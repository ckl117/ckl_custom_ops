Sglang推理框架
# 整体架构
共有四大模块：
1. api_server: 负责接收客户端请求
2. tokenizer: 负责将请求prompts -> token_ids
3. 调度和引擎：负责调度请求和组batch推理。
4. detokenizer: 负责将token_ids -> output_text，返回给api_server.

## kv cache
根据服务启动参数确定kvcache空间大小。原则：mem_fraction_static=(total - 激活 - CUDAGraph)/total
未指定参数`chunked_prefill_size`, `cuda_graph_max_bs`, `mem_fraction_static`时，会自动计算默认值。
1. 根据显存总量确定`chunked_prefill_size`和`cuda_graph_max_bs`. 秉持着大显存使用更大的参数值；小显存和小TP数，使用较小的参数值。TP数越大`cuda_graph_max_bs`越大，因为单卡可用显存会更多.
2. `mem_fraction_static = chunked_prefill_size * 1.5 + cuda_graph_max_bs * 2.5 < allocable_gpu_memory * mem_fraction_static`, 其中还根据MTP、MLA、DP Attention等场景额外加入了特判(感觉是经验值)。

## 调度和引擎
与上游tokenizer、下游(detokenizer或者是tokenizer)模块通信。
从tokenizer拉取新请求，使用zmq.pull。特点：仅单向接收、消息队列、负载均衡、异步。
将结果推送给下游，使用zmq.push。特点：仅单向发送、消息队列、异步。无负载均衡
从rpc异步接收请求。使用zmq.DEALER。特点：双向、消息队列、负载均衡、异步、带路由id。

初始化下面信息：
1. 初始化并行rank和size等信息
2. 初始化模型config
3. 初始化metrics
4. 初始化ipc通信
5. 初始化tokenizer。将tokenizer附带到req中，返回给detokenizer? note: 感觉有点冗余，给请求增加字段，会影响ipc通信效率
6. moe_gemm后端。DeepGEMM、flashinfer、trtllm、triton等
7. mamba。和Transformer不同的一种结构，暂不清楚
8. 初始化worker。包含tp_worker、draft_worker
9. kv cahce。 根据chunk_size、radix_cache、swa等创建kv cache pool(tree_cache)
10. 调度信息。waiting、running、cur_batch、last_batch等信息
11. 初始化chunked_prefill。 主要用于调度，因为prefill优先，会有切chunk的req。以及PP并行时动态调整chunk_size：在pp_rank0上进行生成128个不同长度的数据，记录每次组batch、推理时间，拟合二次函数模型在运行时预测chunk_size。
12. 调度策略。默认fcfs, 确定请求分配的输出tokens数,init_new_token_ratio、min_new_token_ratio、new_token_ratio_decay(衰减比例)
13. 看门狗。监控调度是否hung住。
14. pd分离。分别在P、D节点初始化buffer。
15. overlap。CPU获取新请求和GPU推理overlap。推理一个stream，copy一个stream。future_map用于记录推理batch采样token的位置
16. ...
    
## 调度
1. TP0接收新请求
2. 广播给所有TP成员
3. 根据请求类型dispatch到不同方法
   1. 生成请求会在这里设置好后处理参数，包括max_tokens
   2. 加入到waiting队列
      1. `_prefetch_kvcache`。多级缓存：req.init_next_round_input(tree_cache) 初始化请求，计算prefix cache长度，包括输出token
      2. 加入到waiting队列
4. get_next_batch_to_run：获取下一个需要推理的batch
   1. 去除waiting、running队列中的超时请求
   2. 处理chunked_req，主要是将last_batch的decoder req放到running队列(decoder)，
      1. last_batch = chunked_req(还需要做prefill) + running_req(要做decoder)
      2. last_batch.running_req 加入到running队列
      3. last_batch.chunked_req 去何处？
   3. 合并batch。 
      1. 去除last_batch中的chunked_req, 以及完成的req。last_batch只有decoder_req
      2. 将last_batch中req 合并到 running_batch
   4. 获取新的prefill new_batch。new_batch里的请求顺序为chunked_req、prefill_req(来自waiting队列)，参考`_get_new_batch_prefill_raw()`。
         1. max_bsz = running_bsz(decoder) + new_batch_bsz(prefill)
         2. 依据`schedule_policy`对waiting重新排序，`fcfs`不排序，`lpm`最长前缀匹配，等
         3. 创建 PrefillAdder: prefill插入调度策略，给每个running_req预留 `min(max_new_tokens-len(output_ids), CLIP_MAX_NEW_TOKENS)*self.new_token_ratio`
         4. 插入chunk_req.根据cache剩余量和chunk_size 对chunkd_req进一步切chunk，如果chunked_req没有切chunk，则预留4k后续token(`CLIP_MAX_NEW_TOKENS`控制),否则预留0。
         5. 插入prefill_req(waiting队列)。
            1. 预取第三级缓存(TP组内有一个预取完成即可)，只需要TP组内最小的token数即可。如果没有预取成功，跳过改req
            2. 计算radix cache(cpu+gpu), 分配input_ids
            3. 产生新的chunk_req
         6. 准备batch的输入：input_ids、seq_len等。`new_batch.prepare_for_extend()`
      1. new_batch 非空，执行prefill。
      2. 否则，跳过prefill_only的running_batch，只执行running_batch(decoder)
         1. 剔除running_batch中已完成的req
         2. 根据kv cache剩余量，抢占部分请求`retract_decode()`,更新self.new_token_ratio
         3. 无抢占，衰减 `self.new_token_ratio = max(self.new_token_ratio - self.new_token_ratio_decay,self.min_new_token_ratio)`
         4. 准备 running_batch 的输入
   5. new_batch(prefil)非空，run_batch() 得到未来输出token Tensor，加入到后处理队列
   6. 执行last_batch(decoder)的后处理`process_batch_result`，并将结果发送到detokenizer
      1. 处理P/D: `next_token_ids`追加到 `output_ids`, `logprobs`处理; 检查停止: eos token、不在词表范围内、解码后的str在stop_str内; 发送给detokenizer;释放资源
   7. 同步schedule_stream, 并发送cur_batch结果异步copy到cpu
   8. last_batch = cur_batch
```
pre_and_run = recv_requests + process_input_requests + get_next_batch_to_run

CPU: [pre_and_run N]      -> [pop_and_process N-1] -> [pre_and_run N+1]             -> [pop_and_process N]
                    ↓                                                            ↑
GPU:                 [                   run_batch N                    ]   ->   d2h(async)
```

## PD分离

### 一、初始化
scheduler通过`init_disaggregation()`完成prefill和decode节点的初始化。

| 模式 | 创建的队列 | 作用 |
   | --- | --- | --- |
| DECODE | disagg_decode_prealloc_queue (DecodePreallocQueue) | 握手 + 预分配 KV 内存 |
| DECODE | disagg_decode_transfer_queue (DecodeTransferQueue) | 轮询 KV 传输状态 |
| PREFILL | disagg_prefill_bootstrap_queue (PrefillBootstrapQueue) | 与 Decode 端握手 |
| PREFILL | disagg_prefill_inflight_queue (List[Req]) | 正在发送 KV 的请求 |

两端都会创建 MetadataBuffers(传输首 token、logprobs、hidden states 等元数据) 和 ReqToMetadataIdxAllocator (元数据 buffer 索引分配器)。其中P的`buffer_size`由总token数//context_len确定， D为最大并发*2


prefill:  
1. BootstrapServer <- start_disagg_service <- init_tokenizer_manager(): 启动aiohttp(/route、/health)，注册P信息，让D发现自己的dp/cp/tp/pp等拓扑
```shell
┌─────────────────────────────────────────────────────────────────────┐
│  CommonKVBootstrapServer (aiohttp, 运行在 Prefill 端)                │
│                                                                     │
│  PUT  /route               ← Prefill rank 注册自己                   │
│  GET  /route               ← Decode 端查询 Prefill 拓扑/连接地址       │
│  POST /register_dp_rank    ← Prefill 注册 bootstrap_room→dp_rank     │
│  POST /query_dp_ranks      ← Decode 批量查询 room→dp_rank 映射        │
│  GET  /health              ← 健康检查                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 二、请求路由 _add_request_to_queue
```python
def _add_request_to_queue(self, req, is_retracted=False):
    if mode == NULL:       → self.waiting_queue.append(req)        # 正常模式
    elif mode == PREFILL:  → self.disagg_prefill_bootstrap_queue.add(req, ...)  # Prefill 模式
    elif mode == DECODE:   → self.disagg_decode_prealloc_queue.add(req, ...)    # Decode 模式

```
### 三、事件分发 dispatch_event_loop
|Mode | Normal | Overlap | Pipeline Parallel |
| --- | --- | --- | --- |
| PREFILL | event_loop_normal_disagg_prefill | event_loop_overlap_disagg_prefill | event_loop_pp_disagg_prefill |
| DECODE | event_loop_normal_disagg_decode | event_loop_overlap_disagg_decode | event_loop_pp_disagg_decode |

#### 四、生命周期
##### P
```
┌──────────────────────────────────────────────────────────────────────┐
│  1. PrefillBootstrapQueue                                            │
│     请求到达 → 创建 KVSender → 与 Decode 端握手                       │
│     poll 状态: Bootstrapping → WaitingForInput                       │
│                                                                      │
│  2. waiting_queue                                                    │
│     握手完成 → pop_bootstrapped() → 进入 waiting_queue                │
│     由 PrefillAdder 调度执行 prefill forward                          │
│                                                                      │
│  3. disagg_prefill_inflight_queue                                    │
│     forward 完成 → send_kv_chunk():                                   │
│       - 获取 KV page indices                                         │
│       - 写入首 token/logprobs/hidden_states 到 MetadataBuffers        │
│       - 调用 sender.send(page_indices, state_indices) 发起 RDMA 传输  │
│     poll 状态: WaitingForInput → Transferring → Success               │
│     Success → 释放 KV Cache, 返回结果给用户                           │
└──────────────────────────────────────────────────────────────────────┘
```
后台线程：
一个prefill线程: 负责与D握手. KVPoll.Bootstrapping -> KVPoll.WaitingForInput
多个传输进程: 一个线程池和多个传输队列`transfer_queue_size`，线程池划分成`transfer_queue_size`个子线程池，负责每个队列传输.
握手(poll)信息: mooncake_session_id、kv cache 大小

1. recv_requests(). 同集中式
2. process_input_requests(). 同集中式，包含(`_add_request_to_queue`)
3. waiting_queue.append(disagg_prefill_bootstrap_queue.pop_bootstrapped()). 完成握手

##### 整体时序图
```shell
    Prefill 端                    Bootstrap Server (HTTP)               Decode 端
    ──────────                    ────────────────────────               ──────────
         │                                │                                  │
  ───────┼── 阶段 0: 服务注册 ────────────┼──────────────────────────────────┼────
         │                                │                                  │
    (启动时) PUT /route ─────────────────→│                                  │
         │  {tp_rank, cp_rank, pp_rank,   │                                  │
         │   rank_ip, rank_port, ...}     │                                  │
         │                                │                                  │
         │                                │←──── GET /route (全局拓扑) ──────│ (启动时)
         │                                │──────→ PrefillServerInfo ────────→│
         │                                │        {tp_size, cp_size, ...}   │
         │                                │                                  │
         │                                │     _resolve_rank_mapping()      │
         │                                │     计算 TP/CP/PP rank 对应关系  │
         │                                │                                  │
  ───────┼── 阶段 1: 请求到达 ────────────┼──────────────────────────────────┼────
         │                                │                                  │
    创建 KVSender                         │                            创建 KVReceiver
    状态 → Bootstrapping                  │                            状态 → Bootstrapping
         │                                │                                  │
         │                                │←── GET /route (具体 rank) ───────│
         │                                │    ?prefill_dp_rank=X&           │
         │                                │     target_tp_rank=Y&pp_rank=Z   │
         │                                │───→ PrefillRankInfo ────────────→│
         │                                │     {rank_ip, rank_port}         │
         │                                │                                  │
  ───────┼── 阶段 2: KV Args 注册 ───────┼──────────────────────────────────┼────
         │                                │                                  │
         │←─── ZMQ PUSH (room="None") ──────────────────────────────────────│
         │     {session_id, kv_data_ptrs, │                          _register_kv_args()
         │      aux_data_ptrs, tp_rank,   │                                  │
         │      kv_item_len, ...}         │                                  │
         │                                │                            状态 → WaitingForInput
    Prefill bootstrap_thread 收到后       │                                  │
    存入 decode_kv_args_table             │                                  │
         │                                │                                  │
  ───────┼── 阶段 3: KV 内存预分配 + 地址交换 ─────────────────────────────┼────
         │                                │                                  │
         │                                │                       DecodePreallocQueue:
         │                                │                         _pre_alloc() 预分配 KV 内存
         │                                │                         获取 kv_page_indices
         │                                │                                  │
         │←─── ZMQ PUSH (room=实际ID) ─────────────────────────────────────│
         │     {bootstrap_room, endpoint, │                          receiver.init(
         │      session_id, dst_kv_indices,│                            kv_indices,
         │      dst_aux_index,            │                            aux_index,
         │      dst_state_indices,        │                            state_indices)
         │      required_dst_info_num}    │                                  │
         │                                │                            状态 → Transferring (等待)
    bootstrap_thread 收到后               │                                  │
    存入 transfer_infos[room]             │                                  │
    当所有 dst_info 到齐后               │                                  │
    状态 → WaitingForInput                │                                  │
         │                                │                                  │
  ───────┼── 阶段 4: KV 数据 RDMA 传输 ──┼──────────────────────────────────┼────
         │                                │                                  │
    Scheduler 执行 prefill forward        │                                  │
    send_kv_chunk() →                     │                                  │
    add_transfer_request() →              │                                  │
    transfer_worker 线程:                 │                                  │
      engine.batch_transfer_sync()        │                                  │
      (RDMA one-sided write) ══════════════════════════════════════════════→ GPU 内存
         │                                │                                  │
    最后一个 chunk:                       │                                  │
      send_aux() (元数据传输)             │                                  │
      sync_status_to_decode_endpoint()    │                                  │
         │                                │                                  │
         │──── ZMQ PUSH {room, Success, rank} ─────────────────────────────→│
         │                                │                          decode_thread 收到后
    状态 → Success                        │                          状态 → Success
         │                                │                                  │
    释放 KV Cache                         │                       _commit_transfer_to_req()
                                                                   读取 MetadataBuffers
                                                                   进入 waiting_queue → 解码

```