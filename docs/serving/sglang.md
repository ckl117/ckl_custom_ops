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
      1. _prefetch_kvcache。多级缓存：req.init_next_round_input(tree_cache) 初始化请求，计算prefix cache长度，包括输出token
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
