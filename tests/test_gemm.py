import torch

import toy_hgemm


@torch.no_grad
def as_col_major(x: torch.Tensor):
    # convert a row major tensor -> col major with contiguous storage
    x_trans = x.t()
    print(f"{x_trans.stride()=}, {x_trans.size()=}")
    x_col_major = x_trans.reshape(x.shape)
    print(f"{x_col_major.stride()=}, {x_col_major.size()=}")
    return x_col_major.contiguous()  # must be a contiguous tensor


def test_hgemm_ptx():
    M = 512
    N = 1024
    K = 1024
    dtype = torch.bfloat16
    dtype = torch.half
    a = torch.randn([M, K], dtype=dtype, device="cuda")
    b = torch.randn([K, N], dtype=dtype, device="cuda")
    bias = None
    out = torch.empty([M, N], dtype=dtype, device="cuda")
    toy_hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4(a, b, out)
    print(f"{out=}")
    out_ref = a @ b
    print(f"{out_ref=}")

    print(
        "Max difference:",
        torch.max(torch.abs(out_ref.to(torch.float32) - out.to(torch.float32))),
    )


def test_hgemm_cute():
    M = 512
    N = 1024
    K = 1024
    dtype = torch.bfloat16
    dtype = torch.half
    a = torch.randn([M, K], dtype=dtype, device="cuda")
    b = torch.randn([K, N], dtype=dtype, device="cuda")
    print(b.stride(), b.size())
    b_col_major = as_col_major(b)
    print(b_col_major.stride(), b_col_major.size())
    bias = None
    out = torch.empty([M, N], dtype=dtype, device="cuda")
    toy_hgemm.hgemm_mma_stages_block_swizzle_tn_cute(a, b_col_major, out, 2, False, 1)
    print(f"{out=}")
    out_ref = a @ b
    print(f"{out_ref=}")

    print(
        "Max difference:",
        torch.max(torch.abs(out_ref.to(torch.float32) - out.to(torch.float32))),
    )


if __name__ == "__main__":
    test_hgemm_cute()
