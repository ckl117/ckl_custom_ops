import torch

import toy_hgemm

if __name__ == "__main__":
    
    Ms = [ 2**i for i in range(10)]
    Ms = [1024]
    Ns = [128, 256, 1024, 2048]
    Ns = [300]
    run_tims = 1
    for M in Ms:
        for N in Ns:
            x = torch.randn([M, N], dtype=torch.float, device='cuda')
            y = torch.empty_like(x)
            
            for i in range(run_tims):
                toy_hgemm.online_softmax(x, y)
                print(y)

            for i in range(run_tims):
                baseline = torch.softmax(x, dim=-1)
                print(baseline)

