import time, torch
t = torch.zeros(1024, 1024, device='cuda')  # hold a live CUDA context
while True:
    t = t * 1.0000001
    time.sleep(30)
