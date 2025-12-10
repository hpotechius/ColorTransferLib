from tracemalloc import start
import torch
import time
from basicsr.archs.colorformer_arch import ColorFormer as models


model = models('twins', input_size=[256, 256], num_output_channels=2, last_norm='Spectral', do_normalize=False)
model = model.cuda()
torch.backends.cudnn.benchmark=True

inp = torch.rand((1,3, 256, 256)).cuda()
with torch.no_grad():
    for _  in range(100):
        model(inp)
    torch.cuda.synchronize()
    start = time.time()
    for _  in range(1000):
        model(inp)
    torch.cuda.synchronize()
    print((time.time()-start)/1000)