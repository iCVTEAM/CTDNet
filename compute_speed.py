import numpy as np
import torch
import time
import os
import sys

sys.path.append("../")
from src.net import CTDNet
import src.dataset as dataset


def compute_speed():
    cfg = dataset.Config(mode='test')
    model = CTDNet(cfg)

    print(model)
    # count parameter number
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: %.3fM" % (pytorch_total_params / 1e6))
    backbone_params = sum(p.numel() for p in model.bkbone.parameters())
    print("Backbone number of parameters: %.3fM" % (backbone_params / 1e6))

    model = model.cuda()
    model.eval()

    run_time = list()

    for i in range(0, 1000):
        input = torch.randn(1, 3, 352, 352).cuda()
        # ensure that context initialization and normal_() operations finish before you start measuring time
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(input)

        torch.cuda.synchronize()        # wait for GPU to finish
        end = time.perf_counter()

        print(end - start)

        run_time.append(end - start)

    run_time.pop(0)

    print('Mean running time is: ', np.mean(run_time))
    print("FPS is: ", (1 / np.mean(run_time)))


if __name__ == "__main__":
    compute_speed()


