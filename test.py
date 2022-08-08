import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torch


# def check_dir(file_name=None):
#     dir_name = osp.dirname(file_name)
#     if not os.path.exists(dir_name):
#         os.makedirs(dir_name)

# x = [1, 2, 3]
# y = x
# plt.plot(x, y)
# plt.margins(0,0)
# plt.savefig("./test.pdf")
# plt.show()

path = osp.expanduser('~/datasets')
print(path)
print(torch.__version__)
print(torch.version.cuda)