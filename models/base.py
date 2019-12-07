import os
import re
from os.path import join

import numpy as np
import torch
from torch import nn

from utils import get_module_device


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.global_step = 0

    def increment_global_step(self):
        self.global_step += 1

    def sample_prior(self, n_imgs, **kwargs):
        raise NotImplementedError

    def get_device(self):
        return get_module_device(self)

    def checkpoint(self, ckpt_folder):
        path = join(ckpt_folder, "model_{}.pt".format(self.global_step))
        torch.save(self.state_dict(), path)

    def load(self, ckpt_folder, step=None):
        if step is None:
            filenames = list(filter(lambda n: "model_" in n, os.listdir(ckpt_folder)))
            regex = re.compile(r'\d+')
            numbers = [int(regex.search(n).group(0)) for n in filenames]
            ckpt_name = filenames[np.argmax(numbers)]   # get latest checkpoint
            step = max(numbers)
        else:
            ckpt_name = "model_{}.pt".format(step)
        print("Loading model checkpoint at step {}...".format(step))
        path = join(ckpt_folder, ckpt_name)
        self.load_state_dict(torch.load(path))
        self.global_step = step
        print("Loaded.")
