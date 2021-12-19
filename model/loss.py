import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss():
    def __init__(self, args):
        self.args = args

    def base(self, config, logits, labels):

        return config['criterion'](logits, labels)