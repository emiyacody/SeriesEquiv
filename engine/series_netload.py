import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.quantization import QuantStub, DeQuantStub
from torchsummary import summary
import time

from github.engine.series_layer import SeriesConv
from github.engine.series_layer import SeriesMaxpool
from github.engine.series_layer import SeriesFC
from github.engine.series_layer import SeriesReLU
from github.engine.imageStar import ImageStar


def series_dnn_diff(net, IM, LB, UB, opt='origin', Istar=None):
    # dim = IM.shape
    # LB = np.zeros(dim, dtype=np.single)
    # UB = np.zeros(dim, dtype=np.single)
    # # LB[0, 0, 0] = lb
    # # UB[0, 0, 0] = ub
    # LB[12:15, 12:15] = np.ones((3, 3), dtype=np.single) * lb
    # UB[12:15, 12:15] = np.ones((3, 3), dtype=np.single) * ub
    start = time.time()
    # IM_m = np.concatenate((IM, IM))
    # LB_m = np.concatenate((LB, LB))
    # UB_m = np.concatenate((UB, UB))
    # I1_m = ImageStar(IM_m, LB_m, UB_m)

    if opt == 'origin':
        I1 = ImageStar(IM, LB, UB)
    else:
        I1 = ImageStar(IM, LB, UB)
        I1.d = Istar.d
        I1.C = Istar.C
        I1.numPred = Istar.numPred
    Istar_pre = I1
    p_i = 1
    method = 'exact-star'
    relu_layer = SeriesReLU()
    for name, m in net.named_children():
        # print(name, ">>>", m)
        if isinstance(m, nn.Conv2d):
            print('Start ' + name)
            layer_oper = getattr(net, name)
            c_wei1 = layer_oper.weight.to(torch.float32)
            c_bias1 = layer_oper.bias.to(torch.float32)
            pad = m.padding
            stride = m.stride
            layer = SeriesConv(c_wei1, c_bias1, pad=pad, stride=stride)
            Istar = layer.reach(Istar_pre)
            Istar_pre = Istar
        elif isinstance(m, nn.MaxPool2d):
            print('Start ' + name)
            layer_oper = getattr(net, name)
            pad = m.padding
            pad = np.array([pad, pad])
            stride = m.stride
            stride = np.array([stride, stride])
            pool = m.kernel_size
            pool = np.array([pool, pool])
            layer = SeriesMaxpool(pool, stride, pad, 'maxpool' + str(p_i))
            Istar = layer.reach(Istar_pre, method)
            p_i += 1
            Istar_pre = Istar
        elif isinstance(m, nn.Linear):
            print('Start ' + name)
            layer_oper = getattr(net, name)
            fc_wei1 = layer_oper.weight.to(torch.float32)
            fc_bias1 = layer_oper.bias.to(torch.float32)
            layer = SeriesFC(fc_wei1, fc_bias1)
            Istar = layer.reach(Istar_pre)
            output_size = m.out_features
            Istar_pre = Istar
        elif isinstance(m, nn.ReLU):
            print('Start ' + name)
            Istar = relu_layer.reach(Istar, method)
            Istar_pre = Istar

    # print('Start last layer')
    # fc_last_w1 = np.eye(output_size, dtype=np.single)
    # fc_last_w2 = -np.eye(output_size, dtype=np.single)
    # fc_last_b1 = np.zeros(output_size)
    # fc_last_b2 = np.zeros(output_size)
    # layer = FC(fc_last_w1, fc_last_b1, fc_last_w2, fc_last_b2, 'last')
    # Istar = layer.reach(Istar_pre)
    end = time.time()
    print("Time: " + str(end - start))
    # if opt == 'origin':
    #     max_range = err_range(Istar, output_size)
    # elif opt == 'new':
    #     max_range = err_range(Istar, output_size, opt='new')
    # else:
    #     raise Exception('Wrong option input!')

    return Istar

