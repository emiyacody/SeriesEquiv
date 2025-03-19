import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.quantization import QuantStub, DeQuantStub
import time
from scipy import optimize

from github.engine.layer import MergedConv2d
from github.engine.layer import MaxPooling2d
from github.engine.layer import FC
from github.engine.layer import ReLULayer
from github.engine.imageStar import ImageStar


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


def err_range(Istar, output_size, opt='origin'):
    diff_list = []
    n = len(Istar)
    if opt == 'origin':
        print('Outputs contain %d ImageStar' % n)
        for i in range(n):
            # print('Computing %d ImageStar' % (i+1))
            Star1 = Istar[i].toStar()
            vertices = Star1.toPolyhedron()
            if isinstance(vertices, np.ndarray):
                if len(diff_list) == 0:
                    for ind in range(output_size):
                        diff_list.append([min(vertices[ind, :]), max(vertices[ind, :])])
                else:
                    for ind in range(output_size):
                        diff_list[ind] = [min([diff_list[ind][0], min(vertices[ind, :])]),
                                          max([diff_list[ind][1], max(vertices[ind, :])])]
            else:
                print('Fail to compute vertices!')
                continue
    elif opt == 'new':
        print('Outputs contain %d ImageStar' % n)
        for i in range(n):
            Star1 = Istar[i].toStar()
            vertices = Star1.toPolyhedron_new()
            if isinstance(vertices, np.ndarray):
                if len(diff_list) == 0:
                    for ind in range(output_size):
                        diff_list.append([min(vertices[ind, :]), max(vertices[ind, :])])
                else:
                    for ind in range(output_size):
                        diff_list[ind] = [min([diff_list[ind][0], min(vertices[ind, :])]),
                                          max([diff_list[ind][1], max(vertices[ind, :])])]
            else:
                print('Fail to compute vertices!')
                continue
    return diff_list


def find_maximal_disc(star):
    c_diff = star.V[0, 0, :, 0]
    v_diff = star.V[0, 0, :, 1:]
    C_mat = star.C[1:, :]
    d_mat = star.d[1:, :].squeeze()
    max_temp = 0

    for index in range(star.numChannel):
        fun_max = - v_diff[index, :]
        fun_min = v_diff[index, :]
        pos_result = optimize.linprog(fun_max, A_ub=C_mat, b_ub=d_mat, bounds=(-1, 1))
        neg_result = optimize.linprog(fun_min, A_ub=C_mat, b_ub=d_mat, bounds=(-1, 1))
        max_val = c_diff[index] + np.dot(fun_max, pos_result.x)
        min_val = c_diff[index] + np.dot(fun_min, neg_result.x)
        max_temp = max(max_temp, max(abs(max_val), abs(min_val)))

    return max_temp


def quant_star(input, scale, offset):
    if isinstance(input, list):
        w = input[0].V.shape[1]
        ch = input[0].V.shape[2] // 2
        if w != 1:
            for i in range(len(input)):
                input[i].V[w:, :, :, 0] = np.round(input[i].V[w:, :, :, 0] / scale) + offset
                input[i].V[w:, :, :, 0] = (input[i].V[w:, :, :, 0] - offset) * scale
        elif (w == 1) & (input[0].V.shape[0] == 2):
            for i in range(len(input)):
                input[i].V[1, 0, :, 0] = np.round(input[i].V[1, 0, :, 0] / scale) + offset
                input[i].V[1, 0, :, 0] = (input[i].V[1, 0, :, 0] - offset) * scale
        else:
            for i in range(len(input)):
                input[i].V[0, 0, ch:, 0] = np.round(input[i].V[0, 0, ch:, 0] / scale) + offset
                input[i].V[0, 0, ch:, 0] = (input[i].V[0, 0, ch:, 0] - offset) * scale
    elif isinstance(input, ImageStar):
        w = input.V.shape[1]
        input.V[w:, :, :, 0] = np.round(input.V[w:, :, :, 0] / scale) + offset
        input.V[w:, :, :, 0] = (input.V[w:, :, :, 0] - offset) * scale
    return input


def compute_net_diff(model, net1, net2, IM, LB, UB, opt='origin'):
    # dim = IM.shape
    # LB = np.zeros(dim, dtype=np.single)
    # UB = np.zeros(dim, dtype=np.single)
    # # LB[0, 0, 0] = lb
    # # UB[0, 0, 0] = ub
    # LB[12:15, 12:15] = np.ones((3, 3), dtype=np.single) * lb
    # UB[12:15, 12:15] = np.ones((3, 3), dtype=np.single) * ub
    start = time.time()
    IM_m = np.concatenate((IM, IM))
    LB_m = np.concatenate((LB, LB))
    UB_m = np.concatenate((UB, UB))
    I1_m = ImageStar(IM_m, LB_m, UB_m)

    Istar_pre = I1_m
    p_i = 1
    method = 'exact-star'
    relu_layer = ReLULayer()
    for name, m in model.named_children():
        # print(name, ">>>", m)
        if isinstance(m, nn.Conv2d):
            print('Start ' + name)
            c_wei1 = net1[name+'.weight'].to(torch.float32)
            c_bias1 = net1[name+'.bias'].to(torch.float32)
            c_wei2 = net2[name + '.weight'].to(torch.float32)
            c_bias2 = net2[name + '.bias'].to(torch.float32)
            pad = m.padding
            stride = m.stride
            layer = MergedConv2d(c_wei1, c_bias1, c_wei2, c_bias2,
                                 pad1=pad, pad2=pad, stride1=stride, stride2=stride)
            Istar = layer.reach(Istar_pre)
        elif isinstance(m, nn.MaxPool2d):
            print('Start ' + name)
            pad = m.padding
            pad = np.array([pad, pad])
            stride = m.stride
            stride = np.array([stride, stride])
            pool = m.kernel_size
            pool = np.array([pool, pool])
            layer = MaxPooling2d(pool, stride, pad, 'maxpool'+str(p_i))
            Istar = layer.reach(Istar_pre, method)
            p_i += 1
        elif isinstance(m, nn.Linear):
            print('Start ' + name)
            fc_wei1 = net1[name+'.weight'].to(torch.float32)
            fc_bias1 = net1[name+'.bias'].to(torch.float32)
            fc_wei2 = net2[name + '.weight'].to(torch.float32)
            fc_bias2 = net2[name + '.bias'].to(torch.float32)
            layer = FC(fc_wei1, fc_bias1, fc_wei2, fc_bias2)
            Istar = layer.reach(Istar_pre)
            output_size = m.out_features
        elif isinstance(m, nn.ReLU):
            print('Start ' + name)
            Istar = relu_layer.reach(Istar, method)
        Istar_pre = Istar

    print('Start last layer')
    fc_last_w1 = np.eye(output_size, dtype=np.single)
    fc_last_w2 = -np.eye(output_size, dtype=np.single)
    fc_last_b1 = np.zeros(output_size)
    fc_last_b2 = np.zeros(output_size)
    layer = FC(fc_last_w1, fc_last_b1, fc_last_w2, fc_last_b2, 'last')
    Istar = layer.reach(Istar_pre)
    end = time.time()
    print("Time: " + str(end - start))
    # if opt == 'origin':
    #     max_range = err_range(Istar, output_size)
    # elif opt == 'new':
    #     max_range = err_range(Istar, output_size, opt='new')
    # else:
    #     raise Exception('Wrong option input!')
    max_range = 0
    for j in range(len(Istar)):
        val = find_maximal_disc(Istar[j])
        max_range = max(max_range, val)

    return max_range


def compute_quant_net_diff(model, net1, net2, IM, lb, ub):
    # net1 = torch.load(pth1, map_location=torch.device('cpu'))
    # net2 = torch.load(pth2, map_location=torch.device('cpu'))

    dim = np.array(IM.shape)
    LB = np.zeros(dim, dtype=np.single)
    UB = np.zeros(dim, dtype=np.single)
    LB[0, 0, 0] = lb
    UB[0, 0, 0] = ub
    # LB[0:2, 0:2, 0] = lb
    # UB[0:2, 0:2, 0] = ub
    IM_m = np.concatenate((IM, IM))
    LB_m = np.concatenate((LB, LB))
    UB_m = np.concatenate((UB, UB))
    I1_m = ImageStar(IM_m, LB_m, UB_m)

    Istar_pre = I1_m
    p_i = 1
    method = 'exact-star'
    relu_layer = ReLULayer()
    for name, m in model.named_children():
        if isinstance(m, QuantStub):
            scale = net2[name+'.scale'].numpy()
            offset = net2[name+'.zero_point'].numpy()
            continue
        elif isinstance(m, nn.Conv2d):
            c_wei1 = net1[name+'.weight'].to(torch.float32)
            c_bias1 = net1[name+'.bias'].to(torch.float32)
            c_wei2 = net2[name + '.weight'].dequantize().to(torch.float32)
            c_bias2 = net2[name + '.bias'].dequantize().to(torch.float32)
            pad = m.padding
            stride = m.stride

            Istar_pre = quant_star(Istar_pre, scale, offset)
            layer = MergedConv2d(c_wei1, c_bias1, c_wei2, c_bias2,
                                 pad1=pad, pad2=pad, stride1=stride, stride2=stride)
            Istar = layer.reach(Istar_pre)
            scale = net2[name + '.scale'].numpy()
            offset = net2[name + '.zero_point'].numpy()
        elif isinstance(m, nn.Linear):
            fc_wei1 = net1[name+'.weight'].to(torch.float32)
            fc_bias1 = net1[name+'.bias'].to(torch.float32)
            fc_wei2 = net2[name + '._packed_params._packed_params'][0].dequantize().to(torch.float32)
            fc_bias2 = net2[name + '._packed_params._packed_params'][1].dequantize().to(torch.float32)

            Istar_pre = quant_star(Istar_pre, scale, offset)
            layer = FC(fc_wei1, fc_bias1, fc_wei2, fc_bias2)
            Istar = layer.reach(Istar_pre)
            output_size = m.out_features
            scale = net2[name + '.scale'].numpy()
            offset = net2[name + '.zero_point'].numpy()
        elif isinstance(m, nn.MaxPool2d):
            pad = m.padding
            pad = np.array([pad, pad])
            stride = m.stride
            stride = np.array([stride, stride])
            pool = m.kernel_size
            pool = np.array([pool, pool])
            layer = MaxPooling2d(pool, stride, pad, 'maxpool'+str(p_i))
            Istar = layer.reach(Istar_pre, method)
            p_i += 1
        elif isinstance(m, nn.ReLU):
            Istar = relu_layer.reach(Istar, method)
        elif isinstance(m, DeQuantStub):
            continue
        else:
            raise Exception("%s layer is undefined!" % name)
        Istar_pre = Istar.copy()

    fc_last_w1 = np.eye(output_size, dtype=np.single)
    fc_last_w2 = -np.eye(output_size, dtype=np.single)
    fc_last_b1 = np.zeros(output_size)
    fc_last_b2 = np.zeros(output_size)
    layer = FC(fc_last_w1, fc_last_b1, fc_last_w2, fc_last_b2, 'last')
    Istar_pre = quant_star(Istar_pre, scale, offset)
    Istar = layer.reach(Istar_pre)

    max_range = err_range(Istar, output_size)

    return max_range


def compute_prune_net_diff(model, net1, net2, IM, lb, ub):
    dim = IM.shape
    LB = np.zeros(dim, dtype=np.single)
    UB = np.zeros(dim, dtype=np.single)
    LB[0, 0, 0] = lb
    UB[0, 0, 0] = ub
    IM_m = np.concatenate((IM, IM))
    LB_m = np.concatenate((LB, LB))
    UB_m = np.concatenate((UB, UB))
    I1_m = ImageStar(IM_m, LB_m, UB_m)

    Istar_pre = I1_m
    p_i = 1
    method = 'exact-star'
    relu_layer = ReLULayer()
    for name, m in model.named_children():
        # print(name, ">>>", m)
        if isinstance(m, nn.Conv2d):
            c_wei1 = net1[name + '.weight'].to(torch.float32)
            c_bias1 = net1[name + '.bias'].to(torch.float32)
            c_wei2 = net2[name + '.weight'].to(torch.float32)
            c_bias2 = net2[name + '.bias'].to(torch.float32)
            pad = m.padding
            stride = m.stride
            layer = MergedConv2d(c_wei1, c_bias1, c_wei2, c_bias2,
                                 pad1=pad, pad2=pad, stride1=stride, stride2=stride)
            Istar = layer.reach(Istar_pre)
        elif isinstance(m, nn.MaxPool2d):
            pad = m.padding
            pad = np.array([pad, pad])
            stride = m.stride
            stride = np.array([stride, stride])
            pool = m.kernel_size
            pool = np.array([pool, pool])
            layer = MaxPooling2d(pool, stride, pad, 'maxpool' + str(p_i))
            Istar = layer.reach(Istar_pre, method)
            p_i += 1
        elif isinstance(m, nn.Linear):
            fc_wei1 = net1[name + '.weight'].to(torch.float32)
            fc_bias1 = net1[name + '.bias'].to(torch.float32)
            fc_wei2 = net2[name + '.weight'].to(torch.float32)
            fc_bias2 = net2[name + '.bias'].to(torch.float32)
            layer = FC(fc_wei1, fc_bias1, fc_wei2, fc_bias2)
            Istar = layer.reach(Istar_pre)
            output_size = m.out_features
        elif isinstance(m, nn.ReLU):
            Istar = relu_layer.reach(Istar, method)
        Istar_pre = Istar

    fc_last_w1 = np.eye(output_size, dtype=np.single)
    fc_last_w2 = -np.eye(output_size, dtype=np.single)
    fc_last_b1 = np.zeros(output_size)
    fc_last_b2 = np.zeros(output_size)
    layer = FC(fc_last_w1, fc_last_b1, fc_last_w2, fc_last_b2, 'last')
    Istar = layer.reach(Istar_pre)
    max_range = err_range(Istar, output_size)

    return max_range


if __name__ == '__main__':
    pth1 = '../data/numer1.pth'
    pth2 = '../data/numer2.pth'
    net1 = Net1()
    net2 = Net2()
    net1.load_state_dict(torch.load(pth1))
    net2.load_state_dict(torch.load(pth2))
    fig = plt.figure()
    sample_n = 10
    x = np.linspace(1, sample_n, sample_n, dtype=int)
    upp_list = np.zeros(10)
    low_list = np.zeros(10)
    for iter in range(sample_n):
        IM = np.random.rand(6, 6, 1).astype(np.single)
        lb = -0.05
        ub = 0.05
        max_range = compute_net_diff(pth1, pth2, net1, net2, IM, lb, ub)
        with torch.no_grad():
            IM_input = torch.from_numpy(IM.transpose(2, 0, 1)).unsqueeze(0)
            output1 = net1(IM_input)
            upp_list[iter] = output1.squeeze().numpy() - max_range[0][0]
            low_list[iter] = output1.squeeze().numpy() - max_range[0][1]
            for i in range(20):
                noise = 0.1 * np.random.rand(2, 2) - 0.05
                IM_n = IM.copy()
                IM_n[0:2, 0:2, 0] += noise
                output2 = net2(torch.from_numpy(IM_n.transpose(2, 0, 1)).unsqueeze(0))
                plt.plot(x[iter], output2, color='k', marker='.', markersize=4)
    plt.plot(x, upp_list, color='b', linestyle='solid')
    plt.plot(x, low_list, color='g', linestyle='solid')
    plt.fill_between(x, low_list, upp_list, color='yellow', alpha=0.2)
    plt.title('Numerical Example')
    plt.xlabel('Index of Sample')
    plt.ylabel('Ranges')
    plt.show()