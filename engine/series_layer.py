import numpy as np
from github.engine.imageStar import ImageStar
from github.engine.star import Star
from github.engine.zono import Zono
# from imageStar import ImageStar
# from star import Star
# from zono import Zono
import torch
import torch.nn as nn
import math


class SeriesConv():
    def __init__(self, weight, bias, pad=(0, 0), stride=1, dila=1):
        layer_name = 'Merged Conv2d'
        padding_mat = pad
        stride_mat = stride
        dilation_mat = dila
        self.NumInputs = 1
        self.InputNames = "input"
        self.NumOutputs = 1
        self.OutputNames = "output"

        if isinstance(layer_name, str):
            self.Name = layer_name
        else:
            Exception("Layer name should be a string")

        if isinstance(weight, np.ndarray):
            filter_weights = weight.astype(np.float64)
            filter_bias = bias.astype(np.float64)

            w = filter_weights.shape
            b = filter_bias.shape
            if len(w) != 4:
                Exception("Invalid weights 1 array")
            if len(b) != 3:
                Exception("Invalid bias 1 array")
            if w[3] != b[1]:
                Exception("Inconsistency between filter weights 1 and filter bias 1")

            self.NumFilters = w[3]
            self.NumChannels = w[2]
            self.FilterSize = np.array([w[0], w[1]])
            self.Weights = filter_weights
            self.Bias = filter_bias
            self.PaddingSize = padding_mat
            self.Stride = stride_mat
            self.DilationFactor = dilation_mat

        elif isinstance(weight, torch.Tensor):
            filter_weights = weight
            filter_bias = bias

            w = filter_weights.shape
            self.NumFilters = w[0]
            self.NumChannels = w[1]
            self.FilterSize = np.array([w[2], w[3]])
            self.Weights = filter_weights
            self.Bias = filter_bias
            self.PaddingSize = padding_mat
            self.Stride = stride_mat
            self.DilationFactor = dilation_mat

        else:
            raise Exception('wrong parameter type')

    def evaluate(self, *args):
        nargs = len(args)
        if nargs == 1:
            input = args[0]
            option = []
        elif nargs == 2:
            input = args[0]
            option = args[1]
        else:
            Exception("Invalid number of inputs, should be 1/2")

        y = nn.Conv2d(input, self.Weights, self.Bias, stride=self.Stride, padding=self.PaddingSize)
        return y

    def series_reach(self, image):

        if isinstance(image, ImageStar) is False:
            raise Exception("image is not ImageStar")

        c1 = image.V[:, :, :, 0].copy()
        v1 = image.V[:, :, :, 1:image.numPred + 1].copy()

        k_l = self.FilterSize[0]

        conv_c = nn.Conv2d(c1.shape[2], self.NumFilters, k_l, stride=self.Stride, padding=self.PaddingSize,
                          dilation=self.DilationFactor)
        conv_v = nn.Conv2d(c1.shape[2], self.NumFilters, k_l, bias=False, stride=self.Stride,
                            padding=self.PaddingSize, dilation=self.DilationFactor)

        if isinstance(self.Weights, torch.Tensor):
            conv_c.weight = nn.Parameter(self.Weights, requires_grad=False)
            conv_c.bias = nn.Parameter(self.Bias, requires_grad=False)
            conv_v.weight = nn.Parameter(self.Weights, requires_grad=False)
        else:
            conv_c.weight = nn.Parameter(torch.from_numpy(self.Weights.transpose(3, 2, 0, 1)), requires_grad=False)
            conv_c.bias = nn.Parameter(torch.from_numpy(self.Bias.flatten()), requires_grad=False)
            conv_v.weight = nn.Parameter(torch.from_numpy(self.Weights.transpose(3, 2, 0, 1)), requires_grad=False)

        with torch.no_grad():
            new_c1 = conv_c(torch.from_numpy(c1.transpose(2, 0, 1)).unsqueeze_(0))
            for i in range(v1.shape[3]):
                new_v1_temp = conv_v(torch.from_numpy(v1.transpose(2, 3, 0, 1))[:, i, :, :].unsqueeze(0))
                if i == 0:
                    new_v1 = new_v1_temp
                else:
                    new_v1 = torch.cat((new_v1, new_v1_temp), 0)

            new_y1 = np.concatenate((new_c1.numpy().transpose(2, 3, 1, 0), new_v1.numpy().transpose(2, 3, 1, 0)),
                                    axis=3)

        new_V = new_y1
        S = ImageStar(new_V, image.C, image.d, image.pred_lb, image.pred_ub)
        S.inheritMaxIdx(image.MaxIdxs)
        S.inheritInputSize(image.InputSizes)
        return S

    def multi_series_reach(self, in_images):
        IS = []
        if isinstance(in_images, list):
            n = len(in_images)
            for i in range(n):
                IS.append(self.series_reach(in_images[i]))
        elif isinstance(in_images, ImageStar):
            IS.append(self.series_reach(in_images))

        return IS

    def reach_star_single_input(self, input):

        if isinstance(input, ImageStar) is False:
            Exception("The input is not an ImageStar")

        if input.numChannel != self.NumChannels:
            Exception("Input set contains %d channels while the conv layer has %d channels"
                      % (input.numChannel, self.NumChannels))

        c = nn.Conv2d(input.V[:, :, :, 0])
        V = nn.Conv2d(input.V[:, :, :, 1:input.numPred])
        Y = np.concatenate((c, V), axis=3)
        S = ImageStar(Y, input.C, input.d, input.pred_lb, input.pred_ub)
        return S

    def reach_star_multipleInputs(self, in_images, option):

        n = len(in_images)
        images = []
        for i in range(n):
            images.append(self.reach_star_single_input(in_images[i]))

        return images

    def reach(self, *args):
        nargs = len(args)
        if nargs == 6:
            in_images = args[0]
            method = args[1]
            option = args[2]
        elif nargs == 5:
            in_images = args[0]
            method = args[1]
            option = args[2]
        elif nargs == 4:
            in_images = args[0]
            method = args[1]
            option = args[2]
        elif nargs == 3:
            in_images = args[0]
            method = args[1]
            option = args[2]
        elif nargs == 2:
            in_images = args[0]
            method = args[1]
            option = 'single'
        elif nargs == 1:
            in_images = args[0]
            method = 'approx-star'
            option = 'single'
        else:
            Exception("Invalid number of input, should be 1/2/3/4/5/6")

        IS = self.multi_series_reach(in_images)

        return IS

    def parse(self, conv2d):
        if isinstance(conv2d, SeriesConv) is False:
            Exception("Input is not a Conv2d")

        filter_weights = conv2d.Weights
        filter_bias = conv2d.Bias
        padding_mat = conv2d.PaddingSize
        stride_mat = conv2d.Stride
        dilation_mat = conv2d.DilationFactor

        L = SeriesConv(filter_weights, filter_bias, padding_mat, stride_mat, dilation_mat)
        return L


class SeriesFC(object):

    def __init__(self, weight, bias, option=None):
        if option is None:
            if weight.shape[0] != bias.shape[0]:
                raise Exception("Inconsistent dimension between the weight1 matrix and bias1 vector")
            self.Weights = weight
            self.Bias = bias
            self.option = option
            self.InputSize = weight.shape[1]
            self.OutputSize = weight.shape[0]
        elif option == 'last':
            self.Weights = weight
            self.Bias = bias
            self.option = option
            self.InputSize = weight.shape[1] * 2
            self.OutputSize = weight.shape[0]

    def series_reach(self, image):
        if isinstance(image, ImageStar) is False:
            raise Exception("image is not an ImageStar")

        numPred = image.numPred
        # N = image.height * image.width * image.numChannel
        # if (N / 2 != self.InputSize) & (self.option != 'last'):
        #     raise Exception("Inconsistency between the size of the input image and the InputSize of the network")
        # elif (N != self.InputSize) & (self.option == 'last'):
        #     raise Exception("Inconsistency between the size of the input image and the InputSize of the network")

        temp_V = image.V.copy()
        temp_V = temp_V.transpose(2, 3, 0, 1)

        V = np.zeros((1, 1, self.OutputSize, numPred + 1), dtype=np.single)
        if isinstance(self.Weights, torch.Tensor):
            fc1 = nn.Linear(self.InputSize, self.OutputSize)
            fc1.weight = nn.Parameter(self.Weights, requires_grad=False)
            fc1.bias = nn.Parameter(self.Bias, requires_grad=False)

            fc_v1 = nn.Linear(self.InputSize, self.OutputSize, bias=False)
            fc_v1.weight = nn.Parameter(self.Weights, requires_grad=False)
        else:
            fc1 = nn.Linear(self.InputSize, self.OutputSize)
            fc1.weight = nn.Parameter(torch.from_numpy(self.Weights), requires_grad=False)
            fc1.bias = nn.Parameter(torch.from_numpy(self.Bias), requires_grad=False)

            fc_v1 = nn.Linear(self.InputSize, self.OutputSize, bias=False)
            fc_v1.weight = nn.Parameter(torch.from_numpy(self.Weights), requires_grad=False)

        with torch.no_grad():
            if temp_V.shape[3] != 1:
                h = temp_V.shape[3]
                for i in range(numPred + 1):
                    I1 = torch.flatten(torch.from_numpy(temp_V[:, i, 0:h, :])).unsqueeze_(0)
                    if i == 0:
                        V[0, 0, :, i] = fc1(I1)
                    else:
                        V[0, 0, :, i] = fc_v1(I1)
            elif (temp_V.shape[3] == 1) & (temp_V.shape[2] == 2):
                h = temp_V.shape[3]
                for i in range(numPred + 1):
                    I1 = torch.flatten(torch.from_numpy(temp_V[:, i, 0:h, :])).unsqueeze_(0)
                    if i == 0:
                        V[0, 0, :, i] = fc1(I1)
                    else:
                        V[0, 0, :, i] = fc_v1(I1)
            else:
                c = int(temp_V.shape[0])
                for i in range(numPred + 1):
                    I1 = torch.from_numpy(temp_V[0:c, i, :, :].flatten()).unsqueeze_(0)
                    if i == 0:
                        V[0, 0, :, i] = fc1(I1)
                    else:
                        V[0, 0, :, i] = fc_v1(I1)

        Istar = ImageStar(V, image.C, image.d, image.pred_lb, image.pred_ub)
        Istar.inheritMaxIdx(image.MaxIdxs)
        Istar.inheritInputSize(image.InputSizes)
        return Istar

    def multi_series_reach(self, in_images):
        IS = []
        if isinstance(in_images, list):
            n = len(in_images)
            for i in range(n):
                IS.append(self.series_reach(in_images[i]))
        elif isinstance(in_images, ImageStar):
            IS.append(self.series_reach(in_images))

        return IS

    def reach(self, images):
        IS = self.multi_series_reach(images)

        return IS


class SeriesMaxpool(object):

    def __init__(self, poolsize, stride, paddingsize, name='MaxPooling2d'):
        hasunpoolingoutputs = 0
        numinputs = 1
        inputnames = 'input'
        numoutputs = 1
        outputnames = 'output'

        if isinstance(name, str):
            self.Name = name
        else:
            Exception("Name is not string")
        if poolsize.shape != (2,):
            Exception("Invalid pool size")
        else:
            self.PoolSize = poolsize
        if stride.shape == (2,):
            self.Stride = stride
        else:
            Exception("Invalid stride size")
        if paddingsize.shape == (2,):
            self.PaddingSize = paddingsize
        else:
            Exception("Invalid padding size")
        if hasunpoolingoutputs >= 0:
            self.HasUnpoolingOutputs = hasunpoolingoutputs
        else:
            Exception("Invalid HasUnpoolingOutputs parameter")
        if numinputs >= 1:
            self.NumInputs = numinputs
        else:
            Exception("Invalid number of inputs")
        if numoutputs >= 1:
            self.NumOutputs = numoutputs
        else:
            Exception("Invalid number of outputs")

        self.InputNames = inputnames
        self.OutputNames = outputnames

    def evaluate(self, *args):
        nargs = len(args)
        if nargs == 1:
            input = args[0]
            option = 'cnn'
        elif nargs == 2:
            input = args[0]
            option = args[1]
        else:
            Exception("Invalid number of input")

        y = nn.MaxPool2d(self.PoolSize, self.Stride, self.PaddingSize)
        return y

    def get_zero_padding_input(self, input):
        n = input.shape
        t = self.PaddingSize[0]
        b = self.PaddingSize[0]
        l = self.PaddingSize[1]
        r = self.PaddingSize[1]

        if len(n) == 2:
            # Input has one channel
            h = n[0]
            w = n[1]

            padded_I = np.zeros((t + h + b, l + w + r))
            padded_I[t:t + h, l:l + w] = input

        elif len(n) > 2:
            h = n[0]
            w = n[1]
            d = n[2]

            padded_I = np.zeros((t + h + b, l + w + r, d))
            for i in range(d):
                padded_I[t:t + h, l:l + w, i] = input[:, :, i]

        return padded_I

    def get_zero_padding_imagestar(self, ims):

        if self.PaddingSize.sum() == 0:
            # pad_ims = ims
            if hasattr(ims, 'im_lb'):
                pad_ims = ImageStar(ims.V.copy(), ims.C.copy(), ims.d.copy(), ims.pred_lb.copy(),
                                              ims.pred_ub.copy(), ims.im_lb.copy(), ims.im_ub.copy())
            else:
                pad_ims = ImageStar(ims.V.copy(), ims.C.copy(), ims.d.copy(), ims.pred_lb.copy(),
                                              ims.pred_ub.copy())
        else:
            c = self.get_zero_padding_input(ims.V[:, :, :, 1])
            k = c.shape
            n = ims.numPred
            new_size = (k[0], k[1], k[2], n + 1)
            V1 = np.zeros(new_size, dtype=np.single)
            V1[:, :, :, 0] = c
            for i in range(n):
                V1[:, :, :, i + 1] = self.get_zero_padding_input(ims.V[:, :, :, i + 1])
            if hasattr(ims, 'im_lb'):
                new_im_lb = self.get_zero_padding_input(ims.im_lb)
                new_im_ub = self.get_zero_padding_input(ims.im_ub)
            else:
                new_im_lb = None
                new_im_ub = None
            pad_ims = ImageStar(V1, ims.C, ims.d, ims.pred_lb, ims.pred_ub, new_im_lb, new_im_ub)

        return pad_ims

    def compute_maxmap(self, input):
        I = self.get_zero_padding_input(input)
        m = self.PoolSize
        h, w = self.get_size_maxmap(input)
        map = self.get_startpoints(input)
        maxmap = np.zeros((1, h * w))

        for l in range(h * w):
            a = l % w
            if a == 0:
                i = math.floor(l / w)
                j = w
            else:
                i = a
                j = math.floor(l / w) + 1

            i0 = map[i - 1, j - 1][0]
            j0 = map[i - 1, j - 1][1]
            val = I[i0 - 1, j0 - 1]
            for i in range(i0, i0 + m[0]):
                for j in range(j0, j0 + m[1]):
                    if val < I[i, j]:
                        val = I[i, j]
            maxmap[l] = val
        maxmap = maxmap.reshape([h, w]).T
        return maxmap

    def get_size_maxmap(self, input):
        I = self.get_zero_padding_input(input)
        n = I.shape
        m = self.PoolSize
        h = math.floor((n[0] - m[0]) / self.Stride[0] + 1)
        w = math.floor((n[1] - m[1]) / self.Stride[1] + 1)

        return h, w

    def get_startpoints(self, input):
        I = self.get_zero_padding_input(input)
        m = self.PoolSize

        h, w = self.get_size_maxmap(input)

        startPoints = [[] for i in range(h)]

        for i in range(h):
            for j in range(w):
                startPoints[i].append(np.zeros(2))
                if i == 0:
                    startPoints[i][j][0] = 0
                elif i >= 1:
                    startPoints[i][j][0] = startPoints[i - 1][j][0] + self.Stride[0]
                if j == 0:
                    startPoints[i][j][1] = 0
                elif j >= 1:
                    startPoints[i][j][1] = startPoints[i][j - 1][1] + self.Stride[1]
        return startPoints

    def construct_maxmap(self, input):
        if isinstance(input, ImageStar) is False:
            Exception("Input is not an ImageStar")

        h, w = self.get_size_maxmap(input.V[:, :, 0, 0])
        new_V = np.zeros((h, w, input.numChannel, input.numPred + 1))

        channel_maxPoints = np.zeros((3, h * w, input.numChannel))
        for i in range(input.numChannel):
            channel_maxPoints[:, :, i] = input.max_points[:, i * h * w:(i + 1) * h * w]

        for p in range(input.numPred + 1):
            for k in range(input.numChannel):
                for i in range(h):
                    for j in range(w):
                        ind = i * w + j
                        max_ind = channel_maxPoints[:, ind, k]
                        new_V[:, :, k, p] = input.V[max_ind[0], max_ind[1], k, p]

        image = ImageStar(new_V, input.C, input.d, input.pred_lb, input.pred_ub)
        return image

    def stepSplit(self, *args):
        nargs = len(args)
        if nargs == 4:
            in_image = args[0]
            ori_image = args[1]
            pos = args[2]
            split_index = args[3]
            lp_solver = 'linprog'
        elif nargs == 5:
            in_image = args[0]
            ori_image = args[1]
            pos = args[2]
            split_index = args[3]
            lp_solver = args[4]

        if isinstance(in_image, ImageStar) is False:
            raise Exception('input maxMap is not an ImageStar')
        if isinstance(ori_image, ImageStar) is False:
            raise Exception('reference image is not an ImageStar')

        n = split_index.shape
        if (n[1] != 3) | (n[0] < 1):
            raise Exception('Invalid split index, it should have 3 columns and at least 1 row')

        images = []
        for i in range(n[0]):
            center = split_index[i].copy()
            others = split_index.copy()
            others = np.delete(others, i, axis=0)
            # others[i] = np.zeros(n[1])
            new_C, new_d = ImageStar.isMax(in_image, ori_image, center, others, lp_solver)
            if (new_C is not None) & (new_d is not None):
                V = in_image.V
                V[pos[0], pos[1], pos[2], :] = ori_image.V[center[0], center[1], center[2], :]
                if hasattr(in_image, 'im_lb'):
                    im = ImageStar(V, new_C, new_d, in_image.pred_lb, in_image.pred_ub, in_image.im_lb,
                                             in_image.im_ub)
                else:
                    im = ImageStar(V, new_C, new_d, in_image.pred_lb, in_image.pred_ub)
                im.MaxIdxs = in_image.MaxIdxs.copy()
                im.InputSizes = in_image.InputSizes.copy()
                im.updateMaxIdx(self.Name, center, pos)
                images.append(im)
        return images

    def stepSplitMultipleInputs(self, *args):
        nargs = len(args)
        if nargs == 5:
            in_images = args[0]
            ori_image = args[1]
            pos = args[2].copy()
            split_index = args[3].copy()
            option = args[4].copy()
            lp_solver = 'linprog'
        elif nargs == 6:
            in_images = args[0]
            ori_image = args[1]
            pos = args[2].copy()
            split_index = args[3].copy()
            option = args[4].copy()
            lp_solver = args[5]
        else:
            raise Exception('Invalid number of input arguments!')

        images = []
        if isinstance(in_images, list):
            n = len(in_images)
            for i in range(n):
                images.extend(self.stepSplit(in_images[i], ori_image, pos, split_index, lp_solver))
        elif isinstance(in_images, ImageStar):
            images.extend(self.stepSplit(in_images, ori_image, pos, split_index, lp_solver))

        return images

    def reach_star_approx(self, *args):
        nargs = len(args)
        if nargs == 1:
            in_image = args[0]
            dis_opt = []
            lp_solver = 'linprog'
        elif nargs == 2:
            in_image = args[0]
            dis_opt = args[1]
            lp_solver = 'linprog'
        elif nargs == 3:
            in_image = args[0]
            dis_opt = args[1]
            lp_solver = args[2]
        else:
            raise Exception("Invalid number of input arguments, should be 1/2/3")

        if isinstance(in_image, ImageStar) is False:
            raise Exception("Input image is not an ImageStar")

        h, w = self.get_size_maxmap(in_image.V[:, :, 0, 0])
        startPoints = self.get_startpoints(in_image.V[:, :, 0, 0])
        max_index = [[[] for i in range(w)] for j in range(h)]

        pad_image = self.get_zero_padding_imagestar(in_image)

        np1 = pad_image.numPred
        l = 0
        for k in range(pad_image.numChannel):
            for i in range(h):
                for j in range(w):
                    max_index[i][j].append(pad_image.get_localMax_index(startPoints[i][j], self.PoolSize, k, lp_solver))
                    max_id = max_index[i][j][k]
                    if (len(max_id.shape) > 1) & (max_id.shape[0] > 1):
                        np1 += 1
                        l += 1

        new_V = np.zeros((h, w, pad_image.numChannel, pad_image.numPred + 1))
        new_pred_index = 0
        for k in range(pad_image.numChannel):
            for i in range(h):
                for j in range(w):
                    max_id = max_index[i][j][k]
                    if (len(max_id.shape) == 1) | ((len(max_id.shape) > 1) & (max_id.shape[0] == 1)):
                        for p in range(pad_image.numPred + 1):
                            new_V[i, j, k, p] = pad_image.V[max_id[0], max_id[1], k, p]
                    else:
                        new_V[i, j, k, 0] = 0
                        new_pred_index += 1
                        empty_V = np.zeros((h, w, pad_image.numChannel, 1))
                        new_V = np.concatenate((new_V, empty_V), axis=3)
                        new_V[i, j, k, pad_image.numPred + new_pred_index] = 1

        N = self.PoolSize[0] * self.PoolSize[1]
        new_C = np.zeros((new_pred_index * (N + 1), np1))
        new_d = np.zeros((new_pred_index * (N + 1), 1))
        new_pred_lb = np.zeros((new_pred_index, 1))
        new_pred_ub = np.zeros((new_pred_index, 1))
        new_pred_index = 0
        for k in range(pad_image.numChannel):
            for i in range(h):
                for j in range(w):
                    max_id = max_index[i][j][k]
                    if (len(max_id.shape) > 1) & (max_id.shape[0] > 1):
                        new_pred_index += 1
                        startpoint = startPoints[i][j]
                        points = pad_image.get_localPoints(startpoint, self.PoolSize)
                        C1 = np.zeros((1, np1))
                        C1[0, pad_image.numPred + new_pred_index - 1] = 1
                        lb, ub = pad_image.get_localBound(startpoint, self.PoolSize, k, lp_solver)
                        new_pred_lb[new_pred_index - 1, 0] = lb
                        new_pred_ub[new_pred_index - 1, 0] = ub
                        d1 = ub
                        C2 = np.zeros((N, np1))
                        d2 = np.zeros((N, 1))
                        for g in range(N):
                            point = points[g, :]
                            C2[g, 1:pad_image.numPred] = pad_image.V[point[0], point[1], k, 1:pad_image.numPred]
                            C2[g, pad_image.numPred + new_pred_index - 1] = -1
                            d2[g] = -pad_image.V[point[0], point[1], k, 0]

                        C = np.concatenate((C1, C2))
                        d = np.concatenate((np.array([[d1]]), d2))

                        new_C[(new_pred_index - 1) * (N + 1):new_pred_index * (N + 1), :] = C
                        new_d[(new_pred_index - 1) * (N + 1):new_pred_index * (N + 1)] = d

        n = pad_image.C.shape[0]
        C = np.concatenate((pad_image.C, np.zeros((n, new_pred_index))), axis=1)
        new_C = np.concatenate((C, new_C))
        new_d = np.concatenate((pad_image.d, new_d))
        new_pred_lb = np.concatenate((pad_image.pred_lb, new_pred_lb))
        new_pred_ub = np.concatenate((pad_image.pred_ub, new_pred_ub))

        image = ImageStar(new_V, new_C, new_d, new_pred_lb, new_pred_ub)
        return image

    def reach_star_approx_multipleinputs(self, *args):
        nargs = len(args)
        if nargs == 2:
            in_images = args[0]
            option = args[1]
            dis_opt = []
            lp_solver = 'linprog'
        elif nargs == 3:
            in_images = args[0]
            option = args[1]
            dis_opt = args[2]
            lp_solver = 'linprog'
        elif nargs == 4:
            in_images = args[0]
            option = args[1]
            dis_opt = args[2]
            lp_solver = args[3]
        else:
            Exception("Invalid number of input arguments")

        n = len(in_images)
        IS = []

        for i in range(n):
            IS.append(self.reach_star_approx(in_images[i], dis_opt, lp_solver))

        return IS

    def reach_star_exact(self, *args):
        nargs = len(args)
        if nargs == 2:
            in_image = args[0]
            option = args[1]
            dis_opt = None
            lp_solver = 'linprog'
        elif nargs == 3:
            in_image = args[0]
            option = args[1]
            dis_opt = args[2]
            lp_solver = 'linprog'
        elif nargs == 4:
            in_image = args[0]
            option = args[1]
            dis_opt = args[2]
            lp_solver = args[3]
        else:
            raise Exception('Invalid number of input arguments')

        if isinstance(in_image, ImageStar) is False:
            raise Exception("Input image is not an ImageStar")

        startPoints = self.get_startpoints(in_image.V[:, :, 0, 0])
        h, w = self.get_size_maxmap(in_image.V[:, :, 0, 0])
        pad_image = self.get_zero_padding_imagestar(in_image)

        # check max_id first
        max_index = np.zeros((h, w, pad_image.numChannel), dtype=object)
        maxMap_basis_V = np.zeros((h, w, pad_image.numChannel, pad_image.numPred + 1), dtype=np.single)
        split_pos = []

        maxidx = np.zeros((h, w, pad_image.numChannel), dtype=object)
        for k in range(pad_image.numChannel):
            for i in range(h):
                for j in range(w):
                    max_index[i, j, k] = pad_image.get_localMax_index(startPoints[i][j], self.PoolSize, k, lp_solver)
                    # construct the basis image for the maxMap
                    if len(max_index[i][j][k].shape) == 1:
                        maxMap_basis_V[i, j, k, :] = pad_image.V[max_index[i][j][k][0], max_index[i][j][k][1], k,
                                                     :].copy()
                        maxidx[i, j, k] = max_index[i, j, k]
                    else:
                        split_pos.append([i, j, k])

        n = len(split_pos)
        images = ImageStar(maxMap_basis_V, pad_image.C, pad_image.d, pad_image.pred_lb, pad_image.pred_ub)
        images.inheritMaxIdx(in_image.MaxIdxs)
        images.inheritInputSize(in_image.InputSizes)
        images.addMaxIdx(self.Name, maxidx)
        images.addInputSize(self.Name, [pad_image.height, pad_image.width])
        if n > 0:
            for i in range(n):
                if isinstance(images, list):
                    m1 = len(images)
                else:
                    m1 = 1
                images = self.stepSplitMultipleInputs(images, pad_image, split_pos[i],
                                                      max_index[split_pos[i][0]][split_pos[i][1]][split_pos[i][2]], [])
                if isinstance(images, list):
                    m2 = len(images)
                else:
                    m2 = 1
                # print('\nSplit %d images into %d images' % (m1, m2))
        return images

    def reach_star_exact_multipleInputs(self, *args):
        nargs = len(args)
        if nargs == 2:
            in_images = args[0]
            option = args[1]
            dis_opt = None
            lp_solver = 'linprog'
        elif nargs == 3:
            in_images = args[0]
            option = args[1]
            dis_opt = args[2]
            lp_solver = 'linprog'
        elif nargs == 4:
            in_images = args[0]
            option = args[1]
            dis_opt = args[2]
            lp_solver = args[3]
        else:
            raise Exception('Invalid number of input arguments')

        IS = []
        if isinstance(in_images, list):
            n = len(in_images)
            for i in range(n):
                temp1 = self.reach_star_exact(in_images[i], dis_opt, lp_solver)
                if isinstance(temp1, list):
                    IS.extend(temp1)
                else:
                    IS.append(temp1)
        elif isinstance(in_images, ImageStar):
            temp1 = self.reach_star_exact(in_images, dis_opt, lp_solver)
            if isinstance(temp1, list):
                IS.extend(temp1)
            else:
                IS.append(temp1)

        return IS

    def reach(self, im, me):
        in_images = im
        method = me
        option = []
        dis_opt = []
        lp_solver = 'linprog'

        if method == 'approx-star':
            IS = self.reach_star_approx(in_images, dis_opt, lp_solver)
        elif method == 'abs-dom':
            IS = self.reach_star_approx_multipleinputs(in_images, option, dis_opt, lp_solver)
        elif method == 'exact-star':
            IS = self.reach_star_exact_multipleInputs(in_images, option, dis_opt, lp_solver)

        return IS

    def parse(self, input):

        if isinstance(input, SeriesMaxpool) is False:
            Exception("Input is not a MaxPooling2d")

        L = SeriesMaxpool(input.Name, input.PoolSize, input.Stride, input.PaddingSize, input.HasUnpoolingOutputs,
                         input.NumInputs, input.InputNames, input.NumOutputs, input.OutputNames)
        return L


class SeriesReLU(object):

    def __init__(self, *args):
        nargs = len(args)
        if nargs == 5:
            self.Name = args[0]
            self.NumInputs = args[1]
            self.InputNames = args[2]
            self.NumOutputs = args[3]
            self.OutputNames = args[4]
        elif nargs == 1:
            self.Name = args[0]
        elif nargs == 0:
            self.Name = 'relu_layer'
        else:
            raise Exception("Invalid number of input")

    def evaluate(self, input):

        y = nn.Relu(input)
        return y

    def reach_star_exact(self, *args):
        # using Star to compute approximation reachability
        nargs = len(args)
        lp_solver = 'linprog'
        if nargs == 3:
            I = args[0]
            method = args[1]
            option = args[2]
        elif nargs == 2:
            I = args[0]
            method = args[1]
            option = 'single'
        elif nargs == 1:
            I = args[0]
            method = 'approx-star'
            option = 'single'
        else:
            raise Exception("Invalid number of input, should be 1/2/3")

        if isinstance(I, Star) is False:
            raise Exception("Input is not a Star")

        if I is None:
            S = []
        else:
            lb, ub = I.estimateRanges()
            if (lb is None) | (ub is None):
                S = []
            else:
                map = np.argwhere(ub <= 0)
                V = I.V.copy()
                V[map, :] = 0
                if hasattr(I, 'Z') is False:
                    new_Z = None
                elif I.Z is not None:
                    c1 = I.Z.c.copy()
                    c1[map, :] = 0
                    V1 = I.Z.V.copy()
                    V1[map, :] = 0
                    new_Z = Zono(c1, V1)
                else:
                    new_Z = None
                In = Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub, new_Z)
                map = np.argwhere(np.logical_and(lb < 0, ub > 0))
                m = len(map)
                for i in range(m):
                    In = self.stepReachMultipleInputs(In, map[i], option, lp_solver)
                S = In

        return S

    def reach_star_approx(self, *args):
        # using Star to compute approximation reachability
        nargs = len(args)
        if nargs == 3:
            I = args[0]
            method = args[1]
            option = args[2]
        elif nargs == 2:
            I = args[0]
            method = args[1]
            option = 'single'
        elif nargs == 1:
            I = args[0]
            method = 'approx-star'
            option = 'single'
        else:
            raise Exception("Invalid number of input, should be 1/2/3")

        if isinstance(I, Star) is False:
            raise Exception("Input is not a Star")

        if I is None:
            S = []
        else:
            lb, ub = I.estimateRanges()  # column vector
            if (lb is None) | (ub is None):
                S = []
            else:
                map1 = np.argwhere(ub <= 0)  # gives index
                map2 = np.argwhere(np.logical_and(lb < 0, ub > 0))
                xmax = I.getMaxs(map2)
                map3 = np.argwhere(xmax <= 0)
                n = len(map3)
                map4 = np.zeros((n, 1), dtype=map1.dtype)
                In_exist = True
                for i in range(n):
                    map4[i, 0] = map2[map3[i]]
                map11 = np.concatenate((map1, map4))
                In = I.resetRow(map11)  # reset to zero at the element having ub <= 0

                # find all indexes that have lb < 0 and ub > 0, then apply
                # the over-approximation rule for ReLU
                map5 = np.argwhere(xmax > 0)
                map6 = map2[map5[:]]  # all indexes having ub > 0
                xmax1 = xmax[map5[:]]  # upper bound of all neurons having ub > 0

                xmin = I.getMins(map6)
                map7 = np.argwhere(xmin < 0)
                map8 = map6[map7[:]]  # all indexes having lb < 0 & ub > 0
                lb1 = xmin[map7[:]]  # lower bound of all indexes having lb < 0 & ub > 0
                ub1 = xmax1[map7[:]]  # upper bound of all neurons having lb < 0 & ub > 0

                if In_exist:
                    S = self.multipleStepReachStarApprox_at_one(In, map8, lb1, ub1)
                else:
                    S = I
        return S

    def stepReachMultipleInputs(self, *args):
        nargs = len(args)
        if nargs == 4:
            I = args[0]
            index = args[1]
            option = args[2]
            lp_solver = args[3]
        elif nargs == 3:
            I = args[0]
            index = args[1]
            option = args[2]
            lp_solver = 'linprog'
        else:
            raise Exception("Invalid number of input, should be 3/4")

        if isinstance(I, Star):
            S = []
            temp = self.stepReach(I, index, lp_solver)
            if isinstance(temp, Star):
                S.append(temp)
            elif isinstance(temp, list):
                S.extend(temp)
        elif isinstance(I, list):
            p = len(I)
            S = []
            for i in range(p):
                temp = self.stepReach(I[i], index, lp_solver)
                if isinstance(temp, Star):
                    S.append(temp)
                elif isinstance(temp, list):
                    S.extend(temp)
        else:
            raise Exception("Wrong data type I")

        return S

    def stepReach(self, *args):
        nargs = len(args)
        if nargs == 3:
            I = args[0]
            index = args[1]
            lp_solver = args[2]
        elif nargs == 2:
            I = args[0]
            index = args[1]
            lp_solver = 'linprog'
        else:
            raise Exception("Invalid number of input, should be 2/3")

        if isinstance(I, Star) is False:
            raise Exception("Input is not a star set")

        xmin = I.getMin(index)

        if xmin >= 0:
            S = I
        else:
            xmax = I.getMax(index)
            if xmax <= 0:
                V1 = I.V.copy()
                V1[index, :] = 0
                if hasattr(I, 'Z') is False:
                    new_Z = None
                elif I.Z is not None:
                    c = I.Z.c.copy()
                    c[index] = 0
                    V = I.Z.V.copy()
                    V[index, :] = 0
                    new_Z = Zono(c, V)
                else:
                    new_Z = None
                S = Star(V1, I.C, I.d, I.predicate_lb, I.predicate_ub, new_Z)
            else:
                # S1 = I & x[index]<0
                c = I.V[index, 0].copy()
                V = I.V[index, 1:(I.nVar + 1)].copy()
                new_C = np.concatenate((I.C, V))
                if len(I.d.shape) == 2:
                    new_d = np.concatenate((I.d, -c[:, np.newaxis]))
                else:
                    new_d = np.concatenate((I.d, -c))
                new_V = I.V.copy()
                new_V[index, :] = np.zeros((1, I.nVar + 1))

                # update outer-zono
                if hasattr(I, 'Z') is False:
                    new_Z = None
                elif I.Z is not None:
                    c1 = I.Z.c.copy()
                    c1[index] = 0
                    V1 = I.Z.V.copy()
                    V1[index, :] = 0
                    new_Z = Zono(c1, V1)
                else:
                    new_Z = None

                S1 = Star(new_V, new_C, new_d, I.predicate_lb, I.predicate_ub, new_Z)

                # S2 = I & x[index]>=0
                new_C = np.concatenate((I.C, -V))
                if len(I.d.shape) == 2:
                    new_d = np.concatenate((I.d, c[:, np.newaxis]))
                else:
                    new_d = np.concatenate((I.d, c))
                S2 = Star(I.V, new_C, new_d, I.predicate_lb, I.predicate_ub, I.Z)

                S = [S1, S2]

        return S

    def multipleStepReachStarApprox_at_one(self, I, index, lb, ub):

        if isinstance(I, Star) is False:
            raise Exception("Input is not a Star")

        N = I.dim
        m = len(index)

        if m > 0:
            V1 = I.V
            V1[index, :] = 0
            V2 = np.zeros((N, m))
            for i in range(m):
                V2[index[i], i] = 1
            new_V = np.concatenate((V1, V2), axis=1)
            # case 0: keep the old constraints on the old predicate
            n = I.nVar
            C0 = np.concatenate((I.C, np.zeros((I.C.shape[0], m))), axis=1)
            d0 = I.d
            # case 1: y[index] >= 0
            C1 = np.concatenate((np.zeros((m, n)), -np.eye(m)), axis=1)
            d1 = np.zeros((m, 1))
            # case 2: y[index] >= x[index]
            C2 = np.concatenate((I.V[index, 1:n + 1], -V2[index, 0:m]), axis=1)
            d2 = -I.V[index, 0]
            # case 3: y[index] <= (ub/(ub-lb))*(x-lb)
            a = ub / (ub - lb)
            b = a * lb
            C3 = np.concatenate((-a * I.V[index, 1:n + 1], V2[index, 0:m]), axis=1)
            d3 = a * I.V[index, 0] - b

            new_C = np.concatenate((C0, C1, C2, C3))
            new_d = np.concatenate((d0, d1, d2, d3))
            new_pred_lb = np.concatenate((I.predicate_lb, np.zeros((m, 1))))
            new_pred_ub = np.concatenate((I.predicate_ub, ub))
            S = Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)
        else:
            S = Star(I.V.copy(), I.C.copy(), I.d.copy(), I.predicate_lb.copy(), I.predicate_ub.copy())

        return S

    def reach_star_single_input(self, in_image, method, option):

        if isinstance(in_image, ImageStar) is False:
            raise Exception("Input is not an ImageStar")

        h = in_image.height
        w = in_image.width
        c = in_image.numChannel
        if method == 'approx-star':
            Y = self.reach_star_approx(in_image.toStar(), method, option)
        elif method == 'exact-star':
            Y = self.reach_star_exact(in_image.toStar(), method, option)

        if isinstance(Y, Star):
            images = Y.toImageStar(h, w, c)
            images.inheritMaxIdx(in_image.MaxIdxs)
            images.inheritInputSize(in_image.InputSizes)
        elif isinstance(Y, list):
            n = len(Y)
            images = []
            for i in range(n):
                image = Y[i].toImageStar(h, w, c)
                image.inheritMaxIdx(in_image.MaxIdxs)
                image.inheritInputSize(in_image.InputSizes)
                images.append(image)

        return images

    def reach_star_multipleInputs(self, in_images, method, option):
        if isinstance(in_images, list) is False:
            images = self.reach_star_single_input(in_images, method, option)
        else:
            n = len(in_images)
            images = []
            for i in range(n):
                image = self.reach_star_single_input(in_images[i], method, option)
                if isinstance(image, ImageStar):
                    images.append(image)
                elif isinstance(image, list):
                    images.extend(image)

        return images

    def reach(self, *args):
        nargs = len(args)
        if nargs == 6:
            in_images = args[0]
            method = args[1]
            option = 'single'
            relaxFactor = 0
            dis_opt = []
            lp_solver = 'linprog'
        elif nargs == 5:
            in_images = args[0]
            method = args[1]
            option = 'single'
            relaxFactor = 0
            dis_opt = []
            lp_solver = 'linprog'
        elif nargs == 4:
            in_images = args[0]
            method = args[1]
            option = 'single'
            relaxFactor = 0
            dis_opt = []
            lp_solver = 'linprog'
        elif nargs == 3:
            in_images = args[0]
            method = args[1]
            option = 'single'
            relaxFactor = 0
            dis_opt = []
            lp_solver = 'linprog'
        elif nargs == 2:
            in_images = args[0]
            method = args[1]
            option = 'single'
            relaxFactor = 0
            dis_opt = []
            lp_solver = 'linprog'
        else:
            raise Exception("Invalid number of input, should be 2/3/4/5/6")

        images = self.reach_star_multipleInputs(in_images, method, option)
        return images

    def parse(self, relu_layer):

        if isinstance(relu_layer, SeriesReLU) is False:
            raise Exception("Input is not a ReluLayer")

        L = SeriesReLU(relu_layer.Name, relu_layer.NumInputs, relu_layer.InputNames,
                      relu_layer.NumOutputs, relu_layer.OutputNames)
        return L
