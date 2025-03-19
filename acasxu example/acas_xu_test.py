import onnx
import onnxruntime as ort
from onnx2pytorch import ConvertModel
import re
import numpy as np
import torch
from torchsummary import summary
import time
from scipy import optimize

from github.engine.series_netload import series_dnn_diff
from github.engine.netload import compute_net_diff


def structured_vnnlib_parser(file_path):
    with open(file_path, "r") as file:
        data = file.read()

    # Extract declared input/output variables
    input_vars = re.findall(r"\(declare-const (X_\d+) Real\)", data)
    output_vars = re.findall(r"\(declare-const (Y_\d+) Real\)", data)

    # Extract input constraints
    input_constraints = re.findall(r"\(assert \(and (.+?)\)\)", data, re.DOTALL)

    # Extract output constraints
    output_constraints = re.findall(r"\(assert \((.+?)\)\)", data)

    return input_vars, output_vars, input_constraints, output_constraints


def find_maximal_disc(star_r, star_c):
    c_diff = star_r.V[0, 0, :, 0] - star_c.V[0, 0, :, 0]
    v_diff = star_r.V[0, 0, :, 1:] - star_c.V[0, 0, :, 1:]
    C_mat = star_c.C[1:, :]
    d_mat = star_c.d[1:, :].squeeze()
    max_temp = 0

    for index in range(star_c.numChannel):
        fun_max = - v_diff[index, :]
        fun_min = v_diff[index, :]
        # pos_result = optimize.linprog(fun_max, A_ub=C_mat, b_ub=d_mat, A_eq=C_mat, b_eq=d_mat)
        # neg_result = optimize.linprog(fun_min, A_ub=C_mat, b_ub=d_mat, A_eq=C_mat, b_eq=d_mat)
        pos_result = optimize.linprog(fun_max, A_ub=C_mat, b_ub=d_mat, bounds=(-1, 1))
        neg_result = optimize.linprog(fun_min, A_ub=C_mat, b_ub=d_mat, bounds=(-1, 1))
        max_val = c_diff[index] + np.dot(fun_max, pos_result.x)
        min_val = c_diff[index] + np.dot(fun_min, neg_result.x)
        max_temp = max(max_temp, max(abs(max_val), abs(min_val)))

    return max_temp


if __name__ == "__main__":
    model_path1 = './onnx/ACASXU_run2a_1_2_batch_2000.onnx'
    onnx_model1 = onnx.load(model_path1)
    onnx.checker.check_model(model_path1)
    ort_session1 = ort.InferenceSession(model_path1)
    net1 = ConvertModel(onnx_model1)

    model_path2 = './onnx/ACASXU_run2a_1_3_batch_2000.onnx'
    onnx_model2 = onnx.load(model_path2)
    onnx.checker.check_model(model_path2)
    ort_session2 = ort.InferenceSession(model_path2)
    net2 = ConvertModel(onnx_model2)

    ori_x = np.array([0.64, 0., 0., -0.475, -0.475])
    ori_x = ori_x[:, np.newaxis, np.newaxis]
    LB = np.array([0.6, -0.5, -0.5, 0.45, -0.5])
    LB = LB[:, np.newaxis, np.newaxis]
    UB = np.array([0.679857769, 0.5, 0.5, 0.5, -0.45])
    UB = UB[:, np.newaxis, np.newaxis]

    print("ACAX SU Dataset with Series Method")
    start_total = time.time()
    max_discrepancy = 0
    Istar = series_dnn_diff(net1, ori_x, LB, UB)
    print(len(Istar))
    for i in range(len(Istar)):
        print("Build %d of %d ImageStar" % (i + 1, len(Istar)))
        Istar_temp = series_dnn_diff(net2, ori_x, LB, UB, opt='rebuild', Istar=Istar[i])
        print(len(Istar_temp))
        for j in range(len(Istar_temp)):
            max_temp = find_maximal_disc(Istar[i], Istar_temp[j])
            max_discrepancy = max(max_discrepancy, max_temp)
    print(max_discrepancy)
    end_total = time.time()
    print("Time: " + str(end_total - start_total))

    # print(onnx.helper.printable_graph(onnx_model.graph))
    # print(pytorch_model)
    # ort_session = ort.InferenceSession(model_path)
    # input_vec = np.array([0.62, 0.2, -0.499, 0.48, 0.46], dtype=np.float32)
    # ort_inputs = {'input': input_vec[np.newaxis, np.newaxis, np.newaxis, :]}
    # ort_output = ort_session.run(None, ort_inputs)
    # print(ort_output)

    # summary(pytorch_model, (1, 5, 1))

    # input_tensor = torch.tensor(input_vec[np.newaxis, np.newaxis, np.newaxis, :])
    # ans = pytorch_model(input_tensor)
    # print(ans)

    # for name, m in pytorch_model.named_children():
    #     print("Name: " + name, m)
    # attr1 = getattr(pytorch_model, name)
    # print(attr1.weight)
    # print(pytorch_model.MatMul_linear_7_Add.weight)

    # input_vars, output_vars, input_constraints, output_constraints = structured_vnnlib_parser("prop_1.vnnlib")
    #
    # print("Declared Input Variables:", input_vars)
    # print("Declared Output Variables:", output_vars)
    # print("Input Constraints:", input_constraints)
    # print("Output Constraints:", output_constraints)

    # densenet_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    # print(densenet_model)
