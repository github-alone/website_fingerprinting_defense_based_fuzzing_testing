# -*- coding: utf-8 -*-

import random
from collections import defaultdict
import numpy as np
from datetime import datetime
from keras import backend as K
from keras.models import Model
from keras.models import load_model
import pickle as pickle
import Levenshtein
import heapq
import csv

model_layer_weights_top_k = []

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def init_coverage_tables(model,dnn):
    model_layer_dict = defaultdict(bool)
    init_dict(model,model_layer_dict,dnn)
    return model_layer_dict

def init_coverage_tables(model1, model2, model3,dnn):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1,dnn)
    init_dict(model2, model_layer_dict2,dnn)
    init_dict(model3, model_layer_dict3,dnn)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3

# def init_coverage_tables(model1,dnn):
#     model_layer_dict1 = defaultdict(bool)
#     init_dict(model1, model_layer_dict1,dnn)
#     return model_layer_dict1


def init_dict(model, model_layer_dict,dnn):
    if dnn == 'df':
        layer_names = ['block1_adv_act1', 'block1_adv_act2', "block1_pool", "block1_dropout",
                       'block2_act1', 'block2_act2', "block2_pool", "block2_dropout",
                       'block3_act1', 'block3_act2', "block3_pool", "block3_dropout",
                       'block4_act1', 'block4_act2', "block4_pool", "block4_dropout",
                       'fc1_act', 'fc1_dropout', 'fc2_act', 'fc2_dropout']
    else:
        layer_names = ['layer1_relu', 'layer1_pool',
                       'layer2_block1_conv2', "layer2_block1_relu2", "layer2_block2_relu1",'layer2_block2_relu2',
                       'layer3_block1_relu1', "layer3_block1_relu2", "layer3_block2_relu1",'layer3_block2_relu2',
                       'layer4_block1_relu1', "layer4_block1_relu2", "layer4_block2_relu1",'layer4_block2_relu2',
                       'layer5_block1_relu1', "layer5_block1_relu2", "layer5_block2_relu1",'layer5_block2_relu2',
                       'average_pool']
    for layer in model.layers:
        if layer.name in layer_names:
            for index in range(layer.output_shape[-1]):
                model_layer_dict[(layer.name, index)] = False

def init_coverage_times(model,dnn):
    model_layer_times = defaultdict(int)
    init_times(model,model_layer_times,dnn)
    return model_layer_times

def init_coverage_value(model,dnn):
    model_layer_value = defaultdict(float)
    init_times(model, model_layer_value,dnn)
    return model_layer_value

def init_times(model,model_layer_times,dnn):
    # for layer in model.layers:
    #     if 'flatten' in layer.name or 'input' in layer.name:
    #         continue
    if dnn == 'df':
        layer_names = ['block1_adv_act1', 'block1_adv_act2', "block1_pool", "block1_dropout",
                       'block2_act1', 'block2_act2', "block2_pool", "block2_dropout",
                       'block3_act1', 'block3_act2', "block3_pool", "block3_dropout",
                       'block4_act1', 'block4_act2', "block4_pool", "block4_dropout",
                       'fc1_act', 'fc1_dropout', 'fc2_act', 'fc2_dropout']
    else:
        layer_names = ['layer1_relu', 'layer1_pool',
                       'layer2_block1_conv2', "layer2_block1_relu2", "layer2_block2_relu1",'layer2_block2_relu2',
                       'layer3_block1_relu1', "layer3_block1_relu2", "layer3_block2_relu1",'layer3_block2_relu2',
                       'layer4_block1_relu1', "layer4_block1_relu2", "layer4_block2_relu1",'layer4_block2_relu2',
                       'layer5_block1_relu1', "layer5_block1_relu2", "layer5_block2_relu1",'layer5_block2_relu2',
                       'average_pool']
    for layer in model.layers:
        if layer.name in layer_names:
            for index in range(layer.output_shape[-1]):
                model_layer_times[(layer.name, index)] = 0

def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def neuron_to_cover(not_covered,model_layer_dict):
    if not_covered:
        layer_name, index = random.choice(not_covered)
        not_covered.remove((layer_name, index))
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index

def random_strategy(model,model_layer_times, neuron_to_cover_num):
    loss_neuron = []
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_times.items() if v == 0]
    for _ in range(neuron_to_cover_num):
        layer_name, index = neuron_to_cover(not_covered, model_layer_times)
        loss00_neuron = K.mean(model.get_layer(layer_name).output[..., index])
        # if loss_neuron == 0:
        #     loss_neuron = loss00_neuron
        # else:
        #     loss_neuron += loss00_neuron
        # loss_neuron += loss1_neuron
        loss_neuron.append(loss00_neuron)
    return loss_neuron

def neuron_select_high_weight(model, layer_names, top_k):
    global model_layer_weights_top_k
    model_layer_weights_dict = {}
    for layer_name in layer_names:
        weights = model.get_layer(layer_name).get_weights()
        if len(weights) <= 0:
            continue
        w = np.asarray(weights[0])  # 0 is weights, 1 is biases
        w = w.reshape(w.shape)
        for index in range(model.get_layer(layer_name).output_shape[-1]):
            index_w = np.mean(w[..., index])
            if index_w <= 0:
                continue
            model_layer_weights_dict[(layer_name,index)]=index_w
    # notice!
    model_layer_weights_list = sorted(model_layer_weights_dict.items(), key=lambda x: x[1], reverse=True)

    k = 0
    for (layer_name, index),weight in model_layer_weights_list:
        if k >= top_k:
            break
        model_layer_weights_top_k.append([layer_name,index])
        k += 1


def neuron_selection(model, model_layer_times, model_layer_value, neuron_select_strategy, neuron_to_cover_num, threshold):
    if neuron_select_strategy == 'None':
        return random_strategy(model, model_layer_times, neuron_to_cover_num)

    # 将输入的字符转为数字
    num_strategy = len([x for x in neuron_select_strategy if x in ['0', '1', '2', '3','4']])
    neuron_to_cover_num_each = neuron_to_cover_num // num_strategy

    loss_neuron = []
    # initialization for strategies
    if ('0' in list(neuron_select_strategy)) or ('1' in list(neuron_select_strategy)) or ('2' in list(neuron_select_strategy)):
        i = 0
        neurons_covered_times = []
        neurons_key_pos = {}
        for (layer_name, index), time in model_layer_times.items():
            neurons_covered_times.append(time)
            neurons_key_pos[i] = (layer_name, index)
            i += 1
        # mp.asarray函数功能：将结构数据转为ndarray
        neurons_covered_times = np.asarray(neurons_covered_times)
        times_total = sum(neurons_covered_times)
        print("times_total is :",times_total)

    # python 中 //是整除，/是正常除法
    # select neurons covered often
    if '0' in list(neuron_select_strategy):
        if times_total == 0:
            return random_strategy(model, model_layer_times, 1)#The beginning of no neurons covered
        # neurons_covered_percentage = neurons_covered_times / float(times_total)
        # num_neuron0 = np.random.choice(range(len(neurons_covered_times)), p=neurons_covered_percentage)
        # replace=false表示不能取相同的数字
        # num_neuron0 = np.random.choice(range(len(neurons_covered_times)), neuron_to_cover_num_each, replace=False, p=neurons_covered_percentage)
        num_neuron0 = heapq.nlargest(neuron_to_cover_num_each, range(len(neurons_covered_times)), neurons_covered_times.__getitem__)
        for num in num_neuron0:
            layer_name0, index0 = neurons_key_pos[num]
            # 将该层中的第index0个神经元值添加到loss_neuron中，神经元的值是当前n*m的二维向量中所有值求平均后的值
            loss0_neuron = K.mean(model.get_layer(layer_name0).output[..., index0])
            loss_neuron.append(loss0_neuron)

    # select neurons covered rarely
    if '1' in list(neuron_select_strategy):
        if times_total == 0:
            return random_strategy(model, model_layer_times, 1)
        # np.subtract(x1,x2)返回x1-x2的结果，做减法，如果 x1.shape != x2.shape ，它们必须可以广播到一个公共形状(成为输出的形状)
        # neurons_covered_times_inverse = np.subtract(max(neurons_covered_times), neurons_covered_times)
        # neurons_covered_percentage_inverse = neurons_covered_times_inverse / float(sum(neurons_covered_times_inverse))
        # num_neuron1 = np.random.choice(range(len(neurons_covered_times)), p=neurons_covered_percentage_inverse)
        # num_neuron1 = np.random.choice(range(len(neurons_covered_times)), neuron_to_cover_num_each, replace=False,
        #                                p=neurons_covered_percentage_inverse)
        num_neuron1 = heapq.nsmallest(neuron_to_cover_num_each, range(len(neurons_covered_times)),
                                     neurons_covered_times.__getitem__)
        for num in num_neuron1:
            layer_name1, index1 = neurons_key_pos[num]
            loss1_neuron = K.mean(model.get_layer(layer_name1).output[..., index1])
            loss_neuron.append(loss1_neuron)

    # select neurons covered rarely and often
    if '2' in list(neuron_select_strategy):
        if times_total == 0:
            return random_strategy(model, model_layer_times, 1)
        # np.subtract(x1,x2)返回x1-x2的结果，做减法，如果 x1.shape != x2.shape ，它们必须可以广播到一个公共形状(成为输出的形状)
        # neurons_covered_times_inverse = np.subtract(max(neurons_covered_times), neurons_covered_times)
        # neurons_covered_percentage_inverse = neurons_covered_times_inverse / float(sum(neurons_covered_times_inverse))
        # num_neuron1 = np.random.choice(range(len(neurons_covered_times)), p=neurons_covered_percentage_inverse)
        # num_neuron1 = np.random.choice(range(len(neurons_covered_times)), neuron_to_cover_num_each, replace=False,
        #                                p=neurons_covered_percentage_inverse)
        num_neuron1 = heapq.nsmallest(neuron_to_cover_num_each, range(len(neurons_covered_times)),
                                      neurons_covered_times.__getitem__)
        num_neuron2 = heapq.nlargest(neuron_to_cover_num_each, range(len(neurons_covered_times)),
                                      neurons_covered_times.__getitem__)
        for num in num_neuron1:
            layer_name1, index1 = neurons_key_pos[num]
            loss1_neuron = K.mean(model.get_layer(layer_name1).output[..., index1])
            loss_neuron.append(loss1_neuron)
        for num in num_neuron2:
            layer_name1, index1 = neurons_key_pos[num]
            loss1_neuron = K.mean(model.get_layer(layer_name1).output[..., index1])
            loss_neuron.append(loss1_neuron)

    # select neurons with largest weights (feature maps with largest filter weights)
    if '3' in list(neuron_select_strategy):
        layer_names = [layer.name for layer in model.layers if
                       'flatten' not in layer.name and 'input' not in layer.name]
        k = 0.1
        top_k = k * len(model_layer_times)  # number of neurons to be selected within
        global model_layer_weights_top_k
        if len(model_layer_weights_top_k) == 0:
            neuron_select_high_weight(model, layer_names, top_k)  # Set the value

        num_neuron2 = np.random.choice(range(len(model_layer_weights_top_k)), neuron_to_cover_num_each, replace=False)
        for i in num_neuron2:
            # i = np.random.choice(range(len(model_layer_weights_top_k)))
            layer_name2 = model_layer_weights_top_k[i][0]
            index2 = model_layer_weights_top_k[i][1]
            loss2_neuron = K.mean(model.get_layer(layer_name2).output[..., index2])
            loss_neuron.append(loss2_neuron)

    if '4' in list(neuron_select_strategy):
        above_threshold = []
        below_threshold = []
        above_num = neuron_to_cover_num_each / 2
        below_num = neuron_to_cover_num_each - above_num
        above_i = 0
        below_i = 0
        for (layer_name, index), value in model_layer_value.items():
            if threshold + 0.25 > value > threshold and layer_name != 'fc1' and layer_name != 'fc2' and \
                    layer_name != 'predictions' and layer_name != 'fc1000' and layer_name != 'before_softmax' \
                    and above_i < above_num:
                above_threshold.append([layer_name, index])
                above_i += 1
                # print(layer_name,index,value)
                # above_threshold_dict[(layer_name, index)]=value
            elif threshold > value > threshold - 0.2 and layer_name != 'fc1' and layer_name != 'fc2' and \
                    layer_name != 'predictions' and layer_name != 'fc1000' and layer_name != 'before_softmax' \
                    and below_i < below_num:
                below_threshold.append([layer_name, index])
                below_i += 1
        #
        # loss3_neuron_above = 0
        # loss3_neuron_below = 0
        loss_neuron = []
        if len(above_threshold) > 0:
            for above_item in range(len(above_threshold)):
                loss_neuron.append(K.mean(
                    model.get_layer(above_threshold[above_item][0]).output[..., above_threshold[above_item][1]]))

        if len(below_threshold) > 0:
            for below_item in range(len(below_threshold)):
                loss_neuron.append(-K.mean(
                    model.get_layer(below_threshold[below_item][0]).output[..., below_threshold[below_item][1]]))

        # loss_neuron += loss3_neuron_below - loss3_neuron_above

        # for (layer_name, index), value in model_layer_value.items():
        #     if 0.5 > value > 0.25:
        #         above_threshold.append([layer_name, index])
        #     elif 0.25 > value > 0.2:
        #         below_threshold.append([layer_name, index])
        # loss3_neuron_above = 0
        # loss3_neuron_below = 0
        # if len(above_threshold)>0:
        #     above_i = np.random.choice(range(len(above_threshold)))
        #     loss3_neuron_above = K.mean(model.get_layer(above_threshold[above_i][0]).output[..., above_threshold[above_i][1]])
        # if len(below_threshold)>0:
        #     below_i = np.random.choice(range(len(below_threshold)))
        #     loss3_neuron_below = K.mean(model.get_layer(below_threshold[below_i][0]).output[..., below_threshold[below_i][1]])
        # loss_neuron += loss3_neuron_below - loss3_neuron_above
        if loss_neuron == 0:
            return random_strategy(model, model_layer_times, 1)  # The beginning of no neurons covered

    return loss_neuron

def neuron_scale(loss_neuron):
    loss_neuron_new = []
    loss_sum = K.sum(loss_neuron)
    for loss_each in loss_neuron:
        loss_each /= loss_sum
        loss_neuron_new.append(loss_each)
    return loss_neuron_new

def neuron_scale_maxmin(loss_neuron):
    max_loss = K.max(loss_neuron)
    min_loss = K.min(loss_neuron)
    base = max_loss - min_loss
    loss_neuron_new = []
    for loss_each in loss_neuron:
        loss_each_new = (loss_each - min_loss) / base
        loss_neuron_new.append(loss_each_new)
    return loss_neuron_new

def neuron_covered(model_layer_times):
    covered_neurons = len([v for v in model_layer_times.values() if v > 0])
    total_neurons = len(model_layer_times)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def update_coverage(input_data, model, model_layer_times,threshold,dnn):
    # DF模型中一共有二十层是神经元
    if dnn == 'df':
        layer_names = ['block1_adv_act1', 'block1_adv_act2', "block1_pool", "block1_dropout",
                       'block2_act1', 'block2_act2', "block2_pool", "block2_dropout",
                       'block3_act1', 'block3_act2', "block3_pool", "block3_dropout",
                       'block4_act1', 'block4_act2', "block4_pool", "block4_dropout",
                       'fc1_act', 'fc1_dropout', 'fc2_act', 'fc2_dropout']
    else:
        layer_names = ['layer1_relu', 'layer1_pool',
                       'layer2_block1_conv2', "layer2_block1_relu2", "layer2_block2_relu1",'layer2_block2_relu2',
                       'layer3_block1_relu1', "layer3_block1_relu2", "layer3_block2_relu1",'layer3_block2_relu2',
                       'layer4_block1_relu1', "layer4_block1_relu2", "layer4_block2_relu1",'layer4_block2_relu2',
                       'layer5_block1_relu1', "layer5_block1_relu2", "layer5_block2_relu1",'layer5_block2_relu2',
                       'average_pool']
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold: #and model_layer_dict[(layer_names[i], num_neuron)] == 0:
                model_layer_times[(layer_names[i], num_neuron)] += 1

    return model_layer_times


def update_coverage_value(input_data, model, model_layer_value,dnn):
    if dnn == 'df':
        layer_names = ['block1_adv_act1', 'block1_adv_act2', "block1_pool", "block1_dropout",
                       'block2_act1', 'block2_act2', "block2_pool", "block2_dropout",
                       'block3_act1', 'block3_act2', "block3_pool", "block3_dropout",
                       'block4_act1', 'block4_act2', "block4_pool", "block4_dropout",
                       'fc1_act', 'fc1_dropout', 'fc2_act', 'fc2_dropout']
    else:
        layer_names = ['layer1_relu', 'layer1_pool',
                       'layer2_block1_conv2', "layer2_block1_relu2", "layer2_block2_relu1",'layer2_block2_relu2',
                       'layer3_block1_relu1', "layer3_block1_relu2", "layer3_block2_relu1",'layer3_block2_relu2',
                       'layer4_block1_relu1', "layer4_block1_relu2", "layer4_block2_relu1",'layer4_block2_relu2',
                       'layer5_block1_relu1', "layer5_block1_relu2", "layer5_block2_relu1",'layer5_block2_relu2',
                       'average_pool']

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        # xrange(scaled.shape[-1])
        for num_neuron in range(scaled.shape[-1]):
            model_layer_value[(layer_names[i], num_neuron)] = np.mean(scaled[..., num_neuron])

    return intermediate_layer_outputs
    # return model_layer_value


# 挑选样本：从test数据集中筛选出每个类别limits条流量，一共95 * limits条流量（DF识别准确率>95%)
def generate_good_sample(dnn, data, labels):
    limits = 50
    new_data = []
    new_labels = []
    minlen = 150
    num_traces = {}
    for x, m in zip(data, labels):
        if m not in num_traces:
            num_traces[m] = 0
        count = num_traces[m]
        if count < limits:
            new_x = x.reshape((1, x.shape[0], 1))
            if len(x) >= minlen and m == np.argmax(dnn.predict(new_x)) and np.max(dnn.predict(new_x)) >= 0.95:
                new_data.append(x)
                new_labels.append(m)
                num_traces[m] = count + 1
    new_data = np.array(new_data)
    new_labels = np.array(new_labels)
    if not new_data.size:
        raise ValueError('After filtering, no sequence left.')
    del num_traces
    path1 = "./Dataset/good_samples/95w_{}tra.pkl".format(limits)  # 文件路径
    path2 = "./Dataset/good_samples/95w_{}lab.pkl".format(limits)

    output1 = open(path1, 'wb')
    output2 = open(path2, 'wb')
    pickle.dump(new_data, output1)
    pickle.dump(new_labels, output2)
    output1.close()
    output2.close()
    print("new data shape", new_data.shape)
    print("new labels shape", new_labels.shape)
    print("new labels:", new_labels)
    # np.savez_compressed("../dataset/cleng/good_samples/{}_95w_{}tr.npz".format(dnn, limits),data=new_data,labels=new_labels)
    print("samples saved to ../Dataset/good_samples/{}_95w_{}tra.pkl".format(dnn, limits))
    print("samples saved to ../Dataset/good_samples/{}_95w_{}lab.pkl".format(dnn, limits))

def LoadGoodSample():
    print("Loading non-defended Dataset for closed-world scenario" )
    # Point to the directory storing data
    dataset_dir = './Dataset/good_samples/ClosedWorld/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load good sample
    with open(dataset_dir + '95w_20tra.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + '95w_20lab.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle,encoding='iso-8859-1'))

    print ("Data dimensions:")
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)

    # return X_train, y_train, X_valid, y_valid, X_test, y_test
    return X_test, y_test

# Load data for non-defended Dataset for CW setting
def LoadDataNoDefCW():
    print("Loading non-defended Dataset for closed-world scenario" )
    # Point to the directory storing data
    dataset_dir = 'Dataset/ClosedWorld/Nodef/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_NoDef.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_train_NoDef.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle,encoding='iso-8859-1'))

    # Load validation data
    with open(dataset_dir + 'X_valid_NoDef.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_valid_NoDef.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle,encoding='iso-8859-1'))

    # Load testing data
    with open(dataset_dir + 'X_test_NoDef.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_test_NoDef.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle,encoding='iso-8859-1'))

    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test
    # return X_train, y_train,X_test, y_test

def get_Levenshtein_distance(i,gen_tra, gen_label,data_test,perClass_trace_num):
    leven_distance = []
    temp_tra = gen_tra
    temp_tra = temp_tra.reshape(temp_tra.shape[1])
    temp_data = data_test

    ori_gen_distance = Levenshtein.distance(temp_tra,temp_data[i].reshape(temp_data[i].shape[0]))
    for j in range (gen_label * perClass_trace_num,(gen_label+1) * perClass_trace_num):
        distance1 = Levenshtein.distance(temp_tra,temp_data[j].reshape(temp_data[j].shape[0]))
        leven_distance.append(distance1)
    gen_mean_distance = sum(leven_distance)/len(leven_distance)
    return ori_gen_distance,gen_mean_distance



def get_ClosedWorld_layer_value(data,label,model_layer_values,model1,dnn,model_path):
    # model_layer_time文件路径
    model_layer_values_path = 'Model_layer/{}/ClosedWorld_model_layer_value.csv'.format(dnn)
    f = open(model_layer_values_path, 'a', encoding='utf-8', newline='')
    writer = csv.writer(f)
    # model_layer_values_list = []
    for i in range(label.shape[0]):
        print('the model_layer_values of {}th trace is counting!'.format(i))
        temp_x = data[i]
        temp_x = temp_x.reshape((1,data[i].shape[0],1))
        update_coverage_value(temp_x, model1, model_layer_values,dnn)
        # model_layer_values_list.append(model_layer_values)
        neurons_covered_value = []
        layer_name_index = []
        for (layer_name, index), time in model_layer_values.items():
            neurons_covered_value.append(time)
            str1 = str(layer_name)+','+str(index)
            layer_name_index.append(str1)
        # 写入列名，只写一次
        if i == 0:
            writer.writerow(layer_name_index)
        writer.writerow(neurons_covered_value)
        f.flush()
        # 加快计算速度，因为会向graph中增加新的节点
        if i % 200 == 99:
            K.clear_session()
            model1 = load_model(model_path)
    f.close()
    return model_layer_values

def get_OpenWorld_layer_value(data,label,model_layer_values,model1,dnn,model_path):
    # model_layer_time文件路径
    model_layer_values_path = 'Model_layer/{}/OpenWorld_model_layer_value.csv'.format(dnn)
    f = open(model_layer_values_path, 'a', encoding='utf-8', newline='')
    writer = csv.writer(f)
    # model_layer_values_list = []
    for i in range(label.shape[0]):
        print('the model_layer_values of {}th trace is counting!'.format(i))
        temp_x = data[i]
        temp_x = temp_x.reshape((1,data[i].shape[0],1))
        update_coverage_value(temp_x, model1, model_layer_values,dnn)
        # model_layer_values_list.append(model_layer_values)
        neurons_covered_value = []
        layer_name_index = []
        for (layer_name, index), time in model_layer_values.items():
            neurons_covered_value.append(time)
            str1 = str(layer_name)+','+str(index)
            layer_name_index.append(str1)
        # 写入列名，只写一次
        if i == 0:
            writer.writerow(layer_name_index)
        writer.writerow(neurons_covered_value)
        f.flush()
        # 加快计算速度，因为会向graph中增加新的节点
        if i % 200 == 99:
            K.clear_session()
            model1 = load_model(model_path)
    f.close()
    return model_layer_values

# def get_layer_times(data,label,model_layer_time,threshold,model1):
#     # model_layer_time文件路径
#     model_layer_times_path = './Model_Layer_times/model_layer_time1.csv'
#     f = open(model_layer_times_path, 'a', encoding='utf-8', newline='')
#     writer = csv.writer(f)
#     model_layer_times_list = []
#     for i in range(label.shape[0]):
#         print('the model_layer_times of {}th trace is counting!'.format(i))
#         temp_x = data[i]
#         temp_x = temp_x.reshape((1,data[i].shape[0],1))
#         update_coverage(temp_x, model1, model_layer_time, threshold,dnn)
#         # 100是test数据中每个类别的流量数
#         if i % 100 == 99:
#             model_layer_times_list.append(model_layer_time)
#             neurons_covered_times = []
#             layer_name_index = []
#             for (layer_name, index), time in model_layer_time.items():
#                 neurons_covered_times.append(time)
#                 str1 = str(layer_name)+','+str(index)
#                 layer_name_index.append(str1)
#             # 写入列名，只写一次
#             if i == 99:
#                 writer.writerow(layer_name_index)
#             writer.writerow(neurons_covered_times)
#             f.flush()
#             # 只记录最后每个类别的最后一条流量的统计情况
#             model_layer_time = init_coverage_times(model1)
#         # 加快计算速度，因为会向graph中增加新的节点
#         if i % 200 == 99:
#             K.clear_session()
#             model1 = load_model('./Model/ClosedWorld_NoDef.h5')
#
#     return model_layer_times_list

