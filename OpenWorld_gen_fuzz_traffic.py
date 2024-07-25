from utility import *
from utils_tmp import *
from keras.models import load_model
import numpy as np
import pandas as pd
import sys
import os
import csv
import time
import heapq

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

length = 5000
input_shape = (length, 1)
monitored_class_num = 95

class_per_trace_num = 20
# 定义一次插入包的个数
once_insert_time = 3
unmonitor_class_label = 95

model = load_model('./Model/df_ow_NoDef.h5')
# 网络将learning_phase设为0，表示predict模式;设置为1 表示train模式
K.set_learning_phase(0)
neuron_select_strategy = sys.argv[1]
threshold = float(sys.argv[2])

# neuron_to_cover_num 指定激活的神经元个数
neuron_to_cover_num = int(sys.argv[3])
# subdir 是指定今天的日期，今天跑的实验生成的流量放在此文件夹中,比如输入0113
subdir = sys.argv[4]
# dnn是当前需要预测的模型，只可输入df和varcnn
dnn = sys.argv[5]
neuron_to_cover_weight = 1/(neuron_to_cover_num * threshold)

# log文件路径
# log文件路径
logfile_path = "./Logfile/{}/OpenWorld/'{}_{}_{}.out".format(dnn,subdir, neuron_select_strategy,str(threshold),str(neuron_to_cover_num))

l = open(logfile_path, "a")

# csv文件路径
csv_path = './result/{}/OpenWorld/{}_{}_{}_{}.csv'.format(dnn,subdir,neuron_select_strategy, str(threshold),
                                                          str(neuron_to_cover_num))
f1 = open(csv_path, 'a', encoding='utf-8', newline='')
writer1 = csv.writer(f1)
# 写入csv的列名，并将列名写入表头
# writer1.writerow('obj=loss2-loss1 + loss_neuron')
csv_head = ['i', 'ori_class', 'ori_probability', 'ori_tra_len', 'evading_class', 'evading_evadingClass_probability',
            'evading_oriClass_probability', 'insert_times', 'bandwidth_overhead(%)', 'insert_indexes', 'insert_numbers',
            'spending_time(seconds)', 'total_neurons', 'ori_covered_neurons',
            'gen_covered_neurons', 'ori_coverage', 'gen_coverage']
writer1.writerow(csv_head)
# flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入。
f1.flush()

# model_layer_time和model_layer_value文件路径
model_layer_values_path = 'Model_layer/{}/OpenWorld_model_layer_value.csv'.format(dnn)
if dnn == 'df':
    model_path = './Model/df_ow_NoDef.h5'
else :
    model_path = './Model/var-cnn_ow_NoDef.h5'
X_goodSample, y_goodSample = LoadGoodSampleOW()
# Convert data as float32 type
X_goodSample = X_goodSample.astype('float32')
y_goodSample = y_goodSample.astype('float32')

X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon = LoadDataNoDefOW_Evaluation()
# Convert data as float32 type
X_test_Mon = X_test_Mon.astype('float32')
y_test_Mon = y_test_Mon.astype('float32')
X_test_Unmon = X_test_Unmon.astype('float32')
y_test_Unmon = y_test_Unmon.astype('float32')

traffic_num = y_goodSample.shape[0]

# tra_evad_data：保存插入后成功逃逸的流量
# tra_evad_label:保存插入后成功逃逸流量的label
# insert_num_list:保存当前流量插入的值，通常为1
# insert_index_list：保存当前流量插入的位置
tra_evad_data = []
tra_evad_label = []

insert_num_list = [[] for _ in range(traffic_num)]
insert_index_list = [[] for _ in range(traffic_num)]
result_list = []

model_layer_times1 = init_coverage_times(model,dnn)
model_layer_value1 = init_coverage_value(model,dnn)

def curtime():
    return datetime.utcnow().strftime('%d.%m %H:%M:%S')

def log(str):
    # print("> {}".format(str))
    l.write("{}>\t{}\n".format(curtime(), str))
    l.flush()

def log_close():
    l.close()

def mutation(data, gradient, i, n):
    # 变异操作
    temp_grad = np.array(gradient[0][0])
    max_index = heapq.nlargest(n, range(len(temp_grad)), temp_grad.__getitem__)
    max_index = np.array(max_index)

    temp_x = data[0]
    tra_len = np.count_nonzero(temp_x)

    # 保证插入位置在原始流量长度之前
    while (True):
        if np.sum(max_index >= tra_len) == 0:
            break
        else:
            for j in range(n):
                if max_index[j] >= tra_len:
                    temp_grad[max_index[j]] = float('-inf')
                    max_index = heapq.nlargest(n, range(len(temp_grad)), temp_grad.__getitem__)
                    max_index = np.array(max_index)
    print('inserting index is :', max_index)

    for j in range(n):
        insert_num = np.sign(temp_grad[max_index[j]])
        temp_x = np.insert(temp_x, max_index[j], insert_num)[:-1]
        insert_num_list[i].extend(insert_num)
        insert_index_list[i].extend([max_index[j]])
    return temp_x

def load_model_layer_times(path,model_layer_times):
    df = pd.read_csv(path)
    colnames = df.columns
    model_layer_times_list = []
    for i in range (len(df)):
        for j,name in enumerate(colnames):
            x = colnames[j].split(",")
            layer_name = x[0]
            index = int(x[1])
            model_layer_times[(layer_name, index)] = df[name][i]
        model_layer_times_list.append(model_layer_times)
    return model_layer_times_list

def change_layer_values_to_times(path,model_layer_times,model1,threhold):
    df = pd.read_csv(path)
    colnames = df.columns
    y_test_per_class = 100
    model_layer_times_list = []
    for i in range(monitored_class_num):
        for j, name in enumerate(colnames):
              value = df[name][i * y_test_per_class : (i+1) * y_test_per_class]
              times = np.sum( value > threhold)
              x = colnames[j].split(",")
              layer_name = x[0]
              index = int(x[1])
              model_layer_times[(layer_name, index)] = times
        model_layer_times_list.append(model_layer_times)
        model_layer_times = init_coverage_times(model1,dnn)
    return model_layer_times_list

def CountTrace(data, traffic_num):
    tra_len = []
    for j in range(traffic_num):
        count = np.count_nonzero(data[j])
        tra_len.append(count)

    return tra_len

tra_len = CountTrace(X_goodSample, traffic_num)

ori_all_pred = model.predict(X_goodSample, batch_size=traffic_num)
ori_all_label = []
for i in range(traffic_num):
    ori_all_label.append(np.argmax(ori_all_pred[i]))

print("model_layer_times_list starts")
start_time = time.perf_counter()
if os.path.exists(model_layer_values_path):
    model_layer_times_list = change_layer_values_to_times(model_layer_values_path,model_layer_times1,model,threshold)
else:
    get_OpenWorld_layer_value(X_test_Mon, y_test_Mon, model_layer_value1, model,dnn,model_path)
    model_layer_times_list = change_layer_values_to_times(model_layer_values_path, model_layer_times1, model, threshold)

print("model_layer_values_list spends {} seconds".format(time.perf_counter() - start_time))
#( 1124, 60 * class_per_trace_num)
for i in range(0,1124):
    # 计算图过程中会增加图节点，导致计算时间越来越长，因此对每条流量都重置图，重新导入模型
    K.clear_session()
    model = load_model('./Model/df_ow_NoDef.h5')

    start_time = time.perf_counter()
    print("the {}th tra_len is {}".format(i, tra_len[i]))
    insert_time = 0
    ori_tra = X_goodSample[i].reshape((1, X_goodSample[i].shape[0], 1))
    gen_tra = ori_tra

    pred1 = ori_all_pred[i]
    label1 = ori_all_label[i]
    # model_layer_times1 for each traffic,not for each class
    # 如果model_layer_times1是全局变量，则统计的是所有类别中某个神经元被激活的总次数
    model_layer_times1 = init_coverage_times(model, dnn)
    model_layer_value1 = init_coverage_value(model, dnn)
    update_coverage(ori_tra, model, model_layer_times1,threshold,dnn)
    ori_covered_neurons, total_neurons, ori_coverage = neuron_covered(model_layer_times1)

    orig_label = label1
    orig_pred = pred1[label1]
    gen_pred = pred1[label1]
    gen_ori_label_pred = pred1[label1]

    # 控制带宽开销小于20%
    while (insert_time <= 0.3 * tra_len[i]):
        # 一般插入次数大于50之后，计算梯度的时间会非常长，因此reset图之后会更快一些。
        if insert_time != 0 and insert_time % 30 == 0:
            K.clear_session()
            model = load_model('./Model/df_ow_NoDef.h5')
        pred1 = model.predict(gen_tra)
        label1 = np.argmax(pred1[0])

        # update_coverage(gen_tra, model, model_layer_times1, threshold)

        if label1 == orig_label:
            # np.argsort():返回的是元素值从小到大排序后的索引值的数组
            # [-5:]顺序输出最后5个元素值
            # start_time1 = time.perf_counter()
            label_top5 = np.argsort(pred1[0])[-5:]
            print("ori_label is {},label_top5 is {}".format(label1, label_top5))

            # ’fc3'是最后一层dense层，神经元数量为最终分类的类别数，softmax是一个单调函数
            loss_1 = K.mean(model.get_layer('fc3').output[..., orig_label])
            loss_2 = K.mean(model.get_layer('fc3').output[..., label_top5[-2]])
            # loss_3 = K.mean(model.get_layer('fc3').output[..., label_top5[-3]])
            # loss_4 = K.mean(model.get_layer('fc3').output[..., label_top5[-4]])
            # loss_5 = K.mean(model.get_layer('fc3').output[..., label_top5[-5]])

            # neuron_to_cover_weight = 0.5
            # predict_weight = 10
            # predict_weight=0.5
            layer_output = loss_2 - loss_1
            # layer_output = (predict_weight * (loss_2 + loss_3 + loss_4 + loss_5) - loss_1)

            # neuron coverage loss,此处先选择经常被覆盖的神经元去激活
            loss_neuron = neuron_selection(model, model_layer_times_list[label1], model_layer_value1, neuron_select_strategy,
                                           neuron_to_cover_num, threshold)

            # # extreme value means the activation value for a neuron can be as high as possible ...
            # EXTREME_VALUE = False
            # if EXTREME_VALUE:
            #     neuron_to_cover_weight = 2

            # layer_output += neuron_to_cover_weight * loss_neuron
            layer_output += neuron_to_cover_weight * K.sum(loss_neuron)

            final_loss = K.mean(layer_output)
            # print("Compute the loss spending {} seconds".format(time.perf_counter() - start_time1))
            # log("Compute the loss spending {} seconds".format(time.perf_counter() - start_time1))

            # start_time1 = time.perf_counter()
            # we compute the gradient of the input picture wrt this loss,grads shape is same as input_tensor
            gradient = normalize(K.gradients(final_loss, model.input)[0])
            iterate = K.function([model.input], [gradient])
            grads = iterate([gen_tra])

            # 对流量进行插包
            gen_tra = mutation(gen_tra, grads, i, n=once_insert_time)

            gen_tra = gen_tra.reshape((1, length, 1))
            pred1 = model.predict(gen_tra)
            label1 = np.argmax(pred1[0])
            gen_pred = pred1[0][label1]
            gen_ori_label_pred = pred1[0][orig_label]
            insert_time += once_insert_time

        else:
            gen_tra = gen_tra.reshape((length, 1))
            tra_evad_data.append(gen_tra)
            tra_evad_label.append(label1)
            gen_tra = gen_tra.reshape((1,length, 1))
            # 计算变异前后流量间的编辑距离，变异后与目标分类流量间的编辑距离
            # ori_gen_distance, gen_mean_distance = get_Levenshtein_distance(i, gen_tra, label1, X_goodSample,
            #                                                                perClass_trace_num=class_per_trace_num)
            update_coverage(gen_tra, model, model_layer_times1, threshold,dnn)
            gen_covered_neurons, total_neurons, gen_coverage = neuron_covered(model_layer_times1)

            # print("this trace is successful")
            print("ori_label is {}, ori_len is {},evading label is {},bandwidth overhead is {}%".format(orig_label,
                                                                                                        tra_len[i],
                                                                                                        label1,
                                                                                                        insert_time * 100 /
                                                                                                        tra_len[i]))
            log("ori_label is {}, evading label is {},insert times are {},bandwidth overhead is {}%".format(
                orig_label, label1, insert_time, insert_time * 100 / tra_len[i]))
            log("ori_pred probability is {},evading_pred probability is {}".format(orig_pred, gen_pred))
            print("ori_pred probability is {},evading_pred probability is {}".format(orig_pred, gen_pred))
            print("the {}th trace spends {} seconds to evading, these inserting indexes are{}\n "
                  "these inserting numbers are {}".format(i, time.perf_counter() - start_time, insert_index_list[i],
                                                          insert_num_list[i]))
            log("the {}th trace spends {} seconds to evading, these inserting indexes are{}\n "
                "these inserting numbers are {}".format(i, time.perf_counter() - start_time, insert_index_list[i],
                                                        insert_num_list[i]))
            # columns = ['i', 'ori_class', 'ori_probability','ori_tra_len','evading_class','evading_evadingClass_probability',
            #            'evading_oriClass_probability','insert_times','bandwidth_overhead(%)','insert_indexes','insert_numbers',
            #            'spending_time(seconds)','total_neurons','ori_covered_neurons''gen_covered_neurons','ori_coverage' ,'gen_coverage',
            #            'start_time','end_time']
            result_list = [i, orig_label, orig_pred, tra_len[i], label1, gen_pred, gen_ori_label_pred, insert_time,
                           insert_time * 100 / tra_len[i],
                           insert_index_list[i], insert_num_list[i], time.perf_counter() - start_time,
                           # ori_gen_distance, gen_mean_distance,
                           total_neurons, ori_covered_neurons, gen_covered_neurons, ori_coverage, gen_coverage]
            writer1.writerow(result_list)
            f1.flush()
            break


    if insert_time > 0.3 * tra_len[i]:
        print("the {}th trace is unsuccessful".format(i))
        result_list = [i, orig_label, orig_pred, tra_len[i], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        writer1.writerow(result_list)
        f1.flush()
        log("the {}th trace is unsuccessful, the insert times is larger than 0.3 times the length of original trace".format(i))

