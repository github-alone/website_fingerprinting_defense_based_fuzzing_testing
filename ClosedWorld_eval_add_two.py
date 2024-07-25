import csv
from keras.models import load_model
import numpy as np
import pandas as pd
import math
import sys
import os
import heapq
from utility import *
from keras.utils import np_utils

class_per_trace_num = 20
length=5000
input_shape=(length,1)
class_num = 95
# BO_weight 表示在选择最佳插入方式时 bandwidth overhead 所占的权重
BO_weight = 5
VERBOSE = 2 # Output display mode
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

neuron_select_strategy = sys.argv[1]
threshold = float(sys.argv[2])
# neuron_to_cover_num 指定激活的神经元个数
neuron_to_cover_num = int(sys.argv[3])
# subdir 是指定今天的日期，今天跑的实验生成的流量放在此文件夹中,比如输入0113
subdir = sys.argv[4]
dnn = sys.argv[5]
# csv_path = './result/'+subdir+'{}_{}_{}_half3'.format(neuron_select_strategy,str(threshold),str(neuron_to_cover_num))+'.csv'
csv_path = './result/{}/ClosedWorld/{}_{}_{}_{}'.format(dnn,subdir,neuron_select_strategy, str(threshold),
                                                               str(neuron_to_cover_num)) + '.csv'


# csv_head = ['i', 'ori_class', 'ori_probability', 'ori_tra_len', 'evading_class', 'evading_evadingClass_probability',
#            'evading_oriClass_probability', 'insert_times', 'bandwidth_overhead(%)', 'insert_indexes', 'insert_numbers',
#            'spending_time(seconds)','ori_gen_distance', 'gen_mean_distance',
#             'total_neurons','ori_covered_neurons','gen_covered_neurons','ori_coverage' ,'gen_coverage']
def load_csv(csv_path):
    result_list = []
    with open(csv_path) as f:
        f_csv = csv.reader(f)
        # 获取表头
        headers = next(f_csv)
        for row in f_csv:
            result_list.append(row)
    f.close()
    return result_list
    # result_list = pd.read_csv(csv_path)
    # return result_list

def mutation(data, insert_index, insert_num):
    # 变异操作
    temp_x = data
    count = 0
    for i in range (len(insert_index)):
        if temp_x[insert_index[i]] != 0:
            count +=1
            temp_x = np.insert(temp_x, insert_index[i], insert_num[i])[:-1]
    return temp_x,count

def add_mutation(data,last_insert_index,insert_index,insert_num):
    temp_x = data
    last_insert_index = np.array(last_insert_index)
    count = 0
    for i in range(len(insert_index)):
        sum = np.sum(last_insert_index > insert_index[i])
        new_index = insert_index[i] + sum
        if new_index < length:
            new_index = new_index
        else:
            new_index = length-1
        if temp_x[new_index] != 0:
            count += 1
            temp_x = np.insert(temp_x, new_index, insert_num[i])[:-1]
    return temp_x,count

def CountTrace_all(data, traffic_num):
    tra_len = []
    for j in range(traffic_num):
        count = np.count_nonzero(data[j])
        tra_len.append(count)

    return tra_len

def CountTrace(data):
    tra_len = 0
    for i in range(data.shape[0]):
        if data[i] != 0:
            tra_len += 1

    return tra_len

def acc_best_insert(result_list,data,label,model):
    # best_insert 记录每个类别的最佳插入方式
    best_insert_index_list = []
    best_insert_num_list = []
    best_insert_for_class_list = []

    insert_trace = []
    insert_label = []
    bandwidth_overhead_list = []

    for i in range(len(result_list)):
        # print('result_list[min_acc_flage][i] is :',result_list['min_acc_flage'][i])
        if result_list['min_acc_flage'][i] == '[1]':
            best_insert_index_list.append(eval(result_list['insert_indexes'][i]))
            # print('result_list[insert_indexes][i] is', result_list['insert_indexes'][i])
            best_insert_num_list.append(eval(result_list['insert_numbers'][i]))
            best_insert_for_class_list.append(eval(result_list['ori_class'][i]))
    for i in range(len(best_insert_for_class_list)):
        for j in range (label.shape[0]):
            if label[j] == best_insert_for_class_list[i]:
                temp_x = mutation(data[j],best_insert_index_list[i],best_insert_num_list[i])
                insert_label.append(label[j])
                insert_trace.append(temp_x)
                tra_len = CountTrace(data[j])
                bandwidth_overhead_list.append(len(best_insert_index_list)/tra_len)
    print("len(insert_trace) is ",len(insert_trace))
    insert_trace = np.array(insert_trace)
    insert_label = np.array(insert_label)
    insert_trace = insert_trace[:, :, np.newaxis]
    print('insert_trace shape is ', insert_trace.shape)
    print('insert_label shape is ',insert_label.shape)
    # 必须是onehot编码才能输入到模型中进行评估，model.evaluation
    insert_label = np_utils.to_categorical(insert_label, class_num)
    score_test = model.evaluate(insert_trace, insert_label, verbose=VERBOSE)
    bandwidth_overhead = sum(bandwidth_overhead_list)/len(bandwidth_overhead_list)
    print('bandwidth_overhead:',bandwidth_overhead)
    print("Testing accuracy:", score_test[1])
    # math.ceil 是
    # for i in range(len(result_list)):
    #     temp_list = result_list[ i*class_per_trace_num : (i+1)*class_per_trace_num]
    #     max_score = 0
    #     index = 0
    #     for j in range (class_per_trace_num):
    #        # temp_list[j][5]表示第i类第j条流量的逃逸后分类的概率值，表格中的第5列，'evading_evadingClass_probability'
    #        # temp_list[j][6]表示第i类第j条流量的逃逸后分类为原始类别的概率值，表格中的第6列，'evading_oriClass_probability'
    #        # score = evading_evadingClass_probability - evading_oriClass_probability - BO_weight * bandwidth_overhead
    #        # 根据score来确定当前最好的插入方式，
    #        difference =100 * (float(temp_list[j][5]) - float(temp_list[j][6]))
    #        score = difference - BO_weight * float(temp_list[j][8])
    #        if score > max_score:
    #            max_score = score
    #            index = i * class_per_trace_num + j
    #     best_insert_index_list.append(eval(result_list[index][9]))
    #     best_insert_num_list.append(eval(result_list[index][10]))
    return score_test[1],bandwidth_overhead

def insert_each_pattern(result_list,data,label,model):

    temp_list = result_list
    acc_list = [[] for _ in range(len(result_list))]
    min_acc_flage = [[] for _ in range(len(result_list))]

    for i in range (class_num):
        min_acc = 1
        min_acc_index = 0
        for j in range (class_per_trace_num):
            insert_trace = []
            insert_label = []
            row = i * class_per_trace_num + j
            # row = i * class_per_trace_num + j
            print("row is ",row)
            if eval(temp_list[row][9]) != 0:
                for k in range (label.shape[0]):
                    if label[k] == i and eval(result_list[row][1]) == i:
                        insert_index = eval(temp_list[row][9])
                        insert_num = eval(temp_list[row][10])
                        temp_x = mutation(data[k],insert_index,insert_num)
                        insert_trace.append(temp_x)
                        insert_label.append(label[k])
                    else:
                        continue
                print("len(insert_trace) is ", len(insert_trace))
                insert_trace = np.array(insert_trace)
                insert_label = np.array(insert_label)
                print('insert_trace shape is ', insert_trace.shape)
                insert_trace = insert_trace[:, :, np.newaxis]
                print('insert_trace shape is ', insert_trace.shape)
                print('insert_label shape is ', insert_label.shape)
                insert_label = np_utils.to_categorical(insert_label, class_num)
                score_test = model.evaluate(insert_trace, insert_label, verbose=VERBOSE)
                acc_list[row].append(score_test[1])
                if score_test[1] < min_acc:
                    min_acc = score_test[1]
                    min_acc_index = row
                #     min_acc_flage[min_acc_index].append('1')
                # min_acc = score_test[1] if score_test[1] < min_acc else min_acc
                # min_acc_index = row if score_test[1] < min_acc else min_acc_index
                print("Testing accuracy:", score_test[1])
            else:
                continue

        min_acc_flage[min_acc_index].append(1)
        # min_acc_pattern_index_list.append(min_acc_pattern_index)
    return acc_list,min_acc_flage

def insert_each_pattern_v2(result_list,data,label,model,tra_len):

    # temp_list = result_list
    acc_list = [[] for _ in range(len(result_list))]
    min_acc_flage = [[] for _ in range(len(result_list))]

    min_acc_insert_trace = []
    min_acc_insert_label = []
    min_acc_bandwidth_overhead = []

    # 记录第一次实际插入包的数量。
    count0 = 0

    for i in range (class_num):
        min_acc = 1
        min_acc_row = 0
        min_acc_pattern_index = 0

        insert_trace = [[] for _ in range(class_per_trace_num)]
        insert_label = [[] for _ in range(class_per_trace_num)]
        bandwidth_overhead = [[] for _ in range(class_per_trace_num)]
        # 记录每个类别的每个插入pattern效果，便于后续的叠加
        acc_pattern_list = [[] for _ in range(class_per_trace_num)]

        for j in range (class_per_trace_num):
            row = i * class_per_trace_num + j
            print("row is ",row)
            if eval(result_list['insert_indexes'][row]) != 0:
                for k in range (label.shape[0]):
                    if label[k] == i and result_list['ori_class'][row] == i:
                        # 读取出来的result_list['insert_indexes'][row]是str类型，需转换为int类型
                        insert_index = eval(result_list['insert_indexes'][row])
                        # print(type(insert_index))
                        insert_num = eval(result_list['insert_numbers'][row])
                        temp_x,count0 = mutation(data[k],insert_index,insert_num)
                        BO = count0 / tra_len[k]
                        bandwidth_overhead[j].append(BO)
                        insert_trace[j].append(temp_x)
                        insert_label[j].append(label[k])
                    else: continue
                insert_trace[j] = np.array(insert_trace[j])
                insert_label[j] = np.array(insert_label[j])
                print('insert_trace shape is ', insert_trace[j].shape)
                insert_trace[j] = insert_trace[j][:, :, np.newaxis]
                print('insert_trace shape is ', insert_trace[j].shape)
                print('insert_label shape is ', insert_label[j].shape)
                insert_label[j] = np_utils.to_categorical(insert_label[j], class_num)
                score_test = model.evaluate(insert_trace[j], insert_label[j], verbose=VERBOSE)
                acc_list[row].append(score_test[1])
                acc_pattern_list[j].append(score_test[1])
                print("Testing accuracy:", score_test[1])
                if score_test[1] < min_acc:
                    min_acc = score_test[1]
                    min_acc_pattern_index = j
                    min_acc_row = row
            else:   # 将不能逃逸的此条流量的acc置为1，使得叠加pattern时不会选择此位置
                acc_pattern_list[j].append(1)

        min_acc_flage[min_acc_row].append(1)

        # 如果最小acc大于0.1，说明插入效果不好，则叠加次好的插入pattern插入到原始流量中，
        if min_acc >= 0.1 :
            insert_trace_add = []
            bandwidth_overhead_add = []
            insert_label_add = insert_label[min_acc_pattern_index]
            # temp_data是上次插入后的流量
            temp_data = insert_trace[min_acc_pattern_index]
            print(temp_data.shape)
            temp_data = temp_data.reshape((temp_data.shape[0], temp_data.shape[1]))

            index = heapq.nsmallest(2, range(len(acc_pattern_list)), acc_pattern_list.__getitem__)
            # 取第二小acc的pattern的下标index[1]，叠加插入到已插入最小acc的pattern的流量中
            new_index = index[1]
            row = i * class_per_trace_num + new_index
            insert_index = eval(result_list['insert_indexes'][row])
            insert_num = eval(result_list['insert_numbers'][row])

            for k in range(temp_data.shape[0]):
                # 获取上一次插入的index
                last_insert_index = eval(result_list['insert_indexes'][min_acc_row])
                # temp_x = add_mutation(temp_data[k], last_insert_index,insert_index, insert_num)
                temp_x,count1 = add_mutation(temp_data[k], last_insert_index, insert_index, insert_num)
                BO = (count0 + count1) / tra_len[k]
                insert_trace_add.append(temp_x)
                bandwidth_overhead_add.append(BO)
            insert_trace_add = np.array(insert_trace_add)
            insert_trace_add = insert_trace_add[:, :, np.newaxis]
            # score_test = model.evaluate(insert_trace_add, insert_label_add, verbose=VERBOSE)
            # if score_test[1]<min_acc:
            #     min_acc = score_test[1]

            min_acc_insert_trace.append(insert_trace_add)
            min_acc_insert_label.append(insert_label_add)
            min_acc_bandwidth_overhead.append(bandwidth_overhead_add)

        else:
            min_acc_insert_trace.append(insert_trace[min_acc_pattern_index])
            min_acc_insert_label.append(insert_label[min_acc_pattern_index])
            min_acc_bandwidth_overhead.append(bandwidth_overhead[min_acc_pattern_index])

    min_acc_insert_trace = np.array(min_acc_insert_trace)
    min_acc_insert_label = np.array(min_acc_insert_label)
    min_acc_bandwidth_overhead = np.array(min_acc_bandwidth_overhead)

    min_acc_insert_trace = min_acc_insert_trace.reshape((-1,5000,1))
    print(' min_acc_insert_trace shape is:', min_acc_insert_trace.shape)
    min_acc_insert_label = min_acc_insert_label.reshape((-1,95))
    print(' min_acc_insert_label shape is:', min_acc_insert_label.shape)
    min_acc_bandwidth_overhead = min_acc_bandwidth_overhead.reshape((-1,1))
    print(' min_acc_bandwidth_overhead shape is:', min_acc_bandwidth_overhead.shape)

    acc_all = model.evaluate(min_acc_insert_trace, min_acc_insert_label, verbose=VERBOSE)
    bandwidth_overhead = sum(min_acc_bandwidth_overhead) / len(min_acc_bandwidth_overhead)
    print('bandwidth_overhead:', bandwidth_overhead)
    print("Testing accuracy:", acc_all)

    # # 保存变异后的流量及label
    # path1 = "./Dataset/gen_tra/gen_tra_{}_{}_{}_add_two.pkl".format(neuron_select_strategy, str(threshold),
    #                                                            str(neuron_to_cover_num))  # 文件路径
    # path2 = "./Dataset/gen_tra/gen_label_{}_{}_{}_add_two.pkl".format(neuron_select_strategy, str(threshold),
    #                                                              str(neuron_to_cover_num))
    # output1 = open(path1, 'wb')
    # output2 = open(path2, 'wb')
    # pickle.dump(min_acc_insert_trace, output1)
    # pickle.dump(min_acc_insert_label, output2)
    # output1.close()
    # output2.close()


    return acc_all,bandwidth_overhead,acc_list,min_acc_flage

def run():
    if dnn == 'df':
        model_path = './Model/df_cw_NoDef.h5'
    else:
        model_path = './Model/var-cnn_cw_NoDef.h5'
    model = load_model(model_path)
    X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataNoDefCW()
    # Convert data as float32 type
    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_valid = y_valid.astype('float32')
    y_test = y_test.astype('float32')

    tra_len_train = CountTrace_all(X_train, X_train.shape[0])
    tra_len_test = CountTrace_all(X_test, X_test.shape[0])

    result_list = pd.read_csv(csv_path)
    print("len(result_list) is :", len(result_list))
    acc_all_train,bandwidth_overhead_train,acc_list_train,min_acc_flag_train = insert_each_pattern_v2(result_list, X_train, y_train, model,tra_len_train)
    # acc_all_test, bandwidth_overhead_test, acc_list_test, min_acc_flag_test = insert_each_pattern_v2(result_list, X_test, y_test, model,tra_len_test)
    print('acc_all_train_add_two is :',acc_all_train[1])
    print('bandwidth_overhead_train_add_two is :',bandwidth_overhead_train)
    result_list['acc_insert_pattern_to_class_train'] = acc_list_train
    result_list['min_acc_flag_train'] = min_acc_flag_train

    # print('acc_all_test is :',acc_all_test[1])
    # print('bandwidth_overhead_test is :',bandwidth_overhead_test)
    # result_list['acc_insert_pattern_to_class_test'] = acc_list_train
    # result_list['min_acc_flag_test'] = min_acc_flag_train
    # result_list.to_csv(csv_path, index=False, sep=',')

    train_dataset = ['train_dataset(95*800)']
    # test_dataset = ['test_dataset(95*100)']

    a_train = ['acc_train_add_two', acc_all_train[1]]
    b_train = ['bandwidth_overhead_train_add_two', bandwidth_overhead_train]
    train_dataset = pd.DataFrame(data= train_dataset)
    a_train = pd.DataFrame(data=a_train)
    b_train = pd.DataFrame(data=b_train)
    # mode='a'表示追加, index=True表示给每行数据加索引序号, header=False表示不加标题
    train_dataset.to_csv(csv_path, mode='a',index=False, sep=',')
    a_train.to_csv(csv_path, mode='a',index=False, sep=',')
    b_train.to_csv(csv_path, mode='a',index=False, sep=',')



if __name__ == "__main__":
    run()
