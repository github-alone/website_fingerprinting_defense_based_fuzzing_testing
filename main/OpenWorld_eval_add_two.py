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
length = 5000
input_shape = (length, 1)
monitored_class_num = 95
# BO_weight 表示在选择最佳插入方式时 bandwidth overhead 所占的权重
BO_weight = 5
VERBOSE = 2  # Output display mode
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

neuron_select_strategy = sys.argv[1]
threshold = float(sys.argv[2])
# neuron_to_cover_num 指定激活的神经元个数
neuron_to_cover_num = int(sys.argv[3])
# subdir 是指定今天的日期，今天跑的实验生成的流量放在此文件夹中,比如输入0113
subdir = sys.argv[4]
# csv_path = './result/'+subdir+'{}_{}_{}_half3'.format(neuron_select_strategy,str(threshold),str(neuron_to_cover_num))+'.csv'
csv_path = './result/OpenWorld/' + subdir + '{}_{}_{}'.format(neuron_select_strategy, str(threshold),
                                                               str(neuron_to_cover_num)) + '.csv'

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

def Prediction_monitored(trained_model = None, dataset = None):
    # X_test_Mon = dataset['X_test_Mon'].astype('float32')
    print ("Total testing data ", len(dataset) )
    dataset = dataset[:, :, np.newaxis]
    result_Mon = trained_model.predict(dataset, verbose=2)
    return result_Mon

def Prediction_all(trained_model = None, dataset = None):
    X_test_Mon = dataset['X_test_Mon'].astype('float32')
    X_test_Unmon = dataset['X_test_Unmon'].astype('float32')
    print ("Total testing data ", len(X_test_Mon) + len(X_test_Unmon))
    # X_test_Mon = X_test_Mon[:, :, np.newaxis]
    X_test_Unmon = X_test_Unmon[:, :, np.newaxis]
    result_Mon = trained_model.predict(X_test_Mon, verbose=2)
    result_Unmon = trained_model.predict(X_test_Unmon, verbose=2)
    return result_Mon, result_Unmon

def Evaluation_monitored( monitored_label = None,
                   unmonitored_label = 95, result_Mon = None):
    # print ("Testing with threshold = ", threshold_val)
    TP = 0
    WP = 0
    FN = 0

    # ==============================================================
    # Test with Monitored testing instances
    # evaluation
    for i in range(len(result_Mon)):
        sm_vector = result_Mon[i]
        predicted_class = np.argmax(sm_vector)

        if predicted_class == monitored_label[i]: # predicted as Monitored
            TP = TP + 1 # predicted as correct Monitored_label and actual site is Monitored
        elif (predicted_class in monitored_label and predicted_class != monitored_label[i]):
            WP = WP + 1
        elif predicted_class == unmonitored_label: # predicted as Unmonitored and actual site is Monitored
            FN = FN + 1
    TPR = float(TP) / (TP + FN + WP)

    print ("TPR : ", TPR)

    return TPR

def Evaluation_all(monitored_label = None, unmonitored_label = None,
                   result_Mon = None, result_Unmon = None):
    r = len(result_Unmon) // len(result_Mon)  # a client that visits one monitored webpage for every ten non-monitored webpages

    TP = 0  # the tested packet sequence came from the same page the classifier identified
    FP = 0  # the tested packet sequence came from a non-sensitive page
    WP = 0  # a sensitive page mistaken as another sensitive page

    TN = 0
    FN = 0

    # ==============================================================
    # Test with Monitored testing instances
    # evaluation
    for i in range(len(result_Mon)):
        sm_vector = result_Mon[i]
        predicted_class = np.argmax(sm_vector)

        if predicted_class == monitored_label[i]:  # predicted as Monitored
            TP = TP + 1  # predicted as correct Monitored_label and actual site is Monitored
        elif (predicted_class in monitored_label and predicted_class != monitored_label[i]):
            WP = WP + 1
        elif predicted_class in unmonitored_label:  # predicted as Unmonitored and actual site is Monitored
            FN = FN + 1

    # ==============================================================
    # Test with Unmonitored testing instances
    # evaluation

    for i in range(len(result_Unmon)):
        sm_vector = result_Unmon[i]
        predicted_class = np.argmax(sm_vector)

        if predicted_class in monitored_label: # predicted as Monitored
            FP = FP + 1 # predicted as Monitored and actual site is Unmonitored
        else:  # predicted as Unmonitored and actual site is Unmonitored
            TN = TN + 1


    N_P = TP + FN + WP
    N_N = FP + TN

    print ("TP : ", TP)
    print ("WP : ", WP)
    print ("FN : ", FN)
    print ("TN : ", TN)
    print ("FP : ", FP)
    print("Total  : ", TP + FP + WP + TN + FN)
    TPR = float(TP) / N_P
    print("TPR : ", TPR)
    WPR = float(WP) / N_P
    print("TPR : ", WPR)
    FPR = float(FP) / N_N
    print("TPR : ", FPR)
    Precision = TPR / (TPR + WPR + r * FPR)
    print("Precision : ", Precision)
    Recall = TPR
    print ("Recall : ", Recall)
    F1_score = (2 * Precision * Recall) / (Precision + Recall)
    print("F1_score : ", F1_score)
    print ("\n")

    # log_file.writelines("%.6f,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n"%(threshold_val, TP, FP, TN, FN, TPR, FPR, Precision, Recall))
    return TP, WP, FP, TN, FN, TPR, FPR, Precision, Recall,F1_score

def insert_each_pattern_v2(result_list,data,label,model,tra_len):

    # temp_list = result_list
    tpr_list = [[] for _ in range(len(result_list))]
    min_acc_flage = [[] for _ in range(len(result_list))]

    min_tpr_insert_trace = []
    min_tpr_insert_label = []
    min_tpr_bandwidth_overhead = []

    # 记录第一次实际插入包的数量。
    count0 = 0
    count1 = 0

    for i in range (monitored_class_num):
        min_tpr = 1
        min_tpr_row = 0
        min_tpr_pattern_index = 0

        insert_trace = [[] for _ in range(class_per_trace_num)]
        insert_label = [[] for _ in range(class_per_trace_num)]
        bandwidth_overhead = [[] for _ in range(class_per_trace_num)]
        # 记录每个类别的每个插入pattern效果，便于后续的叠加
        tpr_pattern_list = [[] for _ in range(class_per_trace_num)]

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
                print('insert_label shape is ', insert_label[j].shape)
                # insert_label[j] = np_utils.to_categorical(insert_label[j], class_num)
                result_Mon = Prediction_monitored(trained_model=model,dataset=insert_trace[j])
                TPR = Evaluation_monitored(monitored_label=insert_label[j],unmonitored_label=95,result_Mon=result_Mon)
                tpr_list[row].append(TPR)
                tpr_pattern_list[j].append(TPR)
                print("Testing accuracy:", TPR)
                if TPR < min_tpr:
                    min_tpr = TPR
                    min_tpr_pattern_index = j
                    min_tpr_row = row
            else:   # 将不能逃逸的此条流量的acc置为1，使得叠加pattern时不会选择此位置
                tpr_pattern_list[j].append(1)

        min_acc_flage[min_tpr_row].append(1)

        # 如果最小acc大于0.1，说明插入效果不好，则叠加次好的插入pattern插入到原始流量中，
        if min_tpr >= 0.1 :
            insert_trace_add = []
            bandwidth_overhead_add = []
            insert_label_add = insert_label[min_tpr_pattern_index]
            # temp_data是上次插入后的流量
            temp_data = insert_trace[min_tpr_pattern_index]
            print(temp_data.shape)
            temp_data = temp_data.reshape((temp_data.shape[0], temp_data.shape[1]))

            index = heapq.nsmallest(2, range(len(tpr_pattern_list)), tpr_pattern_list.__getitem__)
            # 取第二小acc的pattern的下标index[1]，叠加插入到已插入最小acc的pattern的流量中
            new_index = index[1]
            row = i * class_per_trace_num + new_index
            insert_index = eval(result_list['insert_indexes'][row])
            insert_num = eval(result_list['insert_numbers'][row])

            for k in range(temp_data.shape[0]):
                # 获取上一次插入的index
                last_insert_index = eval(result_list['insert_indexes'][min_tpr_row])
                last_insert_num = eval(result_list['insert_numbers'][min_tpr_row])
                if insert_index != 0:
                    # temp_x = add_mutation(temp_data[k], last_insert_index,insert_index, insert_num)
                    temp_x,count1 = add_mutation(temp_data[k], last_insert_index, insert_index, insert_num)
                else:
                    temp_x, count1 = add_mutation(temp_data[k], last_insert_index, last_insert_index, last_insert_num)
                BO = (count0 + count1) / tra_len[k]
                insert_trace_add.append(temp_x)
                bandwidth_overhead_add.append(BO)
            # insert_trace_add = np.array(insert_trace_add)
            # insert_trace_add = insert_trace_add[:, :, np.newaxis]

            min_tpr_insert_trace.append(insert_trace_add)
            min_tpr_insert_label.append(insert_label_add)
            min_tpr_bandwidth_overhead.append(bandwidth_overhead_add)

        else:
            min_tpr_insert_trace.append(insert_trace[min_tpr_pattern_index])
            min_tpr_insert_label.append(insert_label[min_tpr_pattern_index])
            min_tpr_bandwidth_overhead.append(bandwidth_overhead[min_tpr_pattern_index])

    return tpr_list,min_acc_flage,min_tpr_insert_trace,min_tpr_insert_label,min_tpr_bandwidth_overhead

def run():

    monitored_class = 95
    dataset = {}
    model = load_model('./Model/OpenWorld_NoDef.h5')
    # X: Training data's shape :  (96000, 5000), Monitored:95*800, Unmonitored:20000
    # y: Training data's shape :  (96000,)
    # X: Validation data's shape :  (9600, 5000)
    # y: Validation data's shape :  (9600,)
    # X: Testing data_Mon's shape :  (9500, 5000)
    # y: Testing data_Mon's shape :  (9500,)
    # X: Testing data_Unmon's shape :  (20000, 5000), it is different from monitored training data.
    # y: Testing data_Unmon's shape :  (20000,)
    X_train, y_train, X_valid, y_valid = LoadDataNoDefOW_Training()
    X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon = LoadDataNoDefOW_Evaluation()



    X_train_monitored = []
    y_train_monitored = []
    for i in range(y_train.shape[0]):
        for j in range(monitored_class):
            if y_train[i] == j:
                X_train_monitored.append(X_train[i])
                y_train_monitored.append(y_train[i])
    X_train_monitored = np.array(X_train_monitored)
    y_train_monitored = np.array(y_train_monitored)

    # Convert data as float32 type
    X_train_monitored = X_train_monitored.astype('float32')
    y_train_monitored = y_train_monitored.astype('float32')

    tra_len_train = CountTrace_all(X_train_monitored, X_train_monitored.shape[0])
    # tra_len_test = CountTrace_all(X_test_Mon, X_test_Mon.shape[0])

    result_list = pd.read_csv(csv_path)
    print("len(result_list) is :", len(result_list))
    tpr_list, min_tpr_flag, min_tpr_insert_trace, \
        min_tpr_insert_label, min_tpr_bandwidth_overhead = insert_each_pattern_v2(result_list, X_train_monitored,
                                                                                  y_train_monitored, model,tra_len_train)
    # tpr_list, min_tpr_flag, min_tpr_insert_trace, \
    # min_tpr_insert_label, min_tpr_bandwidth_overhead = insert_each_pattern_v2(result_list, X_test_Mon,
    #                                                                           y_test_Mon, model, tra_len_test)

    min_tpr_insert_trace = np.array(min_tpr_insert_trace)
    min_tpr_insert_label = np.array(min_tpr_insert_label)
    print(' min_tpr_insert_trace shape is:', min_tpr_insert_trace.shape)
    print(' min_tpr_insert_label shape is:', min_tpr_insert_label.shape)

    min_tpr_insert_trace = min_tpr_insert_trace.reshape((-1, 5000, 1))
    print(' min_tpr_insert_trace shape is:', min_tpr_insert_trace.shape)
    min_tpr_insert_label = min_tpr_insert_label.reshape((-1, 1))
    print(' min_tpr_insert_label shape is:', min_tpr_insert_label.shape)


    dataset['X_test_Mon'] = min_tpr_insert_trace
    dataset['y_test_Mon'] = min_tpr_insert_label
    dataset['X_test_Unmon'] = X_test_Unmon
    dataset['y_test_Unmon'] = y_test_Unmon

    result_Mon, result_Unmon= Prediction_all(trained_model = model, dataset = dataset)
    monitored_label = list(min_tpr_insert_label)
    unmonitored_label = list(y_test_Unmon)

    TP, WP, FP, TN, FN, TPR, FPR, Precision, Recall,F1_score =Evaluation_all(monitored_label=monitored_label,
                                                    unmonitored_label=unmonitored_label, result_Mon=result_Mon,
                                                    result_Unmon=result_Unmon)

    min_tpr_bandwidth_overhead = np.array(min_tpr_bandwidth_overhead)
    min_tpr_bandwidth_overhead = min_tpr_bandwidth_overhead.reshape((-1, 1))
    print(' min_tpr_bandwidth_overhead shape is:', min_tpr_bandwidth_overhead.shape)

    bandwidth_overhead = sum(min_tpr_bandwidth_overhead) / len(min_tpr_bandwidth_overhead)

    # print('r-precision:', 'test_dataset(95*100)_add_two:')
    print('r-precision:','train_dataset(95*800)_add_two:')
    print('bandwidth_overhead is :',bandwidth_overhead)
    print('tpr_all is :',TPR)
    print('fpr_all is :', FPR)
    print('precision', Precision)
    print('recall', Recall)
    print('f1_Score', F1_score)
    result_list['tpr_insert_pattern_to_class_train'] = tpr_list
    result_list['min_tpr_flag_train'] = min_tpr_flag
    result_list.to_csv(csv_path, index=False, sep=',')


    train_dataset = ['r-precision:','train_dataset(95*800)_add_two']
    # test_dataset = ['r-precision:','test_dataset(95*100)_add_two']
   
    a = ['bandwidth_overhead', bandwidth_overhead]
    b = ['tpr', TPR]
    c = ['fpr', FPR]
    d = ['precision', Precision]
    e = ['recall', Recall]
    f = ['f1_Score', F1_score]

    dataset = pd.DataFrame(data= train_dataset)
    a = pd.DataFrame(data=a)
    b = pd.DataFrame(data=b)
    c = pd.DataFrame(data=c)
    d = pd.DataFrame(data=d)
    e = pd.DataFrame(data=e)
    f = pd.DataFrame(data=f)
    # mode='a'表示追加, index=True表示给每行数据加索引序号, header=False表示不加标题
    dataset.to_csv(csv_path, mode='a',index=False, sep=',')
    a.to_csv(csv_path, mode='a',index=False, sep=',')
    b.to_csv(csv_path, mode='a',index=False, sep=',')
    c.to_csv(csv_path, mode='a', index=False, sep=',')
    d.to_csv(csv_path, mode='a', index=False, sep=',')
    e.to_csv(csv_path, mode='a', index=False, sep=',')
    f.to_csv(csv_path, mode='a', index=False, sep=',')

    #
    # a_test = ['acc_test', acc_all_test[1]]
    # b_test = ['bandwidth_overhead_test', bandwidth_overhead_test]
    # test_dataset = pd.DataFrame(data= test_dataset)
    # a_test = pd.DataFrame(data=a_test)
    # b_test = pd.DataFrame(data=b_test)
    # # mode='a'表示追加, index=True表示给每行数据加索引序号, header=False表示不加标题
    # test_dataset.to_csv(csv_path, mode='a',index=False, sep=',')
    # a_test.to_csv(csv_path, mode='a',index=False, sep=',')
    # b_test.to_csv(csv_path, mode='a',index=False, sep=',')

    # 保存变异后的流量及label
    # path1 = "./Dataset/gen_tra/OpenWorld/gen_tra_{}_{}_{}_add_two.pkl".format(neuron_select_strategy, str(threshold),
    #                                                                           str(neuron_to_cover_num))  # 文件路径
    # path2 = "./Dataset/gen_tra/OpenWorld/gen_label_{}_{}_{}_add_two.pkl".format(neuron_select_strategy, str(threshold),
    #                                                                             str(neuron_to_cover_num))
    # output1 = open(path1, 'wb')
    # output2 = open(path2, 'wb')
    # pickle.dump(min_tpr_insert_trace, output1)
    # pickle.dump(min_tpr_insert_label, output2)
    # output1.close()
    # output2.close()


if __name__ == "__main__":
    run()
