import pickle as pkl

data1 = pkl.load(open("./testdata.pkl", "rb"))

data = data1.reset_index().drop(columns=["index"]).drop(columns=["label"])


labels = data1["label"]

import pyspark
import pandas as pd
from math import log2
from copy import deepcopy
import time

from pyspark import SparkContext

sc = SparkContext("spark://spark-master:7077", "Tree")

start_tick = time.time()

dataset = data

labels = labels

features = list(data.columns)

n_parallel = 8

converge = False

cur_dataset = dataset

cur_labels = labels

### a BFS procedure for building decision tree based on ID3
# dataset, labels, splitted nodes


tasks_list = [(0, dataset, labels, 1, {})]
keyid_taskinfo_dict = {
    0: (dataset.index, {}),
}

step = 0

while (not converge):
    step += 1
    if len(tasks_list) < n_parallel:
        n_piece = int(n_parallel / len(tasks_list)) + 1
        fine_tasks_list = []
        for task in tasks_list:
            length_data = task[1].shape[0]
            piece_len = int(length_data / n_parallel) + 1
            for i in range(0, n_parallel):
                fine_tasks_list.append((task[0], task[1].iloc[i * piece_len:(i + 1) * piece_len, :],
                                        task[2].iloc[i * piece_len:(i + 1) * piece_len], task[3], task[4]))
        state = sc.parallelize(fine_tasks_list, n_parallel)
    else:
        state = sc.parallelize(tasks_list, n_parallel)


    def convert_state(state_value):
        key_id = state_value[0]
        cur_dataset = state_value[1]
        labels_value = state_value[2]
        cur_entropy = state_value[3]
        pre_features = state_value[4]
        n_data = cur_dataset.shape[0]
        n_features = cur_dataset.shape[1]
        if n_data == 0:
            return []
        if (n_features == 0) or cur_entropy < 0.000001:
            return []
        features = list(cur_dataset.columns)
        feature_values = []

        for fea in features:
            for idx in cur_dataset.index:
                feature_values.append(((key_id, fea, cur_dataset[fea][idx]), (labels_value[idx], n_data, 1)))
        return feature_values


    # output: (key_id, feature_id, feature_value) -> (label, 1)
    feature_list = state.flatMap(convert_state)
    print("done1")
    print("time1:", time.time() - start_tick)
    start_tick = time.time()
    ## if feature_list is null, break
    try:
        total_feature_list_num = feature_list.first()
    except:
        break

    print("done2")
    print("time2:", time.time() - start_tick)
    start_tick = time.time()

    # output: (feature_id, feature_value) -> (label, 1)
    # feature_list = datasetRdd.flatMap(lambda x: get_feature_list(x))

    # output: (key_id, feature_id, feature_value, label) -> 1
    feature_label_list = feature_list.map(lambda x: ((x[0][0], x[0][1], x[0][2], x[1][0]), 1))

    # output: ((key_id, feature_id, feature_value)-> (N(Ffid=FValue), n_data))
    feature_count = feature_list.reduceByKey(lambda x, y: (-1, x[1], x[2] + y[2])).mapValues(lambda x: (x[2], x[1]))

    # output: ((key_id, feature_id, feature_value) -> (label, N(Ffid=Fvalue, L=label))) for each (key_id, feature_id, feature_value, label)
    feature_label_count = feature_label_list.reduceByKey(lambda x, y: x + y).map(
        lambda x: ((x[0][0], x[0][1], x[0][2]), (x[0][3], x[1])))

    # output: ((key_id, feature_id, feature_value) -> (label, N(Ffid=Fvalue, L=label), N(Ffid=FValue), n_data)) for each (key_id, feature_id, feature_value, label)
    feature_label_info = feature_label_count.join(feature_count).mapValues(
        lambda x: (x[0][0], x[0][1], x[1][0], x[1][1]))

    # print("[DEBUG]feature_label_info:", feature_label_info.collect(), "\n")

    # output: ((key_id, feature_id) -> entropy) for each (key_id, feature_id, feature_value, label)
    log_feature_label_unit = feature_label_info.map(
        lambda x: ((x[0][0], x[0][1]), -1.0 / x[1][3] * x[1][1] * log2(x[1][1] / x[1][2])))

    # (key_id, feature_id) -> entropy for each feature_id
    entropy = log_feature_label_unit.reduceByKey(lambda x, y: x + y)


    def choose_feature_func(x, y):
        if x[1] > y[1]:
            return y
        else:
            return x


    # (key_id -> feature_id_to_split)
    entropy = entropy.map(lambda x: (x[0][0], (x[0][1], x[1]))).reduceByKey(choose_feature_func)
    print("time3:", time.time() - start_tick)
    start_tick = time.time()
    print("done3")

    ## phase 2: choose the feature with lowest entropy
    new_keyid_taskinfo_dict = {}
    new_tasks_list = []

    feature_chosen_dict = entropy.collectAsMap()
    print("time4:", time.time() - start_tick)
    start_tick = time.time()
    print("done4")

    cnt = max(feature_chosen_dict.keys())
    for key_id in feature_chosen_dict:
        feature_chosen = feature_chosen_dict[key_id][0]
        split_entropy = feature_chosen_dict[key_id][1]
        valid_index = keyid_taskinfo_dict[key_id][0]
        splited_features = keyid_taskinfo_dict[key_id][1]

        valid_dataset = dataset.loc[valid_index]
        for val in valid_dataset[feature_chosen].unique():
            # print("[Pre Split]", splited_features, "-> (", feature_chosen," = ", val, ")")
            new_splited_features = deepcopy(splited_features)
            new_splited_features[feature_chosen] = str(val)
            # print(new_splited_features, " entropy=", split_entropy)
            # print(valid_dataset)
            if split_entropy < 0.001:
                # have been completely split, output
                records = valid_dataset.mask(valid_dataset[feature_chosen] != val).dropna()
                if records.shape[0] >= 1:
                    print("[Leaf] ", new_splited_features, ": label=", labels[records.index[0]])
                else:
                    print("[End] No record under ", new_splited_features)
                continue
            new_dataset_index = list(valid_dataset.mask(valid_dataset[feature_chosen] != val).dropna().index)
            new_dataset = valid_dataset.loc[new_dataset_index].copy(deep=True)
            new_dataset.drop(columns=list(new_splited_features.keys()), inplace=True)
            if new_dataset.shape[0] == 0:
                print("[End] No record under ", new_splited_features)
                continue
            if new_dataset.shape[1] == 0:
                # justify which is most, give the result
                label_pred = labels[new_dataset_index].value_counts().index[0]
                print("[Leaf] ", new_splited_features, ": label=", label_pred)
                continue
            print("[Split]", splited_features, "-> (", feature_chosen, " = ", val, ")")
            new_labels = labels.loc[new_dataset_index]
            cnt += 1
            new_keyid_taskinfo_dict[cnt] = (new_dataset_index, new_splited_features)
            new_tasks_list.append((cnt, new_dataset, new_labels, split_entropy, new_splited_features))

    keyid_taskinfo_dict = new_keyid_taskinfo_dict
    tasks_list = new_tasks_list
    print("time5:", time.time() - start_tick)
    start_tick = time.time()
    print("done5")

    if step == 4:
        break

print("elapsed:", time.time() - start_tick)



