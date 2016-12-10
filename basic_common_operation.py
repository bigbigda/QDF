import numpy as np


def read_data():
    train_data = np.genfromtxt('E:\\DataSet\\Letter\\train.txt', delimiter=',')
    test_data = np.genfromtxt('E:\\DataSet\\Letter\\test.txt', delimiter=',')

    train_label = train_data[:, 0].copy()
    train_feature = train_data[:, 1:17]

    test_label = test_data[:, 0].copy()
    test_feature = test_data[:, 1:17]

    # train_data = np.genfromtxt('E:\\DataSet\\opt\\train.txt', delimiter=',')
    # test_data = np.genfromtxt('E:\\DataSet\\opt\\test.txt', delimiter=',')
    #
    # train_label = train_data[:, 0].copy()
    # train_feature = train_data[:, 1:65]
    #
    # test_label = test_data[:, 0].copy()
    # test_feature = test_data[:, 1:65]

    # train_data = np.genfromtxt('E:\\DataSet\\Statlog\\train.txt', delimiter=',')
    # test_data = np.genfromtxt('E:\\DataSet\\Statlog\\test.txt', delimiter=',')
    #
    # train_label = train_data[:, 0].copy()
    # train_feature = train_data[:, 1:37]
    #
    # test_label = test_data[:, 0].copy()
    # test_feature = test_data[:, 1:37]

    category_num = np.max(test_label) + 1
    category_num_int = category_num.astype(int)
    print(category_num_int)
    return train_label, train_feature, test_label, test_feature, category_num_int


def show_top_pro_result(top1, top3, top5, num_of_test_set):
    print("the top1, 3, 5 result is as follows:")
    print(top1)
    print(top3)
    print(top5)
    #
    print(top1 / num_of_test_set)
    print(top3 / num_of_test_set)
    print(top5 / num_of_test_set)


# 将训练集各个类的ndarry类型数据存入list
def generate_class_data(feature_ndarr, label_ndarr, label_to_find):
    index = (np.where(label_ndarr == label_to_find))
    return feature_ndarr[index]


def do_data_split(feature_ndarr, label_ndarr, category_num):
    category_list = []
    for i in range(category_num):
        tmp_array = generate_class_data(feature_ndarr, label_ndarr, i)
        category_list.append(tmp_array)
    return category_list


def gen_mean_vector(data_list):
    category_mean = []
    for i in range(len(data_list)):
        category_mean.append(data_list[i].mean(axis=0))
    return category_mean

