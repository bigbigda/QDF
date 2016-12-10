import numpy as np
import basic_common_operation

def initial_file_process():

    raw_data_lines = open('E:\\DataSet\\iris\\data.txt')
    f_outfile = open('E:\\DataSet\\iris\\data_change_tag.txt', 'w')
    for s in raw_data_lines:
        f_outfile.write(s.replace('Iris-setosa', '0').replace('Iris-versicolor', '1').replace('Iris-virginica', '2'))
    f_outfile.close()


def iris_data():
    data = np.genfromtxt('E:\\DataSet\\iris\\data_change_tag.txt', delimiter=',')

    row_num = data.shape[0]
    col_num = data.shape[1]
    line_id = np.arange(row_num)
    np.random.shuffle(line_id)
    train_num = int(row_num * 0.8)

    train_data = data[0:train_num,:]
    test_data = data[train_num:,:]

    train_label = train_data[:,col_num-1]
    train_feature = train_data[:, 0:col_num-1]

    test_label = test_data[:,col_num-1]
    test_feature = test_data[:0:col_num-1]

    category_num = np.max(test_label) + 1
    category_num_int = category_num.astype(int)
    print(category_num_int)

    return train_label, train_feature, test_label, test_feature, category_num_int


def gen_para_pca_list(train_feature, train_label, category_num):
    data_mean = train_feature.mean(axis=0)
    return


def my_pca():
    train_label, train_feature, test_label, test_feature, category_num = iris_data()
    train_result_para = gen_para_pca_list(train_feature, train_label, category_num)
