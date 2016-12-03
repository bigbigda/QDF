import numpy as np
import QDF from .

def gen_category_prob(data_list, train_label):
    category_prob = []
    for i in range(len(data_list)):
        category_prob.append(data_list[i].shape[0] / train_label.shape[0])
    return category_prob


def gen_mean_vector(data_list):
    category_mean = []
    for i in range(len(data_list)):
        category_mean.append(data_list[i].mean(axis=0))
    return category_mean


def gen_sigma_vector_qdf(data_list, category_prob):
    category_cov = []
    for data in data_list:
        tmp_cov_matrix = np.cov(data, rowvar=False, bias=True)
        # print (type(tmp_cov_matrix))
        category_cov.append(tmp_cov_matrix)

    # 以下为RDA分类器的协方差矩阵平滑部分
    # 计算sigma0
    sigma0 = np.zeros(shape=(16, 16))
    tmp = 0
    for i in range(len(data_list)):
        sigma0 = category_cov[i] * category_prob[i] + sigma0
        tmp = category_prob[i] + tmp

    # return category_cov
    return category_cov


def gen_sigma_vector_ldf(data):
    category_cov = []
    tmp_cov_matrix = np.cov(data, rowvar=False, bias=True)
    # print (type(tmp_cov_matrix))
    for i in range(26):
        category_cov.append(tmp_cov_matrix)

    return category_cov


# 由训练集得出所有参数
def gen_para_list(train_feature, train_label):
    data_list = QDF.do_data_split(train_feature, train_label, 26)

    category_prob = gen_category_prob(data_list, train_label)
    category_mean = gen_mean_vector(data_list)
    category_cov = gen_sigma_vector_qdf(data_list, category_prob)
    # category_cov = gen_sigma_vector_ldf(train_feature)

    return category_prob, category_mean, category_cov

def main():
    print("#################################")
    print('hello python!')

    train_data = np.genfromtxt('D:\\work\\JavaWorkspace\\eclipsePython\\src\\Letter\\train.txt', delimiter=',')
    test_data = np.genfromtxt('D:\\work\\JavaWorkspace\\eclipsePython\\src\\Letter\\test.txt', delimiter=',')

    # print(type(train_data))
    train_label = train_data[:, 0].copy()
    train_feature = train_data[:, 1:17]
    # print(train_label.shape)
    # print(train_feature.shape)

    test_label = test_data[:, 0].copy()
    test_feature = test_data[:, 1:17]
    # print(test_label.shape)
    # print(test_feature.shape)
    # beta = np.linspace (0, 0.05, 5)
    gamma = np.linspace (0, 0.03, 5)
    beta = [0]
    # gamma = [0.001]

    t1 = 0
    b_v = 0
    g_v = 0
    count = 0

    train_result_para = gen_mqdf_para_list(train_feature, train_label)
    predict_top5_pro = QDF.gen_predict_pro(test_feature, train_result_para)
    (top1, top3, top5) = QDF.gen_final_sum(predict_top5_pro, test_label)
    if top1 > t1:
        t1 = top1
        b_v = beta[i]
        g_v = gamma[j]
    print(top1/4000)


    print(t1/ 4000)
    print(b_v)
    print(g_v)
    # print(top1)
    # print(top3)
    # print(top5)
    #
    # print(top1 / 4000)
    # print(top3 / 4000)
    # print(top5 / 4000)


main()