import math
import numpy as np


# f = open("D:\\work\\JavaWorkspace\\eclipsePython\\src\\Letter\\train.txt", 'r')


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


def gen_sigma_vector_qdf(data_list, category_prob, beta, gamma):
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

    print("good")
    print(tmp)
    category_cov_smooth = []
    for j in range(len(data_list)):
        tmp_cov_smooth = (1 - gamma) * ((1 - beta) * category_cov[j] + beta * sigma0) + \
                         gamma * np.trace(category_cov[j]) * np.eye(16) / 16
        category_cov_smooth.append(tmp_cov_smooth)

    # return category_cov
    return category_cov_smooth


def gen_sigma_vector_ldf(data):
    category_cov = []
    tmp_cov_matrix = np.cov(data, rowvar=False, bias=True)
    # print (type(tmp_cov_matrix))
    for i in range(26):
        category_cov.append(tmp_cov_matrix)

    return category_cov


# 由训练集得出所有参数
def gen_para_list(train_feature, train_label, beta, gamma):
    data_list = do_data_split(train_feature, train_label, 26)

    category_prob = gen_category_prob(data_list, train_label)
    category_mean = gen_mean_vector(data_list)
    category_cov = gen_sigma_vector_qdf(data_list, category_prob, beta, gamma)
    # category_cov = gen_sigma_vector_ldf(train_feature)

    return category_prob, category_mean, category_cov


# 对于一个输入样本，求得其对应某一类的条件概率

# 对每一个输入样本，遍历其对应各个类的条件概率，得出一个list
# def clc_condition_pro():
#     for()


# 算出所有测试集向量的top5向量，并产生top1,top3,top5预测概率; to_be_test2d_label为2维 ndarr
def gen_final_sum(predict_top5_pro, golden_label):
    # 将预测结果和实际结果比较
    predict_result_list = []
    to_be_test_ndarr = np.array(predict_top5_pro)
    for top_i_num in range(to_be_test_ndarr.shape[1]):
        tmp_result_ndarry = np.equal(to_be_test_ndarr[:, top_i_num], golden_label).tolist()
        predict_result_list.append(tmp_result_ndarry)

    # 统计top1 top3 top5预测正确的数量
    predict_result_ndarry = np.array(predict_result_list)
    predict_result_ndarry = predict_result_ndarry.transpose()

    top1_sum = predict_result_ndarry[:, 0:1].any(axis=1).sum()
    top3_sum = predict_result_ndarry[:, 0:3].any(axis=1).sum()
    top5_sum = predict_result_ndarry[:, 0:5].any(axis=1).sum()

    return top1_sum, top3_sum, top5_sum


# do_data_split(label, feature, 26)
# create_sigma_vector(category_list)
# print(category_list[1].shape)

# 对于每一个输入样本对应各个类的概率，得出一个top5的向量
def gen_top5_predictor(test_vector, train_result_para):
    category_prob = train_result_para[0]
    category_mean = train_result_para[1]
    category_cov = train_result_para[2]

    pro_list = []
    for cate_num in range(len(category_prob)):
        pro_list.append(math.log(category_prob[cate_num]) - (test_vector - category_mean[cate_num]) @ (
            np.linalg.inv(category_cov[cate_num])) @ (test_vector - category_mean[cate_num])
                        - math.log(np.linalg.det(category_cov[cate_num])))
        # QDF方法
        #
        # pro_list.append(-(test_vector - category_mean[cate_num]) @ (
        #     np.linalg.inv(category_cov[cate_num])) @ (test_vector - category_mean[cate_num])
        #                 - math.log(np.linalg.det(category_cov[cate_num])))

    # 排序
    print(pro_list)
    pro_top5_indices = np.argsort(pro_list)[::-1][:5]
    return pro_top5_indices


def gen_predict_pro(to_be_test_array, train_result_para):
    predict_top5_indices_list = []
    # 由训练后的网络的出预测结果
    for test_sample_num in range(to_be_test_array.shape[0]):
        predict_top5_indices_list.append(gen_top5_predictor(to_be_test_array[test_sample_num], train_result_para))

    return predict_top5_indices_list


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
    # gamma = np.linspace (0, 0.03, 5)
    beta = [0]
    gamma = [0]

    t1 = 0
    b_v = 0
    g_v = 0
    count = 0
    for i in range(len(beta)):
        for j in range (len(gamma)):
            train_result_para = gen_para_list(train_feature, train_label, beta[i], gamma[j])
            predict_top5_pro = gen_predict_pro(test_feature, train_result_para)
            (top1, top3, top5) = gen_final_sum(predict_top5_pro, test_label)
            if top1 > t1:
                t1 = top1
                b_v = beta[i]
                g_v = gamma[j]
            print(top1/4000)
            count += 1
            print(count)

    print(t1/ 4000)
    print(b_v)
    print(g_v)
    print(top1)
    print(top3)
    print(top5)
    #
    print(top1 / 4000)
    print(top3 / 4000)
    print(top5 / 4000)


# main()
