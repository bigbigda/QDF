import numpy as np
import math
import ldf_qdf
import basic_common_operation

def gen_eigs(data_list, category_prob):
    category_cov = []
    for data in data_list:
        tmp_cov_matrix = np.cov(data, rowvar=False, bias=True)
        category_cov.append(tmp_cov_matrix)

    # 以下为计算出协方差矩阵的特征值和特征向量
    # 计算sigma0
    eig_value = []
    eig_vector = []
    for i in range(len(data_list)):
        (value, matrix) = np.linalg.eig(category_cov[i])
        sort_indices = np.argsort(value)[::-1]
        sorted_value = value[sort_indices]
        sorted_matrix = matrix[:, sort_indices]
        eig_value.append(sorted_value)
        eig_vector.append(sorted_matrix)
    # return category_cov
    return eig_value, eig_vector


# 由训练集得出所有参数
def gen_mqdf_para_list(train_feature, train_label, category_num):
    data_list = ldf_qdf.do_data_split(train_feature, train_label, category_num)

    category_prob = ldf_qdf.gen_category_prob(data_list, train_label)
    category_mean = ldf_qdf.gen_mean_vector(data_list)
    (eig_value, eig_vector) = gen_eigs(data_list, category_prob)
    print(eig_vector[1].shape)
    print(eig_value[1].shape)
    # category_cov = gen_sigma_vector_ldf(train_feature)

    return category_prob, category_mean, eig_value, eig_vector


def gen_mqdf_top5_predictor(test_vector, train_result_para):
    category_prob = train_result_para[0]
    category_mean = train_result_para[1]
    category_eig_value = train_result_para[2]
    category_eig_vector = train_result_para[3]
    pro_list = []
    for cate_num in range(len(category_prob)):
        tmp_pro = 0
        distinc_num = 14
        for i in range(distinc_num):
            # print(i)
            # if i == 1:
            #     print(np.dot(test_vector - category_mean[cate_num][i], category_eig_vector[cate_num][:, i]))
            #     print(np.dot(test_vector - category_mean[cate_num][i], category_eig_vector[cate_num][:, i]) ** 2)
            # tmp_pro = tmp_pro - math.log(category_eig_value[cate_num][i])
            tmp_pro = tmp_pro - ((np.dot(test_vector - category_mean[cate_num], category_eig_vector[cate_num][:, i])
                                  ** 2) / category_eig_value[cate_num][i]) - math.log(category_eig_value[cate_num][i])
        for j in range (distinc_num, 16):
            tmp_pro = tmp_pro - ((np.dot(test_vector - category_mean[cate_num], category_eig_vector[cate_num][:, j])
                                  ** 2) / category_eig_value[cate_num][distinc_num]) -\
                                    math.log(category_eig_value[cate_num][distinc_num])
        pro_list.append(tmp_pro)
        # QDF方法
        #
        # pro_list.append(-(test_vector - category_mean[cate_num]) @ (
        #     np.linalg.inv(category_cov[cate_num])) @ (test_vector - category_mean[cate_num])
        #                 - math.log(np.linalg.det(category_cov[cate_num])))

    # 排序
    # print("begin:")
    # print(pro_list)
    pro_top5_indices = np.argsort(pro_list)[::-1][:5]
    return pro_top5_indices


def gen_mqdf_predict_pro(to_be_test_array, train_result_para):
    predict_top5_indices_list = []
    # 由训练后的网络的出预测结果
    for test_sample_num in range(to_be_test_array.shape[0]):
        predict_top5_indices_list.append(gen_mqdf_top5_predictor(to_be_test_array[test_sample_num], train_result_para))

    return predict_top5_indices_list


def mqdf_main():
    print("#################################")
    print('MQDF :')

    train_label, train_feature, test_label, test_feature, category_num = basic_common_operation.read_data()

    train_result_para = gen_mqdf_para_list(train_feature, train_label, category_num)
    predict_top5_pro = gen_mqdf_predict_pro(test_feature, train_result_para)
    (top1, top3, top5) = ldf_qdf.gen_final_sum(predict_top5_pro, test_label)

    basic_common_operation.show_top_pro_result(top1, top3, top5, len(test_label))



