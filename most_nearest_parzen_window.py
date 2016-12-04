import numpy as np
import ldf_qdf
import basic_common_operation


def gen_pw_top5_predictor(test_vector, data_list):
    pro_list = []
    d = data_list[1].shape[1]
    for i in range(len(data_list)):
        tmp_pro = 0
        h = 0.05
        for j in range(data_list[i].shape[0]):
            tmp_pro += (1 / h ** d) * np.exp(-(np.linalg.norm(test_vector - data_list[i][j, :])) / (2 * h * h))
            # tmp_pro = tmp_pro + (1/h ** d) * np.exp(-(np.linalg.norm(test_vector - data_list[i][j,:]))/(2 * h * h))
        pro_list.append(tmp_pro)

    pro_top5_indices = np.argsort(pro_list)[::-1][:5]
    return pro_top5_indices


def gen_pw_predict_pro(to_be_test_array, train_feature, train_label):
    predict_top5_indices_list = []
    # 由训练后的网络的出预测结果
    data_list = ldf_qdf.do_data_split(train_feature, train_label, 26)

    for test_sample_num in range(to_be_test_array.shape[0]):
        tmp_top5_indices = gen_pw_top5_predictor(to_be_test_array[test_sample_num], data_list)
        predict_top5_indices_list.append(tmp_top5_indices)
        print(tmp_top5_indices)
        print(test_sample_num)

    return predict_top5_indices_list


def gen_mnd_top5_predictor(test_vector, train_feature, train_label):
    pro_list = []
    print("zzz")

    for train_num in range(len(train_label)):
        # print(test_vector)
        # print(train_feature[train_num, :])
        pro_list.append(np.linalg.norm(test_vector - train_feature[train_num, :]))

    label_indices = train_label[np.argsort(pro_list)]
    _, idx = np.unique(label_indices, return_index=True)
    label_top5 = label_indices[np.sort(idx)][:5]

    return label_top5


def gen_mnd_predict_pro(to_be_test_array, train_feature, train_label):
    predict_top5_indices_list = []
    the_tenth_near = []
    # 由训练后的网络的出预测结果
    for test_sample_num in range(to_be_test_array.shape[0]):
        tmp_top5_indices = gen_mnd_top5_predictor(to_be_test_array[test_sample_num], train_feature, train_label)
        predict_top5_indices_list.append(tmp_top5_indices)
        #print(tmp_top5_indices)
        print(test_sample_num)
    print("aaaaaaaaaaaaa")
    print(np.mean(the_tenth_near))
    print("aaaaaaaaaaaaa")

    return predict_top5_indices_list


def most_nearest_distance():
    print("#################################")
    print('Most Nearest Distance:')

    train_label, train_feature, test_label, test_feature, category_num = basic_common_operation.read_data()
    # print("bbb:%s " %(train_feature.shape))

    predict_top5_pro = gen_mnd_predict_pro(test_feature, train_feature, train_label)
    (top1, top3, top5) = ldf_qdf.gen_final_sum(predict_top5_pro, test_label)

    basic_common_operation.show_top_pro_result(top1, top3, top5, len(test_label))


def parzen_window():
    print("#################################")
    print('Parzen Window:')

    train_label, train_feature, test_label, test_feature, category_num = basic_common_operation.read_data()

    predict_top5_pro = gen_pw_predict_pro(test_feature, train_feature, train_label)
    (top1, top3, top5) = ldf_qdf.gen_final_sum(predict_top5_pro, test_label)

    basic_common_operation.show_top_pro_result(top1, top3, top5, len(test_label))


# most_nearest_distance()
