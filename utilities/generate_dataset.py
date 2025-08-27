import pandas as pd
import numpy as np
import os
from .parse_config import parameters

data_dir = parameters['data_dir']
window_size = parameters['win_len']
stride = parameters['win_stride']
def generate_dataset2(rootpath, subjectNames, indices,feature_indices,filename='features_100_50.csv'):
    """
    Read the features.csv files from each subject and generate the dataset
    :param rootpath: the root path of the dataset
    :param subjectNames: the list of subject names
    :return: X: the feature matrix, y: the label vector
    """
    first_flag = True
    for subject in subjectNames:
        filepath = os.path.join(rootpath, subject, 'Gsenser Data', filename)
        feature = pd.read_csv(filepath, header=None)
        if first_flag:
            features = feature.to_numpy()
            first_flag = False
        else:
            features = np.vstack((features, feature.to_numpy()))
    features_0 = features[np.isin(features[:, -2], indices)]
    features_1 = features[features[:, -1] == 1]
    sampled_features = np.vstack((features_0, features_1))
    X = sampled_features[:, feature_indices]
    y = sampled_features[:, -1]

    return X, y

def generate_oneclass_dataset_addout(rootpath, subjectNames,indices,feature_indices, percent,filename='features_100_50.csv'):
    first_flag = True
    for subject in subjectNames:
        filepath = os.path.join(rootpath, subject, 'Gsenser Data', filename)
        feature = pd.read_csv(filepath, header=None)
        if first_flag:
            features = feature.to_numpy()
            first_flag = False
        else:
            features = np.vstack((features, feature.to_numpy()))
    # 过滤符合条件的行
    features_0 = features[np.isin(features[:, -2], indices)]
    features_1 = features[features[:, -1] == 1]

    # 确定feature_1的行数
    count1 = features_1.shape[0]
    #随机取出feature_0中count1*percent的行,固定随机种子
    np.random.seed(0)
    sampled_0 = features_0[np.random.choice(len(features_0), int(count1*percent), replace=False)]

    # 合并抽取的行
    sampled_features = np.vstack((sampled_0, features_1))
    # 打乱行顺序
    np.random.shuffle(sampled_features)
    # 分离特征和标签
    X_sampled = sampled_features[:, feature_indices]
    y_sampled = sampled_features[:, -1]

    return X_sampled, y_sampled
def generate_dataset(rootpath, filename):

    first_flag = True

    filepath = os.path.join(rootpath, 'temp', filename)
    feature = pd.read_csv(filepath, header=None)
    if first_flag:
        features = feature.to_numpy()
        first_flag = False
    else:
        features = np.vstack((features, feature.to_numpy()))
    X = features[0:, :-1]
    y = features[0:, -1]
    return X, y



