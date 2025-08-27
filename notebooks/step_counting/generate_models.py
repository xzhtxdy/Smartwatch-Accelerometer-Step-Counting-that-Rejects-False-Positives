from sklearnex import patch_sklearn
patch_sklearn()
import numpy as np
import joblib
np.random.seed(43)
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from utilities import parameters,generate_oneclass_dataset_addout

project_path=parameters["project_dir"]
data_dir = parameters["data_dir"]
window_size = parameters["win_len"]
stride = parameters["win_stride"]
# subject_names_train = parameters['subject_names_train']
subject_names_train = [f"S{i}" for i in range(1, 49)]

def initialize_weights(m):
    """初始化均匀概率权重"""
    return np.ones(m) / m

def compute_kernel_matrix(x, sigma):
    """计算核矩阵"""
    pairwise_sq_dists = np.sum(x**2, axis=1).reshape(-1, 1) + np.sum(x**2, axis=1) - 2 * np.dot(x, x.T)
    kernel_matrix = np.exp(-pairwise_sq_dists / (2 * sigma**2))
    return kernel_matrix

def compute_fk_matrix(kernel_matrix, weights):
    """基于核矩阵计算 f_k(xi)"""
    return np.dot(kernel_matrix, weights)

def compute_fk_prime_matrix(kernel_matrix, weights):
    """基于核矩阵计算 f'_k(xi)"""
    kernel_matrix_no_diag = kernel_matrix - np.diag(np.diag(kernel_matrix))  # 设置对角元素为 0
    return np.dot(kernel_matrix_no_diag, weights)

def update_weights(weights, fk_values, fk_prime_values):
    """基于分类器输出的对数比值更新权重"""
    fk_prime_values = np.clip(fk_prime_values, a_min=1e-10, a_max=None)  # 避免分母为零
    updated_weights = weights + np.log(fk_values / fk_prime_values)
    if np.isnan(updated_weights).any():
        raise ValueError("Updated weights contain NaN values.")
    return updated_weights

def invert_and_normalize_weights(weights):
    """反转并归一化最终权重"""
    inverted_weights = 1 / weights
    normalized_weights = inverted_weights / np.sum(inverted_weights)
    return normalized_weights

def weighted_bagging_svm(x, weights, n_estimators, nu, kernel, gamma):
    """基于加权的 Bagging 集成 One-Class SVM"""
    models = []

    for i in range(n_estimators):
        np.random.seed(43+i)
        indices = np.random.choice(len(x), size=len(x), replace=True, p=weights)
        x_resampled = x[indices]
        model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        model.fit(x_resampled)
        models.append(model)
    return models

def majority_voting_predict(models, x):
    """使用绝对多数投票法的 Bagging 集成 One-Class SVM 进行预测"""
    predictions = np.array([model.predict(x) for model in models])
    positive_votes = np.sum(predictions == 1, axis=0)
    negative_votes = np.sum(predictions == -1, axis=0)
    final_prediction = np.where(positive_votes > negative_votes, 1, -1)
    return final_prediction

def subject_random_split_cross_validation(subject_names, feature_indices,  n_splits, percent):
    folds = []
    universal_array = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    subject_names = np.array(subject_names)
    path=os.path.join(data_dir,'raw', 'data')
    for i in range(n_splits):
        # 原始数据划分为训练集和测试集
        train_subjects=subject_names
        train_X, train_y = generate_oneclass_dataset_addout(path, train_subjects, universal_array, feature_indices, percent)

        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_X)

        folds.append({
            "train_X_scaled": train_X_scaled,
        })

    return folds, scaler

def pca_discard_top_variance(folds, explained_variance_to_discard=0.9):

    retained_components_original = []

    for fold in folds:
        train_X_scaled = fold["train_X_scaled"]
        train_gpu = np.array(train_X_scaled)

        # 初始化权重
        m = train_X_scaled.shape[0]  # 数据点的数量
        weights = np.array(initialize_weights(m))
        sigma = 1

        # 计算核矩阵
        kernel_matrix = compute_kernel_matrix(train_gpu, sigma)

        # 执行权重更新（n 次迭代）
        for k in range(10):
            fk_values = compute_fk_matrix(kernel_matrix, weights)
            fk_prime_values = compute_fk_prime_matrix(kernel_matrix, weights)
            weights = update_weights(weights, fk_values, fk_prime_values)

        # 归一化权重
        final_weights = invert_and_normalize_weights(weights)
        # 保存最终权重
        fold["final_weights"] = final_weights  # 转换为 CPU 格式并存储

        pca = PCA(n_components=0.95)
        train_X_pca_original = pca.fit_transform(train_X_scaled)

        fold["train_X_pca_original"] = train_X_pca_original

        retained_components_original.append(train_X_pca_original.shape[1])

    # print(f"保留主成分数（选择顶部变化 95% 的主成分 -original样本）：{retained_components_original}")

    return folds,pca


def save_bagging_model(models, scaler, weights, pca, file_path):

    bagging_model = {
        "models": models,
        "scaler": scaler,
        "weights": weights,
        "pca": pca
    }
    joblib.dump(bagging_model, file_path)
    print(f"WSB-OCSVM model is saved in {file_path}")

def save_single_model(models, scaler,  pca, file_path):

    bagging_model = {
        "models": models,
        "scaler": scaler,
        "pca": pca
    }
    joblib.dump(bagging_model, file_path)


n_splits = 1
n_estimators = 5
save_path=os.path.join(project_path,'result')
feature_indices= list(range(0, 24))
folds,scaler = subject_random_split_cross_validation(subject_names_train,  feature_indices=feature_indices,n_splits=n_splits,percent=0)

# PCA
folds,pca = pca_discard_top_variance(folds, explained_variance_to_discard=0.8)


for fold_idx, fold in enumerate(folds):

    train_X_pca_original=fold["train_X_pca_original"]
    final_weights=fold["final_weights"]

    single_ocsvm = OneClassSVM(nu=0.051, kernel='rbf', gamma=0.05)
    single_ocsvm.fit(train_X_pca_original)
    save_single_model(single_ocsvm, scaler, pca, os.path.join(save_path,'ocsvm_model.pkl'))

    models = weighted_bagging_svm(train_X_pca_original, final_weights, n_estimators, 0.001, 'rbf', 0.03)
    save_bagging_model(models, scaler, final_weights, pca, os.path.join(save_path,'wsbocsvm_model.pkl'))

