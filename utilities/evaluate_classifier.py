import joblib
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



def evaluate(trainX, trainy, testX, testy, model):
    model.fit(trainX, trainy)
    pred = model.predict(testX)
    accuracy_list = []
    f1score_list = []
    precision_list = []
    recall_list = []
    for id_movement in range(10):
        testy_subset = testy[testy == id_movement]
        pred_subset = pred[testy == id_movement]
        accuracy_list.append(accuracy_score(testy_subset, pred_subset))
        f1score_list.append(f1_score(testy_subset, pred_subset, average='macro'))
        precision_list.append(precision_score(testy_subset, pred_subset, average='macro'))
        recall_list.append(recall_score(testy_subset, pred_subset, average='macro'))
    return accuracy_list, f1score_list, precision_list, recall_list, pred

def predict_with_bagging_model(bagging_model, data):
    """
    使用 Bagging 模型进行预测。
    :param bagging_model: 包含模型、标准化器、PCA 和权重的字典
    :param data: 原始输入数据
    :return: Bagging 模型的预测结果
    """
    scaler = bagging_model["scaler"]
    pca = bagging_model["pca"]
    models = bagging_model["models"]

    # 数据标准化和 PCA 转换
    data_scaled = scaler.transform(data)
    data_pca = pca.transform(data_scaled)

    # 集成模型预测
    predictions = np.array([model.predict(data_pca) for model in models])
    positive_votes = np.sum(predictions == 1, axis=0)
    negative_votes = np.sum(predictions == -1, axis=0)
    final_prediction = np.where(positive_votes > negative_votes, 1, -1)

    return final_prediction


def predict_with_single_model(bagging_model, data):
    """
    使用 Bagging 模型进行预测。
    :param bagging_model: 包含模型、标准化器、PCA 和权重的字典
    :param data: 原始输入数据
    :return: Bagging 模型的预测结果
    """
    scaler = bagging_model["scaler"]
    pca = bagging_model["pca"]
    models = bagging_model["models"]

    # 数据标准化和 PCA 转换
    data_scaled = scaler.transform(data)
    data_pca = pca.transform(data_scaled)

    # 集成模型预测
    predictions = np.array(models.predict(data_pca))

    return predictions
def load_bagging_model(file_path):
    """
    从文件加载 Bagging 模型。
    :param file_path: 模型文件路径
    :return: 加载的 Bagging 模型
    """
    bagging_model = joblib.load(file_path)
    # print(f"Bagging 模型已从 {file_path} 加载")
    return bagging_model
def load_single_model(file_path):
    """
    从文件加载 Bagging 模型。
    :param file_path: 模型文件路径
    :return: 加载的 Bagging 模型
    """
    bagging_model = joblib.load(file_path)
    # print(f"Bagging 模型已从 {file_path} 加载")
    return bagging_model