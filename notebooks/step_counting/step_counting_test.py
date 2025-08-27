from sklearnex import patch_sklearn
patch_sklearn()
from utilities.data import reader
from utilities.data.preprocessing import norm_of_vector
from utilities.labeling import find_peaks
from utilities.features.extract_features import extract_single_features
import numpy as np
np.random.seed(43)
import os
from utilities.evaluate_classifier import predict_with_bagging_model, load_bagging_model
from utilities import generate_dataset,parameters


project_path=parameters["project_dir"]
data_dir = parameters["data_dir"]
window_size = parameters["win_len"]
stride = parameters["win_stride"]
loaded_model = load_bagging_model(os.path.join(project_path, 'result', 'wsbocsvm_model.pkl'))

total_peaks = 0

path = os.path.join(project_path, 'data', 'walking_data', 'test.txt')
data = reader.read_file(path, **{"delimiter": '\t', "header": None}).iloc[:, 0:3]
data=np.array(data)

# 滑窗
for i in range(0, len(data), stride):
    segment2_data= data[i:i + window_size]
    if i == len(data) - 1:
        break
    pd_Features = extract_single_features(segment2_data)
    pred= predict_with_bagging_model(loaded_model,pd_Features)
    pred_flag=pred

    if pred_flag == -1:
        continue
    elif pred_flag == 1:
        segment2 = norm_of_vector(segment2_data)
        mean=np.mean(segment2)
        if mean<1.2:
            peaks = find_peaks(segment2, method='argrelextrema', order=7)
        else:
            peaks = find_peaks(segment2, method='argrelextrema', order=4)
        total_peaks += len(peaks[0])

print(total_peaks)
