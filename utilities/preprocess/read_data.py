import os
import numpy as np
# from ..parse_config import parameters
import utilities as ut
from utilities.parse_config import parameters
from utilities.data import reader

def read_data_for_one_subject(subject, **kwargs):
    """
    Read data for one subject
    :param subject: subject name
    :param kwargs: the same as read_data_for_one_activity.
    :return:
    """
    Data = {"Activity" + str(x): [] for x in [1,2,3,4,5,6,7,8,9]}

    # for act_num in range(1, 11):
    for act_num in  [1,2,3,4,5,6,7,8,9]:
        activity_name = "Activity" + str(act_num)
        norms_all = read_data_for_one_activity(subject, activity_name, **kwargs)
        Data[activity_name] = norms_all
    return Data


def read_data_for_one_activity(subject, activity, **kwargs):
    directory = os.path.join(parameters['data_dir'],'raw','data', subject, "Gsenser Data", activity)
    data = {}
    norms_all = np.array([])

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            try:
                file_data = reader.read_file(filepath, **{"header": None, "encoding": "utf-8"}).iloc[1:, 1:4].to_numpy().astype(float)
            except UnicodeDecodeError:
                file_data = reader.read_file(filepath, **{"header": None, "encoding": "latin1"}).iloc[1:, 1:4].to_numpy().astype(float)
            # file_data = low_pass_filter(file_data, 10, 25)
            data[filename] = file_data
        elif filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            try:
                file_data = reader.read_file(filepath, **{"delimiter": '\t', "header": None, "encoding": "utf-8"}).iloc[:, 0:3].to_numpy().astype(float)
            except UnicodeDecodeError:
                file_data = reader.read_file(filepath, **{"delimiter": '\t', "header": None, "encoding": "latin1"}).iloc[:, 0:3].to_numpy().astype(float)
            # file_data = low_pass_filter(file_data, 10, 25)
            data[filename] = file_data

        if norms_all.size == 0:
            norms_all = data[filename]
        else:
            if norms_all.shape[1] == data[filename].shape[1]:
                norms_all = np.vstack([norms_all, data[filename]])
            else:
                raise ValueError(f"Dimension mismatch: norms_all has {norms_all.shape[1]} columns, but {filename} has {data[filename].shape[1]} columns")

    return norms_all




def axis_exchange(imu_data):
    imu_data = np.array(imu_data)
    imu_data[:, [0, 1, 2, 3, 4, 5]] = imu_data[:, [1, 0, 2, 4, 3, 5]]
    imu_data[:, [1, 4]] = -imu_data[:, [1, 4]]
    imu_data[:, [0, 1, 2]] = imu_data[:, [0, 1, 2]] / 9.8
    return imu_data

def norm_of_vector(acc, offset=0):
    norm = np.zeros((acc.shape[0], 1))
    for i in range(acc.shape[0]):
        norm[i] = (float(acc[i][0]) ** 2 + float(acc[i][1]) ** 2 + float(acc[i][2]) ** 2)
    return norm - offset

def read_smadata_for_one_activity(subject, activity, row):
    directory = os.path.join(parameters['data_dir'], 'raw', 'data', subject, "Gsenser Data", activity)
    data = {}
    norms_all = np.array([])
    num=row//8
    # print(num)
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            try:
                file_data = reader.read_file(filepath, **{"header": None, "encoding": "utf-8"}).iloc[1:num+1, 1:4].to_numpy().astype(float)
            except UnicodeDecodeError:
                file_data = reader.read_file(filepath, **{"header": None, "encoding": "latin1"}).iloc[1:num+1, 1:4].to_numpy().astype(float)
            # file_data = low_pass_filter(file_data, 10, 25)
            data[filename] = file_data
        elif filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            try:
                file_data = reader.read_file(filepath, **{"delimiter": '\t', "header": None, "encoding": "utf-8"}).iloc[1:num+1, 0:3].to_numpy().astype(float)
            except UnicodeDecodeError:
                file_data = reader.read_file(filepath, **{"delimiter": '\t', "header": None, "encoding": "latin1"}).iloc[1:num+1, 0:3].to_numpy().astype(float)
            # file_data = low_pass_filter(file_data, 10, 25)
            data[filename] = file_data

        if norms_all.size == 0:
            norms_all = data[filename]
        else:
            if norms_all.shape[1] == data[filename].shape[1]:
                norms_all = np.vstack([norms_all, data[filename]])
            else:
                raise ValueError(f"Dimension mismatch: norms_all has {norms_all.shape[1]} columns, but {filename} has {data[filename].shape[1]} columns")

    return norms_all