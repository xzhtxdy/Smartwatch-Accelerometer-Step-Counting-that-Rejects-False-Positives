import pandas as pd
from ..config import general_config as config
import os
from utilities.data import preprocessing


def read_file(path, **kwargs):
    """
    Read data from a csv file and return a pandas dataframe.
    """
    return pd.read_csv(path, **kwargs)


def read_files(category='raw'):
    """
    Read data from all csv files in the data directory and return a dictionary
    of dataframes.
    :param category: 'raw', 'processed', or 'open-source'.
    """
    data = {}
    labels = {}
    data_dir = os.path.join(config["project_dir"], 'data', category)
    if category == 'raw':
        for file in os.listdir(data_dir):
            if file.endswith('.csv') or file.endswith('.txt'):
                data[file[:-4]] = read_file(os.path.join(data_dir, file))
    if category == 'open-source':
        for sub_folder in os.listdir(data_dir):
            path = os.path.join(data_dir, sub_folder, "Regular", f"{sub_folder}_Regular.txt")
            accelerometer = read_file(path, **{"delimiter": ' ', "header": None}).iloc[:, 0:3]

            # Filter the data.
            data[sub_folder] = pd.DataFrame(preprocessing.filter(accelerometer, 'lowpass', 5, 3))

            # Read the labels from the steps.txt file.
            label_path = os.path.join(data_dir, sub_folder, "Regular", "steps.txt")
            label = read_file(label_path, **{"delimiter": ' ', "header": None, "index_col": 0})

            # Remove columns before the third active label.
            data[sub_folder].drop(range(0, label.index[2]), axis=0, inplace=True)
            data[sub_folder].drop(range(label.index[-2]+1, data[sub_folder].index[-1]+1), axis=0, inplace=True)
            label.drop([label.index[0], label.index[1]], axis=0, inplace=True)
            label.drop([label.index[-1]], axis=0, inplace=True)

            # Process the labels.
            ## Expand the dataframe to fill the missing values.
            label_filled = preprocessing.fill_df(label)
            ## Fill the missing values with the previous value.
            label_filled.ffill(axis=0, inplace=True)

            labels[sub_folder] = label_filled
    return data, labels
