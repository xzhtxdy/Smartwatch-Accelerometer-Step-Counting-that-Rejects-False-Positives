from utilities.features.extract_features import extract_features
from utilities import parameters
''' read data and save features'''
if __name__ == "__main__":
    window_size = parameters["win_len"]
    stride = parameters["win_stride"]
    kwargs = {"cut": (0, 1000), "columns": True}
    pd_Features = extract_features(window_size, stride, **kwargs)

