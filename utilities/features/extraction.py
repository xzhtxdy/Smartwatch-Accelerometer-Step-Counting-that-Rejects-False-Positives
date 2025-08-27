import numpy as np
import calculation as calc

def exact_features(data):
    """
    Extracts features from data.
    :param data: data to extract features from
    :return: features
    """
    calc_functions = [calc.mean_absolute_value, calc.waveform_length, calc.zero_crossings, calc.slope_sign_changes,
                        calc.root_mean_square, calc.mean, calc.std, calc.skewness, calc.kurtosis,
                        calc.median_abs_deviation]
    features = []
    for calc_function in calc_functions:
        features.append(calc_function(data))
    return np.array(features).flatten()
