import xgboost as xgb


def feature_selection(X, y, accumulative_threshold=0.90):
    model = xgb.XGBClassifier()
    model.fit(X, y)
    importance = model.feature_importances_

    sorted_importance = sorted(enumerate(importance), key=lambda x: x[1], reverse=True)
    top_values = [value for index, value in sorted_importance]
    top_indics = [index for index, value in sorted_importance]

    sum = 0
    for i in range(len(top_values)):
        sum += top_values[i]
        if sum >= accumulative_threshold:
            break
    print("Select {} features".format(i + 1))
    return top_indics[:i + 1], importance
