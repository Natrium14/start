import pandas as pd

from sklearn.metrics import mean_absolute_error


# Метод для получения метрик модели (точности, полноты и тд)
def get_model_metrics(model, data_train, column_test):
    metrics = {}
    metrics['MAE'] = get_MAE(model,data_train,column_test)
    return metrics


# Средняя абсолютная ошибка
def get_MAE(model, data_train, column_test):
    y_true = data_train.loc[column_test].values
    y_pred = model.predict(data_train.drop[column_test])
    return mean_absolute_error(y_true, y_pred)


