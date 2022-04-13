import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,max_error, explained_variance_score
from sklearn.metrics.cluster import homogeneity_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler


# Метод обучения созданной модели в генераторе на тестовой выборке
def model_train(model, data, params):

    metrics = []

    if type(model).__name__ == "RandomForestRegressor":
        columns = params["model_columns"]
        column_train = str(params["column_train"])

        X = data.loc[:, columns].values
        y = data.loc[:, column_train].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        RFR_MSE = mean_squared_error(y_test, y_pred)
        RFR_MAE = mean_absolute_error(y_test, y_pred)
        MAX_ERROR = max_error(y_test, y_pred)
        exp_var_score = explained_variance_score(y_test, y_pred)

        print("MAX ERROR: " + str(MAX_ERROR))
        print("explained_variance_score: " + str(exp_var_score))
        print("MSE: {0}".format(RFR_MSE))
        print("MAE: {0}".format(RFR_MAE))

        metrics.append({"MSE": RFR_MSE})
        metrics.append({"MAE": RFR_MAE})
        metrics.append({"MAX_ERROR": MAX_ERROR})
        metrics.append({"explained_variance_score": exp_var_score})

        return model, metrics

    if type(model).__name__ == "GaussianProcessRegressor":
        columns = params["model_columns"]
        column_train = str(params["column_train"])

        data = data[:] # костыль
        X = data.loc[:, columns].values
        X = np.atleast_2d(X)
        x = X
        y = data.loc[:, column_train].values

        model.fit(X, y)
        y_pred = model.predict(x)
        RFR_MSE = mean_squared_error(y, y_pred)
        RFR_MAE = mean_absolute_error(y, y_pred)
        MAX_ERROR = max_error(y, y_pred)
        exp_var_score = explained_variance_score(y, y_pred)

        print("MSE: {0}".format(RFR_MSE))
        print("MAE: {0}".format(RFR_MAE))
        print("MAX ERROR: " + str(MAX_ERROR))
        print("explained_variance_score: " + str(exp_var_score))

        metrics.append({"MSE": RFR_MSE})
        metrics.append({"MAE": RFR_MAE})
        metrics.append({"MAX_ERROR": MAX_ERROR})
        metrics.append({"explained_variance_score": exp_var_score})

        return model, metrics

    if type(model).__name__ == "LinearRegression":
        columns = params["model_columns"]
        column_train = str(params["column_train"])

        X = data.loc[:, columns].values
        y = data.loc[:, column_train].values

        model.fit(X, y)
        y_pred = model.predict(X)

        RFR_MSE = mean_squared_error(y, y_pred)
        RFR_MAE = mean_absolute_error(y, y_pred)
        MAX_ERROR = max_error(y, y_pred)
        exp_var_score = explained_variance_score(y, y_pred)

        print("MAX ERROR: " + str(MAX_ERROR))
        print("explained_variance_score: " + str(exp_var_score))
        print("MSE: {0}".format(RFR_MSE))
        print("MAE: {0}".format(RFR_MAE))

        metrics.append({"MSE": RFR_MSE})
        metrics.append({"MAE": RFR_MAE})
        metrics.append({"MAX_ERROR": MAX_ERROR})
        metrics.append({"explained_variance_score": exp_var_score})

        return model, metrics
    else:
        scaler = StandardScaler()
        columns = list(params["model_columns"])
        data = data[columns]
        data = scaler.fit_transform(data)
        model = model.fit(data)
        return model, metrics


