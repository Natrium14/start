import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler



# Метод обучения созданной модели в генераторе на тестовой выборке
def model_train(model, data):
    try:
        if type(model).__name__ == "RandomForestRegressor":
            data = data.drop(['_DATE_', '_NumMotor_'], axis=1)
            #profile_id_list = data._VEL_AXIS_.unique()
            #X = data.drop(['_CURR_ACT_', '_VEL_AXIS_', '_MOT_TEMP_'], axis=1).values
            X = data.drop(['_MOT_TEMP_', '_VEL_AXIS_'], axis=1).values
            y = data.loc[:, '_MOT_TEMP_'].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            RFR_MSE = mean_squared_error(y_test, y_pred)
            RFR_MAE = mean_absolute_error(y_test, y_pred)
            print("MSE: {0}".format(RFR_MSE))
            print("MAE: {0}".format(RFR_MAE))
            return model
        if type(model).__name__ == "GaussianProcessRegressor":
            data = data.drop(['_DATE_', '_NumMotor_'], axis=1)[:3000]
            X = data.drop(['_MOT_TEMP_', '_VEL_AXIS_'], axis=1).values
            X = np.atleast_2d(X)
            x = X
            y = data.loc[:, '_MOT_TEMP_'].values

            model.fit(X, y)
            #y_pred, sigma = model.predict(x, return_std=True)
            return model
        else:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            model = model.fit(data)
            return model
    except Exception:
        return None

