import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


# Метод обучения созданной модели в генераторе на тестовой выборке
def model_train(model, data):
    try:
        if type(model).__name__ == "RandomForestRegressor":
            print("3")
            data = data.drop(['_DATE_', '_NumMotor_'], axis=1)
            print("4")
            #profile_id_list = data._VEL_AXIS_.unique()
            X = data.drop(['_CURR_ACT_', '_VEL_AXIS_', '_MOT_TEMP_'], axis=1).values
            print("5")
            y = data.loc[:, '_MOT_TEMP_'].values
            print("6")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            model.fit(X_train, y_train)
            print("7")
            y_pred = model.predict(X_test)
            RFR_MSE = mean_squared_error(y_test, y_pred)
            RFR_MAE = mean_absolute_error(y_test, y_pred)
            print("MSE: {0}".format(RFR_MSE))
            print("MAE: {0}".format(RFR_MAE))
            return model
        else:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            model = model.fit(data)
            return model
    except Exception:
        return None

