import pandas as pd
from sklearn.preprocessing import StandardScaler


# Метод обучения созданной модели в генераторе на тестовой выборке
def model_train(model, data):
    try:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        model = model.fit(data)
        return model
    except Exception:
        return None

