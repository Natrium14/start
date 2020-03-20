import pandas as pd


# Метод обучения созданной модели в генераторе на тестовой выборке
def model_train(train_data, model):
    try:
        features = ["temp_stator", "current_stator", "freq", "load"]
        x = pd.get_dummies(train_data[features])
        y = train_data["state"]
        model.fit(x, y)
        return model
    except Exception:
        return None

