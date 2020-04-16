import pandas as pd


# Метод обучения созданной модели в генераторе на тестовой выборке
def model_train(data, model):
    try:
        model = model.fit(data)
        return model
    except Exception:
        return None

