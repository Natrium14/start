import pandas as pd


# Метод для получения метрик модели (точности, полноты и тд)
def get_model_metrics(model):
    return None


# Метод тестирования обученной модели на сырых/тестовых данных
def model_test(dataset, model):
    x = pd.get_dummies(dataset, columns=["1"])
    y = dataset["State"]
    model.fit(x, y)
    return get_model_metrics(model)


