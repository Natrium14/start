import pandas as pd


# Метод обучения созданной модели в генераторе на тестовой выборке
def model_train(train_data, model):
    x = pd.get_dummies(train_data, columns=["current_stator", "freq", "load"])
    y = train_data["state"]
    model.fit(x, y)
    return model

