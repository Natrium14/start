import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Метод обучения созданной модели в генераторе на тестовой выборке
def model_train(train_data, model):
    x = pd.get_dummies(train_data, columns=["1","2"])
    y = train_data["State"]
    model.fit(x, y)
    return model

