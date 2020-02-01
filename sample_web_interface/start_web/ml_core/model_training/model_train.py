import pandas as pd


# Метод обучения созданной модели в генераторе на тестовой выборке
def model_train(train_data, model):
    print("Данные для обучения: ")
    print(train_data.head())
    features = ["temp_stator", "current_stator", "freq", "load"]
    x = pd.get_dummies(train_data[features])
    print("X:")
    print(x.head())
    y = train_data["state"]
    print("Y:")
    print(y.head())
    print("Получены данные для обучения модели: " + str(model))
    model.fit(x, y)
    print("Модель обучена: " + str(model))
    return model

