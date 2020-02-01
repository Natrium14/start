from sklearn.ensemble import RandomForestClassifier


# Метод создания модели по выбранному методу из библиотеки sklearn
def generate_model():
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    print("Создана модель рандомного леса: " + str(model))
    return model

