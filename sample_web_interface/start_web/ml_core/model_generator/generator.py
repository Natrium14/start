from sklearn.ensemble import RandomForestClassifier


# Метод создания модели по выбранному методу из библиотеки sklearn
def create_model():
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    return model

