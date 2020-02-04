from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# Метод создания модели по выбранному методу из библиотеки sklearn
def generate_model(method):
    if method == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
        return model
    if method == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=1)
        return model
    else:
        return None
