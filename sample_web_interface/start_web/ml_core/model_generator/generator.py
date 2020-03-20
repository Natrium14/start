from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingClassifier


# Метод создания модели по выбранному методу из библиотеки sklearn
def generate_model(method):
    if method == "RandomForestClassifier":
        try:
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
            return model
        except Exception:
            return None
    if method == "GradientBoostingClassifier":
        try:
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=1)
            return model
        except Exception:
            return None
    if method == "AdaBoostClassifier":
        try:
            model = AdaBoostClassifier(n_estimators=100, max_depth=5, random_state=1)
            return model
        except Exception:
            return None
    if method == "AdaBoostRegressor":
        try:
            model = AdaBoostRegressor(n_estimators=100, max_depth=5, random_state=1)
            return model
        except Exception:
            return None
    if method == "BaggingClassifier":
        try:
            model = BaggingClassifier(n_estimators=100, max_depth=5, random_state=1)
            return model
        except Exception:
            return None
    if method == "BaggingRegressor":
        try:
            model = BaggingRegressor(n_estimators=100, max_depth=5, random_state=1)
            return model
        except Exception:
            return None
    if method == "ExtraTreesClassifier":
        try:
            model = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=1)
            return model
        except Exception:
            return None
    if method == "GradientBoostingRegressor":
        try:
            model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=1)
            return model
        except Exception:
            return None
    if method == "StackingClassifier":
        try:
            model = StackingClassifier(n_estimators=100, max_depth=5, random_state=1)
            return model
        except Exception:
            return None
    else:
        return None
