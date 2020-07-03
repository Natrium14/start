from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import Birch

# Метод создания модели по выбранному методу из библиотеки sklearn
def generate_model(method, params):
    if method == "dbscan":
        try:
            eps = 0.5
            min_samples = 2

            if bool(params):
                if params["eps"]:
                    eps = params["eps"]
                if params["min_samples"]:
                    min_samples = params["min_samples"]

            model = DBSCAN(eps=eps, min_samples=min_samples)
            print('1')
            return model
        except Exception:
            return None

    if method == "kmeans":
        try:
            n_clusters = 20
            n_init = 10

            if bool(params):
                if params["n_clusters"]:
                    n_clusters = params["n_clusters"]
                if params["n_init"]:
                    n_init = params["n_init"]

            model = KMeans(n_clusters=n_clusters, n_init=n_init)
            return model
        except Exception:
            return None

    if method == "birch":
        try:
            n_clusters = 3

            if bool(params):
                if params["birch_clusters"]:
                    n_clusters = params["birch_clusters"]

            model = Birch(n_clusters=n_clusters)
            return model
        except Exception:
            return None

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
    else:
        return None
