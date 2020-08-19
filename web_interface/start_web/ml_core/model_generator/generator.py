from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering

# Метод создания модели по выбранному методу из библиотеки sklearn
def generate_model(method, params):
    if method == "dbscan":
        eps = 0.5
        min_samples = 2

        try:
            if params["eps"]:
                eps = params["eps"]
            if params["min_samples"]:
                min_samples = params["min_samples"]
        except:
            pass

        model = DBSCAN(eps=eps, min_samples=min_samples)
        return model


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

    if method == "aggclust":
        try:
            n_clusters = 2

            if bool(params):
                if params["agg_clusters"]:
                    n_clusters = params["agg_clusters"]

            model = AgglomerativeClustering(n_clusters=n_clusters)
            return model
        except Exception:
            return None

    if method == "RandomForestRegressor":
        try:
            n_estimators = 10
            max_depth = 5

            if bool(params):
                if params["n_estimators"]:
                    n_estimators = params["n_estimators"]
                if params["max_depth"]:
                    max_depth = params["max_depth"]

            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
            return model
        except Exception:
            return None

    if method == "GaussianProcessRegressor":
        try:
            kernel = RBF() + C(constant_value=1)
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)
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
