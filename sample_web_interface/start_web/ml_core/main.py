import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier


def hello_world():
    return 'Hello world!'


def create_model(train_data):
    x = pd.get_dummies(train_data, columns=["1","2"])
    y = train_data["State"]
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(x, y)
    return model


