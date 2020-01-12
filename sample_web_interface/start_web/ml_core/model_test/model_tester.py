import pandas as pd


def model_test(model, dataset):
    x = pd.get_dummies(dataset, columns=["1"])
    y = dataset["State"]
    model.fit(x, y)
