import pandas as pd


class Data:
    def __init__(self):
        self.data = None
        self.model = None
        self.db_client = None
        self.model_columns = None
        self.train_column = None
        self.metrics = None

    def set_data(self, data1):
        self.data = data1

    def get_data(self):
        return self.data

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def set_metrics(self, metrics):
        self.metrics = metrics

    def get_metrics(self):
        return self.metrics

    def set_db_client(self, db_client):
        self.db_client = db_client

    def get_db_client(self):
        return self.db_client

    def set_model_columns(self, model_columns):
        self.model_columns = model_columns

    def get_model_columns(self):
        return self.model_columns

    def set_train_column(self, train_column):
        self.train_column = train_column

    def get_train_column(self):
        return self.train_column


def data():
    return Data()


