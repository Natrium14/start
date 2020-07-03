import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import ml_core.model_generator.generator as generator
import ml_core.model_training.model_train as trainer
import ml_core.model_test.model_tester as tester
import ml_core.model_manage.model_saver as saver


# Метод создания и обучения модели в пакете ml_core;
# Результат - получение объекта обученной модели
def model_train(data, method, params):
    try:
        model = generator.generate_model(method, params)
        model = trainer.model_train(model, data)
        return model
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return None


def model_save(model):
    try:
        saver.save_model(model)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return None
