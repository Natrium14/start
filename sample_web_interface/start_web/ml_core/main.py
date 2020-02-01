import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sample_web_interface.start_web.ml_core.model_generator.generator as generator
import sample_web_interface.start_web.ml_core.model_training.model_train as trainer
import sample_web_interface.start_web.ml_core.model_test.model_tester as tester


# Метод создания и обучения модели в пакете ml_core;
# Результат - получение объекта обученной модели
def model_train(dataset):
    model = generator.generate_model()
    print("Переход к обучению модели")
    trainer.model_train(dataset, model)
    print("Возврат модели из ml_core: " + str(model))
    return model


