import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sample_web_interface.start_web.ml_core.model_generator as generator
import sample_web_interface.start_web.ml_core.model_training as trainer
import sample_web_interface.start_web.ml_core.model_test as tester


# Основной метод создания и обучения модели в пакете ml_core;
# Результат - получение метрик (точность, полнота)
def model_train(dataset):
    model = generator.create_model()
    trainer.model_train(dataset, model)
    metrics = tester.model_test(dataset, model)
    return metrics


