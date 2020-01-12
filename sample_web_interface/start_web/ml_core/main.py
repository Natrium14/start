import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sample_web_interface.start_web.ml_core.model_generator as generator
import sample_web_interface.start_web.ml_core.model_training as trainer
import sample_web_interface.start_web.ml_core.model_test as tester


def hello_world():
    return 'Hello world!'


def index(dataset):
    model = generator.create_model()
    trainer.model_train(dataset, model)
    metrics = tester.model_test(dataset, model)
    return metrics


