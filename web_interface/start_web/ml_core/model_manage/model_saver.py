import time

import management.views as m_views
import pickle


# MongoClient
client = None


# метод подключения к БД
def get_client():
    global client
    client = m_views.get_client()
    return client


# метод сохранения модели в БД
def save_model(model):
    global client

    client = get_client()
    model_name = str(type(model).__name__)
    pickled_model = pickle.dumps(model)
    db = client['start']
    collection = db['models']
    info = collection.insert_one(
        {
            model_name: pickled_model,
            'name': model_name,
            'created_time': time.time()
        })
    print(info.inserted_id, ' saved model!')
