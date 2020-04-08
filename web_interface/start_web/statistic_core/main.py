import statistics


# Минимальное значение
def get_min(data):
    return data.min()


# Максимальное значение
def get_max(data):
    return data.max()


# Среднее значение
def get_mean(data):
    return data.mean()


# Мода - наиболее встречающееся
def get_mode(data):
    return statistics.mode(data)


# Медиана
def get_median(data):
    return statistics.median(data)


# Дисперсия
def get_variance(data):
    return statistics.variance(data)


# Стандартное отклонение
def get_stdev(data):
    return statistics.stdev(data)

