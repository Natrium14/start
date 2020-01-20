import statistics


# Среднее значение
def get_mean(data):
    return statistics.mean(data)


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

