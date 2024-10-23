import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from hnsw import HNSW, l2_distance


# Функция для вычисления точного расстояния (ground truth)
def exact_nearest_neighbors(data, query, k):
    neigh = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='l2')
    neigh.fit(data)
    distances, indices = neigh.kneighbors([query])
    return indices[0]


# Функция для расчета recall
def calculate_recall(true_neighbors, found_neighbors):
    intersection = set(true_neighbors).intersection(set(found_neighbors))
    return len(intersection) / len(true_neighbors)


# Эксперимент с различными значениями ef
def run_experiment(hnsw, data, test_queries, k, ef_values):
    results = []
    for ef in ef_values:
        total_recall = 0
        total_computations = 0
        num_queries = len(test_queries)

        for query in test_queries:
            # Точные соседи
            true_neighbors = exact_nearest_neighbors(data, query, k)

            # Поиск в HNSW с текущим значением ef
            found_neighbors = [idx for idx, dist in hnsw.search(query, k=k, ef=ef)]

            # Подсчет recall
            recall = calculate_recall(true_neighbors, found_neighbors)
            total_recall += recall

            # Число вычислений расстояний (аналогично количеству элементов в observed)
            computations = len(hnsw.search(query, k=k, ef=ef, return_observed=True))
            total_computations += computations

        avg_recall = total_recall / num_queries
        avg_computations = total_computations / num_queries

        results.append((ef, avg_recall, avg_computations / len(data)))  # нормализуем на количество вершин

    # Сохранение результатов в CSV
    with open('experiment_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ef', 'recall', 'computations_per_node'])
        writer.writerows(results)

    return results


# Визуализация результатов
def plot_results(results):
    ef_values, recalls, computations = zip(*results)

    plt.figure(figsize=(8, 6))
    plt.scatter(recalls, computations, color='blue')

    # Подписываем ef рядом с каждой точкой
    for i, ef in enumerate(ef_values):
        plt.text(recalls[i], computations[i], f'ef={ef}', fontsize=9)

    plt.title('Recall vs Computations per Node')
    plt.xlabel('Recall')
    plt.ylabel('Computations per Node')
    plt.grid(True)
    plt.show()


# Параметры эксперимента
m = 50
m0 = 50
ef_construction = 30
n = 10000  # Количество точек
dim = 128   # Размерность данных
k = 5     # Количество ближайших соседей
ef_values = [5, 10, 20, 30, 40, 50]  # Значения ef для эксперимента

# Создание модели и данных
hnsw = HNSW(distance_func=l2_distance, m=m, m0=m0, ef_construction=ef_construction)
data = np.array(np.float32(np.random.random((n, dim))))
test_queries = np.array(np.float32(np.random.random((100, dim))))  # 100 тестовых запросов

# Добавление элементов в HNSW
for point in data:
    hnsw.add(point)

# Запуск эксперимента
results = run_experiment(hnsw, data, test_queries, k, ef_values)

# Визуализация результатов
plot_results(results)
