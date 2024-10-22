#!/usr/bin/env python
# coding: utf-8

from heapq import heappop, heappush
from math import log2
import random

import numpy as np


def l2_distance(a, b):
    return np.linalg.norm(a - b)


def heuristic(candidates, curr, k, distance_func, data):
    candidates = sorted(candidates, key=lambda a: a[1])
    result_indx_set = {candidates[0][0]}
    result = [candidates[0]]
    added_data = [data[candidates[0][0]]]
    for c, curr_dist in candidates[1:]:
        c_data = data[c]
        if curr_dist < min(distance_func(c_data, a) for a in added_data):
            result.append((c, curr_dist))
            result_indx_set.add(c)
            added_data.append(c_data)
    for c, curr_dist in candidates:  # optional. uncomment to build neighborhood exactly with k elements.
        if len(result) < k and (c not in result_indx_set):
            result.append((c, curr_dist))

    return result


def k_closest(candidates: list, curr, k, distance_func, data):
    return sorted(candidates, key=lambda a: a[1])[:k]


class HNSW:
    # self._graphs[level][i] contains a {j: dist} dictionary,
    # where j is a neighbor of i and dist is distance

    def _distance(self, x, y):
        return self.distance_func(x, [y])[0]

    def vectorized_distance_(self, x, ys):
        return [self.distance_func(x, y) for y in ys]

    def __init__(
            self,
            distance_func,
            m=5,
            ef=10,
            ef_construction=30,
            m0=None,
            neighborhood_construction=heuristic,
            vectorized=False
    ):
        self.data = []
        self.distance_func = distance_func
        self.neighborhood_construction = neighborhood_construction

        if vectorized:
            self.distance = self._distance
            self.vectorized_distance = distance_func
        else:
            self.distance = distance_func
            self.vectorized_distance = self.vectorized_distance_

        self._m = m
        self._ef = ef
        self._ef_construction = ef_construction
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / log2(m)
        self._graphs = []
        self._enter_point = None

    def add(self, elem, ef=None):
        if ef is None:
            ef = self._ef

        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m

        # Определяем уровень, на котором элемент будет вставлен
        level = int(-log2(random.random()) * self._level_mult) + 1
        idx = len(data)
        data.append(elem)

        if point is not None:  # Если граф не пуст, у нас есть входная точка
            dist = distance(elem, data[point])

            # Обходим граф на уровнях, которые не требуют вставки элемента
            for layer in reversed(graphs[level:]):
                point, dist = self.beam_search(graph=layer, q=elem, k=1, eps=[point], ef=1)[0]

            layer0 = graphs[0]
            for layer in reversed(graphs[:level]):
                level_m = m if layer is not layer0 else self._m0

                # Получаем кандидатов для вставки
                candidates = self.beam_search(graph=layer, q=elem, k=level_m * 2, eps=[point], ef=self._ef_construction)

                # Оптимизация: используем только наиболее релевантные кандидаты
                unique_neighbors = {}
                for c, curr_dist in candidates:
                    if c not in unique_neighbors or curr_dist < unique_neighbors[c]:
                        unique_neighbors[c] = curr_dist

                # Сортируем кандидатов и оставляем только топ-M
                sorted_neighbors = sorted(unique_neighbors.items(), key=lambda item: item[1])[:level_m]
                layer[idx] = sorted_neighbors

                # Обновляем обратные ссылки
                for j, dist in sorted_neighbors:
                    # Оптимизация: используем прямой поиск соседей для обновления
                    if j in layer:
                        candidates_j = layer[j] + [(idx, dist)]
                        neighbors_j = self.neighborhood_construction(candidates=candidates_j, curr=j, k=level_m,
                                                                     distance_func=self.distance_func, data=self.data)
                        layer[j] = neighbors_j
                    else:
                        # В случае отсутствия записи создаем новую
                        layer[j] = [(idx, dist)]

        # Создание новых уровней, если это необходимо
        for _ in range(len(graphs), level):
            graphs.append({idx: []})
            self._enter_point = idx

    # can be used for search after jump
    def search(self, q, k=1, ef=10, level=0, return_observed=True):
        graphs = self._graphs
        point = self._enter_point
        for layer in reversed(graphs[level:]):
            point, _ = self.beam_search(layer, q=q, k=1, eps=[point], ef=1)[0]

        return self.beam_search(graph=graphs[level], q=q, k=k, eps=[point], ef=ef, return_observed=return_observed)

    def beam_search(self, graph, q, k, eps, ef, ax=None, marker_size=20, return_observed=False):
        '''
        graph – the layer where the search is performed
        q - query
        k - number of closest neighbors to return
        eps – entry points [vertex_id, ..., vertex_id]
        ef – size of the beam
        observed – if True returns the full of elements for which the distance were calculated
        returns – a list of tuples [(vertex_id, distance), ... , ]
        '''
        # Priority queue: (negative distance, vertex_id)
        candidates = []
        visited = set()  # set of vertex used for extending the set of candidates
        observed = {}  # dict: vertex_id -> float – set of vertexes for which the distance were calculated

        if ax:
            ax.scatter(x=q[0], y=q[1], s=marker_size, color='red', marker='^')
            ax.annotate('query', (q[0], q[1]))

        # Initialize the queue with the entry points
        for ep in eps:
            dist = self.distance_func(q, self.data[ep])
            heappush(candidates, (dist, ep))
            observed[ep] = dist

        while candidates:
            # Get the closest vertex (furthest in the max-heap sense)
            dist, current_vertex = heappop(candidates)

            if ax:
                ax.scatter(x=self.data[current_vertex][0], y=self.data[current_vertex][1], s=marker_size, color='red')
                ax.annotate(len(visited), self.data[current_vertex])

            # check stop conditions #####
            observed_sorted = sorted(observed.items(), key=lambda a: a[1])
            # print(observed_sorted)
            ef_largets = observed_sorted[min(len(observed) - 1, ef - 1)]
            # print(ef_largets[0], '<->', -dist)
            if ef_largets[1] < dist:
                break
            #############################

            # Add current_vertex to visited set
            visited.add(current_vertex)

            # Check the neighbors of the current vertex
            for neighbor, _ in graph[current_vertex]:
                if neighbor not in observed:
                    dist = self.distance_func(q, self.data[neighbor])
                    # if neighbor not in visited:
                    heappush(candidates, (dist, neighbor))
                    observed[neighbor] = dist
                    if ax:
                        ax.scatter(x=self.data[neighbor][0], y=self.data[neighbor][1], s=marker_size, color='yellow')
                        # ax.annotate(len(visited), (self.data[neighbor][0], self.data[neighbor][1]))
                        ax.annotate(len(visited), self.data[neighbor])

        # Sort the results by distance and return top-k
        observed_sorted = sorted(observed.items(), key=lambda a: a[1])
        if return_observed:
            return observed_sorted
        return observed_sorted[:k]

    def save_graph_plane(self, file_path):
        with open(file_path, "w", encoding='utf-8') as f:
            f.write(f'{len(self.data)}\n')

            for x in self.data:
                s = ' '.join([a.astype('str') for a in x])
                f.write(f'{s}\n')

            for graph in self._graphs:
                for src, neighborhood in graph.items():
                    for dst, _ in neighborhood:
                        f.write(f'{src} {dst}\n')

# n = int(sys.argv[1]) # graph size
# dim = int(sys.argv[2]) # vector dimensionality
# m = int(sys.argv[3]) # avg number of vertex
# m0 = int(sys.argv[3]) # avg number of vertex for the lower layer

# hnsw = HNSW( distance_func=l2_distance, m=5, m0=7, ef=10, ef_construction=30,  neighborhood_construction = heuristic)

# k = 5
# dim = 2
# n = 1000
# data = np.array(np.float32(np.random.random((n, dim))))


# for x in data:
#     hnsw.add(x)
