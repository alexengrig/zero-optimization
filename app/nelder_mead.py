import logging

import numpy as np

logger = logging.getLogger(__name__)


def nelder_mead(func, x0, alpha=1.0, beta=0.5, gamma=2.0, tol=1e-6, max_iter=1000):
    """
    Реализация метода Нелдера-Мида для минимизации функции.

    :param func: Целевая функция.
    :param x0: Начальная точка (вектор).
    :param alpha: Коэффициент отражения.
    :param beta: Коэффициент сжатия.
    :param gamma: Коэффициент растяжения.
    :param tol: Точность ε (эпсилон).
    :param max_iter: Максимальное число итераций.
    :return: Оптимальная точка и значение функции в этой точке.
    """

    def sort_simplex(simplex, func):
        """
        Сортировка точек симплекса по значению функции.
        """
        values = [func(*point) for point in simplex]
        sorted_simplex = [x for _, x in sorted(zip(values, simplex), key=lambda pair: pair[0])]
        return sorted_simplex

    # 1. Инициализация симплекса
    n = len(x0)
    simplex = [x0.tolist()]  # Начальная точка
    for i in range(n):
        x_new = np.copy(x0)
        x_new[i] += 1.0  # Отклоняем одну координату для формирования симплекса
        simplex.append(x_new.tolist())

    simplex = np.array(simplex)
    count_iter = 0

    while count_iter < max_iter:
        # 2. Сортировка точек симплекса по значению функции
        simplex = sort_simplex(simplex, func)
        x_best = simplex[0]
        x_worst = simplex[-1]
        x_second_worst = simplex[-2]
        centroid = np.mean(simplex[:-1], axis=0)

        # 3. Шаг отражения
        x_reflection = centroid + alpha * (centroid - x_worst)
        if func(*x_reflection) < func(*x_second_worst):
            if func(*x_reflection) < func(*x_best):
                # Шаг растяжения
                x_expansion = centroid + gamma * (x_reflection - centroid)
                if func(*x_expansion) < func(*x_reflection):
                    simplex[-1] = x_expansion
                else:
                    simplex[-1] = x_reflection
            else:
                simplex[-1] = x_reflection
        else:
            # Шаг сжатия
            x_contraction = centroid + beta * (x_worst - centroid)
            if func(*x_contraction) < func(*x_worst):
                simplex[-1] = x_contraction
            else:
                # Шаг редукции
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])

        # 4. Проверка на сходимость
        if np.linalg.norm(simplex[0] - simplex[-1]) < tol:
            break

        count_iter += 1

    # Возвращаем оптимальную точку и значение функции
    return simplex[0], func(*simplex[0]), count_iter
