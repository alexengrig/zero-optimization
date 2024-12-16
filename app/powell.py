import numpy as np


def powell(func, x0, tol=1e-6, max_iter=1000):
    """
    Реализация метода Пауэлла для минимизации функции без использования производных.

    :param func: Целевая функция.
    :param x0: Начальная точка (список или numpy массив).
    :param tol: Точность (порог для остановки).
    :param max_iter: Максимальное число итераций.
    :return: Оптимальная точка и значение функции в этой точке.
    """

    def line_search(curr_func, curr_x, curr_direction):
        """
        Линейный поиск минимума вдоль заданного направления.
        :param curr_func: Целевая функция.
        :param curr_x: Текущая точка.
        :param curr_direction: Направление поиска.
        :return: Оптимальное значение шага alpha.
        """
        alpha = 0
        step_size = 1e-3  # Шаг поиска
        best_alpha = alpha
        best_value = curr_func(*(curr_x + alpha * curr_direction))

        # Простая стратегия поиска минимума вдоль направления
        while True:
            alpha += step_size
            new_value = curr_func(*(curr_x + alpha * curr_direction))
            if new_value < best_value:
                best_value = new_value
                best_alpha = alpha
            else:
                break  # Останавливаемся, если улучшения больше нет
        return best_alpha

    # 1. Инициализация
    x = np.array(x0, dtype=float)  # Начальная точка
    n = len(x)  # Размерность задачи
    directions = np.eye(n)  # Набор начальных направлений (единичные векторы)
    count_iter = 0  # Счетчик итераций

    while count_iter < max_iter:
        x_start = np.copy(x)  # Сохраняем начальную точку текущей итерации

        # 2. Поочередный линейный поиск по всем направлениям
        for i in range(n):
            direction = directions[i]
            alpha = line_search(func, x, direction)
            x = x + alpha * direction  # Обновляем текущую точку

        # 3. Генерация нового направления
        new_direction = x - x_start  # Разница между новой и старой точкой
        if np.linalg.norm(new_direction) < tol:  # Если шаг слишком мал, завершаем
            break
        new_direction /= np.linalg.norm(new_direction)  # Нормируем новое направление

        # Добавляем новое направление и убираем самое старое
        directions = np.vstack([directions[1:], new_direction])

        # 4. Проверка сходимости
        if np.linalg.norm(x - x_start) < tol:
            break

        count_iter += 1

    # 5. Возвращаем результат
    return x, func(*x), count_iter
