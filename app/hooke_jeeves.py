import logging

import numpy as np

logger = logging.getLogger(__name__)


def hooke_jeeves(func, x0, step_size=0.5, step_reduction=0.5, tol=1e-6, max_iter=1000):
    """
    Реализация метода Хука-Дживса для минимизации.

    :param func: Целевая функция.
    :param x0: Начальная точка.
    :param step_size: Начальный шаг.
    :param step_reduction: Коэффициент уменьшения шага.
    :param tol: Точность ε (эпсилон).
    :param max_iter: Максимальное число итераций.
    :return: Оптимальная точка и значение функции в этой точке.
    """

    def explore(x, step):
        """
        Исследующий поиск: пробуем перемещаться вдоль каждой переменной в положительном и отрицательном направлениях.
        """
        logger.debug(f"Исследующий поиск для {x} с шагом {step}")
        for i in range(len(x)):
            logger.debug(f"Для x{i}={x[i]}")
            # Движение в положительном/отрицательном направлении
            for direction in [1, -1]:
                logger.debug(f"Направление {direction}")
                x_new = np.copy(x)  # Создаем копию текущей точки
                x_new[i] += direction * step  # Изменяем координату в текущем направлении
                logger.debug(f"Сдвиг {x_new[i]}")
                value = func(*x)
                value_new = func(*x_new)
                # Проверяем, улучшилась ли функция
                if value_new < value:
                    logger.debug(f"Есть улучшение: {value_new} < {value}")
                    x = x_new  # Обновляем текущую точку
                else:
                    logger.debug(f"Нет улучшения: {value_new} >= {value}")
        return x

    # 1. Инициализация: задаем начальную точку, шаг, и счетчик итераций
    x_base = np.array(x0, dtype=float)  # Начальная точка
    x_opt = np.copy(x_base)  # Оптимальная точка, начинаем с x0
    count_iter = 0  # Счетчик итераций

    while step_size > tol and count_iter < max_iter:
        # Логируем текущее состояние перед исследующим поиском
        logger.debug(
            f"Итерация #{count_iter}: шаг={step_size}"
            f", f({', '.join(str(arg) for arg in x_opt)}) = {func(*x_opt)}"
        )

        # 2. Исследующий поиск: пытаемся найти улучшение вдоль каждой координаты
        x_new = explore(x_opt, step_size)

        # 3. Если улучшений нет, уменьшаем шаг
        if np.allclose(x_new, x_opt):
            new_step = step_size * step_reduction  # Уменьшаем шаг
            logger.debug(f"Улучшений не найдено, уменьшаем шаг до {new_step}")
            step_size = new_step
        else:
            # 4. Поиск по образцу: перемещаемся в направлении улучшения
            logger.debug(f"Найдено улучшение: перемещение из {x_base} в {x_new}")
            x_opt = x_new + (x_new - x_base)  # Делаем шаг в направлении улучшения
            x_base = np.copy(x_new)  # Обновляем базовую точку

        # Увеличиваем счетчик итераций
        count_iter += 1

    # 5. Возвращаем оптимальные параметры и значение функции
    logger.debug(f"Оптимизация завершена: f({', '.join(str(arg) for arg in x_opt)}) = {func(*x_opt)}")
    return x_opt, func(*x_opt)
