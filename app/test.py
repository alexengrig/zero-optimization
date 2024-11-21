import csv

import pytest
import sympy as sp

from app.hooke_jeeves import hooke_jeeves
from app.main import get_function, get_x0
from app.nelder_mead import nelder_mead
from app.powell import powell
from app.utils import format_number

# Параметры тестов
argnames = "input_expr, input_x0, expected_args, expected_value"
argvalues = [
    ('(x-2)**2+(y-3)**2', '0 0', [2, 3], 0),
    ('(x-2)**2+(y-3)**2', '2 2.5', [2, 3], 0),  # с начальной точкой вблизи оптимума
    ('(x-2)**2+(y-3)**2', '10 -5', [2, 3], 0),  # с удалённой начальной точкой
    ('(x-2)**2+(y-3)**2', '4 3', [2, 3], 0),  # с симметричной начальной точкой
    ('(x-2)**2+(y-3)**2', '2 3', [2, 3], 0),  # с точкой в оптимуме
    ('(x-2)**2+(y-3)**2', '2.5 2.5', [2, 3], 0),  # с близким значением для другой пары
    ('(x-2)**2+(y-3)**2', '-2 -3', [2, 3], 0),  # с отрицательными значениями для начальной точки
    # Функция Розенброка
    ('100*(y - x**2)**2 + (x - 1)**2', '2 2', [1, 1], 0)
]

# Для хранения результатов
results = []


# Функция подготовки целевой функции и начальной точки
def prepare_func_x0(input_expr, input_x0):
    expr, param_names = get_function(input_expr)
    params = sp.symbols(param_names)
    expr = sp.sympify(expr)
    func = sp.lambdify(params, expr)
    x0 = get_x0(params, input_x0)
    return func, x0


# Общая функция для выполнения тестов
def run_test(method_name, method, input_expr, input_x0, expected_args, expected_value):
    func, x0 = prepare_func_x0(input_expr, input_x0)
    optimal_args, optimal_value = method(func, x0)
    print(f"{method_name}: {optimal_args} = {optimal_value}")
    # Вычисление отклонений
    delta_args = [abs(opt - exp) for opt, exp in zip(optimal_args, expected_args)]
    delta_value = abs(optimal_value - expected_value)

    # Сохранение результата
    results.append([
        input_expr,
        input_x0,
        method_name,
        [format_number(arg) for arg in optimal_args],  # Оптимальные аргументы (с округлением)
        format_number(optimal_value),  # Значение функции (с округлением)
        [format_number(delta) for delta in delta_args],  # Отклонение аргументов
        format_number(delta_value)  # Отклонение значения
    ])


# Тесты
@pytest.mark.parametrize(argnames, argvalues)
def test_hooke_jeeves(input_expr, input_x0, expected_args, expected_value):
    run_test("hooke_jeeves", hooke_jeeves, input_expr, input_x0, expected_args, expected_value)


@pytest.mark.parametrize(argnames, argvalues)
def test_nelder_mead(input_expr, input_x0, expected_args, expected_value):
    run_test("nelder_mead", nelder_mead, input_expr, input_x0, expected_args, expected_value)


@pytest.mark.parametrize(argnames, argvalues)
def test_powell(input_expr, input_x0, expected_args, expected_value):
    run_test("powell", powell, input_expr, input_x0, expected_args, expected_value)


@pytest.mark.parametrize(argnames, argvalues)
def test_scipy_nelder_mead(input_expr, input_x0, expected_args, expected_value):
    def method(func, x0):
        from scipy.optimize import minimize
        result = minimize(lambda x: func(*x), x0, method='Nelder-Mead')
        return result.x, result.fun

    run_test("scipy_nelder_mead", method, input_expr, input_x0, expected_args, expected_value)


@pytest.mark.parametrize(argnames, argvalues)
def test_scipy_powell(input_expr, input_x0, expected_args, expected_value):
    def method(func, x0):
        from scipy.optimize import minimize
        result = minimize(lambda x: func(*x), x0, method='Powell')
        return result.x, result.fun

    run_test("scipy_powell", method, input_expr, input_x0, expected_args, expected_value)


@pytest.fixture(scope="module", autouse=True)
def after_tests_in_module():
    yield  # Все тесты в модуле завершились
    # Заголовки таблицы
    headers = ["Expression", "x0", "Method", "Optimal Args", "Optimal Value", "Delta of Expected Args",
               "Delta of Expected Value"]

    # Сохранение таблицы в CSV
    output_file = "test_results.csv"
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Записываем заголовки
        writer.writerows(results)  # Записываем строки таблицы
