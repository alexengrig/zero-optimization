import logging
import re
import sys

import numpy as np
import sympy as sp

from app.hooke_jeeves import hooke_jeeves
from app.nelder_mead import nelder_mead
from app.powell import powell
from app.utils import format_number

logger = logging.getLogger(__name__)


def get_function(expr_input):
    param_names = sorted(set(re.findall(r'[a-zA-Z]+\d*', expr_input)))
    if len(param_names) == 0:
        raise ValueError("Функция не имеет параметров")
    return expr_input, param_names


def input_function():
    expr_input = input("Выражение целевой функции: ")
    return get_function(expr_input)


def get_x0(params, x0_input):
    x0_input = x0_input.replace(',', '.')
    x0_values = re.findall(r"[-+]?\d*\.\d+|\d+", x0_input)
    x0 = np.array([float(x) for x in x0_values])
    if len(x0) != len(params):
        raise ValueError(f"Ожидаемое кол-во аргументов {len(params)}, но текущее {len(x0)}")
    return x0


def input_x0(params):
    while True:
        x0_input = input(f"Аргументы начальной точки для параметров ({' '.join(str(p) for p in params)}): ")
        try:
            return get_x0(params, x0_input)
        except ValueError as e:
            print(str(e))


def create_func_x0(expr_input, param_names):
    params = sp.symbols(param_names)
    expr = sp.sympify(expr_input)
    func = sp.lambdify(params, expr)
    return func


def main():
    expr, param_names = input_function()
    print(f"f({', '.join(param_names)}) = {expr}")

    params = sp.symbols(param_names)
    expr = sp.sympify(expr)
    func = sp.lambdify(params, expr)
    x0 = input_x0(params)

    print('hooke jeeves')
    optimal_args, optimal_value = hooke_jeeves(func, x0)
    print(f"min f({', '.join(format_number(arg) for arg in optimal_args)}) = {format_number(optimal_value)}")

    print('nelder mead')
    optimal_args, optimal_value = nelder_mead(func, x0)
    print(f"min f({', '.join(format_number(arg) for arg in optimal_args)}) = {format_number(optimal_value)}")

    print('powel')
    optimal_args, optimal_value = powell(func, x0)
    print(f"min f({', '.join(format_number(arg) for arg in optimal_args)}) = {format_number(optimal_value)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    main()
