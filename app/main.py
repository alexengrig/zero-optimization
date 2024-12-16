import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import numpy as np
import sympy as sp

from app.hooke_jeeves import hooke_jeeves
from app.nelder_mead import nelder_mead
from app.powell import powell
from app.utils import format_number


def update_variables(*args):
    for widget in variable_frame.winfo_children():
        widget.destroy()

    try:
        expr_input = function_entry.get()
        if not expr_input.strip():
            return

        param_names = sorted(set(sp.sympify(expr_input).free_symbols), key=str)
        param_names = [str(p) for p in param_names]

        global initial_entries
        initial_entries = []

        # Создаем матрицу для начальных точек (4 колонки)
        rows = len(param_names) // 4 + (len(param_names) % 4)
        for i, var in enumerate(param_names):
            row = i // 4
            col = i % 4

            tk.Label(variable_frame, text=f"{var}").grid(row=row, column=col * 2, padx=5, pady=5, sticky="w")
            initial_entry = tk.Entry(variable_frame, width=10)
            initial_entry.insert(0, "0")
            initial_entry.grid(row=row, column=col * 2 + 1, padx=5, pady=5, sticky="w")
            initial_entries.append(initial_entry)

        update_optimal_fields(param_names)

        # Сбросить подсветку ошибки
        function_label.config(fg="black")

    except Exception as e:
        # Подсветить метку с текстом "Выражение функции" красным
        function_label.config(fg="red")

        # Вывести ошибку в виде alert для пользователя
        messagebox.showerror("Ошибка", f"Ошибка в выражении функции: {e}")


def update_optimal_fields(param_names):
    for widget in optimal_frame.winfo_children():
        widget.destroy()

    global optimal_entries
    optimal_entries = []

    # Создаем матрицу для оптимальных точек (4 колонки)
    rows = len(param_names) // 4 + (len(param_names) % 4)
    for i, var in enumerate(param_names):
        row = i // 4
        col = i % 4

        tk.Label(optimal_frame, text=f"{var}").grid(row=row, column=col * 2, padx=5, pady=5, sticky="w")
        optimal_entry = tk.Entry(optimal_frame, width=10)
        optimal_entry.grid(row=row, column=col * 2 + 1, padx=5, pady=5, sticky="w")
        optimal_entries.append(optimal_entry)


def show_graph():
    update_variables()
    return None


def optimize():
    update_variables()
    try:
        expr_input = function_entry.get()
        param_names = sorted(set(sp.sympify(expr_input).free_symbols), key=str)
        param_names = [str(p) for p in param_names]

        x0 = [float(entry.get()) for entry in initial_entries]

        method = method_combobox.get()
        alpha = float(alpha_entry.get())
        beta = float(beta_entry.get())

        params = sp.symbols(param_names)
        expr = sp.sympify(expr_input)
        func = sp.lambdify(params, expr)

        if method == "Хука-Дживса":
            optimal_args, optimal_value = hooke_jeeves(func, np.array(x0))
        elif method == "Нелдера-Мида":
            optimal_args, optimal_value = nelder_mead(func, np.array(x0))
        elif method == "Пауэлла":
            optimal_args, optimal_value = powell(func, np.array(x0))
        else:
            raise ValueError("Не выбран метод оптимизации")

        for i, entry in enumerate(optimal_entries):
            entry.delete(0, tk.END)
            entry.insert(0, format_number(optimal_args[i]))
        result_entry.delete(0, tk.END)
        result_entry.insert(0, format_number(optimal_value))

    except Exception as e:
        messagebox.showerror("Ошибка", str(e))


root = tk.Tk()
root.title("Оптимизация функции")
root.geometry("400x800")

# Ввод (заголовок)
tk.Label(root, text="Ввод", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=5, padx=5, pady=5, sticky="w")

# Выражение функции
function_label = tk.Label(root, text="Выражение функции")
function_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
function_entry = tk.Entry(root)
function_entry.insert(0, "(x-2)**2+(y-3)**2")
function_entry.grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky="ew")
function_entry.bind("<Return>", update_variables)  # Обновление при нажатии Enter

# Кнопка "Ввод" рядом с выражением функции
tk.Button(root, text="Ввод", command=update_variables).grid(row=1, column=4, padx=5, pady=5, sticky="ew")

# Начальная точка
tk.Label(root, text="Начальная точка").grid(row=2, column=0, padx=5, pady=5, sticky="w")
variable_frame = tk.Frame(root)
variable_frame.grid(row=3, column=0, columnspan=5, padx=5, pady=5, sticky="ew")

# Инпуты для min и max, и кнопка "График"
tk.Label(root, text="min:").grid(row=4, column=0, padx=5, pady=5, sticky="e")  # Выравнивание по левому краю
min_entry = tk.Entry(root, width=10)
min_entry.insert(0, "-100")  # Значение по умолчанию для min
min_entry.grid(row=4, column=1, padx=5, pady=5, sticky="ew")  # Растягиваем по горизонтали

tk.Label(root, text="max:").grid(row=4, column=2, padx=5, pady=5, sticky="w")  # Выравнивание по левому краю
max_entry = tk.Entry(root, width=10)
max_entry.insert(0, "100")  # Значение по умолчанию для max
max_entry.grid(row=4, column=3, padx=5, pady=5, sticky="ew")  # Растягиваем по горизонтали

# Кнопка "График" в одной строке с min и max
tk.Button(root, text="График", command=show_graph).grid(row=4, column=4, padx=5, pady=5,
                                                        sticky="ew")  # Выравнивание по левому краю

# Метод (заголовок)
tk.Label(root, text="Метод", font=("Arial", 12, "bold")).grid(row=5, column=0, columnspan=5, padx=5, pady=5, sticky="w")

# Метод и кнопка "Найти" в одной строке
tk.Label(root, text="Тип").grid(row=6, column=0, padx=5, pady=5, sticky="w")
method_combobox = ttk.Combobox(root, values=["Хука-Дживса", "Нелдера-Мида", "Пауэлла"], state="readonly")
method_combobox.set("Хука-Дживса")
method_combobox.grid(row=6, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

# Кнопка "Найти" справа от выбора метода
tk.Button(root, text="Найти", command=optimize).grid(row=6, column=4, padx=5, pady=5, sticky="ew")

# Параметры метода
tk.Label(root, text="Параметры метода").grid(row=7, column=0, columnspan=5, padx=5, pady=5, sticky="w")

tk.Label(root, text="alpha").grid(row=8, column=0, padx=5, pady=5, sticky="w")
alpha_entry = tk.Entry(root, width=10)
alpha_entry.insert(0, "1.0")
alpha_entry.grid(row=8, column=1, padx=5, pady=5, sticky="ew")

tk.Label(root, text="beta").grid(row=9, column=0, padx=5, pady=5, sticky="w")
beta_entry = tk.Entry(root, width=10)
beta_entry.insert(0, "0.5")
beta_entry.grid(row=9, column=1, padx=5, pady=5, sticky="ew")

# Вывод (заголовок)
tk.Label(root, text="Вывод", font=("Arial", 12, "bold")).grid(row=10, column=0, columnspan=5, padx=5, pady=5,
                                                              sticky="w")

# Результат
tk.Label(root, text="Значение функции").grid(row=11, column=0, padx=5, pady=5, sticky="w")
result_entry = tk.Entry(root, width=30)
result_entry.grid(row=11, column=1, columnspan=4, padx=5, pady=5, sticky="ew")

# Оптимальная точка
tk.Label(root, text="Оптимальная точка").grid(row=12, column=0, padx=5, pady=5, sticky="w")
optimal_frame = tk.Frame(root)
optimal_frame.grid(row=13, column=0, columnspan=5, padx=5, pady=5, sticky="ew")

update_variables()

root.mainloop()
