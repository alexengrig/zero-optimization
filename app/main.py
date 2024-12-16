import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm  # Для цветовой карты

matplotlib.use('TkAgg')

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import numpy as np
import sympy as sp

from app.hooke_jeeves import hooke_jeeves
from app.nelder_mead import nelder_mead
from app.powell import powell
from app.utils import format_number

checkbuttons = []  # Список для переменных BooleanVar
checkbuttons_widgets = []  # Список для чекбоксов


def update_variables(*args):
    for widget in variable_frame.winfo_children():
        widget.destroy()

    try:
        expr_input = function_entry.get()
        if not expr_input.strip():
            return

        param_names = sorted(set(sp.sympify(expr_input).free_symbols), key=str)
        param_names = [str(p) for p in param_names]

        global initial_entries, checkbuttons, checkbuttons_widgets, selected_checkboxes
        initial_entries = []
        checkbuttons = []
        checkbuttons_widgets = []  # Обновляем список чекбоксов
        selected_checkboxes = []

        # Создаем матрицу для начальных точек (2 колонки)
        rows = len(param_names) // 2 + (1 if len(param_names) % 2 != 0 else 0)
        for i, var in enumerate(param_names):
            row = i // 2
            col = i % 2

            tk.Label(variable_frame, text=f"{var}").grid(row=row, column=col * 3, padx=5, pady=5, sticky="w")
            initial_entry = tk.Entry(variable_frame, width=10)
            initial_entry.insert(0, "0")
            initial_entry.grid(row=row, column=col * 3 + 1, padx=5, pady=5, sticky="w")
            initial_entries.append(initial_entry)

            # Чекбокс для выбора переменной
            var_selected = tk.BooleanVar()
            checkbutton = tk.Checkbutton(variable_frame, variable=var_selected,
                                         command=lambda: update_checkboxes())
            checkbutton.grid(row=row, column=col * 3 + 2, padx=5, pady=5)
            checkbuttons.append(var_selected)
            checkbuttons_widgets.append(checkbutton)  # Сохраняем виджет чекбокса

            # Автоматически выделяем первые два чекбокса
            if i < 2:
                var_selected.set(True)
                selected_checkboxes.append(i)

        update_checkboxes()  # Применяем состояние чекбоксов
        update_optimal_fields(param_names)

        # Сбросить подсветку ошибки
        function_label.config(fg="black")

    except Exception as e:
        # Подсветить метку с текстом "Выражение функции" красным
        function_label.config(fg="red")

        # Вывести ошибку в виде alert для пользователя
        messagebox.showerror("Ошибка", f"Ошибка в выражении функции: {e}")
        raise e


def update_checkboxes():
    # Пересоздаем список виджетов чекбоксов
    selected_checkboxes = [i for i, var in enumerate(checkbuttons) if var.get()]

    for i, checkbutton_var in enumerate(checkbuttons):  # Работаем с переменными BooleanVar
        if len(selected_checkboxes) < 2 or i in selected_checkboxes:
            # Чекбокс активен
            checkbuttons_widgets[i].config(state="normal")
        else:
            # Чекбокс отключен
            checkbuttons_widgets[i].config(state="disabled")


def update_optimal_fields(param_names):
    for widget in optimal_frame.winfo_children():
        widget.destroy()

    global optimal_entries
    optimal_entries = []

    # Создаем матрицу для оптимальных точек (3 колонки)
    rows = len(param_names) // 2 + (len(param_names) % 2)
    for i, var in enumerate(param_names):
        row = i // 2
        col = i % 2

        tk.Label(optimal_frame, text=f"{var}").grid(row=row, column=col * 2, padx=5, pady=5, sticky="w")
        optimal_entry = tk.Entry(optimal_frame, width=10)
        optimal_entry.grid(row=row, column=col * 2 + 1, padx=5, pady=5, sticky="w")
        optimal_entries.append(optimal_entry)


def show_graph():
    try:
        # Получаем выражение функции
        expr_input = function_entry.get()
        param_names = sorted(set(sp.sympify(expr_input).free_symbols), key=str)
        param_names = [str(p) for p in param_names]

        # Проверяем, выбрано ли хотя бы одна переменная
        selected_indices = [i for i, var in enumerate(checkbuttons) if var.get()]
        if len(selected_indices) == 0:
            messagebox.showerror("Ошибка", "Для построения графика выберите хотя бы одну переменную.")
            return

        # Если выбрана только одна переменная, строим 2D-график
        if len(selected_indices) == 1:
            var = param_names[selected_indices[0]]
            idx = selected_indices[0]

            # Получаем значения начальной точки
            x0 = [float(entry.get()) for entry in initial_entries]

            # Устанавливаем диапазон для графика
            x_min = float(min_entry.get())
            x_max = float(max_entry.get())

            # Создаем сетку значений для выбранной переменной
            x = np.linspace(x_min, x_max, 100)

            # Вычисляем значения функции
            params = sp.symbols(param_names)
            expr = sp.sympify(expr_input)
            func = sp.lambdify(params, expr)

            # Фиксируем остальные переменные
            Z = np.zeros_like(x)
            for i in range(x.shape[0]):
                x0[idx] = x[i]  # Первая (и единственная) выделенная переменная
                Z[i] = func(*x0)  # Вычисляем значение функции

            # Построение 2D-графика
            plt.plot(x, Z)
            plt.title(f"График функции f({', '.join(param_names)}) = {expr_input}")
            plt.xlabel(var)
            plt.ylabel("f(x)")
            plt.grid(True)
            plt.show()

        # Если выбраны две переменные, строим 3D-график
        elif len(selected_indices) == 2:
            # Определяем две выделенные переменные и их индексы
            var1, var2 = [param_names[i] for i in selected_indices]
            idx1, idx2 = selected_indices

            # Получаем значения начальной точки
            x0 = [float(entry.get()) for entry in initial_entries]

            # Устанавливаем диапазон для графика
            x_min = float(min_entry.get())
            x_max = float(max_entry.get())
            y_min = x_min
            y_max = x_max

            # Создаем сетку значений для выделенных переменных
            x = np.linspace(x_min, x_max, 100)
            y = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x, y)

            # Вычисляем значения функции
            params = sp.symbols(param_names)
            expr = sp.sympify(expr_input)
            func = sp.lambdify(params, expr)

            # Фиксируем остальные переменные
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    x0[idx1] = X[i, j]  # Первая выделенная переменная
                    x0[idx2] = Y[i, j]  # Вторая выделенная переменная
                    Z[i, j] = func(*x0)  # Вычисляем значение функции

            # Построение 3D-графика
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='k', alpha=0.8)

            # Формируем название графика
            ax.set_title(f"График функции f({', '.join(param_names)}) = {expr_input}")

            # Настройки осей
            ax.set_xlabel(var1)
            ax.set_ylabel(var2)
            ax.set_zlabel("f(x, y)")

            # Добавляем цветовую шкалу
            fig.colorbar(surf, shrink=0.5, aspect=5)

            plt.show()

        else:
            messagebox.showerror("Ошибка", "Для построения графика выберите ровно одну или две переменные.")
            return

    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка при построении графика: {e}")
        raise e


def optimize():
    try:
        expr_input = function_entry.get()
        param_names = sorted(set(sp.sympify(expr_input).free_symbols), key=str)
        param_names = [str(p) for p in param_names]

        x0 = [float(entry.get()) for entry in initial_entries]

        method = method_combobox.get()
        tol = float(tol_entry.get())
        max_iter = float(max_iter_entry.get())

        params = sp.symbols(param_names)
        expr = sp.sympify(expr_input)
        func = sp.lambdify(params, expr)

        if method == "Хука-Дживса":
            optimal_args, optimal_value, k = hooke_jeeves(func, np.array(x0), tol=tol, max_iter=max_iter)
        elif method == "Нелдера-Мида":
            optimal_args, optimal_value, k = nelder_mead(func, np.array(x0), tol=tol, max_iter=max_iter)
        elif method == "Пауэлла":
            optimal_args, optimal_value, k = powell(func, np.array(x0), tol=tol, max_iter=max_iter)
        else:
            raise ValueError("Не выбран метод оптимизации")

        for i, entry in enumerate(optimal_entries):
            entry.delete(0, tk.END)
            entry.insert(0, format_number(optimal_args[i]))
        result_entry.delete(0, tk.END)
        result_entry.insert(0, format_number(optimal_value))

        # Обновляем поле для вывода k
        k_entry.delete(0, tk.END)
        k_entry.insert(0, str(k))

    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка при поиске: {e}")
        raise e


def enable_copy_paste(entry_widget):
    """Добавляет обработчики для Ctrl+C и Ctrl+X."""
    entry_widget.bind("<Control-c>", lambda e: entry_widget.event_generate("<<Copy>>"))
    entry_widget.bind("<Control-x>", lambda e: entry_widget.event_generate("<<Cut>>"))


root = tk.Tk()
root.title("Оптимизация функции")
root.geometry("410x800")

# Ввод (заголовок)
tk.Label(root, text="Ввод", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=5, padx=5, pady=5, sticky="w")

# Выражение функции
function_label = tk.Label(root, text="Выражение функции")
function_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
function_entry = tk.Entry(root)
function_entry.insert(0, "(x-2)**2+(y-3)**2")
function_entry.grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky="ew")
function_entry.bind("<Return>", update_variables)  # Обновление при нажатии Enter
enable_copy_paste(function_entry)  # Добавляем поддержку Ctrl+C и Ctrl+X

# Кнопка "Ввод" рядом с выражением функции
tk.Button(root, text="Ввод", command=update_variables).grid(row=1, column=4, padx=5, pady=5, sticky="ew")

# Начальная точка
tk.Label(root, text="Начальная точка").grid(row=2, column=0, padx=5, pady=5, sticky="w")
variable_frame = tk.Frame(root)
variable_frame.grid(row=3, column=0, columnspan=5, padx=5, pady=5, sticky="ew")

# Инпуты для min и max, и кнопка "График"
tk.Label(root, text="min").grid(row=4, column=0, padx=5, pady=5, sticky="e")  # Выравнивание по левому краю
min_entry = tk.Entry(root, width=10)
min_entry.insert(0, "-100")  # Значение по умолчанию для min
min_entry.grid(row=4, column=1, padx=5, pady=5, sticky="ew")  # Растягиваем по горизонтали

tk.Label(root, text="max").grid(row=4, column=2, padx=5, pady=5, sticky="w")  # Выравнивание по левому краю
max_entry = tk.Entry(root, width=10)
max_entry.insert(0, "100")  # Значение по умолчанию для max
max_entry.grid(row=4, column=3, padx=5, pady=5, sticky="ew")  # Растягиваем по горизонтали

# Кнопка "График" в одной строке с min и max
tk.Button(root, text="График", command=show_graph).grid(
    row=4, column=4, padx=5, pady=5, sticky="ew")  # Выравнивание по левому краю

# Метод (заголовок)
tk.Label(root, text="Метод", font=("Arial", 12, "bold")).grid(
    row=5, column=0, columnspan=5, padx=5, pady=5, sticky="w")

# Метод и кнопка "Найти" в одной строке
tk.Label(root, text="Тип").grid(row=6, column=0, padx=5, pady=5, sticky="w")
method_combobox = ttk.Combobox(root, values=["Хука-Дживса", "Нелдера-Мида", "Пауэлла"], state="readonly")
method_combobox.set("Хука-Дживса")
method_combobox.grid(row=6, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

# Кнопка "Найти" справа от выбора метода
tk.Button(root, text="Найти", command=optimize).grid(row=6, column=4, padx=5, pady=5, sticky="ew")

# Параметры метода
tk.Label(root, text="Параметры метода").grid(row=7, column=0, columnspan=5, padx=5, pady=5, sticky="w")

tk.Label(root, text="Критейрий точности (ε)").grid(row=8, column=0, padx=5, pady=5, sticky="w")
tol_entry = tk.Entry(root, width=10)
tol_entry.insert(0, "1e-6")  # Значение по умолчанию для tol
tol_entry.grid(row=8, column=1, padx=5, pady=5, sticky="ew")

tk.Label(root, text="Максимальное кол-во (k)").grid(row=9, column=0, padx=5, pady=5, sticky="w")
max_iter_entry = tk.Entry(root, width=10)
max_iter_entry.insert(0, "1000")  # Значение по умолчанию для max_iter
max_iter_entry.grid(row=9, column=1, padx=5, pady=5, sticky="ew")

# Вывод (заголовок)
tk.Label(root, text="Вывод", font=("Arial", 12, "bold")).grid(
    row=10, column=0, columnspan=5, padx=5, pady=5, sticky="w")

# Результат
tk.Label(root, text="Значение функции").grid(row=11, column=0, padx=5, pady=5, sticky="w")
result_entry = tk.Entry(root, width=30)
result_entry.grid(row=11, column=1, columnspan=4, padx=5, pady=5, sticky="ew")

# Внесение изменений в интерфейс для добавления поля k
tk.Label(root, text="Количество итераций (k)").grid(row=12, column=0, padx=5, pady=5, sticky="w")
k_entry = tk.Entry(root, width=10)
k_entry.grid(row=12, column=1, padx=5, pady=5, sticky="ew")  # Растягиваем по горизонтали

# Оптимальная точка
tk.Label(root, text="Оптимальная точка").grid(row=13, column=0, padx=5, pady=5, sticky="w")
optimal_frame = tk.Frame(root)
optimal_frame.grid(row=14, column=0, columnspan=5, padx=5, pady=5, sticky="ew")

update_variables()

root.mainloop()
