import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import sympy as sp
from datetime import datetime

from app.hooke_jeeves import hooke_jeeves
from app.nelder_mead import nelder_mead
from app.powell import powell
from app.utils import format_number

history_data = []  # Список для хранения истории поиска

checkbuttons = []  # Список для переменных BooleanVar
checkbuttons_widgets = []  # Список для чекбоксов

step_size_entry = None
step_reduction_entry = None
alpha_entry = None
beta_entry = None
gamma_entry = None


def update_variables(*args):
    # Вызываем update_method_parameters при изменении метода
    update_method_params()
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


def update_method_params(*args):
    global step_size_entry, step_reduction_entry, alpha_entry, beta_entry, gamma_entry
    # Сначала очищаем старые параметры
    for widget in method_param_frame.winfo_children():
        widget.grid_forget()

    method = method_combobox.get()

    if method == "Хука-Дживса":
        # Добавляем параметры для метода Хука-Дживса
        tk.Label(method_param_frame, text="Начальный шаг").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        step_size_entry = tk.Entry(method_param_frame, width=10)
        step_size_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        step_size_entry.insert(0, "0.5")  # Значение по умолчанию

        tk.Label(method_param_frame, text="Коэффициент уменьшения шага").grid(
            row=1, column=0, padx=5, pady=5, sticky="w")
        step_reduction_entry = tk.Entry(method_param_frame, width=10)
        step_reduction_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        step_reduction_entry.insert(0, "0.5")  # Значение по умолчанию

    elif method == "Нелдера-Мида":
        # Добавляем параметры для метода Нелдера-Мида
        tk.Label(method_param_frame, text="Коэффициент отражения (α)").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        alpha_entry = tk.Entry(method_param_frame, width=10)
        alpha_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        alpha_entry.insert(0, "1")  # Значение по умолчанию

        tk.Label(method_param_frame, text="Коэффициент сжатия (β)").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        beta_entry = tk.Entry(method_param_frame, width=10)
        beta_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        beta_entry.insert(0, "0.5")  # Значение по умолчанию

        tk.Label(method_param_frame, text="Коэффициент растяжения (γ)").grid(row=2, column=0, padx=5, pady=5,
                                                                             sticky="w")
        gamma_entry = tk.Entry(method_param_frame, width=10)
        gamma_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        gamma_entry.insert(0, "2")  # Значение по умолчанию

    elif method == "Пауэлла":
        # Для метода Пауэлла параметры не меняются, можно оставить пустым или убрать
        pass  # Или добавьте здесь другие параметры, если нужно

    # Важно обновить интерфейс
    method_param_frame.grid(row=10, column=0, columnspan=5, padx=5, pady=5, sticky="w")


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

        # Инициализируем параметры для записи в историю
        method_params = {}

        if method == "Хука-Дживса":
            step_size = float(step_size_entry.get())
            step_reduction = float(step_reduction_entry.get())
            method_params = {
                "step_size": step_size,
                "step_reduction": step_reduction
            }
            optimal_args, optimal_value, k = hooke_jeeves(func, np.array(x0), tol=tol, max_iter=max_iter,
                                                          step_size=step_size, step_reduction=step_reduction)
        elif method == "Нелдера-Мида":
            alpha = float(alpha_entry.get())
            beta = float(beta_entry.get())
            gamma = float(gamma_entry.get())
            method_params = {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma
            }
            optimal_args, optimal_value, k = nelder_mead(func, np.array(x0), tol=tol, max_iter=max_iter, alpha=alpha,
                                                         beta=beta, gamma=gamma)
        elif method == "Пауэлла":
            method_params = {}  # Для метода Пауэлла параметры не меняются, если нет других
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

        # Добавление записи в историю
        history_data.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "function": expr_input,
            "initial_point": "\n".join([entry.get() for entry in initial_entries]),
            "method": method,
            "parameters": f"ε = {tol}\nmax k = {max_iter}\n" + "\n".join(
                [f"{key} = {value}" for key, value in method_params.items()]),
            "iterations": k,
            "function_value": format_number(optimal_value),
            "optimal_point": "\n".join(
                [f"{param}={format_number(val)}" for param, val in zip(param_names, optimal_args)]),
        })

    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка при поиске: {e}")
        raise e


def show_history():
    # Создаем новое окно для истории
    history_window = tk.Toplevel(root)
    history_window.title("История поиска")
    history_window.geometry("600x400")

    # Настройка стиля для Treeview
    s = ttk.Style()
    s.configure('Treeview', rowheight=100)  # Устанавливаем высоту строк

    # Создаем таблицу для отображения истории
    tree = ttk.Treeview(history_window, columns=(
        "Date", "Function", "Initial Point", "Method", "Parameters", "Iterations", "Function Value", "Optimal Point"),
                        show="headings")

    # Настройка заголовков столбцов
    tree.heading("Date", text="Дата")
    tree.heading("Function", text="Функция")
    tree.heading("Initial Point", text="Начальная точка")
    tree.heading("Method", text="Метод")
    tree.heading("Parameters", text="Параметры")
    tree.heading("Iterations", text="Итерации")
    tree.heading("Function Value", text="Значение функции")
    tree.heading("Optimal Point", text="Оптимальная точка")

    # Настройка колонок для автоматической подгонки ширины
    for col in tree["columns"]:
        tree.column(col, width=100, anchor="w")

    # Добавляем данные в таблицу
    for record in history_data:
        values = (
            record["date"], record["function"], record["initial_point"], record["method"], record["parameters"],
            record["iterations"], record["function_value"], record["optimal_point"]
        )
        # Вставляем строку в таблицу
        tree.insert("", "end", values=values)

    tree.pack(fill="both", expand=True)


def enable_copy_paste(entry_widget):
    """Добавляет обработчики для Ctrl+C и Ctrl+X."""
    entry_widget.bind("<Control-c>", lambda e: entry_widget.event_generate("<<Copy>>"))
    entry_widget.bind("<Control-x>", lambda e: entry_widget.event_generate("<<Cut>>"))


root = tk.Tk()
root.title("Оптимизация функции")
root.geometry("410x800")

# Меню
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

history_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="История", command=show_history)

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
method_combobox.bind("<<ComboboxSelected>>", update_method_params)

# Кнопка "Найти" справа от выбора метода
tk.Button(root, text="Найти", command=optimize).grid(row=6, column=4, padx=5, pady=5, sticky="ew")

# Параметры метода
tk.Label(root, text="Параметры метода").grid(row=7, column=0, columnspan=5, padx=5, pady=5, sticky="w")

tk.Label(root, text="Критерий точности (ε)").grid(row=8, column=0, padx=5, pady=5, sticky="w")
tol_entry = tk.Entry(root, width=10)
tol_entry.insert(0, "1e-6")  # Значение по умолчанию для tol
tol_entry.grid(row=8, column=1, padx=5, pady=5, sticky="ew")

tk.Label(root, text="Максимальное кол-во (k)").grid(row=9, column=0, padx=5, pady=5, sticky="w")
max_iter_entry = tk.Entry(root, width=10)
max_iter_entry.insert(0, "1000")  # Значение по умолчанию для max_iter
max_iter_entry.grid(row=9, column=1, padx=5, pady=5, sticky="ew")

# Рамка для параметров метода
method_param_frame = tk.Frame(root)
method_param_frame.grid(row=10, column=0, columnspan=5, padx=5, pady=5, sticky="w")

# Вывод (заголовок)
tk.Label(root, text="Вывод", font=("Arial", 12, "bold")).grid(
    row=11, column=0, columnspan=5, padx=5, pady=5, sticky="w")

# Результат
tk.Label(root, text="Значение функции").grid(row=12, column=0, padx=5, pady=5, sticky="w")
result_entry = tk.Entry(root, width=30)
result_entry.grid(row=12, column=1, columnspan=4, padx=5, pady=5, sticky="ew")

# Внесение изменений в интерфейс для добавления поля k
tk.Label(root, text="Количество итераций (k)").grid(row=13, column=0, padx=5, pady=5, sticky="w")
k_entry = tk.Entry(root, width=10)
k_entry.grid(row=13, column=1, padx=5, pady=5, sticky="ew")  # Растягиваем по горизонтали

# Оптимальная точка
tk.Label(root, text="Оптимальная точка").grid(row=14, column=0, padx=5, pady=5, sticky="w")
optimal_frame = tk.Frame(root)
optimal_frame.grid(row=15, column=0, columnspan=5, padx=5, pady=5, sticky="ew")

update_variables()

root.mainloop()
