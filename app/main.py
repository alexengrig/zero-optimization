import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
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

        for i, var in enumerate(param_names):
            tk.Label(variable_frame, text=f"{var}").grid(row=i, column=0, padx=5, pady=5)
            initial_entry = tk.Entry(variable_frame, width=10)
            initial_entry.insert(0, "0")
            initial_entry.grid(row=i, column=1, padx=5, pady=5)
            initial_entries.append(initial_entry)

        tk.Label(variable_frame, text="min").grid(row=len(param_names), column=0, padx=5, pady=5)
        min_entry = tk.Entry(variable_frame, width=10)
        min_entry.insert(0, "-100")
        min_entry.grid(row=len(param_names), column=1, padx=5, pady=5)

        tk.Label(variable_frame, text="max").grid(row=len(param_names), column=2, padx=5, pady=5)
        max_entry = tk.Entry(variable_frame, width=10)
        max_entry.insert(0, "100")
        max_entry.grid(row=len(param_names), column=3, padx=5, pady=5)

        global range_entries
        range_entries = [min_entry, max_entry]


        update_optimal_fields(param_names)

    except Exception as e:
        pass


def update_optimal_fields(param_names):
    for widget in optimal_frame.winfo_children():
        widget.destroy()

    global optimal_entries
    optimal_entries = []

    for i, var in enumerate(param_names):
        tk.Label(optimal_frame, text=f"Оптимальное {var}").grid(row=i, column=0, padx=5, pady=5)
        optimal_entry = tk.Entry(optimal_frame, width=10)
        optimal_entry.grid(row=i, column=1, padx=5, pady=5)
        optimal_entries.append(optimal_entry)


def optimize():
    try:
        expr_input = function_entry.get()
        param_names = sorted(set(sp.sympify(expr_input).free_symbols), key=str)
        param_names = [str(p) for p in param_names]

        x0 = [float(entry.get()) for entry in initial_entries]

        min_val = float(range_entries[0].get())
        max_val = float(range_entries[1].get())

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
root.geometry("600x800")


tk.Label(root, text="Функция").grid(row=0, column=0, padx=5, pady=5, sticky="w")
function_entry = tk.Entry(root, width=40)
function_entry.insert(0, "x + y + z")
function_entry.grid(row=0, column=1, columnspan=3, padx=5, pady=5)
function_entry.bind("<KeyRelease>", update_variables)


variable_frame = tk.Frame(root)
variable_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5)


tk.Label(root, text="alpha").grid(row=2, column=0, padx=5, pady=5)
alpha_entry = tk.Entry(root, width=10)
alpha_entry.insert(0, "1.0")
alpha_entry.grid(row=2, column=1, padx=5, pady=5)

tk.Label(root, text="beta").grid(row=3, column=0, padx=5, pady=5)
beta_entry = tk.Entry(root, width=10)
beta_entry.insert(0, "0.5")
beta_entry.grid(row=3, column=1, padx=5, pady=5)


tk.Label(root, text="Метод").grid(row=4, column=0, padx=5, pady=5)
method_combobox = ttk.Combobox(root, values=["Хука-Дживса", "Нелдера-Мида", "Пауэлла"], state="readonly")
method_combobox.set("Хука-Дживса")
method_combobox.grid(row=4, column=1, padx=5, pady=5)


optimal_frame = tk.Frame(root)
optimal_frame.grid(row=5, column=0, columnspan=4, padx=5, pady=5)


tk.Button(root, text="Найти", command=optimize).grid(row=6, column=0, columnspan=4, pady=10)


tk.Label(root, text="Значение функции").grid(row=7, column=0, padx=5, pady=5)
result_entry = tk.Entry(root, width=30)
result_entry.grid(row=7, column=1, columnspan=3, padx=5, pady=5)


update_variables()
root.mainloop()
