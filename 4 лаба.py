#Метод Барьерных Функций
from scipy.optimize import minimize

# Целевая функция
def objective_function(x):
    return 4*x[0] - x[1]**2 - 12

# Ограничения
def constraint1(x):
    return x[0]

def constraint2(x):
    return x[1]

def constraint3(x):
    return 10*x[0] - x[0]**2 + 10*x[1] - x[1]**2 - 34

# Начальное приближение
x0 = [2, 4]

# Эпсилон
eps = 0.05

# Функция штрафов
def penalty_function(x):
    penalty = max(0, -constraint1(x)) + max(0, -constraint2(x)) + max(0, -constraint3(x))
    return penalty

# Общая функция (целевая + штрафы)
def total_function(x):
    return objective_function(x) + penalty_function(x)

# Минимизация с использованием метода штрафных функций
result = minimize(total_function, x0, method='Nelder-Mead', tol=eps)

# Вывод результатов
print("Минимальное значение функции:", result.fun)
