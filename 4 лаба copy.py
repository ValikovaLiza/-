import numpy as np

# Целевая функция
def objective_function(x):
    return 4 * x[0] - x[1]**2 - 12

# Ограничения
def constraint(x):
    return 10 * x[0] - x[0]**2 + 10 * x[1] - x[1]**2 - 34

# Функция градиента целевой функции
def gradient_objective_function(x):
    return np.array([4, -2 * x[1]])

# Функция градиента ограничений
def gradient_constraint(x):
    return np.array([10 - 2 * x[0], 10 - 2 * x[1]])

# Функция для метода барьерных функций
def barrier_method(x0, eps):
    mu = 1  # Параметр барьера
    t = 10  # Параметр для увеличения mu

    while constraint(x0) < 0:
        x = x0
        while np.linalg.norm(gradient_objective_function(x)) > eps:
            x = x - (1 / mu) * gradient_objective_function(x)
        x0 = x
        mu *= t  # Увеличиваем mu

    return x0

# Начальное приближение
x0 = np.array([2, 4])
# Точность
eps = 0.05

# Вызов метода барьерных функций
result = barrier_method(x0, eps)

print("Значение целевой функции в минимуме:", objective_function(result))

