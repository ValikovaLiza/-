#Метод Штрафных Функций
import numpy as np

# Определение функции цели
def objective_function(x):
    return 4 * x[0] - x[1]**2 - 12

# Определение ограничений
def constraint1(x):
    return 10 * x[0] - x[0]**2 + 10 * x[1] - x[1]**2 - 34

# Определение штрафных функций
def penalty_function(x, t):
    return t * max(0, constraint1(x))**2

# Определение градиента целевой функции
def gradient_objective(x):
    dfdx0 = 4
    dfdx1 = -2 * x[1]
    return np.array([dfdx0, dfdx1])

# Определение градиента ограничения
def gradient_constraint(x):
    dfdx0 = 10 - 2 * x[0]
    dfdx1 = 10 - 2 * x[1]
    return np.array([dfdx0, dfdx1])

# Метод штрафных функций
def penalty_method(x0, eps, t0, mu, max_iter=1000):
    x = np.array(x0)
    t = t0
    for _ in range(max_iter):
        grad_obj = gradient_objective(x)
        grad_con = gradient_constraint(x)
        grad_penalty = grad_obj + mu * grad_con  # Градиент штрафной функции
        if np.linalg.norm(grad_penalty) < eps:
            break
        x = x - grad_penalty / np.linalg.norm(grad_penalty)  # Градиентный спуск
        t *= mu  # Увеличиваем коэффициент штрафа
    return x

# Начальное приближение
x0 = [2, 4]
# Точность
eps = 0.05
# Начальное значение коэффициента штрафа
t0 = 1
# Множитель увеличения коэффициента штрафа
mu = 2

# Запуск метода штрафных функций
result = penalty_method(x0, eps, t0, mu)
print("Значение функции цели в решении:", objective_function(result))
