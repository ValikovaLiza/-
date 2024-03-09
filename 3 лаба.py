import numpy as np

from scipy.optimize import minimize

# Определяем функцию для оптимизации и ограничения
def objective(x):
    return x[0]**2 + x[1]**2  # Пример целевой функции (квадратичная функция)

def constraint_eq(x):
    return x[0] + x[1] - 1     # Условие равенства, которое должно быть удовлетворено

# Функция для вычисления градиента целевой функции
def gradient_objective_function(x):
    return np.array([2 * x[0], -2 * x[1]])

# Функция для вычисления градиента ограничения
def gradient_constraint_equation(x):
    return np.array([1, 1])

# Метод градиентного спуска
def gradient_descent(objective_func, constraint_eq, grad_objective, grad_constraint, x0, learning_rate=0.01, max_iter=1000, tol=1e-6):
    x = np.array(x0)

    for i in range(max_iter):
        # Вычисление градиента функции Лагранжа
        grad_lagrangian = grad_objective(x) + constraint_eq(x) * grad_constraint(x)

        # Обновление переменных по правилу градиентного спуска
        x = x - learning_rate * grad_lagrangian

        # Проверка условия сходимости
        if np.linalg.norm(grad_lagrangian) < tol:
            break

    return x

# Начальное приближение
initial_guess = [0.5, 0.5]

# Решение задачи оптимизации с ограничениями
optimal_solution = gradient_descent(objective, constraint_eq,
                                    gradient_objective_function, gradient_constraint_equation,
                                    initial_guess)

# Вывод результатов
print("Оптимальные переменные:", optimal_solution)
print("Значение целевой функции в оптимальной точке:", objective(optimal_solution))






def projection_gradient_method(objective_func, constraint_eq, x0, max_iter=100, tol=1e-6):
    x = np.array(x0)
    
    for i in range(max_iter):
        gradient = np.array([4 * x[0], -2 * x[1]])  # Градиент целевой функции

        # Множители Лагранжа
        lagrange_multipliers = np.array([-1])

        # Градиент функции Лагранжа
        grad_lagrangian = gradient - lagrange_multipliers * np.array([constraint_eq(x)])

        # Проекция градиента
        x = x - grad_lagrangian

        # Проверка условия сходимости
        if np.linalg.norm(grad_lagrangian) < tol:
            break

    return x

# Начальное приближение
initial_guess = [0.5, 0.5]

# Решение задачи оптимизации с ограничениями
optimal_solution = projection_gradient_method(objective, constraint_eq, initial_guess)

# Вывод результатов
print("Оптимальные переменные:", optimal_solution)
print("Значение целевой функции в оптимальной точке:", objective(optimal_solution))






def projection_gradient_rosen_method(objective_func, constraint_eq, x0, step_size=0.01, max_iter=1000, tol=1e-6):
    x = np.array(x0)
    
    for i in range(max_iter):
        gradient = np.array([4 * x[0], -2 * x[1]])  # Градиент целевой функции

        # Множители Лагранжа
        lagrange_multipliers = np.array([-1])

        # Градиент функции Лагранжа
        grad_lagrangian = gradient - lagrange_multipliers * np.array([constraint_eq(x)])

        # Проекция градиента
        x = x - step_size * grad_lagrangian

        # Проверка условия сходимости
        if np.linalg.norm(grad_lagrangian) < tol:
            break

    return x

# Начальное приближение
initial_guess = [0.5, 0.5]

# Решение задачи оптимизации с ограничениями
optimal_solution = projection_gradient_rosen_method(objective, constraint_eq, initial_guess)

# Вывод результатов
print("Оптимальные переменные:", optimal_solution)
print("Значение целевой функции в оптимальной точке:", objective(optimal_solution))

