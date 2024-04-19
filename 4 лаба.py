import numpy as np

def penalty_function_method(f, grad_f, constraints, grad_constraints, x0, penalty_func, eps=0.05, max_iter=1000):
    """
    Метод Штрафных Функций для задачи оптимизации с ограничениями
    
    Параметры:
    f : функция
        Функция для минимизации
    grad_f : функция
        Градиент функции для минимизации
    constraints : список функций
        Список функций-ограничений
    grad_constraints : список функций
        Список градиентов функций-ограничений
    x0 : numpy array
        Начальное приближение
    penalty_func : функция
        Штрафная функция
    eps : float, опционально
        Погрешность, по умолчанию 0.05
    max_iter : int, опционально
        Максимальное количество итераций, по умолчанию 1000
    
    Возвращает:
    x_opt : numpy array
        Оптимальное решение
    """
    
    x = x0
    penalty = 1
    
    for _ in range(max_iter):
        # Функция для минимизации с штрафной функцией
        def penalty_opt(x):
            return f(x) + penalty * penalty_func(x, constraints)
        
        # Градиент функции для минимизации с штрафной функцией
        def grad_penalty_opt(x):
            return grad_f(x) + penalty * sum(grad_penalty(x) for grad_penalty in grad_constraints)
        
        # Обновление весов с помощью градиентного спуска
        x_new = x - 0.01 * grad_penalty_opt(x)
        
        # Проверка на соответствие ограничениям
        if all(g(x_new) <= 0 for g in constraints):
            # Проверка на сходимость
            if np.linalg.norm(x_new - x) < eps:
                return x_new
            x = x_new
        else:
            penalty *= 10
    
    return x

# Функция для минимизации
def f(x):
    return 4 * x[0] - x[1]**2 - 12

# Градиент функции для минимизации
def grad_f(x):
    return np.array([4, -2 * x[1]])

# Ограничения
def constraint1(x):
    return 10.0 * x[0] - x[0]**2 + 10.0 * x[1] - x[1]**2 - 34.0

def constraint2(x):
    return -x[0]

def constraint3(x):
    return -x[1]

# Градиенты ограничений
def grad_constraint1(x):
    return np.array([10 - 2 * x[0], 10 - 2 * x[1]])

def grad_constraint2(x):
    return np.array([-1, 0])

def grad_constraint3(x):
    return np.array([0, -1])

# Штрафная функция
def penalty_func(x, constraints):
    return sum(max(0, g(x))**2 for g in constraints)

# Начальное приближение
x0 = np.array([2, 4])

# Вызов метода штрафных функций
x_opt = penalty_function_method(f, grad_f, [constraint1, constraint2, constraint3],
                                [grad_constraint1, grad_constraint2, grad_constraint3],
                                x0, penalty_func, eps=0.05)

print("Optimal solution:", f(x_opt))
