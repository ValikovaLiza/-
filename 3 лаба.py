import numpy as np
from scipy.optimize import minimize

# Определяем функцию для оптимизации и ограничения
def objective(x):
    return x[0]**2 + x[1]**2  # квадратичная функция

def constraint_eq(x):
    return x[0] + x[1] - 1     # Условие равенства, которое должно быть удовлетворено


def lagrangian(x, λ):
    return objective(x) - λ * constraint_eq(x)

# Оптимизация
def optimize():
    # Начальное приближение для x и λ
    x0 = [0.5, 0.5]  # Начальные значения x
    λ0 = 1  # Начальное значение λ

    # Оптимизация с использованием метода множителей Лагранжа
    result = minimize(lambda x: lagrangian(x, λ0), x0, constraints={'type': 'eq', 'fun': constraint_eq})
    
    return result




# Метод Зойтендейка
def penalty_function(x):
    """Штрафная функция для учёта ограничения
    штраф пропорционален степени нарушения ограничений: 
    чем больше нарушение, тем больше штраф."""
    return np.abs(x[0] + x[1] - 1) ** 2

def zoiteindijk_method(objective_function, x0, penalty_coefficient, tolerance=1e-6, max_iterations=1000):
    x = x0
    k = penalty_coefficient
    penalty = penalty_function(x)

    for _ in range(max_iterations):
        # Комбинированная функция для оптимизации
        combined_function = lambda x: objective_function(x) + k * penalty_function(x)

        # Минимизация комбинированной функции
        result = minimize(combined_function, x, method='Nelder-Mead')

        # Обновление текущего приближения
        x = result.x

        # Пересчет значения штрафной функции
        penalty = penalty_function(x)

        # Проверка условия сходимости
        if penalty < tolerance:
            break

        # Увеличение коэффициента штрафа
        k *= 10

    return x


def projection_function(x):
    return np.array([x[0], 1 - x[0]])

# Метод проекции градиента Розенброка
def rosen_gradient_projection(x0, learning_rate=0.1, num_iterations=100):
    x = np.copy(x0)  # Начальное значение x
    for _ in range(num_iterations):
        gradient = 2 * x  # Градиент целевой функции
        x = x - learning_rate * gradient  # Шаг градиентного спуска
        x = projection_function(x)  # Проекция на допустимое множество
    return x


# Начальное приближение для x и lambda
x0 = np.array([0.5, 0.5])

# Метод множителей Лагранжа
result = optimize()
print("Метод множителей Лагранжа:")
print("Значение целевой функции в решении:", result.fun)

# Метод Зойтендейка
penalty_coefficient = 1
solution1 = zoiteindijk_method(objective, x0, penalty_coefficient)
print("\nМетод Зойтендейка:")
print("Значение целевой функции в решении:", objective(solution1))

# Метод проекции градиента Розена
min_point = rosen_gradient_projection(x0)
print("\nМетод Розенброка:")
print("Значение функции в оптимальной точке:", objective(min_point))
