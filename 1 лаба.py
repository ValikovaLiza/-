import math

a = -2
b = 1
L=abs(b-a+1)
eps=1e-5

def func (x):
    return x**2 + 2*x - 4

def dehito(a, b, eps, L):
    while b-a>2*eps:
        mid=(a+b)/2
        x1 = mid - eps/2
        x2 = mid + eps/2
        res1 = func(x1)
        res2 = func(x2)

        if res1 > res2: 
            a=x1
        else: 
            b=x2
    return (func((a+b)/2))

print("Метод дихотомии: ", dehito(a,b,eps,L))
# -4.999999999999492


def fibon(a, b, L, eps):
    n = abs(round(math.log((b - a) / eps, math.e) / math.log((math.sqrt(5)-1)/2, math.e)))
    while n>2:
        F = [None]*n
        F[0] = 1
        F[1] = 1
        for i in range(2,n):
            F[i] = F[i-1] + F[i-2]
        x1 = a + (F[n-2]/F[n-1])*L
        x2 = b - (F[n-2]/F[n-1])*L

        res1 = func(x1)
        res2 = func(x2)

        if res1 > res2: 
            b=x1
            res1=res2
            x1=x2
            L=b-a
            x2 = b - (F[n-2]/F[n-1])*L
            res2 = func(x2)
        else: 
            a=x2
            res2=res1
            x2=x1
            L=b-a
            x1 = a + (F[n-2]/F[n-1])*L
            res1 = func(x1)
        n-=1
    return min(res1, res2)
print("Метод Фибоначчи: ", fibon(a,b,L,eps))
# -4.9999999999321405



def sechen(a, b, L):
    #t = (1+ math.sqrt(5))/2
    t = 0.618
    while L>eps:
        x1 = a + L*t
        x2 = b - L*t

        res1 = func(x1)
        res2 = func(x2)

        if res1 > res2: 
            b=x1
            res1=res2
            x1=x2
            L=b-a
            x2 = b - L*t
            res2 = func(x2)
        else: 
            a=x2
            res2=res1
            x2=x1
            L=b-a
            x1 = a + L*t
            res1 = func(x1)
    return min(res1, res2)
print("Метод золотого сечения: ", sechen(a,b,L))
# -4.999999999999424