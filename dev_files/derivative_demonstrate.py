import time
import math


def test(x1, d):
    # pontuacao
    fx = lambda x: (x - 4) ** 2 - 3
    # derivada da mudança entre x2 de x1
    dx = lambda x1, x2, y1, y2: (y2 - y1) / (x2 - x1)

    # chute inicial
    # d = 0.01
    # valor inicial
    # x1 = 1
    # valor inicial + chute
    x2 = 1 + d

    iter = 0
    while True:
        # f1
        f1 = fx(x1)
        # f2
        f2 = fx(x2)
        # derivada
        df = dx(x1, x2, f1, f2)
        # proximo valor
        x3 = x2 - f2 / df
        # troca os valores
        x1, x2 = x2, x3

        if math.fabs(x1 - x2) < d:
            return x2, iter
        iter += 1


tests = [
    [0.01, 1],
    [0.02, 0.1],
    [0.05, 0.003],
    [0.0001, 0.003],
    [0.0001, -50],
    [0.1, -50],
    [0.05, -50],
    [0.09, -50],
]

for dtest in tests:
    print(f"testing x1 = {dtest[1]} and d = {dtest[0]}")
    v, i = test(dtest[1], dtest[0])
    print(f"got result: {v} in {i} iterations\n")
