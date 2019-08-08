import numpy as np
import matplotlib.pyplot as plt

class Regression:
    def __init__(self):
        pass

    def find_sum(l, p):
        res = 0

        for i in l:
            res += i**p

        return res

    def find_mul_sum(l1, l2):
        res = 0

        for i in range(len(l1)):
            res += (l1[i]*l2[i])

        return res

    def solve_equ(sum_x, sum_x2, sum_y, sum_xy):
        # Equation no 1
        # Ey = a * Ex + b * n

        # Equation no 2
        # Exy = a * Ex^2 + b * Ex

        n = 30

        p = np.array([[sum_x,n], [sum_x2,sum_x]])
        q = np.array([sum_y, sum_xy])

        res = np.linalg.solve(p, q)

        return res

    def predict(x, res):
        y_pred = []

        for i in x:
            y_pred.append(res[0] * i + res[1])

        return y_pred

def main():
    x = [1.1,1.3,1.5,2,2.2,2.9,3,3.2,3.2,3.7,3.9,4,4,4.1,4.5,4.9,5.1,5.3,5.9,6,6.8,7.1,7.9,8.2,8.7,9,9.5,9.6,10.3,10.5]

    y = [39343,46205,37731,43525,39891,56642,60150,54445,64445,57189,63218,55794,56957,57081,61111,67938,66029,83088,81363,93940,91738,98273,101302,113812,109431,105582,116969,112635,122391,121872]

    r = Regression

    sum_x = r.find_sum(x, 1)
    sum_y = r.find_sum(y, 1)
    sum_x2 = r.find_sum(x, 2)
    sum_xy = r.find_mul_sum(x, y)

    res = []

    res = r.solve_equ(sum_x, sum_x2, sum_y, sum_xy)

    y_pred = r.predict(x, res)

    plt.scatter(x, y, color = 'red')
    plt.plot(x, y_pred, color = 'blue')
    plt.title('Ownression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == "__main__":
    main()

