# -*- coding:utf8 -*-
import numpy as np
import matplotlib.pyplot as plt


def cost_function(input_X, _y, theta):
    """
    cost function
    :param input_X: np.matrix input X
    :param _y: np.array y
    :param theta: np.matrix theta
    :return: float
    """
    rows, cols = input_X.shape
    predictions = input_X * theta
    sqrErrors = np.array(predictions - _y) ** 2
    J = 1.0 / (2 * rows) * sqrErrors.sum()

    return J


def gradient_descent(input_X, _y, theta, learning_rate=0.1,
                     iterate_times=3000):
    """
    gradient descent
    :param input_X: np.matrix input X
    :param _y: np.array y
    :param theta: np.matrix theta
    :param learning_rate: float learning rate
    :param iterate_times: int max iteration times
    :return: tuple
    """
    convergence = 0
    rows, cols = input_X.shape
    Js = []

    for i in range(iterate_times):
        errors = input_X * theta - _y
        delta = 1.0 / rows * (errors.transpose() * input_X).transpose()
        theta -= learning_rate * delta
        Js.append(cost_function(input_X, _y, theta))

    return theta, Js


def generate_data():
    """
    generate training data y = 2*x^2 + 4*x + 2
    """
    x = np.linspace(0, 2, 50)
    X = np.matrix([np.ones(50), x, x**2]).T
    y = 2 * X[:, 0] - 4 * X[:, 1] + 2 * X[:, 2] + np.mat(np.random.randn(50)).T / 25
    np.savetxt('linear_regression_using_gradient_descent.csv',
               np.column_stack((X, y)), delimiter=',')


def test():
    """
    main
    :return: None
    """
    m = np.loadtxt('linear_regression_using_gradient_descent.csv', delimiter=',')
    input_X, y = np.asmatrix(m[:, :-1]), np.asmatrix(m[:, -1]).T
    # theta 的初始值必须是 float
    theta = np.matrix([[0.0], [0.0], [0.0]])
    final_theta, Js = gradient_descent(input_X, y, theta)

    t1, t2, t3 = np.array(final_theta).reshape(-1,).tolist()
    print('对测试数据 y = 2 - 4x + 2x^2 求得的参数为: %.3f, %.3f, %.3f\n' % (t1, t2, t3))

    plt.figure('theta')
    predictions = np.array(input_X * final_theta).reshape(-1,).tolist()
    x1 = np.array(input_X[:, 1]).reshape(-1,).tolist()
    y1 = np.array(y).reshape(-1,).tolist()
    plt.plot(x1, y1, '*')
    plt.plot(x1, predictions)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y = 2 - 4x + 2x^2')

    plt.figure('cost')
    x2 = range(1, len(Js) + 1)
    y2 = Js
    plt.plot(x2, y2)
    plt.xlabel('iterate times')
    plt.ylabel('value')
    plt.title('cost function')

    plt.show()


if __name__ == '__main__':
    test()
