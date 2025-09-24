import numpy as np
from sleipnir.autodiff import Jacobian, Variable, VariableMatrix


def test_y_eq_x() -> None:
    x = VariableMatrix(3)
    for i in range(3):
        x[i].set_value(i + 1)

    # y = x
    #
    #         [1  0  0]
    # dy/dx = [0  1  0]
    #         [0  0  1]
    y = x
    J = Jacobian(y, x)

    expected_J = np.diag([1.0, 1.0, 1.0])
    assert (J.get().value() == expected_J).all()
    assert (J.value().todense() == expected_J).all()


def test_y_eq_3x() -> None:
    x = VariableMatrix(3)
    for i in range(3):
        x[i].set_value(i + 1)

    # y = 3x
    #
    #         [3  0  0]
    # dy/dx = [0  3  0]
    #         [0  0  3]
    y = 3 * x
    J = Jacobian(y, x)

    expected_J = np.diag([3.0, 3.0, 3.0])
    assert (J.get().value() == expected_J).all()
    assert (J.value().todense() == expected_J).all()


def test_products() -> None:
    x = VariableMatrix(3)
    for i in range(3):
        x[i].set_value(i + 1)

    #     [x₁x₂]
    # y = [x₂x₃]
    #     [x₁x₃]
    #
    #         [x₂  x₁  0 ]
    # dy/dx = [0   x₃  x₂]
    #         [x₃  0   x₁]
    #
    #         [2  1  0]
    # dy/dx = [0  3  2]
    #         [3  0  1]
    y = VariableMatrix(3)
    y[0] = x[0] * x[1]
    y[1] = x[1] * x[2]
    y[2] = x[0] * x[2]
    J = Jacobian(y, x)

    expected_J = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 2.0], [3.0, 0.0, 1.0]])
    assert (J.get().value() == expected_J).all()
    assert (J.value().todense() == expected_J).all()


def test_nested_products() -> None:
    x = VariableMatrix(1)
    x[0].set_value(3)
    assert x.value(0) == 3.0

    #     [ 5x]   [15]
    # y = [ 7x] = [21]
    #     [11x]   [33]
    y = VariableMatrix(3)
    y[0] = 5 * x[0]
    y[1] = 7 * x[0]
    y[2] = 11 * x[0]
    assert y.value(0) == 15.0
    assert y.value(1) == 21.0
    assert y.value(2) == 33.0

    #     [y₁y₂]   [15⋅21]   [315]
    # z = [y₂y₃] = [21⋅33] = [693]
    #     [y₁y₃]   [15⋅33]   [495]
    z = VariableMatrix(3)
    z[0] = y[0] * y[1]
    z[1] = y[1] * y[2]
    z[2] = y[0] * y[2]
    assert z.value(0) == 315.0
    assert z.value(1) == 693.0
    assert z.value(2) == 495.0

    #     [ 5x]
    # y = [ 7x]
    #     [11x]
    #
    #         [ 5]
    # dy/dx = [ 7]
    #         [11]
    J = Jacobian(y, x)
    assert J.get().value()[0, 0] == 5.0
    assert J.get().value()[1, 0] == 7.0
    assert J.get().value()[2, 0] == 11.0
    assert J.value()[0, 0] == 5.0
    assert J.value()[1, 0] == 7.0
    assert J.value()[2, 0] == 11.0

    #     [y₁y₂]
    # z = [y₂y₃]
    #     [y₁y₃]
    #
    #         [y₂  y₁  0 ]   [21  15   0]
    # dz/dy = [0   y₃  y₂] = [ 0  33  21]
    #         [y₃  0   y₁]   [33   0  15]
    J = Jacobian(z, y)
    expected_J = np.array([[21.0, 15.0, 0.0], [0.0, 33.0, 21.0], [33.0, 0.0, 15.0]])
    assert (J.get().value() == expected_J).all()
    assert (J.value().todense() == expected_J).all()

    #     [y₁y₂]   [5x⋅ 7x]   [35x²]
    # z = [y₂y₃] = [7x⋅11x] = [77x²]
    #     [y₁y₃]   [5x⋅11x]   [55x²]
    #
    #         [ 70x]   [210]
    # dz/dx = [154x] = [462]
    #         [110x] = [330]
    J = Jacobian(z, x)
    expected_J = np.array([[210.0], [462.0], [330.0]])
    assert (J.get().value() == expected_J).all()
    assert (J.value().todense() == expected_J).all()


def test_non_square() -> None:
    x = VariableMatrix(3)
    for i in range(3):
        x[i].set_value(i + 1)

    # y = [x₁ + 3x₂ − 5x₃]
    #
    # dy/dx = [1  3  −5]
    y = VariableMatrix(1)
    y[0] = x[0] + 3 * x[1] - 5 * x[2]
    J = Jacobian(y, x)

    expected_J = np.array([[1.0, 3.0, -5.0]])

    J_get_value = J.get().value()
    assert J_get_value.shape == (1, 3)
    assert (J_get_value == expected_J).all()

    J_value = J.get().value()
    assert J_value.shape == (1, 3)
    assert (J_value == expected_J).all()


def test_mixed_rows() -> None:
    x = VariableMatrix(3)
    for i in range(3):
        x[i].set_value(i + 1)

    # Nonlinear, constant, leaf, and linear rows, with x₃ in none of them
    #
    #     [ x₁x₂  ]
    # y = [   5   ]
    #     [  x₁   ]
    #     [x₁ + 3x₂]
    #
    #         [x₂  x₁  0]
    # dy/dx = [0   0   0]
    #         [1   0   0]
    #         [1   3   0]
    y = VariableMatrix(4)
    y[0] = x[0] * x[1]
    y[1] = Variable(5.0)
    y[2] = x[0]
    y[3] = x[0] + 3.0 * x[1]
    J = Jacobian(y, x)
    J_get = J.get()

    expected_J = np.array(
        [[2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 3.0, 0.0]]
    )
    assert (J_get.value() == expected_J).all()
    assert (J.value().todense() == expected_J).all()

    # The symbolic Jacobian follows wrt's values
    x[0].set_value(3.0)
    x[1].set_value(4.0)
    expected_J[0, 0] = 4.0
    expected_J[0, 1] = 3.0
    assert (J_get.value() == expected_J).all()
    assert (J.value().todense() == expected_J).all()


def test_variable_reuse() -> None:
    x = VariableMatrix(2)
    for i in range(2):
        x[i].set_value(i + 1)

    # y = [x₁x₂]
    y = VariableMatrix(1)
    y[0] = x[0] * x[1]

    jacobian = Jacobian(y, x)

    # dy/dx = [x₂  x₁]
    # dy/dx = [2  1]
    J = jacobian.value()

    assert J.shape == (1, 2)
    assert J[0, 0] == 2.0
    assert J[0, 1] == 1.0

    x[0].set_value(2)
    x[1].set_value(1)
    # dy/dx = [x₂  x₁]
    # dy/dx = [1  2]
    J = jacobian.value()

    assert J.shape == (1, 2)
    assert J[0, 0] == 1.0
    assert J[0, 1] == 2.0
