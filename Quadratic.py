from algms import steepest_descent,newton, conjugate_gradient, bfgs, l_bfgs, PQN, PQN_v2
import numpy as np
import time

def rosenbrock(x):
  return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_rosen(x):
  return np.array([200*(x[1]-x[0]**2)*(-2*x[0]) + 2*(x[0]-1), 200*(x[1]-x[0]**2)])

def hessian_rosen(x):
  return np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]], [-400*x[0], 200]])

def f(x):
    global Q, p
    return (x.T.dot(Q)*x.T).sum() + (p*x).sum()

def g(x):
    global Q, p
    return Q.dot(x) + p

def h(x):
    global Q
    return Q


if __name__ == '__main__':
    error = 1e-6
    max_iterations = 5000
    matrixSize = 100
    x0 = np.zeros(matrixSize)
    np.random.seed(217)

    for i in range(1):
        print('\n====================\n\
        ============= iteration={} ==============================\n'.format(i))
        A = np.random.rand(matrixSize, matrixSize)
        Q = np.dot(A,A.transpose())
        p = np.random.rand(matrixSize)

        print('\n======= Steepest Descent ======\n')
        start = time.time()
        x, n_iter = steepest_descent(f, g, x0,
                                    iterations=max_iterations, error=error)
        end = time.time()
        print("  Steepest Descent terminated in {} iterations, x = {}, f(x) = {}, time elapsed {}, time per iter {}"\
        .format(n_iter, x, f(x), end - start, (end - start) / n_iter))

        print('\n======= Conjugate Gradient Method ======\n')
        start = time.time()
        x, n_iter = conjugate_gradient(f, g, x0,
                                        iterations=max_iterations, error=error)
        end = time.time()
        print("  Conjugate Gradient Method terminated in {} iterations, x = {}, f(x) = {}, time elapsed {}, time per iter {}"\
        .format(n_iter, x, f(x), end - start, (end - start) / n_iter))

        print('\n======= Newton\'s Method ======\n')
        start = time.time()
        x, n_iter = newton(f, g, h, x0,
                            iterations=max_iterations, error=error)
        end = time.time()
        print("  Newton\'s Method terminated in {} iterations, x = {}, f(x) = {}, time elapsed {}, time per iter {}" \
        .format(n_iter, x, f(x), end - start, (end - start) / n_iter))

        print('\n======= Broyden-Fletcher-Goldfarb-Shanno ======\n')
        start = time.time()
        x, n_iter = bfgs(f, g, x0,
                        iterations=max_iterations, error=error)
        end = time.time()
        print("  BFGS terminated in {} iterations, x = {}, f(x) = {}, time elapsed {}, time per iter {}"\
        .format(n_iter, x, f(x), end - start, (end - start) / n_iter))

        print('\n======= Limited memory Broyden-Fletcher-Goldfarb-Shanno ======\n')
        start = time.time()
        x, n_iter = l_bfgs(f, g, x0,
                            iterations=max_iterations, error=error)
        end = time.time()
        print("  l-BFGS terminated in {} iterations, x = {}, f(x) = {}, time elapsed {}, time per iter {}"\
        .format(n_iter, x, f(x), end - start, (end - start) / n_iter))

        print('\n======= Partial Quasi-Newton ======\n')
        start = time.time()
        x, n_iter = PQN(f, g, x0,
                            iterations=max_iterations, error=error, portion=0.1)
        end = time.time()
        print("  PQN terminated in {} iterations, x = {}, f(x) = {}, time elapsed {}, time per iter {}"\
        .format(n_iter, x, f(x), end - start, (end - start) / n_iter))

        print('\n======= Partial Quasi-Newton v2 ======\n')
        start = time.time()
        x, n_iter = PQN_v2(f, g, x0,
                            iterations=max_iterations, error=error, portion=0.1)
        end = time.time()
        print("  PQN_v2 terminated in {} iterations, x = {}, f(x) = {}, time elapsed {}, time per iter {}"\
        .format(n_iter, x, f(x), end - start, (end - start) / n_iter))