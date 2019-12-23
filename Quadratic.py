from algms import steepest_descent,steepest_descent_diag_Hessian, newton, conjugate_gradient, bfgs, l_bfgs, PQN_threshold, PQN_random
import numpy as np
import matplotlib.pyplot as plt
import time


def f(x):
    global Q, p
    return (x.T.dot(Q)*x.T).sum() + p.dot(x) 

def g(x):
    global Q, p
    return Q.dot(x) + p

def h(x):
    global Q
    return Q


if __name__ == '__main__':
    no_instances = 10
    error = 1e-6
    max_iterations = 2000
    matrixSize = 10
    x0 = np.zeros(matrixSize)
    np.random.seed(1)
    algo_errors = np.zeros((8, no_instances))

    for i in range(no_instances):
        print('\n====================\n\
        ============= iteration={} ==============================\n'.format(i))
        A = np.random.rand(matrixSize, matrixSize)
        Q = np.dot(A,A.transpose())
        p = np.random.rand(matrixSize)

        print('\n======= 0. Steepest Descent ======\n')
        start = time.time()
        x, n_iter = steepest_descent(f, g, x0,
                                    iterations=max_iterations, error=error)
        end = time.time()
        print("  Steepest Descent terminated in {} iterations, x = {}, f(x) = {}, time elapsed {}, time per iter {}"\
        .format(n_iter, x, f(x), end - start, (end - start) / n_iter))
        algo_errors[0, i] = np.linalg.norm(g(x))

        print('\n======= 1. Steepest Descent diag Hessian ======\n')
        start = time.time()
        x, n_iter = steepest_descent_diag_Hessian(Q, f, g, x0,
                                    iterations=max_iterations, error=error)
        end = time.time()
        print("  SGD with diag Hessian terminated in {} iterations, x = {}, f(x) = {}, time elapsed {}, time per iter {}"\
        .format(n_iter, x, f(x), end - start, (end - start) / n_iter))
        algo_errors[1, i] = np.linalg.norm(g(x)) if np.linalg.norm(g(x)) > error else 0

        print('\n======= 2. Conjugate Gradient Method ======\n')
        start = time.time()
        x, n_iter = conjugate_gradient(f, g, x0,
                                        iterations=max_iterations, error=error)
        end = time.time()
        print("  Conjugate Gradient Method terminated in {} iterations, x = {}, f(x) = {}, time elapsed {}, time per iter {}"\
        .format(n_iter, x, f(x), end - start, (end - start) / n_iter))
        algo_errors[2, i] = np.linalg.norm(g(x)) if np.linalg.norm(g(x)) > error else 0

        print('\n======= 3. Newton\'s Method ======\n')
        start = time.time()
        x, n_iter = newton(f, g, h, x0,
                            iterations=max_iterations, error=error)
        end = time.time()
        print("  Newton\'s Method terminated in {} iterations, x = {}, f(x) = {}, time elapsed {}, time per iter {}" \
        .format(n_iter, x, f(x), end - start, (end - start) / n_iter))
        algo_errors[3, i] = np.linalg.norm(g(x)) if np.linalg.norm(g(x)) > error else 0

        print('\n======= 4. Broyden-Fletcher-Goldfarb-Shanno ======\n')
        start = time.time()
        x, n_iter = bfgs(f, g, x0,
                        iterations=max_iterations, error=error)
        end = time.time()
        print("  BFGS terminated in {} iterations, x = {}, f(x) = {}, time elapsed {}, time per iter {}"\
        .format(n_iter, x, f(x), end - start, (end - start) / n_iter))
        algo_errors[4, i] = np.linalg.norm(g(x)) if np.linalg.norm(g(x)) > error else 0

        print('\n======= 5. Limited memory Broyden-Fletcher-Goldfarb-Shanno ======\n')
        start = time.time()
        x, n_iter = l_bfgs(f, g, x0,
                            iterations=max_iterations, error=error)
        end = time.time()
        print("  l-BFGS terminated in {} iterations, x = {}, f(x) = {}, time elapsed {}, time per iter {}"\
        .format(n_iter, x, f(x), end - start, (end - start) / n_iter))
        algo_errors[5, i] = np.linalg.norm(g(x)) if np.linalg.norm(g(x)) > error else 0

        print('\n======= 6. Partial Quasi-Newton threshold ======\n')
        start = time.time()
        x, n_iter = PQN_threshold(Q, f, g, x0,
                            iterations=max_iterations, error=error, portion=0.5)
        end = time.time()
        print("  PQN terminated in {} iterations, x = {}, f(x) = {}, time elapsed {}, time per iter {}"\
        .format(n_iter, x, f(x), end - start, (end - start) / n_iter))
        algo_errors[6, i] = np.linalg.norm(g(x)) if np.linalg.norm(g(x)) > error else 0

        print('\n======= 7. Partial Quasi-Newton random ======\n')
        start = time.time()
        x, n_iter = PQN_random(Q, f, g, x0,
                            iterations=max_iterations, error=error, portion=0.5)
        end = time.time()
        print("  PQN_random terminated in {} iterations, x = {}, f(x) = {}, time elapsed {}, time per iter {}"\
        .format(n_iter, x, f(x), end - start, (end - start) / n_iter))
        algo_errors[7, i] = np.linalg.norm(g(x)) if np.linalg.norm(g(x)) > error else 0


    np.savetxt("algo_errors.csv", algo_errors, delimiter=",")
    np.savetxt("sum_of_errors.csv", algo_errors.sum(axis=1), delimiter=",")

    for i in range(8):
        plt.plot(np.log(algo_errors[i, :]), label = str(i))
    plt.legend()
    plt.xlabel('No. of ins')
    plt.ylabel('log error')
    plt.savefig('errors.png', dpi = 1000, quality=100)

    