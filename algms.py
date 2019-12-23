"""
Self-contained implementation of non-linear optimization algorithms:

- steepest descent
- newton's method
- conjuage gradient
- BFGS
- l-BFGS

Following Nocedal & Wright's Numerical Optimization Chapter 3, 5 & 8
"""

import math
import time
import numpy as np
import random


# line-search conditions
def wolfe(f, g, xk, alpha, pk):
  c1 = 1e-4
  return f(xk + alpha * pk) <= f(xk) + c1 * alpha * np.dot(g(xk), pk)


def strong_wolfe(f, g, xk, alpha, pk, c2):
  # typically, c2 = 0.9 when using Newton or quasi-Newton's method.
  #            c2 = 0.1 when using non-linear conjugate gradient method.
  return wolfe(f, g, xk, alpha, pk) and abs(
      np.dot(g(xk + alpha * pk), pk)) <= c2 * abs(np.dot(g(xk), pk))


def gold_stein(f, g, xk, alpha, pk, c):
  return (f(xk) + (1 - c) * alpha * np.dot(g(xk), pk) <= f(xk + alpha * pk)
          ) and (f(xk + alpha * pk) <= f(xk) + c * alpha * np.dot(g(xk), pk))


# line-search step len
def step_length(f, g, xk, alpha, pk, c2):
  return interpolation(f, g,
                       lambda alpha: f(xk + alpha * pk),
                       lambda alpha: np.dot(g(xk + alpha * pk), pk),
                       alpha, c2,
                       lambda f, g, alpha, c2: wolfe(f, g, xk, alpha, pk))


def interpolation(f, g, f_alpha, g_alpha, alpha, c2, strong_wolfe_alpha, iters=20):
  # referred implementation here:
  # https://github.com/tamland/non-linear-optimization
  # http://people.math.sc.edu/kellerlv/Quadratic_Interpolation.pdf
  l = 0.0
  h = 1.0
  for i in range(iters):
    if strong_wolfe_alpha(f, g, alpha, c2):
      return alpha

    half = (l + h) / 2
    alpha = - g_alpha(l) * (h-l) ** 2  / (2 * (f_alpha(h) - f_alpha(l) - g_alpha(l) * (h - l)))
    if alpha < l or alpha > h: # quadratic interpolation failed. reduce by half instead
      alpha = half
    if g_alpha(alpha) > 0:
      h = alpha
    elif g_alpha(alpha) <= 0:
      l = alpha
  return alpha


# optimization algorithms
def steepest_descent(f, g, x0, iterations, error):
  x = x0
  x_old = x
  c2 = 0.9
  for i in range(iterations):
    pk = -g(x)
    alpha = step_length(f, g, x, 1.0, pk, c2)
    x = x + alpha * pk
    if i % 1000 == 0:
      print("  iter={}, g={}, alpha={}, x={}, f(x)={},SquaredGradient={}".\
        format(i, pk, alpha, x, f(x), np.linalg.norm(pk)))

    if np.linalg.norm(pk) < error:
      print("end:  iter={}, SquaredGradient={}, x_error_norm={}".format(i, np.linalg.norm(pk), np.linalg.norm(x - x_old)))
      break
    x_old = x
  return x, i

  # steepest_descent_diag_Hessian algorithms
def steepest_descent_diag_Hessian(Q, f, g, x0, iterations, error):
  x = x0
  x_old = x
  c2 = 0.9
  for i in range(iterations):

    Bk = np.diag(Q)
    Hk = 1 / (Bk + 10e-6)
    pk = -Hk * g(x)
    alpha = step_length(f, g, x, 1.0, pk, c2)
    x = x + alpha * pk
    if i % 1000 == 0:
      print("  iter={}, g={}, alpha={}, x={}, f(x)={},SquaredGradient={}".\
        format(i, pk, alpha, x, f(x), np.linalg.norm(pk)))

    if np.linalg.norm(pk) < error:
      print("end:  iter={}, SquaredGradient={}, x_error_norm={}".format(i, np.linalg.norm(pk), np.linalg.norm(x - x_old)))
      break
    x_old = x
  return x, i


def newton(f, g, H, x0, iterations, error):
  x = x0
  c2 = 0.9
  for i in range(iterations):
    pk = -np.linalg.solve(H(x), g(x))
    alpha = step_length(f, g, x, 1.0, pk, c2)
    x = x + alpha * pk
    if i % 1000 == 0:
      print("  iter={}, direction={}, alpha={}, x={}, f(x)={}, SquaredGradient={}".\
        format(i, pk, alpha, x, f(x), np.linalg.norm(g(x))))
      
    cur_error = np.linalg.norm(g(x))
    x_error = np.linalg.norm(alpha * pk)
    if  cur_error < error:  
      break

  print("end:  iter={}, SquaredGradient={}, x_error_norm={}".\
    format(i, np.linalg.norm(g(x)), x_error))
  return x, i + 1


def conjugate_gradient(f, g, x0, iterations, error):
  xk = x0
  c2 = 0.1

  fk = f(xk)
  gk = g(xk)
  pk = -gk

  for i in range(iterations):
    alpha = step_length(f, g, xk, 1.0, pk, c2)
    xk1 = xk + alpha * pk
    gk1 = g(xk1)
    beta_k1 = np.dot(gk1, gk1) / np.dot(gk, gk)
    pk1 = -gk1 + beta_k1 * pk

    if i % 1000 == 0:
      print("  iter={}, direction={}, alpha={}, x={}, f(x)={}, SquaredGradient={}, x_error_norm={}".\
        format(i, pk, alpha, xk, f(xk), np.linalg.norm(gk), np.linalg.norm(xk1 - xk)))
  
    cur_error = np.linalg.norm(gk1)
    x_error = np.linalg.norm(xk - xk1)
    xk = xk1
    gk = gk1
    pk = pk1

    if  cur_error < error:
      break

    
  print("end:  iter={}, SquaredGradient={}, x_error_norm={}".\
    format(i, np.linalg.norm(gk1), x_error))
  return xk, i + 1


def bfgs(f, g, x0, iterations, error):
  xk = x0
  c2 = 0.9
  I = np.identity(xk.size)
  gk = g(xk)
  k = 30

  for i in range(iterations):
    if i % k == 0:
      Hk = I

    # compute search direction
    pk = -Hk.dot(gk)

    # obtain step length by line search
    alpha = step_length(f, g, xk, 1.0, pk, c2)

    # update x
    xk1 = xk + alpha * pk
    gk1 = g(xk1)

    # define sk and yk for convenience
    sk = xk1 - xk
    yk = gk1 - gk

    # compute H_{k+1} by BFGS update
    ys = yk.dot(sk)
    if ys < 1e-6:
      ys = 1e-6
    rho_k = float(1.0 / ys)

    Hk1 = (I - rho_k * np.outer(sk, yk)).dot(Hk).dot(I - \
           rho_k * np.outer(yk, sk)) + rho_k * np.outer(sk, sk)

    if i % 1000 == 0:
      print("  iter={}, direction={}, alpha={}, x={}, f(x)={}, SquaredGradient={}, x_error_norm={}".\
        format(i, pk, alpha, xk, f(xk), np.linalg.norm(gk), np.linalg.norm(xk1 - xk)))

    cur_error = np.linalg.norm(gk1)
    x_error = np.linalg.norm(xk - xk1)
    Hk = Hk1
    xk = xk1
    gk = gk1

    if  cur_error < error:
      break
    
  print("end:  iter={}, SquaredGradient={}, x_error_norm={}".\
        format(i, np.linalg.norm(gk1), x_error))
  return xk, i + 1


def l_bfgs(f, g, x0, iterations, error, m=10):
  xk = x0
  c2 = 0.9
  I = np.identity(xk.size)
  #Hk = I

  sks = []
  yks = []

  def Hp(H0, p):
    m_t = len(sks)
    q = g(xk)
    a = np.zeros(m_t)
    b = np.zeros(m_t)
    for i in reversed(range(m_t)):
      s = sks[i]
      y = yks[i]
      rho_i = float(1.0 / (y.T.dot(s)  + 1e-8))
      a[i] = rho_i * s.dot(q)
      q = q - a[i] * y

    r = H0.dot(q)

    for i in range(m_t):
      s = sks[i]
      y = yks[i]
      rho_i = float(1.0 / (y.T.dot(s) + 1e-8))
      b[i] = rho_i * y.dot(r)
      r = r + s * (a[i] - b[i])

    return r

  for i in range(iterations):
    # compute search direction
    gk = g(xk)
    pk = -Hp(I, gk)

    # obtain step length by line search
    alpha = step_length(f, g, xk, 1.0, pk, c2)

    # update x
    xk1 = xk + alpha * pk
    gk1 = g(xk1)

    # define sk and yk for convenience
    sk = xk1 - xk
    yk = gk1 - gk

    sks.append(sk)
    yks.append(yk)
    if len(sks) > m:
      sks = sks[1:]
      yks = yks[1:]

    # compute H_{k+1} by BFGS update
    # rho_k = float(1.0 / yk.dot(sk))

    if i % 1000 == 0:
      print("\n  iter={}, direction={}, alpha={}, x={}, f(x)={}, SquaredGradient={}, x_error_norm={}".\
        format(i, pk, alpha, xk, f(xk), np.linalg.norm(gk), np.linalg.norm(xk1 - xk)))

    cur_error = np.linalg.norm(gk1)
    x_error = np.linalg.norm(xk - xk1)
    xk = xk1
    if  cur_error < error:
      break

  print("\n end:  iter={}, direction={}, alpha={},SquaredGradient={}, x_error_norm={}".\
        format(i, pk, alpha,np.linalg.norm(gk1), x_error))

  return xk, i + 1

def random_index(N, M):
    idx = np.array(range(N))
    np.random.shuffle(idx)
    return idx[:M], idx[M:]

def top_sy_product(sy, N, M):
    temp = np.argpartition(sy, -M)
    return temp[-M:], temp[:N-M]

def threshold(lower, gk, N, M):
    sort_idx = np.argsort(gk)

    if gk[sort_idx[N - M]] < lower:
      return sort_idx[-M:], sort_idx[:N - M]

    p_index = np.zeros(M, dtype=int)
    np_index = np.zeros(N - M, dtype=int)
    p_cnt = np_cnt = 0
    for i in range(N):
      if gk[sort_idx[i]] > lower and p_cnt < M:
        p_index[p_cnt] = sort_idx[i]
        p_cnt += 1
      else:
        np_index[np_cnt] = sort_idx[i]
        np_cnt += 1
    return p_index, np_index
        

def PQN(f, g, x0, iterations, error, portion):
    xk = x0
    c2 = 0.9
    N = xk.size
    M = int(N * portion)
    I = np.identity(M)  
    gk = g(xk)
    k = 20
    pk = np.zeros(N)
    Hk = I  
    cumulative_y = np.zeros(N)
    temp = np.argpartition(gk, M-1)
    p_index, np_index = temp[:M], temp[M:]


    for i in range(iterations):
        #Reset Hessian
        if i != 0 and i % k == 0:
            Hk = I        
            p_index, np_index = top_cumulative_g_diff(cumulative_y, N, M)
            cumulative_y = np.zeros(N)

        # compute search direction        
        pk[p_index] = -Hk.dot(gk[p_index])
        pk[np_index] = -gk[np_index]

        # obtain step length by line search
        alpha = step_length(f, g, xk, 1.0, pk, c2)

        # update x
        xk1 = xk + alpha * pk
        gk1 = g(xk1)

        # define sk and yk for convenience
        sk = xk1[p_index] - xk[p_index]
        yk = gk1[p_index] - gk[p_index]
        cumulative_y += np.absolute(gk1 - gk)

        # compute H_{k+1} by BFGS update
        ys = yk.dot(sk)
        if abs(ys) < 1e-6:
          ys = 1e-6
        rho_k = float(1.0 / ys)
        

        Hk1 = (I - rho_k * np.outer(sk, yk)).dot(Hk).dot(I - \
            rho_k * np.outer(yk, sk)) + rho_k * np.outer(sk, sk)

        if i % 1000 == 0:
          print("\n  iter={}, direction={}, alpha={}, x={}, f(x)={}, SquaredGradient={}, x_error_norm={}".format(i, pk, alpha, xk, f(xk), np.linalg.norm(gk), np.linalg.norm(xk1 - xk)))
            
        cur_error = np.linalg.norm(xk1 - xk)
        if cur_error < error:
          xk = xk1
          break

        Hk = Hk1
        xk = xk1
        gk = gk1

    print("\n end:  iter={}, direction_p={}, direction_np={}, alpha={},SquaredGradient={}, x_error_norm={}".\
              format(i, pk[p_index], pk[np_index], alpha, np.linalg.norm(gk1), cur_error))
            
    return xk, i + 1

def PQN_threshold(Q, f, g, x0, iterations, error, portion):
    xk = x0
    c2 = 0.9
    N = xk.size
    M = int(N * portion)
    I = np.identity(M)  
    gk = g(xk)
    k = 30
    pk = np.zeros(N)
    Hk = I  


    for i in range(iterations):
        #Reset Hessian
        if i % k == 0:
            Hk = I        
            p_index, np_index = threshold(10e-2, gk, N, M)
            
        Bk_np = np.diag(Q)[np_index]
        Hk_np = 1 / (Bk_np + 1e-8)
        pk[np_index] = -Hk_np * gk[np_index]
        #BFGS
        pk[p_index] = -Hk.dot(gk[p_index])
        

        # obtain step length by line search
        alpha = step_length(f, g, xk, 1.0, pk, c2)

        # update x
        xk1 = xk + alpha * pk
        gk1 = g(xk1)

        # define sk and yk for convenience
        sk = xk1[p_index] - xk[p_index]
        yk = gk1[p_index] - gk[p_index]

        # compute H_{k+1} by BFGS update
        ys = yk.dot(sk)
        if abs(ys) < 1e-6:
          ys = 1e-6
        rho_k = float(1.0 / ys)
        

        Hk1 = (I - rho_k * np.outer(sk, yk)).dot(Hk).dot(I - \
            rho_k * np.outer(yk, sk)) + rho_k * np.outer(sk, sk)

        if i % 1000 == 0:
          print("\n  iter={}, direction={}, alpha={}, x={}, f(x)={}, SquaredGradient={}, x_error_norm={}".format(i, pk, alpha, xk, f(xk), np.linalg.norm(gk), np.linalg.norm(xk1 - xk)))
            
        cur_error = np.linalg.norm(gk1)
        x_error = np.linalg.norm(xk - xk1)
        Hk = Hk1
        xk = xk1
        gk = gk1

        if cur_error < error:
          break
        

    print("\n end:  iter={}, direction_p={}, direction_np={}, alpha={},SquaredGradient={}, x_error_norm={}".\
              format(i, pk[p_index], pk[np_index], alpha, np.linalg.norm(gk1), x_error))
            
    return xk, i + 1


def PQN_random(Q, f, g, x0, iterations, error, portion):
    xk = x0
    c2 = 0.9
    N = xk.size
    M = int(N * portion)
    I = np.identity(M)  
    gk = g(xk)
    k = 30
    pk = np.zeros(N)
    
    for i in range(iterations):
        #Reset Hessian
        if i % k == 0:
            Hk = I        
            #find the array of index that will be selected to calculate Hessian
            p_index, np_index = random_index(N, M)

        # compute search direction    
        #Gradient descent with diagnal Hessian  
        Bk_np = np.diag(Q)[np_index]
        Hk_np = 1 / (Bk_np + 1e-8)
        pk[np_index] = -Hk_np * gk[np_index]
        #BFGS
        pk[p_index] = -Hk.dot(gk[p_index])
        
        # obtain step length by line search
        alpha = step_length(f, g, xk, 1.0, pk, c2)

        # update x
        xk1 = xk + alpha * pk
        gk1 = g(xk1)

        # define sk and yk for convenience
        sk = xk1[p_index] - xk[p_index]
        yk = gk1[p_index] - gk[p_index]
        

        # compute H_{k+1} by BFGS update
        
        ys = (yk * sk).sum()
        # if ys < 1e-10:
        #   ys = 1e-10
        rho_k = float(1.0 / ys)
        

        Hk1 = (I - rho_k * np.outer(sk, yk)).dot(Hk).dot(I - \
            rho_k * np.outer(yk, sk)) + rho_k * np.outer(sk, sk)

        if i % 1000 == 0:
          print("\n  iter={}, direction_p={}, direction_np={}, alpha={}, x={}, f(x)={}, SquaredGradient={}, x_error_norm={}".\
            format(i, pk[p_index], pk[np_index], alpha, xk, f(xk), np.linalg.norm(gk), np.linalg.norm(xk1 - xk)))
            
        cur_error = np.linalg.norm(gk1)
        x_error = np.linalg.norm(xk - xk1)
        Hk = Hk1
        xk = xk1
        gk = gk1

        if  cur_error < error:  
            break

    print("\n end:  iter={}, direction_p={}, direction_np={}, alpha={},SquaredGradient={}, x_error_norm={}".\
              format(i, pk[p_index], pk[np_index], alpha, np.linalg.norm(gk1), x_error))
        
    return xk, i + 1

