import numpy as np
from time import perf_counter


def FISTA(niter, f, g, x_0, L_0=1, eta=1.1, precision=1e-6, verbose=False):
    if verbose:
        start = perf_counter()

    x_k = x_0
    y_k = x_0
    t_k = 1
    L_k = L_0
    f_cost_k = f(x_k)
    g_cost_k = g(x_k)

    if verbose:
        cost_history = [f_cost_k + g_cost_k]

    for k in range(1, niter + 1):
        criteria = True

        while criteria:
            f_grad = f.grad(y_k)
            prox_step = g.prox(y_k - f_grad / L_k, L_k)
            g_prox_step = g(prox_step)
            diff = (prox_step - y_k).reshape(-1)

            if f(prox_step) + g_prox_step <= f(y_k) + np.dot(diff, f_grad.reshape(-1) + L_k / 2 * diff) + g_prox_step:
                criteria = False

            else:
                L_k *= eta

        t_next = (1 + np.sqrt(1 + 4 * t_k**2)) / 2

        y_k = prox_step + (t_k - 1) / t_next * (prox_step - x_k)

        x_k = prox_step
        t_k = t_next

        if verbose:
            cost_history.append(f(x_k) + g_prox_step)
            final_iter = k

        if abs(f(x_k) - f_cost_k) > precision and abs(g_prox_step - g_cost_k) > precision:
            f_cost_k = f(x_k)
            g_cost_k = g_prox_step

        else:
            break

    if verbose:
        return x_k, cost_history, final_iter, perf_counter() - start

    return x_k