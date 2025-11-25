import math
from utils.general_utils import safe_interact

def conjugate_gradient(
    Ax,             # A @ x, here A is assumed to be SPD
    dot,            # dot(x, y)
    saxpy,          # ax + y
    b,              # right-hand side
    x0,             # initial guess
    ML_inv=1.0,
    MR_inv=1.0,     # preconditioner ML_inv A MR_inv ~= I
    tol=1e-10,
    atol=0.0,
    max_iter=1000,
    restart_iter=1, # restart every `restart_iter` iterations
    callback=None,
    verbose=False
):
    x = x0
    iter_total = 0

    break_flag = False

    b = b

    while iter_total < max_iter:
        print(f"Restarting CG iteration {iter_total + 1}...") if verbose else None

        r = saxpy(-1.0, Ax(x), ML_inv * b)         # r = b - A x
        res = dot(r, r)
        z = ML_inv * r
        p = MR_inv * z

        print(f"[Iter {iter_total}] res: {res:.6e}") if verbose else None

        if iter_total == 0:
            res_init = res

        for k in range(restart_iter):
            gamma = dot(r, z)                       # gamma = <r, z>
            q = Ax(p)                               # q = A p
            delta = dot(p, q)                       # delta = <q, p>
            if delta < tol:
                print(f"Early termination: delta is too small: {delta:.2e}")
                # safe_interact(local=locals(), banner="Debugging CG...")
                break_flag = True
                break
            alpha = gamma / delta
            x = saxpy(alpha, p, x)                  # x = x + alpha * p
            r = saxpy(-alpha, q, r)                 # r = r - alpha * q
            res = dot(r, r)
            z = ML_inv * r                          # z = ML^-1 r
            gamma_prev = gamma
            # TODO: One of these inner products is redundant
            gamma = dot(r, z)                       # Update gamma
            beta = gamma / gamma_prev
            p = saxpy(beta, p, MR_inv * z)          # p = MR^-1 z + beta * p

            # if verbose:
            x_norm = math.sqrt(dot(x, x))
            print(f"[Iter {iter_total+1}] res: {res:.6e}, |x|: {x_norm:.2e}, delta: {delta:.2e}, gamma: {gamma:.2e}") if verbose else None
            # import code; code.interact(local=locals(), banner="Debugging CG...")

            iter_total += 1
            if iter_total >= max_iter:
                break_flag = True
                break

        if break_flag:
            break

    r = saxpy(-1.0, Ax(x), b)         # r = b - A x
    res = dot(r, r)
    print(f"Final residual norm: {res:.2e}")

    res_rel = res / res_init

    return x, res, iter_total, res_rel

def cg_damped(
    Ax,             # A @ x, here A is assumed to be SPD
    dot,            # dot(x, y)
    saxpy,          # ax + y
    b,              # right-hand side
    x0,             # initial guess
    M=None,         # preconditioner M ~ A^-1
    tol=1e-10,
    atol=0.0,
    max_iter=1000,
    restart_iter=5, # restart every `restart_iter` iterations
    callback=None,
    verbose=False
):
    x = x0
    iter_total = 0

    break_flag = False

    while iter_total < max_iter:
        print(f"Restarting CG iteration {iter_total + 1}...") if verbose else None

        r = saxpy(-1.0, Ax(x), b)         # r = b - A x
        res = dot(r, r)
        z = M(r) if M is not None else r  # z = M r
        p = z

        print(f"[Iter {iter_total}] res: {res:.2e}") if verbose else None

        if iter_total == 0:
            res_init = res
        
        safe_interact(local=locals(), banner="debugging pcg")

        for k in range(restart_iter):
            gamma = dot(r, z)                       # gamma = <r, z>
            q = Ax(p)                               # q = A p
            delta = dot(p, q)                       # delta = <q, p>
            if delta < tol:
                print(f"Early termination: delta is too small: {delta:.2e}")
                # safe_interact(local=locals(), banner="Debugging CG...")
                break_flag = True
                break
            alpha = gamma / delta
            x = saxpy(alpha, p, x)                  # x = x + alpha * p
            r = saxpy(-alpha, q, r)                 # r = r - alpha * q
            res = dot(r, r)
            z = M(r) if M is not None else r        # z = M r
            gamma_prev = gamma
            # TODO: One of these inner products is redundant
            gamma = dot(r, z)                       # Update gamma
            beta = gamma / gamma_prev
            p = saxpy(beta, p, z)                   # p = z + beta * p

            # if verbose:
            x_norm = math.sqrt(dot(x, x))
            print(f"[Iter {iter_total+1}] res: {res:.2e}, |x|: {x_norm:.2e}, delta: {delta:.2e}, gamma: {gamma:.2e}") if verbose else None
            # import code; code.interact(local=locals(), banner="Debugging CG...")

            iter_total += 1
            if iter_total >= max_iter:
                break_flag = True
                break

        if break_flag:
            break

    r = saxpy(-1.0, Ax(x), b)         # r = b - A x
    res = dot(r, r)
    print(f"Final residual norm: {res:.2e}")

    res_rel = res / res_init

    return x, res, iter_total, res_rel


def cgls_damped(
    Ax,         # A @ x
    Atx,       # A.T @ y
    dot,            # dot(x, y)
    saxpy,          # ax + y
    b,              # right-hand side
    x0,             # initial guess
    damp=0.0,       # damping factor
    tol=1e-10,
    atol=0.0,
    max_iter=1000,
    restart_iter=5, # restart every `restart_iter` iterations
    callback=None,
    verbose=False
):
    x = x0
    iter_total = 0

    last_res = math.inf

    break_flag = False

    while iter_total < max_iter:
        print(f"Restarting CG iteration {iter_total + 1}...") if verbose else None

        r0 = saxpy(-1.0, Ax(x), b)         # r0 = b - A x0
        s0 = saxpy(-damp, x, Atx(r0))  # s0 = A^T r0 - λ^2 x0
        p0 = s0

        r = r0
        s = s0
        p = p0
        gamma = dot(s, s)  # Initial norm of s

        for k in range(restart_iter):
            q = Ax(p)                  # q = A p
            delta = dot(q, q) + dot(p, p, damp)  # delta = <q, q> + λ^2 * <p, p>
            if delta < 1e-20:
                print("Early termination: delta is too small.")
                break_flag = True
                break
            # print(f"delta: {delta:.4e}, gamma: {gamma:.4e}")
            alpha = gamma / delta
            x = saxpy(alpha, p, x)         # x = x + alpha * p
            r = saxpy(-alpha, q, r)        # r = r - alpha * q
            s = saxpy(-damp, x, Atx(r))  # s = A^T r - λ^2 x
            gamma_prev = gamma
            gamma = dot(s, s)  # Update norm of s
            beta = gamma / gamma_prev
            p = saxpy(beta, p, s)          # p = s + beta * p

            # if verbose:
            cur_r = saxpy(-1.0, Ax(x), b)         # r0 = b - A x0
            res = dot(cur_r, cur_r) + dot(x, x, damp)  # Compute residual norm
            print(f"[Iter {iter_total+1}] res: {res:.2e}")
            if res > last_res:
                print("Warning: Residual norm increased!")
                break_flag = True
                break

            last_res = res

            if gamma < max(tol * (gamma_prev ** 0.5), atol):
                if verbose:
                    print(f"Convergence achieved at iteration {iter_total+1}.")
                break_flag = True
                break

            iter_total += 1
            if iter_total >= max_iter:
                break_flag = True
                break

        if break_flag:
            break

    return x

