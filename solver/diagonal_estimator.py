import torch
from utils.general_utils import safe_interact


def restarted_squared_hutchinson(Hz_func, z_gen_func, D_init, restart_iter=-1, num_iters=200, eps=1e-16, damp=0.0):

    D_sq_accum = 0
    denom_iters = 0

    S_hat = 1 / (D_init + eps).sqrt()
    S_hat_sq = S_hat * S_hat

    for it in range(num_iters):
        denom_iters += 1

        z = z_gen_func()
        Sz = S_hat * z

        HSz = Hz_func(Sz)

        Di = Sz * HSz / (S_hat_sq)

        D_sq = Di * Di

        # D_sq_vec = D_sq.as_1d_tensor()
        # D_sq_vec[D_sq_vec < (damp ** 2)] = damp ** 2
        # D_sq.load_1d_tensor(D_sq_vec)

        D_sq_accum += D_sq

        if (restart_iter > 0 and denom_iters >= restart_iter) or (it + 1 == num_iters):
            D_est = (D_sq_accum / denom_iters).sqrt()
            D_vec = D_est.as_1d_tensor()

            S_hat = 1 / (D_est + eps).sqrt()
            S_hat_sq = S_hat * S_hat

            D_sq_accum = 0
            denom_iters = 0

            print("restart at ", it + 1)


    return D_est
