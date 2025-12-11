import torch
from utils.general_utils import safe_interact

def restarted_hutchinson(Hz_func, z_gen_func, D_init, restart_iter=-1, num_iters=200, eps=1e-16, damp=0.0, 
                         with_xyz=True,
                         with_features_dc=True,
                         with_features_rest=True,
                         with_scaling=True,
                         with_rotation=True, 
                         with_opacity=True):

    D_accum = 0
    denom_iters = 0

    S_hat = 1 / (D_init + eps).sqrt()
    S_hat_sq = S_hat * S_hat

    D_est = D_init

    for it in range(num_iters):
        denom_iters += 1

        z = z_gen_func()

        if not with_xyz:
            z.xyz *= 0.0
        if not with_features_dc:
            z.features_dc *= 0.0
        if not with_features_rest:
            z.features_rest *= 0.0
        if not with_scaling:
            z.scaling *= 0.0
        if not with_rotation:
            z.rotation *= 0.0
        if not with_opacity:
            z.opacity *= 0.0

        Sz = S_hat * z

        HSz = Hz_func(Sz)

        Di = Sz * HSz / (S_hat_sq)

        D_accum += Di

        # DEBUG
        del z, Sz, HSz
        torch.cuda.empty_cache()

        if (restart_iter > 0 and denom_iters >= restart_iter) or (it + 1 == num_iters):
            D_est = D_accum / denom_iters
            D_vec = D_est.as_1d_tensor()

            S_hat = 1 / (D_est.abs() + eps).sqrt()
            S_hat_sq = S_hat * S_hat

            D_accum = 0
            denom_iters = 0

            # print("restart at ", it + 1)


    return D_est

def restarted_squared_hutchinson(Hz_func, z_gen_func, D_init, restart_iter=-1, num_iters=200, eps=1e-16, damp=0.0):

    D_sq_accum = 0
    denom_iters = 0

    S_hat = 1 / (D_init + eps).sqrt()
    S_hat_sq = S_hat * S_hat

    D_est = D_init

    for it in range(num_iters):
        denom_iters += 1

        z = z_gen_func()
        Sz = S_hat * z

        HSz = Hz_func(Sz)

        Di = Sz * HSz / (S_hat_sq)

        D_sq = Di * Di

        D_sq_accum += D_sq

        if (restart_iter > 0 and denom_iters >= restart_iter) or (it + 1 == num_iters):
            D_est = (D_sq_accum / denom_iters).sqrt()
            D_vec = D_est.as_1d_tensor()

            S_hat = 1 / (D_est + eps).sqrt()
            S_hat_sq = S_hat * S_hat

            D_sq_accum = 0
            denom_iters = 0

            # print("restart at ", it + 1)


    return D_est
