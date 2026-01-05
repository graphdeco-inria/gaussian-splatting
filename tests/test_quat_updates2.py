import numpy as np
import torch
from matplotlib import pyplot as plt

rotation_activation = torch.nn.functional.normalize

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def quat_to_rot(quat, normalize=True):
    R = torch.zeros(quat.shape[0], 3, 3, device=quat.device, dtype=quat.dtype)
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    q = (x**2 + y**2 + z**2 + w**2).sqrt()
    R[:, 0, 0] = q ** 2 - 2 * (y ** 2 + z ** 2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = q ** 2 - 2 * (x ** 2 + z ** 2)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = q ** 2 - 2 * (x ** 2 + y ** 2)

    if normalize:
        R /= (q ** 2).unsqueeze(-1).unsqueeze(-1)

    return R

def quat_to_drot_da(quat, delta_quat):
    dR = torch.zeros(quat.shape[0], 3, 3, device=quat.device, dtype=quat.dtype)
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    delta_w = delta_quat[:, 0]
    delta_x = delta_quat[:, 1]
    delta_y = delta_quat[:, 2]
    delta_z = delta_quat[:, 3]

    for i in range(4):
        dRt = torch.zeros(quat.shape[0], 3, 3, device=quat.device, dtype=quat.dtype)
        wrt = ["x", "y", "z", "w"][i]
        delta_t = [delta_x, delta_y, delta_z, delta_w][i]
        if wrt == "x":
            dRt[:, 0, 0] = 2 * x
            dRt[:, 0, 1] = 2 * y
            dRt[:, 0, 2] = 2 * z
            dRt[:, 1, 0] = 2 * y
            dRt[:, 1, 1] = -2 * x
            dRt[:, 1, 2] = -2 * w
            dRt[:, 2, 0] = 2 * z
            dRt[:, 2, 1] = 2 * w
            dRt[:, 2, 2] = -2 * x
        if wrt == "y":
            dRt[:, 0, 0] = -2 * y
            dRt[:, 0, 1] = 2 * x
            dRt[:, 0, 2] = 2 * w
            dRt[:, 1, 0] = 2 * x
            dRt[:, 1, 1] = 2 * y
            dRt[:, 1, 2] = 2 * z
            dRt[:, 2, 0] = -2 * w
            dRt[:, 2, 1] = 2 * z
            dRt[:, 2, 2] = -2 * y
        if wrt == "z":
            dRt[:, 0, 0] = -2 * z
            dRt[:, 0, 1] = -2 * w
            dRt[:, 0, 2] = 2 * x
            dRt[:, 1, 0] = 2 * w
            dRt[:, 1, 1] = -2 * z
            dRt[:, 1, 2] = 2 * y
            dRt[:, 2, 0] = 2 * x
            dRt[:, 2, 1] = 2 * y
            dRt[:, 2, 2] = 2 * z
        if wrt == "w":
            dRt[:, 0, 0] = 2 * w
            dRt[:, 0, 1] = -2 * z
            dRt[:, 0, 2] = 2 * y
            dRt[:, 1, 0] = 2 * z
            dRt[:, 1, 1] = 2 * w
            dRt[:, 1, 2] = -2 * x
            dRt[:, 2, 0] = -2 * y
            dRt[:, 2, 1] = 2 * x
            dRt[:, 2, 2] = 2 * w
        dR += delta_t * dRt
    return dR

def quat_to_d2rot_da2(quat, delta_quat):
    d2R = torch.zeros(quat.shape[0], 3, 3, device=quat.device, dtype=quat.dtype)
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    delta_w = delta_quat[:, 0]
    delta_x = delta_quat[:, 1]
    delta_y = delta_quat[:, 2]
    delta_z = delta_quat[:, 3]
    for i in range(4):
        d2Rt = torch.zeros(quat.shape[0], 3, 3, device=quat.device, dtype=quat.dtype)
        wrt = ["x", "y", "z", "w"][i]
        delta_t = [delta_x, delta_y, delta_z, delta_w][i]
        if wrt == "x":
            d2Rt[:, 0, 0] = 2 * delta_x
            d2Rt[:, 0, 1] = 2 * delta_y
            d2Rt[:, 0, 2] = 2 * delta_z
            d2Rt[:, 1, 0] = 2 * delta_y
            d2Rt[:, 1, 1] = -2 * delta_x
            d2Rt[:, 1, 2] = -2 * delta_w
            d2Rt[:, 2, 0] = 2 * delta_z
            d2Rt[:, 2, 1] = 2 * delta_w
            d2Rt[:, 2, 2] = -2 * delta_x
        if wrt == "y":
            d2Rt[:, 0, 0] = -2 * delta_y
            d2Rt[:, 0, 1] = 2 * delta_x
            d2Rt[:, 0, 2] = 2 * delta_w
            d2Rt[:, 1, 0] = 2 * delta_x
            d2Rt[:, 1, 1] = 2 * delta_y
            d2Rt[:, 1, 2] = 2 * delta_z
            d2Rt[:, 2, 0] = -2 * delta_w
            d2Rt[:, 2, 1] = 2 * delta_z
            d2Rt[:, 2, 2] = -2 * delta_y
        if wrt == "z":
            d2Rt[:, 0, 0] = -2 * delta_z
            d2Rt[:, 0, 1] = -2 * delta_w
            d2Rt[:, 0, 2] = 2 * delta_x
            d2Rt[:, 1, 0] = 2 * delta_w
            d2Rt[:, 1, 1] = -2 * delta_z
            d2Rt[:, 1, 2] = 2 * delta_y
            d2Rt[:, 2, 0] = 2 * delta_x
            d2Rt[:, 2, 1] = 2 * delta_y
            d2Rt[:, 2, 2] = 2 * delta_z
        if wrt == "w":
            d2Rt[:, 0, 0] = 2 * delta_w
            d2Rt[:, 0, 1] = -2 * delta_z
            d2Rt[:, 0, 2] = 2 * delta_y
            d2Rt[:, 1, 0] = 2 * delta_z
            d2Rt[:, 1, 1] = 2 * delta_w
            d2Rt[:, 1, 2] = -2 * delta_x
            d2Rt[:, 2, 0] = -2 * delta_y
            d2Rt[:, 2, 1] = 2 * delta_x
            d2Rt[:, 2, 2] = 2 * delta_w
        d2R += delta_t * d2Rt
    return d2R
    
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def compute_trace_delta(S, Q_tilde, dQ_tilde):
    quat = rotation_activation(Q_tilde)
    quat_new = rotation_activation(Q_tilde + dQ_tilde)

    R = quat_to_rot(quat, normalize=True).transpose(1, 2)
    R_new = quat_to_rot(quat_new, normalize=True).transpose(1, 2)
    R_delta = R.transpose(1, 2) @ R_new

    covar_delta = (S ** -2).unsqueeze(-1) * R_delta.transpose(1, 2) * (S ** 2).unsqueeze(1) @ R_delta
    trace_delta = torch.diagonal(covar_delta, dim1=1, dim2=2).sum(dim=1) - 3.0

    covar1 = build_covariance_from_scaling_rotation(S, 1.0, quat)
    covar_new1 = build_covariance_from_scaling_rotation(S, 1.0, quat_new)

    covar_delta1 = torch.linalg.solve(covar1, covar_new1)
    trace_delta1 = torch.diagonal(covar_delta1, dim1=1, dim2=2).sum(dim=1) - 3.0

    return trace_delta

torch.manual_seed(10115)

p = 1

# S = torch.exp(torch.randn(p, 3))
# Q_tilde = torch.randn(p, 4) 
# S = torch.tensor([[1.0000056, 10.60, 0.00000060]], dtype=torch.double)
# Q_tilde = torch.tensor([[0.01, -0.002,  0.000954,  -0.000356]], dtype=torch.double) 
# Q_tilde = torch.tensor([[206070.5938, 162166.0625, 130921.7109, 262986.6875]], dtype=torch.double)
S = torch.tensor([[0.0931, 0.0892, 0.0892]], dtype=torch.double)
Q_tilde = torch.tensor([[206070.5938, 162166.0625, 130921.7109, 262986.6875]], dtype=torch.double)
delta_Q_tilde = torch.tensor([[-0.4, 1.0, 0.1, 0.5]], dtype=torch.double)
delta_Q_tilde /= delta_Q_tilde.norm(dim=1, keepdim=True)
quat = rotation_activation(Q_tilde)
R = quat_to_rot(quat).transpose(1, 2)

R_tilde = quat_to_rot(Q_tilde, normalize=False).transpose(1, 2)

w = Q_tilde[:,0]
x = Q_tilde[:,1]
y = Q_tilde[:,2]
z = Q_tilde[:,3]
r = (x**2 + y**2 + z**2 + w**2).sqrt()

delta_w = delta_Q_tilde[:,0]
delta_x = delta_Q_tilde[:,1]
delta_y = delta_Q_tilde[:,2]
delta_z = delta_Q_tilde[:,3]

sum_t_delta_t = (w * delta_w + x * delta_x + y * delta_y + z * delta_z)

dr = sum_t_delta_t / r
d2r = 1 / r - (sum_t_delta_t ** 2) / (r ** 3)

dR_tilde = quat_to_drot_da(Q_tilde, delta_Q_tilde).transpose(1, 2)
d2R_tilde = quat_to_d2rot_da2(Q_tilde, delta_Q_tilde).transpose(1, 2)

G = torch.zeros(p, 3, 3, dtype=torch.double)
dG = R.transpose(1, 2) @ ((r ** -2) * dR_tilde - 2 * (r ** -3) * dr * R_tilde)
d2G = R.transpose(1, 2) @ (-2 * (r ** -3) * dr * dR_tilde + (r ** -2) * d2R_tilde + 6 * (r ** -4) * (dr ** 2) * R_tilde - 2 * (r** -3) * dr * dR_tilde - 2 * (r ** -3) * d2r * R_tilde)

SdGSinv = S.unsqueeze(-1) * dG * (S ** -1).unsqueeze(1)
coeff = 2 * ((SdGSinv * SdGSinv).sum(dim=(1,2)) + torch.diagonal(d2G, dim1=1, dim2=2).sum(dim=1))

trace_delta_list = []
trace_delta_est_list = []

fig = plt.figure(figsize=(12, 8))


max_stepsize = 0.1 * r.item()
stepsize = max_stepsize / 50.0
for i in range(-50, 50, 1):
    a = i * stepsize
    dQ_tilde = a * delta_Q_tilde

    trace_delta = compute_trace_delta(S, Q_tilde, dQ_tilde)
    trace_delta_est = 0.5 * coeff * (a ** 2)

    trace_delta_list.append(trace_delta[0].cpu().numpy())
    trace_delta_est_list.append(trace_delta_est[0].cpu().numpy())


plt.plot(np.arange(-50, 50, 1) * stepsize, trace_delta_list, label='Actual Trace Delta')
plt.plot(np.arange(-50, 50, 1) * stepsize, trace_delta_est_list, label='Estimated Trace Delta')
plt.xlabel('Step Size (a)')
plt.ylabel('Trace Delta')
plt.title('Trace Delta vs Step Size')
plt.legend()

plt.savefig('figures/trace_delta_vs_stepsize_multivariate.png')


