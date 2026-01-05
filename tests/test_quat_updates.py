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

def quat_to_drot(quat, wrt="x", normalize=True):
    dR = torch.zeros(quat.shape[0], 3, 3, device=quat.device, dtype=quat.dtype)
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    q = (x**2 + y**2 + z**2 + w**2).sqrt()
    if wrt == "x":
        dR[:, 0, 0] = 2 * x
        dR[:, 0, 1] = 2 * y
        dR[:, 0, 2] = 2 * z
        dR[:, 1, 0] = 2 * y
        dR[:, 1, 1] = -2 * x
        dR[:, 1, 2] = -2 * w
        dR[:, 2, 0] = 2 * z
        dR[:, 2, 1] = 2 * w
        dR[:, 2, 2] = -2 * x
    if wrt == "y":
        dR[:, 0, 0] = -2 * y
        dR[:, 0, 1] = 2 * x
        dR[:, 0, 2] = 2 * w
        dR[:, 1, 0] = 2 * x
        dR[:, 1, 1] = 2 * y
        dR[:, 1, 2] = 2 * z
        dR[:, 2, 0] = -2 * w
        dR[:, 2, 1] = 2 * z
        dR[:, 2, 2] = -2 * y
    if wrt == "z":
        dR[:, 0, 0] = -2 * z
        dR[:, 0, 1] = -2 * w
        dR[:, 0, 2] = 2 * x
        dR[:, 1, 0] = 2 * w
        dR[:, 1, 1] = -2 * z
        dR[:, 1, 2] = 2 * y
        dR[:, 2, 0] = 2 * x
        dR[:, 2, 1] = 2 * y
        dR[:, 2, 2] = 2 * z
    if wrt == "w":
        dR[:, 0, 0] = 2 * w
        dR[:, 0, 1] = -2 * z
        dR[:, 0, 2] = 2 * y
        dR[:, 1, 0] = 2 * z
        dR[:, 1, 1] = 2 * w
        dR[:, 1, 2] = -2 * x
        dR[:, 2, 0] = -2 * y
        dR[:, 2, 1] = 2 * x
        dR[:, 2, 2] = 2 * w
    if normalize:
        dR /= q.unsqueeze(-1).unsqueeze(-1)
    return dR

def quat_to_d2rot(quat, wrt="x", normalize=True):
    d2R = torch.zeros(quat.shape[0], 3, 3, device=quat.device, dtype=quat.dtype)
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    q = (x**2 + y**2 + z**2 + w**2).sqrt()
    if wrt == "x":
        d2R[:, 0, 0] = 2
        d2R[:, 1, 1] = -2
        d2R[:, 2, 2] = -2
    if wrt == "y":
        d2R[:, 0, 0] = -2
        d2R[:, 1, 1] = 2
        d2R[:, 2, 2] = -2
    if wrt == "z":
        d2R[:, 0, 0] = -2
        d2R[:, 1, 1] = -2
        d2R[:, 2, 2] = 2
    if wrt == "w":
        d2R[:, 0, 0] = 2
        d2R[:, 1, 1] = 2
        d2R[:, 2, 2] = 2
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

# def compute_trace_delta(S, Q_tilde, dQ_tilde):
#     quat = rotation_activation(Q_tilde)
#     quat_new = rotation_activation(Q_tilde + dQ_tilde)
# 
#     covar = build_covariance_from_scaling_rotation(S, 1.0, quat)
#     covar_new = build_covariance_from_scaling_rotation(S, 1.0, quat_new)
# 
#     covar_delta = torch.linalg.solve(covar, covar_new)
#     trace_delta = torch.diagonal(covar_delta, dim1=1, dim2=2).sum(dim=1) - 3.0
# 
#     return trace_delta
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
# S = torch.tensor([[0.0056, 0.0060, 0.0060]])
# Q_tilde = torch.tensor([[0.9969, -0.0218,  0.0954,  0.0356]])
S = torch.tensor([[1.0000056, 10.60, 0.00000060]], dtype=torch.double)
# Q_tilde = torch.tensor([[0.01, -0.002,  0.000954,  -0.000356]], dtype=torch.double) 
Q_tilde = torch.tensor([[206070.5938, 162166.0625, 130921.7109, 262986.6875]], dtype=torch.double)
# S = torch.tensor([[0.0053, 0.0064, 0.0062]])
# Q_tilde = torch.tensor([[61.1971, 32.9973, 12.8884, 25.6220]])
quat = rotation_activation(Q_tilde)
R = quat_to_rot(quat).transpose(1, 2)

R_tilde = quat_to_rot(Q_tilde, normalize=False).transpose(1, 2)

w = Q_tilde[:,0]
x = Q_tilde[:,1]
y = Q_tilde[:,2]
z = Q_tilde[:,3]
q = (x**2 + y**2 + z**2 + w**2).sqrt()

G = torch.zeros(p, 3, 3)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.4, wspace=0.4)

coeffs = []

for j in range(4):
    param = ["w", "x", "y", "z"][j]
    t = [w, x, y, z][j]

    dR_tilde = quat_to_drot(Q_tilde, wrt=param, normalize=False).transpose(1, 2)
    d2R_tilde = quat_to_d2rot(Q_tilde, wrt=param, normalize=False).transpose(1, 2)


    dG = R.transpose(1, 2) @ ((q ** -2) * dR_tilde - 2 * t * (q ** -4) * R_tilde)
    d2G = R.transpose(1, 2) @ (-2 * t * (q ** -4) * dR_tilde + ((q ** -2) * d2R_tilde + 8 * (t ** 2) * (q ** -6) * R_tilde - 2 * t * (q ** -4) * dR_tilde - 2 * (q ** -4) * R_tilde))

    SdGSinv = S.unsqueeze(-1) * dG * (S ** -1).unsqueeze(1)
    coeff = 2 * ((SdGSinv * SdGSinv).sum(dim=(1,2)) + torch.diagonal(d2G, dim1=1, dim2=2).sum(dim=1))
    coeffs.append(coeff)
    # import code; code.interact(local=dict(globals(), **locals()), banner="Trace Delta Analysis")

    trace_delta_list = []
    trace_delta_est_list = []

    max_stepsize = 0.1 * q.item()
    stepsize = max_stepsize / 50.0
    for i in range(-50, 50, 1):
        dQ_tilde = torch.zeros(p, 4)
        dQ_tilde[:, j] = i * stepsize
        trace_delta = compute_trace_delta(S, Q_tilde, dQ_tilde)
        trace_delta_est = 0.5 * coeff * (i * stepsize) ** 2

        # import code; code.interact(local=locals(), banner="Trace Delta Analysis loop")

        trace_delta_list.append(trace_delta[0].cpu().numpy())
        trace_delta_est_list.append(trace_delta_est[0].cpu().numpy())

    ax = [ax1, ax2, ax3, ax4][j]
    ax.plot(np.arange(-50, 50, 1) * stepsize, trace_delta_list, label="Exact")
    ax.plot(np.arange(-50, 50, 1) * stepsize, trace_delta_est_list, label="Est")
    ax.set_title(f"Perturbation in {param}")
    ax.legend()


plt.savefig("figures/trace_delta_vs_dQ_tilde_all_params.png")

exit()

dGx = R.transpose(1, 2) @ ((q ** -2) * dRx_tilde - 2 * x * (q ** -4) * R_tilde)
d2Gx = R.transpose(1, 2) @ (-2 * x * (q ** -4) * dRx_tilde + ((q ** -2) * d2Rx_tilde + 8 * (x ** 2) * (q ** -6) * R_tilde - 2 * x * (q ** -4) * dRx_tilde - 2 * (q ** -4) * R_tilde))

SdGSinv = S.unsqueeze(-1) * dGx * (S ** -1).unsqueeze(1)

coeff = (SdGSinv * SdGSinv).sum(dim=(1,2)) + torch.diagonal(d2Gx, dim1=1, dim2=2).sum(dim=1)

trace_delta_list = []
trace_delta_est_list = []

stepsize = 0.01
for i in range(-50, 50, 1):
    dQx_tilde = torch.zeros(p, 4)
    dQx_tilde[:, 1] = i * stepsize
    trace_delta = compute_trace_delta(S, Q_tilde, dQx_tilde)
    trace_delta_est = coeff * (i * stepsize) ** 2

    trace_delta_list.append(trace_delta[0].cpu().numpy())
    trace_delta_est_list.append(trace_delta_est[0].cpu().numpy())

    quat_new = rotation_activation(Q_tilde + dQx_tilde)
    R_new = quat_to_rot(quat_new).transpose(1, 2)
    G_ref = R.transpose(1, 2) @ R_new - torch.eye(3)
    G_est = G + dGx * (i * stepsize) + 0.5 * d2Gx * (i * stepsize) ** 2

    # import code; code.interact(local=dict(globals(), **locals()), banner="Trace Delta Analysis loop")

fig, ax = plt.subplots()
ax.plot(np.arange(-50, 50, 1) * stepsize, trace_delta_list, label="Exact")
ax.plot(np.arange(-50, 50, 1) * stepsize, trace_delta_est_list, label="Est")

ax.legend()
ax.set_xlabel("dQx_tilde")
ax.set_ylabel("Trace Delta")
plt.savefig("figures/trace_delta_vs_dQx_tilde.png")

import code; code.interact(local=dict(globals(), **locals()), banner="Trace Delta Analysis")

torch.manual_seed(0)

S = torch.exp(torch.randn(1, 3, dtype=torch.double))
Q_tilde = torch.randn(1, 4, dtype=torch.double)
offset = torch.tensor([0.0, -0.1, 0.0, 0.0], dtype=torch.double)
Q_tilde_new = Q_tilde + offset
quat = rotation_activation(Q_tilde)
quat_new = rotation_activation(Q_tilde_new)
R = quat_to_rot(quat).transpose(1, 2)
R_new = quat_to_rot(quat_new).transpose(1, 2)
R_delta = R.transpose(1, 2) @ R_new
dR = R_delta - torch.eye(3)

R1 = (S ** -2).unsqueeze(-1) * R_delta.transpose(1, 2) * (S ** 2).unsqueeze(1)
R2 = R1 @ R_delta

R11 = (1 + dR[:,0,0]) ** 2 + (S[:,0] ** (-2)) * (dR[:,1,0] ** 2) * (S[:,1] ** 2) + (S[:,0] ** (-2)) * (dR[:,2,0] ** 2) * (S[:,2] ** 2)
R22 = (S[:,1] ** (-2)) * (dR[:,0,1] ** 2) * (S[:,0] ** 2) + (1 + dR[:,1,1]) ** 2 + (S[:,1] ** (-2)) * (dR[:,2,1] ** 2) * (S[:,2] ** 2)
R33 = (S[:,2] ** (-2)) * (dR[:,0,2] ** 2) * (S[:,0] ** 2) + (S[:,2] ** (-2)) * (dR[:,1,2] ** 2) * (S[:,1] ** 2) + (1 + dR[:,2,2]) ** 2

dR11 = dR[:,0,0]
dR22 = dR[:,1,1]
dR33 = dR[:,2,2]

T11 = (1 + dR[:,0,0]) ** 2
T12 = (S[:,0] ** (-2)) * (dR[:,1,0] ** 2) * (S[:,1] ** 2)
T13 = (S[:,0] ** (-2)) * (dR[:,2,0] ** 2) * (S[:,2] ** 2)
T21 = (S[:,1] ** (-2)) * (dR[:,0,1] ** 2) * (S[:,0] ** 2)
T22 = (1 + dR[:,1,1]) ** 2
T23 = (S[:,1] ** (-2)) * (dR[:,2,1] ** 2) * (S[:,2] ** 2)
T31 = (S[:,2] ** (-2)) * (dR[:,0,2] ** 2) * (S[:,0] ** 2)
T32 = (S[:,2] ** (-2)) * (dR[:,1,2] ** 2) * (S[:,1] ** 2)
T33 = (1 + dR[:,2,2]) ** 2


x = Q_tilde[:,0]
y = Q_tilde[:,1]
z = Q_tilde[:,2]
w = Q_tilde[:,3]
q = (x**2 + y**2 + z**2 + w**2).sqrt()

Rx = torch.zeros_like(R)
Rx[:,0,0] = 0
Rx[:,0,1] = 2 * y
Rx[:,0,2] = 2 * z
Rx[:,1,0] = 2 * y
Rx[:,1,1] = -4 * x
Rx[:,1,2] = -2 * w
Rx[:,2,0] = 2 * z
Rx[:,2,1] = 2 * w
Rx[:,2,2] = -4 * x

RTRx = R * Rx
T1_est = 2 * RTRx.sum(dim=(1,2)) / (q ** 2) * (offset[0])

dRx = 2 * x / (q ** 4) * R + (1 / (q ** 2)) * Rx
SinvdRxS = (S ** -1).unsqueeze(-1) * dRx * (S ** 1).unsqueeze(1)

trace_delta_est = 2 * (dR11 + dR22 + dR33) + SinvdRxS.norm(dim=(1,2))**2 * (offset[0] ** 2)
trace_delta_est1 = T1_est + SinvdRxS.norm(dim=(1,2))**2 * (offset[0] ** 2)

trace_delta = compute_trace_delta(S, Q_tilde, offset)

import code; code.interact(local=dict(globals(), **locals()), banner="Trace Delta Analysis")
