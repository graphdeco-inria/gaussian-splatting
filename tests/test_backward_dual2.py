import torch
import torch.autograd.forward_ad as fwAD

def has_tangent(x):
    if isinstance(x, float):
        return False
    elif isinstance(x, list):
        return any(has_tangent(xi) for xi in x)
    return fwAD.unpack_dual(x).tangent is not None

def get_tangent(x):
    if isinstance(x, float):
        return 0.0
    elif isinstance(x, list):
        return [get_tangent(xi) for xi in x]
    elif has_tangent(x):
        return fwAD.unpack_dual(x).tangent
    elif isinstance(x, torch.Tensor):
        return torch.zeros_like(x)
    else:
        raise ValueError(f"Unsupported type for tangent extraction: {type(x)}")

def _F_forward(x_primal, x_tangent):

    m = x_primal.shape[0]

    res_primal = torch.zeros((m + 1), dtype=x_primal.dtype, device=x_primal.device)
    res_tangent = torch.zeros((m + 1), dtype=x_primal.dtype, device=x_primal.device)

    for i in range(m + 1):
        res_primal[i] = x_primal[0] * (x_primal[1] ** (m - i)) * (x_primal[2] ** (i + 1))
        res_tangent[i] = x_tangent[0] * (x_primal[1] ** (m - i)) * (x_primal[2] ** (i + 1)) \
                         + x_tangent[1] * (m - i) * x_primal[0] * (x_primal[1] ** (m - i - 1)) * (x_primal[2] ** (i + 1)) \
                         + x_tangent[2] * (i + 1) * x_primal[0] * (x_primal[1] ** (m - i)) * (x_primal[2] ** i)

    import code; code.interact(local=dict(globals(), **locals()), banner="In _F_forward after primal/tangent computation")


    return res_primal, res_tangent


def _F_backward(x_primal, grad_primal, x_tangent, grad_tangent):
    """
    Simulates a custom backward pass
    x_primal: The point at which the gradient and Hessian are evaluated
    grad_primal: The incoming gradient from the backward pass, i.e., ∇_F L
    x_tangent: The JVP from the previous stage
    grad_tangent: The HVP from the previous stage, i.e., ∇_F^2 L v
    """
    m = x_primal.shape[0]
    res_primal = torch.zeros((3), dtype=x_primal.dtype, device=x_primal.device)
    res_tangent = torch.zeros((3), dtype=x_primal.dtype, device=x_primal.device)

    with fwAD.dual_level():
        x_dual = fwAD.make_dual(x_primal, x_tangent)
        # Note: There is a weird thing where we need to clone the gradients here
        grad_dual = fwAD.make_dual(grad_primal.clone(), grad_tangent.clone())
        res_dual = torch.zeros((3), dtype=x_primal.dtype, device=x_primal.device)

        for i in range(m + 1):
            res_dual[0] += grad_dual[i] * (x_dual[1] ** (m + 1 - i)) * (x_dual[2] ** (i + 1))
            res_dual[1] += grad_dual[i] * ((m + 1 - i) * x_dual[0] * (x_dual[1] ** (m - i)) * (x_dual[2] ** (i + 1)))
            res_dual[2] += grad_dual[i] * ((i + 1) * x_dual[0] * (x_dual[1] ** (m + 1 - i)) * (x_dual[2] ** i))

        res_primal, res_tangent = fwAD.unpack_dual(res_dual)


    return res_primal, res_tangent

class _F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_primal, x_tangent):
        res_primal, res_tangent = _F_forward(x_primal, x_tangent)
        ctx.save_for_backward(x_primal, x_tangent)
        return res_primal, res_tangent

    @staticmethod
    def backward(ctx, grad, grad_tangent):
        """
        When taking backward pass on the JVP
        grad is ∇_F^2 Lv, where L is the scalar loss function, v is res_tangent
        grad_tangent is ∇_F L
        res_primal is the gradient, res_tangent is the Hessian-vector product
        But in the return format, we need to return HVP, gradient
        """

        x_primal, x_tangent, = ctx.saved_tensors
        import code; code.interact(local=dict(globals(), **locals()), banner="In _F backward")
        res_primal, res_tangent = _F_backward(x_primal, grad_tangent, x_tangent, grad)
        return res_tangent, res_primal


"""
L = x_1 x_2^4 x_3^1 + x_1 x_2^3 x_3^2 + x_1 x_2^2 x_3^3 + x_1 x_2^1 x_3^4
"""   
x = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)

L_ref = x[0] * (x[1] ** 4) * (x[2] ** 1) + x[0] * (x[1] ** 3) * (x[2] ** 2) + x[0] * (x[1] ** 2) * (x[2] ** 3) + x[0] * (x[1] ** 1) * (x[2] ** 4)

J_ref = torch.zeros((4, 3), dtype=x.dtype, device=x.device)
for i in range(4):
    J_ref[i, 0] = (x[1] ** (4 - i)) * (x[2] ** (i + 1))
    J_ref[i, 1] = (4 - i) * x[0] * (x[1] ** (3 - i)) * (x[2] ** (i + 1))
    J_ref[i, 2] = (i + 1) * x[0] * (x[1] ** (4 - i)) * (x[2] ** i)

H_ref = torch.zeros((3, 3), dtype=x.dtype, device=x.device)
H_ref[0, 0] = 0
H_ref[1, 0] = 4 * (x[1] ** 3) * (x[2] ** 1) + 3 * (x[1] ** 2) * (x[2] ** 2) + 2 * (x[1] ** 1) * (x[2] ** 3) + 1 * (x[1] ** 0) * (x[2] ** 4)
H_ref[2, 0] = 1 * (x[1] ** 4) * (x[2] ** 0) + 2 * (x[1] ** 3) * (x[2] ** 1) + 3 * (x[1] ** 2) * (x[2] ** 2) + 4 * (x[1] ** 1) * (x[2] ** 3)
H_ref[0, 1] = H_ref[1, 0]
H_ref[1, 1] = 12 * x[0] * (x[1] ** 2) * (x[2] ** 1) + 6 * x[0] * (x[1] ** 1) * (x[2] ** 2) + 2 * x[0] * (x[1] ** 0) * (x[2] ** 3)
H_ref[2, 1] = 4 * x[0] * (x[1] ** 3) * (x[2] ** 0) + 6 * x[0] * (x[1] ** 2) * (x[2] ** 1) + 6 * x[0] * (x[1] ** 1) * (x[2] ** 2) + 4 * x[0] * (x[1] ** 0) * (x[2] ** 3)
H_ref[0, 2] = H_ref[2, 0]
H_ref[1, 2] = H_ref[2, 1]
H_ref[2, 2] = 2 * x[0] * (x[1] ** 3) * (x[2] ** 0) + 6 * x[0] * (x[1] ** 2) * (x[2] ** 1) + 12 * x[0] * (x[1] ** 1) * (x[2] ** 2)

for i in range(3):
    x_tangent = torch.zeros_like(x)
    x_tangent[i] = 1.0
    x_tangent.requires_grad = True

    x.grad = None
    x_tangent.grad = None

    with fwAD.dual_level():
        x_primal = x
        z_primal, z_tangent = _F.apply(x_primal, x_tangent)

        import code; code.interact(local=dict(globals(), **locals()), banner="After _F.apply")

        z_dual = fwAD.make_dual(z_primal, z_tangent)
        t_dual = z_dual.sum()

        t_primal, t_tangent = fwAD.unpack_dual(t_dual)

    t_tangent.backward()

    Hv = x.grad
    g = x_tangent.grad

    import code; code.interact(local=dict(globals(), **locals()), banner="After Hv computation")

Hv_expected = 8 * x_tangent
g_expected = 8 * x

assert torch.allclose(Hv, Hv_expected)
assert torch.allclose(g, g_expected)

print("Test passed.")



