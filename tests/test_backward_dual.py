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

def F(x):

    jvp_flag = has_tangent(x)

    if not jvp_flag:
        return _F.apply(x)
    else:
        x_tangent = get_tangent(x)
        return _F.apply(x, jvp_flag, x_tangent)

class _F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, jvp_flag=False, x_tangent=None):
        if not jvp_flag:
            res = x ** 4
        else:
            res = x ** 4
            res_tangent = 4 * (x ** 3) * x_tangent
            ctx.save_for_forward(x, res_tangent)

        ctx.save_for_backward(x)
        return res

    @staticmethod
    def jvp(ctx, grad_x, grad_jvp_flag, grad_x_tangent):
        x, res_tangent, = ctx.saved_tensors
        return res_tangent

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return 4 * (x ** 3) * grad_output, None, None

class HessianFunc:
    """
    Test class to compute Hessian-vector products using forward-mode AD. 
    """

    def __init__(self):
        pass

    def forward(self, x, x_tangent):
        self.x = x
        self.x_tangent = x_tangent
        self.x.requires_grad_(True)
        self.x_tangent.requires_grad_(True)

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, x_tangent)
            y_dual = x_dual * 2

            # Checkpoint y_dual
            y_primal, y_tangent = fwAD.unpack_dual(y_dual)
            self.y_primal = y_primal.detach().clone()
            self.y_tangent = y_tangent.detach().clone()
            self.y_primal.requires_grad_(True)
            self.y_tangent.requires_grad_(True)
            self.y_dual = fwAD.make_dual(self.y_primal, self.y_tangent)

            z_dual = F(self.y_dual)

            # Checkpoint z_dual
            z_primal, z_tangent = fwAD.unpack_dual(z_dual)
            self.z_primal = z_primal.detach().clone()
            self.z_tangent = z_tangent.detach().clone()
            self.z_primal.requires_grad_(True)
            self.z_tangent.requires_grad_(True)
            self.z_dual = fwAD.make_dual(self.z_primal, self.z_tangent)

            t_dual = (self.z_dual ** 0.5).sum()
            self.t_primal, self.t_tangent = fwAD.unpack_dual(t_dual)

            import code; code.interact(local=dict(globals(), **locals()))

        return self.t_primal, self.t_tangent

    def zero_grad(self):
        self.x.grad = None
        self.x_tangent.grad = None
        self.y_primal.grad = None
        self.y_tangent.grad = None
        self.z_primal.grad = None
        self.z_tangent.grad = None

    def hvp(self):
        self.t_tangent.backward()
        import code; code.interact(local=dict(globals(), **locals()))

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
x_tangent = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)

hess_func = HessianFunc()
t_primal, t_tangent = hess_func.forward(x, x_tangent)

print("t = ", t_primal, t_tangent)

hess_func.hvp()

import code; code.interact(local=dict(globals(), **locals()))
