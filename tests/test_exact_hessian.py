import torch
import torch.autograd.forward_ad as fwAD

class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_forward(x)
        ctx.save_for_backward(x)
        return x**2

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return 2 * x * grad_output

    @staticmethod
    def jvp(ctx, x_tangent):
        (x,) = ctx.saved_tensors
        # directional derivative of f(x) = x^2 is 2x * v
        return 2 * x * x_tangent
 
x = torch.tensor([2.0, 3.0], requires_grad=True)
v = torch.tensor([1.0, -1.0])

with fwAD.dual_level():
    x_dual = fwAD.make_dual(x, v)
    y_dual = MyFunc.apply(x_dual)  # uses your custom Function
    z_dual = y_dual.sum()  # scalar output
    z_primal, jvp = fwAD.unpack_dual(z_dual)

print("JVP =", jvp)  # ∇f(x) @ v

# Now backprop through jvp to get Hessian-vector product
Hv = torch.autograd.grad(jvp, x)[0]
print("Hv =", Hv)

