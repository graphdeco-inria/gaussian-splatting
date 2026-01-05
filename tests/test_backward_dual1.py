import torch

class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        ctx.save_for_forward(x)
        return x**2

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return 2 * x * grad_output

    @staticmethod
    def jvp(ctx, x_tangent):
        (x,) = ctx.saved_tensors
        return 2 * x * x_tangent


x = torch.tensor([2.0, 3.0], requires_grad=True)
v = torch.tensor([1.0, -1.0])

with torch.autograd.forward_ad.dual_level():
    x_dual = torch.autograd.forward_ad.make_dual(x, v)
    y_dual = MyFunc.apply(x_dual)
    z_dual = y_dual.sum()
    z, jvp = torch.autograd.forward_ad.unpack_dual(z_dual)
    import code; code.interact(local=dict(globals(), **locals()))

jvp.backward()
print("Hv =", x.grad)

