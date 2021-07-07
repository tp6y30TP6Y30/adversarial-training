import torch
import matplotlib.pyplot as plt

def projected_gradient_descent(model, x, y, loss_fn, num_steps = 2, step_size = 0.5, step_norm = 2, eps = 0.03, eps_norm = 2,
                               clamp=(0,1), y_target=None, visualize=False):
    """Performs the projected gradient descent attack on a batch of images."""
    x_adv = x.clone().detach().requires_grad_(True).to(x.device)
    targeted = y_target is not None
    num_channels = x.shape[1]

    pred = model(x_adv)
    loss = loss_fn(pred, y)
    loss.backward()

    with torch.no_grad():
        # Force the gradient step to be a fixed size in a certain norm
        if step_norm == 'inf':
            gradients = x_adv.grad.sign() * step_size
        else:
            # Note .view() assumes batched image data as 4D tensor
            grad_norm = x_adv.grad.view(x_adv.shape[0], x_adv.shape[1], -1).norm(step_norm, dim=-1).view(x_adv.shape[0], x_adv.shape[1], 1, 1)
            gradients = x_adv.grad * step_size / grad_norm

        if targeted:
            # Targeted: Gradient descent with on the loss of the (incorrect) target label
            # w.r.t. the image data
            x_adv -= gradients
        else:
            # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
            # the model parameters
            x_adv += gradients

    # Project back into l_norm ball and correct range
    if eps_norm == 'inf':
        # Workaround as PyTorch doesn't have elementwise clip
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
    else:
        delta = x_adv - x

        # Assume x and x_adv are batched tensors where the first dimension is
        # a batch dimension
        mask = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1) <= eps

        scaling_factor = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1)
        scaling_factor[mask] = eps

        # .view() assumes batched images as a 4D Tensor
        delta *= eps / scaling_factor.view(-1, 1, 1, 1)

        x_adv = x + delta
        
    x_adv = x_adv.clamp(*clamp)
    return loss, x_adv.detach()