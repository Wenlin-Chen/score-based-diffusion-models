import torch
import numpy as np
from tqdm import tqdm
from scipy import integrate

def generation(
        score_model, 
        sample_shape,
        sampler, 
        sampler_params, 
        device,
        eps=1e-3,
    ):
    if sampler == "em":
        samples = Euler_Maruyama_sampler(
            score_model,
            forward_pdf_std=sampler_params["forward_pdf_std"],
            diffusion_coeff=sampler_params["diffusion_coeff"],
            sigma=sampler_params["sigma"],
            num_steps=sampler_params["num_steps"],
            batch_size=sample_shape[0],
            in_channels=sample_shape[1],
            height=sample_shape[2],
            width=sample_shape[3],
            device=device,
            eps=eps,
        )
    
    elif sampler == "pc":
        samples = predictor_corrector_sampler(
            score_model,
            forward_pdf_std=sampler_params["forward_pdf_std"],
            diffusion_coeff=sampler_params["diffusion_coeff"],
            sigma=sampler_params["sigma"],
            num_steps=sampler_params["num_steps"],
            snr=sampler_params["snr"],
            batch_size=sample_shape[0],
            in_channels=sample_shape[1],
            height=sample_shape[2],
            width=sample_shape[3],
            device=device,
            eps=eps,
        )

    elif sampler == "langevin":
        samples = langevin_sampler(
            score_model,
            forward_pdf_std=sampler_params["forward_pdf_std"],
            sigma=sampler_params["sigma"],
            num_steps=sampler_params["num_steps"],
            snr=sampler_params["snr"],
            batch_size=sample_shape[0],
            in_channels=sample_shape[1],
            height=sample_shape[2],
            width=sample_shape[3],
            device=device,
            eps=eps,
        )

    elif sampler == "ode":
        samples = ode_sampler(
            score_model,
            forward_pdf_std=sampler_params["forward_pdf_std"],
            diffusion_coeff=sampler_params["diffusion_coeff"],
            sigma=sampler_params["sigma"],
            rtol=sampler_params["rtol"],
            atol=sampler_params["atol"],
            batch_size=sample_shape[0],
            in_channels=sample_shape[1],
            height=sample_shape[2],
            width=sample_shape[3],
            device=device,
            eps=eps,
        )

    else:
        raise NotImplementedError

    samples = torch.clamp(samples, 0.0, 1.0)
    return samples

@torch.no_grad()
def Euler_Maruyama_sampler(
        score_model, 
        forward_pdf_std, 
        diffusion_coeff, 
        sigma,
        num_steps,
        batch_size,
        in_channels,
        height,
        width,
        device,
        eps=1e-3,
    ):
    # forward SDE: dx_t = \sigma^t*dw, 0<=t<=1
    # reverse SDE: dx_t = -\sigma^{2t}*\nabla_x \log p_t(x) dt + \sigma^t d\tilde{w}
    # p(x_1) ~= N(x_1| 0 , (\simga^2 - 1) / (2*\log(\sigma)) * I)
    # score model + discretization: x_{t-\delta_t} = x_t +\sigmq^{2t}*score*\delta_t + \sigma^t*\sqrt{\delta_t}z_t
    
    t_1 = torch.ones(batch_size, device=device)  # (B, )
    std_1 = forward_pdf_std(t_1, sigma)[:, None, None, None]  # (B, )
    x = torch.randn(batch_size, in_channels, height, width, device=device) * std_1  # (B, C_in, H, W)
    time_steps = torch.linspace(1.0, eps, num_steps, device=device)  # (T, )
    delta_t = time_steps[0] - time_steps[1]
    
    for t in tqdm(time_steps):
        batch_t = torch.ones(batch_size, device=device) * t  # (B, )
        g = diffusion_coeff(batch_t, sigma)[:, None, None, None]  # (B, 1, 1, 1)
        x_mean = x + g**2 * score_model(x, batch_t) * delta_t  # (B, C_in, H, W)
        x = x_mean + g * torch.sqrt(delta_t) * torch.randn_like(x, device=device)  # (B, C_in, H, W)

    return x_mean


@torch.no_grad()
def predictor_corrector_sampler(
        score_model, 
        forward_pdf_std, 
        diffusion_coeff,
        sigma,
        num_steps,
        snr,
        batch_size,
        in_channels,
        height,
        width,
        device,
        eps=1e-3,
    ):
    # EM sampler (predict) + Langevin sampler (correct)

    t_1 = torch.ones(batch_size, device=device)  # (B, )
    std_1 = forward_pdf_std(t_1, sigma)[:, None, None, None]  # (B, )
    x = torch.randn(batch_size, in_channels, height, width, device=device) * std_1  # (B, C_in, H, W)
    time_steps = torch.linspace(1.0, eps, num_steps, device=device)  # (T, )
    delta_t = time_steps[0] - time_steps[1]

    for t in tqdm(time_steps):
        batch_t = torch.ones(batch_size, device=device) * t  # (B, )

        # Langevin sampler (correct)
        grad = score_model(x, batch_t)  # (B, C_in, H, W)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = np.sqrt(np.prod(x.shape[1:]))
        lg_step_size = 2 * (snr * noise_norm / grad_norm)**2
        x = x + lg_step_size * grad + torch.sqrt(2.0 * lg_step_size) * torch.randn_like(x, device=device)

        # EM sampler (predict)
        g = diffusion_coeff(batch_t, sigma)[:, None, None, None]  # (B, 1, 1, 1)
        x_mean = x + g**2 * score_model(x, batch_t) * delta_t  # (B, C_in, H, W)
        x = x_mean + g * torch.sqrt(delta_t) * torch.randn_like(x, device=device)  # (B, C_in, H, W)

    return x_mean


@torch.no_grad()
def langevin_sampler(
        score_model, 
        forward_pdf_std, 
        sigma,
        num_steps,
        snr,
        batch_size,
        in_channels,
        height,
        width,
        device,
        eps=1e-3,
    ):
    # Langevin sampler

    t_1 = torch.ones(batch_size, device=device)  # (B, )
    std_1 = forward_pdf_std(t_1, sigma)[:, None, None, None]  # (B, )
    x = torch.randn(batch_size, in_channels, height, width, device=device) * std_1  # (B, C_in, H, W)
    time_steps = torch.linspace(1.0, eps, num_steps, device=device)  # (T, )

    for t in tqdm(time_steps):
        batch_t = torch.ones(batch_size, device=device) * t  # (B, )

        grad = score_model(x, batch_t)  # (B, C_in, H, W)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = np.sqrt(np.prod(x.shape[1:]))
        lg_step_size = 2 * (snr * noise_norm / grad_norm)**2
        x_mean = x + lg_step_size * grad 
        x = x_mean + torch.sqrt(2.0 * lg_step_size) * torch.randn_like(x, device=device)

    return x_mean


def ode_sampler(
        score_model, 
        forward_pdf_std, 
        diffusion_coeff,
        sigma,
        atol,
        rtol,
        batch_size,
        in_channels,
        height,
        width,
        device,
        eps=1e-3,
    ):
    # probability flow ODE sampler
    # dx = -0.5 * \sigma^{2t} * score * dt

    t_1 = torch.ones(batch_size, device=device)  # (B, )
    std_1 = forward_pdf_std(t_1, sigma)[:, None, None, None]  # (B, )
    x = torch.randn(batch_size, in_channels, height, width, device=device) * std_1  # (B, C_in, H, W)
    x_shape = x.shape

    @torch.no_grad()
    def eval_score(x, batch_t):
        x = torch.tensor(x, dtype=torch.float32, device=device).reshape(x_shape)  # (B, C_in, H, W)
        score = score_model(x, batch_t)
        return score

    def ode_func(t, x):
        batch_t = torch.ones(batch_size, device=device) * t  # (B, )
        g = diffusion_coeff(batch_t, sigma)[:, None, None, None]
        ode = - 0.5 * (g**2) * eval_score(x, batch_t)
        return ode.reshape(-1).cpu().numpy()


    res = integrate.solve_ivp(ode_func, (1.0, eps), x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method="RK45")
    print("Number of function evaluations: {}".format(res.nfev))
    x = torch.tensor(res.y[:, -1], device=device).reshape(x_shape)

    return x