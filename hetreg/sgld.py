import math
import numpy as np
import torch
import wandb
import torch.nn.utils as utils
from torch.optim.optimizer import Optimizer, required


def valid_performance(
    model,
    loader, 
    criterion, 
    device,
    head='gaussian', 
    head_activation='softplus', 
    likelihood='heteroscedastic' 
):
    """
    Calculates performance (MSE based on model output f[:,0]) and
    Negative Log-Likelihood (NLL) on a validation or test set.
    """
    model.eval() 
    N = len(loader.dataset)
    if N == 0:
        return 0.0, 0.0 

    total_mse_num = 0.0 
    total_nll_num = 0.0 
    nan_detected = False

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)

            try:
                f = model(X)

                if not torch.isfinite(f).all():
                    print(f"Warning: NaN or Inf detected in model output f during validation (Batch {batch_idx}).")
                    nan_detected = True
                    break 

                try:
                    mu_for_mse = f[:, 0].view_as(y) # Ensure shapes match
                    batch_mse_sum = ((mu_for_mse - y)**2).sum().item()
                except IndexError:
                     if likelihood == 'homoscedastic' and f.ndim >= 1:
                         mu_for_mse = f.view_as(y) # Assume f is mu directly
                         batch_mse_sum = ((mu_for_mse - y)**2).sum().item()
                     else:
                         print(f"Warning: Could not extract mean for MSE calculation from model output shape {f.shape} (Batch {batch_idx}). Skipping MSE calculation for batch.")
                         batch_mse_sum = np.nan 


                nll_per_sample = criterion(f, y, head=head, head_activation=head_activation, likelihood=likelihood)

                if not torch.isfinite(nll_per_sample).all():
                    print(f"Warning: NaN or Inf detected in NLL calculation during validation (Batch {batch_idx}).")
                    nan_detected = True
                    break 

                batch_nll_sum = nll_per_sample.sum().item()

                if not np.isnan(batch_mse_sum): # Avoid adding NaN MSE
                    total_mse_num += batch_mse_sum
                else:
                    total_mse_num = np.nan # Mark final MSE as NaN

                total_nll_num += batch_nll_sum

            except Exception as e:
                import traceback
                print(f"Error during validation processing (Batch {batch_idx}): {e}")
                nan_detected = True
                break

    if nan_detected:
        final_mse = np.nan
        final_nll = np.inf 
    else:
        if N > 0:
            final_mse = total_mse_num / N
            final_nll = total_nll_num / N
        else:
            final_mse = 0.0
            final_nll = 0.0

        if not np.isfinite(final_mse):
            final_mse = np.nan
        if not np.isfinite(final_nll):
            final_nll = np.inf

    return final_mse, final_nll


class pSGLD(Optimizer):
    """
    Pre-conditioned Stochastic Gradient Langevin Dynamics (Li et al., 2016).
    RMSProp–style running 2nd moment gives a per-parameter preconditioner.
    """
    def __init__(self, params, lr=required, norm_sigma=0.0,
                 precond_decay=0.99, eps=1e-8, addnoise=True):

        weight_decay = 1.0 / (norm_sigma ** 2) if norm_sigma != 0 else 0.0
        if lr is not required and lr <= 0:
            raise ValueError("lr must be positive")
        defaults = dict(lr=lr, weight_decay=weight_decay,
                        precond_decay=precond_decay, eps=eps,
                        addnoise=addnoise)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr   = group['lr']
            wd   = group['weight_decay']
            rho  = group['precond_decay']
            eps  = group['eps']
            noise_flag = group['addnoise']

            for p in group['params']:
                if p.grad is None: continue
                g = p.grad

                # --- 1st & 2nd moment (RMSProp style) --------------------
                state = self.state[p]
                if not state:                       # init
                    state['square_avg'] = torch.zeros_like(p)

                square_avg = state['square_avg']
                square_avg.mul_(rho).addcmul_(g, g, value=1.0-rho)

                precond = 1.0 / (square_avg.sqrt() + eps)  # element-wise

                # Weight-decay term (Gaussian prior)
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                # Langevin update
                if noise_flag:
                    noise = torch.randn_like(p) * torch.sqrt(precond)
                else:
                    noise = 0.0

                p.add_(-0.5*lr * precond * g + math.sqrt(lr) * noise)


class SGLD(Optimizer):
    """
    Stochastic Gradient Langevin Dynamics optimizer.
    """
    def __init__(self, params, lr=required, norm_sigma=0, addnoise=True):
        # prior precision lambda = 1 / sigma^2
        weight_decay = 1.0 / (norm_sigma ** 2) if norm_sigma != 0 else 0.0 # Handle sigma=0
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, weight_decay=weight_decay, addnoise=addnoise)
        super(SGLD, self).__init__(params, defaults)

    def step(self):
        loss = None 
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr = group['lr']
            addnoise = group['addnoise']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data 

                if weight_decay != 0:
                    
                    d_p.add_(p.data, alpha=weight_decay)

                if addnoise:
                    noise_term = torch.randn_like(p.data) / np.sqrt(lr) 
                    update = (0.5 * d_p + noise_term).mul_(-lr) 
                else:
                    update = (0.5 * d_p).mul_(-lr) 

                p.data.add_(update) 

        return loss

def sgld_optimization(
    model,
    train_loader,
    valid_loader=None,
    n_epochs=500,
    lr=1e-3,
    prior_prec_init=1.0,
    addnoise=True,
    use_wandb=False,
    max_grad_norm=1.0, 
    head='gaussian', 
    head_activation='softplus', 
    likelihood='heteroscedastic' 
):
    """
    Run SGLD training for a regression model.

    Returns:
        model, valid_perfs, valid_nlls
    """
    device = next(model.parameters()).device 
    N_train = len(train_loader.dataset)

    sigma = 1.0 / np.sqrt(prior_prec_init) if prior_prec_init > 0 else 0.0


    # optimizer = pSGLD(model.parameters(), lr=lr,
    #                     norm_sigma=sigma, addnoise=addnoise)

    optimizer = SGLD(model.parameters(), lr=lr, norm_sigma=sigma, addnoise=addnoise)

    valid_perfs = []
    valid_nlls = []
    training_failed = False 

    print(f"Starting SGLD training: epochs={n_epochs}, lr={lr}, prior_prec={prior_prec_init}, head={head}, head_act={head_activation}")

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        model.train()
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            # Forward pass
            f = model(X)

            if not torch.isfinite(f).all(): 
                print(f"NaN/Inf in RAW model output at epoch {epoch}, batch {batch_idx}")
                training_failed = True
                break # Exit batch loop

            f_stabilized = f 
            if f.shape[1] > 0: 
                 f[:, 0].clamp_(min=-20.0, max=20.0) 

            if likelihood == 'heteroscedastic':
                if head == 'natural':
                    f[:, 1].clamp_(max=-1e-6) 
                elif head == 'gaussian' and head_activation == 'softplus':
                    f[:, 1].clamp_(min=-20.0, max=30.0) 

                if not torch.isfinite(f).all():
                    print(f"NaN/Inf in model output AFTER ALL CLAMPING at epoch {epoch}, batch {batch_idx}")
                    training_failed = True
                    break 

            try:
                loss = gaussian_log_likelihood_loss(f, y, head=head, head_activation=head_activation, likelihood=likelihood).mean()

                if not torch.isfinite(loss): 
                    print(f"NaN/Inf LOSS detected at epoch {epoch}, batch {batch_idx}")
                    print(f"  Prior Precision: {prior_prec_init}")
                    print(f"  LR: {lr}")
                    print(f"  Input y shape: {y.shape}, example: {y.flatten()[0].item():.4f}, requires_grad: {y.requires_grad}")
                    print(f"  Input f_stabilized shape: {f_stabilized.shape}, example: {f_stabilized[0].tolist()}, requires_grad: {f_stabilized.requires_grad}")

                    mu_loss = f_stabilized[:, 0]
                    var_loss = torch.tensor(1.0) 
                    if likelihood == 'heteroscedastic':
                         if head == 'natural':
                             eta2 = f_stabilized[:, 1]
                             var_loss = -0.5 / eta2
                         elif head == 'gaussian' and head_activation == 'softplus':
                             v = f_stabilized[:, 1]
                             var_loss = torch.nn.functional.softplus(v) 
                         else: 
                             var_loss = f_stabilized[:, 1].clamp(min=0)

                         print(f"  Inside Loss - mu shape: {mu_loss.shape}, example: {mu_loss.flatten()[0].item():.4f}")
                         print(f"  Inside Loss - var (pre-eps) shape: {var_loss.shape}, example: {var_loss.flatten()[0].item():.4f}, min: {var_loss.min().item():.4g}, max: {var_loss.max().item():.4g}")
                         var_loss = var_loss.clamp(min=0) + 1e-8 # Add epsilon
                         var_loss = var_loss.clamp(min=1e-8, max=1e7) # Clamp after epsilon
                         print(f"  Inside Loss - var (post-eps/clamp) shape: {var_loss.shape}, example: {var_loss.flatten()[0].item():.4f}, min: {var_loss.min().item():.4g}, max: {var_loss.max().item():.4g}")
                         term1 = -0.5 * torch.log(2 * torch.pi * var_loss)
                         term2 = -0.5 * ((y.view_as(mu_loss) - mu_loss)**2 / var_loss)
                         print(f"  Inside Loss - Term1 (log): example: {term1.flatten()[0].item():.4f}, min: {term1.min().item():.4g}, max: {term1.max().item():.4g}")
                         print(f"  Inside Loss - Term2 (sq_err/var): example: {term2.flatten()[0].item():.4f}, min: {term2.min().item():.4g}, max: {term2.max().item():.4g}")
                    # --------------------------------
                    training_failed = True
                    break 

                loss.backward()

                # --- Gradient Clipping ---
                utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                # --- Optimizer Step ---
                optimizer.step()

                if not torch.isnan(loss): 
                     epoch_loss += loss.item()
                else:
                     epoch_loss = np.nan 

            except Exception as e:
                 print(f"Error during loss calculation or backward pass at epoch {epoch}, batch {batch_idx}: {e}")
                 import traceback
                 traceback.print_exc() 
                 training_failed = True
                 break 

        if training_failed:
            print(f"Stopping training at epoch {epoch} due to NaN/Inf detected.")
            if valid_loader is not None:
                 valid_perfs.append(np.nan)
                 valid_nlls.append(np.inf) 
            break 

        epoch_loss /= len(train_loader)

        current_valid_nll = np.inf 
        current_valid_perf = np.nan
        if valid_loader is not None:
            model.eval()
            with torch.no_grad():
                perf, nll = valid_performance(model, valid_loader, gaussian_log_likelihood_loss, device, head=head, head_activation=head_activation, likelihood=likelihood)
            valid_perfs.append(perf)
            valid_nlls.append(nll)
            current_valid_nll = nll if not (np.isnan(nll) or np.isinf(nll)) else np.inf
            current_valid_perf = perf if not (np.isnan(perf) or np.isinf(perf)) else np.nan

            if np.isnan(valid_nlls[-1]) or np.isinf(valid_nlls[-1]):
                print(f"WARNING: NaN/Inf in validation NLL at epoch {epoch}")

        if use_wandb:
            log_dict = {'epoch': epoch}
            
            if not np.isnan(epoch_loss):
                log_dict['train/loss'] = epoch_loss
            else:
                log_dict['train/loss'] = float('nan') 

            if valid_loader is not None:
                log_dict['valid/perf'] = current_valid_perf 
                log_dict['valid/nll'] = current_valid_nll if current_valid_nll != np.inf else float('inf')

            wandb.log(log_dict, step=epoch)

    if training_failed:
        print("Training stopped prematurely due to NaNs/Infs.")

    return model, valid_perfs, valid_nlls


def gaussian_log_likelihood_loss(output, target, head='gaussian', head_activation='softplus', likelihood='heteroscedastic', epsilon=1e-8):
    """Calculates Gaussian NLL with stabilization."""

    if likelihood == 'homoscedastic':
        mu = output.squeeze(-1)
        raise NotImplementedError("Homoscedastic NLL requires sigma_noise handling")

    else: 
        mu = output[:, 0] 

        if head == 'natural':
            eta2 = output[:, 1] 
            var = -0.5 / eta2
        elif head == 'gaussian' and head_activation == 'softplus':
            v = output[:, 1] 
            var = torch.nn.functional.softplus(v)
        elif head == 'gaussian' and head_activation is None: 
             var = output[:, 1]
             var = var.clamp(min=0)
        else: 
            var = output[:, 1] 
            var = var.clamp(min=0) 

        var = torch.nan_to_num(var, nan=epsilon, posinf=1e6, neginf=epsilon)
        var = var.clamp(min=0) + epsilon
        var = var.clamp(min=epsilon, max=1e7) 
        var = torch.nan_to_num(var, nan=epsilon, posinf=1e7, neginf=epsilon)
        target = target.view_as(mu) 
        var = var.view_as(mu) 
        log_var = torch.log(var) 
        sq_err_over_var = ((target - mu)**2) / var

        log_likelihood = -0.5 * (np.log(2 * np.pi) + log_var + sq_err_over_var)

        if not torch.isfinite(log_likelihood).all():
            print("Warning: NaN/Inf detected in final log_likelihood calculation.")
            log_likelihood = torch.nan_to_num(log_likelihood, nan=-30, posinf=-30, neginf=-30) 

        return -log_likelihood
