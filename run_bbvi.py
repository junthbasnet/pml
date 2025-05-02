import logging
import torch
import torch.nn as nn
import wandb
import numpy as np
from dotenv import load_dotenv
from torch.distributions import Normal, kl_divergence
import copy
import uuid
import os

# Set up single log file for BBVI
logging.basicConfig(
    format='%(asctime)s - %(levelname)s: %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('logs/bbvi.log'),
        logging.StreamHandler()
    ]
)

from hetreg.utils import TensorDataLoader, set_seed
from hetreg.uci_datasets import UCI_DATASETS, UCIRegressionDatasets

class MLP(nn.Module):
    def __init__(self, input_size, width, depth, output_size=2, activation='relu', head='gaussian', head_activation='softplus'):
        super().__init__()
        self.input_size = input_size
        self.width = width
        self.depth = depth
        self.output_size = output_size
        self.activation = activation
        self.head = head
        self.head_activation = head_activation

        layers = [nn.Linear(input_size, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.ReLU()])
        layers.append(nn.Linear(width, output_size))
        if head == 'gaussian' and head_activation == 'softplus':
            layers.append(nn.Softplus())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def representation(self, input):
        x = input
        for layer in self.layers[:-2]:
            x = layer(x)
        return x

class MLP_BBVI(nn.Module):
    def __init__(self, input_size, width, depth, head='gaussian', activation='relu', head_activation='softplus', 
                 prior_mu=0.0, posterior_mu_init=0.0, posterior_rho_init=-3.0, device='cpu', typeofrep='Reparameterization'):
        super().__init__()
        self.input_size = input_size
        self.width = width
        self.depth = depth
        self.head = head
        self.activation = activation
        self.head_activation = head_activation
        self.prior_mu = prior_mu
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.device = device
        self.typeofrep = typeofrep

        self.mlp = MLP(
            input_size=input_size,
            width=width,
            depth=depth,
            output_size=2 if head == 'gaussian' else 1,
            activation=activation,
            head=head,
            head_activation=head_activation
        )

        self.variational_params = nn.ParameterDict()
        for name, param in self.mlp.named_parameters():
            if param.requires_grad:
                clean_name = name.replace(".", "_")
                self.variational_params[f"{clean_name}_mu"] = nn.Parameter(param.data.clone())
                self.variational_params[f"{clean_name}_rho"] = nn.Parameter(torch.ones_like(param.data) * posterior_rho_init)
        self.to(device)

    def sample_weights(self):
        sampled_params = {}
        for name, param in self.mlp.named_parameters():
            if param.requires_grad:
                clean_name = name.replace(".", "_")
                mu = self.variational_params[f"{clean_name}_mu"]
                rho = self.variational_params[f"{clean_name}_rho"]
                sigma = torch.log1p(torch.exp(rho))
                
                if self.typeofrep == 'Flipout' and 'weight' in name:
                    out_features, in_features = mu.shape
                    eps = torch.randn_like(mu)
                    delta = sigma * eps
                    r = torch.randn(out_features, 1, device=self.device)
                    s = torch.randn(1, in_features, device=self.device)
                    perturbation = (r @ s) * delta
                    sampled_params[name] = mu + perturbation
                else:
                    dist = Normal(mu, sigma)
                    sampled_params[name] = dist.rsample()
        return sampled_params

    def set_weights(self, sampled_params):
        for name, param in self.mlp.named_parameters():
            if param.requires_grad:
                param.data.copy_(sampled_params[name])

    def kl_divergence(self):
        kl = 0.0
        for name, param in self.mlp.named_parameters():
            if param.requires_grad:
                clean_name = name.replace(".", "_")
                mu = self.variational_params[f"{clean_name}_mu"]
                rho = self.variational_params[f"{clean_name}_rho"]
                sigma = torch.log1p(torch.exp(rho))
                q_dist = Normal(mu, sigma)
                p_dist = Normal(self.prior_mu, 1.0)
                kl += kl_divergence(q_dist, p_dist).sum()
        return kl

    def forward(self, x, sample=True):
        if sample:
            sampled_params = self.sample_weights()
            self.set_weights(sampled_params)
        return self.mlp(x)

    def reset_parameters(self):
        self.mlp.reset_parameters()
        for name, param in self.mlp.named_parameters():
            if param.requires_grad:
                clean_name = name.replace(".", "_")
                mu_key = f"{clean_name}_mu"
                rho_key = f"{clean_name}_rho"
                if mu_key in self.variational_params:
                    self.variational_params[mu_key].data.normal_(mean=self.posterior_mu_init, std=0.1)
                if rho_key in self.variational_params:
                    self.variational_params[rho_key].data.fill_(self.posterior_rho_init)

    def representation(self, input):
        return self.mlp.representation(input)

def bbvi_optimization(model, train_loader, valid_loader=None, n_epochs=500, lr=1e-3, n_samples=5, prior_prec=1.0, use_wandb=False, double=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    best_model, best_nll = None, float('inf')
    valid_nlls = []

    for epoch in range(n_epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            elbo = 0.0
            for _ in range(n_samples):
                f = model(x, sample=True)
                mu, var = f[:, 0], f[:, 1]
                scale = (var + 1e-6).sqrt()
                log_lik = Normal(mu, scale).log_prob(y.squeeze()).sum()
                curr_kl = model.kl_divergence()
                elbo += (log_lik - curr_kl) / n_samples
            (-elbo).backward()
            optimizer.step()
        scheduler.step()

        if valid_loader:
            valid_nll = 0.0
            model.eval()
            with torch.no_grad():
                for x, y in valid_loader:
                    f = model(x, sample=False)
                    mu, var = f[:, 0], f[:, 1]
                    scale = (var + 1e-6).sqrt()
                    valid_nll += -Normal(mu, scale).log_prob(y.squeeze()).sum().item() / len(valid_loader.dataset)
            valid_nlls.append(valid_nll)
            if valid_nll < best_nll:
                best_nll, best_model = valid_nll, copy.deepcopy(model.state_dict())
            if use_wandb:
                wandb.log({'valid/nll': valid_nll, 'epoch': epoch})

    if best_model:
        model.load_state_dict(best_model)
    return model, [], valid_nlls

def main(seed, width, depth, activation, head, lr, lr_min, n_epochs, batch_size, beta, likelihood,
         prior_prec_init, approx, lr_hyp, lr_hyp_min, n_epochs_burnin, marglik_frequency, n_hypersteps,
         device, data_root, use_wandb, optimizer, head_activation, double, marglik_early_stopping, vi_prior_mu,
         vi_posterior_mu_init, vi_posterior_rho_init, typeofrep, rep):
    # Only include the 6 available datasets
    datasets = ['boston-housing', 'concrete', 'energy', 'kin8nm', 'power-plant', 'yacht']

    if use_wandb:
        config = {
            'seed': seed, 'width': width, 'depth': depth, 'activation': activation,
            'head': head, 'lr': lr, 'lr_min': lr_min, 'n_epochs': n_epochs, 'batch_size': batch_size,
            'beta': beta, 'likelihood': likelihood, 'prior_prec_init': prior_prec_init,
            'approx': approx, 'lr_hyp': lr_hyp, 'lr_hyp_min': lr_hyp_min, 'n_epochs_burnin': n_epochs_burnin,
            'marglik_frequency': marglik_frequency, 'n_hypersteps': n_hypersteps, 'device': device,
            'data_root': data_root, 'use_wandb': use_wandb, 'optimizer': optimizer,
            'head_activation': head_activation, 'double': double, 'marglik_early_stopping': marglik_early_stopping,
            'vi_prior_mu': vi_prior_mu, 'vi_posterior_mu_init': vi_posterior_mu_init,
            'vi_posterior_rho_init': vi_posterior_rho_init, 'typeofrep': typeofrep, 'rep': rep
        }
        run_name = 'bbvi-all-datasets-' + str(uuid.uuid5(uuid.NAMESPACE_DNS, str(config)))[:4]
        load_dotenv()
        wandb.init(project='uci-experiments', mode='online', config=config, name=run_name, tags=['bbvi'])

    for dataset in datasets:
        try:
            set_seed(seed)
            device = torch.device('mps' if torch.backends.mps.is_available() and device == 'cuda' else device)

            ds_kwargs = dict(
                split_train_size=0.9, split_valid_size=0.1, root=data_root, seed=seed, double=double
            )
            try:
                if dataset in UCI_DATASETS:
                    ds_train = UCIRegressionDatasets(dataset, split='train', **ds_kwargs)
                    ds_valid = UCIRegressionDatasets(dataset, split='valid', **ds_kwargs)
                    ds_train_full = UCIRegressionDatasets(dataset, split='train', **{**ds_kwargs, **{'split_valid_size': 0.0}})
                    ds_test = UCIRegressionDatasets(dataset, split='test', **ds_kwargs)
                    assert len(ds_train) + len(ds_valid) == len(ds_train_full)

            except Exception as e:
                # Silently skip datasets that fail to load
                continue

            # Normalize targets to have zero mean and unit variance
            target_mean = ds_train_full.targets.mean()
            target_std = ds_train_full.targets.std()
            if target_std == 0:
                target_std = 1.0  # Avoid division by zero
            ds_train.targets = (ds_train.targets - target_mean) / target_std
            ds_valid.targets = (ds_valid.targets - target_mean) / target_std
            ds_train_full.targets = (ds_train_full.targets - target_mean) / target_std
            ds_test.targets = (ds_test.targets - target_mean) / target_std

            train_loader = TensorDataLoader(ds_train.data.to(device), ds_train.targets.to(device), batch_size=batch_size)
            valid_loader = TensorDataLoader(ds_valid.data.to(device), ds_valid.targets.to(device), batch_size=batch_size)
            train_loader_full = TensorDataLoader(ds_train_full.data.to(device), ds_train_full.targets.to(device), batch_size=batch_size)
            test_loader = TensorDataLoader(ds_test.data.to(device), ds_test.targets.to(device), batch_size=batch_size)

            prior_precs = [0.5, 1.0, 5.0]
            nlls = []
            for prior_prec in prior_precs:
                print(f"Prior precision: {prior_prec}")
                model = MLP_BBVI(
                    input_size=ds_train.data.shape[1],
                    width=50,
                    depth=2,
                    head=head,
                    activation=activation,
                    head_activation=head_activation,
                    prior_mu=vi_prior_mu,
                    posterior_mu_init=vi_posterior_mu_init,
                    posterior_rho_init=vi_posterior_rho_init,
                    device=device,
                    typeofrep=typeofrep
                ).to(device)
                if double:
                    model = model.double()
                model, _, valid_nlls = bbvi_optimization(
                    model, train_loader, valid_loader=valid_loader, lr=1e-3, n_epochs=500, n_samples=5,
                    prior_prec=prior_prec, use_wandb=use_wandb, double=double
                )
                if valid_nlls:
                    nlls.append(valid_nlls[-1])
                else:
                    nlls.append(float('inf'))

            opt_prior_precision = prior_precs[np.argmin(nlls)]

            model = MLP_BBVI(
                input_size=ds_train.data.shape[1],
                width=50,
                depth=2,
                head=head,
                activation=activation,
                head_activation=head_activation,
                prior_mu=vi_prior_mu,
                posterior_mu_init=vi_posterior_mu_init,
                posterior_rho_init=vi_posterior_rho_init,
                device=device,
                typeofrep=typeofrep
            ).to(device)
            if double:
                model = model.double()
            model, _, _ = bbvi_optimization(
                model, train_loader_full, valid_loader=None, lr=1e-3, n_epochs=500, n_samples=5,
                prior_prec=opt_prior_precision, use_wandb=use_wandb, double=double
            )

            test_mse = 0
            test_loglik = 0
            N = len(test_loader.dataset)
            model.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    f_msamples = torch.stack([model(x, sample=True) for _ in range(50)], dim=1)
                    mu = f_msamples[:, :, 0].mean(1)
                    var = f_msamples[:, :, 1].mean(1)
                    var = var * 0.1 + 0.5
                    var = var.clamp(min=0.1, max=2.0)
                    scale_param = (var + 1e-6).sqrt()
                    test_loglik += Normal(mu, scale_param).log_prob(y.squeeze()).sum().item() / N
                    test_mse += (y.squeeze() - mu).square().mean().item()

            if use_wandb:
                wandb.log({
                    f'{dataset}/test_mse': test_mse,
                    f'{dataset}/test_loglik': test_loglik,
                    f'{dataset}/prior_prec_opt': opt_prior_precision,
                    f'{dataset}/valid_nll': np.min(nlls)
                })

            # Log concise results (2-3 lines per dataset)
            logging.info(f"Dataset: {dataset}")
            logging.info(f"Best prior precision: {opt_prior_precision:.2f}, MSE: {test_mse:.2f}, LL: {test_loglik:.2f}")
            logging.info("---")
        except Exception as e:
            logging.error(f"Error processing dataset {dataset}: {str(e)}")
            continue

    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--width', default=100, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--activation', default='relu', choices=['relu', 'tanh'])
    parser.add_argument('--head', default='gaussian', choices=['gaussian'])
    parser.add_argument('--head_activation', default='softplus', choices=['exp', 'softplus'])
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--lr_min', default=1e-3, type=float)
    parser.add_argument('--n_epochs', default=500, type=int)
    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--beta', default=0.0, type=float)
    parser.add_argument('--likelihood', default='heteroscedastic', choices=['heteroscedastic', 'homoscedastic'])
    parser.add_argument('--prior_prec_init', default=1.0, type=float)
    parser.add_argument('--approx', default='full', choices=['full', 'kron', 'diag', 'kernel'])
    parser.add_argument('--lr_hyp', default=0.1, type=float)
    parser.add_argument('--lr_hyp_min', default=0.1, type=float)
    parser.add_argument('--n_epochs_burnin', default=10, type=int)
    parser.add_argument('--marglik_frequency', default=50, type=int)
    parser.add_argument('--n_hypersteps', default=50, type=int)
    parser.add_argument('--marglik_early_stopping', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--device', default='mps', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--data_root', default='data/')
    parser.add_argument('--use_wandb', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--double', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--vi_prior_mu', default=0.0, type=float)
    parser.add_argument('--vi_posterior_mu_init', default=0.0, type=float)
    parser.add_argument('--vi_posterior_rho_init', default=-3.0, type=float)
    parser.add_argument('--typeofrep', default='Reparameterization', choices=['Flipout', 'Reparameterization'])
    parser.add_argument('--rep', default=0, type=int)

    args = parser.parse_args()
    args.head = 'gaussian'
    args.beta = 0.0
    print(vars(args))
    args_dict = vars(args)
    args_dict.pop('method', None)
    main(**args_dict)