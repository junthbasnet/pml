import logging
import torch
import wandb
import numpy as np
from dotenv import load_dotenv
from torch.distributions import Normal
from hetreg.utils import TensorDataLoader, set_seed
from hetreg.uci_datasets import UCI_DATASETS, UCIRegressionDatasets
from hetreg.models import MLP, ACTIVATIONS, HEADS
from hetreg.sgld import sgld_optimization


logging.basicConfig(
    filename='logs/sgld.log',
    format='%(asctime)s - [%(filename)s:%(lineno)s]%(levelname)s: %(message)s',
    level=logging.INFO
)

def main(seed, dataset, width, depth, activation, head, lr, n_epochs, batch_size, method, likelihood,
         device, data_root, use_wandb, head_activation, double):
    set_seed(seed)

    # Set up data. 90% train and 10% test is default since Hernandez-Lobato
    # Take 10% of train for validation when not using marglik
    ds_kwargs = dict(
        split_train_size=0.9, split_valid_size=0.1, root=data_root, seed=seed, double=double
    )
    if dataset in UCI_DATASETS:
        ds_train = UCIRegressionDatasets(dataset, split='train', **ds_kwargs)
        ds_valid = UCIRegressionDatasets(dataset, split='valid', **ds_kwargs)
        ds_train_full = UCIRegressionDatasets(dataset, split='train', **{**ds_kwargs, **{'split_valid_size': 0.0}})
        ds_test = UCIRegressionDatasets(dataset, split='test', **ds_kwargs)
        assert len(ds_train) + len(ds_valid) == len(ds_train_full)

    train_loader = TensorDataLoader(ds_train.data.to(device), ds_train.targets.to(device), batch_size=batch_size)
    valid_loader = TensorDataLoader(ds_valid.data.to(device), ds_valid.targets.to(device), batch_size=batch_size)
    train_loader_full = TensorDataLoader(ds_train_full.data.to(device), ds_train_full.targets.to(device), batch_size=batch_size)
    test_loader = TensorDataLoader(ds_test.data.to(device), ds_test.targets.to(device), batch_size=batch_size)

    # Set up model.
    input_size = ds_train.data.size(1)

    if method == 'sgld':
        output_size = 1 if likelihood == 'homoscedastic' else 2
        if head == 'natural' and output_size != 2:
            print("Warning: Natural head requires output_size=2. Forcing.")
            output_size = 2

        # ——— Cross-validate Prior Precision ———
        prior_precs = np.logspace(0, 3, 10)
        nlls = [] # To store validation NLLs
        logging.info(f'Dataset: {dataset}')
        logging.info(f"Starting prior precision cross-validation for SGLD...")
        for prior_prec in prior_precs:
            print(f"Testing prior precision: {prior_prec:.4f}")
            model = MLP(
                input_size, width, depth,
                output_size=output_size,
                activation=activation,
                head=head, 
                head_activation=head_activation
            ).to(device)
            if double:
                model = model.double()
            model.reset_parameters() 

            # Run SGLD training
            try:
                _, _, valid_nlls_run = sgld_optimization( 
                    model,
                    train_loader, 
                    valid_loader=valid_loader,
                    n_epochs=n_epochs, 
                    lr=lr,
                    prior_prec_init=prior_prec, 
                    addnoise=True,
                    use_wandb=False, 
                    max_grad_norm=1.0, 
                    head=head, 
                    head_activation=head_activation, 
                    likelihood=likelihood 
                )
                
                final_nll = valid_nlls_run[-1] if valid_nlls_run else np.inf
                if np.isnan(final_nll) or np.isinf(final_nll):
                    nlls.append(np.inf) 
                    print(f"  Validation NLL: {final_nll}") 
                else:
                    nlls.append(final_nll)
                    print(f"  Validation NLL: {final_nll:.4f}")

            except Exception as e:
                
                import traceback
                print(f"  ERROR during CV run for prior_prec {prior_prec:.4f}: {e}")
                nlls.append(np.inf) 

        # Choose the best prior precision
        if not nlls or all(np.isinf(n) for n in nlls):
            print("\nWarning: All prior precisions resulted in Inf/NaN validation NLLs during CV.")
            finite_nlls = [(p, n) for p, n in zip(prior_precs, nlls) if not np.isinf(n) and not np.isnan(n)]
            if finite_nlls:
                opt_idx = np.argmin([n for p,n in finite_nlls])
                opt_prior_prec = finite_nlls[opt_idx][0]
                best_nll = finite_nlls[opt_idx][1]
                print(f"Choosing best finite NLL prior: {opt_prior_prec:.4f} (NLL: {best_nll:.4f})")
            else:
                opt_prior_prec = 1.0
                best_nll = np.inf
                print(f"Falling back to default prior precision: {opt_prior_prec:.4f}")
        else:
            opt_idx = np.argmin(nlls)
            opt_prior_prec = prior_precs[opt_idx]
            best_nll = nlls[opt_idx]
            logging.info(f'Best prior precision found via CV: {opt_prior_prec:.4f}')

        if use_wandb:
            wandb.run.summary['sgld_prior_prec_opt'] = opt_prior_prec
            wandb.run.summary['sgld_valid_nll_at_opt_prior'] = best_nll if best_nll != np.inf else float('inf')


        # ——— Retrain on full training data with the optimal prior precision ———
        logging.info(f"Retraining SGLD on full dataset with optimal prior precision: {opt_prior_prec:.4f}")
        model = MLP(
            input_size, width, depth,
            output_size=output_size,
            activation=activation,
            head=head, 
            head_activation=head_activation 
        ).to(device)
        if double:
            model = model.double()
        model.reset_parameters() 

        model, _, final_valid_nlls = sgld_optimization( 
            model,
            train_loader_full, 
            valid_loader=valid_loader, 
            n_epochs=n_epochs,
            lr=lr,
            prior_prec_init=opt_prior_prec, 
            addnoise=True,
            use_wandb=use_wandb, 
            max_grad_norm=1.0, 
            head=head, 
            head_activation=head_activation, 
            likelihood=likelihood 
        )

        scale = ds_train.s if hasattr(ds_train, 's') else 1.0 
        test_mse    = 0.0
        test_loglik = 0.0
        N_test = len(test_loader.dataset)
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                f = model(x)

                if likelihood == 'homoscedastic':
                    mu = f.squeeze(-1)
                    sigma_noise = getattr(model, 'sigma_noise', 1.0) 
                    var = torch.full_like(mu, sigma_noise**2)
                else: 
                    if head == 'natural':
                        eta1, eta2 = f[:,0], f[:,1]
                        eta2_clamped = eta2.clamp(max=-1e-8) 
                        var = -0.5 / eta2_clamped
                        mu  = eta1 * var
                    elif head == 'gaussian' and head_activation == 'softplus':
                        mu, v = f[:,0], f[:,1]
                        var = torch.nn.functional.softplus(v) + 1e-8 
                    elif head == 'gaussian' and head_activation is None: 
                        mu, var = f[:, 0], f[:, 1]
                        var = var.clamp(min=1e-8) 
                    else: 
                        mu, var = f[:,0], f[:,1]
                        var = var.clamp(min=1e-8) 

                    var = torch.nan_to_num(var, nan=1e-6, posinf=1e6, neginf=1e-6) 
                    var = var.clamp(min=1e-8, max=1e6) 


                mu = mu.view_as(y)
                var = var.view_as(y)

                y_std = torch.sqrt(var)
                dist = Normal(loc=mu * scale, scale=y_std * scale) 

                test_mse += ((mu - y)**2).sum().item() 
                test_loglik += dist.log_prob(y * scale).sum().item() 

        test_mse /= N_test
        test_loglik /= N_test

        # Log final test metrics
        if use_wandb:
            wandb.run.summary['test/mse'] = test_mse
            wandb.run.summary['test/loglik'] = test_loglik
        logging.info(f'Final test performance: MSE={test_mse:.3f}, LogLik={test_loglik:.3f}')

    else:
        raise ValueError('Invalid method')


if __name__ == '__main__':
    import sys
    import argparse
    from arg_utils import set_defaults_with_yaml_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--dataset', default='energy', choices=UCI_DATASETS)
    # architecture
    parser.add_argument('--width', default=50, type=int)
    parser.add_argument('--depth', default=2, type=int)
    parser.add_argument('--activation', default='gelu', choices=ACTIVATIONS)
    parser.add_argument('--head', default='gaussian', choices=HEADS)
    parser.add_argument('--head_activation', default='softplus', choices=['exp', 'softplus'])
    # optimization (general)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--method', default='sgld', help='Method', choices=['sgld',])
    parser.add_argument('--likelihood', default='heteroscedastic', choices=['heteroscedastic', 'homoscedastic'])
    # others
    parser.add_argument('--device', default='mps', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--data_root', default='data/')
    parser.add_argument('--use_wandb', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--double', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--config', nargs='+')

    set_defaults_with_yaml_config(parser, sys.argv)
    args = vars(parser.parse_args())
    args.pop('config')
    if args['use_wandb']:
        import uuid
        import copy
        tags = [args['dataset'], args['method']]
        config = copy.deepcopy(args)
        run_name = '-'.join(tags)
        run_name += '-' + str(uuid.uuid5(uuid.NAMESPACE_DNS, str(args)))[:4]
        load_dotenv()
        wandb.init(
            project='uci-experiments',
            entity='junthbasnet-indian-institute-of-technology-kanpur',
            mode='online',
            config=config,
            name=run_name,
            tags=tags
        )
    print(args)
    for dataset in UCI_DATASETS:
        args['dataset'] = dataset
        method = args['method']
        print(f'{dataset} Dataset')
        logging.info(f'Method: {method} Dataset: {dataset}')
        main(**args)
