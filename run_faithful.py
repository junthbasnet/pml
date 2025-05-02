import logging
import torch
import wandb
import numpy as np
from dotenv import load_dotenv
from torch.distributions import Normal

from hetreg.utils import TensorDataLoader, set_seed
from hetreg.uci_datasets import UCI_DATASETS, UCIRegressionDatasets
from hetreg.models import ACTIVATIONS, HEADS
from hetreg.models import MLPFaithful
from hetreg.faithful import faithful_optimization


logging.basicConfig(
    filename='logs/faithful.log',
    format='%(asctime)s - [%(filename)s:%(lineno)s]%(levelname)s: %(message)s',
    level=logging.INFO
)

def main(seed, dataset, width, depth, activation, head, lr, lr_min, n_epochs, batch_size, method, beta,
         device, data_root, use_wandb, optimizer, head_activation, double):
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

    if method == 'faithful':
        output_size = 2
        model = MLPFaithful(
            input_size, width, depth, activation=activation,
            head=head, head_activation=head_activation
        ).to(device)
        if double:
            model = model.double()

        #grid_setting_names = ['lr', 'prior_precision']
        prior_precs = np.logspace(-4, 4, 9)
        #grid_settings = [[1e-4, 3e-4, 7e-4, 1e-3, 3e-3, 7e-3], prior_precs]
        #for settings in itertools.product(*grid_settings):
        #    conf = {grid_setting_names[idx]: value for idx, value in enumerate(settings)}
        nlls = []
        for prior_prec in prior_precs:
            print(prior_prec)
            model.reset_parameters()
            model, valid_perfs, valid_nlls = faithful_optimization(
                model, train_loader, valid_loader=valid_loader, lr=lr, lr_min=lr_min, n_epochs=n_epochs, beta=beta,
                 prior_structure='scalar', scheduler='cos', optimizer=optimizer, prior_prec_init=prior_prec, use_wandb=use_wandb
            )
            nlls.append(valid_nlls[-1])

        # choose best prior precision and rerun on combined train + validation set
        opt_prior_prec = prior_precs[np.argmin(nlls)]
        if use_wandb:
            wandb.run.summary['prior_prec_opt'] = opt_prior_prec
            wandb.run.summary['valid/nll'] = np.min(nlls)
        logging.info(f'Best prior precision found: {opt_prior_prec}')
        model.reset_parameters()
        model, _, _ = faithful_optimization(
            model, train_loader_full, lr=lr, lr_min=lr_min, n_epochs=n_epochs, prior_structure='scalar', beta=beta,
            scheduler='cos', optimizer=optimizer, prior_prec_init=opt_prior_prec, use_wandb=use_wandb
        )

        
        # Evaluate the trained model on test set.
        scale = ds_train.s
        test_mse = 0
        test_loglik = 0
        N = len(test_loader.dataset)
        for x, y in test_loader:
            f = model(x)
            mu = f[: , 0]
            #std = f[:,1]
            std = torch.sqrt(f[: , 1])
            #test_loglik += -gaussian_log_likelihood_loss(f.detach(), y).sum().item()
            test_loglik += Normal(scale * mu, scale * std).log_prob(y.squeeze() * scale).sum().item() / N
            #Normal(scale * mu, scale * std).log_prob(y).sum().item() / N
            test_mse += (y.squeeze() - mu).square().sum() / N

    else:
        raise ValueError('Invalid method')

    if use_wandb:
        wandb.run.summary['test/mse'] = test_mse
        wandb.run.summary['test/loglik'] = test_loglik
        
    logging.info(f'Final test performance: MSE={test_mse:.3f}, LogLik={test_loglik:.3f}')


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
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_min', default=1e-5, type=float, help='Cosine decay target')
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--method', default='faithful', help='Method', choices=['faithful',])
    parser.add_argument('--beta', default=0.5, type=float)

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
