import torch
from torch import nn
from torch.nn import functional as F
from bayesian_torch.layers import LinearReparameterization
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator


ACTIVATIONS = ['relu', 'selu', 'tanh', 'silu', 'gelu']
HEADS = ['natural', 'meanvar', 'gaussian']

def stable_softplus(input):
   return F.softplus(input) + 1e-8

def get_activation(act_str):
    if act_str == 'relu':
        return nn.ReLU
    elif act_str == 'tanh':
        return nn.Tanh
    elif act_str == 'selu':
        return nn.SELU
    elif act_str == 'silu':
        return nn.SiLU
    elif act_str == 'gelu':
        return nn.GELU
    else:
        raise ValueError('invalid activation')


def get_head_activation(act_str):
    if act_str == 'exp':
        return torch.exp
    elif act_str == 'softplus':
        return stable_softplus
    else:
        raise ValueError('invalid activation')


class NaturalHead(nn.Module):

    def __init__(self, activation='softplus') -> None:
        super().__init__()
        self.act_fn = get_head_activation(activation)

    def forward(self, input):
        return torch.stack([input[:, 0], -0.5 * self.act_fn(input[:, 1])], 1)


class GaussianHead(nn.Module):

    def __init__(self, activation='softplus') -> None:
        super().__init__()
        self.act_fn = get_head_activation(activation)

    def forward(self, input):
        f1, f2 = input[:, 0], self.act_fn(input[:, 1])
        return torch.stack([f1, f2], 1)


def get_head(head_str):
    if head_str == 'natural':
        return NaturalHead
    elif head_str == 'gaussian':
        return GaussianHead
    else:
        return nn.Identity


class NaturalReparamHead(nn.Module):
    # Transform mean-var into natural parameters
    def forward(self, input):
        f1, f2 = input[:, 0], input[:, 1]
        eta_1 = f1 / f2
        eta_2 = - 1 / (2 * f2)
        return torch.stack([eta_1, eta_2], 1)


class MLP(nn.Sequential):

    def __init__(self, input_size, width, depth, output_size, activation='gelu',
                 head='natural', head_activation='exp', skip_head=False, dropout=0.0):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.width = width
        self.depth = depth
        hidden_sizes = depth * [width]
        self.activation = activation
        act = get_activation(activation)
        self.rep_layer = f'layer{depth}'

        self.add_module('flatten', nn.Flatten())
        if len(hidden_sizes) == 0:  # i.e. when depth == 0.
            # Linear Model
            self.add_module('lin_layer', nn.Linear(self.input_size, output_size, bias=True))
        else:
            # MLP
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            for i, (in_size, out_size) in enumerate(in_outs):
                self.add_module(f'layer{i+1}', nn.Linear(in_size, out_size, bias=True))
                if dropout > 0.0:
                    self.add_module(f'dropout{i+1}', nn.Dropout(p=dropout))
                self.add_module(f'{activation}{i+1}', act())
            self.add_module('out_layer', nn.Linear(hidden_sizes[-1], output_size, bias=True))
        if not skip_head:
            self.add_module('head', get_head(head)(activation=head_activation))

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def representation(self, input):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        handle = getattr(self, self.rep_layer).register_forward_hook(get_activation(self.rep_layer))
        self.forward(input)
        rep = activation[self.rep_layer]
        handle.remove()
        return rep.detach()

def make_bayesian(model, prior_mu, prior_sigma, posterior_mu_init, posterior_rho_init, typeofrep):
    const_bnn_prior_parameters = {
        "prior_mu": prior_mu,
        "prior_sigma": prior_sigma,
        "posterior_mu_init": posterior_mu_init,
        "posterior_rho_init": posterior_rho_init,
        "type": typeofrep,  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
    }
    dnn_to_bnn(model, const_bnn_prior_parameters)
    return model

@variational_estimator
class MLPVI(nn.Sequential):

    def __init__(self, input_size, width, depth, output_size, activation='gelu',
                 head='natural', head_activation='exp', skip_head=False, priorsigma1=0.1, priorsigma2=0.4):
        super(MLPVI, self).__init__()
        self.input_size = input_size
        self.width = width
        self.depth = depth
        hidden_sizes = depth * [width]
        self.activation = activation
        act = get_activation(activation)
        self.priorsigma1 = priorsigma1
        self.priorsigma2 = priorsigma2
        self.rep_layer = f'layer{depth}'

        self.add_module('flatten', nn.Flatten())
        if len(hidden_sizes) == 0:  # i.e. when depth == 0.
            # Linear Model
            self.add_module('lin_layer', BayesianLinear(self.input_size, output_size, bias=True, prior_sigma_1=self.priorsigma1, prior_sigma_2=self.priorsigma2))
        else:
            # MLP
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            for i, (in_size, out_size) in enumerate(in_outs):
                self.add_module(f'layer{i+1}', BayesianLinear(in_size, out_size, bias=True, prior_sigma_1=self.priorsigma1, prior_sigma_2=self.priorsigma2))
                self.add_module(f'{activation}{i+1}', act())
            self.add_module('out_layer', BayesianLinear(hidden_sizes[-1], output_size, bias=True, prior_sigma_1=self.priorsigma1, prior_sigma_2=self.priorsigma2))
        if not skip_head:
            self.add_module('head', get_head(head)(activation=head_activation))

    def representation(self, input):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        handle = getattr(self, self.rep_layer).register_forward_hook(get_activation(self.rep_layer))
        self.forward(input)
        rep = activation[self.rep_layer]
        handle.remove()
        return rep.detach()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def init_parameters_vi(self):
        for module in self.modules():
            if isinstance(module, LinearReparameterization):
                module.init_parameters()
            if isinstance(module, BayesianLinear):
                module.reset_parameters()


class MLPFaithfulseq(nn.Sequential):
    def __init__(self, input_size, width, depth, output_size, activation='gelu',
                 head='natural', head_activation='exp'):
        super(MLPFaithfulseq, self).__init__()
        self.input_size = input_size
        self.width = width
        self.depth = depth
        hidden_sizes = depth * [width]
        self.activation = activation
        act = get_activation(activation)

        self.add_module('flatten', nn.Flatten())
        if len(hidden_sizes) == 0:  # i.e. when depth == 0.
            # Linear Model
            self.add_module('lin_layer', nn.Linear(self.input_size, output_size, bias=True))
        else:
            # MLP
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            for i, (in_size, out_size) in enumerate(in_outs):
                self.add_module(f'layer{i + 1}', nn.Linear(in_size, out_size, bias=True))
                self.add_module(f'{activation}{i + 1}', act())


class MLPFaithful(nn.Module):
    def __init__(self, input_size, width, depth, activation='gelu',
                 head='natural', head_activation='exp'):
        super(MLPFaithful, self).__init__()

        self.z_layer = MLPFaithfulseq(input_size, width, depth-1, output_size=width, activation=activation,
                 head=head, head_activation=head_activation)

        self.out_layer_mu = nn.Linear(width, 1, bias=True)
        self.out_layer_var = nn.Linear(width, 1, bias=True)

        self.head_layer = get_head(head)(activation=head_activation)

    def forward(self, input):
        z = self.z_layer(input)
        mu_z = self.out_layer_mu(z)
        var_z = self.out_layer_var(z.detach())     # Stop gradient here
        return self.head_layer(torch.cat([mu_z, var_z], dim=-1))

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def representation(self, input):
        return self.z_layer(input).detach()


if __name__ == '__main__':
    print('...')
