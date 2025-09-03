
# Bayesian Heteroscedastic Regression with Deep Neural Networks

[Seed paper](https://openreview.net/pdf?id=A6EquH0enk)

## Setup
For experiments, we used python version 3.9 and torch version 1.12.1

Additional online dependencies are listed in `requirements.txt` and have to be installed with `pip install -r requirements.txt`.

Further, `dependencies/` contain modified versions of [`laplace-torch`](https://github.com/aleximmer/Laplace) and [`asdl`](https://github.com/kazukiosawa/asdl) and have to be installed with `pip install -e dependencies/laplace` and `pip install -e dependencies/asdl`.
To install local utilities, run `pip install -e .` in the root directory of this repository.

## Implementation Details
The author extended laplace-torch to support (natural) heteroscedastic Gaussian likelihoods and asdl to support their Fisher/GGN curvature approximations as described in the seed paper.
The modifications can be found in `dependencies/`.
Laplace requires the natural parameterization of the likelihood for positie semidefiniteness of the Hessian as described in the seed paper.

### Models
In `hetreg/models.py` there's implementations of an `MLP`.

## Running Experiments
Experiment has been performed on UCI regression dataset.

For running each method, you can just run and can pass arguments as well or just change the default in the code and run it.
For example, to run mcdropout

```
python run_mcdropout.py
```

to run mean_field_vi
```
python run_mean_field_vi.py
```

Logs will be generated at logs/ or alternatively you can use wandb for logging as well. just change entity accordingly.

However, logs should give the general idea of LL and MSE
```
2025-05-01 23:48:56,153 - [run_mcdropout.py:165]INFO: Method: mcdropout Dataset: boston-housing
2025-05-01 23:49:35,410 - [run_mcdropout.py:70]INFO: Best prior precision found: 1.0
2025-05-01 23:49:39,445 - [run_mcdropout.py:109]INFO: Final test performance: MSE=0.058, LogLik=-2.752
2025-05-01 23:49:39,445 - [run_mcdropout.py:165]INFO: Method: mcdropout Dataset: concrete
2025-05-01 23:50:32,502 - [run_mcdropout.py:70]INFO: Best prior precision found: 0.001
2025-05-01 23:50:36,405 - [run_mcdropout.py:109]INFO: Final test performance: MSE=0.071, LogLik=-2.730
2025-05-01 23:50:36,405 - [run_mcdropout.py:165]INFO: Method: mcdropout Dataset: energy
2025-05-01 23:51:27,626 - [run_mcdropout.py:70]INFO: Best prior precision found: 0.01
2025-05-01 23:51:31,396 - [run_mcdropout.py:109]INFO: Final test performance: MSE=0.045, LogLik=-1.478
2025-05-01 23:51:31,396 - [run_mcdropout.py:165]INFO: Method: mcdropout Dataset: kin8nm
2025-05-01 23:52:20,459 - [run_mcdropout.py:70]INFO: Best prior precision found: 1.0
2025-05-01 23:52:25,307 - [run_mcdropout.py:109]INFO: Final test performance: MSE=0.074, LogLik=1.316
2025-05-01 23:52:25,307 - [run_mcdropout.py:165]INFO: Method: mcdropout Dataset: naval-propulsion-plant
2025-05-01 23:53:21,261 - [run_mcdropout.py:70]INFO: Best prior precision found: 0.01
2025-05-01 23:53:26,347 - [run_mcdropout.py:109]INFO: Final test performance: MSE=0.223, LogLik=4.978
2025-05-01 23:53:26,348 - [run_mcdropout.py:165]INFO: Method: mcdropout Dataset: power-plant
2025-05-01 23:54:16,687 - [run_mcdropout.py:70]INFO: Best prior precision found: 0.001
2025-05-01 23:54:21,320 - [run_mcdropout.py:109]INFO: Final test performance: MSE=0.057, LogLik=-2.791
2025-05-01 23:54:21,320 - [run_mcdropout.py:165]INFO: Method: mcdropout Dataset: wine-quality-red
2025-05-01 23:55:02,873 - [run_mcdropout.py:70]INFO: Best prior precision found: 10.0
2025-05-01 23:55:07,414 - [run_mcdropout.py:109]INFO: Final test performance: MSE=0.474, LogLik=-0.849
2025-05-01 23:55:07,415 - [run_mcdropout.py:165]INFO: Method: mcdropout Dataset: yacht
2025-05-01 23:55:52,243 - [run_mcdropout.py:70]INFO: Best prior precision found: 0.01
2025-05-01 23:55:55,702 - [run_mcdropout.py:109]INFO: Final test performance: MSE=0.001, LogLik=-0.425

```
