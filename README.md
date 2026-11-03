# aBMF

MATLAB code for **Asymmetric Factorization for Low-Rank PSD Learning: When Is the Relaxation Exact?**, NeurIPS 2026.

Enliang Hu, Juho Kannala, Guoying Zhao, Quanming Yao, and Kun Yue.

The paper studies the asymmetric formulation

$$
\min_{X,Y}\ f(XY^\top)+\frac{h(X)+h(Y)}{2}
+\frac{\gamma}{2}\|X-Y\|_F^2,
$$

and gives sufficient conditions for critical points to have equal factors. The experiments compare penalty choices and their effect on factor agreement, accuracy, and runtime.

## Repository contents

| Directory | Contents |
| --- | --- |
| `aBMF_two toy examples/` | Two scalar examples of penalty updates |
| `aBMF_TopK/` | Synthetic top-k PSD approximation |
| `aBMF_MaxCut/` | Max-Cut experiments on Gset graphs; MATLAB and Slurm entry points |
| `aBMF_nonconvex_nn_verification/` | A nonconvex neural-network verification surrogate |
| `aBMF_nonconvex_nn_gamma_sensitivity/` | Fixed-penalty multiplier experiments for the same NN surrogate |
| `aBMF_Low-rank Matrix Recovery/` | An AM example on MovieLens-1M |

This README covers the folders above. Dedicated drivers for the paper's sNMF, link-prediction, and matrix-sensing experiments are not included in this release. The MovieLens folder is an example, rather than the full multi-method, three-dataset comparison.

## Requirements

- MATLAB R2022b is recommended.
- The MovieLens example calls `normrnd` and requires the Statistics and Machine Learning Toolbox.
- The other examples listed below use MATLAB functions and the supplied data. 
- The optional Max-Cut cluster launcher requires Slurm, Bash, and Python 3.8 or newer. Its default MATLAB module is `matlab-R2022b`.

Download the repository, or clone it:

```bash
git clone https://github.com/ynelhu/aBMF.git
cd aBMF
```

Run each MATLAB block below from the repository root, unless stated otherwise. Use a fresh output directory for an independent run. The toy and MovieLens scripts clear the MATLAB workspace.

## 1. Scalar examples

```matlab
cd('aBMF_two toy examples');
run('code_for_first_toy_example.m');
cd('..');
```

Run the second example separately if you want to retain the first example's figures:

```matlab
cd('aBMF_two toy examples');
code_for_second_toy_example;
cd('..');
```

Each example creates four MATLAB figures. Figures are not automatically exported. The first example includes a penalty schedule that causes numerical overflow; this behavior is part of the comparison.

## 2. Top-k PSD approximation

Use the benchmark function directly. A small test is:

```matlab
cd('aBMF_TopK');
summary = run_topk_benchmark( ...
    'fixed_n',200,'rank_list',5, ...
    'fixed_rank',3,'n_list',100, ...
    'matrix_types',{'PSD'},'num_trials',1, ...
    'max_iter',30,'data_mode','matrix_free', ...
    'output_dir',fullfile(pwd,'results','topk_smoke'));
cd('..');
```

The two paper-sized conditions are `(n,r) = (10000,50)` and `(50000,1000)`, with five trials per condition:

```matlab
cd('aBMF_TopK');
summary = run_topk_benchmark( ...
    'fixed_n',50000,'rank_list',1000, ...
    'fixed_rank',50,'n_list',10000, ...
    'matrix_types',{'PSD'},'num_trials',5, ...
    'seed',1,'max_iter',1000,'tol',1e-6, ...
    'nu',0.9,'epsilon0',1e-6, ...
    'data_mode','matrix_free','save_histories',true, ...
    'output_dir',fullfile(pwd,'results','topk_main'));
cd('..');
```

The large condition requires substantially more memory and time than the small test. Synthetic matrices are generated internally; matrix-free mode uses matrix–factor products without storing a dense `n`-by-`n` matrix.

The driver compares `sBMF-GD`, `sBMF-GN`, `aBMF-adp`, and `aBMF-our`. Outputs include `all_trial_results.csv`, `summary_mean_std.csv`, `config.json`, sweep summaries, and plots. With `save_histories=true`, it also saves per-trial histories.

## 3. Max-Cut

The data folder contains G20, G55, G67, and a small test graph. For a local test:

```matlab
cd(fullfile('aBMF_MaxCut','src'));
summary = run_maxcut_benchmark( ...
    'condition_list',{'data/tiny_gset.txt',2}, ...
    'num_trials',1,'rounds',20,'max_iter',100, ...
    'full_sdp',false, ...
    'output_dir',fullfile(pwd,'maxcut_results_quick'));
cd('../..');
```

The paper uses G55 at width 50 and G67 at widths 50 and 100, with ten trials and 100 shared rounding hyperplanes per condition:

```matlab
cd(fullfile('aBMF_MaxCut','src'));
conditions = {'data/G55.txt',50; ...
              'data/G67.txt',50; ...
              'data/G67.txt',100};
summary = run_maxcut_benchmark( ...
    'condition_list',conditions,'num_trials',10, ...
    'rounds',100,'max_iter',3000,'tol',1e-6, ...
    'constraint_tol',1e-6,'full_sdp',false, ...
    'gamma_nu',0.9,'gamma_mix_epsilon0',1e-6, ...
    'output_dir',fullfile(pwd,'maxcut_results_main'));
cd('../..');
```

Local runs write `raw_trial_results.csv`, `summary_mean_std.csv`, `raw_results_and_assignments.mat`, and `config.json` to the chosen output directory. The driver supports signed graph weights and reports objectives on the original graph scale.

For the supplied Slurm setup:

```bash
cd aBMF_MaxCut
bash RUN_FORMAL.sh --run-dir "$PWD/results/main_run"
```

The default allocation is the `CU` partition, 64 CPU cores, and 64 GB RAM per task. Adjust the resource options and MATLAB module for your cluster. Check progress with:

```bash
bash STATUS.sh results/main_run
```

Aggregated results are written under `statistics/`; result packages are placed under `exports/` within the run directory. The full experiment has 120 method records. See `aBMF_MaxCut/README_CN.txt` for launcher options.

## 4. Nonconvex NN surrogate

The supplied data contain a fixed 12-input, 16-hidden-unit ReLU classifier and Digits 3-versus-8 samples in PCA-standardized feature coordinates. No download is needed for this benchmark.

```matlab
cd('aBMF_nonconvex_nn_verification');
self_check;
summary = example_quick;
cd('..');
```

For the full comparison:

```matlab
cd('aBMF_nonconvex_nn_verification');
summary = example_nonconvex_nn;
cd('..');
```

The full setting uses three samples, perturbation radii `{0.6,0.8}`, factor widths `{8,16}`, and ten trials per condition. Four methods give 480 runs in total, with a maximum of 3000 iterations per run. Methods share initialization within each trial and common iteration and line-search settings.

New results are saved in `results/`, including `raw_results.csv`, `summary_mean_std.csv`, `rebuttal_aggregate.csv`, and `results_and_histories.mat`. The supplied experiment records are in `results-1/`.

Both the loss and diagnostics use `Z = (X*Y' + Y*X')/2`. The reported objective excludes the factor-agreement penalty. Success means a finite objective and affine, inequality, complementarity, and relative factor-disagreement residuals at most `0.05`; PSD violation is reported separately.

Perturbations are applied in feature space, not image-pixel space. This nonconvex surrogate is outside the convex-loss assumptions of the exactness theorem and does not provide robustness certificates.

### Penalty multiplier experiment

```matlab
cd('aBMF_nonconvex_nn_gamma_sensitivity');
self_check_gamma_sensitivity;
aggregate = example_nn_gamma_sensitivity_quick;
cd('..');
```

For the full sweep:

```matlab
cd('aBMF_nonconvex_nn_gamma_sensitivity');
aggregate = example_nn_gamma_sensitivity;
cd('..');
```

The full sweep uses fixed penalties `gamma = c*gamma_ref`, with `c = [1,2,4,8,16]`, and contains 600 runs. Here `gamma_ref` is a gradient-based reference scale for the nonconvex experiment, not a certified penalty threshold.

Outputs are written to `results_gamma_sensitivity/`. Back up the supplied results in this directory before rerunning the experiment.

## 5. MovieLens example

```matlab
cd('aBMF_Low-rank Matrix Recovery');
Test_me;
cd('..');
```

This example runs `AM_our` on the bundled MovieLens-1M matrix. Its defaults are:

| Setting | Value |
| --- | --- |
| Rating normalization | `data/max(abs(data(:)))` |
| Training split | Each observed rating is assigned to training independently with probability 0.7 |
| Factor width | 2 |
| Regularization `eta` | `1e-3` |
| Initial penalty | 1 |
| Maximum iterations | 100 |
| Objective-change tolerance | `1e-4`, over three iterations |
| Initialization | Independent Gaussian `X0` and `Y0` |

The script creates figures for the penalty, factor difference, testing RMSE, and objective. Testing RMSE is measured on normalized ratings. Predictions use the two user/item blocks of `V = (X+Y)/2`.

These are demonstration settings. The paper's MovieLens comparison uses 3000 iterations at most, 50 trials for 100K and 1M, and ten trials for 10M. This folder does not contain the complete table-generation driver or all compared methods.

## Penalty strategies

In the paper, `adp` denotes the adaptive strategy, `agd` the algorithm-dependent strategy, `pca` the proportional-to-computational-accuracy strategy, and `our` the proposed strategy. Some drivers use `aBMF-our` and `aBMF-adp` for methods labelled `AM_our` and `AM_adp` in the paper.

Penalty updates depend on the problem. Top-k and Max-Cut use the mixed directional/norm update; the NN surrogate uses a nondecreasing gradient-norm update. See the relevant solver for its exact rule. The fixed-penalty convergence theorem does not automatically cover these changing-penalty sequences.

## Data sources and use

- MovieLens: [GroupLens](https://grouplens.org/datasets/movielens/).
- Gset graphs: [Stanford Gset collection](https://web.stanford.edu/~yyye/yyye/Gset/).
- Digits: [scikit-learn Digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). Prepared features and fixed classifier weights are supplied with the NN examples.

Third-party data remain subject to their original terms of use. For questions about code reuse or commercial authorization, contact **ynel.hu@gmail.com**.

## Citation

```bibtex
@inproceedings{hu2026asymmetric,
  title     = {Asymmetric Factorization for Low-Rank PSD Learning: When Is the Relaxation Exact?},
  author    = {Hu, Enliang and Kannala, Juho and Zhao, Guoying and Yao, Quanming and Yue, Kun},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
