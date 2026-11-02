# Controlled gamma-multiplier sensitivity on the nonconvex NN surrogate

This addition isolates the effect of overestimating one theorem-motivated **reference scale** while keeping the model, sample, perturbation radius, rank, initialization, solver, stopping rule, and line-search settings fixed.

For each matched run, the code constructs the same initialization used by `solve_abmf_nc` and computes

\[
\gamma_{\mathrm{ref}}
=
\frac12\|\nabla_Z f_{\mathrm{nc}}(Z_0)\|_F+\epsilon_\gamma.
\]

It then solves `aBMF-fixed` with

\[
\gamma=c\,\gamma_{\mathrm{ref}},
\qquad
c\in\{1,2,4,8,16\}.
\]

Only `c` changes inside each matched sample/epsilon/rank/trial group.

## Important interpretation

The NN surrogate is nonconvex, so `gamma_ref` is a theorem-motivated gradient scale, not a theorem-certified lower bound for this experiment. In a rebuttal, describe the result as a **controlled penalty-scale sensitivity study outside the formal assumptions of Theorem 1**.

## Run

First perform static checks and a short smoke test:

```matlab
self_check_gamma_sensitivity;
aggregate = example_nn_gamma_sensitivity_quick;
```

Then run the full experiment:

```matlab
aggregate = example_nn_gamma_sensitivity;
```

The full design uses the same three samples, two perturbation radii, two ranks, and ten matched trials as the main NN benchmark. It runs five fixed-gamma multipliers, giving 600 total runs.

## Outputs

The `results_gamma_sensitivity` folder contains:

- `gamma_sensitivity_raw.csv`: every individual run;
- `gamma_sensitivity_by_condition.csv`: mean/std for each sample, epsilon, rank, and multiplier;
- `gamma_sensitivity_aggregate.csv`: compact overall table grouped by multiplier;
- `gamma_sensitivity_matched_vs_c1.csv`: matched changes relative to `c=1`;
- `gamma_sensitivity_results.mat`: complete MATLAB structures;
- four separate PNG sensitivity plots for objective, disagreement, iterations, and runtime.

The most useful controlled quantities for the rebuttal are:

- surrogate objective difference relative to `c=1`;
- relative-factor-disagreement ratio relative to `c=1`;
- iteration-count ratio relative to `c=1`;
- runtime ratio relative to `c=1`;
- diagnostic success-rate change.

A typical interpretation is that increasing `c` may reduce factor disagreement, while excessive penalties can make the block optimization more conservative and increase runtime or iteration count. State this only after the actual CSV results confirm the trend.
