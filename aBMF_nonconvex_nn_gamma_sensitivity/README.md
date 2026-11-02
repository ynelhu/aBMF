# Nonconvex PSD-Lifted Neural-Network Verification Stress Test

This MATLAB project provides a compact empirical stress test for asymmetric low-rank PSD factorization on a genuinely nonconvex ReLU-network verification surrogate.

## Rebuttal positioning

The experiment is **outside the convex-loss assumptions of the paper's exactness theorem**. It should be described only as empirical evidence. The factorized values are not formal robustness certificates.

The nonconvexity is introduced through the ReLU complementarity penalty

\[
\Psi_{\mathrm{comp}}(Z)
=
\sum_i
\left[a_i(Z)\bigl(a_i(Z)-s_i(Z)\bigr)\right]^2.
\]

The lifted surrogate is

\[
f_{\mathrm{nc}}(Z)
=
\text{margin}(Z)
+\rho_{\mathrm{eq}}\Psi_{\mathrm{eq}}(Z)
+\rho_{\mathrm{ineq}}\Psi_{\mathrm{ineq}}(Z)
+\rho_{\mathrm{comp}}\Psi_{\mathrm{comp}}(Z)
+\lambda\|Z\|_F^2.
\]

The compared low-rank models are:

- `sBMF`: \(Z=XX^\top\);
- `aBMF-fixed`: \(Z=\operatorname{sym}(XY^\top)\) with fixed \(\gamma\);
- `aBMF-adp`: multiplicative adaptive baseline;
- `aBMF-our` (AMour): a nondecreasing theorem-motivated spectral safeguard.

For `aBMF-our`, the penalty update is

\[
\widehat\gamma_k
=
\frac12\|\nabla_Z f_{\mathrm{nc}}(Z_k)\|_F+\epsilon_\gamma,
\qquad
\gamma_k
=
\max\{\widehat\gamma_k,\gamma_{k-1}\}.
\]

The same safeguard is applied at initialization before the first factor step. The Frobenius norm conservatively controls the spectral term that appears in Theorem 1. Since the lifted verification objective is nonconvex and the penalty changes during the run, this remains a theorem-motivated empirical rule rather than a theorem-covered convergence result.

## Data and network

The package contains `data/digits_3_8_prepared.mat`, produced from the scikit-learn Digits dataset restricted to digits 3 and 8.

- input dimension after PCA and standardization: 12;
- hidden ReLU units: 16;
- binary test accuracy of the bundled fixed network: approximately 99.07%;
- selected verification samples: correctly classified points near the 25th, 50th, and 75th signed-margin percentiles;
- perturbation radii: \(0.6\) and \(0.8\);
- lifted dimension: \(p=1+12+16=29\).

The data and network are bundled so that no MATLAB toolbox or internet download is required.

## Running in MATLAB or VS Code

Open the project folder in MATLAB, or open it in VS Code with the MATLAB extension and set the working directory to this folder.

Run a short smoke test:

```matlab
summary = example_quick;
```

Run the full rebuttal benchmark:

```matlab
summary = example_nonconvex_nn;
```

## Default full benchmark

The `full` mode uses:

- methods: `sBMF`, `aBMF-fixed`, `aBMF-adp`, and `aBMF-our`;
- three selected test samples;
- perturbation radii `0.6` and `0.8`;
- ranks `8` and `16`;
- ten matched trials;
- at most `3000` iterations;
- identical data, initialization, stopping rule, and line-search parameters across matched methods.

This gives 120 runs per method and 480 runs in total. Run `self_check` and `example_quick` before the full benchmark.

## Outputs

The `results` directory contains:

- `raw_results.csv`;
- `summary_mean_std.csv`;
- `rebuttal_aggregate.csv`;
- `results_and_histories.mat`;
- `gamma_trajectory.png`;
- `complementarity_trajectory.png`;
- `factor_disagreement_trajectory.png`.

The main reported diagnostics are:

- nonconvex surrogate objective;
- signed verification-margin surrogate;
- affine-constraint residual;
- inequality residual;
- ReLU complementarity residual;
- relative factor disagreement;
- PSD violation of \(\operatorname{sym}(XY^\top)\);
- runtime and common diagnostic success rate.

## Important interpretation

Do not state that this experiment extends the theorem to nonconvex losses. Do not label a positive factorized margin as a certified robustness bound. A suitable rebuttal description is:

> We additionally evaluated the methods on a PSD-lifted ReLU verification surrogate with a nonconvex complementarity penalty. This is an empirical stress test outside the convex-loss assumptions of our theory and is not used as a formal robustness certificate.

## Package correction

The project includes `aggregate_rebuttal_results.m`, which constructs the compact method-level table written to `results/rebuttal_aggregate.csv`.

## Controlled fixed-gamma multiplier sensitivity

A separate controlled sensitivity experiment is included for reviewer questions about penalty overestimation. Read `GAMMA_SENSITIVITY_README.md`, then run:

```matlab
self_check_gamma_sensitivity;
aggregate = example_nn_gamma_sensitivity_quick;
aggregate = example_nn_gamma_sensitivity;
```

This experiment keeps all non-gamma variables fixed and tests `gamma = c * gamma_ref` for `c = [1, 2, 4, 8, 16]`. Because the NN surrogate is nonconvex, `gamma_ref` must be described as a theorem-motivated reference scale rather than a theorem-certified lower bound.
