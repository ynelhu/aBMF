# Gamma sensitivity v1

- Added a controlled fixed-gamma sweep with `gamma = c * gamma_ref`.
- Defined `gamma_ref = 0.5 * ||grad f_nc(Z0)||_F + epsilon_gamma` from the matched initial lift.
- Added multipliers `c = [1, 2, 4, 8, 16]` in full mode.
- Added condition-level, overall, and matched-against-`c=1` CSV summaries.
- Added four separate sensitivity plots.
- Kept the original main NN benchmark unchanged.
