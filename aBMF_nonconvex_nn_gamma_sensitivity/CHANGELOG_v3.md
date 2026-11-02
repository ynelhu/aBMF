# Version 3 changes

- `aBMF-our` now uses
  \[
  \gamma_k=\max\left\{\gamma_{k-1},\frac12\|\nabla f_{\rm nc}(Z_k)\|_F+\epsilon_\gamma\right\}.
  \]
- The same safeguard is applied to the initial penalty before the first factor update.
- The full rebuttal benchmark now uses ten trials and at most 3000 iterations.
- `aggregate_rebuttal_results.m` remains included as a separate callable function.
- The interpretation remains empirical: the nonconvex loss and adaptive penalty are outside the formal assumptions of Theorem 1.
