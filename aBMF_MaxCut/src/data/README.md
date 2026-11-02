Place standard Gset graph files in this directory. Each file must begin with the usual header

n m

followed by m lines of

i j weight

The function `example_gset` now discovers these files automatically, reads `n` from the header, and selects a compact rebuttal configuration near n = 3000, 10000, and 20000 with rank 50, plus rank 100 on the largest selected graph when feasible.

Do not rename files to `G_small`, `G_large`, or `G_largest`; those names were placeholders in v1-v2 and are no longer used.

When no external Gset files are present, `example_gset` automatically runs `tiny_gset.txt` as a short smoke test. The tiny result is only for checking the program and must not be reported in the rebuttal.

Negative edge weights are allowed. They occur in valid signed-weight Gset instances; `example_gset` runs them with `require_nonnegative_weights=false`.
