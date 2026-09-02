# Loan-default example

Place an authorized loan-default CSV in this directory, identify the target column, and run:

```bash
vizjudge analyze loans.csv --target default --output ../../outputs/loan-default
```

Review high-scoring features for temporal leakage and protected-attribute proxies before using
them in a model.

