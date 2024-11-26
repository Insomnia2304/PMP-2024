import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

Y = [0, 5, 10]
theta = [0.2, 0.5]

# Daca Y este fixat, n este invers proportional cu theta.
# Daca theta este fixat, n este direct proportional cu Y.

fig, axes = plt.subplots(len(Y), len(theta), figsize=(10, 8), constrained_layout=True)

for i, y in enumerate(Y):
    for j, t in enumerate(theta):
        with pm.Model() as model:
            n = pm.Poisson('n', mu=10)
            observed_Y = pm.Binomial('y', n=n, p=t, observed=y)
            trace = pm.sample(1_000, return_inferencedata=True)

        az.plot_posterior(trace, var_names=['n'], hdi_prob=0.95, ax=axes[i, j])
        axes[i, j].set_title(f'Y={y}, theta={t}')

fig.suptitle('n posterior')
plt.show()
