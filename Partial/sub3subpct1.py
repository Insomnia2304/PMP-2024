# Am rulat codul in Google Colab, si am incarcat pozele cu rezultatele

import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

# 0 = ban
# 1 = stema
obs = [1, 0, 0, 0, 1, 0, 1, 0, 0, 0]
n_heads = sum(obs)
n_tails = len(obs) - n_heads

with pm.Model() as model:
    p = pm.Beta('p', alpha=1, beta=1)
    likelihood = pm.Bernoulli('likelihood', p=p, observed=obs)
    trace = pm.sample(2000, return_inferencedata=False)

plt.figure(figsize=(8, 6))
plt.hist(trace['p'], bins=30, density=True, alpha=0.6, color='g')
plt.title('Distribuția a Posteriori pentru Probabilitatea de Stema')
plt.xlabel('Probabilitatea de Stema')
plt.ylabel('Densitatea')
plt.grid(True)
plt.show()
