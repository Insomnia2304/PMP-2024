# Am rulat codul in Google Colab, si am incarcat pozele cu rezultatele

import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

# 0 = ban
# 1 = stema
obs = [1, 0, 0, 0, 1, 0, 1, 0, 0, 0]

with pm.Model() as model1:
    p = pm.Beta('p', alpha=1, beta=1)
    likelihood = pm.Bernoulli('likelihood', p=p, observed=obs)
    trace1 = pm.sample(2000, return_inferencedata=False)

posterior_samples = trace1['p']

obs = [1, 1, 0, 1, 0, 0, 1, 0, 0, 1]

with pm.Model() as model2:
    p = pm.Beta('p', alpha=np.mean(posterior_samples), beta=np.std(posterior_samples))

    likelihood = pm.Bernoulli('likelihood', p=p, observed=obs)
    trace2 = pm.sample(2000, return_inferencedata=False)

plt.figure(figsize=(8, 6))
plt.hist(trace2['p'], bins=30, density=True, alpha=0.6, color='g')
plt.title('Distribuția a Posteriori pentru Probabilitatea de Stema')
plt.xlabel('Probabilitatea de Stema')
plt.ylabel('Densitatea')
plt.grid(True)
plt.show()
