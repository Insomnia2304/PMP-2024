import numpy as np
import pymc as pm

obs = [56, 60, 58, 55, 57, 59, 61, 56, 58, 60]
x = np.mean(obs)

with pm.Model() as model:
    mu = pm.Normal('mu', mu=x, sigma=100)
    sigma = pm.HalfNormal('sigma', sigma=10)
    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=obs)

with model:
    trace = pm.sample(1000, return_inferencedata=True)

print(f'Intervalul HDI pentru mu: {pm.hdi(trace.posterior["mu"], hdi_prob=0.95)["mu"].values}')
print(f'Intervalul HDI pentru sigma: {pm.hdi(trace.posterior["sigma"], hdi_prob=0.95)["sigma"].values}')
