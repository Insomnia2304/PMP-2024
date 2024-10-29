import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm

states = ["Dificil", "Mediu", "Usor"]
n_states = len(states)

observations = ["FB", "B", "S", "NS"]
n_observations = len(observations)

start_probability = np.array([1/3, 1/3, 1/3])

transition_probability = np.array([
    [0, 0.5, 0.5],    # Dificil
    [0.5, 0.25, 0.25], # Mediu
    [0.5, 0.25, 0.25]  # Usor
])

emission_probability = np.array([
    [0.1, 0.2, 0.4, 0.3],  # Dificil
    [0.15, 0.25, 0.5, 0.1], # Mediu
    [0.2, 0.3, 0.4, 0.1]    # Usor
])

model = hmm.CategoricalHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

observations_sequence = np.array([0, 0, 2, 1, 1, 2, 1, 1, 3, 1, 1, 2]).reshape(-1, 1)
hidden_states = model.predict(observations_sequence)
print("Most likely hidden states:", hidden_states)

# Plot the results for visualization
sns.set_style("darkgrid")
plt.plot(hidden_states, '-o', label="Hidden State")
plt.xlabel("Time Step")
plt.ylabel("Hidden State (Word or Silence)")
plt.yticks(ticks=range(n_states), labels=states)
plt.legend()
plt.title("Predicted Hidden States Over Time")
plt.show()
