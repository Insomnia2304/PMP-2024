from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dimensiunea gridului
dimensiune_grid = (10, 10)

# Lista de culori predefinite
culori = [
    "red", "blue", "green", "yellow", 
    "purple", "orange", "pink", "cyan", 
    "brown", "lime"
]

# Citirea gridului
df = pd.read_csv('grid_culori.csv', header=None)
grid_culori = df.to_numpy()

# Generarea secvenței de culori observate
observatii = ['red', 'red', 'lime', 'yellow', 'blue']

# Mapare culori -> indecși
culoare_to_idx = {culoare: idx for idx, culoare in enumerate(culori)}
idx_to_culoare = {idx: culoare for culoare, idx in culoare_to_idx.items()}

# Transformăm secvența de observații în indecși
observatii_idx = [culoare_to_idx[c] for c in observatii]

# Definim stările ascunse ca fiind toate pozițiile din grid (100 de stări)
numar_stari = dimensiune_grid[0] * dimensiune_grid[1]
stari_ascunse = [(i, j) for i in range(dimensiune_grid[0]) for j in range(dimensiune_grid[1])]
stare_to_idx = {stare: idx for idx, stare in enumerate(stari_ascunse)}
idx_to_stare = {idx: stare for stare, idx in stare_to_idx.items()}

# Matrice de tranziție
transitions = np.zeros((numar_stari, numar_stari))

for i, j in stari_ascunse:
    vecini = [
        (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)  # sus, jos, stânga, dreapta
    ]
    vecini_valizi = [stare_to_idx[(x, y)] for x, y in vecini if 0 <= x < 10 and 0 <= y < 10]
    transitions[stare_to_idx[(i, j)], stare_to_idx[(i, j)]] = 0.25 # subpunctul 1.
    # transitions[stare_to_idx[(i, j)], stare_to_idx[(i, j)]] = 0.0 # subpunctul 2.
    for vecin in vecini_valizi: 
        transitions[stare_to_idx[(i, j)], vecin] = 0.75 / len(vecini_valizi) # subpunctul 1.
        # transitions[stare_to_idx[(i, j)], vecin] = 1 / len(vecini_valizi) # subpunctul 2.

# Matrice de emisie
emissions = np.zeros((numar_stari, len(culori)))
for i in range(dimensiune_grid[0]):
    for j in range(dimensiune_grid[1]):
        stare_idx = stare_to_idx[(i, j)]
        culoare = grid_culori[i, j]
        culoare_idx = culoare_to_idx[culoare]
        emissions[stare_idx, culoare_idx] = 1.0

model = hmm.CategoricalHMM(n_components=numar_stari)
model.startprob_ = np.ones(numar_stari) / numar_stari
model.transmat_ = transitions
model.emissionprob_ = emissions

# Rulăm algoritmul Viterbi pentru secvența de observații
logprob, secventa_stari = model.decode(np.array(observatii_idx).reshape(-1, 1), algorithm="viterbi")
print(f'Prob acestui drum: {np.exp(logprob)}')

# Convertim secvența de stări în poziții din grid
drum = [idx_to_stare[idx] for idx in secventa_stari]

# Vizualizăm drumul pe grid
fig, ax = plt.subplots(figsize=(8, 8))
for i in range(dimensiune_grid[0]):
    for j in range(dimensiune_grid[1]):
        culoare = grid_culori[i, j]
        ax.add_patch(plt.Rectangle((j, dimensiune_grid[0] - i - 1), 1, 1, color=culoare))
        ax.text(j + 0.5, dimensiune_grid[0] - i - 0.5, culoare, 
                color="white", ha="center", va="center", fontsize=8, fontweight="bold")

# Evidențiem drumul rezultat
for idx, (i, j) in enumerate(drum):
    ax.add_patch(plt.Circle((j + 0.5, dimensiune_grid[0] - i - 0.5), 0.3, color="black", alpha=0.7))
    ax.text(j + 0.5, dimensiune_grid[0] - i - 0.5, str(idx + 1), 
            color="white", ha="center", va="center", fontsize=10, fontweight="bold")

# Setări axă
ax.set_xlim(0, dimensiune_grid[1])
ax.set_ylim(0, dimensiune_grid[0])
ax.set_xticks(range(dimensiune_grid[1]))
ax.set_yticks(range(dimensiune_grid[0]))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(visible=True, color="black", linewidth=0.5)
ax.set_aspect("equal")
plt.title("Drumul rezultat al stărilor ascunse", fontsize=14)
plt.show()