import numpy as np
import pandas as pd
import entropy as ent
import stochastic.processes.noise as sn

# Generate time-series with increasing B exponent
betas = np.arange(-2, 2.1, 0.1)
n_ts = betas.size
n_samples = 1000
sf = 10
ts = np.empty((n_ts, n_samples + 1))

for i, b in enumerate(betas):
    rng = np.random.default_rng(42)
    ts[i] = sn.ColoredNoise(beta=b, rng=rng).sample(n_samples)


df = pd.DataFrame()

for i in range(n_ts):
    df = df.append({
        'PermEnt': ent.perm_entropy(ts[i], order=3, normalize=True),
        'SVDEnt' : ent.svd_entropy(ts[i], order=3, normalize=True),
        'SpecEnt' : ent.spectral_entropy(ts[i], sf, normalize=True, 
                                         method='welch', nperseg=50),
        #'AppEnt': ent.app_entropy(ts[i], order=2),
        'SampleEnt': ent.sample_entropy(ts[i], order=2),
        'PetrosianFD': ent.petrosian_fd(ts[i]),
        'KatzFD': ent.katz_fd(ts[i]),
        'HiguchiFD': ent.higuchi_fd(ts[i]),
        'DFA': ent.detrended_fluctuation(ts[i])}, ignore_index=True)


# Describe
print(f"\n\n\nScore: {df.mean().round(6)['DFA']}\n\n\n")
