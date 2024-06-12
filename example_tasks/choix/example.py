import choix
import numpy as np

np.set_printoptions(precision=3, suppress=True)

n_items = 8
data = [
    (7, 3), (2, 0), (5, 2), (4, 2), (2, 1),
    (4, 5), (6, 3), (5, 4), (7, 0), (2, 3),
    (4, 0), (0, 4), (6, 5), (3, 2), (3, 4),
    (3, 4), (5, 2), (7, 3), (7, 6), (6, 5),
]

logit_mean, logit_cov = choix.ep_pairwise(n_items, data, 0.1, model="logit")
print("Logit mean:")
print(logit_mean)
print("\nLogit covariance:")
print(logit_cov)
print("\ntotal Logit variance: {:3f}".format(np.trace(logit_cov)))

probit_mean, probit_cov = choix.ep_pairwise(n_items, data, 0.1, model="probit")
print("Probit mean:")
print(probit_mean)
print("\nProbit covariance:")
print(probit_cov)
print("\ntotal Probit variance: {:3f}".format(np.trace(probit_cov)))

print(f"\n\n\nScore: {np.trace(probit_cov)}\n\n\n")
