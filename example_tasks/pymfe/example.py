from sklearn.datasets import load_iris
from pymfe.mfe import MFE

# Load a dataset
data = load_iris()
y = data.target
X = data.data

###############################################################################
# Extracting default measures
mfe = MFE(groups = ['info-theory'])
mfe.fit(X, y)
ft = mfe.extract()


print(f"Score: {ft[1][ft[0].index('mut_inf.mean')]}")
