import numpy as np
from sklearn.linear_model import LinearRegression

from seglearn.pipe import Pype
from seglearn.split import temporal_split
from seglearn.transform import FeatureRep, SegmentXYForecast, last

t = np.arange(5000) / 100.
y = np.sin(t) * t * 2.5 + t * t

# with forecasting, X can include the target
X = np.stack([t, y], axis=1)

# remember for a single time series, we need to make a list
X = [X]
y = [y]

# split the data along the time axis (our only option since we have only 1 time series)
X_train, X_test, y_train, y_test = temporal_split(X, y, test_size=0.25)

# create a feature representation pipeline
# setting y_func = last, and forecast = 200 makes us predict the value of y
# 200 samples ahead of the segment
# other reasonable options for y_func are ``mean``, ``all`` (or create your own function)
# see the API documentation for further details
clf = Pype([('segment', SegmentXYForecast(width=200, overlap=0.5, y_func=last, forecast=200)),
            ('features', FeatureRep()),
            ('lin', LinearRegression())])

# fit and score
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

print("N series in train: ", len(X_train))
print("N series in test: ", len(X_test))
print("N segments in train: ", clf.N_train)
print("N segments in test: ", clf.N_test)
print(f"\n\n\nScore: {score}\n\n\n")
