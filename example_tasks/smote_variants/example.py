import smote_variants as sv
import imblearn.datasets as imb_datasets

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import platform

libras= imb_datasets.fetch_datasets()['libras_move']
X, y= libras['data'], libras['target']

oversampler = ('smote_variants', 'MulticlassOversampling',
                {'oversampler': 'distance_SMOTE', 'oversampler_params': {}})

classifier = ('sklearn.neighbors', 'KNeighborsClassifier', {})

# Constructing a pipeline with oversampling and classification as the last step
model= Pipeline([('scale', StandardScaler()),
                ('clf', sv.classifiers.OversamplingClassifier(oversampler, classifier))])

param_grid= {'clf__oversampler':[('smote_variants', 'distance_SMOTE', {'proportion': 0.5}),
                                ('smote_variants', 'distance_SMOTE', {'proportion': 1.0}),
                                ('smote_variants', 'distance_SMOTE', {'proportion': 1.5})]}

# Specifying the gridsearch for model selection
grid= GridSearchCV(model,
                  param_grid=param_grid,
                  cv=3,
                  n_jobs=1,
                  verbose=0,
                  scoring='accuracy')

# Fitting the pipeline
grid.fit(X, y)

print(f"\n\n\nScore: {grid.best_score_}\n\n\n")
