from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeCV
import numpy as np


diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X,y)
feature_names = np.array(diabetes.feature_names)

sfs_forward = SequentialFeatureSelector(ridge, n_features_to_select=2, direction="forward").fit(X,y)

print(feature_names[sfs_forward.get_support()])

