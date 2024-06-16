# -*- coding: utf-8 -*-
"""superager_neural_network.ipynb
Original file is located at (not avalible to public)
    https://colab.research.google.com/drive/1JD1I0y4U9OaSkPputIrlzpuLeyrycXd3
"""

!pip install keras

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin

"""# Random Forest and Gini Index"""

df = pd.read_csv("promoter_M_values_filtered_by_pvalue_0.05_transposed.csv", index_col=0)

df

X_features_list = df.columns[:-1].tolist()
Y_feature_list = []
Y_feature_list.append(df.columns[-1])

X_features = pd.DataFrame(df, columns = X_features_list)
Y_feature = pd.DataFrame(df, columns = Y_feature_list)
Y_feature['group'] = Y_feature['group'].replace({'s': 0, 't': 1})

x_train, x_test, y_train, y_test = train_test_split(
   X_features,
   Y_feature,
   test_size=0.20,
   random_state=4)

# Define a function to create the neural network model
# 'optimizer': 'adam', 'epochs': 50, 'batch_size': 40, 'activation': 'relu'
# auc 0.72
def create_model(optimizer='adam', activation='relu'):
    model = Sequential()
    model.add(Dense(12, input_dim=x_train.shape[1], activation=activation))
    model.add(Dense(8, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['AUC'])
    return model

# Custom wrapper to use Keras model with scikit-learn
class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, optimizer='adam', activation='relu', batch_size=10, epochs=50, verbose=1):
        self.optimizer = optimizer
        self.activation = activation
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.model = None

    def fit(self, X, y):
        self.model = create_model(optimizer=self.optimizer, activation=self.activation)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")

    def predict_proba(self, X):
        return self.model.predict(X)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'activation': ['relu', 'tanh'],
    'batch_size': [10, 20, 40],
    'epochs': [50, 100]
}

# Initialize the custom KerasClassifierWrapper
model = KerasClassifierWrapper(verbose=1)

# Initialize RandomizedSearchCV with the custom KerasClassifierWrapper and the parameter grid
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=20, cv=3, scoring='roc_auc', n_jobs=-1, verbose=10, random_state=42)

# Perform the random search on the training data
random_search.fit(x_train, y_train.values.ravel())

# Retrieve the best model from random search
best_nn = random_search.best_estimator_

# Make predictions on the test set using the best model
y_pred_proba = best_nn.predict_proba(x_test)

# Calculate the ROC curve and AUC for the best model
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Plot the ROC curve
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('ROC_curve_nn_improved.png', format='png', bbox_inches='tight')
plt.show()

# Print the best hyperparameters and the AUC
print(f'Best hyperparameters: {random_search.best_params_}')
print(f'ROC AUC: {roc_auc:.2f}')

