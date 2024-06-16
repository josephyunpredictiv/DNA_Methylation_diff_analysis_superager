# -*- coding: utf-8 -*-
"""
Original file is located at (Not avaliable to public)
    https://colab.research.google.com/drive/1cR-cx0vJ3xHVVJZdkaM1RCqQfYvfipOa
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib as mpl

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

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

gini_values = model.feature_importances_
feature_importances_df = pd.DataFrame({
    'Feature': X_features_list,
    'Importance': gini_values
})
feature_importances_df = feature_importances_df[feature_importances_df['Importance'] > 0]
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
feature_importances_df['Order'] = range(1, len(feature_importances_df) + 1)

feature_importances_df[["Feature", "Importance"]].to_csv("Feature_Importance.csv", index=False)

plt.figure(figsize=(40, 6), dpi=300)
plt.plot(feature_importances_df['Order'], feature_importances_df['Importance'], marker='o', linestyle='-', color='b')
plt.title('Feature Gini Importance by Order')
plt.xlabel('Feature')
plt.ylabel('Gini Importance')
plt.xticks(feature_importances_df['Order'], feature_importances_df['Feature'], rotation=45, ha='right')
plt.rc('xtick', labelsize=10)
plt.grid(True)
plt.savefig('Feature_importance.png', format='png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(150, 6), dpi=300)
plt.plot(feature_importances_df['Order'], feature_importances_df['Importance'], marker='o', linestyle='-', color='b')
plt.title('Feature Gini Importance by Order')
plt.xlabel('Feature')
plt.ylabel('Gini Importance')
plt.xticks(feature_importances_df['Order'], feature_importances_df['Feature'], rotation=45, ha='right')
plt.rc('xtick', labelsize=10)
plt.grid(True)
plt.savefig('Feature_importance_pvalue_0.05.png', format='png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 8), dpi=300)
plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances from RandomForestClassifier')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.savefig('Feature_importance_pvalue_0.05_sideways.png', format='png', bbox_inches='tight')
plt.show()

gini_values = model.feature_importances_
feature_importances_df = pd.DataFrame({
    'Feature': X_features_list,
    'Importance': gini_values
})
feature_importances_df = feature_importances_df[feature_importances_df['Importance'] > 0]
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False).head(50)
feature_importances_df['Order'] = range(1, len(feature_importances_df) + 1)

plt.figure(figsize=(10, 8))
plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances from RandomForestClassifier')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.show()

plt.figure(figsize=(20, 6), dpi=300)
plt.plot(feature_importances_df['Order'], feature_importances_df['Importance'], marker='o', linestyle='-', color='b')
plt.title('Feature Gini Importance by Order')
plt.xlabel('Feature')
plt.ylabel('Gini Importance')
plt.xticks(feature_importances_df['Order'], feature_importances_df['Feature'], rotation=45, ha='right')
plt.rc('xtick', labelsize=10)
plt.grid(True)
plt.savefig('Feature_importance_top50_pvalue_0.05.png', format='png', bbox_inches='tight')
plt.show()

y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:, 1]

# Calculate the ROC curve and AUC
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
plt.savefig('ROC_curve.png', format='png', bbox_inches='tight')
plt.show()

# Print the AUC
print(f'ROC AUC: {roc_auc:.2f}')

"""# hyperparameter tuning"""

#0.65 when bootstrap=False, min_samples_leaf=4, min_samples_split=10, random_state=42

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

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize a RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV with the RandomForestClassifier and the parameter grid
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)

# Perform the grid search on the training data
grid_search.fit(x_train, y_train.values.ravel())

# Retrieve the best model from grid search
best_rf = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred_proba = best_rf.predict_proba(x_test)[:, 1]

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
plt.savefig('ROC_curve.png', format='png', bbox_inches='tight')
plt.show()

# Print the best hyperparameters and the AUC
print(f'Best hyperparameters: {grid_search.best_params_}')
print(f'ROC AUC: {roc_auc:.2f}')

best_rf

gini_values = model.feature_importances_
feature_importances_df = pd.DataFrame({
    'Feature': X_features_list,
    'Importance': gini_values
})
feature_importances_df = feature_importances_df[feature_importances_df['Importance'] > 0]
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
feature_importances_df['Order'] = range(1, len(feature_importances_df) + 1)

feature_importances_df[["Feature", "Importance"]].to_csv("Feature_Importance.csv", index=False)

plt.figure(figsize=(40, 6), dpi=300)
plt.plot(feature_importances_df['Order'], feature_importances_df['Importance'], marker='o', linestyle='-', color='b')
plt.title('Feature Gini Importance by Order')
plt.xlabel('Feature')
plt.ylabel('Gini Importance')
plt.xticks(feature_importances_df['Order'], feature_importances_df['Feature'], rotation=45, ha='right')
plt.rc('xtick', labelsize=10)
plt.grid(True)
plt.savefig('Feature_importance.png', format='png', bbox_inches='tight')
plt.show()

df.columns[:-1].tolist()[0:10]

