import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the dataset into 70% training and 30% testing (to reduce overfitting)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the data (needed for SVM and k-NN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dictionary to store model performance
results = {}

# Logistic Regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
results['Logistic Regression'] = accuracy_score(y_test, y_pred)
print("\nLogistic Regression:")
print(classification_report(y_test, y_pred))

# k-Nearest Neighbors (k=5)
k_nn = KNeighborsClassifier(n_neighbors=5)
k_nn.fit(X_train, y_train)
y_pred = k_nn.predict(X_test)
results['k-NN'] = accuracy_score(y_test, y_pred)
print("\nk-NN:")
print(classification_report(y_test, y_pred))

# Decision Tree (pruned by setting max_depth)
dec_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
dec_tree.fit(X_train, y_train)
y_pred = dec_tree.predict(X_test)
results['Decision Tree'] = accuracy_score(y_test, y_pred)
print("\nDecision Tree:")
print(classification_report(y_test, y_pred))

# Random Forest (reducing number of trees and setting max_depth)
rand_forest = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
rand_forest.fit(X_train, y_train)
y_pred = rand_forest.predict(X_test)
results['Random Forest'] = accuracy_score(y_test, y_pred)
print("\nRandom Forest:")
print(classification_report(y_test, y_pred))

# Support Vector Machine (RBF Kernel for non-linearity)
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
results['SVM'] = accuracy_score(y_test, y_pred)
print("\nSupport Vector Machine:")
print(classification_report(y_test, y_pred))

# Display results
print("\nFinal Accuracy Scores:")
for model, acc in results.items():
    print(f"{model}: {acc:.2f}")
