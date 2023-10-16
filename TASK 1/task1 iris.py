
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the Iris dataset
data = pd.read_csv('Iris.csv')

# Display basic information about the dataset
data.info()

# Display descriptive statistics of the dataset
data.describe()

# Count the occurrences of each species in the dataset
data['Species'].value_counts()

# Drop the 'Id' column and visualize pairplots
tmp = data.drop('Id', axis=1)
g = sns.pairplot(tmp, hue='Species', markers='+')
plt.show()

# Create violin plots for each feature
g = sns.violinplot(y='Species', x='SepalLengthCm', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='Species', x='SepalWidthCm', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='Species', x='PetalLengthCm', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='Species', x='PetalWidthCm', data=data, inner='quartile')
plt.show()

# Prepare the data for classification
X = data.drop(['Id', 'Species'], axis=1)
y = data['Species']

# Perform k-Nearest Neighbors (KNN) classification with different values of k
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    scores.append(metrics.accuracy_score(y, y_pred))
    
# Plot the accuracy scores for different values of k
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()

# Perform Logistic Regression classification
logreg = LogisticRegression()
logreg.fit(X, y)
y_pred = logreg.predict(X)
print("Accuracy of Logistic Regression (full dataset):", metrics.accuracy_score
      (y, y_pred))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
                                                    random_state=5)

# Train a Logistic Regression model on the training set
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Make predictions on the test set and calculate accuracy
y_pred = logreg.predict(X_test)
print("Accuracy of Logistic Regression (test set):", metrics.accuracy_score
      (y_test, y_pred))





