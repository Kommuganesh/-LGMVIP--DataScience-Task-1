# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, plot_confusion_matrix,classification_report,confusion_matrix

# Loading the Dataset
data=pd.read_csv("Iris.csv")
data.head(10)
data.sample(10)
data.shape
# Dataset Columns
data.columns
#Dataset Summary
data.info()
#Dataset Statistical Summary
data.describe()
#Checking Null Values
data.isnull().sum()
data['Species'].unique()
#Checking columns count of "Species"
data['Species'].value_counts()
sns.pairplot(data,hue='Species')
fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(16,5))
sns.scatterplot(x='SepalLengthCm',y='PetalLengthCm',data=data,hue='Species',ax=ax1,s=300,marker='o')
sns.scatterplot(x='SepalWidthCm',y='PetalWidthCm',data=data,hue='Species',ax=ax2,s=300,marker='o')
sns.violinplot(y='Species', x='SepalLengthCm', data=data, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='SepalWidthCm', data=data, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='PetalLengthCm', data=data, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='PetalWidthCm', data=data, inner='quartile')
plt.show()

#Pie plot to show the overall types of Iris classifications
colors = ['#66b3ff','#ff9999','green']
data['Species'].value_counts().plot(kind = 'pie',  autopct = '%1.1f%%', shadow = True,colors=colors, explode = [0.08,0.08,0.08])

plt.figure(figsize=(7,5))
sns.heatmap(data.corr(), annot=True,cmap='CMRmap')
plt.show()

#Defining independent and dependent variables
features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = data.loc[:, features].values   #defining the feature matrix
y = data.Species

#Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33,random_state=0)

#Defining the decision tree classifier and fitting the training set
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

#Prediction on test data
y_pred = dtree.predict(X_test)
y_pred

#Checking the accuracy of the model
score=accuracy_score(y_test,y_pred)
print("Accuracy:",score)

def report(model):
    preds=model.predict(X_test)
    print(classification_report(preds,y_test))
    plot_confusion_matrix(model,X_test,y_test,cmap='nipy_spectral',colorbar=True)
    
    print('Decision Tree Classifier')
    report(dtree)
    print(f'Accuracy: {round(score*100,2)}%')
    
    confusion_matrix(y_test, y_pred)
    


