import pandas as pd
import numpy as py
import os

os.chdir(r"C:/Users/gaura/OneDrive/Desktop/Machine Learning Codes")

stars=pd.read_csv("star_classification.csv")
#Copying the dataframe
df=stars.copy()
#df.info()
print(df.describe())
print(df['class'].value_counts())
#Most of our data identifies Galaxies followed by Star and then QSO

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='class', y='redshift', data=df)
plt.show()
# Interpretation of the boxplot:
# - GALAXY: Low redshift values with a narrow range (low variability), and a few high outliers.
# - QSO: Wide range of redshift values (high variability) with a high median and many outliers.
# - STAR: Very small redshift values, tightly clustered with minimal variability, and a few close outliers.


# Next I'll drop the irrelevant colums that bear no value for identification
df2 = df.drop(['obj_ID', 'spec_obj_ID', 'run_ID', 'rerun_ID', 'plate', 'fiber_ID', 'MJD'], axis=1)
total_rows = df2.shape[0]  # .shape[0] gives the number of rows
print(f"Total rows (using .shape): {total_rows}")
#Lets map the class labels:
    
# Map the class column to numeric values
from sklearn.preprocessing import LabelEncoder
enco = LabelEncoder()
df2['class'] = enco.fit_transform(df2['class'])

#Normalizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df2[['u', 'g', 'r', 'i', 'z']] = scaler.fit_transform(df2[['u', 'g', 'r', 'i', 'z']])

##Start Training


### Splitting the data
from sklearn.model_selection import train_test_split
X = df2.drop('class', axis=1)
y = df2['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =============================================================================
# for k in range(1, 11):  # Try values of k from 1 to 10
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#     print(f"k={k}")
#     print(classification_report(y_test, y_pred))
# =============================================================================


knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("KNN:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(" ")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

### Trying Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='newton-cg', max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression:")
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))

cm2 = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm2, annot=True, fmt='d', cmap="Blues")
plt.title('Logistic Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()