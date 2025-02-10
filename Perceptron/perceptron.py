'''
# Perceptron: Mathematical functionor or models which works on the supervised ML. 
- Perceptron is a line in 2D space which create a region in order to divide the dataset into two classes means, peceptron performs the binary classification. 
- In 2D - Perceptron acts as a Line. 
- In 3D - Perceptron acts as a Plane.

# Limitations of Perceptrons:
- Perceptron only works on linear data or the sort of linear data. 
- Perceptorn will fails on the completly non-linear data. 
'''

# Importing the required Libraries. 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.linear_model import Perceptron
from mlextend.plotting import plot_decision_boundary

# Loading an dataset. 
df = pd.read_csv("data/placement.csv")

print(df.shape)
print(df.head())

# Visualizing the dataset. 
print(sns.scatterplot(x=df['cgpa'],y=df['resume_score'],hue=df['placed']))
print(plt.show())

# Separating X and y.
X = df.iloc[:, 0:2]
y = df.iloc[:, -1]

# Initializing the Perceptron class.
perceptron = Perceptron()

# Training the perceptron means calculating the values of `weight` and `bais`. 
perceptron.fit(X, y) 

# Displaying the coefficient value. 
print(perceptron.coef_)

# Displaying the interncept value. 
print(perceptron.intercept_)

# Plotting the decision boundary. 
print(plot_decision_boundary(X.values, y.values, clf=perceptron, legend=2))