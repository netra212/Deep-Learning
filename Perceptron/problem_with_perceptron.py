'''
# Perceptron does not works in the non-linear data. 
# This is the practical implementation showing that perceptron does not work with the non-linear data. 
'''
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Creating an DataFrame. 
or_data = pd.DataFrame()
and_data = pd.DataFrame()
xor_data = pd.DataFrame()

# Filling Or Data. 
or_data["Input1"] = [1, 1, 0, 0]
or_data["Input2"] = [1, 0, 1, 0]
or_data["Output"] = [1, 1, 1, 0]

# Filling And Data. 
and_data["Input1"] = [1, 1, 0, 0]
and_data["Input2"] = [1, 0, 1, 0]
and_data["Output"] = [1, 0, 0, 0]

# Filling Xor Data. 
xor_data["Input1"] = [1, 1, 0, 0]
xor_data["Input2"] = [1, 0, 1, 0]
xor_data["Output"] = [0, 1, 1, 0]

print("Or Data: ")
print(or_data)
print("And Data: ")
print(and_data)
print("XOR Data: ")
print(xor_data)

# PLottijng an and_data. 
print(sns.scatterplot(x=and_data["Input1"], y=and_data["Input2"], hue=and_data["Output"],s=200))
print(sns.scatterplot(x=xor_data["Input1"], y=xor_data["Input2"], hue=xor_data["Output"],s=200))
print(sns.scatterplot(x=or_data["Input1"], y=or_data["Input2"], hue=or_data["Output"],s=200))
print(plt.show())

# Building an Perceptron models. 
from sklearn.linear_model import Perceptron

clf1 = Perceptron()
clf2 = Perceptron()
clf3 = Perceptron()

# fitting the models. 
clf1.fit(and_data.iloc[:, 0:2].values, and_data.iloc[:, -1].values)
clf2.fit(or_data.iloc[:, 0:2].values, or_data.iloc[:, -1].values)
clf3.fit(xor_data.iloc[:, 0:2].values, xor_data.iloc[:, -1].values)

# Calculating the coefficients. 

# For the Coefficients 1.
print("Coefficients 1: ")
print(clf1.coef_)
print(clf1.intercept_)

# For the Coefficients 2. 
print("Coefficients 1: ")
print(clf2.coef_)
print(clf2.intercept_)

# For the Coefficients 3. 
print("Coefficients 3: ")
print(clf3.coef_)
print(clf3.intercept_)

# 
x = np.linspace(-1, 1, 5)
y = -x + 1

# Plotting the 
print(plt.plot(x, y))
sns.scatterplot(x=and_data["Input1"], y=and_data["Input2"], hue=and_data["Output"], s=200)
print(plt.show())


# 
print("Displaying the coefficient and intercept for the 2nd models: ")
print(clf2.coef_)
print(clf2.intercept_)

# 
x1=np.linspace(-1,1,5)
y1=-x+0.5


# Visualizing the model 2. 
plt.plot(x1, y1)
sns.scatterplot(x=or_data["Input1"], y=or_data["Input2"], hue=or_data["Output"], s=200)
plt.show()


# Calculating the coefficients and Intercept. 
print("Displaying the coefficients and intercepts of the model 3: ")
print(clf3.coef_)
print(clf3.intercept_)


# Visualizing the model 3 which shows that perceptron does not works on the non-linear datasets. 
plot_decision_boundary(xor_data.iloc[:, 0:2].values, xor_data.iloc[:, -1].values, clf=clf3, legend=2)
plt.show()