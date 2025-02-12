'''
# Problem with Perceptron Trick:
- Line move when only there is missclassified point means the value of weight such as w1, w2 ..etc and bias such as b1, b2 will change but if there is no missclassified point, the value of the weight and bias will not change so that we cannot quantify the result or we cannot say how good is this perceptron model. 

# Loss function: 
- Way or methods to tells how good is our model. 
- If we take an above example, Loss function will be the function of f(w1, w2, b). If this f(w1, w2, b) gives value such as 25 means this 25 is error. 

# Loss function in Perceptron:
- Mathematical function of w1, w2 and b or f(w1, w2, b). 
- #missclassifiedPoint <-- Can be a loss function in our case or in our perceptron models.
- Loss function of the Perceptron: L(w,b) = 1/n * np.sum(L(yi, f(xi)))  + alpha * Regularization(w1, w2)
- In this case, L(yi, f(xi)) = max(00, yi * f(xi))
- What is f(xi) ? f(xi) -> w1x1 + w2x2 + b
- n -> Number of rows in the data. 
- In this case, Loss function is depend on the w1, w2 & b. 
'''
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Making an Dataset. 
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=1, 
    n_redundant=0, 
    n_classes=2, 
    n_clusters_per_class=1,
    random_state=41, 
    hypercube=False, 
    class_sep=15
)

# 
print("Shape of Input Data: ")
print(X.shape)

print("Shape of output: ")
print(y.shape)

# Plotting an Dataset. 
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="winter", s=100)
plt.show()

# Building an Loss function of the Perceptron. 
def perceptron_loss_function(X, y):

    # selecting the random values of w1, w2 and b. 
    w1 = w2 = b = 1

    # setting the learning rate of 0.1 
    learning_rate = 0.1 

    #
    for j in range(1000):
        for i in range(X.shape[0]):

            # Checking an condition. 
            z = w1 * X[i][0] + w2 * X[i][1] + b 

            # 
            if z * y[i]  < 0:
                # Updating the values of w1, w2, & b. 
                w1 = w1 + learning_rate * X[i][0]
                w2 = w2 + learning_rate * X[i][1]
                b = b + learning_rate * y[i]
    
    return w1, w2, b

# 
w1, w2, b = perceptron_loss_function(X, y)

print("Values of w1, w2 and b.")
print("W1 : ", w1)
print("W2 : ", w2)
print("b : ", b)

# since the perceptron is just a line so the equation of line is y = mx + c. 
# Now, calculating the values of m & c. 

# m -> represent the slope here. 
m = -(w1/w2)

# c -> represent the intercept term here. 
c = -(b/w2)

# 
print("\nDisplying the value of m & c: ")
print("Value of m: ", m)
print("Value of c: ", c)

# making an inputs and output. 
x_input = np.linspace(-3, 3, 100)
y_input = m * x_input + c

# 
plt.figure(figsize=(10, 6))
plt.plot(x_input, y_input, color="red", linewidth=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="winter", s=100)
plt.ylim(-3, 2)
plt.show()