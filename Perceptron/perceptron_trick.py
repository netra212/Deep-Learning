from sklearn.datasets import make_classification
import numpy as np 
import matplotlib.pyplot as plt 


# Separating the data into X and y. 
X, y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=10)


# Plotting the figures. 
print(plt.figure(figsize=(10, 6)))
print(plt.scatter(X[:, 0], X[:, 1], c=y, cmap="winter", s=100))
print(plt.show())


# Defining the step functions in perceptron. 
def step(z):
    return 1 if z>0 else 0


# Defining the perceptron algorithms. 
def perceptron(X, y):

    # Adding one to all the Input rows which is bias term. 
    X = np.insert(X, 0, 1, axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1 

    for i in range(1000):
        # Picking an random number from 0 to 100. 
        j = np.random.randint(0, 100)
        z = np.dot(X[j], weights) # w_1 * x_1 + w_1 * x_2 + bias_term
        y_hat = step(z) # passing an z to the step function. 

        weights = weights + lr * (y[j] - y_hat) * X[j]
    
    return weights[0], weights[1:]

intercept_, coef_ = perceptron(X, y)

print("Intercept Term: ", intercept_)
print("Coefficient Term: ", coef_)

# Calculating the value of `m` & `b`. 
m = -(coef_[0]/coef_[1])
b = -(intercept_/coef_[1])

# Passing an Inputs. 
x_input = np.linspace(-3,3,100)
y_input = m*x_input + b

# Visualizing the inputs. 
plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='red',linewidth=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
plt.ylim(-3,2)
