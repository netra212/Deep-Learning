from sklearn.datasets import make_classification
import numpy as np 
import matplotlib.pyplot as plt 

'''
# Perceptron algorithms: 
- Create a line. 
- Let's suppose we have a 4 Points, two reds and two greeen. Let's suppose one red point, one green point are in the one side of the line with equation Ax1 + By1 + C >= 0. Similarly, 2nd red point, 2nd green point are in the other side of the lien with equation Ax1 + By1 + C < 0. This mean, one red with co-ordinate (4, 5) and one green point are in the wrong side of the line with co-ordinate (1, 3). In order bring this point in their respective region, we have to performs the some kinds of the transformation in the Perceptron algorithm or Line. So, In this case, we will add the one to each co-ordinate of wrongly classified point resulting (4, 5, 1) & (1, 3, 1). Now, we will perform the addition and substraction between the co-ordinates and line coefs. 
such as 
- Eq of line: 2x + 3y + 5 = 0
For (4, 5, 1)
     2  3  5
  -  4  5  1
 ---------------
-2x - 2y + 4 = 0

For (1, 3, 1)
    2  3  5
 +  1  3  1
---------------
3x + 6y + 6 = 0

But we don't add & subs co-ordinates directly, we takes the learning rates into an account. 
Like 
- New_coef = Old_Coef - eta * Co-Ordinate 

- Ax + By + C = 0
- W0 + W1.x1 + W2.x2 = 0
- W0 * X0 + W1.X1 + W2.X2 = 0
- np.sum(Wi*Xi) = 0


- Algorithms:
epochs = 1000, eta = 0.001
for i in range(1000):
    select a point randomly. 
    if xi belongs to N & np.sum(Wi*Xi) >=0 # negative point falls under the trap of positive region. 
        Wnew = Wold - eta * Xi
    
    if Xi belongs to P & np.sum(Wi.Xi) < 0 # positive point falls under the trap of the negative region. 
        Wnew = Wold + eta * Xi

# Simplified Version of above Algorithms:
for i in 1000:
    select random row:
        Wn = Wo + eta * (y_i - y_hat) * Xi
'''


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
