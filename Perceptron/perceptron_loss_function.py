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
- 
'''