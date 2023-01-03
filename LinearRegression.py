import numpy as np

#Visit https://github.com/VaradBelwalkar/helpful-tutorials/blob/main/machine_learning/learn_regression.tex
#for explaination in math terminology

class LinearRegression:
        
        def __init__(self,lr = 0.001,n_iters = 1000):
            self.lr = lr
            self.n_iters = n_iters 
            self.slopes = None
            self.y_intercept = None

        def fit(self,X,y):
            n_samples, n_features = X.shape
            self.slopes = np.zeros(n_features)
            self.y_intercept = 0
            for _ in range(self.n_iters):
                y_pred = np.dot(X,self.slopes) + self.y_intercept

                #Partial derivative w.r.t "m"
                d_slopes = (1/n_samples) * np.dot(X.T,(y_pred-y))

                #Partial derivative w.r.t "c"
                d_intercept = (1/n_samples) * np.sum(y_pred -y)


                self.slopes = self.slopes -self.lr * d_slopes
                self.y_intercept = self.y_intercept - self.lr * d_intercept

        def predict(self, X):
            y_pred = np.dot(X,self.slopes) + self.y_intercept
            return y_pred




def MSE(y_test,predictions):
    return np.mean((y_test - predictions)**2)
