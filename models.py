import numpy as np

class LinearRegression:
    def __init__(self, alpha, iterations):
        self.alpha = alpha
        self.iterations = iterations
        self.X = None
        self.y = None
        self.theta = None

    def _cost_function(self, X, y, theta, alpha):
        m = len(y)
        return np.dot((np.dot(X, theta) - y), (np.dot(X, theta) - y)) / (2 * m) + alpha * sum(np.power(theta[1:], 2))
         
    def _gradient_descent(self, X, y, theta, alpha, iterations):
        m = len(y)
        cost = np.zeros(iterations)
        thetaMod = theta.copy()
        thetaHist = np.zeros(iterations)
        for i in range(iterations):
            thetaMod = thetaMod - np.dot(self.X.T, (np.dot(self.X, thetaMod) - self.y)) * self.alpha / m
            thetaHist[i] = thetaMod[1]
            cost[i] = self._cost_function(X, y, theta, alpha)
        return thetaMod, cost, thetaHist

    def fit(self, X, y):
        ones = np.ones((X.shape[0], 1))
        theta = np.zeros(X.shape[1] + 1)
        self.X = np.hstack(([ones, X]))
        self.y = y
        self.theta, cost, thetaHistory = self._gradient_descent(self.X, self.y, theta, self.alpha, self.iterations) 
        return cost, thetaHistory
    
    def predict(self, X_test):
        return np.dot(self.X, self.theta.reshape(-1, 1))

    def get_iterations(self):
        return self.iterations

if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv('/home/artem/Programming/Machine_Learning/ML_Andrew_Ng/Linear_Regression/ex1data2.txt', names=['Size', 'Bedrooms', 'Price'])
    X = data.loc[:, ("Size", "Bedrooms")]
    y = data.loc[:, "Price"]
    iterations = 1000
    alpha = 0.01
    model = LinearRegression(alpha, iterations)
    cost = model.fit(X, y)
    
    
