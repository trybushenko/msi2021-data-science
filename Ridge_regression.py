"""Module with class to fit a ridge regression model and also make predictions."""
import numpy as np


class RidgeRegression:
    """Class to fit a ridge regression model on a given training set, and make predictions on data.
    Specifically, the cost function with L_2 regularization is assumed to be:
        ||w - Xw||^2_2 + alpha * ||w||2_2
    where,
        w = Model weights,
        X = Data design matrix,
        alpha = regularization parameter.
    The class uses batch gradient descent to optimize the model weights.
    """

    def __init__(self, learning_rate: float, reg_strength: float, max_iter: int) -> None:
        """Initialize.
        Args:
            learning_rate (float): Learning rate for gradient descent.
            reg_strength (float): Regularization parameter, to control the bias-variance tradeoff.
            max_iter (int): Number of iterations to run gradient descent.
        """
        self.learn_rate = learning_rate
        self.reg_strength = reg_strength
        self.max_iter = max_iter
        self._weights: np.array
        self.design_matrix: np.array
        self.y_train: np.array
        self.train_size: int
        self.num_feats: int

    @property
    def weights(self):
        """Return ridge regression model weights."""
        return self._weights

    def fit(self, x_train, y_train) -> None:
        """Fit model weights to training data."""
        self.train_size, self.num_feats = x_train.shape
        self.design_matrix = np.append(np.ones(self.train_size).reshape(
            -1, 1), x_train, axis=1)  # Add one for the intercept term for all training examples.
        self.y_train = y_train.to_numpy()
        self._weights = np.zeros(self.num_feats + 1)  # +1 for the intercept.

        self.optimize_weights()

    def optimize_weights(self) -> None:
        """Optimize model weights for self.max_iter iterations."""
        for _ in range(self.max_iter):
            self.update_step()

    def update_step(self) -> None:
        """Update model weights with batch gradient descent step.
        Recall, the update statement for a model weight theta_k is:
            theta_k = theta_k - (learn_rate * J_theta_k)
            where J_theta_k = cost function
                  J_theta_k = (2 / train_size) * ( ((y_hat - y_real) * x_k) + (alpha * theta_k^2))
        """
        y_hat = (self._weights * self.design_matrix).sum(axis=1)
        errors = (y_hat - self.y_train).reshape(-1, 1)

        j_theta = (2 / self.train_size) * ((errors * self.design_matrix).sum(axis=0) +
                                           (self.reg_strength * self._weights))
        step = self.learn_rate * j_theta

        self._weights = self._weights - step.reshape(-1)

    def predict(self, x_test):
        """Make predictions for x_test using trained model.
        Args:
            x_test (np.array): Input test data set.
        Returns:
            np.array: Predictions for x_test.
        """
        test_size = x_test.shape[0]
        x_test = np.append(np.ones(test_size).reshape(-1, 1), x_test, axis=1)

        return (self._weights * x_test).sum(axis=1)

    def RMSE(self, X_test, y_test) -> np.float64:
        return np.sqrt(np.sum(np.power(y_test.ravel() - self.predict(X_test), 2)) / y_test.shape[0])
