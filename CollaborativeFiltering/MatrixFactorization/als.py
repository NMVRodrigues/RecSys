import numpy as np
from sklearn.metrics import mean_squared_error

"""
Attempt to fix an existim implementation, will look at it in the future
"""

class ALS:
    """
    Alternating Least Squares matrix factorization algorithm

    Parameters
    ----------
    n_iters : int
        number of iterations to train the algorithm

    n_factors : int
        number of latent factors to use in matrix
        factorization model, some machine-learning libraries
        denote this as rank

    reg : float
        regularization term for item/user latent factors,
        since lambda is a keyword in python we use reg instead
    """

    def __init__(self, n_iters, n_factors, reg):
        self.reg = reg
        self.n_iters = n_iters
        self.n_factors = n_factors

    def create_UV(self, mode="random"):
        """
        Creates U V user-item matrixes
        """
        if mode == "random":
            self.user_factors = np.random.random((self.n_user, self.n_factors))
            self.item_factors = np.random.random((self.n_item, self.n_factors))


    def fit(self, x, x_val=None):
        """
        pass in training and testing at the same time to record
        model convergence, assuming both dataset is in the form
        of User x Item matrix with cells as ratings
        """
        self.n_user, self.n_item = x.shape

        self.create_UV()

        # record the training and testing mse for every iteration
        # to show convergence later (usually, not worth it for production)
        self.val_mse_history = []
        self.train_mse_history = []

        for _ in range(self.n_iters):
            self.user_factors = self._als_step(x, self.item_factors)
            self.item_factors = self._als_step(x.T, self.user_factors)

            predictions = self.predict()

            self.val_mse_history.append(self.compute_mse(x_val, predictions))
            self.train_mse_history.append(self.compute_mse(x, predictions))


    def _als_step(self, ratings, target):
        """
        Update the target matrix
        """
        A = target.T.dot(target) + np.eye(self.n_factors) * self.reg
        b = ratings.dot(target)
        A_inv = np.linalg.inv(A)
        solve_vecs = b.dot(A_inv)
        return solve_vecs

    def predict(self):
        """predict ratings for every user and item"""
        pred = self.user_factors.dot(self.item_factors.T)
        return pred

    @staticmethod
    def compute_mse(y_true, y_pred):
        """ignore zero terms prior to comparing the mse"""
        mask = np.nonzero(y_true)
        mse = mean_squared_error(y_true[mask], y_pred[mask])
        return mse