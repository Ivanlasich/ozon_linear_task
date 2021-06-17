import numpy as np
from scipy.special import expit
import time
from random import shuffle


class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=None,
        step_alpha=1,
        step_beta=0, 
        tolerance=1e-5,
        max_iter=1000,
        random_seed=153,
        **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.weights = None
        self.y = None
        self.X = None

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """
        self.y = y
        self.X = X
        index = np.arange(X.shape[0])
        shuffle(index)
        X = X[index]
        y = y[index]

        if (w_0 is None):
            self.weights = np.ones(X.shape[1])
            w_0 = np.zeros(X.shape[1])
        else:
            self.weights = w_0
            w_0 = np.zeros(X.shape[1])

        k = 0

        if (trace):
            history = {}
            history['time'] = []
            history['func'] = []

            if((not X_val is None) and (not y_val is None)):
                history['func_val'] = []

        iteration = 0

        while(np.linalg.norm(self.weights - w_0) >= self.tolerance and self.max_iter > iteration):

            k += 1
            learning_rate = self.step_alpha / k**self.step_beta

            if(self.batch_size is None):
                start_time = time.time()
                w_0 = self.weights
                dw = self.loss_function.grad(X, y, w_0)
                self.weights = w_0 - learning_rate * dw
                iteration += 1
                end_time = time.time() - start_time
            else:
                n = X.shape[0] // self.batch_size
                if (X.shape[0] % self.batch_size != 0):
                    n += 1
                start_time = time.time()
                for i in range(n):
                    X_batch = X[i * self.batch_size:(i + 1) * self.batch_size]
                    y_batch = y[i * self.batch_size:(i + 1) * self.batch_size]
                    w_0 = self.weights
                    dw = self.loss_function.grad(X_batch, y_batch, w_0)
                    self.weights = w_0 - learning_rate * dw
                iteration += 1
                end_time = time.time() - start_time

            if(trace):
                history['time'].append(end_time)
                history['func'].append(self.loss_function.func(X, y, self.weights))
                if ((not X_val is None) and (not y_val is None)):
                    history['func_val'].append(self.loss_function.func(X_val, y_val, self.weights))

        if (trace):
            return history

    def predict(self, X, threshold=0):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.
        threshold : float
            Chosen target binarization threshold.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """
        threshold = self.get_optimal_threshold(self.X, self.y)
        predict = X @ self.weights
        answ = np.array([1 if i > threshold else -1 for i in predict])

        return answ

    def get_optimal_threshold(self, X, y):
        """
        Get optimal target binarization threshold.
        Balanced accuracy metric is used.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y : numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : float
            Chosen threshold.
        """
        if self.loss_function.is_multiclass_task:
            raise TypeError('optimal threhold procedure is only for binary task')

        weights = self.get_weights()
        scores = X.dot(weights)
        y_to_index = {-1: 0, 1: 1}

        # for each score store real targets that correspond score
        score_to_y = dict()
        score_to_y[min(scores) - 1e-5] = [0, 0]
        for one_score, one_y in zip(scores, y):
            score_to_y.setdefault(one_score, [0, 0])
            score_to_y[one_score][y_to_index[one_y]] += 1

        # ith element of cum_sums is amount of y <= alpha
        scores, y_counts = zip(*sorted(score_to_y.items(), key=lambda x: x[0]))
        cum_sums = np.array(y_counts).cumsum(axis=0)

        # count balanced accuracy for each threshold
        recall_for_negative = cum_sums[:, 0] / cum_sums[-1][0]
        recall_for_positive = 1 - cum_sums[:, 1] / cum_sums[-1][1]
        ba_accuracy_values = 0.5 * (recall_for_positive + recall_for_negative)
        best_score = scores[np.argmax(ba_accuracy_values)]
        return best_score

    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        """
        return self.weights

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        return self.loss_function.func(X, y, self.get_weights())