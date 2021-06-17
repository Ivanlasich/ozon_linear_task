import numpy as np
import scipy
from scipy.special import expit

class BaseLoss:
    """
    Base class for loss function.
    """

    def func(self, X, y, w):
        """
        Get loss function value at w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, X, y, w):
        """
        Get loss function gradient value at w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogisticLoss(BaseLoss):
    """
    Loss function for binary logistic regression.
    It should support l2 regularization.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef
        self.is_multiclass_task = False

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : float
        """
        a = - y * (X @ w)

        return np.mean(np.logaddexp(0, a)) + self.l2_coef * (w[1:]@w[1:])

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : 1d numpy.ndarray
        """
        w_ = np.array([0])
        w_ = np.append(w_, w[1:])
        if scipy.sparse.issparse(X):
            expr = y*expit(-y*(X@w))
            return np.squeeze(np.asarray((-np.mean((scipy.sparse.csr_matrix(X.T).multiply(scipy.sparse.csr_matrix(expr)).todense().T), axis=0)))) +\
                   self.l2_coef * 2 * w_

        return -np.mean(((X.T*y)*expit(-y*(X@w))).T, axis = 0) + self.l2_coef * 2 * w_

class MultinomialLoss(BaseLoss):
    """
    Loss function for multinomial regression.
    It should support l2 regularization.
    
    w should be 2d numpy.ndarray.
    First dimension is class amount.
    Second dimesion is feature space dimension.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef
        self.is_multiclass_task = True

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 2d numpy.ndarray

        Returns
        -------
        : float
        """
        pass

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 2d numpy.ndarray

        Returns
        -------
        : 2d numpy.ndarray
        """
        pass

