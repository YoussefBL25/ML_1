"""Baseline ML optimization algorithms for EPFL CS-433 Project 1.

This module implements the 6 required linear and logistic regression methods:
1. mean_squared_error_gd: Linear regression using gradient descent
2. mean_squared_error_sgd: Linear regression using stochastic gradient descent
3. least_squares: Least squares regression using normal equations
4. ridge_regression: Ridge regression using normal equations
5. logistic_regression: Logistic regression using gradient descent
6. reg_logistic_regression: Regularized logistic regression using gradient descent
"""

import numpy as np


def compute_mse_loss(y, tx, w):
    """Compute the Mean Squared Error (MSE) loss with 1/(2N) factor.

    Formula:
        L_MSE(w) = (1 / (2 * N)) * sum_{i=1}^N (y_i - x_i^T w)^2
                 = (1 / (2 * N)) * (y - tx @ w)^T @ (y - tx @ w)

    Args:
        y (np.ndarray): Target vector of shape (N,).
        tx (np.ndarray): Feature matrix of shape (N, D).
        w (np.ndarray): Weight vector of shape (D,).

    Returns:
        np.float64: 0-D scalar MSE loss value.
    """
    # Compute residual vector: e has shape (N,)
    e = y - tx @ w
    # Loss scalar: 0.5 * MSE consistent with lecture notes
    return np.float64(np.dot(e, e) / (2.0 * len(y)))


def compute_mse_gradient(y, tx, w):
    """Compute the gradient of the MSE loss with respect to w.

    Formula:
        grad_MSE(w) = - (1 / N) * tx^T @ (y - tx @ w)

    Args:
        y (np.ndarray): Target vector of shape (N,).
        tx (np.ndarray): Feature matrix of shape (N, D).
        w (np.ndarray): Weight vector of shape (D,).

    Returns:
        np.ndarray: Gradient vector of shape (D,).
    """
    # Residual error vector: e has shape (N,)
    e = y - tx @ w
    # Dimension analysis: tx.T has shape (D, N), e has shape (N,) -> output has shape (D,)
    return -(tx.T @ e) / len(y)


def sigmoid(t):
    """Compute the sigmoid activation with numerical clipping.

    Formula:
        sigma(t) = 1 / (1 + exp(-t))

    Numerical Stability:
        To prevent floating-point overflow in exp(-t) for large negative inputs
        (which causes exp(>709) to overflow to inf in float64), we clip inputs
        to [-500.0, 500.0].

    Args:
        t (np.ndarray): Input values of arbitrary shape.

    Returns:
        np.ndarray: Element-wise sigmoid output in range (0, 1).
    """
    t_clipped = np.clip(t, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-t_clipped))


def compute_logistic_loss(y, tx, w):
    """Compute the Negative Log-Likelihood (NLL) loss for binary logistic regression.

    Formula:
        L_log(w) = - (1 / N) * sum_{i=1}^N [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
        where p_i = sigma(x_i^T w)

    Numerical Stability:
        Predictions p are clipped to [1e-15, 1.0 - 1e-15] to prevent log(0)
        which results in NaN or -inf.

    Args:
        y (np.ndarray): Binary target labels in {0, 1} of shape (N,).
        tx (np.ndarray): Feature matrix of shape (N, D).
        w (np.ndarray): Weight vector of shape (D,).

    Returns:
        np.float64: 0-D scalar logistic loss value.
    """
    # Linear projection: tx @ w has shape (N,)
    pred = sigmoid(tx @ w)
    # Numerical clip to avoid log(0) undefined domain
    eps = 1e-15
    pred = np.clip(pred, eps, 1.0 - eps)
    # Negative log-likelihood average over N samples
    loss = -np.mean(y * np.log(pred) + (1.0 - y) * np.log(1.0 - pred))
    return np.float64(loss)


def compute_logistic_gradient(y, tx, w):
    """Compute the gradient of the logistic loss with respect to w.

    Formula:
        grad_log(w) = (1 / N) * tx^T @ (sigma(tx @ w) - y)

    Args:
        y (np.ndarray): Binary target labels in {0, 1} of shape (N,).
        tx (np.ndarray): Feature matrix of shape (N, D).
        w (np.ndarray): Weight vector of shape (D,).

    Returns:
        np.ndarray: Gradient vector of shape (D,).
    """
    # Prediction probabilities: pred has shape (N,)
    pred = sigmoid(tx @ w)
    # Dimension analysis: tx.T is (D, N), (pred - y) is (N,) -> output has shape (D,)
    return (tx.T @ (pred - y)) / len(y)


def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
    """Generate mini-batches for stochastic gradient descent.

    Args:
        y (np.ndarray): Target vector of shape (N,).
        tx (np.ndarray): Feature matrix of shape (N, D).
        batch_size (int): Number of samples per batch. Default: 1.
        num_batches (int): Number of batches to yield. Default: 1.
        shuffle (bool): Whether to randomly permute indices. Default: True.

    Yields:
        tuple[np.ndarray, np.ndarray]: Mini-batch pair (y_batch, tx_batch).
    """
    data_size = len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index >= data_size:
            break
        yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using Gradient Descent (GD).

    Mathematical Formulation:
        Cost function: L_MSE(w) = (1 / (2 * N)) * ||y - tx @ w||^2
        Gradient: grad(w) = - (1 / N) * tx^T @ (y - tx @ w)
        Update step: w_{t+1} = w_t - gamma * grad(w_t)

    Args:
        y (np.ndarray): Target labels vector of shape (N,).
        tx (np.ndarray): Input feature matrix of shape (N, D).
        initial_w (np.ndarray): Initial weight vector of shape (D,).
        max_iters (int): Total number of gradient descent iterations.
        gamma (float): Learning rate / step size parameter.

    Returns:
        tuple[np.ndarray, np.float64]:
            w (np.ndarray): Optimal weight vector of shape (D,).
            loss (np.float64): Final MSE loss scalar (half MSE).
    """
    w = np.asarray(initial_w, dtype=np.float64).ravel().copy()

    for _ in range(max_iters):
        grad = compute_mse_gradient(y, tx, w)
        w = w - gamma * grad

    loss = compute_mse_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using Stochastic Gradient Descent (SGD) with batch size 1.

    Mathematical Formulation:
        Cost function: L_MSE(w) = (1 / (2 * N)) * ||y - tx @ w||^2
        Stochastic Gradient on sample i: grad_i(w) = - x_i * (y_i - x_i^T w)
        Update step: w_{t+1} = w_t - gamma * grad_i(w_t)

    Args:
        y (np.ndarray): Target labels vector of shape (N,).
        tx (np.ndarray): Input feature matrix of shape (N, D).
        initial_w (np.ndarray): Initial weight vector of shape (D,).
        max_iters (int): Total number of stochastic gradient descent steps.
        gamma (float): Learning rate / step size parameter.

    Returns:
        tuple[np.ndarray, np.float64]:
            w (np.ndarray): Optimal weight vector of shape (D,).
            loss (np.float64): Final MSE loss computed over the full dataset (y, tx).
    """
    w = np.asarray(initial_w, dtype=np.float64).ravel().copy()

    for _ in range(max_iters):
        # Sample exactly 1 random data point per iteration (mini-batch size = 1)
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad = compute_mse_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad

    loss = compute_mse_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using Normal Equations.

    Mathematical Formulation:
        Solves: (tx^T @ tx) @ w = tx^T @ y
        w* = solve(tx^T @ tx, tx^T @ y)

    Numerical Stability:
        np.linalg.solve is used instead of matrix inversion (np.linalg.inv)
        as it is faster and numerically stabler by solving the linear system
        directly via LU/Cholesky factorization.
        numpy.linalg.lstsq is strictly avoided per project specifications.

    Args:
        y (np.ndarray): Target labels vector of shape (N,).
        tx (np.ndarray): Input feature matrix of shape (N, D).

    Returns:
        tuple[np.ndarray, np.float64]:
            w (np.ndarray): Optimal weight vector of shape (D,).
            loss (np.float64): Final MSE loss scalar (half MSE).
    """
    # Dimension analysis: tx.T @ tx is (D, D), tx.T @ y is (D,)
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b).ravel()

    loss = compute_mse_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using Normal Equations with L2 penalty.

    Mathematical Formulation:
        Objective: L_ridge(w) = (1 / (2 * N)) * ||y - tx @ w||^2 + lambda_ * ||w||^2
        Setting gradient to zero:
            -(1 / N) * tx^T @ (y - tx @ w) + 2 * lambda_ * w = 0
            (tx^T @ tx + 2 * N * lambda_ * I) @ w = tx^T @ y
        w* = solve(tx^T @ tx + 2 * N * lambda_ * I, tx^T @ y)

    Contractual Note:
        The returned loss scalar strictly excludes the regularization penalty
        term (lambda_ * ||w||^2) and represents the unpenalized half MSE loss.

    Args:
        y (np.ndarray): Target labels vector of shape (N,).
        tx (np.ndarray): Input feature matrix of shape (N, D).
        lambda_ (float): Regularization parameter.

    Returns:
        tuple[np.ndarray, np.float64]:
            w (np.ndarray): Optimal weight vector of shape (D,).
            loss (np.float64): Final unpenalized MSE loss scalar (half MSE).
    """
    n_samples = len(y)
    n_features = tx.shape[1]

    # Dimension analysis: A has shape (D, D), b has shape (D,)
    # Regularization penalty scaled by 2 * N * lambda_
    A = tx.T @ tx + (2.0 * n_samples * lambda_) * np.eye(n_features)
    b = tx.T @ y
    w = np.linalg.solve(A, b).ravel()

    # Returned loss does NOT include lambda * ||w||^2
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using Gradient Descent for binary classification (y in {0, 1}).

    Mathematical Formulation:
        Objective: Negative Log-Likelihood (NLL)
            L(w) = - (1 / N) * sum_{i=1}^N [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
            where p_i = sigma(x_i^T w)
        Gradient: grad(w) = (1 / N) * tx^T @ (sigma(tx @ w) - y)
        Update step: w_{t+1} = w_t - gamma * grad(w_t)

    Args:
        y (np.ndarray): Binary target labels in {0, 1} of shape (N,).
        tx (np.ndarray): Input feature matrix of shape (N, D).
        initial_w (np.ndarray): Initial weight vector of shape (D,).
        max_iters (int): Total number of gradient descent iterations.
        gamma (float): Learning rate / step size parameter.

    Returns:
        tuple[np.ndarray, np.float64]:
            w (np.ndarray): Optimal weight vector of shape (D,).
            loss (np.float64): Final Negative Log-Likelihood loss scalar.
    """
    w = np.asarray(initial_w, dtype=np.float64).ravel().copy()

    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w)
        w = w - gamma * grad

    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using Gradient Descent with L2 penalty.

    Mathematical Formulation:
        Objective: L_reg(w) = L_logistic(w) + lambda_ * ||w||^2
        Gradient: grad_reg(w) = (1 / N) * tx^T @ (sigma(tx @ w) - y) + 2 * lambda_ * w
        Update step: w_{t+1} = w_t - gamma * grad_reg(w_t)

    Contractual Note:
        The returned loss scalar strictly excludes the regularization penalty
        term (lambda_ * ||w||^2) and represents the unpenalized NLL loss.

    Args:
        y (np.ndarray): Binary target labels in {0, 1} of shape (N,).
        tx (np.ndarray): Input feature matrix of shape (N, D).
        lambda_ (float): Regularization parameter.
        initial_w (np.ndarray): Initial weight vector of shape (D,).
        max_iters (int): Total number of gradient descent iterations.
        gamma (float): Learning rate / step size parameter.

    Returns:
        tuple[np.ndarray, np.float64]:
            w (np.ndarray): Optimal weight vector of shape (D,).
            loss (np.float64): Final unpenalized Negative Log-Likelihood loss scalar.
    """
    w = np.asarray(initial_w, dtype=np.float64).ravel().copy()

    for _ in range(max_iters):
        # Regularized gradient includes 2 * lambda_ * w
        grad = compute_logistic_gradient(y, tx, w) + 2.0 * lambda_ * w
        w = w - gamma * grad

    # Returned loss does NOT include lambda * ||w||^2
    loss = compute_logistic_loss(y, tx, w)
    return w, loss
