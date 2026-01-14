import numpy as np

def gaussian_pdf(x, mean, cov):
    n = len(x)
    diff = x - mean
    det = np.linalg.det(cov)
    if det <= 0:
        return 0.0
    norm = np.sqrt((2*np.pi)**n * det)
    return np.exp(-0.5 * diff.T @ np.linalg.inv(cov) @ diff) / norm


def log_gaussian_pdf(x, mean, cov, eps=1e-9):
    n = len(x)
    diff = x - mean

    # Regularize covariance
    cov = cov + eps * np.eye(n)

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return -np.inf

    # Solve L y = diff
    y = np.linalg.solve(L, diff)

    maha = y @ y
    logdet = 2.0 * np.sum(np.log(np.diag(L)))

    return -0.5 * (maha + logdet + n * np.log(2 * np.pi))
