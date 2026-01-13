import numpy as np

def gaussian_pdf(x, mean, cov):
    n = len(x)
    diff = x - mean
    det = np.linalg.det(cov)
    if det <= 0:
        return 0.0
    norm = np.sqrt((2*np.pi)**n * det)
    return np.exp(-0.5 * diff.T @ np.linalg.inv(cov) @ diff) / norm