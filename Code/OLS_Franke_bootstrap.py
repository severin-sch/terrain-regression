from setup import create_designmatrix, MakeData, OLS_reg, train_test_splitter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from assessment import bootstrap

def run_bootstrap_OLS():
    """Runs an instance of bootstrap"""
    x, y, z, P, complexity = MakeData(p=7)[:5]
    BootVariances = np.zeros(complexity)
    BootBias = np.zeros(complexity)
    BootMSE = np.zeros(complexity)


    for p in P:
        """Performs bootstrap resampling on each polynomial
         order with N number of bootstraps"""
        X = create_designmatrix(x, y, p)
        data = train_test_splitter(X, z, testsize=0.2)
        N = 1000
        BootBias[p-1], BootVariances[p-1], BootMSE[p-1] = bootstrap(data, N, OLS_reg)

    return BootVariances, BootBias, BootMSE, P, x, y, z

if __name__ == "__main__":

    """Example run of Bootstrap for OLS"""

    BootVariances, BootBias, BootMSE, P = run_bootstrap_OLS()[:4]

    plt.style.use('ggplot')
    fig = plt.figure().gca()
    plt.plot(P, BootVariances, label="Variance")
    plt.plot(P, BootBias, label="Bias")
    plt.plot(P, BootMSE, label="MSE")
    plt.legend()
    plt.xlabel("Complexity")
    plt.ylabel("Error")
    plt.title("Bias-Variance Tradeoff")
    plt.ylim((0, 0.1))
    fig.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()
