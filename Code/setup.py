import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def R2score(Y_test, Y_tilde):
    """Find R2 score"""
    return 1 - np.sum(np.square(Y_test - Y_tilde)) / np.sum(np.square(Y_test - np.mean(Y_test)))


def MakeData(p=5, N = 10, s=3):
    """Takes in max polynomial order and the squareroot of number of datapoints.
    Returns input data: x,y and output: z, and a list of polynomial orders and Pmax
    x_mat and y_mat is returned for plotting."""
    np.random.seed(s)

    P = list(range(1, p+1))

    x_rand = np.random.uniform(0, 1, size=N)
    y_rand = np.random.uniform(0, 1, size=N)

    ind_sort_row = np.argsort(x_rand)
    ind_sort_col = np.argsort(y_rand)

    x = x_rand[ind_sort_row]
    y = y_rand[ind_sort_col]

    x_mat, y_mat = np.meshgrid(x, y)

    noise_strength = 0.1
    noise = np.random.normal(0, 1, (N,N))
    z_mat = FrankeFunction(x_mat, y_mat) + noise_strength * noise
    z = z_mat.ravel()
    x = x_mat.ravel()
    y = y_mat.ravel()

    return x, y, z, P, P[-1], x_mat, y_mat, z_mat


def train_test_splitter(X, y, testsize=0.2):
    np.random.seed(10)
    arr_rand = np.random.rand(X.shape[0])
    share = (1-testsize)*100
    split = arr_rand < np.percentile(arr_rand, share)

    X_train = X[split]
    y_train = y[split]
    X_test = X[~split]
    y_test = y[~split]

    return X_train, X_test, y_train, y_test



"""Numba doesn't like the if-statements the following
lines remove the warnings of inefficiency"""
from numba.core.errors import NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)

from numba import jit
@jit
def create_designmatrix(x, y, n, scale=True):
    """Takes in x and y data and creates a designmatrix for a polynomial to the nth order"""
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)
    N = len(x)
    l = int((n + 1) * (n + 2) / 2)   # Number of elements in beta
    X = np.ones((N,l))
    for i in range(1, n + 1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
               X[:,q+k] = (x**(i-k))*(y**k)
    if scale:
        scaler = StandardScaler()
        scaler.fit(X[:, 1:])
        X[:, 1:] = scaler.transform(X[:, 1:])
    return X

def OLS_reg(X, Y, lam=0):
    """Returns the Beta values for OLS.
    lam is added as an unused parameter for compatibility throughout"""
    return np.linalg.inv(X.T @ X) @ X.T @ Y

def ridge_reg(X, y, lam):
    I = np.eye(X.shape[1])
    Beta = np.linalg.inv(X.T @ X + lam * I) @ X.T @ y
    return Beta


def MSE(y_data, y_tilde):
    n = np.size(y_tilde)
    return np.sum((y_data - y_tilde) ** 2) / n


def plot_Franke(x_mat, y_mat, prediction, title):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x_mat, y_mat, prediction, cmap=cm.viridis, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_zlim(-0.15, 1.30)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.view_init(30, 30)
    ax.set_title(title)
    plt.show()


if __name__ == "__main__":

    #Creates data
    x, y, z, P, complexity, x_plot, y_plot, z_plot = MakeData(N=10)

    #Plots input and output data
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x_plot, y_plot, z_plot, cmap=cm.viridis, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.view_init(30, 30)
    plt.show()