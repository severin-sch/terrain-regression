import numpy as np
from sklearn.preprocessing import StandardScaler

def MSE(y_test, y_tilde):
    return np.mean(np.square(y_tilde - y_test))

def R2score(Y_test, Y_tilde):
    """Find R2 score"""
    return 1 - np.sum(np.square(Y_test - Y_tilde)) / np.sum(np.square(Y_test - np.mean(Y_test)))

def bootstrap(data, n, model, lam=0, s=2):
    """Bootstrap of the data X with n number of bootstraps returning bias and variance of Ytilde"""
    X_train, X_test, Y_train, Y_test = data

    #A matrix with all Ytilde for every bootstrap
    Ytilde = np.zeros((len(X_test), n))

    np.random.seed(s)
    for k in range(n):
        i = np.random.randint(0, len(X_train), len(X_train))
        Xboot = X_train[i]
        Yboot = Y_train[i]
        Beta = model(Xboot, Yboot, lam)
        Ytilde[:, k] = X_test @ Beta

    Variance = np.mean(Ytilde.var(axis=1))
    Bias = np.mean((Y_test - Ytilde.mean(axis=1))**2)
    MSEs = MSE(Y_test[:, None], Ytilde)
    return Bias, Variance, MSEs


def kfold_cross_validation(X, Y, function, lam=0, k=5, s=3):
    """Performs K-fold cross validation on a dataset and returns the average MSE"""

    np.random.seed(s)
    n = len(X)
    ind = np.arange(n)

    np.random.shuffle(ind)

    X = X[ind]
    Y = Y[ind]
    K_size = int(round(n / k))
    inds = list(range(n))

    Kfold = []
    for i in range(0, n, K_size):
        Kfold.append(inds[i:i + K_size])
    while len(Kfold) > k:
        Kfold.pop()

    MSEs = np.zeros(k)
    R2 = np.zeros(k)
    for i, g in enumerate(Kfold):
        X_test = X[g]
        Y_test = Y[g]
        X_train = np.delete(X, g, axis=0)
        Y_train = np.delete(Y, g, axis=0)

        scaler = StandardScaler()
        scaler.fit(X_train[:, 1:])
        X_train[:, 1:] = scaler.transform(X_train[:, 1:])
        X_test[:, 1:] = scaler.transform(X_test[:, 1:])

        Beta = function(X_train, Y_train, lam)
        Y_tilde = X_test @ Beta
        MSEs[i] = MSE(Y_tilde, Y_test)
        R2[i] = R2score(Y_test, Y_tilde)

    return np.mean(MSEs), np.mean(R2), np.std(MSEs)