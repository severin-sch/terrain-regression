import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from setup import create_designmatrix, MakeData, MSE, train_test_splitter, R2score

@ignore_warnings(category=ConvergenceWarning)
def kfold_cross_validation_lasso(X, Y, function=0, lam=0, k=5, s=3):
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

    MSE = np.zeros(k)
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

        clf_lasso = skl.Lasso(alpha=lam, fit_intercept=False).fit(X_train, Y_train)

        Y_tilde = clf_lasso.predict(X_test)

        MSE[i] = np.mean(np.square(Y_test - Y_tilde))
        R2[i] = R2score(Y_test,Y_tilde)

    return np.mean(MSE), np.mean(R2), np.std(MSE)

if __name__ == "__main__":

    lambdas = np.logspace(-4, 0, 25)

    x, y, z, P, complexity = MakeData(p=13)[:5]

    # Each column is a polynomial order, rows are lambda values
    Kfold_MSE_Lasso = np.zeros((complexity, len(lambdas)))


    for j, p in enumerate(P):
        X = create_designmatrix(x, y, p)
        for i, lam in enumerate(lambdas):
            Kfold_MSE_Lasso[j, i] = kfold_cross_validation_lasso(X, z, lam=lam)[0]

    #Find minimum MSE and its index
    min_MSE = np.amin(Kfold_MSE_Lasso)
    min_index = np.where(Kfold_MSE_Lasso == np.amin(Kfold_MSE_Lasso))

    print("Minimum MSE value for Lasso", min_MSE)

    #Plot the MSEs and the corresponding polynomial degree and lambda value

    f, ax = plt.subplots(figsize=(9, 6))

    polymincrop = 5

    ax.add_patch(Rectangle((min_index[1][0], min_index[0][0]-polymincrop), 1, 1, fill=False, edgecolor='pink', lw=3))

    lambdas_labels = np.array_str(np.log10(lambdas))
    Kfold_MSE_Lasso_scaled = 100 * Kfold_MSE_Lasso
    Poly_lables = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]
    sns.set(font_scale=1.2)
    ax = sns.heatmap(Kfold_MSE_Lasso_scaled[polymincrop:, :10],
                     cbar=False,
                     annot=True,
                     square=True,
                     yticklabels=Poly_lables[polymincrop-1:],
                     fmt='.2f')
    plt.xlabel(r"$\lambda$ values enumerated low->high")
    plt.ylabel("Polynomial order")
    plt.title(r'MSE of Lasso, scaled $10^2$')
    plt.tight_layout()
    plt.show()

