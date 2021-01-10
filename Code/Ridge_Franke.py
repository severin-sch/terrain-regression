import numpy as np
import matplotlib.pyplot as plt
from setup import create_designmatrix, MakeData, OLS_reg, MSE, train_test_splitter, ridge_reg
from assessment import kfold_cross_validation
import seaborn as sns
from matplotlib.patches import Rectangle

if __name__ == "__main__":

    lambdas = np.logspace(-4, 0, 25)

    x, y, z, P, complexity = MakeData(p=9)[:5]

    # Each column is a polynomial order, rows are lambda values
    KfoldMSE_Ridge = np.zeros((complexity, len(lambdas)))

    for j, p in enumerate(P):
        X = create_designmatrix(x, y, p, scale=False)
        for i, lam in enumerate(lambdas):
            KfoldMSE_Ridge[j, i] = kfold_cross_validation(X, z, ridge_reg, lam)[0]


    #Find minimum MSE value:
    min_MSE = np.amin(KfoldMSE_Ridge)
    index1 = np.where(KfoldMSE_Ridge == np.amin(KfoldMSE_Ridge))

    print("Minimum MSE value for Lasso", min_MSE)

    #Find second and third minimum
    temp = np.copy(KfoldMSE_Ridge)
    temp[index1[1][0], index1[0][0]] += 10
    index2 = np.where(temp == np.amin(temp))

    temp[index2[0][0], index2[1][0]] += 10
    index3 = np.where(temp == np.amin(temp))

    # Plot the MSEs and the corresponding polynomial degree and lambda value
    KfoldMSE_Ridge_plot = 100 * KfoldMSE_Ridge
    Poly_lables = ["2", "3", "4", "5", "6", "7", "8", "9"]

    f, ax = plt.subplots(figsize=(9, 6))
    ax.add_patch(Rectangle((index1[1][0], index1[0][0] - 1), 1, 1, fill=False, edgecolor='pink', lw=3))
    ax.add_patch(Rectangle((index2[1][0], index2[0][0] - 1), 1, 1, fill=False, edgecolor='green', lw=3))
    ax.add_patch(Rectangle((index3[1][0], index3[0][0] - 1), 1, 1, fill=False, edgecolor='yellow', lw=3))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(KfoldMSE_Ridge_plot[1:, :14],
                     cbar=False,
                     annot=True,
                     square=True,
                     yticklabels=Poly_lables,
                     fmt='.2f')
    plt.xlabel(r"$\lambda$ values enumerated low->high")
    plt.ylabel("Polynomial order")
    plt.title(r'MSE of Ridge regression, scaled $10^2$')
    plt.tight_layout()
    plt.show()


    p = 5
    DesignMatrix = create_designmatrix(x, y, p, scale=True)
    DesignMatrix, X_test, Y_train, Y_test = train_test_splitter(DesignMatrix, z, testsize=0.2)

    MSE_Ridge = np.zeros(len(lambdas))

    for i, lam in enumerate(lambdas):
        Beta = ridge_reg(DesignMatrix, Y_train, lam)
        y_tilde = X_test @ Beta

        MSE_Ridge[i] = MSE(Y_test, y_tilde)

    plt.plot(np.log10(lambdas), MSE_Ridge, label="Ridge, p = 5")
    plt.plot(np.log10(lambdas), KfoldMSE_Ridge[4,:], label="Ridge- cross validation, p=5")
    plt.xlabel(r"Complexity, $log_{10}(\lambda)$")
    plt.ylabel("Mean Squared Error")
    plt.title("Comparison of single Ridge training iterations vs Cross validation")
    plt.legend()
    plt.show()
