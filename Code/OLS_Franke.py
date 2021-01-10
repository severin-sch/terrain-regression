from setup import create_designmatrix, MakeData, OLS_reg, train_test_splitter
import numpy as np
import matplotlib.pyplot as plt
from assessment import R2score
from matplotlib.ticker import MaxNLocator

"""ORDINARY LEAST SQUARES REGRESSION ON THE FRANKE FUNCTION"""

"""Set Print_ConfInfs_Beta to 1 if you wish to print the confidence 
intervals for the betas of the polynomial of the first grade. 0 if not.

Set it to 2 if you wish to print the confidence intervals for
betas for all polynomials."""
Print_ConfInfs_Beta = 1

np.random.seed(42)

#Initiates the synthetic data used
x, y, z, P, complexity, x_plot, y_plot, z_plot = MakeData(N = 10, p=7)


def OLS_analysis(x, y, P, z, R2 = False, confinf_beta=False, print_error=False, save_X=False):
    """
    Performs OLS regression on the dataset for polynomials from order 1 to 5

    The data is scaled, split, and then the model is trained. The mean squared
    error of the testing and the R2 score is printed for each polynomial.
    """
    R2_list = []
    ConfInfB = []
    MSE_list = np.zeros((len(P), 2))
    Betas = []
    X_list = []

    if print_error:
        print("  P  |     MSE   |   R2   ")
        print("--------------------------")

    for i, p in enumerate(P):

        X = create_designmatrix(x, y, p)
        X_train, X_test, Y_train, Y_test = train_test_splitter(X, z, testsize=0.2)

        # Finding Beta
        B = OLS_reg(X_train, Y_train)
        Betas.append(B)

        # Predicting using OLS
        ytilde_train = X_train @ B
        ytilde_test = X_test @ B

        # Finding MSE of training and test data
        MSE_list[i, 1] = np.mean(np.square(Y_test - ytilde_test))  # Test MSE
        MSE_list[i, 0] = np.mean(np.square(Y_train - ytilde_train))  # Training MSE

        if save_X:
            X_list.append(X)

        if R2:
            R2_list.append(R2score(Y_test, ytilde_test))

        if confinf_beta:
            sig2 = np.mean(np.square(Y_train - ytilde_train))
            VarB = sig2 * np.linalg.inv(X_train.T @ X_train).diagonal()
            ConfInf = np.zeros((len(B), 2))
            ConfInf[:, 0] = B - 2 * np.sqrt(VarB)
            ConfInf[:, 1] = B + 2 * np.sqrt(VarB)
            ConfInfB.append(ConfInf)

        if print_error:
            print(f" {p:2.0f}  |   {MSE_list[p-1][0]:.4f}  |  {R2_list[p-1]:.4f}")

    return Betas, MSE_list, X_list, R2_list, ConfInfB,



if __name__ == "__main__":
    Betas, MSE_list, X_list, R2_list, ConfInfB = OLS_analysis(x, y, P, z,
                                                              R2=True,
                                                              confinf_beta=True,
                                                              print_error=True)

    plt.style.use('ggplot')
    fig = plt.figure().gca()
    plt.plot(P, MSE_list[:, 1], label="Testing MSE")
    plt.plot(P, MSE_list[:, 0], label="Training MSE")
    plt.legend()
    plt.xlabel("Complexity")
    plt.ylabel("MSE")
    fig.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Overfitting example OLS on Franke")
    plt.show()

    if Print_ConfInfs_Beta == 1:
        print("--------------------------")
        print("Confidence intervals of the \n Betas for the fourth polynomial:")
        for conf, bet in zip(ConfInfB[3], Betas[3]):
            print(f"  {bet:6.2f} & {conf[0]:6.2f} & {conf[1]:6.2f} ")
            print("\hline")

    elif Print_ConfInfs_Beta == 2:
        print("--------------------------")
        print("Confidence intervals of the \n Betas for the all polynomials:")
        print(ConfInfB)
