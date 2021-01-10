from setup import create_designmatrix, MakeData, OLS_reg, R2score, train_test_splitter
from OLS_Franke_bootstrap import run_bootstrap_OLS
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from assessment import kfold_cross_validation

if __name__ == "__main__":
    BootMSE, P, x, y, z = run_bootstrap_OLS()[2:]

    KfoldMSE = []
    KfoldR2 = []

    for p in P:
        X = create_designmatrix(x, y, p, scale=False)
        MSE, R2 = (kfold_cross_validation(X, z, OLS_reg))[:2]
        KfoldMSE.append(MSE)
        KfoldR2.append(R2)

    print("  P  |     MSE   |   R2   ")
    print("--------------------------")
    for i in range(len(KfoldMSE)):
        print(f" {P[i]:2.0f}  |   {KfoldMSE[i]:.4f}  |  {KfoldR2[i]:.4f}")

    plt.style.use('ggplot')
    fig = plt.figure().gca()
    plt.plot(P, BootMSE, label="Bootstrap MSE")
    plt.plot(P, KfoldMSE, label="Cross validation MSE")
    plt.legend()
    plt.ylim((-0.01, 0.15))
    plt.xlabel("Complexity")
    plt.ylabel("MSE")
    plt.title("MSE analysis of OLS")
    fig.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

    fig = plt.figure().gca()
    plt.plot(P, KfoldR2, label="R2 score")
    plt.legend()
    plt.xlabel("Complexity")
    plt.ylabel("R2")
    fig.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()