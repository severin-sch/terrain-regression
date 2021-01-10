import numpy as np
from sklearn.preprocessing import StandardScaler
from terrain_setup import train, test, val, plot_terrain
from assessment import bootstrap, kfold_cross_validation
from setup import ridge_reg, create_designmatrix

PolyMax = 20
Polys = np.arange(1, PolyMax+1)


Bias = np.zeros(PolyMax)
Variance = np.zeros(PolyMax)
MSE_boot = np.zeros(PolyMax)
MSE_cv = np.copy(Bias)
R2_cv = np.copy(Bias)
MSE_std_cv = np.copy(Bias)

for i, p in enumerate(Polys):
    X_train = create_designmatrix(train[0], train[1], p)
    Y_train = train[2].ravel()
    X_val = create_designmatrix(val[0], val[1], p)
    Y_val = val[2].ravel()

    data = [X_train, X_val, Y_train, Y_val]
    """Bootstrap"""
    N_boots = 100
    Bias[i], Variance[i], MSE_boot[i] = bootstrap(data, N_boots, ridge_reg)

    """Cross-validation"""
    X_cv = np.concatenate((X_train, X_val))
    Y_cv = np.concatenate((Y_train, Y_val))
    MSE_cv[i], R2_cv[i], MSE_std_cv[i]= kfold_cross_validation(X_cv, Y_cv, ridge_reg, k=10, lam=1e-10)


if __name__ == "__main__":

    MSE_min_boot = np.amin(MSE_boot)
    ind_min_boot = np.where(MSE_boot == MSE_min_boot)[0][0]
    print(f"Minimum MSE for bootstrap= {MSE_min_boot}, P = {ind_min_boot+1}")
    print(MSE_boot[ind_min_boot])

    MSE_min_cv = np.amin(MSE_cv)
    ind_min_cv = np.where(MSE_cv == MSE_min_cv)[0][0]
    print(f"Minimum MSE for bootstrap= {MSE_min_cv}, P = {ind_min_cv+1}")
    print(MSE_cv[ind_min_cv])

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    """Plot Bias-variance analysis"""
    plt.style.use('ggplot')
    fig = plt.figure().gca()
    plt.plot(Polys, Variance, label="Variance")
    plt.plot(Polys, Bias, label="Bias")
    plt.plot(Polys, MSE_boot, label="MSE")
    plt.legend()
    plt.xlabel("Complexity")
    plt.ylabel("Error")
    plt.title("Bias-Variance Tradeoff")
    plt.ylim((-10, 30000))
    fig.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

    STD_line = [MSE_std_cv[ind_min_cv] + MSE_cv[ind_min_cv]] * len(Polys)

    for i, error in enumerate(MSE_cv):
        if error < MSE_std_cv[ind_min_cv] + MSE_cv[ind_min_cv]:
            chosen_model = Polys[i]
            break

    """Plot MSE of bootstrap and cross validation"""
    plt.style.use('ggplot')
    fig = plt.figure().gca()
    #plt.plot(Polys, MSE_boot, label="Bootstrap MSE")
    plt.plot(Polys, MSE_cv, label="Cross validation MSE")
    plt.plot(Polys, STD_line, color="Orange", linestyle="--")
    plt.vlines(chosen_model, -10, 30000, colors='Orange', linestyles='--')
    plt.legend()
    plt.ylim((-10, 30000))
    plt.xlabel("Complexity")
    plt.ylabel("MSE")
    plt.title("MSE analysis of OLS")
    fig.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()