import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
import seaborn as sns

from terrain_setup import train, test, val, plot_terrain
from assessment import bootstrap, kfold_cross_validation
from setup import OLS_reg, ridge_reg, create_designmatrix

def Model_selection_terrain(model, modelname, PolyMax=12, s=3):
    Polys = np.arange(1, PolyMax+1)

    Bias = np.zeros(PolyMax)
    Variance = np.zeros(PolyMax)
    MSE_boot = np.zeros(PolyMax)
    MSE_cv = np.copy(Bias)
    R2_cv = np.copy(Bias)
    MSE_std_cv = np.copy(Bias)

    for i, p in enumerate(Polys):
        X_train = create_designmatrix(train[0], train[1], p, scale=False)
        Y_train = train[2].ravel()
        X_val = create_designmatrix(val[0], val[1], p, scale=False)
        Y_val = val[2].ravel()

        """Cross-validation"""
        X_cv = np.concatenate((X_train, X_val))
        Y_cv = np.concatenate((Y_train, Y_val))
        MSE_cv[i], R2_cv[i], MSE_std_cv[i] = kfold_cross_validation(X_cv, Y_cv, model, k=10, s=s)


        """Bootstrap"""
        scaler = StandardScaler()
        scaler.fit(X_train[:, 1:])
        X_train[:, 1:] = scaler.transform(X_train[:, 1:])
        X_val[:, 1:] = scaler.transform(X_val[:, 1:])
        data = [X_train, X_val, Y_train, Y_val]
        N_boots = 100
        Bias[i], Variance[i], MSE_boot[i] = bootstrap(data, N_boots, model, s=s)


    MSE_min_boot = np.amin(MSE_boot)
    ind_min_boot = np.where(MSE_boot == MSE_min_boot)[0][0]
    print(f"Minimum MSE for bootstrap= {MSE_min_boot}, P = {ind_min_boot+1}")

    MSE_min_cv = np.amin(MSE_cv)
    ind_min_cv = np.where(MSE_cv == MSE_min_cv)[0][0]
    print(f"Minimum MSE for cross validation= {MSE_min_cv}, P = {ind_min_cv+1}")

    """Plot Bias-variance analysis"""
    plt.style.use('ggplot')
    fig = plt.figure().gca()
    plt.plot(Polys, Variance, label="Variance")
    plt.plot(Polys, Bias, label="Bias")
    plt.plot(Polys, MSE_boot, label="MSE")
    plt.legend()
    plt.xlabel("Complexity")
    plt.ylabel("Error")
    plt.title(f"Bias-Variance Tradeoff {modelname}")
    plt.ylim((-10, 30000))
    fig.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

    STD_line = [MSE_std_cv[ind_min_cv] + MSE_cv[ind_min_cv]] * len(Polys)

    for i, error in enumerate(MSE_cv):
        if error < MSE_std_cv[ind_min_cv] + MSE_cv[ind_min_cv]:
            chosen_model = Polys[i]
            break

    """Plot MSE of boo
    tstrap and cross validation"""
    plt.style.use('ggplot')
    fig = plt.figure().gca()
    plt.plot(Polys, MSE_boot, label="Bootstrap MSE")
    plt.plot(Polys, MSE_cv, label="Cross validation MSE")
    plt.plot(Polys, STD_line, color="Orange", linestyle="--")
    plt.vlines(chosen_model, -10, 30000, colors='Orange', linestyles='--')
    plt.legend()
    plt.ylim((-10, 30000))
    plt.xlabel("Complexity")
    plt.ylabel("MSE")
    plt.title(f"MSE analysis of {modelname}")
    fig.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

    return chosen_model


def Model_selection_terrain_with_lambda(model, modelname, lambdas=np.logspace(-4, 0, 25), PolyMin=1, PolyMax=12, cv_model=kfold_cross_validation):

    Polys = np.arange(PolyMin, PolyMax + 1)

    # Each column is a polynomial order, rows are lambda values
    MSE_cv = np.zeros((len(Polys), len(lambdas)))
    MSE_std_cv = np.zeros((len(Polys), len(lambdas)))

    for j, p in enumerate(Polys):
        X_train = create_designmatrix(train[0], train[1], p, scale=False)
        Y_train = train[2].ravel()
        X_val = create_designmatrix(val[0], val[1], p, scale=False)
        Y_val = val[2].ravel()
        X_cv = np.concatenate((X_train, X_val))
        Y_cv = np.concatenate((Y_train, Y_val))
        for i, lam in enumerate(lambdas):
            MSE_cv[j, i], MSE_std_cv[j,i] = cv_model(X_cv, Y_cv, model, lam, k=7)[0:3:2]


    #Find minimum MSE value:
    min_MSE = np.amin(MSE_cv)
    index1 = np.where(MSE_cv == np.amin(MSE_cv))
    min_MSE_std = MSE_std_cv[index1[0][0], index1[1][0]]
    print("Minimum MSE value for Lasso", min_MSE)

    #Find simplest model within one standard error
    temp = np.where(MSE_cv < min_MSE+min_MSE_std)
    print("Minimum MSE + std(Minimum MSE) = ",min_MSE + min_MSE_std)


    for i, j in enumerate((MSE_cv < min_MSE + min_MSE_std).T):
        for k, l in enumerate(j):
            if l:
                break
        else:
            continue
        break

    chosenmodelpol = k + 1
    chosenmodellam = [i, lambdas[i]]

    # Plot the MSEs and the corresponding polynomial degree and lambda value
    Scaled_for_plot = 0.01 * MSE_cv
    Pplot = (1, PolyMax)
    Lamplot = (0, len(lambdas))
    Poly_lables = [str(x) for x in range(Pplot[0], Pplot[1]+1)]
    Lam_lables = [str(x) for x in range(Lamplot[0], Lamplot[1])]

    f, ax = plt.subplots(figsize=(9, 6))
    ax.add_patch(Rectangle((index1[0][0], index1[1][0]), 1, 1, fill=False, edgecolor='pink', lw=3))

    sns.set(font_scale=1)
    ax = sns.heatmap(Scaled_for_plot.T,
                     cbar=False,
                     annot=True,
                     square=True,
                     #yticklabels=Lam_lables,
                     xticklabels=Poly_lables,
                     fmt='3.0f')
    plt.ylabel(r"$\lambda$ values enumerated low->high")
    plt.xlabel("Polynomial order")
    plt.title(f'MSE of {modelname} regression,' + r"scaled $10^{-2}$")
    plt.tight_layout()
    plt.show()

    return chosenmodelpol, chosenmodellam

