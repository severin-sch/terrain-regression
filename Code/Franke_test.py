import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sklearn.linear_model as skl

from setup import create_designmatrix, MakeData, OLS_reg, MSE, train_test_splitter, R2score, plot_Franke
from Ridge import ridge_reg

np.random.seed(1)


lambdas = np.logspace(-4, 0, 25)

Ridge_lam = lambdas[8]

Lasso_lam = lambdas[1]

"Training data"
x, y, Y_train = MakeData(N=10, s=3)[:3]


"""OLS: polynomial degree = 4"""
X_OLS = create_designmatrix(x, y, 4)
X_OLS_train, Y_OLS_train = train_test_splitter(X_OLS, Y_train, testsize=0.2)[0:3:2]
OLS_betas = OLS_reg(X_OLS_train, Y_OLS_train)

"""Ridge: polynomial degree = 5"""
X_Ridge = create_designmatrix(x, y, 5)
X_Ridge_train, Y_Ridge_train = train_test_splitter(X_Ridge, Y_train, testsize=0.2)[0:3:2]
Ridge_betas = ridge_reg(X_Ridge_train, Y_Ridge_train, Ridge_lam)

"""Lasso: polynomial degree = 12"""
X_Lasso = create_designmatrix(x, y, 12)
X_Lasso_train, Y_Lasso_train = train_test_splitter(X_Lasso, Y_train, testsize=0.2)[0:3:2]
clf_lasso = skl.Lasso(alpha=Lasso_lam, fit_intercept=False, max_iter=2000).fit(X_Lasso_train, Y_Lasso_train)



"""Test data"""
N = 20
x, y, Y_test, void, void, x_plot, y_plot, Franke = MakeData(N=N, s=42)

X_OLS_test = create_designmatrix(x, y, 4)

X_Ridge_test = create_designmatrix(x, y, 5)

X_Lasso_test = create_designmatrix(x, y, 12)

OLS_tilde = X_OLS_test @ OLS_betas

Ridge_tilde = X_Ridge_test @ Ridge_betas

Lasso_tilde = clf_lasso.predict(X_Lasso_test)


MSEscores = np.zeros(3)
R2scores = np.zeros(3)
models = [OLS_tilde, Ridge_tilde, Lasso_tilde, Franke]
model_names = ["  OLS  ", " Ridge ", " Lasso ", " Data "]
print("  Model  |    MSE   |   R2   ")
print("------------------------------")
for i in range(3):
    MSEscores[i] = MSE(Y_test, models[i])
    R2scores[i] = R2score(Y_test, models[i])
    print(f" {model_names[i]} | {MSEscores[i]:5.5f}  |  {R2scores[i]:5.4f}")

for i, model in enumerate(models[:3]):
    prediction = model.reshape(N, N)
    plot_Franke(x_plot, y_plot, prediction, f"{model_names[i]}, MSE = {MSEscores[i]:.4f}, R2 = {R2scores[i]:.3f}")

plot_Franke(x_plot, y_plot, Franke, f"Data set, N = {N**2}")