import numpy as np
import sklearn.linear_model as skl

from terrain_setup import train, test, val, plot_terrain
from Lasso_Franke import kfold_cross_validation_lasso
from setup import OLS_reg, ridge_reg, create_designmatrix, MSE, R2score
from terrain_analysis_tools import Model_selection_terrain, Model_selection_terrain_with_lambda
from terrain_setup import test, plot_terrain

OLS_degree = Model_selection_terrain(OLS_reg, "OLS", s=4, PolyMax=12)

Ridge_Lambdas = np.logspace(-10, -4, 10)
Ridge_model = Model_selection_terrain_with_lambda(ridge_reg, "Ridge", lambdas=Ridge_Lambdas, PolyMin=1, PolyMax=20)

Lasso_lambdas = np.logspace(-4, -0, 10)
Lasso_model = Model_selection_terrain_with_lambda(0, "Lasso", lambdas=Lasso_lambdas, PolyMin=1, PolyMax=20, cv_model = kfold_cross_validation_lasso)


print(f"  Model   |   P   |  lambda")
print(f"---------------------------")
print(f"   OLS    |   {OLS_degree}   |   n/a")
print(f"  Ridge   |   {Ridge_model[0]}  |   {Ridge_model[1][1]}")
print(f"  Lasso   |   {Lasso_model[0]}   |   {Lasso_model[1][1]}")


np.random.seed(1)

Ridge_lam = Ridge_model[1][1]

Lasso_lam = Lasso_model[1][1]

"""Y train"""
Y_train = train[2].ravel()
Y_val = val[2].ravel()
Y_train = np.concatenate((Y_train, Y_val))

"""OLS"""

X_train = create_designmatrix(train[0], train[1], OLS_degree, scale=True)
X_val = create_designmatrix(val[0], val[1], OLS_degree, scale=True)
X_OLS_train = np.concatenate((X_train, X_val),)


OLS_betas = OLS_reg(X_OLS_train, Y_train)

"""Ridge"""

X_train = create_designmatrix(train[0], train[1], Ridge_model[0], scale=True)
X_val = create_designmatrix(val[0], val[1], Ridge_model[0], scale=True)
X_Ridge_train = np.concatenate((X_train, X_val))

Ridge_betas = ridge_reg(X_Ridge_train, Y_train, Ridge_lam)

"""Lasso"""

X_train = create_designmatrix(train[0], train[1], Lasso_model[0], scale=True)
X_val = create_designmatrix(val[0], val[1], Lasso_model[0], scale=True)
X_Lasso_train = np.concatenate((X_train, X_val))



clf_lasso = skl.Lasso(alpha=Lasso_lam, fit_intercept=False).fit(X_Lasso_train, Y_train)



"""Test data"""
N = test[0].shape[0]
x1 = test[0]
x2 = test[1]
Y_test = test[2].ravel()

X_OLS_test = create_designmatrix(x1, x2, OLS_degree, scale=True)

X_Ridge_test = create_designmatrix(x1, x2, Ridge_model[0], scale=True)

X_Lasso_test = create_designmatrix(x1, x2, Lasso_model[0], scale=True)

OLS_tilde = X_OLS_test @ OLS_betas

Ridge_tilde = X_Ridge_test @ Ridge_betas

Lasso_tilde = clf_lasso.predict(X_Lasso_test)


MSEscores = np.zeros(3)
R2scores = np.zeros(3)
models = [OLS_tilde, Ridge_tilde, Lasso_tilde]
model_names = ["  OLS  ", " Ridge ", " Lasso "]
print("  Model  |    MSE   |   R2   ")
print("------------------------------")
for i in range(3):
    MSEscores[i] = MSE(Y_test, models[i])
    R2scores[i] = R2score(Y_test, models[i])
    print(f" {model_names[i]} | {MSEscores[i]:5.5f}  |  {R2scores[i]:5.4f}")


for i, model in enumerate(models[:3]):
    prediction = model.reshape(N, N)
    plot_terrain(x1, x2, prediction, f"{model_names[i]}, MSE = {MSEscores[i]:.0f}, R2 = {R2scores[i]:.3f}")

plot_terrain(x1, x2, test[2], f"Data set, N = {Y_test.shape[0]}")