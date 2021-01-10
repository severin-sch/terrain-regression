import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from setup import create_designmatrix, MakeData, ridge_reg, OLS_reg, MSE
import sklearn.linear_model as skl



x, y, z, P, complexity = MakeData(N=10)[:5]

nlambdas = 100
_lambda = np.logspace(-4, 1, nlambdas)

MSEPredictLasso = np.zeros(nlambdas)
MSEPredictRidge = np.zeros(nlambdas)
MSEPredictOLS = np.zeros(nlambdas)
MSEPredictRidge_mine = np.zeros(nlambdas)

p = 10
X = create_designmatrix(x, y, p)
X_train, X_test, Y_train, Y_test = train_test_split(X, z, test_size=0.2)
for i in range(nlambdas):
    lam = _lambda[i]

    clf_ridge = skl.Ridge(alpha=lam, fit_intercept=False).fit(X_train, Y_train)
    ridge_beta_mine = ridge_reg(X_train, Y_train, lam)
    clf_lasso = skl.Lasso(alpha=lam, fit_intercept=False).fit(X_train, Y_train)
    ols_beta = OLS_reg(X_train, Y_train)

    yridge = clf_ridge.predict(X_test)
    yridge_mine = X_test @ ridge_beta_mine
    ylasso = clf_lasso.predict(X_test)
    yols = X_test @ ols_beta

    MSEPredictLasso[i] = MSE(Y_test, ylasso)
    MSEPredictRidge[i] = MSE(Y_test, yridge)
    MSEPredictRidge_mine[i] = MSE(Y_test,yridge_mine)
    MSEPredictOLS[i] = MSE(Y_test, yols)

#then plot the results
plt.figure()
plt.plot(np.log10(_lambda), MSEPredictRidge_mine, 'r--', label = 'MSE Ridge Test')
plt.plot(np.log10(_lambda), MSEPredictLasso, 'g--', label = 'MSE Lasso Test')

plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.legend()
plt.ylim((0,0.1))
plt.show()