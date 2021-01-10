  FYS-STK3155 Project 1 - Regression analysis and resampling methods
 ====================================================================
By Severin Schirmer

This project aimed to solve a regression problem using linear regression including OLS, Ridge, and Lasso. 

## Repo structure
 
 - `code` - All code related to the project and the terrain data
     - `assessment.` - Functions used for model assessment including bootstrap and cross validation across all files
     - `Franke_test` - Implementation of the selected models on a test set from Franke's function
     - `Comparison_lasso_ridge` - Plots comparison of ridge and lasso for different lambda values
     - `Lasso_Franke` - Model selection for Lasso on Franke's, including seperate function for cross validation for Lasso
     - `OLS_Franke` - Simple introductory analysis of OLS on Franke's
     - `OLS_Franke_bootstrap` - Bias-variance analysis of OLS on Franke's used in model selection
     - `OLS_Franke_crossvalidation` - 5 fold cross validation analysis used in model selectionon Franke's
     - `Ridge_Franke` - Cross validation used for model selection on Franke's
     - `Ridge_terrain` - Bootstrap and cross validation of Ridge on terrain data for single lambda values
     - `setup`- Setup files including implementation of design matrix, producing synthetic Franke data, and MSE and R2 functions
     - `SRTM_data_Norway_1` - Terrain data used in project
     - `terrain_analysis` - Model selection and final test for OLS, Ridge, and Lasso on the terrain data
     - `terrain_analysis_tools`- Model selection functions based on terrain data for OLS, Ridge and Lasso
     - `terrain_setup` - Import of data with splitting into train, validation and test set, and plotting of terrain.
 - `report` - The report 
 - `resources` - All figures produced by code including some that are not in the report.

