# Multivariate Regression Model to Predict Cancer Death Rates

## Project Description

The dataset contains information about cancer death rates and various demographic, environmental, and socioeconomic factors across counties. The goal is to:

Identify highly correlated features and remove redundancy.

Train a regression model to predict the death rate.

Identify significant variables that impact the predictions.


## Features of the Project

Exploratory Data Analysis (EDA) to understand the dataset.

Data preprocessing to handle highly correlated features.

Implementation of an Ordinary Least Squares (OLS) regression model.

Identification of statistically significant variables based on p-values.



Summary of the ModeL:
```plaintext
OLS Regression Results                            
==============================================================================
Dep. Variable:       TARGET_deathRate   R-squared:                       0.676
Model:                            OLS   Adj. R-squared:                  0.658
Method:                 Least Squares   F-statistic:                     37.62
Date:                Tue, 14 Jan 2025   Prob (F-statistic):               0.00
Time:                        06:52:58   Log-Likelihood:                -10164.
No. Observations:                2437   AIC:                         2.059e+04
Df Residuals:                    2308   BIC:                         2.133e+04
Df Model:                         128                                         
Covariance Type:            nonrobust                                         
================================================================================
coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------
const                               104.1644      3.911     26.632      0.000      96.494     111.834
avgAnnCount                          -0.0292      0.004     -7.910      0.000      -0.036      -0.022
avgDeathsPerYear                      0.1128      0.012      9.514      0.000       0.090       0.136
incidenceRate                         0.2116      0.007     29.017      0.000       0.197       0.226
PctHS18_24                            0.0901      0.044      2.067      0.039       0.005       0.176
PctBachDeg18_24                      -0.3978      0.117     -3.398      0.001      -0.627      -0.168
PctBachDeg25_Over                    -2.1012      0.100    -21.061      0.000      -2.297      -1.906
county_Addison County              4.679e-12    2.6e-12      1.803      0.072   -4.11e-13    9.77e-12
county_Aleutians West Census Area    53.1135     16.798      3.162      0.002      20.173      86.054
...

