# ETT Transformer Anomaly Detection

This project investigates deep learning-based time series forecasting and anomaly detection using the Electricity Transformer Temperature (ETT) datasets. The main goal is to predict oil temperature (OT) using historical multivariate time series data and identify potential anomalies based on prediction errors.

## Project Objectives

- Explore and preprocess ETT time series datasets.
- Build an LSTM baseline model for oil temperature forecasting.
- Develop a Transformer-based forecasting model.
- Compare model performance using MAE and RMSE.
- Detect anomalies using prediction errors.
- Visualize forecasting results and detected anomalies.

## Dataset

The project uses four ETT datasets:

- ETTh1
- ETTh2
- ETTm1
- ETTm2

The target variable is `OT` (Oil Temperature).

## Methods

- Exploratory Data Analysis
- Sliding window sequence generation
- LSTM forecasting
- Transformer forecasting
- Prediction-error-based anomaly detection

## Project Structure

```text
data/
notebooks/
src/
results/