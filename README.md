# Time-Series-Project

A personal project comparing **TimesFM** and **ARIMA(p, d, q)** models for time series prediction and forecasting.

## Project Overview

This project evaluates and compares two distinct approaches to time series forecasting:

- **ARIMA (AutoRegressive Integrated Moving Average)**: A classical statistical approach for time series modeling and forecasting
- **TimesFM (Time Series Foundation Model)**: A modern machine learning-based foundation model for time series analysis

## Objective

Propose an optimal time series specification and forecast methodology by comparing the performance and characteristics of ARIMA and TimesFM models across different datasets and scenarios.

## Technologies

- **Languages**: R, Python
- **Focus Areas**: 
  - Time series analysis and forecasting
  - Model comparison and evaluation
  - Statistical and machine learning approaches

## Structure

The project contains implementations in both R and Python, leveraging the strengths of each language for time series analysis and modeling.

---

## Results & Findings

### Key Conclusions

Through rigorous comparison and experimentation, I discovered that:

1. **Foundation Models (TimesFM) Out-of-the-Box Performance**: Foundation models demonstrate superior performance without extensive tuning, showcasing their ability to leverage pre-trained knowledge on large-scale time series datasets.

2. **Machine Learning Model Optimization**: However, when optimized properly, traditional machine learning models like ARIMA actually outperform foundation models. The key lies in:
   - Proper hyperparameter tuning
   - Feature engineering
   - Statistical validation and model diagnostics
   - Domain-specific optimization strategies

3. **Practical Implications**: While ARIMA models require more careful calibration and domain expertise, the performance gains justify the effort for production forecasting systems where accuracy is critical.

---

*This is a personal research project exploring modern and classical approaches to time series forecasting.*
