import timesfm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error  
from statsmodels.tsa.arima.model import ARIMA  

# Load data
Case_stdy_data = pd.read_csv("./Case_study.csv")
y = Case_stdy_data["V82"].values


train_data = y[0:100].astype(np.float32)
test_data  = y[100:200].astype(np.float32)

# ARIMA (from your Step 4 — fit on train, predict 100 steps)
arima_model  = ARIMA(train_data, order=(3, 1, 1))
arima_fitted = arima_model.fit() #estimating the model parameters
arima_preds  = arima_fitted.forecast(steps=100)

#  Load TimesFM
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        context_len=128,
        horizon_len=100, 
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend="cpu",
    ),
    checkpoint=timesfm.TimesFmCheckpoint(  #pretrained model checkpoint
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    ),
)

# Zero-shot Forecast
forecast_results, experimental_quantiles = tfm.forecast(
    inputs=[train_data],
    freq=[0],   # 0 = high/unknown frequency
)
timesfm_preds = forecast_results[0]   # forecast results 100 points

#MSE Comparison 
mse_arima   = mean_squared_error(test_data, arima_preds)
mse_timesfm = mean_squared_error(test_data, timesfm_preds)

print(f"ARIMA MSE:   {mse_arima:.4f}")
print(f"TimesFM MSE: {mse_timesfm:.4f}")

#plot

train_indices = np.arange(0, 100)
test_indices  = np.arange(100, 200)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(train_indices[-24:], train_data[-24:],
        label="Training Data (last 24)", color="black")

ax.plot(test_indices, test_data,
        label="Actual", color="black", linestyle="--", linewidth=2)

ax.plot(test_indices, arima_preds,
        label=f"ARIMA Forecast (MSE={mse_arima:.2f})",
        color="blue", marker="o", markersize=3)

ax.plot(test_indices, timesfm_preds,
        label=f"TimesFM Zero-Shot (MSE={mse_timesfm:.2f})",
        color="red", marker="s", markersize=3)

ax.axvline(x=100, color="gray", linestyle=":", label="Forecast Start")  # ✅ FIX 9
ax.set_title("ARIMA vs TimesFM Zero-Shot Forecast")
ax.set_xlabel("Time Index")
ax.set_ylabel("V82")
ax.legend()
plt.tight_layout()
plt.savefig("timesfm_vs_arima.png", dpi=150)
plt.show()
