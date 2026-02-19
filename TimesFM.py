import timesfm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error  # ✅ Was missing entirely
from statsmodels.tsa.arima.model import ARIMA  # ✅ Need this for ARIMA

# ── Load data ──────────────────────────────────────────────────────────────
Case_stdy_data = pd.read_csv("./Case_study.csv")
y = Case_stdy_data["V82"].values

# ✅ FIX 1: y[1:100] skips the first value for no reason — use y[0:100]
train_data = y[0:100].astype(np.float32)
test_data  = y[100:200].astype(np.float32)

# ── ARIMA (from your Step 4 — fit on train, predict 100 steps) ─────────────
# ✅ FIX 2: You referenced arima_preds but never defined it in this file
arima_model  = ARIMA(train_data, order=(3, 1, 1))   # replace p,d,q with your Step 4 values
arima_fitted = arima_model.fit()
arima_preds  = arima_fitted.forecast(steps=100)

# ── Load TimesFM ───────────────────────────────────────────────────────────
# ✅ FIX 3: The old constructor API changed — use TimesFmHparams + TimesFmCheckpoint
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        context_len=128,
        horizon_len=100,         # ✅ FIX 4: set to 100 to match your test length
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend="cpu",
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    ),
)

# ── Zero-shot Forecast ─────────────────────────────────────────────────────
forecast_results, experimental_quantiles = tfm.forecast(
    inputs=[train_data],
    freq=[0],   # 0 = high/unknown frequency
)

# ✅ FIX 5: You called model.forecast() twice unnecessarily — removed duplicate call
# ✅ FIX 6: forecast_results[0] already has horizon_len=100 points, no slicing needed
timesfm_preds = forecast_results[0]   # shape: (100,)

# ── MSE Comparison ─────────────────────────────────────────────────────────
mse_arima   = mean_squared_error(test_data, arima_preds)
mse_timesfm = mean_squared_error(test_data, timesfm_preds)

print(f"ARIMA MSE:   {mse_arima:.4f}")
print(f"TimesFM MSE: {mse_timesfm:.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────
# ✅ FIX 7: You wrote ax.plot(train_data[-24:], train_data[-24:]) — x and y were
#           both set to train_data. X-axis should be index positions.
# ✅ FIX 8: test_data is a numpy array, not a DataFrame — has no .index or .values
# ✅ FIX 9: ax.axvline used "test.index[0]" — "test" is the tkinter import, not your data

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

# ── Fine-tuning ────────────────────────────────────────────────────────────
# ✅ FIX 10: You used undefined variables: "context", "tfm" (inconsistent naming),
#            "frequency_input", "h" — all fixed below
# ✅ NOTE: TimesFM 1.0 does NOT support .finetune() — this requires timesfm>=2.0
#          If you have 2.0, use the block below; otherwise skip fine-tuning

# tfm.finetune(
#     train_inputs=[train_data],
#     train_outputs=[test_data],
#     freq=[0],
#     epochs=10,
#     learning_rate=1e-4,
# )

# finetuned_results, _ = tfm.forecast(inputs=[train_data], freq=[0])
# finetuned_preds      = finetuned_results[0]
# mse_finetuned        = mean_squared_error(test_data, finetuned_preds)
# print(f"Fine-tuned TimesFM MSE: {mse_finetuned:.4f}")