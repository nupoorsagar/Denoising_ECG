# final_conclusion_plot.py
# =====================================================
# Final comparison of ECG denoising methods
# =====================================================

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pywt
from sklearn.metrics import mean_squared_error
import math
import pandas as pd

# =====================================================
# Utility functions
# =====================================================

def snr(original, denoised):
    """Signal-to-Noise Ratio"""
    noise = original - denoised
    return 10 * np.log10(np.sum(original**2) / np.sum(noise**2))

def prd(original, denoised):
    """Percent Root Mean Square Difference"""
    return 100 * np.linalg.norm(original - denoised) / np.linalg.norm(original)

def corr_coeff(original, denoised):
    """Correlation Coefficient"""
    return np.corrcoef(original.flatten(), denoised.flatten())[0,1]

# =====================================================
# Load test results from deep learning models
# =====================================================
with open("test_results_DRNN.pkl", "rb") as f:
    X_test_drnn, y_test_drnn, y_pred_drnn = pickle.load(f)

with open("test_results_FCN-DAE.pkl", "rb") as f:
    X_test_fcn, y_test_fcn, y_pred_fcn = pickle.load(f)

# Ensure same reference test set
X_test = X_test_drnn
y_test = y_test_drnn

# Pick one example for visualization
sample_idx = 50  # you can change this index
raw_signal = X_test[sample_idx].flatten()
clean_signal = y_test[sample_idx].flatten()
denoised_drnn = y_pred_drnn[sample_idx].flatten()
denoised_fcn = y_pred_fcn[sample_idx].flatten()

# =====================================================
# Apply baseline filters
# =====================================================

# Savitzky-Golay filter
denoised_savgol = savgol_filter(raw_signal, window_length=21, polyorder=3)

# Wavelet Denoising
def wavelet_denoising(signal, wavelet='db4', level=1):
    coeff = pywt.wavedec(signal, wavelet, mode="per")
    sigma = np.median(np.abs(coeff[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeff[1:] = (pywt.threshold(c, value=uthresh, mode='soft') for c in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode="per")

denoised_wavelet = wavelet_denoising(raw_signal)

# =====================================================
# Plot signals
# =====================================================
plt.figure(figsize=(12,6))
plt.plot(raw_signal, label="Raw signal", linewidth=1)
plt.plot(denoised_savgol, label="Savitzky-Golay filter", linewidth=1)
plt.plot(denoised_wavelet, label="Wavelet Transform", linewidth=1)
plt.plot(denoised_drnn, label="DRNN", linewidth=1)
plt.plot(denoised_fcn, label="FCN-DAE", linewidth=1)
plt.plot(clean_signal, label="Clean (Ground truth)", linewidth=1.2, linestyle='--', color='black')

plt.title("ECG Denoising Comparison (Sample {})".format(sample_idx))
plt.xlabel("Time / Samples")
plt.ylabel("Amplitude [mV]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================================================
# Compute metrics for each method
# =====================================================
methods = {
    "Savitzky-Golay": denoised_savgol,
    "Wavelet": denoised_wavelet,
    "DRNN": denoised_drnn,
    "FCN-DAE": denoised_fcn
}

results = []
for name, denoised in methods.items():
    mse = mean_squared_error(clean_signal, denoised)
    snr_val = snr(clean_signal, denoised)
    prd_val = prd(clean_signal, denoised)
    corr = corr_coeff(clean_signal, denoised)
    results.append([name, mse, snr_val, prd_val, corr])

df_results = pd.DataFrame(results, columns=["Method", "MSE", "SNR (dB)", "PRD (%)", "Correlation"])
print("\nComparison of Denoising Methods:\n")
print(df_results.to_string(index=False))

# ----------------------------------------------------
# Extra Plots with 200 and 500 Samples (with ground truth)
# ----------------------------------------------------

def plot_metrics_samples(denoised_drnn, denoised_fcn, raw_signal, denoised_wavelet, denoised_savgol, clean_signal, num_samples):
    """
    Plots signal and denoising results for given sample size, including clean ground truth.
    """
    end_idx = min(len(raw_signal), num_samples)

    plt.figure(figsize=(14, 6))
    plt.title(f"ECG Denoising Comparison ({num_samples} samples)", fontsize=14)

    plt.plot(raw_signal[:end_idx], label="Raw signal", alpha=0.7)
    plt.plot(denoised_savgol[:end_idx], label="Savitzky-Golay filter")
    plt.plot(denoised_wavelet[:end_idx], label="Wavelet Transform")
    plt.plot(denoised_drnn[:end_idx], label="DRNN")
    plt.plot(denoised_fcn[:end_idx], label="FCN-DAE")
    plt.plot(clean_signal[:end_idx], label="Clean (Ground truth)", linestyle='--', color='black', linewidth=1.2)

    plt.xlabel("Time / Sample", fontsize=12)
    plt.ylabel("Amplitude [mV]", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---- Call for 200 and 500 samples ----
plot_metrics_samples(denoised_drnn, denoised_fcn, raw_signal, denoised_wavelet, denoised_savgol, clean_signal, 200)
plot_metrics_samples(denoised_drnn, denoised_fcn, raw_signal, denoised_wavelet, denoised_savgol, clean_signal, 500)
