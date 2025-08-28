import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Step 1 & 2: Load MIT-BIH ECG record
record_path = r'C:\Users\Nimisha Majgawali\Desktop\ecg ft\mit-bih-arrhythmia-database-1.0.0\100'
record = wfdb.rdrecord(record_path)
ecg_signal = record.p_signal[:, 0]  # Extract first ECG lead (usually MLII)
sample_rate = record.fs

# Step 3: Visualize raw ECG signal (first 2000 samples)
plt.figure(figsize=(12, 4))
plt.plot(ecg_signal[0:2000])
plt.title('Raw ECG Signal')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

# Step 4: Normalize the ECG signal (scale between 0 and 1)
ecg_signal_norm = (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))

# Visualize normalized ECG signal (first 2000 samples)
plt.figure(figsize=(12, 4))
plt.plot(ecg_signal_norm[0:2000])
plt.title('Normalized ECG Signal')
plt.xlabel('Sample Number')
plt.ylabel('Normalized Amplitude')
plt.grid()
plt.show()

# Step 5: Apply Fourier Transform on normalized ECG signal
N = len(ecg_signal_norm)
yf = fft(ecg_signal_norm)                  # FFT coefficients
xf = fftfreq(N, 1 / sample_rate)           # Frequency bins (Hz)

# Step 6: Visualize frequency spectrum (positive frequencies)
plt.figure(figsize=(12, 4))
plt.plot(xf[:N // 2], np.abs(yf[:N // 2]))
plt.title('Frequency Spectrum of Normalized ECG Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

# Step 7 & 8: Interpret frequency components (done through visualization)
# Typical noise frequencies to examine:
# - Around 50 or 60 Hz (powerline interference)
# - Low frequencies (<0.5 Hz) for baseline wander
# - High frequency for muscle artifacts

# (Optional) You can add later steps for frequency domain filtering and inverse FFT reconstruction
