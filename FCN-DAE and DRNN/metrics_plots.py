import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from prettytable import PrettyTable

# Define metrics
def SSD(y, y_pred):
    return np.sum(np.square(y - y_pred), axis=1)

def MAD(y, y_pred):
    return np.max(np.abs(y - y_pred), axis=1)

def RMSE(y, y_pred):
    return np.sqrt(np.mean(np.sum(np.square(y - y_pred), axis=1), axis=1))

# Experiments list (must match names in training step)
dl_experiments = ['DRNN', 'FCN-DAE']

# Load results
with open('test_results_' + dl_experiments[0] + '.pkl', 'rb') as f:
    test_DRNN = pickle.load(f)

with open('test_results_' + dl_experiments[1] + '.pkl', 'rb') as f:
    test_FCN_DAE = pickle.load(f)

print('Calculating metrics ...')

# Unpack DRNN results
[X_test_1, y_test_1, y_pred_1] = test_DRNN
SSD_DRNN = SSD(y_test_1, y_pred_1)
MAD_DRNN = MAD(y_test_1, y_pred_1)
RMSE_DRNN = RMSE(y_test_1, y_pred_1)

# Unpack FCN-DAE results
[X_test_2, y_test_2, y_pred_2] = test_FCN_DAE
SSD_FCN = SSD(y_test_2, y_pred_2)
MAD_FCN = MAD(y_test_2, y_pred_2)
RMSE_FCN = RMSE(y_test_2, y_pred_2)

# Organize results
SSD_all = [SSD_DRNN, SSD_FCN]
MAD_all = [MAD_DRNN, MAD_FCN]
RMSE_all = [RMSE_DRNN, RMSE_FCN]
metrics = ['SSD', 'MAD', 'RMSE']
metric_values = [SSD_all, MAD_all, RMSE_all]

# Pretty results table
def generate_table(metrics, metric_values, Exp_names):
    print("\n")
    tb = PrettyTable()
    tb.field_names = ['Method/Model'] + metrics

    for ind, exp_name in enumerate(Exp_names):
        tb_row = [exp_name]
        for metric in metric_values:
            m_mean = np.mean(metric[ind])
            m_std = np.std(metric[ind])
            tb_row.append(f"{m_mean:.3f} ({m_std:.3f})")
        tb.add_row(tb_row)
    print(tb)

generate_table(metrics, metric_values, dl_experiments)

# Example plot: index 3390
plt.figure(figsize=(10,5))
plt.plot(X_test_1[3390], label="ECG + Noise")
plt.plot(y_test_1[3390], label="ECG")
plt.plot(y_pred_1[3390], label="ECG denoised - DRNN")
plt.plot(y_pred_2[3390], label="ECG denoised - FCN-DAE")
plt.xlabel("Samples")
plt.ylabel("Amplitude [mV]")
plt.legend()
plt.show()

# Random 50 examples
for x in np.random.randint(len(X_test_1), size=50):
    plt.figure(figsize=(10,5))
    plt.plot(X_test_1[x], label="ECG + Noise")
    plt.plot(y_test_1[x], label="ECG")
    plt.plot(y_pred_1[x], label="ECG denoised - DRNN")
    plt.plot(y_pred_2[x], label="ECG denoised - FCN-DAE")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude [mV]")
    plt.legend()
    plt.show()
