# ------------------------------
# Package Installation (Run once in VS Code terminal)
# ------------------------------
# You can run these commands directly in VS Code terminal (not in the script)
# pip uninstall tensorflow tensorflow-gpu seaborn --yes
# pip install h5py==2.10.0 tensorflow-gpu==1.14.0 keras==2.2.5 numpy==1.19.2 seaborn==0.10.1 scipy==1.5.2 scikit-learn==0.23.2 prettytable==1.0.1 wfdb==3.1.1

# ------------------------------
# Imports
# ------------------------------
import os
import shutil
import glob
import numpy as np
from scipy.signal import resample_poly
import wfdb
import math
import _pickle as pickle
import matplotlib.pyplot as plt

# ------------------------------
# Paths to datasets
# ------------------------------
qt_db_path = r"C:\Users\APOORVA\OneDrive\Desktop\CAPSTONE DATASET\qt-database-1.0.0"
nstdb_path = r"C:\Users\APOORVA\OneDrive\Desktop\CAPSTONE DATASET\mit-bih-noise-stress-test-database-1.0.0"
data_dir = r"C:\Users\APOORVA\OneDrive\Desktop\CAPSTONE DATASET\data"

# Create a 'data' directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Copy datasets into data directory (skip if already present)
for src_path in [qt_db_path, nstdb_path]:
    dst_path = os.path.join(data_dir, os.path.basename(src_path))
    if not os.path.exists(dst_path):
        shutil.copytree(src_path, dst_path)

print("Data setup: Done")

# ------------------------------
# Verify files
# ------------------------------
print("Files in data directory:")
for folder in os.listdir(data_dir):
    print(folder)

# ------------------------------
# QT Database Processing (Windows / VS Code compatible)
# ------------------------------
import os
import glob
import math
from scipy.signal import resample_poly
import wfdb
import _pickle as pickle
import numpy as np

# Paths
QTpath = os.path.join(data_dir, "qt-database-1.0.0")
newFs = 360  # Desired sampling frequency

# Get all .dat files in QT database
dat_files = glob.glob(os.path.join(QTpath, "*.dat"))

# Dictionary to store processed signals and beats
QTDatabaseSignals = dict()

for dat_file in dat_files:
    # Remove extension for WFDB
    record_name = os.path.splitext(dat_file)[0]  # <-- No '.dat'
    register_name = os.path.basename(record_name)

    # Read signal and header
    signal, fields = wfdb.rdsamp(record_name)  # WFDB automatically finds .dat + .hea
    n_samples = len(signal)

    # Read annotations
    ann = wfdb.rdann(record_name, 'pu1')  # Same: no '.dat'
    anntype = ann.symbol
    annSamples = ann.sample

    # Get P wave start positions
    Anntype = np.array(anntype)
    Pidx = annSamples[Anntype == 'p']
    Sidx = annSamples[Anntype == '(']
    Ridx = annSamples[Anntype == 'N']

    # Compute P wave start relative to preceding S wave
    ind = np.zeros(len(Pidx), dtype=int)
    for j in range(len(Pidx)):
        arr = np.where(Pidx[j] > Sidx)[0]
        ind[j] = arr[-1]
    Pstart = Sidx[ind]

    # Shift 40ms before P wave start
    Pstart = Pstart - int(0.04 * fields['fs'])

    # Extract first channel
    auxSig = signal[:n_samples, 0]

    # Separate beats and remove outliers
    beats = []
    for k in range(len(Pstart) - 1):
        remove = (Ridx > Pstart[k]) & (Ridx < Pstart[k + 1])
        if np.sum(remove) < 2:
            beats.append(auxSig[Pstart[k]:Pstart[k + 1]])

    # Process each beat
    beatsRe = []
    for beat in beats:
        L = math.ceil(len(beat) * newFs / fields['fs'])
        # Pad data to avoid edge effects
        normBeat = list(reversed(beat)) + list(beat) + list(reversed(beat))
        # Resample
        res = resample_poly(normBeat, newFs, fields['fs'])
        res = res[L - 1: 2 * L - 1]
        beatsRe.append(res)

    # Store beats for this signal
    QTDatabaseSignals[register_name] = beatsRe

# Save processed QT database
pickle_file_path = os.path.join(data_dir, "QTDatabase.pkl")
with open(pickle_file_path, 'wb') as output:
    pickle.dump(QTDatabaseSignals, output)

print("=========================================================")
print(f"MIT QT database saved as pickle file at: {pickle_file_path}")

# ------------------------------
# MIT-BIH Noise Stress Test Database (NSTDB) Processing
# ------------------------------
import os
import wfdb
import numpy as np
import _pickle as pickle

# Base path
NSTDB_base = os.path.join(data_dir, "mit-bih-noise-stress-test-database-1.0.0")

# Folders to process
folders = ['bw', 'ma', 'em']
output_names = ['NoiseBWL', 'NoiseMA', 'NoiseEM']

for folder, out_name in zip(folders, output_names):
    NSTDBPath = os.path.join(NSTDB_base, folder)
    
    # WFDB reads record using full path without extension
    signals, fields = wfdb.rdsamp(NSTDBPath)
    
    # Print header info
    print(f"Fields for {folder}:")
    for key in fields:
        print(key, fields[key])
    
    # Save as .npy
    npy_path = os.path.join(data_dir, f"{out_name}.npy")
    np.save(npy_path, signals)
    
    # Save as pickle
    pkl_path = os.path.join(data_dir, f"{out_name}.pkl")
    with open(pkl_path, 'wb') as output:
        pickle.dump(signals, output)
    
    print('=========================================================')
    print(f'MIT BIH NSTDB ({folder}) saved as pickle and npy')

# ------------------------------
# Combine QTDatabase with NSTDB Noise (Windows/VS Code compatible)
# ------------------------------
import os
import numpy as np
import _pickle as pickle

print('Getting the Data ready ... ')

# Reproducibility
seed = 1234
np.random.seed(seed=seed)

# Load QT Database
with open(os.path.join(data_dir, 'QTDatabase.pkl'), 'rb') as input_file:
    qtdb = pickle.load(input_file)  # dict {register_name: beats_list}

# Load NSTDB
with open(os.path.join(data_dir, 'NoiseBWL.pkl'), 'rb') as input_file:
    nstdbwl = pickle.load(input_file)

with open(os.path.join(data_dir, 'NoiseMA.pkl'), 'rb') as input_file:
    nstdma = pickle.load(input_file)

with open(os.path.join(data_dir, 'NoiseEM.pkl'), 'rb') as input_file:
    nstdem = pickle.load(input_file)

# Separate noise channels
noise_channel1 = nstdbwl[:, 0]
noise_channel2 = nstdbwl[:, 1]
noise_channel3 = nstdma[:, 0]
noise_channel4 = nstdma[:, 1]
noise_channel5 = nstdem[:, 0]
noise_channel6 = nstdem[:, 1]

# Split into train/test
def split_noise(ch1, ch2, ratio=0.13):
    split_idx1 = int(ch1.shape[0] * ratio)
    split_idx2 = int(ch2.shape[0] * ratio)
    test = np.concatenate((ch1[:split_idx1], ch2[:split_idx2]))
    train = np.concatenate((ch1[split_idx1:], ch2[split_idx2:]))
    return train, test

noise_train_bwt, noise_test_bwt = split_noise(noise_channel1, noise_channel2)
noise_train_ma, noise_test_ma = split_noise(noise_channel3, noise_channel4)
noise_train_em, noise_test_em = split_noise(noise_channel5, noise_channel6)

# QTDatabase: split beats into train/test
beats_train = []
beats_test = []

test_set = [
    'sel123', 'sel233', 'sel307', 'sel820', 'sel853',
    'sel16420', 'sel16795', 'sele0106', 'sele0121',
    'sel32', 'sel49', 'sel14046', 'sel15814'
]

samples = 512
skip_beats = 0
init_padding = 16

for signal_name, beat_list in qtdb.items():
    for b in beat_list:
        if len(b) > (samples - init_padding):
            skip_beats += 1
            continue
        b_np = np.zeros(samples)
        b_sq = np.array(b)
        b_np[init_padding:len(b_sq) + init_padding] = b_sq - (b_sq[0] + b_sq[-1]) / 2
        if signal_name in test_set:
            beats_test.append(b_np)
        else:
            beats_train.append(b_np)

# Add noise to train and test
def add_noise(beats, noise_train, samples):
    sn_list = []
    noise_index = 0
    rnd_values = np.random.randint(low=20, high=200, size=len(beats)) / 100

    for i in range(len(beats)):
        beat_max = np.max(beats[i]) - np.min(beats[i])

        noise_bwt = noise_train[noise_index:noise_index + samples]
        noise_ma = noise_train[noise_index:noise_index + samples]
        noise_em = noise_train[noise_index:noise_index + samples]

        # Normalize noise amplitude
        alpha = rnd_values[i] / (np.ptp(noise_bwt) / beat_max)
        beta = rnd_values[i] / (np.ptp(noise_ma) / beat_max)
        gamma = rnd_values[i] / (np.ptp(noise_em) / beat_max)

        signal_noise = beats[i] + alpha/3 * noise_bwt + beta/3 * noise_ma + gamma/3 * noise_em
        sn_list.append(signal_noise)

        noise_index += samples
        if noise_index > len(noise_train) - samples:
            noise_index = 0
    return sn_list

sn_train = add_noise(beats_train, noise_train_bwt, samples)
sn_test = add_noise(beats_test, noise_test_bwt, samples)

# Convert to numpy arrays and expand dims
X_train = np.expand_dims(np.array(sn_train), axis=2)
y_train = np.expand_dims(np.array(beats_train), axis=2)

X_test = np.expand_dims(np.array(sn_test), axis=2)
y_test = np.expand_dims(np.array(beats_test), axis=2)

Dataset = [X_train, y_train, X_test, y_test]

print('Dataset ready to use.')
print("DATASET SHAPE")
print(f'Training: {X_train.shape}')
print(f'Testing: {X_test.shape}')
# ------------------------------
# Deep Learning Models for ECG Denoising
# ------------------------------
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv1D, BatchNormalization, Lambda, LSTM, Conv2DTranspose
import keras.backend as K

# 1D Convolution Transpose function
def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, activation='relu', padding='same'):
    """
    Custom 1D transposed convolution using 2D Conv2DTranspose.
    Expands dims -> applies Conv2DTranspose -> squeezes dims back.
    """
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters,
                        kernel_size=(kernel_size, 1),
                        activation=activation,
                        strides=(strides, 1),
                        padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

# ------------------------------
# FCN Denoising Autoencoder
# ------------------------------
def FCN_DAE():
    """
    Fully Convolutional Denoising Autoencoder (FCN-DAE)
    Reference: Chiang et al., IEEE Access, 2019
    """
    input_shape = (512, 1)
    input_tensor = Input(shape=input_shape)

    # Encoder
    x = Conv1D(40, kernel_size=16, activation='elu', strides=2, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Conv1D(20, kernel_size=16, activation='elu', strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(20, kernel_size=16, activation='elu', strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(20, kernel_size=16, activation='elu', strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(40, kernel_size=16, activation='elu', strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1, kernel_size=16, activation='elu', strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Decoder
    x = Conv1DTranspose(x, filters=1, kernel_size=16, activation='elu', strides=1)
    x = BatchNormalization()(x)
    x = Conv1DTranspose(x, filters=40, kernel_size=16, activation='elu', strides=2)
    x = BatchNormalization()(x)
    x = Conv1DTranspose(x, filters=20, kernel_size=16, activation='elu', strides=2)
    x = BatchNormalization()(x)
    x = Conv1DTranspose(x, filters=20, kernel_size=16, activation='elu', strides=2)
    x = BatchNormalization()(x)
    x = Conv1DTranspose(x, filters=20, kernel_size=16, activation='elu', strides=2)
    x = BatchNormalization()(x)
    x = Conv1DTranspose(x, filters=40, kernel_size=16, activation='elu', strides=2)
    x = BatchNormalization()(x)
    predictions = Conv1DTranspose(x, filters=1, kernel_size=16, activation='linear', strides=1)

    model = Model(inputs=input_tensor, outputs=predictions)
    return model

# ------------------------------
# Deep Recurrent Denoising Network (DRNN)
# ------------------------------
def DRRN_denoising():
    """
    Deep Recurrent Neural Network for ECG Denoising
    Reference: Antczak, 2018
    """
    model = Sequential()
    model.add(LSTM(64, input_shape=(512, 1), return_sequences=True))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

# ------------------------------
# Training and Testing Functions
# ------------------------------
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split

# ------------------------------
# Loss function used in FCN_DAE
# ------------------------------
def ssd_loss(y_true, y_pred):
    """
    Sum of squared differences over the second-to-last axis.
    Used for FCN_DAE training.
    """
    return K.sum(K.square(y_pred - y_true), axis=-2)

# ------------------------------
# Train Deep Learning Model
# ------------------------------
def train_dl(Dataset, experiment):
    """
    Train the specified model (FCN-DAE or DRNN) on Dataset.
    Dataset = [X_train, y_train, X_test, y_test]
    """
    print(f'Deep Learning pipeline: Training the model for exp {experiment}')

    X_train, y_train, X_test, y_test = Dataset

    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.3, shuffle=True, random_state=1
    )

    # Select model
    if experiment == 'FCN-DAE':
        model = FCN_DAE()
        model_label = 'FCN_DAE'
        criterion = ssd_loss
    elif experiment == 'DRNN':
        model = DRRN_denoising()
        model_label = 'DRNN'
        criterion = keras.losses.mean_squared_error
    else:
        raise ValueError("Experiment must be 'FCN-DAE' or 'DRNN'")

    print(f'\n{model_label}\n')
    model.summary()

    # Training hyperparameters
    epochs = 100
    batch_size = 128
    lr = 1e-3
    minimum_lr = 1e-10

    model.compile(
        loss=criterion,
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[keras.losses.mean_squared_error, ssd_loss]
    )

    # Callbacks
    model_filepath = f'{model_label}_weights.best.hdf5'

    checkpoint = ModelCheckpoint(
        model_filepath,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode='min',
        save_weights_only=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        min_delta=0.05,
        mode='min',
        patience=2,
        min_lr=minimum_lr,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=0.05,
        mode='min',
        patience=10,
        verbose=1
    )

    # Train model
    model.fit(
        x=X_train, y=y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint, reduce_lr, early_stop]
    )

    # Clear session
    K.clear_session()

# ------------------------------
# Test Deep Learning Model
# ------------------------------
def test_dl(Dataset, experiment):
    """
    Test the trained model and return predictions.
    Dataset = [X_train, y_train, X_test, y_test]
    """
    print('Deep Learning pipeline: Testing the model')

    _, _, X_test, y_test = Dataset
    batch_size = 32

    # Select model
    if experiment == 'FCN-DAE':
        model = FCN_DAE()
        model_label = 'FCN_DAE'
        criterion = ssd_loss
    elif experiment == 'DRNN':
        model = DRRN_denoising()
        model_label = 'DRNN'
        criterion = keras.losses.mean_squared_error
    else:
        raise ValueError("Experiment must be 'FCN-DAE' or 'DRNN'")

    print(f'\n{model_label}\n')
    model.summary()

    model.compile(
        loss=criterion,
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        metrics=[keras.losses.mean_squared_error, ssd_loss]
    )

    # Load best weights
    model_filepath = f'{model_label}_weights.best.hdf5'
    model.load_weights(model_filepath)

    # Predict
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)

    # Clear session
    K.clear_session()

    return X_test, y_test, y_pred

# ------------------------------
# Deep Learning Experiments
# ------------------------------

dl_experiments = ['DRNN', 'FCN-DAE']

for experiment in dl_experiments:
    print(f"\nStarting experiment: {experiment}\n")
    
    # Train the model
    train_dl(Dataset, experiment)

    # Test the model
    X_test, y_test, y_pred = test_dl(Dataset, experiment)

    # Save test results
    test_results = [X_test, y_test, y_pred]
    results_file = f'test_results_{experiment}.pkl'
    
    with open(results_file, 'wb') as output:
        pickle.dump(test_results, output)
    
    print(f'Results from experiment {experiment} saved as {results_file}')
