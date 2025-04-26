import numpy as np
from scipy.signal import firwin, filtfilt
from scipy.io import loadmat, savemat
from pykalman import KalmanFilter
import pywt
import os

def read_subject_file(file_path):
    data = loadmat(file_path)
    keys = list(data.keys())
    return data, keys


def apply_fir_filter(data, fs, lowcut=1.0, highcut=45.0, order=101):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    fir_coeff = firwin(order, [low, high], pass_zero=False)
    filtered_data = filtfilt(fir_coeff, 1.0, data)
    return filtered_data

def remove_eye_blink_artifacts(data):
    coeffs = pywt.wavedec(data, 'db4', level=5)
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(2 * np.log(len(data)))
    coeffs = list(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs))
    filtered_data = pywt.waverec(coeffs, 'db4')

    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    filtered_state_means, _ = kf.smooth(filtered_data[:, np.newaxis])
    filtered_state_means = filtered_state_means.ravel()

    # Signal reconstruction and matching the size of the output signal to the input signal
    if len(filtered_state_means) > len(data):
        filtered_state_means = filtered_state_means[:len(data)]
    elif len(filtered_state_means) < len(data):
        filtered_state_means = np.pad(filtered_state_means, (0, len(data) - len(filtered_state_means)), 'constant')

    return filtered_state_means


directory_path = "/content/gdrive/MyDrive/EEG_128channels_resting_lanzhou_2015"
file_list = os.listdir(directory_path)

output_directory = "/content/gdrive/MyDrive/Preprocessing_EEG_128channels"
os.makedirs(output_directory, exist_ok=True)

# Sampling frequency (Hz)
fs = 250

for file_name in file_list:
    file_path = os.path.join(directory_path, file_name)
    data, keys = read_subject_file(file_path)

    key = keys[3]
    data = data[key]

    for channel in range(128):
        data[channel,:] = apply_fir_filter(data[channel,:], fs)
        data[channel,:] = remove_eye_blink_artifacts(data[channel,:])

    # Save cleaned_data
    name = f"{file_name.split('.')[0]}_Preprocessing.mat"
    cleaned_file_path = os.path.join(output_directory, name)
    savemat(cleaned_file_path, {key: data})
