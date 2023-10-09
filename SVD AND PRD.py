import numpy as np
import pywt  
import csv
from sklearn.metrics import mean_squared_error
import scipy.signal 
import pywt.data


with open('234.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    data_matrix = []
    for row in csv_reader:
        # Check if the row is empty
        if any(row):
            data_matrix.append([float(val) for val in row])

    #for row in csv_reader:
     #   if row:
      #      row[0] = float(row[0])
       #     row[1] = float(row[1])
        #    MLII.append(row[0])
         #   V1.append(row[1])
# Sample ECG data 

ml_ii_signal = [row[0] for row in data_matrix]  # Example MLII signal
v1_signal = [row[1] for row in data_matrix]  # Example V1 signal

# Combine MLII and V1 signals 
ecg_signal = np.array([ml_ii_signal, v1_signal])

# Perform compression (e.g., wavelet compression)
compression_level = 100 # Adjust this according to your needs

compressed_ecg_signal = []
for signal in ecg_signal:
    coeffs = pywt.wavedec(signal, 'db4', level=compression_level)
    compressed_coeffs = [coeffs[0]] + [np.zeros_like(coeff) for coeff in coeffs[1:]]
    compressed_signal = pywt.waverec(compressed_coeffs, 'db4')
    compressed_ecg_signal.append(compressed_signal)

print("Compressed ECG Signal:")
for i, signal in enumerate(compressed_ecg_signal):
    print(f"Signal {i+1}:", signal)

compressed_ecg_array = np.array(compressed_ecg_signal)

# Save the compressed ECG signal as a CSV file
np.savetxt('compressed_ecg_signal.csv', compressed_ecg_array, delimiter=',')


# Calculate PRD
original_signal = ecg_signal
reconstructed_signal = np.array(compressed_ecg_signal)
num_samples = len(original_signal[0])  

sum_squared_diff = np.sum((original_signal - reconstructed_signal) ** 2)
sum_squared_original = np.sum(original_signal ** 2)
prd = (np.sqrt(sum_squared_diff) / np.sqrt(sum_squared_original)) * 100.0

print("PRD (Percentage Root Mean Square Difference): {:.2f}%".format(prd))
# Step 1: Preprocess the signal to make it zero-mean
def remove_mean(signal):
    return signal - np.mean(signal)

# Step 2: Transform the 1D ECG signal into a 2D matrix using block-coding
def transform_to_matrix(ecg_signal, frame_length):
    return scipy.signal.cwt(ecg_signal, scipy.signal.ricker, np.arange(1, frame_length + 1))

# Step 3: Compute the Singular Value Decomposition (SVD) of the data matrix
def compute_svd(data_matrix):
    U, S, VT = np.linalg.svd(data_matrix, full_matrices=False)
    return U, S, VT

def compute_wavelet_coeffs(signal, wavelet_name):
    coeffs = pywt.wavedec(signal, wavelet_name)
    return coeffs

def discard_small_coeffs(coeffs, threshold):
    return [coeff if abs(coeff) >= threshold else 0 for coeff in coeffs]

def run_length_encode(data):
    encoded_data = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            encoded_data.extend([data[i - 1], count])
            count = 1
    encoded_data.extend([data[-1], count])  # Add the last run
    return encoded_data

def calculate_prd(original_signal, reconstructed_signal):
    # Ensure both signals have the same length
    if len(original_signal) != len(reconstructed_signal):
        raise ValueError("Both signals must have the same length")

    # Calculate the PRD
    sum_squared_diff = np.sum((original_signal - reconstructed_signal) ** 2)
    sum_squared_original = np.sum(original_signal ** 2)
    prd = (np.sqrt(sum_squared_diff) / np.sqrt(sum_squared_original)) * 100.0

    return prd


# Step 4: When PRD is predefined
def compress_ecg_signal(ecg_signal, predefined_prd, threshold_level):
    while True:
        # Step 1: Preprocess the signal (make it zero-mean)
        ecg_signal = remove_mean(ecg_signal)
        
        # Step 2: Transform to a 2D matrix
        data_matrix = transform_to_matrix(ecg_signal, frame_length=650000)  # You can adjust the frame length
        
        # Step 3: Compute SVD of the data matrix
        U, S, VT = compute_svd(data_matrix)
        
        # Step 4.1: Consider the largest singular value and truncate
        truncated_S = S[0]
        reconstructed_matrix = np.outer(U[:, 0], truncated_S * VT[0, :])
        reconstructed_ecg_signal = np.sum(reconstructed_matrix, axis=0)
        
        # Step 4.2: Compute wavelet coefficients using biorthogonal wavelets
        # (You'll need a library for wavelet transformations)
        wavelet_coeffs = compute_wavelet_coeffs(reconstructed_ecg_signal, 'biorthogonal_wavelet')
        
        # Step 4.3: Discard coefficients smaller than the predefined threshold
        wavelet_coeffs_thresholded = discard_small_coeffs(wavelet_coeffs, threshold_level)
        
        # Step 4.4: Use a modified run-length coding technique to decrease the number of bits
        compressed_coeffs = run_length_encode(wavelet_coeffs_thresholded)
        
        # Step 4.5: Calculate PRD using equation (12)
        prd = calculate_prd(ecg_signal, reconstructed_ecg_signal)
        
        # Step 4.6: Check if PRD is less than predefined PRD
        if prd < predefined_prd:
            # Calculate the compression ratio using equation (13)
            original_bits = len(ecg_signal) * 16  # Assuming 16 bits per sample
            compressed_bits = len(compressed_coeffs) * 16  # Assuming 16 bits per coefficient
            compression_ratio = original_bits / compressed_bits
            break  # Stop compression

        # Increase the number of considered singular values and continue
        truncated_S = S[:len(truncated_S) + 1]

    return compressed_coeffs, compression_ratio

# Example usage:
original_ecg_signal = ecg_signal  # Replace with your actual ECG data
predefined_prd = 100 # Adjust this according to your desired PRD
threshold_level = 0.1  # Adjust the threshold level

compressed_coeffs, compression_ratio = compress_ecg_signal(original_ecg_signal, predefined_prd, threshold_level)
print("Compression Ratio:", compression_ratio)