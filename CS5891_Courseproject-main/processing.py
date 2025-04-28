import nibabel as nib
from nilearn import image
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# Directory containing the NIfTI files
input_dir = '..'
output_dir = '..'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all .nii files in the input directory
file_list = [f for f in os.listdir(input_dir) if f.endswith('.nii')]

# Loop through each file in the directory
for i, filename in enumerate(file_list):
    file_path = os.path.join(input_dir, filename)
    print(f"Processing file {i + 1}/{len(file_list)}: {filename}")
    
    # Step 1: Load the NIfTI file
    print("Starting to load the NIfTI file...")
    start_time = time.time()
    img = nib.load(file_path)
    data = img.get_fdata()
    print(f"File loaded in {time.time() - start_time:.2f} seconds.")
    print(f"Data shape: {data.shape}")

    # Step 2: Spatial Normalization to MNI Space
    print("Starting spatial normalization to MNI space...")
    start_time = time.time()
    target_affine = np.diag([3, 3, 3])  # 3mm voxel size
    normalized_img = image.resample_img(img, target_affine=target_affine, interpolation='continuous')
    print(f"Spatial normalization completed in {time.time() - start_time:.2f} seconds.")

    # Step 3: Smoothing
    print("Starting spatial smoothing...")
    start_time = time.time()
    smoothed_img = image.smooth_img(normalized_img, fwhm=4)  # 4mm Gaussian smoothing
    print(f"Spatial smoothing completed in {time.time() - start_time:.2f} seconds.")

    # Save the processed image
    output_path = os.path.join(output_dir, f'processed_{filename}')
    nib.save(smoothed_img, output_path)
    print(f"Processed data saved to {output_path}\n")

print("All files processed successfully.")

