from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import numpy as np
import os

# AAL
aal_atlas = datasets.fetch_atlas_aal()
masker = NiftiLabelsMasker(labels_img=aal_atlas.maps, standardize=True)


input_dir = '..'
output_dir = '..'


os.makedirs(output_dir, exist_ok=True)

file_list = [f for f in os.listdir(input_dir) if f.startswith('processed_') and f.endswith('.nii')]

# 对每个参与者的数据进行ROI时间序列提取
for i, filename in enumerate(file_list):
    file_path = os.path.join(input_dir, filename)
    print(f"Extracting ROI time series for file {i + 1}/{len(file_list)}: {filename}")

    # 加载处理后的fMRI数据
    img = nib.load(file_path)
    
    # 提取时间序列
    time_series = masker.fit_transform(img)
    
    output_file = os.path.join(output_dir, f'timeseries_{filename}.csv')
    np.savetxt(output_file, time_series, delimiter=",")
    print(f"Time series for {filename} saved to {output_file}\n")

print("ROI time series extraction completed for all files.")
