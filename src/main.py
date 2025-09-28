import os
import SimpleITK as sitk
import numpy as np
from dicom_utils import extract_video, rename_dicom_files_sequentially
from volumeReconstructor import VolumeReconstructor
from itertools import product

# ------------------- SET UP ------------------
# Extract videos from DICOM files

print("\n" + "="*50)
print("VIDEO EXTRACTION")
print("="*50)


input_dir = 'data/ultrasound'

rename_dicom_files_sequentially(input_dir)

for filename in os.listdir(input_dir):
        
    input_path = os.path.join(input_dir, filename)
    if os.path.isfile(input_path):
        try:
            video_path = extract_video(input_path)
            print(f"Video extracted in: {video_path}")
        except Exception as e:
            print(f"Could not extract video from {input_path}: {e}")

extracted_videos_dir = 'data/videos'

# -------------------- VOLUME RECONSTRUCTION ------------------
# Generate volume reconstruction for each view
# stacks each frame into a 3D volume where each cut is a z plane. 
# It also crops each frame so only the ultrasound region is kept. 

print("\n" + "="*50)
print("VOLUME RECONSTRUCTION")
print("="*50)


volumes_output_dir = 'data/volumes'
if not os.path.exists(volumes_output_dir):
    os.makedirs(volumes_output_dir)

mask_path = "data/crop_masks/mascara.png" 

for filename in os.listdir(extracted_videos_dir):
    video_path = os.path.join(extracted_videos_dir, filename)
    if os.path.isfile(video_path) and filename.endswith('.AVI'):
        try:
            print(f"\nProcessing video: {filename}")
            
            # Create reconstructor and extract volume
            reconstructor = VolumeReconstructor(video_path, mask_path=mask_path)
            volume = reconstructor.create_volume()
            
            # Generate output filename
            base_name = os.path.splitext(filename)[0]  # Remove .AVI extension
            output_filename = f"volume_{base_name}.nii.gz"
            output_path = os.path.join(volumes_output_dir, output_filename)
            
            # Save volume
            sitk.WriteImage(volume, output_path)
            print(f"Volume saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# -------------------- REGISTRATION ------------------
# Perform volume registration and fusion

print("\n" + "="*50)
print("VOLUME REGISTRATION AND FUSION")
print("="*50)

volumes_output_dir = "data/volumes"  
output_path = "data/results/final_volume.nii.gz"
angle_per_side = 60 
if not os.path.exists('data/results'):
    os.makedirs('data/results')

filenames = sorted([os.path.join(volumes_output_dir, f)
                    for f in os.listdir(volumes_output_dir) if f.endswith(".nii.gz")])
volumes = [sitk.ReadImage(fn, sitk.sitkFloat32) for fn in filenames]
names = [os.path.basename(fn).replace('.nii.gz','') for fn in filenames]

if len(volumes) != 6:
    raise ValueError("Expected exactly 6 volumes for a hexagon.")

# Use the geometric center of the first volume as the pot center
fixed = volumes[0]
size = np.array(fixed.GetSize())
spacing = np.array(fixed.GetSpacing())
center = fixed.TransformContinuousIndexToPhysicalPoint(size / 2)

# Create a larger volume for the hexagon
max_dim = max(size * spacing) * 1.5 
new_spacing = spacing
new_size = [int(max_dim / s) for s in new_spacing]

reference = sitk.Image(new_size, sitk.sitkFloat32)
reference.SetSpacing(new_spacing)
reference.SetOrigin(np.array(center) - np.array(reference.GetSpacing()) * np.array(reference.GetSize()) / 2)
reference.SetDirection(fixed.GetDirection())

fused_array = np.zeros(sitk.GetArrayFromImage(reference).shape, dtype=np.float32)
count_array = np.zeros_like(fused_array, dtype=np.float32)

for i, vol in enumerate(volumes):
    print(f"Processing volume {i} ({names[i]})...")
    t = sitk.Euler3DTransform()
    t.SetCenter(center)  # rotate around pot center
    t.SetRotation(0, 0, np.deg2rad(i * angle_per_side))  # rotate around z-axis

    resampled = sitk.Resample(vol, reference, t, sitk.sitkLinear, 0.0, vol.GetPixelID())
    arr = sitk.GetArrayFromImage(resampled)
    mask = arr > 0
    fused_array[mask] += arr[mask]
    count_array[mask] += 1

# Normalize fused volume
count_array[count_array == 0] = 1  # avoid division by zero
fused_array /= count_array

fused_img = sitk.GetImageFromArray(fused_array)
fused_img.CopyInformation(reference)

# Save fused volume
sitk.WriteImage(fused_img, output_path)
print(f"Fused hexagonal volume saved to: {output_path}")

# -------------------- SEGMENTATION ------------------
print("\n" + "="*50)
print("VOLUME SEGMENTATION")
print("="*50)

print("Applying segmentation to the final fused volume...")

# Use the fused volume for segmentation
volume = fused_img

# Gaussian filter (optional - uncomment if needed for noise reduction)
# volume = sitk.GradientMagnitudeRecursiveGaussian(volume, sigma=0.1)

# Median filter - removes noise
print("Applying median filter...")
median = sitk.MedianImageFilter()
filtered_volume = median.Execute(volume)

# Otsu thresholding - segments the object from the background
print("Applying Otsu thresholding...")
otsu_filter = sitk.OtsuThresholdImageFilter()
otsu_filter.SetInsideValue(0)  # sets the pixel value for the background as 0
otsu_filter.SetOutsideValue(1)  # sets the pixel value for the object as 1
seg = otsu_filter.Execute(filtered_volume)

# Optional morphological operations (uncomment if needed)
# seg = sitk.BinaryFillhole(seg)
# seg = sitk.BinaryMorphologicalOpening(seg)

# Write output files
print("Saving segmented volumes...")
outVol = "data/results/final_volume.nii.gz"
outSeg = "data/results/final_volume_segmentation.nii.gz"

sitk.WriteImage(filtered_volume, outVol)
sitk.WriteImage(seg, outSeg)

print(f"Filtered final volume saved to: {outVol}")
print(f"Segmented final volume saved to: {outSeg}")

            
