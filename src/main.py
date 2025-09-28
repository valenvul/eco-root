import os
import SimpleITK as sitk
from dicom_utils import extract_video
from NewVolumeReconstructor import VolumeReconstructor

# ------------------- SET UP ------------------
# Extract videos from DICOM files

input_dir = 'data/ultrasound'

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
            volume = reconstructor.extract_and_crop_frames()
            
            # Generate output filename
            base_name = os.path.splitext(filename)[0]  # Remove .AVI extension
            output_filename = f"volume_{base_name}.nii.gz"
            output_path = os.path.join(volumes_output_dir, output_filename)
            
            # Save volume
            sitk.WriteImage(volume, output_path)
            print(f"Volume saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
