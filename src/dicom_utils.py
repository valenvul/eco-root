import pydicom
import numpy as np
import cv2
import os

def  extract_video(dicom_file_path):
    """
    Extracts video from a DICOM file and saves it as an AVI file.

    input: dicom_file_path: str
    output: video_path: str
    """
    
    dicom_data = pydicom.dcmread(dicom_file_path)

    # Check if the DICOM file contains pixel data
    if 'PixelData' not in dicom_data:
        raise ValueError("DICOM file does not contain pixel data.")

    frames = dicom_data.pixel_array

    # Get frame dimensions
    # grayscale
    if len(frames.shape) == 3:
        num_frames, height, width = frames.shape
    #color
    if len(frames.shape) == 4:
        num_frames, height, width, _ = frames.shape
    else:
        raise ValueError("Unsupported frame shape in DICOM file.")

    # Create output directory if it doesn't exist
    output_dir = "data/videos"
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, os.path.splitext(os.path.basename(dicom_file_path))[0] + ".AVI")

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = float(dicom_data.get('CineRate', 30))  # Use 30 fps by default if not present
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=False)

    # Write frames to video 
    for i in range(num_frames):
        frame = frames[i]
        if len(frame.shape) == 3:  # Convert color to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(frame.astype(np.uint8))
    out.release()

    return video_path
