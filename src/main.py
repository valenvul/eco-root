import os
from dicom_utils import extract_video

# ------------------- SET UP ------------------
# Extract videos from DICOM files

input_dir = 'primera_prueba'

for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    if os.path.isfile(input_path):
        try:
            video_path = extract_video(input_path)
            print(f"Video extraído y guardado en: {video_path}")
        except Exception as e:
            print(f"No se pudo extraer video de {input_path}: {e}")


# -------------------- VOLUME RECONSTRUCTION ------------------
# Generate volume reconstruction for each view

