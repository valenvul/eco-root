import pydicom
import numpy as np
import cv2
import os

def  extract_video(dicom_file_path):
    """
    input: file_path: str
        path al archivo .dcm
    output: video_path: str
        path al video extraido del dicom
    """
    # Leer el archivo DICOM
    dicom_data = pydicom.dcmread(dicom_file_path)

    # Verificar si el archivo DICOM contiene un video
    if 'PixelData' not in dicom_data:
        raise ValueError("El archivo DICOM no contiene datos de píxeles.")

    # Extraer los frames del DICOM
    frames = dicom_data.pixel_array
    # si es B y N
    if len(frames.shape) == 3:
        num_frames, height, width = frames.shape
    # si es RGB
    if len(frames.shape) == 4:
        num_frames, height, width, _ = frames.shape
    else:
        raise ValueError("Formato de frames no soportado.")

    # Crear carpeta de salida si no existe
    output_dir = "videos"
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path.split('/')[-1]))[0] + ".AVI")

    # Definir el codec y crear el objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = float(dicom_data.get('CineRate', 30))  # Usar 30 fps por defecto si no está presente
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=len(frames.shape) == 4)

    # Escribir los frames en el video
    for i in range(num_frames):
        frame = frames[i]
        if len(frame.shape) == 2:  # Escala de grises
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame.astype(np.uint8))
    out.release()

    return video_path
