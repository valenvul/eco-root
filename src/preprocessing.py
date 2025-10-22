import SimpleITK as sitk
import numpy as np


def volume_denoising(volume):
    """
    Apply speckle-aware denoising to a 3D volume using Perona-Malik anisotropic diffusion.
    
    Args:
        volume: SimpleITK.Image
            Input 3D volume to be denoised.
    
    Returns:
        SimpleITK.Image
            Denoised 3D volume.
    """
    denoised_volume = sitk.CurvatureAnisotropicDiffusion(
        volume,
        timeStep=0.0116,  
        conductanceParameter=0.7,
        numberOfIterations=20
    )
    return denoised_volume

