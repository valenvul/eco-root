
import os
import numpy as np
import SimpleITK as sitk

class VolumeRegistrator:
    def __init__(self, volumes_dir, output_dir="data/results", angle_per_side=60, num_volumes=6):
        """
        Initialize VolumeRegistrator for fusing multiple volumes in a hexagonal pattern.
        
        Args:
            volumes_dir: str
                Directory containing the input volume files (.nii.gz)
            output_dir: str
                Directory where the fused volume will be saved (default: "data/results")
            angle_per_side: float
                Rotation angle between adjacent volumes in degrees (default: 60)
            num_volumes: int
                Expected number of volumes for fusion (default: 6)
        """
        self.volumes_dir = volumes_dir
        self.output_dir = output_dir
        self.angle_per_side = angle_per_side
        self.num_volumes = num_volumes
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.volumes = []
        self.filenames = []
        self.names = []
        
        self.reference_volume = None
        self.fused_volume = None

    def load_volumes(self):
        """
        Load all volume files from the volumes directory.
        
        Returns:
            bool: True if volumes loaded successfully, False otherwise
        """
        try:
            self.filenames = sorted([os.path.join(self.volumes_dir, f)
                                   for f in os.listdir(self.volumes_dir) if f.endswith(".nii.gz")])
            self.volumes = [sitk.ReadImage(fn, sitk.sitkFloat32) for fn in self.filenames]
            self.names = [os.path.basename(fn).replace('.nii.gz','') for fn in self.filenames]
            
            if len(self.volumes) != self.num_volumes:
                raise ValueError(f"Expected exactly {self.num_volumes} volumes, but found {len(self.volumes)}.")
                
            print(f"Successfully loaded {len(self.volumes)} volumes from {self.volumes_dir}")
            return True
            
        except Exception as e:
            print(f"Error loading volumes: {e}")
            return False

    def create_reference_volume(self):
        """
        Create a larger reference volume for hexagonal fusion based on the first volume.
        
        Returns:
            SimpleITK.Image: Reference volume for registration
        """
        if not self.volumes:
            raise RuntimeError("No volumes loaded. Call load_volumes() first.")
            
        fixed = self.volumes[0]
        size = np.array(fixed.GetSize())
        spacing = np.array(fixed.GetSpacing())

        # get the center point of the view
        center = fixed.TransformContinuousIndexToPhysicalPoint(size / 2)
        
        # Create a larger volume for the hexagon
        max_dim = max(size * spacing) * 1.5 
        new_spacing = spacing
        new_size = [int(max_dim / s) for s in new_spacing]
        
        reference = sitk.Image(new_size, sitk.sitkFloat32)
        reference.SetSpacing(new_spacing)
        reference.SetOrigin(np.array(center) - np.array(reference.GetSpacing()) * np.array(reference.GetSize()) / 2)
        reference.SetDirection(fixed.GetDirection())
        
        self.reference_volume = reference
        return reference, center

    def register_and_fuse_volumes(self):
        """
        Register and fuse all loaded volumes into a single hexagonal volume.
        
        Returns:
            SimpleITK.Image: Fused volume
        """
        if not self.volumes:
            raise RuntimeError("No volumes loaded. Call load_volumes() first.")
            
        reference, center = self.create_reference_volume()
        
        fused_array = np.zeros(sitk.GetArrayFromImage(reference).shape, dtype=np.float32)
        count_array = np.zeros_like(fused_array, dtype=np.float32)
        
        for i, vol in enumerate(self.volumes):
            print(f"Processing volume {i} ({self.names[i]})...")
            # Create transformation
            t = sitk.Euler3DTransform()
            t.SetCenter(center)  
            t.SetRotation(0, 0, np.deg2rad(i * self.angle_per_side))  # rotate around z-axis
            
            resampled = sitk.Resample(vol, reference, t, sitk.sitkLinear, 0.0, vol.GetPixelID())
            arr = sitk.GetArrayFromImage(resampled)
            
            mask = arr > 0
            fused_array[mask] += arr[mask]
            count_array[mask] += 1
        
        count_array[count_array == 0] = 1  # avoid division by zero
        fused_array /= count_array
        
        fused_img = sitk.GetImageFromArray(fused_array)
        fused_img.CopyInformation(reference)
        
        self.fused_volume = fused_img
        return fused_img

    def save_fused_volume(self, output_filename="final_volume.nii.gz"):
        """
        Save the fused volume to disk.
        
        Args:
            output_filename: str
                Name of the output file (default: "final_volume.nii.gz")
                
        Returns:
            str: Full path to the saved file
        """
        if self.fused_volume is None:
            raise RuntimeError("No fused volume available. Call register_and_fuse_volumes() first.")
            
        output_path = os.path.join(self.output_dir, output_filename)
        sitk.WriteImage(self.fused_volume, output_path)
        print(f"Fused hexagonal volume saved to: {output_path}")
        return output_path

    def process_volumes(self, output_filename="final_volume.nii.gz"):
        """
        Complete pipeline: load volumes, register and fuse them, then save the result.
        
        Args:
            output_filename: str
                Name of the output file (default: "final_volume.nii.gz")
                
        Returns:
            str: Full path to the saved file, or None if processing failed
        """
        try:
            if not self.load_volumes():
                return None                
            self.register_and_fuse_volumes()
            return self.save_fused_volume(output_filename)
            
        except Exception as e:
            print(f"Error in processing pipeline: {e}")
            return None

