import numpy as np
import cv2
import SimpleITK as sitk
from skimage import morphology
import scipy.ndimage as nd
import os.path
import dicom_utils

#class VolumeReconstructor:
#    def __init__(self, pathToVideo, mask):

def skeleton(mask):


    # Lobster from http://www9.informatik.uni-erlangen.de/External/vollib/
    # pvm2raw from http://sourceforge.net/projects/volren/?source=typ_redirect
    # ./pvm2raw Lobster.pvm Lobster.raw
    # reading PVM file
    # found volume with width=301 height=324 depth=56 components=1
    # and edge length 1/1/1.4
    # and data checksum=CFCD4D44

    mask = nd.binary_dilation(mask, structure=morphology.ball(3))
    mask = nd.binary_fill_holes(mask)

    mask = nd.binary_erosion(mask, structure=morphology.ball(3))

    # 3D skeletonization from ITK/SimpleITK
    res = sitk.BinaryThinning(mask)
    # np_res = sitk.GetArrayFromImage(res)

    sitk.WriteImage(res, "res.nii.gz")

# ======================= Configuration ================================================================

# read dicom file
video_path = open_dicom.extract_dicom_video("primera_prueba/imagen_3361207482046.dcm")

# Input video
inputVideo = video_path

# Mask to crop the video before generating the volume
mask =  cv2.imread("mascara.png", cv2.IMREAD_GRAYSCALE)

# Number of samples per second that will be sampled from the video
samplingRate = 3

# Start frame
startFrame = 0

# End Frame
endFrame = -1

# ======================= Pre-processing steps ===============================================

# Equalize?
equalize = False

# ======================= Code ================================================================

# Extract video parameters ...

cap = cv2.VideoCapture(inputVideo)

fps= cap.get(5) # gets video's fps rate

totalFrames = cap.get(7) # gets total number of frames in the video

if endFrame == -1:
    endFrame = int(totalFrames - 1)

stepSampling = int(fps // samplingRate)

framesToSample = list(range(startFrame, endFrame + 1, stepSampling)) # samples the video


print( "FPS: " + str(fps))
print ("Total number of frames: " + str(totalFrames))
print ("Frames to sample " + str(framesToSample))

# Extract mask bounding box
# finds the min and max where thw mask is white
minCoords = np.min(np.argwhere(mask == 255), axis=0)
maxCoords = np.max(np.argwhere(mask == 255), axis=0)

size = maxCoords - minCoords

print ("Mask size" + str(size))

# Create volumetric image container

# array to store the sampled frames cropped according to the mask size
vol = np.zeros(shape=(len(framesToSample),size[0],size[1]), dtype=float)

#sitk.Image(size[0],size[1],len(framesToSample),sitk.sitkUInt8)

i = 0
z = 0
ret = True

while i<totalFrames:
    ret, frame = cap.read()
    print (i)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if i in framesToSample:
#        cv2.imwrite(os.path.join(outDir, str(i) + ".png"), gray)
        if equalize:
            gray = cv2.equalizeHist(gray)

        # crops the sampled frame according to the mask bounding box
        vol[z,:,:] = gray[minCoords[0]:maxCoords[0],minCoords[1]:maxCoords[1]]

        z += 1

        print ("Filling Z: " + str(z))

    i+=1

output = sitk.GetImageFromArray(vol)
output.SetSpacing((1,1,5)) # sets spacing for the voxels, defines real worls size of eaxh voxel

# Post-processing steps:

# Gaussian filter
#output = sitk.GradientMagnitudeRecursiveGaussian(output, sigma=.1)

# Median filter
median = sitk.MedianImageFilter() # removes noise
output = median.Execute(output)



# Otsu

# segmentates the object from the background
otsu_filter = sitk.OtsuThresholdImageFilter() # automatically finds threshold
otsu_filter.SetInsideValue(0) # sets the pixel value for the background as 0
otsu_filter.SetOutsideValue(1) # sets the pixel value for the object as 1
seg = otsu_filter.Execute(output) # applies the filter
# seg = sitk.BinaryFillhole(seg)
# seg = sitk.BinaryMorphologicalOpening(seg)


# Write output

# Output dir where the volume will be stored
if not os.path.exists("tmp2"):
    os.makedirs("tmp2")
outVol = "tmp2/vol.nii.gz"
outSeg = "tmp2/vol_seg.nii.gz"

sitk.WriteImage(output, outVol)
sitk.WriteImage(seg, outSeg)
