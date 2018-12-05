from gslam import GImage
import numpy as np


image=GImage(20,20,24,0,False)
image_np=np.array(image, copy = False)

print(image_np.shape,image_np.dtype)
