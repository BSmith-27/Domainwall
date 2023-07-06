The dynamics model share notebook contains the training procedure for our dynamics model and is set-up to be run on Google Colab for ease of access to our SPM images. We will eventually store these images on a permanent server.

All image data is contained on this Google Drive folder: https://drive.google.com/drive/folders/13rHG2ASMLo5R9Z43drJLCWI5yDqJxDBz?usp=sharing

To run the notebook yourself you will need the SPM images from both datasets: 

Images from Dataset 1 can be found here: https://drive.google.com/file/d/1QHwyzQ0E7kGu_7qxIg0gdog40xqZi5rA/view?usp=drive_link

Images from Dataset 2 can be found here: https://drive.google.com/drive/folders/1X3AtDX2yUmLB3awOVEDW2AoNXpnstFln?usp=sharing

Note that this notebook will replicate the training process, but will not produce exactly identical images as the ones in the manuscript, although they should be similar. 

The pre-trained notebook is nearly the same, but does not contain the training loop and uses pre-trained model weights. To run this notebook, you need to use the trained_weights_5_12.h5 file which contains the dynamics model weights.
