# SR-SRGAN
Tensorflow implementation of 'Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network'.
We only implement the x4 single image super-resolution. 
We are not intended to reproduce the performace reported in [1], just implement it for fun. 
Therefore, we have not verified the performance of the trained model, and compared it with performance in [1]. 
This implementation is based on the Tensorlayer implementation, but our implementation of SRGAN is only depend on Tensorflow and Slim.

Windows7 / Ubuntu 14.04 + CUDA8.0 + CUDNN 5.1 + Tensorflow 1.4

run main.py

Before run, please prepare the training dataset, trained EPSCN and VGG19. 
We have tried ImageNet and DIV2K. 
For ImageNet data, one can follow the description in [1] to prepare the dataset. Dirty image should be carefully handled. 
For DIV2K, one just use the training set. Then, modify the Line 16 to 32 in main.py to the parameters you want to try.

For EPSCN, one can refer to our implementation in 'https://github.com/qiaopTDUN/SR-ESPCN'.

For VGG19, one can download it in 'https://github.com/tensorflow/models/tree/master/research/slim'.

During our exploration of training SRGAN, we found it indeed hard to train. 
We only try a little hyperparameters combination, such as weights for mse loss, context loss and adversarial loss.
These weights change a little while the performance of the trained model changes a lot.
Very often, the train fails, mostly because the model collapses.

We have tried original GAN, WGAN. 
Adversarial loss + MSE loss, Adversarial loss + MSE loss + context loss.
Adam, SGD, RMSProp.

# Train
set 'is_train' to True.

set 'batch_size' to fit your GPU memory, larger is better.

set 'image_size' to fit your GPU memory, larger is better, but harder to train.

It may take some time before the training is done.

# Test
set 'is_train' to False

[1] C. Ledig, L. Theis, F. Huszar, J. Caballero, A. Cunningham,´A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang, et al. Photo-realistic single image super-resolution using a generative adversarial network. In CVPR, 2017.
