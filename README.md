# Image Super Resolution with GANs
- This project implements a Generative AI model based on SRGAN (Super-Resolution Generative Adversarial Network) to upscale low-resolution images, enhancing their clarity and realism. The model is built using PyTorch and trained on custom datasets to generate high-quality super-resolved images.

## Description
- The SRGAN model uses a deep convolutional neural network (CNN) architecture consisting of a generator and discriminator trained adversarially. The generator upscales low-resolution images, while the discriminator learns to distinguish generated images from real high-resolution images. This adversarial training helps produce visually realistic super-resolution outputs.

## Technologies & Libraries Used

- Python – Core programming language

- PyTorch – Deep learning framework

- SRGAN – Super-Resolution Generative Adversarial Network for image enhancement

- CNN (Convolutional Neural Network) – Backbone architecture in the generator and discriminator networks

- NumPy – Numerical computations

- OpenCV / PIL – Image processing

- Matplotlib – Visualization

## Dataset

- Dataset: **DIV2K**

The model is trained using paired low-resolution and high-resolution images.

- High-Resolution Images (HR): Serve as the ground truth for training and evaluation
- Low-Resolution Images (LR): Used as input to the generator

**Note:** Due to size limitations, the dataset is not included in this repo. You can download it from the [DIV2K Official Site](https://data.vision.ee.ethz.ch/cvl/DIV2K/).


## Model Architecture

- Generator: Learns to upscale images with perceptual and content loss.

- Discriminator: Trained to distinguish between real and generated high-resolution images.

- Loss Functions: Adversarial loss, perceptual loss (using VGG19), and content loss to produce high-quality images.

## Environment Setup & Usage

### Step 1: Set Up Environment (using Anaconda)
1. Open Anaconda Prompt and navigate to the project folder:
```
cd path/to/Image-Super-Resolution-GAN
```
2. Create the environment using `environment.yml`:
```
conda env create -f environment.yml
```
3. Activate the new environment:
 ```  
conda activate srganenv_gpu
```
5. Install PyTorch (select one based on your system):
   
For GPU:
```
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
```
For CPU:
```
conda install pytorch-cpu==1.1.0 torchvision-cpu==0.3.0 cpuonly -c pytorch
```
### Step 2: Train the SRGAN Model

Ensure your training dataset is organized as:
```
custom_dataset/
  ├── hr_train_LR/   ← Low-resolution images        
  └── hr_train_HR/   ← Corresponding high-resolution images
```
Then run:
```
python main.py --LR_path custom_dataset_cars/hr_train_LR --GT_path custom_dataset_cars/hr_train_HR
```
### Step 3: Test the Model

Test your trained model with:
```
python main.py --mode test_only --LR_path test_data/cars --generator_path ./model/srgan_custom.pt
```
**Note:** All output images will be saved in the `results` folder.
