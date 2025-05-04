# NEURAL-STYLE-TRANSFER

**COMPANY**: CODTECH IT SOLUTIONS  
**NAME**: DIGANT KATHIRIYA  
**INTERN ID**: CODF51  
**DOMAIN**: Artificial Intelligence Markup Language  
**DURATION**: 4 WEEKS

*Project Images*:
*NST-outputs*
![NST-outputs.png](https://github.com/digantk31/NEURAL-STYLE-TRANSFER/blob/main/project%20images/NST-outputs.png)

---

## Project Overview

In this internship task, you will develop and implement a Neural Style Transfer (NST) pipeline using PyTorch. Neural Style Transfer is a technique that blends the content of one image with the artistic style of another, producing visually striking results that merge real-world photographs with painterly brushstrokes or abstract patterns. Over the next four weeks, you will journey through the fundamentals of deep feature extraction, loss function design, and optimization routines that drive the NST process. By the end of this project, you will deliver a fully documented codebase capable of transforming arbitrary content images into stylized masterpieces, alongside sample outputs that demonstrate varying style and content weightings.

## Objectives

1. **Understand the NST Algorithm**: Gain a thorough grasp of how pre-trained convolutional neural networks (CNNs) like VGG19 capture content and style representations at different layers.
2. **Implement Preprocessing and Postprocessing Pipelines**: Build robust functions to load, resize, normalize, and convert images between NumPy arrays and PyTorch tensors.
3. **Extract Deep Feature Maps**: Wrap a VGG19 feature extractor to retrieve intermediate activations corresponding to designated content and style layers.
4. **Design Loss Functions**: Construct content loss (mean-squared error between feature maps), style loss (mean-squared error between Gram matrices of feature maps), and total variation loss (to encourage spatial smoothness).
5. **Optimize Image Pixels**: Set up an optimization loop using L-BFGS (or alternative optimizers) that updates the generated image tensor to minimize the combined loss.
6. **Experiment with Hyperparameters**: Explore different weights for content, style, and total variation losses, as well as varying numbers of optimization steps and image resolutions.
7. **Present Sample Outputs**: Generate at least five distinct stylized images using different source photographs and style references, documenting observations about convergence speed and visual fidelity.
8. **Write Comprehensive Documentation**: Provide clear, step-by-step instructions for installation, usage, and parameter tuning so that others can reproduce and extend your work.

## Task Description

1. **Foundations and Preprocessing**

   * Set up the development environment with Python 3.9+, PyTorch, torchvision, OpenCV, and other dependencies.
   * Implement image loading and resizing functions that preserve aspect ratio and handle edge cases like missing files.
   * Develop normalization transforms using ImageNet mean and standard deviation, ensuring consistency with pre-trained model expectations.

2. **Feature Extraction and Model Wrapper**

   * Load a pre-trained VGG19 network and freeze its parameters.
   * Identify which convolutional blocks correspond to low-level texture (style) and high-level content features.
   * Build a custom `StyleContentExtractor` module that slices VGG19 at specified layer indices and returns activations for content and style layers.

3. **Loss Computation and Optimization**

   * Write functions to compute the Gram matrix for capturing correlations between feature channels.
   * Define content loss (MSE between content features of the generated image and the original content image).
   * Define style loss (MSE between Gram matrices of generated and style images across multiple layers).
   * Incorporate total variation loss to reduce noise and improve visual coherence.
   * Choose and configure an optimizer (L-BFGS or Adam) to iteratively update the generated image tensor.

4. **Experimentation, Output Generation, and Documentation**

   * Run experiments at different image resolutions (256×256, 512×512, 1024×1024) and compare results.
   * Vary the ratio of content to style weights (e.g., 1e5:1e3, 1e4:1e4) and observe the aesthetic trade-offs.
   * Save final stylized images in an organized directory structure along with intermediate checkpoints if desired.
   * Package the project for sharing: include a `requirements.txt`, clear folder hierarchy (`data/`, `models/`, `scripts/`, `outputs/`), and sample command-line invocations.

## Deliverables

* **`README.md`**: Detailed project overview, installation steps, usage examples, and explanation of design decisions.
* **`nst.py`**: Well-commented script or Jupyter notebook implementing the Neural Style Transfer pipeline end to end.
* **`data/` Folder**: Placeholders for content and style images, along with any sample files.
* **`models/` Folder**: Saved PyTorch model definitions or state dictionaries if applicable.
* **`outputs/` Folder**: At least five stylized images demonstrating different style and content combinations.
* **`requirements.txt`**: Exact versions of Python packages used.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/digantk31/NEURAL-STYLE-TRANSFER.git
   cd Neural-Style-Transfer
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Prepare your images:

   * Drop your content image(s) into `data/content/`
   * Drop your style image(s) into `data/style/`
4. Run the script or notebook:

   ```bash
   python nst.py --content data/content/your_photo.jpg \
                  --style data/style/your_style.jpg \
                  --output outputs/styled.jpg \
                  --size 512 --steps 300 --content-weight 1e5 --style-weight 1e4
   ```
