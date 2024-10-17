# Font Detection and Classification using PyTorch

This project demonstrates how to train an object detection model from scratch using PyTorch to detect instances of "Hello, World!" in images and classify the font used. The code is contained in a Jupyter Notebook and uses custom-generated data.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Data Generation](#data-generation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Running Inference](#running-inference)
- [Visualizing Results](#visualizing-results)
- [Notes](#notes)
- [Acknowledgments](#acknowledgments)

---

## Prerequisites

- **Operating System**: Linux (Ubuntu recommended), macOS, or Windows
- **Python**: Version 3.6 or higher
- **Jupyter Notebook**: To run the `.ipynb` file
- **PyTorch**: Version 1.7 or higher with CUDA support (if using GPU)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended for faster training)

---

## Project Structure

```
.
├── fonts/                   # Directory containing .ttf font files
├── Font_Detection.ipynb     # Jupyter Notebook containing all code
└── README.md                # Instructions and documentation (this file)
```

---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/font-detection-pytorch.git
cd font-detection-pytorch
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages

Install the necessary Python packages using `pip`:

```bash
pip install torch torchvision numpy pandas matplotlib Pillow jupyter
```

Alternatively, you can install `jupyterlab` if you prefer:

```bash
pip install jupyterlab
```

---

## Data Generation

### Why Generate Synthetic Data?

- **Control Over Data**: Generating synthetic data allows us to have complete control over the dataset, including the fonts used, text content, and image properties.
- **Diversity**: By varying fonts, sizes, positions, and other properties, we can create a diverse dataset that helps the model generalize better.
- **Resource Efficiency**: Collecting and annotating real-world data for object detection is time-consuming and resource-intensive. Synthetic data generation streamlines this process.

### Steps to Generate Data

#### 1. Place Font Files

Ensure that the `fonts/` directory contains the `.ttf` font files you intend to use. The font filenames should match those specified in the notebook.

#### 2. Open the Jupyter Notebook

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook  # or jupyter lab
```

Open the `Font_Detection.ipynb` notebook.

#### 3. Run Data Generation Cells

In the notebook, locate the **Data Generation** section and run the cells to generate synthetic images and annotations.

- **Font Selection**: We randomly select fonts from our list to simulate different text styles.
- **Text Placement**: Text is placed at random positions within the image to mimic real-world variability.
- **Font Size Range**: We choose a range of font sizes appropriate for the image dimensions to ensure the text is legible and fits within the image.

**Adjustable Parameters**:

- `num_images`: Number of images to generate.
- `image_width` and `image_height`: Dimensions of the generated images.
- `font_size`: Range of font sizes for the text.
- `num_instances`: Number of "Hello, World!" instances per image.

After running these cells, the generated images and annotations will be saved in the specified directories.

---

## Model Architecture

### Overview

The model consists of three main components:

1. **Convolutional Neural Network (CNN) Backbone**
2. **Region Proposal Network (RPN)**
3. **Font Classifier**

### Why a CNN Backbone?

- **Feature Extraction**: A CNN is used to extract rich features from the input images, capturing spatial hierarchies and patterns.
- **Locality**: Convolutional layers are effective at detecting local features, which is essential for identifying text regions.
- **Efficiency**: A shallow CNN with fewer layers reduces computational complexity, making it suitable for training from scratch on a smaller dataset.

**Implementation Details**:

- **Layers**: The backbone consists of convolutional layers followed by ReLU activations and max-pooling layers to reduce spatial dimensions.
- **Output Channels**: The final output has 32 channels, providing a feature map for subsequent processing.

### Why an RPN?

- **Objectness Prediction**: The RPN predicts objectness scores for various regions, indicating the likelihood of containing an object of interest ("Hello, World!" in this case).
- **Bounding Box Regression**: It also predicts bounding box coordinates, refining anchor boxes to better fit the detected objects.
- **Efficiency**: By sharing computation with the backbone and focusing on promising regions, the RPN accelerates the detection process.

**Implementation Details**:

- **Anchors**: Predefined boxes (anchors) of various scales and aspect ratios are placed uniformly across the image.
- **Convolutional Layers**: The RPN uses additional convolutional layers to predict objectness scores and bounding box deltas for each anchor.
- **Activation Functions**: Sigmoid activation is used for objectness scores to output probabilities between 0 and 1.

### Why Use Anchors?

- **Multi-Scale Detection**: Anchors allow the model to detect objects of different sizes and aspect ratios.
- **Spatial Coverage**: By placing anchors at various positions, scales, and ratios, the model can potentially detect objects anywhere in the image.
- **Bounding Box Regression**: Anchors serve as references for the model to predict adjustments (deltas) to better fit the objects.

**Implementation Details**:

- **Scales and Ratios**: We use scales like `[128]` and ratios like `[1.0]` to define the size and shape of anchors.
- **Anchor Generation**: A function generates all possible anchors over the feature map grid.
- **IoU Calculation**: Intersection over Union (IoU) is computed between anchors and ground truth boxes to assign labels during training.

### Font Classifier

- **Purpose**: Classifies the font of the detected "Hello, World!" instances.
- **Architecture**: Consists of an adaptive average pooling layer followed by a fully connected layer.
- **Why Adaptive Pooling?**: It allows the model to handle variable input sizes by producing a fixed-size output regardless of the input dimensions.

---

## Training the Model

### Loss Functions

- **Objectness Loss**: Binary Cross-Entropy Loss is used to measure how well the model predicts objectness scores for anchors.
- **Bounding Box Regression Loss**: Smooth L1 Loss is used for bounding box regression targets, balancing between L1 and L2 losses for robustness.
- **Classification Loss**: Cross-Entropy Loss is used for font classification, measuring the difference between predicted and true font labels.

### Training Loop

- **Data Loading**: Uses a custom `Dataset` and `DataLoader` to efficiently load and batch the training data.
- **Anchor Label Assignment**: Assigns positive or negative labels to anchors based on IoU with ground truth boxes.
- **Gradient Descent**: Optimizer updates the model parameters using gradients computed from the total loss.
- **Batch Processing**: Processes images in batches to utilize computational resources effectively.

**Adjustable Parameters**:

- `num_epochs`: Number of times the entire training dataset is passed through the model.
- `learning_rate`: Controls the step size during optimization.
- `batch_size`: Number of samples processed before the model is updated.

### Why These Choices?

- **Custom Training Loop**: Provides flexibility to handle multiple losses and custom data processing steps.
- **Loss Balancing**: Combining different loss functions allows the model to learn both detection and classification tasks simultaneously.
- **Optimization Strategy**: Using Adam optimizer for faster convergence on smaller datasets.

---

## Running Inference

### Steps

1. **Model Evaluation Mode**: Set the model to evaluation mode using `model.eval()` to disable dropout and batch normalization layers' training behaviors.
2. **Image Preprocessing**: The test image is preprocessed using the same transformations as the training data to ensure consistency.
3. **Model Prediction**: The model outputs objectness scores, bounding box deltas, and font scores for the input image.
4. **Reshaping Outputs**: The outputs are reshaped to align the dimensions for further processing.
5. **Applying Thresholds**: Objectness scores are thresholded to filter out low-confidence detections.
6. **Bounding Box Decoding**: Predicted bounding box deltas are applied to anchors to obtain final bounding boxes (not fully implemented in the code provided).
7. **Font Classification**: Font probabilities are computed using softmax, and the most probable font is selected for each detection.

### Why These Steps?

- **Consistency**: Preprocessing ensures that the model receives inputs in the format it was trained on.
- **Efficiency**: Thresholding reduces the number of boxes to consider, focusing on high-confidence detections.
- **Interpretability**: Decoding bounding boxes and mapping font indices to font names make the outputs understandable.

---

## Visualizing Results

### Importance of Visualization

- **Validation**: Helps verify that the model is making reasonable predictions.
- **Debugging**: Visual outputs can reveal issues with bounding box scaling, positioning, or classification errors.
- **Demonstration**: Provides an intuitive way to showcase model performance.

### Steps

1. **Plotting the Image**: Uses `matplotlib` to display the test image.
2. **Drawing Bounding Boxes**: Overlays predicted bounding boxes on the image using rectangles.
3. **Annotating Fonts**: Adds text labels for the predicted fonts and confidence scores near the bounding boxes.
4. **Adjusting Thresholds**: Visualization helps in fine-tuning thresholds for objectness and confidence.

---

## Notes

- **Font Files**: Ensure that the font filenames in the code match the actual filenames in the `fonts/` directory.
- **Adjusting Paths**: Update any file paths in the notebook to match your directory structure.
- **GPU Usage**: The notebook automatically detects and uses a GPU if available. Training on a GPU is significantly faster.
- **Batch Size**: If you encounter memory issues during training, try reducing the `batch_size` in the DataLoader.
- **Data Consistency**: Ensure that data preprocessing steps during training and inference are consistent.
- **Model Checkpoints**: Implement saving and loading model checkpoints if you wish to pause and resume training.
- **Bounding Box Decoding**: The code currently lacks proper decoding of bounding box deltas to obtain final bounding boxes. Implement this step using the anchors and predicted deltas for accurate localization.
- **Threshold Tuning**: Adjust objectness and confidence thresholds based on validation results to balance precision and recall.

---

## Acknowledgments

- **PyTorch Documentation**: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- **Font Sources**: [Google Fonts](https://fonts.google.com/)

---

## Contact

For any questions or issues, please contact [bvks2134@gmail.com](bvks2134@gmail.com).