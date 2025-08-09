Leaf and Skin Segmentation with CA_SAUNet
This repository contains a PyTorch implementation of the CA_SAUNet model for semantic segmentation on two datasets: PlantDoc (leaf segmentation) and ISIC 2018 (skin lesion segmentation). The model supports dataset selection via a command-line argument, allowing flexible training and evaluation on either dataset.
Project Structure
The codebase is organized into the following files:

dataset.py: Defines dataset classes for PlantDoc and ISIC, along with data loader utilities.
model.py: Implements the CA_SAUNet model, supporting both PlantDoc and ISIC architectures.
utils.py: Contains metric functions for evaluation (Pixel Accuracy, Mean IoU, Dice Coefficient, F1 Score).
train.py: Main script for training and evaluating the model, with dataset selection.
requirements.txt: Lists required Python dependencies.

Requirements
To run the code, ensure you have Python 3.6+ and install the required dependencies:
pip install -r requirements.txt

The dependencies include:

torch
torchvision
numpy
Pillow
matplotlib
opencv-python
torchinfo

Dataset Setup
The datasets should be organized in the following structure under your data directory:
data_dir/
├── plantdoc/
│   ├── images/
│   └── masks/
├── isic/
│   ├── images/
│   └── masks/


PlantDoc: Images and masks for leaf segmentation. Masks should have pixel values of 38 for the foreground (leaf) and 0 for the background.
ISIC 2018: Images and masks for skin lesion segmentation. Masks should be binary (foreground > 128, background ≤ 128).

Ensure the data_dir path points to the root directory containing the plantdoc and isic folders.
Usage
To train and evaluate the model, use the train.py script with the desired dataset and configuration.
Command-Line Arguments

--dataset: Specifies the dataset to use (plantdoc or isic). Default: plantdoc.
--data_dir: Path to the root directory containing the datasets. Default: /content/drive/MyDrive/Igdir-Uni-Genel/Z_Works-2025.
--num_epochs: Number of training epochs. Default: 100.
--batch_size: Batch size for training. Default: 16.

Example Commands
Train on the PlantDoc dataset:
python train.py --dataset plantdoc --data_dir /path/to/data

Train on the ISIC dataset:
python train.py --dataset isic --data_dir /path/to/data

Output

Training: The script trains the model, saves checkpoints (checkpoint_<dataset>.pth), and saves the best model (best_model_<dataset>.pth) based on validation loss.
Evaluation: After training, the script evaluates the model on the test set, displaying metrics (Pixel Accuracy, Mean IoU, Dice Coefficient, F1 Score) and visualizing predictions for one sample per batch.
Loss Plot: A plot of training and validation loss is displayed at the end of training.

Notes

The CA_SAUNet model uses a simple U-Net architecture for PlantDoc and a DenseNet-based architecture with a shape stream for ISIC.
Training uses early stopping (patience: 20 epochs) to prevent overfitting.
Ensure a GPU is available for faster training. The script automatically selects CUDA if available, otherwise falls back to CPU.
Checkpoints and models are saved in the data_dir specified in the command-line arguments.

