# MNIST PyTorch Digit Recognizer
This project implements a system to recognize handwritten digits using a Convlutional Neural Network (CNN) in PyTorch.
It includes:
- A CNN architecture with two convolutional layers
- Training and evaluations scripts with GPU support
- An interactive GUI that loads the trained model and lets you draw a digit with your mouse to see real-time prediction

### Setup
```bash
# Clone the repository
git clone https://github.com/alessiopiroli/mnist.git
cd mnist

# Create and activate a Conda environment
conda create --name <environment_name> python=3.11
conda activate <environment_name>

# Install dependencies
conda install pytorch torchvision pillow matplotlib -c pytorch -c nvidia

# Download the data from http://yann.lecun.com/exdb/mnist/
```

### Training the model
Run the `train.py` script to train the model from scratch. This creates the `mnist_cnn.pth` file containing the learned weights
```bash
python src/train.py
```

### Testing
Run the `train.py` script to launch the interactive application. This will load the trained `mnist_cnn.pth` file and open a window.
```bash
python src/test.py
```
Draw a single digit (0-9) in the black box and click the `Predict` button to see the model's preduction.
<div align="center">
<img width="412" height="528" alt="image" src="https://github.com/user-attachments/assets/4bfaa173-8fc7-4115-94e9-974234e48f57" />
<img width="412" height="528" alt="image" src="https://github.com/user-attachments/assets/8786b2ed-b01b-4a95-a713-12feaf9348bf" />
</div>

### Kernel Visualization
The project also includes scripts to visualize what the convolutional layers have learned.
Below are the 16 and 32 kernels learned from the first and second convolutional layers respectively.
<div align="center">
<p float="left">
  <img src="https://github.com/user-attachments/assets/83fbc8ca-dd30-44ce-933f-e54d28c117c0" height="300"/>
  <img src="https://github.com/user-attachments/assets/86fd042d-1d01-4fc6-bea9-737b9767540c" height="300"/>
</p>
</div>
