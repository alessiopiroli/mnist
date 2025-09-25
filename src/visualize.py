from pathlib import Path
import torch
from model import CNN
import matplotlib.pyplot as plt

model_path = Path('trained_model/mnist_cnn.pth')

if model_path.is_file():
    model = CNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # saving first batch of kernels
    kernels1 = model.conv1.weight.detach().cpu()

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
    fig.suptitle('First Convolutional Layer Kernels', fontsize=16)

    for i, ax in enumerate(axes.flat):
        kernel = kernels1[i]
        kernel_2d = kernel.squeeze()

        ax.imshow(kernel_2d, cmap='gray')
        ax.set_title(f'Kernel {i+1}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('kernels_visualizations/conv1_kernels.pdf')

    # saving the second batch of kernels
    kernels2 = model.conv2.weight.detach().cpu()

    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(12, 6))
    fig.suptitle('Second Convolutional Layer Kernels', fontsize=16)

    for i, ax in enumerate(axes.flat):
        filter_3d = kernels2[i]
        
        kernel_2d = filter_3d[0]

        ax.imshow(kernel_2d, cmap='gray')
        ax.set_title(f'Kernel {i+1}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('kernels_visualizations/conv2_kernels.pdf')

else:
    print("No model was trained")
    print(f"<mnist/train.py> to train the model")
