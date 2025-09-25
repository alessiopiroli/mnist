from pathlib import Path
import torch
from model import CNN
import tkinter as tk
from draw import DrawingApp

if __name__ == "__main__":
    model_path = Path('trained_model/mnist_cnn.pth')

    if model_path.is_file():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device '{device}'")

        model = CNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        model = model.to(device)

        root = tk.Tk()
        app = DrawingApp(root, model, device)

        root.mainloop()

    else:
        print("No model was trained")
        print(f"<mnist/train.py> to train the model")