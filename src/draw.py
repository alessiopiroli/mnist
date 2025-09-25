import tkinter as tk
from PIL import Image, ImageDraw
import torch
import numpy as np

class DrawingApp:
    def __init__(self, root, model, device):
        self.root = root
        self.model = model
        self.device = device
        self.root.title("MNIST Digit Recognizer")

        self.canvas = tk.Canvas(root, width=280, height=280, bg='black')
        self.canvas.pack()

        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict_digit)
        self.predict_button.pack()

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.result_label = tk.Label(root, text="Draw a digit and click Predict", font=("Helvetica", 16))
        self.result_label.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        self.draw.ellipse([x1, y1, x2, y2], fill='white', outline='white')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill='black')
        self.result_label.config(text="Draw a digit and click Predict")

    def predict_digit(self):
        img_resized = self.image.resize((28, 28), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img_resized))
        img_tensor = (img_tensor.float() / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)
    
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction_text = f"Prediction: {predicted.item()}"
            self.result_label.config(text=prediction_text)