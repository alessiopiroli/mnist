import torch
import torch.nn as nn
from dataset import MNISTDataset
from torch.utils.data import DataLoader
from model import CNN

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device '{device}'")

    training_dataset = MNISTDataset(training=True)
    train_loader = DataLoader(dataset=training_dataset, batch_size=64, shuffle=True)
    
    model = CNN()
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5

    print(f"--- Starting training ---")

    for i in range(num_epochs):
        print(f"--- Epoch {i+1}/5 ---")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

    print("--- Ending training ---")

    print("--- Testing ---")

    testing_dataset = MNISTDataset(training=False)
    test_loader = DataLoader(dataset=testing_dataset, batch_size=64, shuffle=False)

    n_correct = 0
    n_samples = 0

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

    accuracy = 100.0 * n_correct / n_samples
    print(f"Accuracy: {accuracy:.2f}%")

    save_path = 'trained_model/mnist_cnn.pth'
    torch.save(model.state_dict(), save_path)

