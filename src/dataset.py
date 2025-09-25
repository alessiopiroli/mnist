from torch.utils.data import Dataset
import read_data

class MNISTDataset(Dataset):
    def __init__(self, training=True):
        if training:
            self.images, self.labels = read_data.get_training_data()
        else:
            self.images, self.labels = read_data.get_test_data()

    def __len__(self):
        return self.images.size(0)
    
    def __getitem__(self, idx):
        return (self.images[idx].float()/255.0).unsqueeze(0), self.labels[idx]