from torch.utils.data import DataLoader
from torchvision import transforms, datasets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    transforms.Resize((224,224))
])

training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
testing_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

size = 64
train_data = DataLoader(dataset=training_data, batch_size=size, shuffle=True, drop_last=True)
test_data = DataLoader(dataset=testing_data, batch_size=size, shuffle=True, drop_last=True)