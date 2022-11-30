from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision

def dataloader(dataset, input_size, batch_size, split='train'):
    # transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=([0.5]), std=([0.5]))])
    if dataset == 'mnist':
        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'fashion-mnist':
        data_loader = DataLoader(
            datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar10':
        data_loader = DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'faces_anime':
        trainset = torchvision.datasets.ImageFolder('./data/dataset_anime/data/' ,transform=transform)
        data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

    return data_loader
