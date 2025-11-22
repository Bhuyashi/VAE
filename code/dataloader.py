from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def data_loader(dataset, batch_size):
    if dataset == "mnist":
        train_dataset = datasets.MNIST(root='../data/', download=True,train=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root='../data/', train=False, transform=transforms.ToTensor())
    elif dataset == "fashion_mnist":
        train_dataset = datasets.FashionMNIST(root='./data/', download=True, train=True, transform=transforms.ToTensor())
        test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor())

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader