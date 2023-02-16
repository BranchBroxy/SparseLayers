import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def init_function():
    # Use a breakpoint in the code line below to debug your script.
    print(torch.cuda.current_device())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

if __name__ == '__main__':
    init_function()
    from get_data import load_mnist
    batch_size = 10
    train_dataloader, test_dataloader = load_mnist(batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from models import LinearNN, DenseModel, CNNModel, SparseModel
    # model = LinearNN().to(device)
    from handle_model import train, test
    model = DenseModel(in_features=28*28, hidden_features=128, out_features=10, bias=True)
    model = CNNModel(in_features=28*28, hidden_features=128, out_features=10, bias=True)
    model = SparseModel(in_features=28*28, hidden_features=128, out_features=10, bias=True)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 5

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

