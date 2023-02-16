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
    from get_data import load_mnist, add_noise_to_mnist_dataset
    batch_size = 10
    train_dataloader, test_dataloader = load_mnist(batch_size=batch_size)

    noisy_train_dataset = add_noise_to_mnist_dataset(train_dataloader.dataset, noise_level=0.9)
    noisy_train_loader = torch.utils.data.DataLoader(dataset=noisy_train_dataset, batch_size=10, shuffle=False)

    noisy_test_dataset = add_noise_to_mnist_dataset(test_dataloader.dataset, noise_level=0.9)
    noisy_test_loader = torch.utils.data.DataLoader(dataset=noisy_test_dataset, batch_size=10, shuffle=False)

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2)
    from torchvision.utils import make_grid

    image = noisy_train_loader.dataset[0][0]
    ax1.imshow(image[0, :, :], cmap='gray')
    image = train_dataloader.dataset[0][0]
    ax2.imshow(image[0, :, :], cmap='gray')
    plt.show()
    from models import LinearNN, DenseModel, CNNModel, SparseModel
    dense_model = DenseModel(in_features=28 * 28, hidden_features=128, out_features=10, bias=True)
    # model = CNNModel(in_features=28 * 28, hidden_features=128, out_features=10, bias=True)
    sparse_model = SparseModel(in_features=28 * 28, hidden_features=128, out_features=10, bias=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from handle_model import handle_model
    dense_model_runner = handle_model(dense_model, noisy_train_loader, noisy_test_loader)
    dense_model_runner.run()

    sparse_model_runner = handle_model(sparse_model, noisy_train_loader, noisy_test_loader)
    sparse_model_runner.run()





