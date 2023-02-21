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
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)



def create_dataframe(data):
    import pandas as pd
    # Erstelle einen leeren DataFrame
    df = pd.DataFrame()
    # Erstelle die restlichen Spalten mit den zweiten Elementen
    for sublist in data:
        df[sublist[0]] = sublist[0:]
    column_names = ["Epoch"]
    for i in range(len(data)):
        string_to_append = "Acc" + str(i+1)
        column_names.append(string_to_append)
    df.columns = column_names
    return df

def compare_acc(acc_list, save_string="Acc_compare.png"):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    figure, axes = plt.subplots()
    axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.grid()
    axes.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.ylabel("Accuracy in %")
    plt.xlabel("Epochs")
    plt.title("Accuracy comparison of NN with MNIST")
    df = create_dataframe(acc_list)
    # df = pd.DataFrame(data=acc_list, columns=["Epoch", "Accuracy"])
    sns.lineplot(data=df, x=df.Epoch, y=df.Acc1, ax=axes, label="Dense")
    sns.lineplot(data=df, x=df.Epoch, y=df.Acc2, ax=axes, label="Sparse")
    plt.legend(loc='lower right')
    # axes.legend(labels=["Acc1", "Acc2"])
    plt.savefig(save_string)
    plt.close()

def compare_models_robustness(mode: str, *models: nn.Module, noise_levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) -> None:
    from handle_model import handle_model
    acc_list_per_noise_level = []
    # noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for noise_level in noise_levels:
        print(noise_level)
        noisy_train_dataset = add_noise_to_mnist_dataset(train_dataloader.dataset, noise_level=noise_level)
        noisy_train_loader = torch.utils.data.DataLoader(dataset=noisy_train_dataset, batch_size=10, shuffle=False)

        noisy_test_dataset = add_noise_to_mnist_dataset(test_dataloader.dataset, noise_level=noise_level)
        noisy_test_loader = torch.utils.data.DataLoader(dataset=noisy_test_dataset, batch_size=10, shuffle=False)

        model_handlers = []
        model_names = ["Noise Level"]
        for model in models:
            model_handlers.append(handle_model(model, noisy_train_loader, noisy_test_loader))
            model_names.append(model.__class__.__name__)
            #models[0].__class__.__name__
        acc_list = []
        for model_runner in model_handlers:
            model_runner.run()
            acc_list.append(model_runner.training_acc[-1][-1])

        list_to_append = []
        list_to_append.append(noise_level)
        for acc in acc_list:
            list_to_append.append(acc)

        acc_list_per_noise_level.append(list_to_append)

    import pandas as pd
    df = pd.DataFrame(acc_list_per_noise_level)

    df.columns = model_names
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    figure, axes = plt.subplots()
    axes.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.grid()
    axes.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.ylabel("Accuracy in %")
    plt.xlabel("Noise Level")
    plt.title("Accuracy comparison of NN with MNIST")
        # df = pd.DataFrame(data=acc_list, columns=["Epoch", "Accuracy"])

    for index, row in df.iterrows():
        print(row)
    for i in range(df.shape[1]-1):
        sns.lineplot(data=df, x=df.iloc[:, 0], y=df.iloc[:, i+1], ax=axes, label=df.columns[i+1], marker="*",
                     markersize=8)

    plt.legend(loc='lower right')
        # axes.legend(labels=["Acc1", "Acc2"])
    plt.savefig("Compare.png")
    plt.close()




    print(acc_list_per_noise_level)
if __name__ == '__main__':
    #init_function()
    from get_data import load_mnist, add_noise_to_mnist_dataset
    batch_size = 10
    train_dataloader, test_dataloader = load_mnist(batch_size=batch_size)

    noisy_train_dataset = add_noise_to_mnist_dataset(train_dataloader.dataset, noise_level=0.25)
    noisy_train_loader = torch.utils.data.DataLoader(dataset=noisy_train_dataset, batch_size=10, shuffle=False)

    noisy_test_dataset = add_noise_to_mnist_dataset(test_dataloader.dataset, noise_level=0.25)
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
    cnn_model = CNNModel(in_features=28 * 28, hidden_features=128, out_features=10, bias=True)
    sparse_model = SparseModel(in_features=28 * 28, hidden_features=128, out_features=10, bias=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    compare_models_robustness("plot", dense_model, sparse_model, cnn_model)

    from handle_model import handle_model

    dense_model_runner = handle_model(dense_model, noisy_train_loader, noisy_test_loader)
    dense_model_runner.run()
    dense_model_runner.plot_training_acc(save_string="dense_training_acc_plot.png")

    sparse_model_runner = handle_model(sparse_model, noisy_train_loader, noisy_test_loader)
    sparse_model_runner.run()
    sparse_model_runner.plot_training_acc(save_string="sparse_training_acc_plot.png")

    #acc_list = [dense_model_runner.training_acc, sparse_model_runner.training_acc]
    #compare_acc(acc_list)













