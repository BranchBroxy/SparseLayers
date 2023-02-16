import torch


def train(dataloader, model, loss_fn, optimizer, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    size = len(dataloader.dataset)
    n_total_steps = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch + 1) % 1000 == 0:
            epoch = 1
            num_epochs = 1
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
            #print(model.sl2.weight)
            #print(model.sl2.connections)




def test(dataloader, model, loss_fn, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")