import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple



def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               device: torch.device) -> Tuple[float, float]:
    
    """
    Trains the model for a single epoch 
    Args:
        model: a pytorch model to be trained
        dataloader: pytorch dataset
        loss: a loss function to monimize
        optimizer: optimzer to help minimise the function
        device: device to compute on (cpu or gpu)

    """


    model.train()
    
    train_loss, train_acc = 0, 0

    for _, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred_logits = model(X)
        loss = loss_fn(y_pred_logits, y)
        train_loss +=loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Calculate accuracy 
        y_pred_class = y_pred_logits.argmax(dim=1)#because pytorch doesnt use softmax in the model ...
        train_acc+= ((y_pred_class == y).sum().item()/len(y_pred_class))
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc




def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    
    """
    Validation step using the test data (validation data)
    """

    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for _, (X,y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_y_pred_logits = model(X)
            loss = loss_fn(test_y_pred_logits, y)
            test_loss +=loss

            test_y_pred_class = test_y_pred_logits.argmax(dim=1)
            test_acc += ((test_y_pred_class == y).sum().item() / len(test_y_pred_class))
    
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc



def train(model: torch.nn.Module,
          train_ds: torch.utils.data.DataLoader,
          test_ds: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device)-> Dict[str, List]:
    

    """
        This function will perform the training of the provided Model for N epochs. 

        Args:
            model: model you wish to train
            train_ds: train dataloader
            test_ds: validation dataloader
            optimizer: optimizer you wish yo use
            loss_fn: loss function you want to minimize
            epochs: how many epochs to train
            device: "cpu" or "gpou" depending on what is availbale

    """

    history = {"loss": [],
               "val_loss": [],
               "accuracy": [],
               "val_accuracy": []}
    

    for epoch in tqdm(range(epochs)):
        
        train_loss, train_acc = train_step(model, train_ds, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_ds, loss_fn, device)

        print(f"Epoch: {epoch+1} | " f"loss: {train_loss:.4f} | " f"accuracy: {train_acc:.4f} | " f"val_loss: {test_loss:.4f} | " f"val_accuracy: {test_acc:.4f}")

        # Update results dictionary
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(test_loss)
        history["val_accuracy"].append(test_acc)


    return history




