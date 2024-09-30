import torchmetrics
import mlxtend
import torch
import pandas as pd

from torch import nn
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# %%writefile train_step.py
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               device: str ="cpu"):
    """ Function to perform train step in CNN model classification

    Args:
    model: an nn.Module model
    dataloader: the (train) dataloader of images
    optimizer: an optimizer
    loss_fn: the loss function

    Kwargs:
    device: a device with default value "cpu"
    """
    model.train()
    
    train_loss = 0
    train_acc = 0
    
    for batch, (X, y) in enumerate(dataloader):
        # print(X.shape)
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y)
        # if batch % 10 == 0:
        #   print(f"Processed {batch} batches")

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: str ="cpu"):
    """ Function to perform test step in CNN model classification

    Args:
    model: an nn.Module model
    dataloader: the (test) dataloader of images
    loss_fn: the loss function

    Kwargs:
    device: a device with default value "cpu"
    """
    model.eval()
    
    test_loss = 0
    test_acc = 0
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y)
            y_pred_prob = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            test_acc += (y_pred_prob == y).sum() / len(y)
            
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module =nn.CrossEntropyLoss(),
          epochs: int =5,
          device: str="cpu"):
    """ Function to loop through train and test in CNN model classification

    Args:
    model: an nn.Module model
    train_dataloader: the (train) dataloader of images
    test_dataloader: the (test) dataloader of images
    optimizer: an optimizer

    Kwargs:
    loss_fn: the loss function, nn.CrossEntropyLoss is the default
    epochs: how many times to run the loop with default 5
    device: a device with default value "cpu"
    """
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model,
                                           train_dataloader,
                                           optimizer,
                                           loss_fn,
                                           device=device)
        test_loss, test_acc = test_step(model,
                                        test_dataloader,
                                        loss_fn,
                                        device=device)
    
        # if epochs <= 10:
        print(f"Epoch: {epoch+1} | Train loss: {train_loss:.5f} | Train acc: {train_acc:.5f} | Test loss: {test_loss:.5f} | Test acc: {test_acc:.5f}")
        # else:
        #     if epoch % 5 == 0:
        #         print(f"Epoch: {epoch+1}")
        #         print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.5f} | Test loss: {test_loss:.5f} | Test acc: {test_acc:.5f}")

        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
    
    return results

def confusion_matrix_plot(model: nn.Module,
                          test_dataloader: torch.utils.data.DataLoader,
                          class_names: list):
    # calculating the predictions on the test dataset
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        y_pred = []
        y_true = []
        for X, y in test_dataloader:
            y_logit = model(X.to(device))
            y_predictions = torch.softmax(y_logit.to("cpu"), dim=1).argmax(dim=1)
            y_pred.append(y_predictions)
            y_true.append(y)
        
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
    
    # calculating the confusion matrix
    confmat = ConfusionMatrix(task='multiclass',num_classes=len(class_names))
    confmat_tensor = confmat(preds=y_pred,
                             target=y_true).numpy()
    
    plot_confusion_matrix(confmat_tensor, class_names=class_names);


def plot_model_results(results: pd.DataFrame,
                       figsize: int=12):
    """ plotting train and test loss and accuracy over several epochs

    Args:
        results: a pd.DataFrame with 4 columns: train_loss, train_acc, test_loo, test_acc
        figsize1: figure width
        figsize2: figure height
    """
    
    plt.figure(figsize=(figsize, figsize))
    plt.subplot(2, 2, 1)
    plt.plot(range(len(results["train_loss"])), results["train_loss"])
    plt.plot(range(len(results["train_loss"])), results["test_loss"])
    plt.title("Loss")
    
    plt.subplot(2, 2, 2)
    plt.plot(range(len(results["train_loss"])), results["train_acc"])
    plt.plot(range(len(results["train_loss"])), results["test_acc"])
    plt.title("Accuracy")
    
    plt.subplot(2, 2, 3)
    plt.scatter(results["train_loss"], results["train_acc"])
    plt.title("Train loss vs train accuracy")
    
    plt.subplot(2, 2, 4)
    plt.scatter(results["test_loss"], results["test_acc"])
    plt.title("Test loss vs test accuracy");

def plot_random_test(model: torch.nn.Module,
                     directory,
                     class_names,
                     transform,
                     total_pictures: int=4,
                     device: str="cpu"):
    picture_list = list(Path(directory).blob("*/*.jpg"))
    random_pics = random.sample(range(len(picture_list)), total_pictures)
    #
    model.eval()
    for pic in random_pics:
        picture = Image.open(picture_list[pic])
        pic_for_model = transform(picture).to(device)
        true_class = picture_list[pic].parent.name
        with torch.inference_mode():
            pred = torch.softmax(model(pic_for_model.unsqueeze(dim=0)),dim=1)
        plt.figure()
        plt.imshow(picture)
        plt.title(f"True: {true_class}, Pred: {class_names[torch.argmax(pred, dim=1)]}, Prob: {pred.max():.3f}")
        plt.axis(False);


def save_model(model: nn.Module,
               model_path: str):
    torch.save(obj=model.state_dict(),
               f=Path(model_path))
