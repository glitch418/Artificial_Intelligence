import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    
    if not isinstance(training, bool):
        raise TypeError("training parameter must be boolean")

    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    try:
        dataset = datasets.FashionMNIST('./data', train=training,download=True, transform=custom_transform)
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        loader = torch.utils.data.DataLoader(dataset, batch_size=64,shuffle=training)
        return loader
    except Exception as e:
        raise RuntimeError("Failed to load dataset")


def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    try:
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        return model
    except Exception as e:
        raise RuntimeError("Error in buildModel")


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    
    if not isinstance(model, nn.Module):
        raise TypeError("model must be of type nn.Module")
    if not isinstance(train_loader, torch.utils.data.DataLoader):
        raise TypeError("train_loader must be of type torch.utils.data.DataLoader")
    if not isinstance(criterion, nn.Module):
        raise TypeError("criterion must be of type nn.Module")
    if not isinstance(T, int):
        raise TypeError("T must be of type int")
    if T <= 0:
        raise ValueError("T must be a positive integer")

    
    try:
        opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        model.train()

        for epoch in range(T):
            correct = 0
            total_loss = 0
            total_samples = 0

            for data, target in train_loader:
                if torch.isnan(data).any() or torch.isinf(data).any():
                    raise ValueError("wrong values of input")
                if not torch.all((target >= 0) & (target < 10)):
                    raise ValueError("wrong values of target")
                opt.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                opt.step()

                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

            avg_loss = total_loss / total_samples
            print('Train Epoch: ' + str(epoch) + ' Accuracy: ' + str(correct) + '/' + str(total_samples) + ' (' + 
                str(100. * correct / total_samples) + '%) Loss: ' + str(avg_loss))
    except Exception as e:
        raise RuntimeError("Error in training model")
    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    
    if not isinstance(model, nn.Module):
        raise TypeError("model must be of type nn.Module")
    if not isinstance(test_loader, torch.utils.data.DataLoader):
        raise TypeError("test_loader must be of type torch.utils.data.DataLoader")
    if not isinstance(criterion, nn.Module):
        raise TypeError("criterion must be of type nn.Module")
    if not isinstance(show_loss, bool):
        raise TypeError("show_loss must be of type bool")

    try:
        model.eval()
        correct = 0
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                total_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)

        if show_loss:
            avg_loss = total_loss / total_samples
            print('Average loss: ' + str(avg_loss))

        print('Accuracy: ' + str(round(100. * correct / total_samples, 2)) + '%')
    except Exception as e:
        raise RuntimeError("Error in evaluating data")



def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be of type nn.Module")
    if not isinstance(test_images, torch.Tensor):
        raise TypeError("test_images must be of type torch.Tensor")
    if not isinstance(index, int):
        raise TypeError("index must be of type int")
    if index < 0 or index >= len(test_images):
        raise ValueError("Index cannot be less than 0 or out of bounds")

    try:
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

        model.eval()
        with torch.no_grad():
            output = model(test_images[index:index+1])
            probabilities = F.softmax(output, dim=1)[0]

            top_prob, top_class = torch.topk(probabilities, 3)

            for i in range(3):
                print(f"{class_names[top_class[i]]}: {top_prob[i].item()*100:.2f}%")
    except Exception as e:
        raise RuntimeError("Error in predict labels")
    

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()