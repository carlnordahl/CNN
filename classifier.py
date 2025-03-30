import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Net(nn.Module):
    """ A convolutional neural network
    """
    def __init__(self):
        super().__init__()
        # define the layers of the network
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 36, 3, padding=1)
        self.fc1 = nn.Linear(36 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # through first convolutional layer
        x = self.pool(F.relu(self.conv1(x)))
        # through second
        x = self.pool(F.relu(self.conv2(x)))
        # through third
        x = F.relu(self.conv3(x))
        # print("Shape before flattening:", x.shape) 
        # flatten to one-dimensional array
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def load_data(batch_size):
    """ Downloads or loads the needed data (depending if it's already downloaded).
    The data is transformed, normalized and gruped into batches of size batch_size.
    Returns a dataloader for the testing and training data.
    """
    # this transforms the data to a tensor and also normalizes the values
    transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), 
                                                     (0.5, 0.5, 0.5))])

    # download or at least load in the train and testing data
    print("Downloading training data...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    print("Downloading testing data...")
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader

def preview_batch(batch_size, trainloader):
    """ Preview a batch of the data
    """
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # display a batch
    for i in range(batch_size):
        plt.subplot(2, int(batch_size / 2), i + 1)
        img = images[i]
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.title(classes[labels[i]])

    plt.suptitle('Preview of training data', size = 20)
    plt.show()

def train_model(net, trainloader, epochs, path):
    print('training...')
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

    for epoch in range(epochs):
        for i, data in enumerate(tqdm(trainloader, desc = f'Epoch {epoch + 1} of {epochs}', leave=True, ncols=80)):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save the trained model
    torch.save(net.state_dict(), path)

def test_model(net, testloader, classes, batch_size):
    """ Test the model on a random batch of the data
    """
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # run images through network
    outputs = net(images)
    # select class with highest score (probability)
    _, predicted = torch.max(outputs, 1)

    # display the result
    for i in range(batch_size):
        plt.subplot(2, int(batch_size / 2), i + 1)
        img = images[i]
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')

        color = 'green'
        label = classes[predicted[i]]
        if classes[labels[i]] != label:
            color = 'red'
            label = f"({label})"
        plt.title(label, color=color)

    plt.suptitle('Objects found by model', size = 20)
    plt.show()

def evaluate(net, testloader, classes):
    """ Evaluate model by calculating accuracy for each class.
    Return arrays of all predictions and labels in the testing data for
    further evaluation.
    """
    correct_pred = {classname : 0 for classname in classes}
    total_pred = {classname : 0 for classname in classes}

    all_predictions = np.array([])
    all_labels = np.array([])

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            all_labels = np.append(all_labels, labels)

            outputs = net(images)

            _, predictions = torch.max(outputs, 1)

            all_predictions = np.append(all_predictions, predictions)

            # count number of correct predictions per class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
        
    # print accuracy per class and total accuracy
    avg_accuracy = 0.0
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        avg_accuracy += accuracy
        print(f'Accuracy for class {classname:5s} is {accuracy : .1f} %')
    avg_accuracy = avg_accuracy / 10
    print(f'Average accuracy is {avg_accuracy : .1f} %')
    
    return all_predictions, all_labels

def display_confusion_matrix(predictions, labels, classes):
    """ Display a confusion matrix for the labeled data
    """
    conf_matrix = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=classes)
    disp.plot(cmap='magma')
    plt.savefig('confusion_matrix.png')
    plt.show()


if __name__ == '__main__':
    # define the classes of the data
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    batch_size = 10

    trainloader, testloader = load_data(batch_size)

    # preview a random batch from the training data
    preview_batch(batch_size, trainloader)

    # create instance of model and train it
    net = Net()

    PATH = './cifar-net2epochs10.pth'
    train_model(net, trainloader, 10, PATH)

    # load trained model and test it
    net = Net()
    net.load_state_dict(torch.load(PATH))

    test_model(net, testloader, classes, batch_size)

    predictions, labels = evaluate(net, testloader, classes)
    display_confusion_matrix(predictions, labels, classes)

