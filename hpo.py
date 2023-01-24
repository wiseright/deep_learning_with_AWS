#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms as T
import torch.nn.functional as F
#import webdataset as wds
from torchvision.datasets import ImageFolder
import sys
import subprocess

import argparse

# Definisce una map su ogni elemento Resize immagine
def preprocess(sample):
    image, cls = sample
    #image = T.Resize(size=(224,224))(T.ToTensor()(image))
    image = T.ToTensor()(image)
    image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    image = T.Resize(size=(224,224))(image)
    label = cls
    
    return 1-image, label

def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   
    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model

def create_loader(url, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    # Install webdataset
    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'webdataset'])
    
    import webdataset as wds
    
    #Create Dataset from tar archive
    dataset = wds.WebDataset(url).shuffle(1000).decode("rgb").to_tuple("jpg.input.jpg", "jpg.target.cls").map(preprocess)
    dataset = dataset.batched(16)
    
    #Create Loader
    loader = wds.WebLoader(dataset, num_workers=2, batch_size=None)
    loader = loader.unbatched().shuffle(1000).batched(batch_size)
    
    #transform = transforms.Compose([
    #    transforms.RandomHorizontalFlip(p=0.5),
    #    transforms.Resize((224, 224)),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    #)
    #
    #dataset = ImageFolder(root=url, transform=transform)
    #loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #
    return loader


def main(args):
    print(args)
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''      
    train_loader = create_loader(url = 'https://sagemaker-us-east-1-265345480326.s3.amazonaws.com/project_3/dogImages_dataset.tar',
                                batch_size=args.batch_size)
    #train_loader = create_loader(url = './dogImages/train',
    #                             batch_size=args.batch_size)
    
    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test_loader = create_loader(url = 'https://sagemaker-us-east-1-265345480326.s3.amazonaws.com/project_3/dogImages_dataset.tar',
                                batch_size=args.test_batch_size)
    #test_loader = create_loader(url = './dogImages/test',
    #                             batch_size=args.test_batch_size)
    
    test(model, test_loader, criterion)
    
    # training and testing
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, './model.pt')

   
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )

    # Container environment
    # parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    # parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    # parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    # parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    # parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args=parser.parse_args()
    
    main(args)
