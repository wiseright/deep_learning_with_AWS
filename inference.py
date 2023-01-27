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
#import sys
#import subprocess
import os
import sys
import logging
import argparse
import json
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# filename: inference.py
# from: https://hubofco.de/machinelearning/2020/03/22/Deploy-a-pytorch-model-to-Sagemaker/
def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading the model.')
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
  
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 3),
                   #nn.LogSoftmax(dim=1) 
    )
    

    with open(os.path.join(model_dir, 'model_0.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()
    logger.info('Done loading model')
    return model

def input_fn(request_body, request_content_type):
    logger.info('Deserializing the input data.')
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        url = input_data['url']
        logger.info(f'Image url: {url}')
        image_data = Image.open(requests.get(url, stream=True).raw)
               
        image_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return image_transform(image_data)
    raise Exception(f'Requested unsupported ContentType in content_type {content_type}')

def predict_fn(input_data, model):
    logger.info('Generating prediction based on input parameters.')
    if torch.cuda.is_available():
        input_data = input_data.view(1, 3, 224, 224).cuda()
    else:
        input_data = input_data.view(1, 3, 224, 224)
    with torch.no_grad():
        model.eval()
        out = model(input_data)
        ps = torch.exp(out)
    return ps

def output_fn(prediction, content_type):
    logger.info('Serializing the generated output.')
    classes = {0: 'Affenpinscher', 1: 'Afghan_hound', 2: 'Airedale_terrier'}

    topk, topclass = prediction_output.topk(3, dim=1)
    result = []
    
    for i in range(3):
        pred = {'prediction': classes[topclass.cpu().numpy()[0][i]], 'score': f'{topk.cpu().numpy()[0][i] * 100}%'}
        logger.info(f'Adding pediction: {pred}')
        result.append(pred)

    if accept == 'application/json':
        return json.dumps(result), accept
    raise Exception(f'Requested unsupported ContentType in Accept:{accept}')