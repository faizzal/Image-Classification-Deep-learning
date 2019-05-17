import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import json
import argparse

from PIL import Image
def args_function():
        arguments = argparse.ArgumentParser(description='make prediction from saveed modal.')
        arguments.add_argument('--img_path',  type=str, help='add the path of image')
        arguments.add_argument('--json_file',  type=str, help='add the json file')
        arguments.add_argument('--topk', defult=5, type=int, help='add num of topk') 
        arguments.add_argument('--checkpoint', defult= 'modal.pt', type=str, help='file for checkpoint for modal') 
        arguments.add_argument('--gpu', defult= False, type=bool, help='a') 

        args = arguments.args()
        return args
def loading_chackpoint(file):
    checkpoint = torch.load(file)
    up_model   = checkpoint['model']
    up_optimizer  = checkpoint['optimizer']
    up_epoch = checkpoint['epoch']
    return up_model, up_optimizer, up_epoch
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    pil_image = Image.open(image)
    
    praper_img_trnsform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                                          std=(0.229, 0.224, 0.225))
                                    ])
    
    pil_image = praper_img_trnsform(pil_image)
    
    return pil_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    
    input_img = process_image(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_img.to(device)
    image = input_img.unsqueeze(0)
    classes_nam = []
    
    with torch.no_grad():
        output = model.forward(image) 
        prebs, class_cat = torch.topk(output, topk)        
        prebs = prebs.exp()
        
        prebs = np.asarray(prebs.cpu().detach().numpy().ravel())
        idx_classes = {model.class_to_idx[i]: i for i in model.class_to_idx}
        
        
        
        for c in class_cat.numpy()[0]:
            classes_nam.append(idx_classes[l])
        print(f'probability: {prebs},Calasses:{classes_nam}')     
        return prebs, classes_nam

def main():
    args = args_function()
    img_path = args.img_path
    json_file = args.json_file
    topk = args.topk
    checkpoint = args.checkpoint
    gpu = args.gpu
    model, __, __, = loading_chackpoint(checkpoint)
    classes_name = model.classes_name
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
         device = torch.device("cpu")
     
    print(f"the device is used is {device}")
    model.to(device)
    with open (json_file,'r') as names:
        cat_to_name = json.load(names)
        
    print('prediction\n\n')
    prebs,classes = predict(image_path, model, topk)
    names_of_flower =[cat_to_name[cat_to_name[classes[e]]] for e in classes]
    for prob, f_name in zip(prebs,classes):
        print (f'flower Name: {f_name} , propability = {f_name:.f4} ')