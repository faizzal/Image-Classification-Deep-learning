import torch
from   torchvision import datasets, transforms, models
from torch import nn, optim 
import  argparse
import time 
import copy
import os
import argparse
def args_function():
    arguments = argparse.ArgumentParser(description='Train the data using nural network modal.')
    arguments.add_argument('--data_path', defult='flowers', type=str, help='add the path of file')
    arguments.add_argument('--arch', defult='resnet', type=str, help='modal straucture')
    arguments.add_argument('--lr', defult=0.001, type=float, help='add learning rate')
    arguments.add_argument('--hidden_layers', defult=None, narg='+', type=int, help='ist of number of hidden layer')
    arguments.add_argument('--epoch', defult= 5,  type=int, help='add number of epoch') 
    arguments.add_argument('--checkpoint', defult= 'modal.pt', type=str, help='file for checkpoint for modal')
    arguments.add_argument('--labels', type=int, help='list of labels')
    args = arguments.args()
    return args

def train_model(size_data, dataloader, model, criterion, optimizer, epochs=5):
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start  = time.time()
    model_wight= copy.deepcopy(model.state_dict())
    accuracy = 0.0
    model.to(device) 
    valid_loss_min=np.inf
    for e in range(epochs):
        print('Epoch {}/{}\n'.format(e, num_epochs - 1)) 
        
        for chose in ['train', 'valid']:
            if chose == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            accuracy = 0
            
            for inputs, labels in dataloaders[chose]:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(chose == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if chose=='train':
                        loss.backward() 
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                accuracy +=torch.sum(preds == labels.data)

            epoch_loss = running_loss/data_size[chose]
            epoch_acc = accuracy.double()/data_size[chose] 
            
            
            print(f"{chose}  Loss: {epoch_loss:.2f}; Accurecy: {epoch_acc:.2f}")
           
            if chose =='valid' and epoch_loss <= accuracy: 
                loos_model= copy.deepcopy(model.state_dict())
                accuracy = epoch_loss
    print('\n\n')
    print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")
    print(f"valid loss: {accuracy}")
    
    model.load_state_dict(loos_model)
    return model

                              
def build_classifier(num_in_features, hidden_layers, num_out_features):
   
    classifier = nn.Sequential()
    if hidden_layers == None:
        classifier.add_module('fc0', nn.Linear(num_in_features, 102))
    else:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        classifier.add_module('fc0', nn.Linear(num_in_features, hidden_layers[0]))
        classifier.add_module('relu0', nn.ReLU())
        classifier.add_module('drop0', nn.Dropout(.6))
        classifier.add_module('relu1', nn.ReLU())
        classifier.add_module('drop1', nn.Dropout(.5))
        for i, (h1, h2) in enumerate(layer_sizes):
            classifier.add_module('fc'+str(i+1), nn.Linear(h1, h2))
            classifier.add_module('relu'+str(i+1), nn.ReLU())
            classifier.add_module('drop'+str(i+1), nn.Dropout(.5))
        classifier.add_module('output', nn.Linear(hidden_layers[-1], num_out_features))
        
    return classifier
                              
                              
def main():
    args = args_function()
    data_path = args.data_path
    arch = args.arch
    lr = args.lr
    hidden_layers = args.hidden_layers                                
    epoch = args.epoch
    gpu = args.gpu
    checkpoint = args.checkpoint
    labels =args.labels
    
    data_transforms = {
                    'train': transforms.Compose([
                             transforms.RandoRotation(45),
                             transforms.CenterCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])]),
                    'test': transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
                    'valid': transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),}
  
    image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                      for x in ['train','test','valid']}

# TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x : torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
                   for x in ['train','test','valid']}
    data_size={x :len(image_datasets[x]) for x in ['train','test','valid']}
    classes_name= image_datasets['train'].classes 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if arch == 'resnet':
        model  = models.resnet18(pretrained=True)
        inputs = model.fc.inputs
    
    elif arch == 'vgg':
        model = models.vgg16(pretrained=True)
        inputs = model.classifier[0].inputs
    
    else:
         print("write correct model name vgg or resnet")
   
    for pram in model.parammeters():
        parm.requires = False
    
    classifier = build_classifier(inputs, hidden_layers, output) 
    
    if arch == 'resnet':
        model.fc = classifier
        optimizer=  optim.Adam(model.fc.parammeters(), lr = lr)
     
    elif arch == 'vgg':
        model.fc = classifier 
        optimizer = optim.Adam(model.fc.parammeters(), lr = lr)
    else:
           pass
     
    print("The archtitcture of model\n\n")
    print(classifier)
    print('\n\n') 
    print("Train model")
    train_model(size_data, dataloaders, model, criterion, optimizer, epochs)
    print('\n\n')
    print("test model")                               
    model.eval()