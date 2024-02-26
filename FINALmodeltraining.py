import os
import random
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torch import nn
from torchinfo import summary
from tqdm.auto import tqdm
from timeit import default_timer as timer
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


folderpath= os.path.join(os.path.dirname(__file__), 'new dataset')
folderpathp=Path(folderpath)
items=os.listdir(folderpath)


train_dir=os.path.join(folderpath,'train')
test_dir=os.path.join(folderpath,'test')


class MyTinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.layer_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size=3,stride=1,padding=0),#values we can set in our nn s are called hyperparameters
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  
        )
        self.layer_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=0),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)    
        )
        self.layer_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=0),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)    
        )
        self.clasifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*12*12,out_features=output_shape),    
        )
    def forward(self,x):
        '''x=self.layer_block_1(x)
        print(x.shape)
        x=self.layer_block_2(x)
        print(x.shape)
        x=self.layer_block_3(x)
        print(x.shape)
        x=self.layer_block_4(x)
        print(x.shape)
        x=self.clasifier(x)
        print(x.shape)
         return x
        '''

       
        return self.clasifier(self.layer_block_3(self.layer_block_2(self.layer_block_1(x))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_step(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               ):
    model.train()

    train_loss,train_acc=0,0
    for batch, (X,Y) in enumerate (dataloader):
        X,Y=X.to(device),Y.to(device)
        
        y_pred=model(X)
        loss=loss_fn(y_pred,Y)
        train_loss+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class=y_pred.argmax(dim=1)
        train_acc+=torch.sum(y_pred_class==Y).item()/len(Y)
    train_loss=train_loss/len(dataloader)
    train_acc=train_acc/len(dataloader)
    return train_loss,train_acc

def teststep(model,data_loader,loss_fn):
    model.eval()
    testloss,testacc=0,0
    y_true = []
    y_pred = []
    with torch.inference_mode():
        for batch, (X,Y) in enumerate(data_loader):
            X,Y=X.to(device),Y.to(device)
            y_pred_batch=model(X)
            loss=loss_fn(y_pred_batch,Y)
            testloss+=loss
            y_pred_class=y_pred_batch.argmax(dim=1)
            testacc+=torch.sum(y_pred_class==Y).item()/len(Y)
            y_true.extend(Y.cpu().numpy())
            y_pred.extend(y_pred_class.cpu().numpy())
        testloss=testloss/len(data_loader)
        testacc=testacc/len(data_loader)
        cm = confusion_matrix(y_true, y_pred)

        # Display the confusion matrix
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_data_simple.classes)
        display.plot(cmap=plt.cm.Blues, values_format="d")
        plt.show()        
    return testloss,testacc


def train(model:torch.nn.Module,
        train_dataloader:torch.utils.data.DataLoader,
        test_dataloader:torch.utils.data.DataLoader,
        loss_fn:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        epochs:int=10):
    
    results={'train_loss':[],'train_acc':[],'test_loss':[],'test_acc':[]}
    for epoch in tqdm(range(epochs)):
        train_loss,train_acc=train_step(model,train_dataloader,loss_fn,optimizer)
        test_loss,test_acc=teststep(model,test_dataloader,loss_fn)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        print(f'Epoch {epoch+1}/{epochs}, train loss: {train_loss:.5f}, train acc: {train_acc:.2f}%, test loss: {test_loss:.5f}, test acc: {test_acc:.2f}%')
    return results
    

def plot_loss_curves(results: dict[str, list[float]]):
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [l.item() for l in loss], label='train_loss')  # Move loss to CPU and convert to float
    plt.plot(epochs, [l.item() for l in test_loss], label='test_loss')  # Move test_loss to CPU and convert to float
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a for a in accuracy], label='train_accuracy')  # Move accuracy to CPU and convert to float
    plt.plot(epochs, [a for a in test_accuracy], label='test_accuracy')  # Move test_accuracy to CPU and convert to float
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

#plot_loss_curves(results)

train_transform_trivial=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomPerspective(),
    transforms.ToTensor(),
    
])

test_transform_trivial=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

train_data_augment=datasets.ImageFolder(train_dir,transform=train_transform_trivial)

test_data_simple=datasets.ImageFolder(test_dir,transform=test_transform_trivial)  

BATCH_SIZE=32
NUM_WORKERS=0
torch.manual_seed(42)

train_dataloader_augment=DataLoader(train_data_augment,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)

test_dataloader_simple=DataLoader(test_data_simple,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)

model_1=MyTinyVGG(input_shape=3,hidden_units=20,output_shape=len(train_data_augment.classes)).to(device)

start_time=timer()

NUM_EPOCHS=30

results_1=train(model_1,train_dataloader_augment,test_dataloader_simple,loss_fn=nn.CrossEntropyLoss(),optimizer=torch.optim.AdamW(model_1.parameters(),lr=0.001),epochs=NUM_EPOCHS)

end_time=timer()

#print(results_1)

print(f'Training time: {end_time-start_time:.2f}s')

plot_loss_curves(results_1)

save_dir = os.path.join(os.path.dirname(__file__), 'mymodels')
os.makedirs(save_dir, exist_ok=True)

# Save the model in the "mymodels" directory
print(f"Saving model to: {save_dir}")
model_path = os.path.join(save_dir, 'finalmodelv3.3.pth')
torch.save(model_1.state_dict(), model_path)
#model_1_df=pd.DataFrame(results_1)
