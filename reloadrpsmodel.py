import torch
from torch import nn
import torchvision
import random
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm.auto import tqdm
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix 
from pathlib import Path
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folderpath= os.path.join(os.path.dirname(__file__), 'Rock-Paper-Scissors')

test_dir=os.path.join(folderpath,'test')

custom_image_transform = transforms.Compose([
    transforms.Resize((128,128)),
])
test_data_simple=datasets.ImageFolder(test_dir,transform=custom_image_transform)

class_names=test_data_simple.classes




class TinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.layer_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size=3,stride=1,padding=0),#values we can set in our nn s are called hyperparameters
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  
        )
        self.layer_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)    
        )
        self.layer_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=0),
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
        x=self.clasifier(x)
        print(x.shape)
        return x'''
        return self.clasifier(self.layer_block_3(self.layer_block_2(self.layer_block_1(x))))
    
def teststep(model,data_loader,loss_fn):
    model.eval()
    testloss,testacc=0,0
    with torch.inference_mode():
        for batch, (X,Y) in enumerate(data_loader):
            X,Y=X.to(device),Y.to(device)
            y_pred=model(X)
            loss=loss_fn(y_pred,Y)
            testloss+=loss
            y_pred_class=y_pred.argmax(dim=1)
            testacc+=torch.sum(y_pred_class==Y).item()/len(Y)
        testloss=testloss/len(data_loader)
        testacc=testacc/len(data_loader)
    return testloss,testacc

model_path = os.path.join(os.path.dirname(__file__), 'mymodels', 'first_rps_model.pth')

model_1=TinyVGG(input_shape=3,hidden_units=20,output_shape=3).to(device)

model_1.load_state_dict(torch.load(model_path, map_location=device))

custom_image =torchvision.io.read_image(str("C:/Users/roryu/Desktop/learnpytoarch/rps33/paper/3tNHP8R9YHru011u.png")).type(torch.float32)

if custom_image.shape[0] == 4:
    # If it's a 4-channel image (RGBA), convert it to 3-channel (RGB)
    custom_image = custom_image[:3, :, :]

# Divide the image pixel values by 255 to get them between [0, 1]
custom_image = custom_image / 255. 

# Print out image data
'''
print(f"Custom image tensor:\n{custom_image}\n")
print(f"Custom image shape: {custom_image.shape}\n")
print(f"Custom image dtype: {custom_image.dtype}")
'''

'''plt.imshow(custom_image.permute(1, 2, 0)) # need to permute image dimensions from CHW -> HWC otherwise matplotlib will error
plt.title(f"Image shape: {custom_image.shape}")
plt.axis(False)
plt.show()
'''

custom_image_transform = transforms.Compose([
    transforms.Resize((128,128)),
])

# Transform target image
custom_image_transformed = custom_image_transform(custom_image)

model_1.eval()
with torch.inference_mode():
    # Add an extra dimension to image
    custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
    
    # Print out different shapes
    print(f"Custom image transformed shape: {custom_image_transformed.shape}")
    print(f"Unsqueezed custom image shape: {custom_image_transformed_with_batch_size.shape}")
    
    # Make a prediction on image with an extra dimension
    custom_image_pred = model_1(custom_image_transformed.unsqueeze(dim=0).to(device))


    # Print out prediction tensor

    print(f"Prediction tensor:\n{custom_image_pred}\n")

    # Get predicted class

    custom_image_pred_class = custom_image_pred.argmax(dim=1)

    # Print predicted class

    print(f"Predicted class: {custom_image_pred_class.item()}")

    # print prediction probabilities

    print(f"Prediction probabilities: {torch.softmax(custom_image_pred, dim=1)}")


