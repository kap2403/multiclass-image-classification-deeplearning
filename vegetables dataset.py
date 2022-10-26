#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import cv2


# In[2]:


import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader


# path=r"F:\deep learning\Vegetable Images\test"

# In[3]:


class training(Dataset):
    def __init__(self,path,transform=None):
        self.path=path   
        self.transform=transform
        image=[]
        label=[]
        classes=[]
        length=len(image)
        self.calsses=classes
        for i in os.listdir(path):
            path1=os.path.join(path,i)
            classes.append(i)
    
            for j in os.listdir(path1):
                img_path=os.path.join(path1,j)
                image.append(img_path)
                x=classes.index(i)
                label.append(x)
        self.images=image
        self.labels=label
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        image_path=self.images[idx]
        image=cv2.imread(image_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        target=self.labels[idx]
        if self.transform is not None:
            image=self.transform(image)
        target=torch.tensor([target])
        label=target
        return image,label


# In[4]:


train_transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Resize((100, 100)),
  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# In[5]:


dataset=training(r"F:\deep learning\Vegetable Images\train",transform=train_transforms)


# In[ ]:


len(dataset.calsses)


# In[ ]:


len(dataset.labels)


# In[ ]:


img, label = dataset[14999]
print(img.shape, label)
plt.imshow(img.permute(1,2,0))


# In[ ]:


len(dataset.labels)


# In[11]:


import torch.nn as nn
import torch.nn.functional as F


# In[12]:


#model.eval()


# In[14]:


import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# In[28]:


image,label=dataset[2999]
x=image.unsqueeze(0)
output=model(x)
print(output)
print(output.argmax())
print(label)


# In[16]:


from torch.utils.data import DataLoader 


# In[17]:


dataset_loader = torch.utils.data.DataLoader(
    dataset, batch_size=50, shuffle=True, num_workers=0)


# In[ ]:





# In[19]:


import math
block1 =100
pool1 =math.ceil((block1-3)/2 +1)
print(pool1)


block2=pool1

pool2 =math.ceil((block2-3)/2 +1)
print(pool2)



block3=pool2
pool3 =math.ceil((block3-3)/2 +1)
print(pool3)

block4=pool3
pool4 =math.ceil((block4-3)/2 +1)
print(pool4)


block5=pool4
pool5 =math.ceil((block5-3)/2 +1)
print(pool5)


#After flatten 
flatten= pool5 * pool5 * 512
print(f'After flatten:: {flatten}')


# In[32]:


class VGG16_NET(nn.Module):
    def __init__(self):
        super(VGG16_NET, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)#50*50
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)#50*50
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)#25*25
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)#25*25
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)#12*12
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)#12*12
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)#12*12
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)#6*6
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)#6*6
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)#6*6
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)#3*3
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)#3*3
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)#3*3

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc14 = nn.Linear(4608, 4096)
        self.fc15 = nn.Linear(4096, 4096)
        self.fc16 = nn.Linear(4096, 15)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.maxpool(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.maxpool(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc14(x))
        x = F.dropout(x, 0.5) #dropout was included to combat 
        x = F.relu(self.fc15(x))
        x = F.dropout(x, 0.5)
        x = self.fc16(x)
        return x


# In[33]:


model=VGG16_NET()


# In[34]:


image,label=dataset[1]
x=image.unsqueeze(0)
output=model(x)
print(output)
print(output.argmax())
print(label)


# In[35]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


# In[36]:


print(device)


# In[37]:


model = VGG16_NET() 
model = model.to(device=device) 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.001) 


# In[224]:


from tqdm import tqdm
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# criterion = nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss(reduction='sum')

MAX_EPOCHS = 100

for epoch in range(MAX_EPOCHS):
    running_loss = 0.0
    for input_tensor, labels in tqdm(dataset_loader):
        images = input_tensor.to(device).float()
        labels = labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)
        # loss = criterion(labels, torch.max(outputs, 1)[1])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print('epoch', epoch, 'loss', running_loss)


# In[ ]:





# In[ ]:




