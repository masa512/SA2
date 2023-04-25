import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from torch.utils.data import Dataset

def read_image(image_path = ''):
    """
    Wrapper function to read image and raw image
    """
    I = plt.imread(image_path)
    # Rescale to 0-1
    I = I/255

    return I

class simple_dataset(Dataset):
  
  def __init__(self,path = 'data', mode = 'train', box_dim = (50,50)):
    self.path = os.path.join(path,mode)
    self.file_list = [fn for fn in os.listdir(self.path) if fn.endswith('.jpg')]
    self.box_dim = box_dim
  
  def __len__(self):
    return len(self.file_list)

  def __getitem__(self,idx):
    image_path = os.path.join(self.path,self.file_list[idx])
    I = np.mean(read_image(image_path),axis=-1,keepdims=False)
    I = torch.Tensor(I).view(1,I.shape[1],I.shape[0]).float()
    
    # Generate mask
    M = torch.ones_like(I).float()
    #min0 = M.shape[0]//2 - self.box_dim[0]//2
    min0 = 128-20
    #min1 = M.shape[1]//2 - self.box_dim[1]//2
    min1 = min0
    #max0 = M.shape[0]//2 + self.box_dim[0]//2
    max0 = 128+20
    #max1 = M.shape[1]//2 + self.box_dim[1]//2
    max1=max0
    M[0,min0:max0,min1:max1] = 0.0

    # Mask the input
    return I,M
