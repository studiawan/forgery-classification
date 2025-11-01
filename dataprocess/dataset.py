from torchvision import datasets, transforms
import os
import time
from torch.utils import data
import numpy as np
from PIL import Image
import copy

class ImageForgery(data.Dataset):
  def __init__(self, root, trans=None, train=True, test=False):
    self.test = test
    self.train = train
    imgs = [os.path.join(root, img) for img in os.listdir(root)]

    if test:
      sorted(imgs, key=lambda x: int(x.split(".")[-2].split("\\")[-1])) 
    else:
      sorted(imgs, key=lambda x: int(x.split(".")[-2])) 
    
    # shuffle
    np.random.seed(100)
    imgs = np.random.permutation(imgs)
    
    # split dataset
    if self.test:
      self.imgs = imgs
    elif train:
      self.imgs = imgs[:int(0.7*len(imgs))]
    else:
      self.imgs = imgs[int(0.7*len(imgs)):]
    
    if trans==None:
      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
      # test and dev dataset do not need to do data augemetation
      if self.test or not self.train:
        self.trans = transforms.Compose([
                                        transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize
                                        ])
      else:
        self.trans = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224), # RandomSizedCrop(224)??
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                        ])
      
    
  def __getitem__(self, index):
    imgpath = self.imgs[index]
    if self.test:
      label = int(imgpath.split(".")[-2].split("\\")[-1])
    else:
      kind = imgpath.split(".")[-3].split("\\")[-1]

      if kind == "asli":
        label = 0
      elif kind == "palsu":
        label = 1

    img = Image.open(imgpath).convert('RGB')
    img = self.trans(img)
    return img, label
  
  def __len__(self):
    return len(self.imgs)