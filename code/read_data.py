import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from osgeo import gdalnumeric

def load_img(path):
    img = gdalnumeric.LoadFile(path)
    img = np.array(img, dtype="float")
    # subtract mean of channels
    '''B, G, R = cv2.split(img)
    B = (B - np.mean(B))
    G = (G - np.mean(G))
    R = (R - np.mean(R))
    img_new = cv2.merge([B, G, R])'''
    img_new = img / 255.0
    return img_new

def default_loader(filename,root1,root2):
    img=load_img(root1 + '/' + filename)
    mask = gdalnumeric.LoadFile(root2 + '/' +filename).astype(np.float32)
    mask[mask > 0] = 1
    # data agumentation
    '''rot_p = random.random()
    flip_p = random.random()
    if (rot_p < 0.5):
        pass
    elif (rot_p >= 0.5):
        for k in range(3):
            img1[k, :, :] = np.rot90(img1[k, :, :])
            img2[k, :, :] = np.rot90(img2[k, :, :])
        mask = np.rot90(mask)
    if (flip_p < 0.25):
        pass
    elif (flip_p < 0.5):
        for k in range(3):
            img1[k, :, :] = np.fliplr(img1[k, :, :])
            img2[k, :, :] = np.fliplr(img2[k, :, :])
        mask = np.fliplr(mask)
    elif (flip_p < 0.75):
        for k in range(3):
            img1[k, :, :] = np.flipud(img1[k, :, :])
            img2[k, :, :] = np.flipud(img2[k, :, :])
        mask = np.flipud(mask)

    elif (flip_p < 1.0):
        for k in range(3):
            img1[k, :, :] = np.fliplr(np.flipud(img1[k, :, :]))
            img2[k, :, :] = np.fliplr(np.flipud(img2[k, :, :]))
        mask = np.fliplr(np.flipud(mask))'''
    mask=np.expand_dims(mask,axis=0)
    return  img,mask
class ImageFolder(data.Dataset):

    def __init__(self, trainlist, root1,root2):
        table = pd.read_table(trainlist, header=None, sep=',')
        trainlist = table.values
        self.ids = trainlist[:32]
        self.loader = default_loader
        self.root1 = root1
        self.root2 = root2




    def __getitem__(self, index):
        id = self.ids[index][0]
        img, mask = self.loader(id, self.root1,self.root2)
        img = torch.Tensor(img.copy())
        mask = torch.Tensor(mask.copy())
        return img,mask

    def __len__(self):
        return len(self.ids)

