import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import os
import numpy as np
from PIL import Image

#optimizer
LR = 0.01
MOMENTUM = 0.9
STEP_SIZE = 5
GAMMA = 0.1
#train
TRAIN_EPOCH = 5
TRAIN_BATCH_SIZE = 64
#test
TEST_EPOCH = 5
TEST_BATCH_SIZE = 128
#dir
DATA_DIR = '/Users/davidwu/Desktop/data/cifar-10-batches-py/'

##1.dataset
#也可以用torchvision内置的torchvision.datasets类来标准化处理
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class Cifar10_TrainDataset(Dataset):
    #如果enumerate出来的是np.ndarray，可以用torchvision.transforms.ToTensor()来转换，用transform来调用，同时也可以来数据增强等等处理，用transforms.Compose来组合操作；label也可以来转化格式
    def __init__(self,transform=None):
        self.transform = transform
        train = {'labels':[],'data':[]}
        for i in range(1,6):
            file = ''.join(('data_batch_',str(i)))
            file_dir = os.path.join(DATA_DIR,file)
            train_batch = unpickle(file_dir)
            #train['labels'].append(train_batch[b'labels']) #以后扩展list时append或者extend尽量不用，用+=[...]
            train['labels'] += train_batch[b'labels']
            train['data'].append(train_batch[b'data'])
        train['data'] = np.vstack(train['data']).reshape(-1, 3, 32, 32)
        #这种分开的东西，都是这样用list的append或者extend来先放到一个list里，再用vstack或者concatenate，记住：stack是新开一个维度来连接，cat是以现有轴来做连接
        #train['data'] = np.concatenate((train['data'])).reshape(-1, 3, 32, 32)   #也可以
        self.imgs = train['data'].transpose((0, 2, 3, 1))  #convert to HWC
        self.labels = train['labels']

    def __getitem__(self,index):
        img,label = self.imgs[index],self.labels[index]
        #doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        #label也可以遵循img的转化，从PIL.Image或者np.ndarray转化成tensor
        #if self.target_transform is not None:
        #    label = self.target_transform(label)
        return img,label

    def __len__(self):
        return len(self.imgs)

class Cifar10_TestDataset(Dataset):
    def __init__(self,transform=None):
        self.transform = transform
        test = {'labels':[],'data':[]}
        file = 'test_batch'
        file_dir = os.path.join(DATA_DIR,file)
        test = unpickle(file_dir)
        self.imgs = test[b'data']
        self.labels = test[b'labels']

    def __getitem__(self,index):
        img, label = self.imgs[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        #if self.target_transform is not None:
        #    label = self.target_transform(label)
        return img,label

    def __len__(self):
        return len(self.imgs)


##2.model
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    #features层用make_layers来造，features层是来抽图片的featuremap
    def __init__(self,features,num_classes=10):
        super(VGG,self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(64*512*7*7,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,num_classes),
        )
        self.initialize_weigths()

    def forward(self,x):
        x = self.features(x)
        print(x.shape)
        x = self.avgpool(x)
        print(x.shape)
        x = x.flatten()
        x = self.classifier(x)
        return x

    def initialize_weigths(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                #pytorch所有函数后面加下划线，表示执行in-place操作，即不会开辟新内存空间，在原变量上赋值
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                #bias全部置零
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

def make_layers(cfg,batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            ##list添加元素不用append，直接+=[...]
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
            if batch_norm:
                layers += [conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
            else:
                layers += [conv2d,nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg16():
    return VGG(make_layers(cfgs['D'],batch_norm=False))

##3.train
def main():
    #dataset&dataloader
    #transforms.ToTenser: Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    train_dataset = Cifar10_TrainDataset(transform=torchvision.transforms.ToTensor())
    test_dataset = Cifar10_TestDataset(transform=torchvision.transforms.ToTensor())
    #Dataloader类直接把label从int等格式转化成tensor
    train_loader = DataLoader(train_dataset,batch_size=TRAIN_BATCH_SIZE)
    test_loader = DataLoader(test_dataset,batch_size=TEST_BATCH_SIZE)
    #也可以用torchvision这一行内置的标准数据集模块来搞定
    #train_dataset = torchvision.datasets.CIFAR10(root='/Users/davidwu/Desktop/data',train=True,download=False,transform=torchvision.transforms.ToTensor())
    #train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=TRAIN_BATCH_SIZE)
    
    #model
    model = vgg16()

    #optimizer
    optimizer = optim.Adam(model.parameters(),lr=LR,)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=STEP_SIZE,gamma=GAMMA)

    #loss function
    criterion = nn.CrossEntropyLoss()

    #train
    train(train_loader,model,criterion,optimizer,scheduler)

    #test
    test(test_loader,model,criterion)

def train(train_loader,model,criterion,optimizer,sceduler):
    model.train()
    for epoch in range(TRAIN_EPOCH):
        for cur_iter,(img_batch,label_batch) in enumerate(train_loader):
            #print(img_batch.shape)
            output = model(img_batch)
            loss = criterion(output,label_batch)
            #backward learning;scheduler在optimizer更新后再更新，且不需要zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            #if cur_iter % 100 == 0:
            print('epoch:',epoch)
            print('cur_iter:',cur_iter)
            print('loss:',loss)


def test(test_loader,model,loss):
    model.eval()
    for epoch in range(TEST_EPOCH):
        for cur_iter,(img_batch,label_batch) in enumerate(test_loader):
            output = model(img_batch)
            loss = criterion(output,label_batch)
            if cur_iter % 100 == 0:
                print('epoch:',epoch)
                print('cur_iter:',cur_iter)
                print('loss:',loss)

if __name__ == '__main__':
    main()