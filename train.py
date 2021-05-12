import os
import time

import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torchvision.models import resnet50

import settings
from arcface import Arcface

from focal_loss import FocalLoss


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


class NormalizeImage:
    def __call__(self, pic):
        # Our processing step is as follows
        # Each pixel is normalized by subtracting 127.5 and dividing by 128 after
        pic_as_array = np.array(pic)
        pic_as_array = (pic_as_array - 127.5) / 128.0
        return pic_as_array


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_transforms = transforms.Compose([
    NormalizeImage(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.ConvertImageDtype(dtype=torch.float32)
])

image_dataset = torchvision.datasets.ImageFolder(settings.image_dataset_dir, transform=image_transforms)
train_loader = data.DataLoader(image_dataset, shuffle=True)

num_of_classes = len(train_loader.dataset.class_to_idx)
criterion = torch.nn.CrossEntropyLoss() if settings.loss_function == "softmax" \
    else FocalLoss(gamma=2)
softmax = torch.nn.Softmax()
model = resnet50(pretrained=False)
if settings.loss_function == "softmax":
    model.fc = torch.nn.Linear(in_features=2048, out_features=num_of_classes)
    optimizer = torch.optim.SGD([{'params': model.parameters()},
                                 ],
                                lr=1e-3, weight_decay=5e-4)
    model.to(device)
    model = DataParallel(model)
else:
    model.fc = torch.nn.Linear(in_features=2048, out_features=512)
    metric_fc = Arcface(512, num_of_classes=num_of_classes)
    optimizer = torch.optim.SGD([{'params': model.parameters()},
                                 {'params': metric_fc.parameters()}
                                 ],
                                lr=1e-3,
                                weight_decay=5e-4,
                                momentum=0.9)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
correct_guesses = torch.tensor([], device=device, dtype=torch.bool)

for i in range(settings.max_epoch):

    correct_guesses = torch.tensor([], device=device, dtype=torch.bool)
    correct_guesses.to(device)
    model.train()
    for ii, (image_data, image_label_id) in enumerate(train_loader):
        optimizer.zero_grad()
        image_data, image_label_id = image_data.to(device), image_label_id.to(device)
        feature = model(image_data)
        # output = feature

        output = feature if settings.loss_function == "softmax" else softmax(metric_fc(feature, image_label_id))
        correct_guesses = torch.cat([correct_guesses, output.argmax() == image_label_id])
        correct_guesses.to(device)
        loss = criterion(output, image_label_id)
        loss.backward()
        optimizer.step()

        iters = i * len(train_loader) + ii
        acc = correct_guesses.sum() / len(correct_guesses)
        time_str = time.asctime(time.localtime(time.time()))
        print('{} train epoch {} iter {}  iters/s loss {} acc {}'.format(time_str, i, ii, loss.item(), acc))

    scheduler.step()
    model.eval()

    save_model(model, settings.models_dir,
               f"resnet50-{settings.loss_function}", i)
