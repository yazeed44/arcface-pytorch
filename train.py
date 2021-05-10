import time

import os
import numpy as np
import torch
import torchvision.datasets
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torchvision.models import resnet50
from torch.nn import DataParallel

import settings
from arc_margin_product import ArcMarginProduct
from face_identity_dataset import FaceIdentityDataset
import torchvision.transforms as transforms



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

image_dataset = torchvision.datasets.ImageFolder("dataset/", transform=image_transforms)
train_loader = data.DataLoader(image_dataset, shuffle=True)

num_of_classes = len(train_loader.dataset.class_to_idx)
criterion = torch.nn.CrossEntropyLoss()
model = resnet50(pretrained=False)
metric_fc = torch.nn.Linear(in_features=2048, out_features=num_of_classes, bias=True)

model.fc = metric_fc
model.to(device)
model = DataParallel(model)
# metric_fc.to(device)
# metric_fc = DataParallel(metric_fc)
optimizer = torch.optim.SGD([{'params': model.parameters()},
                             # {'params': metric_fc.parameters()}
                             ],
                            lr=1e-1, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
correct_guesses = torch.tensor([], device=device, dtype=torch.bool)
correct_guesses.to(device)

for i in range(50):

    correct_guesses = torch.tensor([], device=device, dtype=torch.bool)
    correct_guesses.to(device)
    model.train()
    for ii, (image_data, image_label_id) in enumerate(train_loader):
        image_data, image_label_id = image_data.to(device), image_label_id.to(device)
        feature = model(image_data)
        correct_guesses = torch.cat([correct_guesses, feature.argmax() == image_label_id])
        correct_guesses.to(device)
        loss = criterion(feature, image_label_id)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    iters = i * len(train_loader) + ii
    acc = correct_guesses.sum() / len(correct_guesses)
    time_str = time.asctime(time.localtime(time.time()))
    print('{} train epoch {} iter {}  iters/s loss {} acc {}'.format(time_str, i, ii, loss.item(), acc))

    model.eval()
    # acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
    # if opt.display:
    #     visualizer.display_current_results(iters, acc, name='test_acc')
    save_model(model, "models/", "resnet50-crossentropy-", i)
