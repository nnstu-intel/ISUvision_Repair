import os
import cv2
import shutil
from tqdm import tqdm
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, models

from efficientnet_pytorch import EfficientNet

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Добавление к словарю пути к изображению
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

'''

# Деление датасета на train и val
def prepare_folders_images(train_dir, val_dir, class_names, data_root):
    for dir_name in [train_dir, val_dir]:
       for class_name in class_names:
            os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)

    for class_name in class_names:
        source_dir = os.path.join(data_root, class_name)
        for i, file_name in enumerate(tqdm(os.listdir(source_dir))):
            if i % 6 != 0:
                dest_dir = os.path.join(train_dir, class_name)
            else:
                dest_dir = os.path.join(val_dir, class_name)
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))

# Обучение модели | сами шаги выполнения
def train_model(model, loss, optimizer, scheduler, num_epochs, train_dataloader, val_dataloader):
    global_loss = []
    global_acc = []

    print('Training:')
    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                dataloader = val_dataloader
                model.eval()   # Set model to evaluate mode

            running_loss = 0.
            running_acc = 0.

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                # statistics
                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)

        global_loss.append(epoch_loss)
        global_acc.append(epoch_acc)

    x_numpy = np.arange(0.0, num_epochs, 1.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    ax1.plot(x_numpy, np.array(global_acc))
    ax1.set(title='ACCURACY')
    ax1.grid()

    ax2.plot(x_numpy, np.array(global_loss))
    ax2.set(title='LOSS')
    ax2.grid()

    plt.show()

    return model

# Главная функция обучения: Предобработка | Формирование слоёв | Запуск обучения | Сохранение модели
def main_training(epochs):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)

    batch_size = 8
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #model = EfficientNet.from_pretrained('efficientnet-b4')
    model = models.resnet50(pretrained=True)


    # Заморозка слоёв модели
    for param in model.parameters():
        param.requires_grad = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #model._fc = torch.nn.Linear(model._fc.in_features, 4)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    model = model.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_model(model, loss, optimizer, scheduler, epochs, train_dataloader, val_dataloader)

    path_to_model = 'EfficientNet_House_clf_v3.pth'
    torch.save(model.state_dict(), path_to_model)

# Добавление к словарю пути к изображению
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

# Получение предсказания модели на конкретное изображение
def get_predictions(model, input_image):
    img = input_image
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    image = img
    img = transforms.ToTensor().__call__(img)
    img = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).__call__(img)
    img = img.unsqueeze_(0).to(device)
    img_dataset = torch.utils.data.TensorDataset(img)
    img_loader = torch.utils.data.DataLoader(img_dataset, batch_size=1)
    for img in img_loader:
        imag = img[0]
        with torch.set_grad_enabled(False):
            preds = model(imag)
        prediction = torch.nn.functional.softmax(preds, dim=1).data.cpu().numpy()

        return image, prediction

if __name__ == "__main__":
    data_root = 'dataset_houses'
    train_dir = 'train'
    val_dir = 'val'
    class_names = [
        'COSMETIC_REPAIR',
        'LUXURY',
        'STANDART',
        'WITHOUT_MODIFY'
    ]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Путь к модели
    path_to_model = 'ResNet50_House_clf_v1.pth'
    # Режим Обучение / использование модели
    train_bool = False

    if train_bool:
        prepare_folders_images(train_dir, val_dir, class_names, data_root)
        main_training(60)
    else:
        #model = EfficientNet.from_pretrained('efficientnet-b4')
        #model._fc = torch.nn.Linear(model._fc.in_features, 4)
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 4)

        model.load_state_dict(torch.load(path_to_model))
        model = model.to(device)
        model.eval()

        testDir = os.listdir('test/unknown/')
        lenTestDir = len(testDir)
        pred_true_count = 0

        for i in range(lenTestDir):
            image = cv2.imread('test/unknown/' + str(testDir[i]), cv2.IMREAD_UNCHANGED)
            img, predictions = get_predictions(model, image)

            pred = predictions[0].argmax()
            print(pred)

            rt = testDir[i].split('.')[0]
            y = int(int(rt)/10 - 1)
            if y == pred:
                 pred_true_count += 1

            BoxClass = class_names[pred]

            plt.imshow(img)
            plt.title(BoxClass)
            plt.show()

            print(str(testDir[i])+'  --/--  ' + BoxClass)

        print('Accuracy: ' + str(round(pred_true_count / lenTestDir, 2)) + ' it`s: ' + str(pred_true_count) + '/' + str(lenTestDir))
'''

class ResNetAndEfficientNetClf(object):
    def __init__(self):
        self.data_root = 'dataset_houses'
        self.train_dir = 'train'
        self.val_dir = 'val'
        self.class_names = [
            'COSMETIC_REPAIR',
            'LUXURY',
            'STANDART',
            'WITHOUT_MODIFY'
        ]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Путь к модели
        self.path_to_model = 'ResNet50_House_clf_v1.pth'
        # Режим Обучение / использование модели
        self.train_bool = False

        # self.model = EfficientNet.from_pretrained('efficientnet-b4')
        # self.model._fc = torch.nn.Linear(self.model._fc.in_features, 4)
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 4)

        self.model.load_state_dict(torch.load(self.path_to_model))
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_predictions(self, model, input_image):
        img = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
        #img = input_image
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        image = img
        img = transforms.ToTensor().__call__(img)
        img = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).__call__(img)
        img = img.unsqueeze_(0).to(self.device)
        img_dataset = torch.utils.data.TensorDataset(img)
        img_loader = torch.utils.data.DataLoader(img_dataset, batch_size=1)
        for img in img_loader:
            imag = img[0]
            with torch.set_grad_enabled(False):
                preds = model(imag)
            prediction = torch.nn.functional.softmax(preds, dim=1).data.cpu().numpy()

            return image, prediction

    def main_training(self, epochs):
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = torchvision.datasets.ImageFolder(self.train_dir, train_transforms)
        val_dataset = torchvision.datasets.ImageFolder(self.val_dir, val_transforms)

        batch_size = 8
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # self.model = EfficientNet.from_pretrained('efficientnet-b4')
        self.model = models.resnet50(pretrained=True)

        # Заморозка слоёв модели
        for param in self.model.parameters():
            param.requires_grad = False

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        # model._fc = torch.nn.Linear(model._fc.in_features, 4)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 4)
        self.model = self.model.to(device)

        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), amsgrad=True, lr=1.0e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        self.train_model(self.model, loss, optimizer, scheduler, epochs, train_dataloader, val_dataloader)

        path_to_model = 'ResNet_House_clf_v3.pth'
        torch.save(self.model.state_dict(), path_to_model)

    def train_model(self, model, loss, optimizer, scheduler, num_epochs, train_dataloader, val_dataloader):
        global_loss = []
        global_acc = []

        print('Training:')
        for epoch in range(num_epochs):
            print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    dataloader = train_dataloader
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    dataloader = val_dataloader
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.
                running_acc = 0.

                # Iterate over data.
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    # forward and backward
                    with torch.set_grad_enabled(phase == 'train'):
                        preds = model(inputs)
                        loss_value = loss(preds, labels)
                        preds_class = preds.argmax(dim=1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss_value.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss_value.item()
                    running_acc += (preds_class == labels.data).float().mean()

                epoch_loss = running_loss / len(dataloader)
                epoch_acc = running_acc / len(dataloader)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)

            global_loss.append(epoch_loss)
            global_acc.append(epoch_acc)

        x_numpy = np.arange(0.0, num_epochs, 1.0)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        ax1.plot(x_numpy, np.array(global_acc))
        ax1.set(title='ACCURACY')
        ax1.grid()

        ax2.plot(x_numpy, np.array(global_loss))
        ax2.set(title='LOSS')
        ax2.grid()

        plt.show()

        return model

