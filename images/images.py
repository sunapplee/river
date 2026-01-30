#!/usr/bin/env python
# coding: utf-8

#  # 🔹 1) начальная предобработка для нейросетей

# 1. импорты библиотек

# In[1]:


import os
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# 2. приведем картинки к одному виду и формату тензора

# In[2]:


from torchvision import transforms

basic_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

from torchvision import datasets

train_ds = datasets.ImageFolder("data_fruits/train", transform=basic_tfms)
val_ds   = datasets.ImageFolder("data_fruits/val",   transform=basic_tfms)
test_ds  = datasets.ImageFolder("data_fruits/test",  transform=basic_tfms)


# 3. сощдаем ImageFolder, который нужен, чтобы не писать свой код обхода папок
# ImageFolder:
# - смотрит на подпапки в train/val/test,
# - считает каждую подпапку отдельным классом,
# - каждому изображению даёт номер класса

# In[3]:


from torchvision import datasets

train_ds = datasets.ImageFolder(root="data_fruits/train", transform=basic_tfms)
val_ds   = datasets.ImageFolder(root="data_fruits/val",   transform=basic_tfms)
test_ds  = datasets.ImageFolder(root="data_fruits/test",  transform=basic_tfms)


# 4. DataLoader

# In[4]:


from torch.utils.data import DataLoader

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=32, shuffle=False)
test_dl  = DataLoader(test_ds, batch_size=32, shuffle=False)


# # 🔹 2) Классический ML для изображений
# 
# 1 вариант 
# 1. Превратить картинку в числовой вектор признаков.
# т.е
#     - переводим картинку в RGB;
#     - уменьшаем размер (для скорости);
#     - считаем гистограмму по каждому каналу (R, G, B);
#     - соединяем три гистограммы в один вектор.
# 2. Обучить классический алгоритм (например, SVM) на этих признаках.
# 3. Получить метрики качества (accuracy, precision, recall, F1).
# 
# 2 вариант
# просто больше размер + flatten

# 1 вариант

# In[5]:


class_names = sorted([p.name for p in Path("data_fruits/train").iterdir() if p.is_dir()])

def img_features1(path, bins=16):
    """
    Из одной картинки делаем вектор признаков.
    - переводим картинку в RGB;
    - уменьшаем размер (для скорости);
    - считаем гистограмму по каждому каналу (R, G, B);
    - соединяем три гистограммы в один вектор.
    - добавляем среднее и стандартное отклонение по каждому каналу (ещё 6 чисел)
    """
    # 1) RGB
    img = Image.open(path).convert("RGB")

    # 2) уменьшаем размер
    img = img.resize((128, 128))

    # 3) гистограмма по каналам
    arr = np.array(img)
    hist_list = []
    for ch in range(3): 
        channel = arr[..., ch]
        hist, _ = np.histogram(
            channel,
            bins=bins,
            range=(0, 256),
            density=True
        )
        hist_list.append(hist)


    means = arr.mean(axis=(0, 1))
    stds  = arr.std(axis=(0, 1))

    # 4) один общий вектор
    features = np.concatenate(hist_list + [means, stds])
    return features


# 2 вариант

# In[6]:


class_names = sorted([p.name for p in Path("data_fruits/train").iterdir() if p.is_dir()])
train_dir = Path("data_fruits/train")

def img_features2(path):
    """
    Признаки из картинки:
    - RGB
    - уменьшаем до 32x32
    - расплющиваем в один вектор (32*32*3 = 3072 числа)
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((32, 32))
    arr = np.array(img) / 255.0
    return arr.flatten()


# делаем матрицы x и y для 2 варианта

# In[7]:


X, y = [], []

for cls_idx, cls in enumerate(class_names):
    folder = train_dir / cls
    for name in os.listdir(folder):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            X.append(img_features2(folder / name))
            y.append(cls_idx)

X = np.array(X)
y = np.array(y)


# делаем матрицы x и y для 1 варианта

# In[8]:


X, y = [], []

for cls_idx, cls in enumerate(class_names):
    folder = Path("data_fruits/train") / cls
    for name in os.listdir(folder):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            path = folder / name
            X.append(img_features1(path))  # признаки
            y.append(cls_idx)  # номер класса

X = np.array(X)
y = np.array(y)


# делим на train и val

# In[9]:


from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# обучим несколько моделей классификации

# 1. логистичская регрессия

# In[ ]:


from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_tr, y_tr)

y_pred_lr = log_reg.predict(X_val)
print(classification_report(y_val, y_pred_lr, target_names=class_names))


# результаты не очень тк данных очень мало

# 2. SVM

# In[ ]:


from sklearn.svm import SVC

svm_clf = SVC(kernel="rbf", C=5, gamma="scale")
svm_clf.fit(X_tr, y_tr)

y_pred_svm = svm_clf.predict(X_val)
print(classification_report(y_val, y_pred_svm, target_names=class_names))


# 3. random forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_tr, y_tr)

y_pred_rf = rf_clf.predict(X_val)
print(classification_report(y_val, y_pred_rf, target_names=class_names))


# 🔹 3) Нейронные сети 
# 
# на примере ResNet18

# - выбираем устройство (`device`);
# - считаем количество классов (`num_classes`);
# - импортируем нужные библиотеки.

# In[ ]:


import torch
from torch import nn, optim
from torchvision import models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# device: куда отправляем модель и данные (cuda, если есть GPU, иначе cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# num_classes: сколько у нас разных фруктов (классов)
num_classes = len(class_names)


# Создаём ResNet18 (можно использовать также ResNet50 / ResNet101 )

# In[16]:


# создаём ResNet18 без предобученных весов
resnet = models.resnet18(weights=None)

#resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) - обучение с весами(мой ноут просто не тянет)


# Меняем последний слой под наши классы
# 
# - `resnet.fc` — последний полносвязный слой.
# - У него есть входная размерность `in_features`.
# - Заменяем его на `nn.Linear(in_features, num_classes)`, чтобы модель могла предсказывать наши классы.

# In[17]:


in_features = resnet.fc.in_features
resnet.fc = nn.Linear(in_features, num_classes)


# Перенос модели и настройка обучения
# 
# - Переносим модель на `device`.
# - Задаём:
#   - `criterion` — функцию потерь (CrossEntropyLoss для многоклассовой задачи).
#   - `optimizer_resnet` — оптимизатор Adam с маленьким шагом обучения.

# In[18]:


# переносим модель на GPU/CPU
resnet = resnet.to(device)

# функция потерь: сравнивает логиты и правильные метки
criterion = nn.CrossEntropyLoss()

# оптимизатор: Adam, обучаем все параметры модели
optimizer_resnet = optim.Adam(
    resnet.parameters(),
    lr=1e-4,
    weight_decay=1e-4,   # это L2‑регуляризация, она штрафует слишком большие веса и помогает бороться с переобучением
)



# обучаем эпоху
# На вход:
# - `model` — наша ResNet18;
# - `loader` — train_dl (батчи картинок и меток);
# - `optimizer` — объект Adam.
# 
# Процесс:
# 1. Переводим модель в режим обучения `model.train()`.
# 2. Для каждого батча:
#    - переносим данные на `device`;
#    - обнуляем старые градиенты;
#    - считаем предсказания (`logits`);
#    - считаем loss;
#    - считаем градиенты (`backward`);
#    - делаем шаг оптимизатора (`step`);
#    - копим loss и количество верных ответов.
# 3. Возвращаем средний loss и accuracy по всей эпохе

# In[19]:


def train_one_epoch(model, loader, optimizer):
    model.train()     

    total_loss = 0.0             
    total_correct = 0        
    total = 0                        

    for images, labels in loader:
        # переносим картинки и метки на устройство
        images = images.to(device)
        labels = labels.to(device)

        # шаг 1: обнуляем градиенты
        optimizer.zero_grad()

        # шаг 2: прямой проход (forward)
        logits = model(images)

        # шаг 3: считаем функцию потерь
        loss = criterion(logits, labels)

        # шаг 4: обратное распространение ошибки
        loss.backward()

        # шаг 5: обновляем веса
        optimizer.step()

        # считаем статистику по батчу
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        # предсказанный класс = индекс максимального логита
        preds = logits.argmax(1)
        total_correct += (preds == labels).sum().item()
        total += batch_size

    # средний loss и accuracy по эпохе
    avg_loss = total_loss / total
    avg_acc = total_correct / total
    return avg_loss, avg_acc


# оценка
# На вход:
# - `model` — ResNet18;
# - `loader` — val_dl или test_dl.
# 
# Процесс:
# 1. Переводим модель в режим оценки `model.eval()`.
# 2. Отключаем градиенты `torch.no_grad()`.
# 3. Для каждого батча:
#    - считаем loss и предсказания;
#    - копим loss и количество верных ответов;
#    - сохраняем все метки и все предсказания.
# 4. Возвращаем:
#    - средний loss,
#    - accuracy,
#    - массив всех истинных меток `y_true`,
#    - массив всех предсказаний `y_pred`

# In[20]:


def evaluate(model, loader):
    model.eval()      
    total_loss = 0.0
    total_correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad(): 
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            preds = logits.argmax(1)
            total_correct += (preds == labels).sum().item()
            total += batch_size

            # сохраняем для отчётов
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    avg_loss = total_loss / total
    avg_acc = total_correct / total
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    return avg_loss, avg_acc, y_true, y_pred


# Обучение ResNet18 по эпохам
# 
# - `num_epochs` — сколько раз проходим по train_dl.
# - На каждой эпохе:
#   - считаем `train_loss`, `train_acc` на train_dl;
#   - считаем `val_loss`, `val_acc` на val_dl;
#   - печатаем результаты для контроля.

# In[21]:


num_epochs = 5  # можно увеличить, если нужно лучшее качество

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(resnet, train_dl, optimizer_resnet)
    val_loss, val_acc, _, _ = evaluate(resnet, val_dl)

    print(
        f"[ResNet] Epoch {epoch+1}/{num_epochs} | "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
        f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
    )


# Финальная оценка на тестовом наборе
# 
# - считаем `test_loss` и `test_acc` на `test_dl`;
# - строим `classification_report` (precision, recall, F1 по каждому классу);
# - получаем матрицу ошибок `confusion_matrix`.

# In[25]:


test_loss, test_acc, y_true_res, y_pred_res = evaluate(resnet, test_dl)

print("ResNet18 test_loss:", test_loss)
print("ResNet18 test_acc :", test_acc)

print(classification_report(
    y_true_res,
    y_pred_res,
    target_names=class_names,
    zero_division=0
))

cm_resnet = confusion_matrix(y_true_res, y_pred_res)
cm_resnet


# плохие результаты тк нет предобученных весов

# # Также можно использовать  EfficientNet-B0 или Vision Transformer ViT-B/16

# # 🔹 Метрики 

# In[26]:


from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# accuracy 
acc = accuracy_score(y_true_res, y_pred_res)
print("Accuracy:", acc)

# macro precision/recall/F1
prec = precision_score(y_true_res, y_pred_res, average="macro", zero_division=0)
rec  = recall_score(y_true_res, y_pred_res, average="macro", zero_division=0)
f1   = f1_score(y_true_res, y_pred_res, average="macro", zero_division=0)

print("Macro precision:", prec)
print("Macro recall   :", rec)
print("Macro F1       :", f1)

# подробный отчёт по каждому классу
print(classification_report(y_true_res, y_pred_res, target_names=class_names, zero_division=0))


# - accuracy_score даёт общую долю правильных ответов.
# 
# - precision_score/recall_score/f1_score с average="macro" дают среднее по классам.
# 
# - Для **accuracy, precision, recall, F1**: всегда **чем выше, тем лучше** 

# # Сохранение модели классификации фруктов

# In[24]:


torch.save(resnet.state_dict(), 'fruit_model.pth')


# ###

# # Функция для дообучения
# 
# Часто на чемпионатах просят организовать непрерывное обучение моделей с помощью AirFlow. 
# 
# Создадим функцию для дообучения моделей и добавления новых классов, которую можно будет перенести в Airflow или интерфейс в будущем.

# In[28]:


def fine_tuning_fruit(new_data: DataLoader) -> None:
    # Загружаем модель, переносим на gpu при наличии
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 9)
    model.load_state_dict(torch.load("fruit_model.pth"))
    model.to(device)

    # Создаем оптимизатор с параметрами модели
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Берем уже написанную функцию
    for epoch in range(5):
        loss, acc = train_one_epoch(model, new_data, optimizer)
        print(f"{epoch}: loss={loss:.4f}, acc={acc:.4f}")


    # Оценка качества
    test_ds  = datasets.ImageFolder("data_fruits/test",  transform=basic_tfms)
    test_dl  = DataLoader(test_ds, batch_size=32, shuffle=False)

    test_loss, test_acc, y_true_res, y_pred_res = evaluate(model, test_dl)

    prec = precision_score(y_true_res, y_pred_res, average="macro", zero_division=0)
    rec  = recall_score(y_true_res, y_pred_res, average="macro", zero_division=0)
    f1   = f1_score(y_true_res, y_pred_res, average="macro", zero_division=0)
    acc = accuracy_score(y_true_res, y_pred_res)
    metrics = {
    "accuracy": acc,
    "macro_precision": prec,
    "macro_recall": rec,
    "macro_f1": f1
    }

    # Сохраняем обновлённую модель
    torch.save(model.state_dict(), "fruit_model.pth")

    return metrics




# ## Проверяем функцию

# In[29]:


from torch.utils.data import Dataset

class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# In[30]:


new_images = ["coconut.jpg", ]
new_labels = [3, ]

dataset = SimpleImageDataset(new_images, new_labels, transform=basic_tfms)
fine_tuning_loader = DataLoader(dataset, batch_size=4, shuffle=True)


# In[31]:


fine_tuning_fruit(fine_tuning_loader)


# #### Данный способ подходит для простого дообучения модели, новый класс таким образом не добавить. Его нужно использовать в простых случаях непрерывного обучения.
# 
# #### Но если нужно добавить новый класс, то нужно добавлять данные к основному набору и переобучать модель ResNet .

# ###

# # Функция добавления нового класса.

# Важно! В папке data_fruits добавили новый таргет coconut, куда добавили наше новое изображение.

# In[32]:


get_ipython().system('ls data_fruits/train/')


# In[40]:


def retrain_model(num_epochs=5):
    basic_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder("data_fruits/train", transform=basic_tfms)
    val_ds   = datasets.ImageFolder("data_fruits/val",   transform=basic_tfms)
    test_ds  = datasets.ImageFolder("data_fruits/test",  transform=basic_tfms)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_dl  = DataLoader(test_ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = sorted([p.name for p in Path("data_fruits/train").iterdir() if p.is_dir()])
    num_classes = len(class_names)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # переносим модель на GPU/CPU
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_dl, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_dl)

        print(
            f"[ResNet] Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

    test_loss, test_acc, y_true_res, y_pred_res = evaluate(model, test_dl)

    prec = precision_score(y_true_res, y_pred_res, average="macro", zero_division=0)
    rec  = recall_score(y_true_res, y_pred_res, average="macro", zero_division=0)
    f1   = f1_score(y_true_res, y_pred_res, average="macro", zero_division=0)
    acc = accuracy_score(y_true_res, y_pred_res)
    metrics = {
    "accuracy": acc,
    "macro_precision": prec,
    "macro_recall": rec,
    "macro_f1": f1
    }

    report = classification_report(
    y_true_res,
    y_pred_res,
    target_names=class_names,
    zero_division=0
    )

    torch.save(model.state_dict(), "fruit_model.pth")

    return metrics, report


# In[41]:


metrics, report = retrain_model(num_epochs=3)

print(metrics)


# In[42]:


print(report)


# Как видим из вывода, у нас добавился новый класс coconut fruit.
