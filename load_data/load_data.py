#!/usr/bin/env python
# coding: utf-8

# # Module A для табличных данных

# ### Загрузка CSV

# In[1]:


import pandas as pd

def load_csv(file_path, chunksize=None):
    """
    Загружает CSV файл.
    Если chunksize указан, возвращает генератор DataFrame для ленивой загрузки.
    """
    if chunksize:
        return pd.read_csv(file_path, chunksize=chunksize)
    else:
        return pd.read_csv(file_path)

# Пример:
# df = load_csv("data.csv")
# для больших файлов (возвращает список из DataFrame'ов): 
df_iter = load_csv("data.csv", chunksize=1000)

[i for i in df_iter][0].shape


# 
# ### Загрузка Excel

# In[2]:


import pandas as pd

def load_excel(file_path, sheet_name=0):
    """
    Загружает Excel файл (xls/xlsx).
    sheet_name=0 загружает первый лист, можно указать название листа.
    """
    return pd.read_excel(file_path, sheet_name=sheet_name)

# Пример:
df = load_excel("excel_file.xlsx")
df.sample(2)


# ### Загрузка TXT (как таблицу, разделитель по табуляции или другой)

# In[3]:


import pandas as pd

def load_txt(file_path, sep=',', chunksize=None):
    """
    Загружает TXT файл как таблицу.
    sep — разделитель (по умолчанию табуляция).
    chunksize — для ленивой загрузки больших файлов.
    """
    if chunksize:
        return pd.read_csv(file_path, sep=sep, chunksize=chunksize)
    else:
        return pd.read_csv(file_path, sep=sep)

# Пример:
df = load_txt("data.txt")
# для больших файлов: df_iter = load_txt("data.txt", chunksize=10000)
df.sample(3)


# ### Загрузка Parquet

# In[4]:


import pandas as pd

def load_parquet(file_path):
    """
    Загружает Parquet файл.
    """
    return pd.read_parquet(file_path)

# Пример:
df = load_parquet("data.parquet")
df.sample(3)


# ### Загрузка бинарных файлов

# In[5]:


import struct
import numpy as np

def load_bin_image(path):
    """
    Загружает бинарный файл с изображением в numpy.ndarray
    Формат:
    - magic (4 байта)
    - array_length (uint32)
    - version (uint8)
    - data_type_code (uint8)
    - reserved (2 байта)
    - data (array_length * dtype)
    """
    with open(path, 'rb') as file:
        # Заголовок
        magic = file.read(4)
        array_length = struct.unpack('<I', file.read(4))[0]
        version = struct.unpack('<B', file.read(1))[0]
        data_type_code = struct.unpack('<B', file.read(1))[0]
        reserved = file.read(2)

        # Таблица типов данных
        data_types = {
            0: np.uint8,
            1: np.float32,
            2: np.int32
        }

        if data_type_code not in data_types:
            raise ValueError(f"Неизвестный data_type_code: {data_type_code}")

        dtype = data_types[data_type_code]
        element_size = np.dtype(dtype).itemsize

        # Чтение данных (без лишних копий)
        raw = file.read(array_length * element_size)
        data = np.frombuffer(raw, dtype=dtype)

        # Восстановление формы (квадратное изображение)
        side = int(array_length ** 0.5)
        image = data.reshape((side, side))

    return image


# In[6]:


load_bin_image('data.mybin').shape


# При работе с бинарным файлом любого типа (изображение, таблица, аудио, временной ряд и т.д.) в первую очередь необходимо строго определить и описать его формат: структуру заголовка (сигнатура, версия, размеры, тип данных, служебные байты), порядок байтов (endianness) и способ хранения данных; затем файл следует читать последовательно в бинарном режиме (rb), извлекая метаданные через struct.unpack, после чего основные данные загружать в подходящую структуру без лишних копий (например, numpy.frombuffer или numpy.memmap при ограниченной оперативной памяти), приводя их к нужной форме (shape) и типу (dtype), а уже на этом уровне интерпретировать содержимое как изображение, таблицу (DataFrame), аудиосигнал или другой объект для дальнейшего анализа, визуализации или обучения моделей.

# ### Загрузка изображений (PIL)

# In[7]:


from PIL import Image
import os
import glob

def load_images(folder_path, extensions=None):
    """
    Загружает изображения из папки как PIL.Image
    """

    images = []
    for path in os.listdir(folder_path):
        img = Image.open(folder_path + path)
        images.append(img)

    return images

# Пример:
images = load_images("images/")
images[3]


# Если файлов очень много — лучше хранить пути, а не сами изображения.

# ### Загрузка аудио

# In[8]:


import librosa
import os
import glob

def load_audio(folder_path):
    """
    Загружает аудиофайлы с помощью librosa
    """

    audio_data = []
    for path in os.listdir(folder_path):
        y, sr = librosa.load(folder_path + path)
        audio_data.append({
            "path": path,
            "signal": y,
            "sr": sr
        })

    return audio_data

# Пример:
audios = load_audio("audio/")
audios[2]


# ### Загрузка видео

# In[13]:


import cv2
import os
import glob

def load_videos(folder_path):
    """
    Загружает видео как cv2.VideoCapture (ленивое чтение)
    """

    videos = []
    for path in os.listdir(folder_path):
        cap = cv2.VideoCapture(folder_path + path)
        videos.append({
            "path": path,
            "capture": cap
        })

    return videos


videos = load_videos('videos/')
videos[1]


# Работа с видео сводится к обработке каждого кадра, также как к изображениям.

# In[14]:


videos[1]['capture'].read()

