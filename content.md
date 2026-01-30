### requirements
- requirements/scripts.md
  — Пошаговая установка окружения: система, Python, VS Code, Docker, Ollama
  — Создание виртуальных окружений: general, PyTorch, Unsloth
  — Работа с Jupyter в VS Code

- requirements/scripts.ipynb
  — Jupyter-версия инструкций по установке

- requirements/requirements-general.txt
  — Зависимости для общего окружения (ML, геоданные, CV, web-разработка)

- requirements/requirements-torch.txt
  — Зависимости для PyTorch (deep learning, CUDA, vision, метрики)

- requirements/requirements-unsloth.txt
  — Зависимости для дообучения LLM моделей (Unsloth + LoRA)

- requirements/scripts.ipynb
  — Jupyter-версия инструкций по установке



### docker
- docker/docker.md
  — Работа с Docker
  — Работа с scp



### ollama
- ollama/ollama_rag.md
  — Пошаговая сборка RAG-пайплайна на LangChain

- ollama/ollama.md
  — Установка Ollama, управление моделями
  — Запуск текстовых, VLM и embedding-моделей
  — Поддержка моделей из HuggingFace в GGUF формате

- ollama/Modelfile
  — Конфигурация для сборки собственной модели в формате Ollama

- ollama/ollama_finetuning.py
  — Подготовка данных и дообучение Qwen3 (4-bit) с помощью Unsloth + LoRA



### images
- images/image.py
  — Загрузка датасета: по папкам или по меткам в имени
  — Предобработка изображений (resize, normalize, RGB)
  — Балансировка классов через WeightedRandomSampler
  — Аугментации: flips, rotation, jitter, erasing, grayscale; расширение датасета
  — Train/test split со стратификацией
  — Обучение моделей: RandomForest, LogisticRegression, YOLOv8-cls

- images/video.py
  — Извлечение кадров из видео
  — Классификация через YOLO-CLS
  — Аугментации: grayscale → RGB, color jitter, cutout

- images/images.py
  — Подготовка данных для нейросетей (ImageFolder, трансформации)
  — Ручные признаки: гистограммы / уменьшение + flatten
  — Дообучение ResNet18
  — Fine-tuning и полное переобучение модели



### load_data
- load_data/load_data.py
  — Загрузка CSV, TXT, Excel, включая ленивую обработку через chunksize
  — Загрузка бинарных форматов с кастомной структурой (magic, headers, dtype)



### text
- text/text.py
  — Преобразование текста в числовые признаки (TF-IDF)



### table
- table/table.py
  — Сохранение таблиц в БД SQLite (to_sql)
  — Хранение нереляционных данных через shelve
  — Боксплоты для визуализации
  — Анализ плотности, нормальности (Шапиро-Уилк), skew/kurtosis

- table/table_clustering.py
  — Определение оптимального числа кластеров (метод локтя)
  — Кластеризация: KMeans, HDBSCAN, BIRCH
  — PCA-визуализация кластеров
  — Метрики качества кластеризации (Silhouette, CH, DB)

- table/table_dashboard.py
  — Интерактивный Streamlit-дашборд
  — Фильтры по колонкам и диапазону дат
  — Метрики и визуализация загрязнителей (линии, бары, scatter)

- table/table_feature_importance.py
  — chunksize для больших данных
  — Анализ признаков и распределений
  — Корреляционная матрица
  — SHAP-анализ
  — Permutation Importance
  — Оптимизация типов данных + сохранение в parquet (макс. сжатие)

- table/table_time_series.py
  — Интерполяция пропусков
  — STL-декомпозиция: тренд, сезонность, остаток
  — Прогнозирование: простая ML-модель + SARIMAX

- table/table3.py
  — Подбор гиперпараметров (RandomizedSearchCV)
  — Балансировка классов через SMOTE
  — ROC-AUC для мульткласса
  — Сохранение моделей (CatBoost, MLP, scaler)
  — Fine-tuning регрессионных моделей



### audio
- audio/audio.py
  — Вычисление аудио-статистик (длительность, громкость, вариативность)
  — Визуализация: временной сигнал, STFT-спектрограмма, Mel-спектрограмма
  — Извлечение признаков: MFCC, chroma, centroid, bandwidth, rolloff (mean/std)

- audio/audio2.py
  — Визуализация waveform
  — Разрезание аудио на сегменты + фильтрация тишины
  — Формирование датасета + train/val/test
  Представления аудио:
  — Waveform
  — Mel-спектрограммы (2D)
  — MFCC признаки
  Дополнительно:
  — Обучение Logistic Regression на MFCC
  — Корреляции MFCC и гистограммы по классам

- audio/audio3.py
  — Извлечение признаков: MFCC
  — Подбор гиперпараметров (GridSearchCV)
  — Fine-tuning аудио-классификатора



### geo
- geo/geo_gpx.py
  — Работа с GPX-треками: точки, LineString, проекции
  — Визуализация на OSM + генерация изображений
  — Извлечение окружения через OSMnx

- geo/geo_shp_tif.py
  — Анализ векторных геоданных (SHP): породы, полигоны, визуализация
  — Обработка растров (TIF): чтение, выравнивание, аугментации
  — Интерактивные карты (OSM, спутник)

- geo/geo_image.py
  — Чтение HDF5 спутниковых снимков
  — Чтение band_stats, label_map, partition
  — Расчёт индексов NDVI и NDMI + визуализация
  — Формирование набора: 13 каналов + NDVI + NDMI = 15 каналов
  — PyTorch Dataset + DataLoader

- geo/geo.py
  — Карта с предсказаниями (геовизуализация классификации)



### API
- API/api.py
  — REST-сервер на FastAPI, принимающий параметры и файлы, вызывающий функции инференса/дообучения и отдающий ответы клиентам
- API/api_testing.py
  — Скрипт с тестовыми запросами к FastAPI-серверу, отправляемыми через requests для проверки всех эндпоинтов
- API/api_docs.md
  — Документация REST API для машинного обучения



### telegram_bot
- telegram_bot/tg_bot.py
  — Телеграм-бот на aiogram, принимающий фото и текстовые команды и возвращающий результаты классификации через удобный чат-интерфейс



### streamlit
- streamlit/streamlit.py
  — Графический веб-интерфейс на Streamlit для отправки данных и файлов в API, отображения результатов и интерактивной визуализации геоданных



### docs
- docs/instruction.docx
  — Руководство по эксплуатации
- docs/presentation.pptx
  — Презентация
- docs/api_docs.md
  — Документация REST API для машинного обучения
