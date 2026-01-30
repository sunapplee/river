# 🔹 1. подтягивание файлов из git через curl wget

## Что должно стоять
```bash
sudo apt update && sudo apt install -y wget curl unzip git jupyter
```
## Создать структуру папок
```bash
mkdir -p ~/files
cd ~/files
pwd
```

## Скачать файл
```bash
# Вариант 1: wget
wget https://raw.githubusercontent.com/sunapplee/river/main/requirements/scripts.ipynb

# Вариант 2: curl
curl -L -O https://raw.githubusercontent.com/sunapplee/river/main/requirements/scripts.ipynb

# Вариант 3: curl и запуск в консоли
curl https://raw.githubusercontent.com/sunapplee/river/main/requirements/scripts.md
```

*Расположение файлов доступно в ```content.md```*

## Проверить результат
```bash
ls -la docker.ipynb
file docker.ipynb
```

## Запустить Jupyter
```bash
jupyter notebook docker.ipynb
```


## Чистка
```bash
rm -f docker.ipynb
```

# 🔹 Установка день 0

# 0. Базовая информация и проверки

## Проверка ОС и системных ресурсов

```bash
uname -a
lsb_release -a  # если есть
df -h /        # свободное место на корневом разделе
free -h        # RAM
nvidia-smi     # если есть GPU NVIDIA
```

## Проверка интернета

```bash
ping -c 3 google.com
```

***

# 1. Обновление системы

```bash
sudo apt update
sudo apt upgrade -y
```

***

# 2. Базовые пакеты

```bash
sudo apt install -y \
  curl \
  wget \
  git \
  build-essential \
  ca-certificates \
  software-properties-common \
  gnupg \
  unzip \
  htop \
  tree \
  ffmpeg
```

***

# 3. Установка Python

## Проверка версии

```bash
python3 --version
```

## Установка Python3 (если не установлен)

```bash
sudo apt install -y python3 python3-pip python3-venv
```

## Проверка установки

```bash
python3 --version
pip3 --version
```

***

# 4. Установка VS Code

## Добавление репозитория Microsoft

```bash
wget -qO- https://packages.microsoft.com/keys/microsoft.asc \
  | sudo gpg --dearmor \
  > /usr/share/keyrings/microsoft-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/microsoft-archive-keyring.gpg] https://packages.microsoft.com/repos/code stable main" \
  | sudo tee /etc/apt/sources.list.d/vscode.list
```

## Установка

```bash
sudo apt update
sudo apt install -y code
```

## Запуск

```bash
code
```

***

# 5. Расширения VS Code

В VS Code откройте панель расширений:

- `Ctrl+Shift+X` → установить:
  - Python (Microsoft)
  - Pylance
  - Jupyter
  - Jupyter Notebook Renderers
  - Docker (Microsoft)
  - YAML

После установки перезапустите VS Code.

***

# 6. Структура проекта и репозиторий

## Создание корневой папки

```bash
mkdir -p ~/Rea
cd ~/Rea
```

## Инициализация Git

```bash
# Проверяем, что мы в Rea
pwd

git init
git remote add origin https://github.com/твойusername/Rea.git
git add .
git commit -m "initial commit"
git branch -M main
git push -u origin main
```

***

# 7. Общее Python-окружение `rea`

## Создание окружения

```bash
cd ~/Rea
python3 -m venv rea
source rea/bin/activate
```

## Проверка

```bash
python -V
pip -V
```

## Обновление pip

```bash
pip install --upgrade pip
```

***

# 8. Jupyter и ядро для VS Code

## Установка Jupyter

```bash
pip install jupyter notebook ipykernel
```

## Создание ядра для VS Code

```bash
python -m ipykernel install --user --name=rea --display-name "Python (rea)"
```

***

# 9. Общее окружение `rea`

Базовое окружение для ML, геоданных, CV, web-разработки.

```bash
cd ~/Rea
source rea/bin/activate
pip install -r requirements-general.txt
```

**Зависимости:** [requirements-general.txt](requirements-general.txt)

***

# 10. Окружение PyTorch

Для проектов с deep learning (CUDA, vision, метрики).

```bash
cd ~/Rea
python3 -m venv pytorch_env
source pytorch_env/bin/activate
pip install --upgrade pip

# Установка PyTorch (выберите версию CUDA на pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Дополнительные пакеты
pip install -r requirements-torch.txt
```

**Зависимости:** [requirements-torch.txt](requirements-torch.txt)

***

# 11. Окружение Unsloth

Для дообучения LLM моделей (требует PyTorch + CUDA).

```bash
cd ~/Rea
python3 -m venv unsloth_env
source unsloth_env/bin/activate
pip install --upgrade pip

# Сначала PyTorch (см. раздел 10)
# Затем следуйте инструкциям в файле зависимостей
```

**Зависимости:** [requirements-unsloth.txt](requirements-unsloth.txt)

***

# 12. Использование окружений

```bash
# Общее окружение
cd ~/Rea
source rea/bin/activate

# PyTorch
source ~/Rea/pytorch_env/bin/activate

# Unsloth
source ~/Rea/unsloth_env/bin/activate

# Деактивация
deactivate
```


# 13. Установка Docker

## Удаление старых версий

```bash
sudo apt remove -y docker docker-engine docker.io containerd runc
```

## Добавление репозитория Docker

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

## Установка Docker

```bash
sudo apt update
sudo apt install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin
```

## Запуск и автозапуск

```bash
sudo systemctl enable docker
sudo systemctl start docker
```

## Разрешить запуск без sudo

```bash
sudo usermod -aG docker $USER
# затем перелогиниться или перезайти в сессию
```

## Проверка

```bash
docker run hello-world
```

***

# 14. Работа с VS Code и Jupyter

## Открыть проект

```bash
cd ~/Rea
source rea/bin/activate
code .
```

## Создать ноутбук и выбрать ядро

- `Ctrl+Shift+P`
- `Jupyter: Create New Blank Notebook`
- В правом верхнем углу выбрать Kernel → `Python (rea)`

***

# 15. Установка Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Проверка

```bash
ollama --version
```

***