Полная настройка локального ИИ с браузерным интерфейсом и ML-моделями для кодирования.

***

## **1. ПОДГОТОВКА СИСТЕМЫ**

```bash
# Обновление + базовые пакеты 
sudo apt update && sudo apt upgrade -y
sudo apt install curl docker.io python3-pip zstd pciutils lshw -y

# Docker права (перелогинься после!)
sudo usermod -aG docker $USER
newgrp docker
```

***

## **2. УСТАНОВКА OLLAMA**

```bash
# Установка (авто-детект GPU/CPU, systemd сервис)
curl -fsSL https://ollama.com/install.sh | sh

# Автозапуск
sudo systemctl enable --now ollama

# Проверка
systemctl status ollama
```

**Проверка API**:
```bash
curl http://localhost:11434
# Ответ: "Ollama is running"
```

***

## **3. МОДЕЛИ ДЛЯ КОДИРОВАНИЯ**

### **3.1 Основные модели для ML/Python**
| Модель | Размер | Время загрузки | Назначение | Команда |
|--------|--------|----------------|------------|---------|
| `deepseek-coder-v2:16b` | 10 ГБ | 10-15 мин | PyTorch/YOLO/пайплайны | `ollama pull deepseek-coder-v2:16b` |
| `codeqwen:14b` | 9 ГБ | 8-12 мин | Data Science/Sklearn | `ollama pull codeqwen:14b` |
| `codestral:22b` | 13 ГБ | 12-18 мин | Длинный контекст | `ollama pull codestral:22b` |

### **3.2 Qwen3 семейство**
| Тип | Модель | Размер | Команда |
|-----|--------|--------|---------|
| Базовая | `qwen3:7b` | 4 ГБ | `ollama pull qwen3:7b` |
| Vision | `qwen3-vl:8b` | 5 ГБ | `ollama pull qwen3-vl:8b` |
| Embeddings | `qwen3-embedding:0.6b` | 0.5 ГБ | `ollama pull qwen3-embedding:0.6b` |

```bash
# Список загруженных
ollama list
```

***

## **4. WEBUI В БРАУЗЕРЕ**

```bash
# Open WebUI
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data --name open-webui \
  --restart always ghcr.io/open-webui/open-webui:main
```

**Открыть**: `http://localhost:3000`
1. Регистрация (первый запуск)  
2. Выбор модели
3. Чат-интерфейс

***

## **5. ТЕРМИНАЛ + REST API**

### **5.1 Интерактивный режим**
```bash
ollama run deepseek-coder-v2:16b
>>> Напиши код для обработки данных с pandas
>>> Найди ошибку в коде
/bye
```

### **5.2 REST API**
```bash
# Chat
curl http://localhost:11434/api/chat -d '{
  "model": "deepseek-coder-v2:16b",
  "messages": [{"role": "user", "content": "Создай ML пайплайн"}]
}'
```

***

## **6. PYTHON ИНТЕГРАЦИЯ**

```bash
pip3 install ollama
```

```python
import ollama

# Код
resp = ollama.chat(model='deepseek-coder-v2:16b', 
                  messages=[{'role': 'user', 'content': 'ML пайплайн с sklearn'}])
print(resp['message']['content'])

# Vision
resp_vlm = ollama.chat(model='qwen3-vl:8b', 
                      messages=[{'role': 'user', 'content': 'Опиши график', 'images': ['plot.png']}])

# Embeddings  
resp_emb = ollama.embeddings(model='qwen3-embedding:0.6b', prompt='ML код')
```

***

## **7. HUGGINGFACE МОДЕЛИ**

```bash
# GGUF с HF
ollama pull hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M
```

***

## **8. ПРОВЕРКА**

```bash
systemctl status ollama     # active (running)
docker ps                   # open-webui
ollama list                 # модели
curl http://localhost:11434 # API
curl -I http://localhost:3000 # WebUI
```

***
