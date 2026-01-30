from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


# Загружаем исходный текст
with open('text.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

# Оборачиваем текст в формат документа LangChain
documents = [Document(page_content=raw_text)]

# Дробим текст на чанки — важно для качественных эмбеддингов и поиска
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Создаём эмбеддинги (через Ollama) и векторное хранилище FAISS
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
vectorstore = FAISS.from_documents(texts, embeddings)

# Настраиваем retriever для поиска релевантных фрагментов
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM для генерации ответа
llm = OllamaLLM(model="qwen3:4b", temperature=0)

# Промпт в стиле RAG: отвечать строго по контексту
template = """Ответь на вопрос, используя только предоставленный контекст. Если ответ не найден в контексте,
скажи "Ответ не найден в документах.".
Контекст:\n{context}
Вопрос:\n{question}
Ответ:"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Пример запроса
question = "Ваш запрос"

# Ищем релевантные документы
docs = retriever.invoke(question)
context = "\n".join([doc.page_content for doc in docs])

# Формируем финальный промпт для LLM
formatted_prompt = prompt.format(context=context, question=question)

# Генерируем ответ модели
response = llm.invoke(formatted_prompt)

print(response)