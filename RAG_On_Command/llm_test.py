from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

# Ollama 모델 초기화
llm = ChatOllama(model="llama3-instruct-8b:latest", temperature=0.0)

# 프롬프트 템플릿 정의
template = """
You are a helpful AI assistant. Please answer the user's question based on the given context.

Context: {context}

Conversation History:
{chat_history}

User: {question}
AI Assistant: """

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    # input_variables=["context", "question"],
    template=template
)