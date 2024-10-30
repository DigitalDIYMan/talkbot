import os
import tempfile
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser

from llm_test import llm, prompt

def get_timestamp_with_underscore():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

def main():
    # 임베딩 초기화
    embeddings = HuggingFaceEmbeddings(
        model_name='BAAI/bge-m3',
        model_kwargs={'device': 'mps'},  # Apple Silicon GPU 사용
        # model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )

    # 벡터 저장소 경로 설정
    get_currenttime = get_timestamp_with_underscore()
    vectorstore_path = f'./vectorstore_{get_currenttime}/'

    # PDF 파일 경로 입력 받기
    pdf_path = input("Enter the path to your PDF file: ")

    # PDF 로드 및 처리
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            # 텍스트 분할
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(pages)

            # 벡터 저장소 생성
            os.makedirs(vectorstore_path, exist_ok=True)
            vs = Chroma.from_documents(texts, embeddings, persist_directory=vectorstore_path)
            vs.persist()
            print("Vectorstore created and persisted!")

            # Retriever 초기화
            retriever = vs.as_retriever(search_kwargs={'k': 3})

            # 대화 기록 초기화
            chat_history = []

            while True:
                # 사용자 입력 받기
                user_input = input("Enter your question (or 'quit' to exit): ")
                if user_input.lower() == 'quit':
                    break

                # 문서 검색
                retrieved_docs = retriever.get_relevant_documents(user_input)
                formatted_docs = format_docs(retrieved_docs)
                print(f"formatted_docs: {formatted_docs}")
                print(f"prompt: {prompt}")
                print(f"user_input: {user_input}")

                # RAG 체인 실행
                rag_chain = prompt | llm | StrOutputParser()
                # answer = rag_chain.invoke({'context': formatted_docs, "question": user_input})
                answer = rag_chain.invoke({'context': formatted_docs, "question": user_input, "chat_history": chat_history})


                # 답변 출력
                print("--------------------------------------------------------------------------------------------------------")
                print("AI: ", answer)

                # 대화 기록 업데이트
                chat_history.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": answer}
                ])

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()