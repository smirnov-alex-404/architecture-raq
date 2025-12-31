import json
import os
import time
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class KnowledgeBaseIndexer:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Инициализация модели эмбеддингов: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    @staticmethod
    def load_documents(knowledge_base_path: str) -> list[Document]:
        print(f"Загрузка документов из {knowledge_base_path}")
        documents = []
        knowledge_dir = Path(knowledge_base_path)
        txt_files = list(knowledge_dir.glob("*.txt"))
        if not txt_files:
            raise ValueError(f"Не найдены .txt файлы в {knowledge_base_path}")
        print(f"Найдено файлов: {len(txt_files)}")

        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "file_path": str(file_path),
                    }
                )
                documents.append(doc)
                print(f"Загружен: {file_path.name}")
            except Exception as e:
                print(f"Ошибка при загрузке {file_path.name}: {str(e)}")

        return documents

    @staticmethod
    def split_documents(documents: list[Document]) -> list[Document]:
        print("\nРазбиение документов на чанки...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            if "position" not in chunk.metadata:
                chunk.metadata["position"] = i

        print(f"Создано чанков: {len(chunks)}")
        return chunks

    def create_faiss_index(self, chunks: list[Document]) -> FAISS:
        print("Создание FAISS индекса...")
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )
        return vector_store

    def save_index(self, vector_store: FAISS, output_path: str = "faiss_index"):
        print(f"Сохранение индекса в {output_path}")
        os.makedirs(output_path, exist_ok=True)
        vector_store.save_local(output_path)
        self.save_metadata(vector_store, output_path)
        print(f"Индекс успешно сохранен")

    @staticmethod
    def save_metadata(vector_store: FAISS, output_path: str):
        """Сохранение метаданных для отладки"""
        try:
            metadata = []
            for i in range(len(vector_store.index_to_docstore_id)):
                doc_id = vector_store.index_to_docstore_id[i]
                doc = vector_store.docstore.search(doc_id)

                if doc and hasattr(doc, 'metadata'):
                    metadata.append({
                        "chunk_id": i,
                        **doc.metadata,
                        "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    })

            metadata_path = os.path.join(output_path, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"Метаданные сохранены в {metadata_path}")

        except Exception as e:
            print(f"Ошибка при сохранении метаданных: {str(e)}")


def main():
    start_time = time.time()
    try:
        indexer = KnowledgeBaseIndexer(model_name="all-MiniLM-L6-v2")
        documents = indexer.load_documents("../knowledge_base/articles")
        chunks = indexer.split_documents(documents)
        vector_store = indexer.create_faiss_index(chunks)
        indexer.save_index(vector_store, "../faiss_index")

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"\n{'='*50}")
        print("СТАТИСТИКА:")
        print(f"Модель эмбеддингов: all-MiniLM-L6-v2")
        print(f"Количество чанков в индексе: {len(chunks)}")
        print(f"Время генерации: {processing_time:.2f} секунд")
        print("="*50)

    except Exception as e:
        print(f"Ошибка при создании индекса: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
