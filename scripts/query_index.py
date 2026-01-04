import time

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class VectorIndexQuery:
    def __init__(self, index_path: str = '../faiss_index'):
        print(f'Загрузка индекса из {index_path}')
        self.embeddings = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        # Загружаем FAISS индекс
        self.vector_store = FAISS.load_local(
            index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f'Индекс успешно загружен')

    def search(self, query: str, k: int = 5):
        print(f"\nПоиск по запросу: '{query}'")
        print(f'Количество результатов: {k}')
        print('-' * 60)
        start_time = time.time()

        results = self.vector_store.similarity_search_with_score(query, k=k)

        search_time = time.time() - start_time
        for i, (doc, score) in enumerate(results):
            print(f'\nРезультат #{i+1} (Score: {score:.4f})')
            print(f"Файл: {doc.metadata.get('filename', 'Неизвестно')}")
            print(f"Чанк ID: {doc.metadata.get('chunk_id', 'N/A')}")
            print(f'Содержимое: {doc.page_content[:250]}...')
            print('-' * 40)

        print(f'\nВремя поиска: {search_time:.3f} секунд')
        return results


def main():
    try:
        query_system = VectorIndexQuery()
        queries = [
            'Who is z-bot-e2-e4?',
            'Which baddzedai do you know?',
            'What baddzedai places was visiting?',
        ]
        for query in queries:
            query_system.search(query, k=3)
            print('\n' + '='*60 + '\n')

    except Exception as e:
        print(f'Ошибка при выполнении поиска: {str(e)}')

if __name__ == '__main__':
    main()