from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional


embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embedding_model():
    global embeddings

    if embeddings is None:

        embeddings = HuggingFaceEmbeddings(
            model_name='intfloat/multilingual-e5-large',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder="/root/.cache/hf_models/e5_large"
        )

        print(f"Инициализация эмбеддера успешна")
    return embeddings