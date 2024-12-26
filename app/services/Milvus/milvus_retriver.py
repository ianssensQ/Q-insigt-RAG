from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    WeightedRanker,
    RRFRanker
)

import nltk
from nltk.corpus import stopwords

# Загружаем стоп-слова для русского языка
# nltk.download("stopwords")
nltk.data.path.append("/usr/local/share/nltk_data")

russian_stopwords = stopwords.words("russian")

# Класс для гибридного поиска в milvus
class HybridRetriever:
    def __init__(self, uri, collection_name="Messages", dense_embedding_function=None):
        self.uri = uri
        self.collection_name = collection_name
        self.embedding_function = dense_embedding_function
        self.client = MilvusClient(uri=uri)

    def build_collection(self, recreation=False):
        dense_dim = len(self.embedding_function.embed_query('test'))

        if self.client.has_collection(collection_name=self.collection_name) and recreation:
            self.client.drop_collection(collection_name=self.collection_name)
            print("Коллекция уже есть и будет пересоздана")
        elif self.client.has_collection(collection_name=self.collection_name) and not recreation:
            print("Коллекция уже есть и не пересоздаётся")
            return "Collection already exists"

        tokenizer_params = {
            "tokenizer": "standard",
            "filter": [
                {"type": "stemmer", "language": "russian"},
                {
                    "type": "stop",
                    "stop_words": russian_stopwords,
                },
            ],
        }

        schema = MilvusClient.create_schema()
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )
        schema.add_field(
            field_name="message",
            datatype=DataType.VARCHAR,
            max_length=20000,
            analyzer_params=tokenizer_params,
            enable_match=True,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )
        schema.add_field(
            field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim
        )
        schema.add_field(
            field_name="chat_message_id",
            datatype=DataType.INT64
        )
        schema.add_field(
            field_name="chat_name", datatype=DataType.VARCHAR, max_length=200
        )

        functions = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["message"],
            output_field_names="sparse_vector",
        )

        schema.add_function(functions)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        index_params.add_index(
            field_name="dense_vector", index_type="FLAT", metric_type="IP"
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        print("Создали новую коллекцию  в Milvus")

    def insert_data(self, metadata):
        embedding = self.embedding_function.embed_documents([metadata['message']])
        dense_vec = embedding[0]
        self.client.insert(
            self.collection_name, {"dense_vector": dense_vec, **metadata}
        )

    async def search(self, query: str, chat_names: list, k: int = 20, mode="hybrid", weights=[0.5, 0.5], k_rerank=100):

        output_fields = [
            "message",
            "chat_name",
        ]
        # filter_expression = f"chat_name in [{", ".join(f'\"{name}\"' for name in chat_names)}]"
        # filter_expression = f"chat_name in [{', '.join(f'\"{name}\"' for name in chat_names)}]"
        # filter_expression = f"chat_name in [{', '.join(f'"{name}"' for name in chat_names)}]"
        filter_expression = f"chat_name in [{', '.join(repr(name) for name in chat_names)}]"

        if mode in ["dense", "hybrid"]:
            embedding = self.embedding_function.embed_query(query)
            dense_vec = embedding

        if mode == "sparse":
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query],
                anns_field="sparse_vector",
                limit=k,
                filter=filter_expression,
                output_fields=output_fields,
            )
        elif mode == "dense":
            results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_vec],
                anns_field="dense_vector",
                limit=k,
                filter=filter_expression,
                output_fields=output_fields,
            )
        elif mode == "hybrid":
            full_text_search_params = {"metric_type": "BM25"}
            full_text_search_req = AnnSearchRequest(
                [query], "sparse_vector", full_text_search_params, limit=k, expr=filter_expression,
            )

            dense_search_params = {"metric_type": "IP"}
            dense_req = AnnSearchRequest(
                [dense_vec], "dense_vector", dense_search_params, limit=k, expr=filter_expression,
            )

            results = self.client.hybrid_search(
                self.collection_name,
                [full_text_search_req, dense_req],
                # ranker=WeightedRanker(*weights),
                ranker=RRFRanker(k=k_rerank),
                limit=k,
                filter=filter_expression,
                output_fields=output_fields,
            )
        else:
            raise ValueError("Invalid mode")
        return [
            {
                "message": doc["entity"]["message"],
                "chat_name": doc["entity"]["chat_name"],
                "score": doc["distance"],
            }
            for doc in results[0]
        ]
