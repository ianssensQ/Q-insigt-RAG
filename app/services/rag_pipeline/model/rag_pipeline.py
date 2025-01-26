import asyncio
import os
import warnings
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from typing import Dict, List, Tuple

from .milvus_retriver import HybridRetriever


class AnswerGenerator:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=0.05,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )

        prompts_dir = Path(__file__).parent / 'prompts'
        with open(str(prompts_dir / 'context_answer_generation.txt'), 'r') as prompt_file:
            self.generation_prompt_text =  prompt_file.read()

    async def generate_answer(
        self,
        user_query: str,
        chat_names: List[str],
        retriever_params: Dict = None
    ) -> Tuple[str, List[str]]:
        if retriever_params is None:
            retriever_params = {}

        retrieved_documents = await self.retriever.search(
            query=user_query,
            chat_names=chat_names,
            **retriever_params
        )

        context_str = "\n".join(
            f"# Чат: {doc['chat_name']}\n# Сообщение: {doc['message']}\n" for doc in retrieved_documents
        )
        context_list = [doc['message'] for doc in retrieved_documents]

        if context_str == '':
            warnings.warn("The retrieved context_str is empty", UserWarning)

        prompt = PromptTemplate(
            template=self.generation_prompt_text,
            input_variables=['user_query', 'context'],
        )

        chain = prompt | self.llm

        chain_output = chain.invoke(
            {
                'input_language': 'Russian',
                'output_language': 'Russian',
                'user_query': user_query,
                'context': context_str
            }
        )

        return chain_output.content, context_list


async def main():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    retriever = HybridRetriever(
        uri="http://localhost:19530",
        collection_name="SampleTest",
        dense_embedding_function=embeddings
    )
    retriever.build_collection(recreation=True)

    # insert test data
    retriever.insert_data(
        {"message": 'Сегодня холодно', 'chat_name': 'Погода', 'chat_message_id': 0}
    )
    retriever.insert_data(
        {"message": 'Завтра 10 градусов и сильный ветер', 'chat_name': 'Погода', 'chat_message_id': 1}
    )
    retriever.insert_data(
        {"message": '10 декабря сильный ветер и -10 градусов', 'chat_name': 'Погода', 'chat_message_id': 2}
    )

    retriever.insert_data(
        {"message": 'Сегодня концер Егора крида', 'chat_name': 'Афиша', 'chat_message_id': 0}
    )
    retriever.insert_data(
        {"message": '10 ноября стендап Абрамова ветер', 'chat_name': 'Афиша', 'chat_message_id': 1}
    )
    retriever.insert_data(
        {"message": '11 декабря стендап Усовича', 'chat_name': 'Афиша', 'chat_message_id': 2}
    )

    retriever.insert_data(
        {"message": 'Сегодня концер Егора крида', 'chat_name': 'Концерты', 'chat_message_id': 0}
    )
    retriever.insert_data(
        {"message": '10 ноября стендап Абрамова ветер', 'chat_name': 'Концерты', 'chat_message_id': 1}
    )
    retriever.insert_data(
        {"message": '11 декабря стендап Усовича', 'chat_name': 'Концерты', 'chat_message_id': 2}
    )

    generator = AnswerGenerator(retriever)
    answer = await generator.generate_answer(
        'Сегодня 28 ноября. Какая погода будет 10 декабря?',
        chat_names=['Погода'],
        retriever_params={'k': 1}
    )
    return answer

if __name__ == '__main__':
    print(asyncio.run(main()))