import asyncio
import os
from pathlib import Path

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from typing import Dict, List


from app.services.Milvus.milvus_retriver import HybridRetriever


class AnswerGenerator:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        # self.llm = ChatMistralAI(
        #     model='gpt-4o-mini',
        #     temperature=0.7,
        #     openai_api_key=os.getenv('openai_api_key')
        # )
        self.llm = ChatMistralAI(
            model='ministral-8b-latest',
            temperature=0,
            mistral_api_key='W2LIYnvdAAbIMJBiqajgxbthjbhSrDG4'
        )

        prompts_dir = Path(__file__).parent / 'prompts'
        self.generation_prompt_text = open(str(prompts_dir / 'context_answer_generation.txt'), 'r').read()

    async def generate_answer(self, user_query: str, chat_names: List[str], retriever_params: Dict = None):
        if retriever_params is None:
            retriever_params = {}

        retrieved_documents = await self.retriever.search(
            query=user_query,
            chat_names=chat_names,
            **retriever_params
        )

        context = "\n".join(
            f"# Чат: {doc['chat_name']}\n# Сообщение: {doc['message']}\n" for doc in retrieved_documents
        )

        print(context)

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
                'context': context
            }
        )

        return chain_output.content


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
    ans = await generator.generate_answer('Сегодня 28 ноября. Какая погода будет 10 декабря?', chat_names=['Погода'], retriever_params={'k': 1})
    print(ans)

if __name__ == '__main__':
    asyncio.run(main())