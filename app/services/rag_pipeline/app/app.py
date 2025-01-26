import logging
import os

from fastapi import FastAPI, HTTPException, Depends
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from typing import List, Dict

from model import AnswerGenerator, HybridRetriever

logging.basicConfig(
    filename='logs/app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = FastAPI()


def init_generator():
    milvus_host = os.getenv('MILVUS_HOST')
    milvus_port = os.getenv('MILVUS_PORT')
    collection_name = os.getenv('COLLECTION_NAME')

    uri = f"http://{milvus_host}:{milvus_port}"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    retriever = HybridRetriever(uri=uri, collection_name=collection_name, dense_embedding_function=embeddings)
    return AnswerGenerator(retriever=retriever)


class QueryRequest(BaseModel):
    user_query: str
    chat_names: List[str]
    retriever_params: Dict | None = None


class QueryResponse(BaseModel):
    answer: str
    context: List[str]


@app.post("/generate-answer", response_model=QueryResponse)
async def generate_answer(
    query_request: QueryRequest,
    answer_generator: AnswerGenerator = Depends(init_generator)
):
    """
    Endpoint to generate an answer based on the user's query.
    """
    try:
        logger.info(f"user query: {query_request.user_query}")

        answer, context = await answer_generator.generate_answer(
            user_query=query_request.user_query,
            chat_names=query_request.chat_names,
            retriever_params=query_request.retriever_params,
        )

        logger.info(f"Generated answer successfully for query: {query_request.user_query}")

        return QueryResponse(answer=answer, context=context)

    except Exception as e:
        logger.error(f"Error generating answer for query: {query_request.user_query}. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/")
def read_root():
    """
    Root endpoint
    """
    logger.info("Root endpoint accessed.")
    return {"message": "RAG system API is running."}
