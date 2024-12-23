from infrastructure.database.database import Base, engine
from infrastructure.models_tables.usertable import UserTable
from infrastructure.models_tables.taskstable import TaskTable
from infrastructure.models_tables.channelstable import ChannelTable
from infrastructure.models_tables.poststable import PostTable

from services.Milvus.milvus_retriver import HybridRetriever
from services.Milvus.embed import get_embedding_model

import os


def init_db():
    Base.metadata.create_all(bind=engine)


def drop_db():
    Base.metadata.drop_all(bind=engine)


if __name__ == "__main__":
    # drop_db()
    init_db()
    retriver = HybridRetriever(uri=os.getenv("URI_MILVUS"), dense_embedding_function=get_embedding_model())
    retriver.build_collection()
