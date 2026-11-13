"""嵌入模型单例：FAQ 与知识库共用，避免重复加载 400MB 模型"""

from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
