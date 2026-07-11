"""Milvus 初始化脚本

用法：
  # 在容器内
  docker exec ai-interview-app python scripts/init_milvus.py
  # 在宿主机（连接到 localhost:19530）
  MILVUS_HOST=localhost python scripts/init_milvus.py

行为：
- 连接 Milvus
- 为 knowledge_chunks 和 question_bank 两个 collection 建表 + 索引
- 已存在则跳过
- 打印最终状态
"""
import asyncio
import logging
import sys

from app.core.config import settings
from app.vector_db.client import get_milvus_client, health_check_milvus
from app.vector_db.collections import knowledge, question_bank

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    logger.info(f"=== Milvus 初始化 ===")
    logger.info(f"Target: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}/{settings.MILVUS_DB_NAME}")

    # 健康检查
    if not health_check_milvus():
        logger.error(f"无法连接 Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        return 1

    client = get_milvus_client()
    dim = settings.KNOWLEDGE_EMBEDDING_DIM
    logger.info(f"Embedding dim = {dim}")

    # FORCE RESET（如果存在且用的是旧 COSINE 索引，drop 重建以切换到 L2）
    existing = client.list_collections()
    logger.info(f"已存在 collections: {existing}")
    if knowledge.COLLECTION_NAME in existing:
        logger.warning(f"Drop & recreate {knowledge.COLLECTION_NAME} (L2 index)")
        client.drop_collection(knowledge.COLLECTION_NAME)
    if question_bank.COLLECTION_NAME in existing:
        logger.warning(f"Drop & recreate {question_bank.COLLECTION_NAME} (L2 index)")
        client.drop_collection(question_bank.COLLECTION_NAME)

    # 建 knowledge_chunks
    knowledge.create_collection(client, dim=dim)
    # 建 question_bank
    question_bank.create_collection(client, dim=dim)

    # 校验最终状态
    logger.info("=== 初始化完成，最终状态 ===")
    for name in [knowledge.COLLECTION_NAME, question_bank.COLLECTION_NAME]:
        if not client.has_collection(name):
            logger.error(f"FAIL: {name} 不存在")
            return 1
        info = client.describe_collection(name)
        field_count = len(info.get("fields", []))
        logger.info(f"  ✓ {name}: {field_count} fields, dim={settings.KNOWLEDGE_EMBEDDING_DIM}")

    logger.info("两个 collection + HNSW 索引 (L2) 就绪")
    return 0


if __name__ == "__main__":
    sys.exit(main())