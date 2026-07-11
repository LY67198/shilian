"""知识库服务（Milvus 版）

业务元数据存 PostgreSQL，embedding 存 Milvus。
- 写路径：PG 拿 ID → Milvus 写同 id；失败回滚 PG
- 读路径：直接调 Milvus，不需要 PG 事务
"""
from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge import KnowledgeChunk, KnowledgeDocument
from app.llm.embedding import embed_text, embed_texts
from app.vector_db import get_milvus_client
from app.vector_db.collections import knowledge as knowledge_vdb
from app.vector_db.collections.knowledge import KnowledgeChunkPayload

logger = logging.getLogger(__name__)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", ".", " "],
)


def _extract_text(file_bytes: bytes, file_type: str) -> str:
    """从文件字节中提取纯文本"""
    file_type = file_type.lower().lstrip(".")
    if file_type == "pdf":
        import pdfplumber
        text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    # txt / md / markdown
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="ignore")


def _hash_content(text: str) -> str:
    """生成 chunk 内容的 hash（用于去重 / 调试）"""
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class KnowledgeService:
    """知识库服务，提供文档摄入、切分配置、向量化写入（PG + Milvus 双写）和语义检索功能。"""

    async def ingest_document(
        self,
        doc_id: int,
        file_bytes: bytes,
        file_type: str,
        db: AsyncSession,
    ) -> int:
        """
        摄入文档：提取文本 → 切分 → 批量向量化 → 写 PG + Milvus

        写策略：先 PG 拿 auto-increment id，再 Milvus 同 id 写；
                任一失败则回滚 PG（删已 insert 的 chunks）。
        """
        # 1. 标记 indexing
        await db.execute(
            update(KnowledgeDocument)
            .where(KnowledgeDocument.id == doc_id)
            .values(status="indexing")
        )
        await db.commit()

        milvus_client = get_milvus_client()

        try:
            # 2. 提取 + 切分
            raw_text = _extract_text(file_bytes, file_type)
            if not raw_text.strip():
                raise ValueError("文档提取不到文本内容")
            chunks = _splitter.split_text(raw_text)
            if not chunks:
                raise ValueError("文档切分后为空")

            # 3. PG：批量插入拿 ID
            chunk_records = [
                KnowledgeChunk(
                    document_id=doc_id,
                    chunk_index=i,
                    content=text,
                    content_hash=_hash_content(text),
                )
                for i, text in enumerate(chunks)
            ]
            db.add_all(chunk_records)
            await db.flush()  # 触发 auto-increment
            chunk_ids = [c.id for c in chunk_records]
            logger.info(f"PG 插入 {len(chunk_ids)} chunk 记录（document_id={doc_id}）")

            # 4. 向量化
            embeddings = await embed_texts(chunks)

            # 5. Milvus：同 id 写
            payloads = [
                KnowledgeChunkPayload(
                    id=cid,
                    document_id=doc_id,
                    chunk_index=i,
                    content=text,
                    content_hash=_hash_content(text),
                    embedding=emb,
                )
                for i, (cid, text, emb) in enumerate(zip(chunk_ids, chunks, embeddings))
            ]
            knowledge_vdb.insert_chunks(milvus_client, payloads)

            # 6. 标记 indexed
            await db.execute(
                update(KnowledgeDocument)
                .where(KnowledgeDocument.id == doc_id)
                .values(status="indexed", chunk_count=len(chunks))
            )
            await db.commit()
            logger.info(f"文档 {doc_id} 索引完成，共 {len(chunks)} 个 chunk")
            return len(chunks)

        except Exception as e:
            # 回滚 PG
            await db.execute(
                delete(KnowledgeChunk).where(KnowledgeChunk.document_id == doc_id)
            )
            await db.execute(
                update(KnowledgeDocument)
                .where(KnowledgeDocument.id == doc_id)
                .values(status="failed", error_message=str(e))
            )
            await db.commit()
            logger.error(f"文档 {doc_id} 索引失败: {e}")
            raise

    async def ingest_from_path(self, doc_id: int, file_path: str, file_type: str) -> int:
        """从本地磁盘路径读取文件并摄入文档（用于 BackgroundTasks，自带独立 DB 会话）。

        Args:
            doc_id: 文档 ID。
            file_path: 本地文件路径。
            file_type: 文件类型（如 pdf、txt、md）。

        Returns:
            生成的 chunk 数量。
        """
        from app.db.base import get_session_local
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        async with get_session_local()() as db:
            return await self.ingest_document(doc_id, file_bytes, file_type, db)

    async def delete_document(self, doc_id: int, db: AsyncSession) -> None:
        """删除文档及其所有 chunk（双删：先 Milvus 后 PG）。

        Args:
            doc_id: 文档 ID。
            db: 数据库会话。
        """
        # 先 Milvus（如果失败，至少 PG 还能查出来）
        try:
            milvus_client = get_milvus_client()
            knowledge_vdb.delete_by_document(milvus_client, doc_id)
        except Exception as e:
            logger.warning(f"Milvus 删除 document_id={doc_id} 失败（继续删 PG）: {e}")

        # 再 PG（CASCADE 已自动删 chunks）
        await db.execute(
            delete(KnowledgeDocument).where(KnowledgeDocument.id == doc_id)
        )
        await db.commit()

    async def reindex_document(
        self,
        doc_id: int,
        file_bytes: bytes,
        file_type: str,
        db: AsyncSession,
    ) -> int:
        """删除旧 chunk 后重新索引文档。

        Args:
            doc_id: 文档 ID。
            file_bytes: 文件字节内容。
            file_type: 文件类型（如 pdf、txt、md）。
            db: 数据库会话。

        Returns:
            重新索引后生成的 chunk 数量。
        """
        milvus_client = get_milvus_client()
        # 双删
        try:
            knowledge_vdb.delete_by_document(milvus_client, doc_id)
        except Exception as e:
            logger.warning(f"Milvus reindex 旧数据删除失败: {e}")
        await db.execute(
            delete(KnowledgeChunk).where(KnowledgeChunk.document_id == doc_id)
        )
        await db.commit()
        return await self.ingest_document(doc_id, file_bytes, file_type, db)

    async def retrieve_chunks(
        self,
        query: str,
        db: AsyncSession = None,  # noqa 保留参数仅为向后兼容（不再用）
        k: int = 4,
        category: Optional[str] = None,  # noqa 当前未在 Milvus 端过滤，留作接口兼容
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        语义检索文档 chunk（走 Milvus）

        返回列表：[{id, document_id, chunk_index, content, content_hash, metadata, similarity}, ...]
        """
        query_vec = await embed_text(query)
        milvus_client = get_milvus_client()
        return knowledge_vdb.search(
            milvus_client,
            query_vec,
            top_k=k,
            document_ids=None,  # category 过滤后续接入 Milvus 端 metadata 查询
            min_score=min_score,
        )

    async def get_document_list(
        self,
        db: AsyncSession,
        page: int = 1,
        size: int = 20,
        category: Optional[str] = None,
        status: Optional[str] = None,
        search: Optional[str] = None,
    ) -> dict:
        """分页查询知识库文档列表，支持分类、状态和关键词筛选。

        Args:
            db: 数据库会话。
            page: 页码，默认 1。
            size: 每页条数，默认 20。
            category: 文档分类筛选。
            status: 文档状态筛选（如 indexing、indexed、failed）。
            search: 标题关键词模糊搜索。

        Returns:
            包含 items、total、page、size 的字典。
        """
        stmt = select(KnowledgeDocument).where(KnowledgeDocument.id > 0)
        if category:
            stmt = stmt.where(KnowledgeDocument.category == category)
        if status:
            stmt = stmt.where(KnowledgeDocument.status == status)
        if search:
            stmt = stmt.where(KnowledgeDocument.title.ilike(f"%{search}%"))

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar_one()

        stmt = stmt.order_by(KnowledgeDocument.created_at.desc()).offset((page - 1) * size).limit(size)
        items = (await db.execute(stmt)).scalars().all()
        return {"items": items, "total": total, "page": page, "size": size}


knowledge_service = KnowledgeService()
