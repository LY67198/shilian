"""数据访问层（Repository Pattern）

Phase 1 引入。每个 ORM model 对应一个 repository，封装所有 SQL 操作。
业务服务（`app/services/`）和 workflow 节点（`app/workflows/<name>/nodes/`）
**不再直接写 SQL**，统一通过 repository 访问数据库。

## 规则
1. **纯 DB 操作**：repository 函数不调 LLM / Milvus / Redis
2. **不抛 HTTPException**：用普通 Python exceptions（NotFoundError / ValidationError / ConflictError）
   或返回 None / Optional[T]
3. **事务边界**：repository 函数不 `commit()`，由调用方（service 或 workflow node）管事务
4. **async 优先**：所有 DB 操作 async
5. **类型提示**：函数签名必须明确返回类型

## 命名约定
- 函数名用动词：`get_by_id` / `list_by_user` / `create` / `update` / `delete` / `count_by_*`
- 不要在 repo 里加业务判断（如"如果 active 才能删除"），留给 service / node

## 迁移计划
当前 P1 已建 base.py + 1 个示例（interview_repo）。
完整迁移在 Phase 3（多轮面试 graph 化）时做，service 层调用点同步切换。
"""
from app.repositories.base import BaseRepository

__all__ = ["BaseRepository"]