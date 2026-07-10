"""ExceptionGroup 解包工具

Python 3.11+ 引入 ExceptionGroup / TaskGroup，多个并发任务失败时会包装。
对日志和错误响应需要展开成扁平列表。
"""
from __future__ import annotations

from typing import List


def format_exception_chain(exc: BaseException) -> str:
    """展开 ExceptionGroup / TaskGroup 为人类可读的多行文本

    例：
        RuntimeError: task failed
          caused by: ValueError: bad input
          caused by: TypeError: ...
    """
    sub_exceptions = getattr(exc, "exceptions", None)
    if sub_exceptions is not None:
        lines = [str(exc)]
        for i, sub in enumerate(sub_exceptions):
            lines.append(f"  [{i}] {format_exception_chain(sub)}")
        return "\n".join(lines)

    msg = f"{type(exc).__name__}: {exc}"
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return f"{msg}\n  caused by: {format_exception_chain(cause)}"
    return msg


def flatten_exceptions(exc: BaseException) -> List[BaseException]:
    """扁平化 ExceptionGroup，返回所有子异常列表（去重 + 保序）"""
    result: List[BaseException] = []
    seen = set()

    def walk(e: BaseException) -> None:
        eid = id(e)
        if eid in seen:
            return
        seen.add(eid)
        sub = getattr(e, "exceptions", None)
        if sub is not None:
            for s in sub:
                walk(s)
        else:
            result.append(e)

    walk(exc)
    return result