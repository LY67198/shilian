"""
Pytest 全局 fixtures

设计原则：
- 单元测试（@pytest.mark.unit）不应依赖任何外部服务
- 集成测试（@pytest.mark.integration）需要 DB/Redis，CI 中通过 docker compose service
- 端到端测试（@pytest.mark.e2e）需要完整 AI API，默认跳过

运行方式：
- 快速 CI：pytest -m "unit"           只有纯单元测试
- 本地全量：pytest -m "unit"          同上
- 集成：    pytest -m "integration"   需 DB / Redis（在容器内执行）
"""
import os
import pytest
import httpx


@pytest.fixture(scope="session")
def api_base_url() -> str:
    return os.environ.get("API_BASE_URL", "http://localhost:8006")


@pytest.fixture(scope="session")
def api_client(api_base_url: str) -> httpx.Client:
    with httpx.Client(
        base_url=api_base_url,
        timeout=httpx.Timeout(30.0),
        headers={"Content-Type": "application/json"},
    ) as client:
        yield client


def pytest_collection_modifyitems(config, items):
    """根据环境变量自动跳过需要 AI 服务的端到端测试"""
    if not os.environ.get("RUN_E2E"):
        skip_e2e = pytest.mark.skip(reason="需 RUN_E2E=1 才运行")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)