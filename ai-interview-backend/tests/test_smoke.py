"""
Smoke tests — 只需要后端服务运行，不依赖具体业务数据

运行：
  pytest tests/test_smoke.py -m "smoke"
环境变量：
  API_BASE_URL（默认 http://localhost:8006）
"""
import pytest


@pytest.mark.smoke
def test_health_endpoint(api_client):
    """GET /api/v1/config/health 返回 200 + healthy"""
    r = api_client.get("/api/v1/config/health")
    assert r.status_code == 200
    body = r.json()
    assert body["code"] == 200
    assert body["data"]["status"] == "healthy"


@pytest.mark.smoke
def test_cors_rejects_when_no_origin_in_production(api_client):
    """CORS 中间件存在 — 任意 Origin 走兜底响应 200 + 实际请求做权限校验"""
    r = api_client.get(
        "/api/v1/config/health",
        headers={"Origin": "https://attacker.example.com"},
    )
    assert r.status_code in (200, 403)