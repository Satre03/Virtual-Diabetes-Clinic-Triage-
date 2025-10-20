import pytest
from fastapi.testclient import TestClient
from src.main import app

# 关键改动：我们不再在全局创建 client
# 相反，我们创建一个 pytest fixture，它会在每个测试运行前被调用
@pytest.fixture(scope="module")
def test_client():
    # 使用 "with" 语句来创建 TestClient
    # 这种方式可以确保 FastAPI 的 "startup" 和 "shutdown" 事件被正确触发！
    with TestClient(app) as client:
        yield client # "yield" 把 client 对象提供给测试函数使用

def test_health_check_success(test_client):
    """测试 /health 接口能否成功返回"""
    # 注意：测试函数现在接收 test_client 作为参数
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    # 我们不再硬编码版本号，因为未来可能会变
    assert "model_version" in response.json()

def test_predict_success(test_client):
    """测试 /predict 接口在输入正确时能否成功预测"""
    payload = {
        "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
        "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02,
        "s5": 0.02, "s6": -0.001
    }
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)

def test_predict_bad_input(test_client):
    """测试 /predict 接口在输入错误时能否返回正确的错误码"""
    payload = {
        "age": 0.02, "sex": -0.044, "bmi": 0.06
    }
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422
