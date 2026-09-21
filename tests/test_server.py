import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import localdecision as ld  # noqa: E402
from localdecision.backends.mock import MockBackend  # noqa: E402
from localdecision.server import create_app  # noqa: E402

REQUEST = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "localdecision-latest",
    "questions": {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, payouts",
                "technical": "Bugs, outages",
                "sales": None,
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}


@pytest.fixture
def client():
    return TestClient(create_app(ld.Engine(MockBackend()), api_key=""))


def test_wire_shape(client):
    r = client.post("/v1/systemone", json=REQUEST)
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"model", "answers", "usage"}
    assert body["usage"]["output_tokens"] == 0
    a = body["answers"]
    assert set(a["is_urgent"]) == {"type", "noul"}
    assert set(a["department"]) == {"type", "choice", "probabilities", "confidence"}
    assert set(a["frustration"]) == {"type", "score", "legend", "probabilities", "confidence"}
    assert a["department"]["choice"] == "billing"
    assert set(a["frustration"]["probabilities"]) == {"0", "1", "2"}


def test_extensions_can_be_switched_off(client):
    body = client.post("/v1/systemone", json={**REQUEST, "settings": {"diagnostics": False}}).json()
    assert "diagnostics" not in body and "timing" not in body


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "?", "criteria": {"only": None}},
        {"type": "choice", "instructions": "?", "criteria": {f"o{i}": None for i in range(256)}},
        {"type": "score", "instructions": "?", "criteria": [str(i) for i in range(11)]},
        {"type": "noul", "instructions": ""},
        {"type": "maybe", "instructions": "?"},
        {"type": "noul", "instructions": "?", "unknown_field": 1},
    ],
)
def test_invalid_questions_are_422(client, question):
    r = client.post("/v1/systemone", json={"state": "x", "questions": {"q": question}})
    assert r.status_code == 422


def test_empty_questions_and_state_are_422(client):
    assert client.post("/v1/systemone", json={"state": "x", "questions": {}}).status_code == 422
    assert (
        client.post(
            "/v1/systemone", json={"state": "", "questions": REQUEST["questions"]}
        ).status_code
        == 422
    )


def test_engine_errors_are_422():
    app = create_app(ld.Engine(MockBackend(max_context=100)), api_key="")
    r = TestClient(app).post("/v1/systemone", json=REQUEST)
    assert r.status_code == 422 and "context limit" in r.json()["detail"]


def test_bearer_auth():
    c = TestClient(create_app(ld.Engine(MockBackend()), api_key="secret"))
    assert c.post("/v1/systemone", json=REQUEST).status_code == 401
    ok = c.post("/v1/systemone", json=REQUEST, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
    assert c.get("/health").status_code == 200


def test_models_and_health(client):
    names = [m["name"] for m in client.get("/v1/models").json()["models"]]
    assert names == ["localdecision-latest", "overlap"]
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["shared_prefix_tokenization"] is True
