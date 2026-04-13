import pytest
from flask_app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.mark.integration
def test_index(client):
    res = client.get("/")
    assert res.status_code == 200


@pytest.mark.integration
def test_next_letter(client):
    res = client.get("/next_letter")
    assert res.status_code == 200
    assert "target" in res.json


@pytest.mark.integration
def test_toggle_mask(client):
    res = client.get("/toggle_mask")
    assert res.status_code == 200
