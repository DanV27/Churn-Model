"""Endpoint behaviour: the contract the dashboard and any other client rely on."""
import pytest


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_dashboard_is_served(client):
    """The frontend is mounted on the API itself, so /app must return HTML."""
    response = client.get("/app/", follow_redirects=True)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Churn Risk Dashboard" in response.text


def test_predict_returns_full_contract(client, customer):
    body = client.post("/predict", json=customer).json()

    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["will_churn"], bool)
    assert 0.0 < body["threshold"] < 1.0
    assert len(body["top_factors"]) == 3

    for factor in body["top_factors"]:
        assert factor["direction"] in {"increases risk", "decreases risk"}
        assert set(factor) == {"feature", "impact", "direction"}


def test_high_risk_customer_scores_above_low_risk_one(client, customer):
    """A short-tenure month-to-month customer must outrank a loyal two-year one."""
    high = client.post("/predict", json=customer).json()["churn_probability"]

    customer.update(tenure=65, Contract="Two year", InternetService="DSL",
                    PaymentMethod="Credit card (automatic)", OnlineSecurity="Yes",
                    PaperlessBilling="No", MonthlyCharges=24.50)
    low = client.post("/predict", json=customer).json()["churn_probability"]

    assert high > low


def test_predictions_are_deterministic(client, customer):
    first = client.post("/predict", json=customer).json()
    second = client.post("/predict", json=customer).json()
    assert first == second


@pytest.mark.parametrize("field, bad_value", [
    ("Contract", "Bogus plan"),
    ("InternetService", "Satellite"),
    ("PaymentMethod", "Cash"),
    ("OnlineSecurity", "Maybe"),
    ("PaperlessBilling", "Sometimes"),
])
def test_invalid_category_is_rejected(client, customer, field, bad_value):
    """Literal-typed fields must 422 and name the offending field."""
    customer[field] = bad_value
    response = client.post("/predict", json=customer)

    assert response.status_code == 422
    assert any(field in detail["loc"] for detail in response.json()["detail"])


def test_missing_field_is_rejected(client, customer):
    del customer["tenure"]
    assert client.post("/predict", json=customer).status_code == 422


def test_non_numeric_tenure_is_rejected(client, customer):
    customer["tenure"] = "twelve"
    assert client.post("/predict", json=customer).status_code == 422
