"""The decision threshold is chosen by expected cost, not left at 0.5."""
import json

import main


def test_default_threshold_comes_from_the_trained_artifact():
    """train.py writes the cost-minimising threshold; the API must honour it."""
    saved = json.loads(main.THRESHOLD_PATH.read_text())
    assert main.DEFAULT_THRESHOLD == saved["threshold"]
    assert 0.0 < main.DEFAULT_THRESHOLD < 1.0


def test_threshold_is_not_the_naive_default():
    """A 5:1 cost asymmetry should not land back on 0.5 — that would mean the
    sweep did nothing and the whole exercise is decorative."""
    assert main.DEFAULT_THRESHOLD != 0.5


def test_response_reports_the_threshold_applied(client, customer):
    body = client.post("/predict", json=customer).json()
    assert body["threshold"] == main.DEFAULT_THRESHOLD


def test_threshold_override_changes_the_verdict(client, customer):
    """Same customer, different thresholds, opposite calls."""
    probability = client.post("/predict", json=customer).json()["churn_probability"]

    customer["threshold"] = round(probability - 0.01, 4)
    assert client.post("/predict", json=customer).json()["will_churn"] is True

    customer["threshold"] = round(probability + 0.01, 4)
    assert client.post("/predict", json=customer).json()["will_churn"] is False


def test_override_does_not_change_the_probability(client, customer):
    """The threshold moves the decision, never the model output."""
    baseline = client.post("/predict", json=customer).json()["churn_probability"]

    customer["threshold"] = 0.9
    assert client.post("/predict", json=customer).json()["churn_probability"] == baseline


def test_out_of_range_threshold_is_rejected(client, customer):
    for bad in (0.0, 1.0, -0.5, 1.5):
        customer["threshold"] = bad
        assert client.post("/predict", json=customer).status_code == 422


def test_threshold_is_not_treated_as_a_model_feature():
    """It rides on the request schema, so it must be excluded from the columns
    used to map SHAP values back to features."""
    assert "threshold" not in main.INPUT_COLUMNS
