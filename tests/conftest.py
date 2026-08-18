import sys
from pathlib import Path

# make main.py importable without installing the project as a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

import main


VALID_CUSTOMER = {
    "tenure": 12,
    "MonthlyCharges": 79.90,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "OnlineSecurity": "No",
    "PaperlessBilling": "Yes",
}


@pytest.fixture(scope="session")
def client():
    return TestClient(main.app)


@pytest.fixture
def customer():
    return dict(VALID_CUSTOMER)
