"""The saved pipeline and the SHAP attribution built on top of it."""
import pytest

import main


def test_pipeline_carries_its_own_preprocessing():
    """Encoder, scaler and classifier are one artifact — that is what keeps the
    API from scaling inputs differently than training did."""
    assert set(main.pipeline.named_steps) == {"preprocessor", "classifier"}


def test_model_unpickled_under_the_pinned_versions():
    """Guards the failure that broke the container: a newer scikit-learn cannot
    load this ColumnTransformer ('_RemainderColsList' is gone)."""
    assert main.pipeline.predict_proba is not None
    assert len(main.feature_names) == 12


@pytest.mark.parametrize("encoded, expected", [
    ("num__tenure", "tenure"),
    ("num__MonthlyCharges", "MonthlyCharges"),
    ("cat__Contract_One year", "Contract"),
    ("cat__Contract_Two year", "Contract"),
    ("cat__InternetService_Fiber optic", "InternetService"),
    ("cat__PaymentMethod_Credit card (automatic)", "PaymentMethod"),
    ("cat__OnlineSecurity_No internet service", "OnlineSecurity"),
    ("cat__PaperlessBilling_Yes", "PaperlessBilling"),
])
def test_encoded_columns_map_back_to_their_source_feature(encoded, expected):
    assert main.original_feature(encoded) == expected


def test_every_encoded_column_resolves_to_a_real_input_feature():
    """No encoded column may fall through to an invented feature name."""
    for name in main.feature_names:
        assert main.original_feature(name) in main.INPUT_COLUMNS


def test_underscored_feature_names_are_not_split_apart():
    """Regression: splitting on the first underscore silently mis-grouped any
    feature whose name contains one, attributing 'Tech_Support' to 'Tech'."""
    columns = sorted(["Tech", "Tech_Support"], key=len, reverse=True)
    original = main.INPUT_COLUMNS
    main.INPUT_COLUMNS = columns
    try:
        assert main.original_feature("cat__Tech_Support_Yes") == "Tech_Support"
        assert main.original_feature("cat__Tech_Yes") == "Tech"
    finally:
        main.INPUT_COLUMNS = original


def test_shap_factors_are_ranked_by_absolute_impact(client, customer):
    factors = client.post("/predict", json=customer).json()["top_factors"]
    magnitudes = [abs(f["impact"]) for f in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_factor_direction_matches_the_sign_of_its_impact(client, customer):
    for factor in client.post("/predict", json=customer).json()["top_factors"]:
        if factor["impact"] > 0:
            assert factor["direction"] == "increases risk"
        else:
            assert factor["direction"] == "decreases risk"
