import pandas as pd
import pytest

from data_cleaner import deterministic_clean, execute_cleaning_code, extract_code, inspect_dataframe


def test_metadata_reports_nulls_duplicates_and_skewness():
    df = pd.DataFrame({"age": [10, 20, None, 20], "name": ["A", "B", "C", "B"]})
    metadata = inspect_dataframe(df)
    assert metadata["rows"] == 4
    assert metadata["null_counts"]["age"] == 1
    assert metadata["duplicate_rows"] == 1
    assert "age" in metadata["skewness"]


def test_deterministic_clean_normalizes_and_fills_numeric_values():
    df = pd.DataFrame({" Age ": [10, None, 10], "Name ": ["A", "B", "A"]})
    cleaned = deterministic_clean(df)
    assert cleaned.columns.tolist() == ["age", "name"]
    assert len(cleaned) == 2
    assert cleaned["age"].isna().sum() == 0


def test_valid_generated_code_executes_on_a_copy():
    df = pd.DataFrame({"amount": [1, None]})
    cleaned = execute_cleaning_code(df, "df = df.copy()\ndf['amount'] = pd.to_numeric(df['amount']).fillna(0)")
    assert cleaned["amount"].tolist() == [1, 0]
    assert df["amount"].isna().sum() == 1


@pytest.mark.parametrize("code", [
    "import os\ndf = df",
    "df = eval('df')",
    "open('secret.txt')",
    "df = df.apply(lambda x: x)",
])
def test_unsafe_generated_code_is_rejected(code):
    with pytest.raises((ValueError, SyntaxError)):
        execute_cleaning_code(pd.DataFrame({"x": [1]}), code)


def test_extract_code_removes_markdown_fences():
    assert extract_code("```python\ndf = df.drop_duplicates()\n```") == "df = df.drop_duplicates()"


def test_huggingface_generation_requires_hf_token(monkeypatch):
    from data_cleaner import generate_cleaning_code
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        generate_cleaning_code({"rows": 1})


def test_huggingface_credit_error_is_explained(monkeypatch):
    import data_cleaner

    class CreditError(Exception):
        status_code = 402

    class FakeCompletions:
        def create(self, **kwargs):
            raise CreditError("no remaining credits")

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr("openai.OpenAI", FakeClient)
    with pytest.raises(RuntimeError, match="no remaining credits"):
        data_cleaner.generate_cleaning_code({"rows": 1})


def test_column_string_normalization_is_allowed():
    df = pd.DataFrame({" Customer Name ": ["Alice"]})
    cleaned = execute_cleaning_code(
        df,
        "df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')",
    )
    assert cleaned.columns.tolist() == ["customer_name"]


def test_median_imputation_is_allowed():
    df = pd.DataFrame({"age": [20, None, 40]})
    cleaned = execute_cleaning_code(df, "df['age'] = df['age'].fillna(df['age'].median())")
    assert cleaned["age"].tolist() == [20.0, 30.0, 40.0]


def test_valid_pandas_import_is_available_to_generated_code():
    cleaned = execute_cleaning_code(
        pd.DataFrame({"amount": ["1", "2"]}),
        "import pandas as pd\ndf['amount'] = pd.to_numeric(df['amount'])",
    )
    assert cleaned["amount"].tolist() == [1, 2]
