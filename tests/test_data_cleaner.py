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
