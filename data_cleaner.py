"""Autonomous CSV cleaning and EDA engine with constrained LLM code generation."""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

ALLOWED_IMPORTS = {"pandas"}
ALLOWED_CALLS = {
    "pd.to_numeric", "pd.to_datetime", "df.dropna", "df.fillna", "df.rename",
    "df.drop", "df.astype", "df.replace", "df.columns", "df.copy",
    "df.select_dtypes", "df.duplicated", "df.drop_duplicates",
}
ALLOWED_CHAINED_METHODS = {"fillna", "dropna", "astype", "replace", "drop_duplicates", "reset_index"}
ALLOWED_STRING_METHODS = {"strip", "lower", "upper", "replace", "title"}
FORBIDDEN_NAMES = {"__import__", "eval", "exec", "open", "compile", "globals", "locals", "input", "os", "subprocess", "pathlib"}


def inspect_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-friendly EDA metadata for an input DataFrame."""
    numeric = df.select_dtypes(include="number")
    skewness = numeric.skew(numeric_only=True).replace({float("nan"): None}).to_dict()
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [str(column) for column in df.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
        "null_counts": {str(column): int(value) for column, value in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "skewness": {str(key): (None if value is None else float(value)) for key, value in skewness.items()},
        "sample_rows": df.head(3).fillna("").astype(str).to_dict(orient="records"),
    }


def build_cleaning_prompt(metadata: dict[str, Any]) -> str:
    """Build a precise prompt asking for code that transforms the variable ``df``."""
    return f"""You are a careful pandas data-cleaning assistant.
Return ONLY Python code. The code must transform the existing pandas DataFrame named df.
Rules: do not read files, write files, import modules except pandas as pd, call network APIs, use exec/eval, or access the operating system.
Prefer explicit, conservative operations: trim column names, remove exact duplicate rows, coerce clearly numeric columns, and fill missing numeric values with medians only when justified.
The final DataFrame must remain named df.

EDA metadata:
{json.dumps(metadata, indent=2, default=str)}
"""


def extract_code(text: str) -> str:
    """Strip Markdown fences if a provider returns them despite the code-only request."""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def validate_cleaning_code(code: str) -> ast.Module:
    """Reject unsafe generated code before execution and return its parsed AST."""
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.With, ast.AsyncWith, ast.Try, ast.Raise, ast.Delete, ast.Global, ast.Nonlocal)):
            raise ValueError(f"Unsupported generated construct: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"Forbidden name in generated code: {node.id}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS or alias.asname != "pd":
                    raise ValueError("Only 'import pandas as pd' is allowed")
        if isinstance(node, ast.ImportFrom):
            raise ValueError("from-import statements are not allowed")
        if isinstance(node, ast.Call):
            qualified = _qualified_name(node.func)
            is_chained_method = isinstance(node.func, ast.Attribute) and node.func.attr in ALLOWED_CHAINED_METHODS
            is_column_string_method = (
                ".str." in qualified
                and qualified.rsplit(".", 1)[-1] in ALLOWED_STRING_METHODS
            )
            if qualified not in ALLOWED_CALLS and not is_chained_method and not is_column_string_method:
                raise ValueError(f"Call is not allow-listed: {qualified}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is not allowed")
    return tree


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return "<dynamic>"


def execute_cleaning_code(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Execute validated code in a minimal namespace and return a copied DataFrame."""
    validate_cleaning_code(code)
    working = df.copy(deep=True)
    namespace = {"__builtins__": {}, "pd": pd, "df": working}
    exec(compile(code, "<generated-cleaning-code>", "exec"), namespace, namespace)
    result = namespace.get("df")
    if not isinstance(result, pd.DataFrame):
        raise ValueError("Generated code must leave a pandas DataFrame in variable df")
    return result


def deterministic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """API-free conservative fallback for demos and offline testing."""
    cleaned = df.copy(deep=True)
    cleaned.columns = [str(column).strip().lower().replace(" ", "_") for column in cleaned.columns]
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            try:
                cleaned[column] = pd.to_numeric(cleaned[column])
            except (TypeError, ValueError):
                pass
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    return cleaned


def generate_cleaning_code(metadata: dict[str, Any], model: str = "Qwen/Qwen2.5-Coder-32B-Instruct:featherless-ai") -> str:
    """Ask Hugging Face Router for constrained pandas code. Requires HF_TOKEN."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Use --dry-run for API-free testing.")
    from openai import OpenAI
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=700,
            messages=[
                {"role": "system", "content": "Return only safe pandas code. The input DataFrame is named df."},
                {"role": "user", "content": build_cleaning_prompt(metadata)},
            ],
        )
    except Exception as exc:
        if getattr(exc, "status_code", None) == 402 or "no remaining credits" in str(exc).lower():
            raise RuntimeError(
                "Hugging Face Inference Providers reports no remaining credits. "
                "Use --dry-run for the free offline cleaner, or use a Hugging Face "
                "account/provider with available inference credits."
            ) from exc
        raise
    return extract_code(response.choices[0].message.content or "")


def write_outputs(df: pd.DataFrame, metadata: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "cleaned_data.csv", index=False)
    df.to_excel(output_dir / "cleaned_data.xlsx", index=False)
    (output_dir / "eda_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def run(input_path: Path, output_dir: Path, dry_run: bool, model: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    before = inspect_dataframe(df)
    if dry_run:
        cleaned = deterministic_clean(df)
        generated = "# Offline deterministic fallback\n" \
            "df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]\n" \
            "df = df.drop_duplicates().reset_index(drop=True)"
    else:
        generated = generate_cleaning_code(before, model=model)
        cleaned = execute_cleaning_code(df, generated)
    after = inspect_dataframe(cleaned)
    metadata = {"input_file": str(input_path), "before": before, "after": after, "generated_code": generated, "mode": "dry-run" if dry_run else "huggingface"}
    write_outputs(cleaned, metadata, output_dir)
    print(json.dumps({"before": before, "after": after, "output_dir": str(output_dir)}, indent=2, default=str))
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean a CSV with EDA metadata and optional Hugging Face-generated pandas code.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--dry-run", action="store_true", help="Use the deterministic offline cleaner; no API key required")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct:featherless-ai")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input_csv, args.output_dir, args.dry_run, args.model)
