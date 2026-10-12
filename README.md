# Day 8 — Autonomous CSV Data Cleaner & EDA Engine

A Python data-quality agent that reads a messy CSV, profiles data types, null counts, duplicates, and numeric skewness, optionally asks Groq to generate conservative pandas cleaning code, validates that code with the Python AST, executes it in memory, and writes cleaned CSV, Excel, and metadata outputs.

> **Safety note:** executing model-generated code is risky. This project does not execute arbitrary LLM output directly: it parses and rejects unsupported imports, calls, and language constructs before execution. Treat the allow-list as a learning control, not a production sandbox.

## Architecture

See [docs/architecture.md](docs/architecture.md).

```text
Messy CSV
   ↓
pandas DataFrame
   ↓
EDA metadata: dtypes, nulls, duplicates, skewness
   ↓
Groq generates constrained pandas code (optional)
   ↓
AST allow-list validation
   ↓
In-memory execution on a DataFrame copy
   ↓
Clean CSV + Excel + EDA JSON
```

## Requirements

- Python 3.10–3.12 recommended.
- `GROQ_API_KEY` only for LLM-generated cleaning code.
- No API key is needed for the offline dry-run.
- Internet access is needed for Groq mode.

## Windows PowerShell setup

```powershell
git clone https://github.com/shallow2024/Day8-autonomous-csv-data-cleaner-eda-engine.git
cd D:\Day8-autonomous-csv-data-cleaner-eda-engine

py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run without an API key first

The offline mode uses a deterministic conservative cleaner and is the recommended first test:

```powershell
.\.venv\Scripts\python.exe data_cleaner.py examples\messy_customers.csv --dry-run
```

It writes:

```text
outputs\cleaned_data.csv
outputs\cleaned_data.xlsx
outputs\eda_metadata.json
```

The Excel file is included for convenient inspection in Microsoft Excel.

## Run with Groq

Set the API key for the current PowerShell session without writing it into the repository:

```powershell
$env:GROQ_API_KEY = "your-key-here"
```

Then run:

```powershell
.\.venv\Scripts\python.exe data_cleaner.py examples\messy_customers.csv
```

Optional model selection:

```powershell
.\.venv\Scripts\python.exe data_cleaner.py examples\messy_customers.csv --model llama-3.3-70b-versatile
```

The program sends only the EDA metadata and sample rows to Groq, not the entire CSV. Review your data governance requirements before sending metadata externally.

## What the agent checks

The metadata includes:

- Row and column counts.
- Column names and pandas dtypes.
- Null counts.
- Exact duplicate count.
- Numeric-column skewness.
- Three sample rows.

The generated code must preserve the DataFrame variable `df`. Before execution, the AST validator rejects:

- File or network access.
- `eval`, `exec`, imports other than `import pandas as pd`.
- Functions, classes, loops, `try`, `with`, and other unsupported constructs.
- Calls outside the explicit pandas/DataFrame allow-list.

## Test the project

Tests use no Groq API and do not access files outside the test fixture:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Limitations

- The AST allow-list is a project demonstration, not a complete security boundary. For production, run untrusted code in a separate locked-down process/container with OS-level restrictions, resource limits, and no secrets.
- Automatic cleaning can change meaning if a column is misclassified. Review the generated code and output before using it.
- Skewness is a descriptive statistic, not proof that a transformation is appropriate.
- The example dataset is intentionally small. Use a governed, representative dataset for real analysis.

## Portfolio value

This project demonstrates:

- Automated EDA and data-quality profiling.
- LLM-assisted code generation.
- AST-based validation before execution.
- In-memory pandas transformation.
- CSV and Excel deliverables.
- Reproducible API-free tests.

## License

MIT. See [LICENSE](LICENSE). Provider and data-governance terms remain applicable.
