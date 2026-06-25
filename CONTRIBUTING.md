# Contributing

Contributions are welcome after the project is made public.

## Development setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Running experiments

```bash
python scripts/run_all.py
```

## Contribution guidelines

- Keep simulation claims separate from field-validation claims.
- Do not add private farm data, real UAV images or sensitive sensor data unless they are explicitly anonymized and licensed for release.
- Preserve reproducibility: update scripts, tables and figures together when experiment logic changes.
- Add a checkpoint or short report for major experiment additions.
- Use clear file names for new figures and tables.
