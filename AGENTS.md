# AGENTS.md
Repository guidance for coding agents working in `mammoth`.

## Scope
- Applies to the entire repository.
- Primary language: Python (>=3.10).
- Packaging/build metadata lives in `pyproject.toml` (setuptools backend).
- Main runtime entrypoint: `main.py`.
- Main tests: `tests/` (`pytest`).

## Environment and Setup
Preferred setup uses `uv` (`uv.lock` is present).

1. Install `uv` (if needed):
   - `pip install uv`
2. Sync dependencies (including dev group):
   - `uv sync --group dev`
3. Run all commands through the environment:
   - `uv run <command>`

Fallback (pip + venv):
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Optional extras/dev:
   - `pip install -r requirements-optional.txt`
   - `pip install pytest mypy ruff autopep8`

## Build Commands
There is no dedicated package build CI job, but these are useful checks:

- Editable install:
  - `uv run pip install -e .`
- Validate packaging/build artifacts:
  - `uv run python -m build`
- Entrypoint sanity check:
  - `uv run python main.py --help`

Docs build (matches GitHub workflow):
- `cd docs && uv run ../utils/args.py && uv run sphinx-build -j auto . ../_build`
- Alternative via Makefile:
  - `cd docs && make html`

## Test Commands
Baseline (full suite):
- `uv run pytest`

Quick smoke tests:
- `uv run pytest tests/test_import.py`
- `uv run pytest tests/test_basic_functionality.py`

Single test file:
- `uv run pytest tests/test_er_example.py`

Single test function (node id):
- `uv run pytest tests/test_er_example.py::test_er[seq-mnist]`
- Generic form: `uv run pytest path/to/test_file.py::test_name`

By name pattern:
- `uv run pytest -k "checkpoint and not wandb"`

Repo-specific pytest options:
- `--force_device=<id>` sets `MAMMOTH_DEVICE` and `CUDA_VISIBLE_DEVICES`.
- `--include_dataset_reload` enables the long dataset re-download path.

Examples:
- `uv run pytest --force_device=0`
- `uv run pytest tests/test_datasets.py --include_dataset_reload`

## Lint and Format Commands
Formatting in this repository is autopep8-based.

Format exactly as README/CI expects:
- `uv run autopep8 --recursive --in-place --aggressive --max-line-length=200 --ignore=E402 .`

Lint/type tooling defined in `pyproject.toml`:
- Ruff:
  - `uv run ruff check .`
- Mypy:
  - `uv run mypy .`

Notes:
- Ruff config currently only specifies lint ignore `E402`.
- Do not assume Black or isort conventions here.

## Code Style Guidelines

### Imports
- Group imports as stdlib, third-party, local modules.
- Keep one import per line when practical.
- `E402` is intentionally tolerated (some modules adjust `sys.path` early).
- Use lazy imports only when needed (cycle break, heavy optional dependency, etc.).
- Prefer `TYPE_CHECKING` for type-only imports.

### Formatting
- Match local style in the touched file; avoid repo-wide style rewrites.
- Effective line-length target is 200 (autopep8 config).
- Keep docstrings where they improve clarity for modules/classes/functions.
- Avoid introducing cosmetic-only diffs in unrelated sections.

### Types and Signatures
- Add type hints to new/modified public APIs where feasible.
- Follow existing typing patterns (`Optional`, `Union`, `TYPE_CHECKING`, forward refs).
- Preserve framework-compatible method signatures, especially in model/dataset hooks.
- Do not introduce strict typing requirements that conflict with current codebase norms.

### Naming Conventions
- Functions/variables/modules: `snake_case`.
- Classes: `PascalCase`.
- Constants/class constants: `UPPER_SNAKE_CASE`.
- Model classes conventionally define:
  - `NAME = '<model-id>'`
  - `COMPATIBILITY = [...]`
- Dataset classes conventionally define base fields (`SETTING`, `N_TASKS`, etc.).

### Error Handling and Validation
- Follow existing patterns:
  - Use `assert` for internal invariants and argument sanity checks.
  - Raise explicit exceptions (`ValueError`, `NotImplementedError`, etc.) for invalid states.
- Keep fail-fast validation behavior in CLI/model/dataset paths.
- Preserve current error messages unless behavior is intentionally changed.

### Logging and Side Effects
- Prefer `logging` over `print` for runtime diagnostics.
- Respect environment-driven logging (`LOG_LEVEL`, `MAMMOTH_TEST`).
- Keep side effects explicit: checkpoint paths, base paths, env vars, device selection.

### Testing Expectations for Changes
- Add/adjust tests under `tests/` when behavior changes.
- Start with targeted tests, then broaden scope.
- Run expensive dataset-reload tests only when relevant.

### Performance and Device Assumptions
- GPU/CUDA is common; preserve CPU fallback behavior.
- Do not hardcode a single device when device selection is already abstracted.
- Preserve reproducibility hooks (`seed`, deterministic loader behavior) in training/data code.

## Repository Architecture Hints
- `main.py`: argument parsing, initialization, training orchestration.
- `models/`: continual learning algorithms and model utilities.
- `datasets/`: continual dataset definitions and task splitting.
- `backbone/`: network backbones and registration helpers.
- `utils/`: args, logging, training loop, checkpoints, metrics, misc helpers.

When adding a model/dataset/backbone, follow existing registration and parser patterns.

## Cursor/Copilot Rule Files
Checked locations:
- `.cursor/rules/`
- `.cursorrules`
- `.github/copilot-instructions.md`

At time of writing, none of these files exist in this repository.
If they are added later, treat them as higher-priority repository instructions and update this file.

## Agent Workflow Recommendations
- Read nearby code before editing; match existing conventions.
- Prefer minimal, focused diffs over broad refactors.
- Run lint/tests relevant to touched areas.
- Include exact commands in PR notes when useful for reproducibility.
