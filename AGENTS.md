# AGENTS.md

This role of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in the project that surprises you, please alert the developer working with you and indicate that this is the case in the Agent.md file to help prevent future agents from having the same issue.

## Commands to interact with the codebase which you should run:

### Build Docs (only use this command verbatim from the project root)

```bash
rm -rf docs/source/generated docs/source/auto_examples && uv run sphinx-build -b html docs/source docs/build/html
```

### Run Pre-commit (takes only 3s)

```bash
uv run pre-commit run --all-files
```

## Environment notes

- 2026-06-16: In one Windows PowerShell session, `uv`, `python`, and `py` were not
  available on PATH, and `.venv\Scripts\python.exe` pointed at a missing Python 3.12
  install. If verification commands fail this way, repair the Python/uv environment
  before assuming the tests themselves are broken.
