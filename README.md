# Ikaria

Ikaria is the home of the myth of Icarus and his the risky endeavour. As such, it symbolises the journey of achievements and failure. Not every flight is permitted. But they all fly.

## Structure

### Actual

- `style-transfer`: Experiments on transferring styles between images.

### Guidelines

- Repository:
  - Group experiments according to topics etc.
  - Path, example: `style-transfer/legacy/`
  - Each study gets a single project.
- Studies:
  - Notebook structure:
    - Header: Inform about essentials from goal over results, structure to changelog and what is required to get it run.
    - Preparation: Gather imports and relevant global variables.
    - Data fetching and preparations: Load and transform right here.
    - Study.
    - Evaluation.
- Coding style:
  - Try to separate code logic from conceptual logic. A functional supports this. Keep functions short.
  - Choose good names for functions and variables.
  - Annotate functions, add short docstrings.
  - If code is re-used several times, provide that in our libraries.
- Provide changelog

```markdown
| version | date | change |
|---------|------|--------|
| 1 | 2024-10-18 | initial creation |
| 2 | 2024-11-14 | ... |
| - |  |  |
```

### Libraries

- Local libraries allow to keep notebooks simple, but may hide important details.
- Use them to pre-structure code that can be moved to big libraries.

## Observed code sweets and sours

### Environment

<!--
1. `netcdf4`: The update from `1.7.2` to `1.7.3` made reading `netcdf` files more rigorous causing troubles reading own `netcdf` files. Solve this issue for allowing further updates.
-->

### Linting

<!--
1. `@typing.overload`:
   - `mypy` does not distinguish between `np.array` and `pd.DataFrame`, but `mypy` needs a subclassing hierarchy. Resolve (ignore) by `# type: ignore[overload-cannot-match]` at subsequent overloads.
   - `mypy` does not distingiush between `pd.Series` and `pd.DataFrame`. If static types are not that relevant, use `PANDA = typing.TypeVar("PANDA", pd.Series, pd.DataFrame)`.
   - `flake8` does not like ellipses on the same line as the function definition. Resolve (ignore) by `# noqa: E704`. Needs to follow `mypy` ignores.
2. Function signature not correctly identified by `mypy`. Resolve (ignore) by `# type: ignore [call-arg]`.
3. `docsig`: To skip checks on docstrings _temporarily_, use the directive `# docsig: disable`.
-->