# Contributing

Thank you for your interest in contributing to this project. This is a research replication package for epistemic network models in philosophy of science — contributions are welcome in several forms, from bug reports to new replications.

---

## Ways to contribute

### Report a bug

If the model produces unexpected results, a simulation crashes, or the dashboard behaves incorrectly, please [open an issue](https://github.com/sprologuglie/Interactive-Epistemic-ABM-Zollman-Using-Mesa/issues) with:

- A minimal reproducible example (model parameters + seed that reproduce the problem)
- The full error message and traceback if applicable
- Your Python and Mesa version (`python --version`, `pip show mesa`)

### Report a replication discrepancy

If you find a difference between the results produced by this package and those reported in one of the original papers, please open an issue labelled **replication discrepancy** including:

- Which paper and which figure or table is affected
- The parameters you used
- A description of what you expected vs. what you obtained
- Whether you think the discrepancy is due to implementation choices, parameter mapping, or something else

Replication discrepancies are scientifically valuable — even unresolved ones are worth documenting.

### Suggest a feature or extension

Open an issue labelled **enhancement** describing:

- What you want to add and why it is scientifically motivated
- Which paper or theoretical argument supports the extension
- Whether you plan to implement it yourself

### Contribute code

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes. See the code guidelines below.

3. Add or update tests in `tests/`. All existing tests must pass:
   ```bash
   pytest tests/
   ```

4. Open a pull request against `main` with a clear description of what you changed and why.

---

## Code guidelines

**Style:** follow [PEP 8](https://peps.python.org/pep-0008/). Method names use `snake_case`. The codebase has some legacy `PascalCase` method names that are being phased out — new code should use `snake_case`.

**Naming:** parameter names should match those used in the original papers where possible. Deviations must be documented in the docstring and in the README parameter mapping table.

**Docstrings:** all methods should have a one-line docstring explaining what they do, not how.

**Two model versions:** the repository maintains two separate model implementations — `scripts/model.py` for the dashboard and `batch_run/batch_run_scripts/model.py` for batch runs. Changes to model logic should be applied to both, or clearly motivated if they apply to only one.

**No breaking changes to the batch run API:** the `Bandit` model constructor signature is used by the replication notebooks. Adding parameters is fine as long as they have sensible defaults that preserve existing behaviour.

---

## Adding a replication notebook

If you want to add a replication of a paper not currently covered:

1. Create a new notebook in `notebooks/` following the structure of the existing ones (header → parameters table → one section per analysis → discrepancies → conclusions).
2. Import the model from `batch_run_scripts.model` — do not copy the model code inline.
3. Document parameter mapping explicitly, including any deviations and their justification.
4. Report all discrepancies honestly, including unresolved ones.
5. Add the paper to the **Replicated papers** section of the README.

---

## Running the tests

```bash
# All tests
pytest tests/

# Specific file
pytest tests/test_model.py

# With output
pytest tests/ -v
```

The test suite covers both model versions. If you add a new parameter, add a corresponding test.

---

## Questions

For questions about the model or the replication methodology, open a discussion on GitHub or contact the maintainer via the information in [`CITATION.cff`](CITATION.cff).