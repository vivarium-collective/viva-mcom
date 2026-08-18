# Contributing to viva-mcom

## Development setup

uv is required. Install with `brew install uv` or `pip install uv`.

    uv venv .venv
    source .venv/bin/activate
    uv pip install -e ".[dev]"
    pytest

Run the fidelity suite specifically (compares the reproduction against real MCoM
Julia output in `datasets/golden`):

    pytest tests/test_golden.py -v

## Fidelity first

This package is a **clean-room reproduction** (see `REPRODUCTION.md`). Any change
to `viva_mcom/engine.py` or `viva_mcom/config.py` must keep `tests/test_golden.py`
green — those tests assert bit-for-bit agreement with the upstream Julia engine
for deterministic configs. If you extend the reproduction to a new config path,
add a golden fixture (a real MCoM Julia run + its config) rather than a
self-referential test.

The bare name `MComProcess` is reserved for a future bridge to the real Julia
engine; keep reproduction code under the `MComReproductionProcess` /
`…Reproduction…` names.

## Releasing to PyPI

Tag a commit with `git tag v<VERSION>` and push the tag. The
`.github/workflows/release.yml` workflow publishes to PyPI automatically using
trusted publishing (no tokens needed after initial setup). PyPI trusted
publishing must be configured once per repo — see
<https://docs.pypi.org/trusted-publishers/>.
