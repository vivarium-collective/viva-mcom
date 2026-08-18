"""Custom bigraph-schema type registration for viva-mcom.

MCoM's ports use only built-in types (``float``, ``maybe[float]``,
``map[string,float]``), so there are no custom types to register. The
module-level ``register_types`` hook is kept so ``build_core`` and
``recursive_dynamic_import`` have a stable entry point (and a place to add
types later, e.g. a units-annotated concentration type).
"""

from __future__ import annotations


def register_types(core):
    # No custom types required; return the core unchanged (required contract).
    return core
