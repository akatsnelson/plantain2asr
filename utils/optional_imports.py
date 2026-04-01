from __future__ import annotations

import importlib
from typing import Dict, Tuple


OptionalExportMap = Dict[str, Tuple[str, str]]


def resolve_optional_export(
    package_name: str,
    export_name: str,
    export_map: OptionalExportMap,
):
    """
    Lazily resolve optional exports for package-level __getattr__ handlers.

    Args:
        package_name: Absolute package name such as ``plantain2asr``.
        export_name: Requested attribute name.
        export_map: Mapping ``name -> (module_path, install_hint)`` where
            ``module_path`` is package-relative (e.g. ``".analysis"``).
    """
    if export_name not in export_map:
        raise AttributeError(f"module '{package_name}' has no attribute '{export_name}'")

    module_path, install_hint = export_map[export_name]
    try:
        module = importlib.import_module(module_path, package_name)
        return getattr(module, export_name)
    except ImportError as exc:
        raise ImportError(
            f"{export_name} requires optional dependencies. {install_hint}"
        ) from exc
