"""
Shared helpers for building Jupyter notebooks programmatically.

Both build_handcoded.py and build_agent.py use these primitives so the
two tracks render with identical nbformat structure.
"""

from __future__ import annotations

import json
import pathlib
import textwrap


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).strip("\n").splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(src).strip("\n").splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    for i, cell in enumerate(cells):
        cell.setdefault("id", f"cell-{i:03d}")
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(out_dir: pathlib.Path, name: str, cells: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    out.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    print(f"  wrote {out}")
