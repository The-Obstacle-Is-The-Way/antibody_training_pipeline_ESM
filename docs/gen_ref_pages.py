"""
Auto-generate API reference pages for MkDocs.

This script automatically generates API documentation markdown files
from Python source code in src/antibody_training_esm/.

Usage:
    Called automatically by mkdocs-gen-files plugin during build.
"""

from pathlib import Path

import mkdocs_gen_files

# Root of the Python package
nav = mkdocs_gen_files.Nav()
src = Path(__file__).parent.parent / "src"

# Iterate through all Python files in the package
for path in sorted(src.rglob("*.py")):
    # Get module path relative to src/
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src / "antibody_training_esm").with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    # Get the Python import path (e.g., "antibody_training_esm.core.trainer")
    parts = tuple(module_path.parts)

    # Skip __init__.py files (they don't need dedicated pages)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].startswith("_"):
        # Skip private modules (e.g., _utils.py)
        continue

    # Add to navigation
    nav[parts] = doc_path.as_posix()

    # Create the API reference markdown file
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # Create the module identifier (e.g., "antibody_training_esm.core.trainer")
        identifier = ".".join(parts)

        # Write the markdown content
        print(f"::: {identifier}", file=fd)

    # Set edit path to source code (not generated markdown)
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(src.parent))

# Write the navigation file (SUMMARY.md) for literate-nav plugin
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
