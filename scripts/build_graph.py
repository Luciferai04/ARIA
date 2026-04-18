#!/usr/bin/env python3
"""
scripts/build_graph.py
──────────────────────
One-time offline script to build GraphRAG entity-relationship index.
Run ONCE before enabling graph_enabled in config.yaml:
    python scripts/build_graph.py

This takes ~30 min on first run depending on corpus size and API rate limits.
Requires: pip install graphrag
Requires: OPENAI_API_KEY in .env (graphrag uses OpenAI for entity extraction)
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "graphrag"


def main():
    print("=" * 60)
    print("ARIA GraphRAG Index Builder")
    print("=" * 60)

    if not RAW_DIR.exists() or not list(RAW_DIR.glob("*")):
        print(f"No documents found in {RAW_DIR}. Add PDFs/text files first.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import subprocess

        print(f"\nIndexing documents from: {RAW_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")
        print("\nThis may take 20-30 minutes...\n")

        # Initialize graphrag workspace
        result = subprocess.run(
            ["python", "-m", "graphrag.index",
             "--root", str(OUTPUT_DIR),
             "--input", str(RAW_DIR)],
            capture_output=False, text=True, timeout=3600
        )

        if result.returncode == 0:
            print(f"\nGraphRAG index built successfully at {OUTPUT_DIR}")
            print("Enable graph_enabled: true in config.yaml to use it.")
        else:
            print(f"\nGraphRAG indexing failed with exit code {result.returncode}")

    except ImportError:
        print("graphrag not installed. Run: pip install graphrag")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
