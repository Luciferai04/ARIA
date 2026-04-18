# aria/parent_store.py
# Pickle-backed InMemoryStore for ParentDocumentRetriever parent documents.

import pickle
from pathlib import Path
from langchain.storage import InMemoryStore

STORE_PATH = Path(__file__).parent.parent / "data" / "parent_store.pkl"

_store = None  # module-level singleton


def get_parent_store() -> InMemoryStore:
    """Load or create the parent document store."""
    global _store
    if _store is None:
        _store = InMemoryStore()
        if STORE_PATH.exists():
            try:
                with open(STORE_PATH, "rb") as f:
                    data = pickle.load(f)
                    for key, value in data.items():
                        _store.mset([(key, value)])
                print(f"[ParentStore] Loaded {len(data)} parent documents from disk.")
            except Exception as e:
                print(f"[ParentStore] Could not load existing store: {e}")
    return _store


def save_parent_store():
    """Persist the parent store to disk."""
    global _store
    if _store is not None:
        try:
            STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Extract all items from InMemoryStore
            items = {}
            for key in list(_store.store.keys()):
                items[key] = _store.store[key]
            with open(STORE_PATH, "wb") as f:
                pickle.dump(items, f)
            print(f"[ParentStore] Saved {len(items)} parent documents to disk.")
        except Exception as e:
            print(f"[ParentStore] Save failed: {e}")
