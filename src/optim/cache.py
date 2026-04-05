"""
On-disk tokenized-prompt cache for PhysBench.

PhysBench val is static. Tokenizing ~200 prompts takes a few seconds per run,
but multiplied by every debug / smoke / ablation run during a 2-week sprint
that's hours of wasted time. This module caches tokenized prompts keyed by
(model_name, prompt_hash) so every script after the first eats the cost once.

Key design choice: we cache ONLY the tokenized text. Image tensors are huge
(~1 MB each) and cheap to re-process with the vision encoder; text tokens are
small and expensive to re-tokenize for chat-template VLMs.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional


class PromptCache:
    """Simple file-backed prompt cache.

    Layout:
        cache_dir/{model_safe_name}/
            index.json         {prompt_hash: relative_pkl_path}
            chunks/*.pkl       pickled dict: {"input_ids": [...], "attention_mask": [...], ...}

    Chunks are per-hash so concurrent writers (never used in this project but
    just in case) don't clobber each other. JSON index is updated atomically
    via write-to-temp + rename.
    """

    def __init__(self, cache_dir: Path, model_name: str):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
        self.root = Path(cache_dir) / safe
        self.chunks_dir = self.root / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self._index: Dict[str, str] = self._load_index()

    def _load_index(self) -> Dict[str, str]:
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_index(self) -> None:
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._index))
        tmp.replace(self.index_path)

    @staticmethod
    def hash_prompt(prompt: str, extra: str = "") -> str:
        h = hashlib.sha256()
        h.update(prompt.encode("utf-8"))
        if extra:
            h.update(b"\x00")
            h.update(extra.encode("utf-8"))
        return h.hexdigest()[:32]

    def get(self, prompt: str, extra: str = "") -> Optional[Dict[str, Any]]:
        key = self.hash_prompt(prompt, extra)
        rel = self._index.get(key)
        if rel is None:
            return None
        path = self.root / rel
        if not path.exists():
            # Stale index entry — drop it.
            self._index.pop(key, None)
            self._save_index()
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def put(self, prompt: str, tokenized: Dict[str, Any], extra: str = "") -> None:
        key = self.hash_prompt(prompt, extra)
        rel = f"chunks/{key}.pkl"
        path = self.root / rel
        with open(path, "wb") as f:
            pickle.dump(tokenized, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._index[key] = rel
        self._save_index()

    def __len__(self) -> int:
        return len(self._index)

    def clear(self) -> None:
        """Wipe all cached entries for this model."""
        import shutil
        if self.chunks_dir.exists():
            shutil.rmtree(self.chunks_dir)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self._index = {}
        self._save_index()
