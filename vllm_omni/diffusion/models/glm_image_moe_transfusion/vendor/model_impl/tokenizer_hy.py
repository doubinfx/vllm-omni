"""Tokenizer loading that is safe when ranks share the HF dynamic module cache."""

import importlib
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, Union

from filelock import FileLock
from transformers import AutoTokenizer, dynamic_module_utils

from . import base as logging


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.get_logger(__name__)
_TOKENIZER_LOAD_LOCK = threading.RLock()
_CACHE_LOCKS: Dict[Path, FileLock] = {}


def load_tokenizer(tokenizer_path: Union[str, Path], *inputs: Any, **kwargs: Any) -> "PreTrainedTokenizerBase":
    """Load with unchanged AutoTokenizer options, serializing shared-cache access.

    Transformers copies local remote-code modules with shutil.copy, which first
    truncates the destination. Its import lock only protects threads in one
    process; another rank can import an empty/partial file during that copy.
    Lock the actual modules cache, rather than the checkpoint or rank group:
    distinct jobs and checkpoints with the same basename can share this cache.
    No process-group initialization or distributed barrier is needed.
    """
    cache = Path(dynamic_module_utils.HF_MODULES_CACHE).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with _TOKENIZER_LOAD_LOCK:
        # Reuse the FileLock object so a custom tokenizer that recursively loads
        # another tokenizer in the same thread can reenter the cache lock.
        cache_lock = _CACHE_LOCKS.setdefault(cache, FileLock(str(cache / ".veomni-tokenizer.lock")))
        return _load_under_cache_lock(cache_lock, tokenizer_path, inputs, kwargs)


def _load_under_cache_lock(cache_lock, tokenizer_path, inputs, kwargs):
    with cache_lock:
        try:
            return AutoTokenizer.from_pretrained(tokenizer_path, *inputs, **kwargs)
        except AttributeError as error:
            module = getattr(error, "obj", None)
            if not (
                isinstance(module, ModuleType)
                and module.__name__.startswith(dynamic_module_utils.TRANSFORMERS_DYNAMIC_MODULE_NAME + ".")
                and hasattr(module, "__transformers_module_hash__")
                and sys.modules.get(module.__name__) is module
            ):
                raise

            # A previous unprotected load can hash the complete file, execute
            # it while empty, then cache that hash on an empty module. Repairing
            # the disk file alone does not make HF reload this poisoned module.
            sys.modules.pop(module.__name__)
            importlib.invalidate_caches()
            logger.warning("Reloading cached tokenizer module after missing attribute: %s", module.__name__)
            # Retry only once. An actually missing class or invalid checkpoint
            # still raises; do not retry unrelated tokenizer exceptions.
            return AutoTokenizer.from_pretrained(tokenizer_path, *inputs, **kwargs)
