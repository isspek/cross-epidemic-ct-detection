from pathlib import Path
from langchain.cache import SQLiteCache

from code.pysettings import LLM_CACHE_FOLDER

_LANGCHAIN_SQLITE_CACHE = None

def _create_langchain_sqlite_cache():
    return SQLiteCache(database_path=Path(LLM_CACHE_FOLDER)/"langchain-sqlite-cache.db")

def langchain_sqlite_cache():
    global _LANGCHAIN_SQLITE_CACHE
    if _LANGCHAIN_SQLITE_CACHE is None:
        _LANGCHAIN_SQLITE_CACHE = _create_langchain_sqlite_cache()
    return _LANGCHAIN_SQLITE_CACHE