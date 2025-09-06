import os
import enum
import json
import time
import atexit

from grappa import should
from typing import Set, Dict
from collections import defaultdict


CACHE_FILE = "app/qualify/cache.json"
TIME_DELTA = 10  # seconds


@enum.unique
class QualifyStatus(str, enum.Enum):
    GOOD = "GOOD"
    """Qualified for unit testing.
    
    MOSA can generate tests for this module without any matter.
    """

    BAD = "BAD"
    """Unqualified for unit testing
    
    Pynguin return SETUP_FAILED when trying to generate tests,
    or MOSA managed achieve 100% branch coverage on this module.
    """

    NOT_QUALIFIED = "NOT_QUALIFIED"
    """Need qualification by Pynguin"""


def _dict_with_ignored() -> Dict[str, int | Set]:
    return {'__ignored__': set()}


class QualifyCache:
    def __init__(self):
        self.is_write_cache_disabled: bool = False
        self.__data = defaultdict(_dict_with_ignored)
        if not os.path.exists(CACHE_FILE):
            self._time_until_next_update = 0
        else:
            with open(CACHE_FILE, "r") as file:
                for k, v in json.load(file).items():
                    self.__data[k] = v
                    self.__data[k]['__ignored__'] = \
                        set(v['__ignored__'])
            self._time_until_next_update = int(time.time()) + TIME_DELTA
        self._project_name: str = ""

    def disable_write_cache(self):
        self.is_write_cache_disabled | should.be.false
        self.is_write_cache_disabled = True

    def enable_write_cache(self):
        self.is_write_cache_disabled | should.be.true
        self.is_write_cache_disabled = False

    @property
    def _data(self):
        return self.__data[self._project_name]

    def set_project_name(self, project_name: str):
        self._project_name = project_name

    def set_project_path(self, project_path: str):
        tmp = project_path.split('/')
        self.set_project_name(tmp[-1])

    def _to_dict(self):
        ret = {}
        for k, v in self.__data.items():
            ret[k] = v
            ret[k]['__ignored__'] = \
                sorted(list(v['__ignored__']))
        return ret

    def _write_cache(self, now: int = None):
        data = self._to_dict()
        if not self.is_write_cache_disabled:
            with open(CACHE_FILE, "w") as file:
                json.dump(data, file, indent=4)
        if now is None:
            now = time.time()
        self._time_until_next_update = now + TIME_DELTA

    def _lazy_write_cache(self):
        now = int(time.time())
        if now >= self._time_until_next_update:
            self._write_cache(now)

    def get_status(self, module_name: str):
        if module_name in self._data:
            v = self._data[module_name]
            return QualifyStatus.GOOD if v else QualifyStatus.BAD
        elif module_name in self._data['__ignored__']:
            return QualifyStatus.BAD
        else:
            return QualifyStatus.NOT_QUALIFIED

    def add_good(self, module_name: str):
        assert self.get_status(module_name) != QualifyStatus.BAD
        self._data[module_name] = 1
        self._lazy_write_cache()

    def add_bad(self, module_name: str):
        assert self.get_status(module_name) != QualifyStatus.GOOD
        self._data[module_name] = 0
        self._lazy_write_cache()

    def add_ignored(self, module_name: str):
        assert self.get_status(module_name) == QualifyStatus.GOOD
        del self._data[module_name]
        self._data['__ignored__'].add(module_name)


_cache = QualifyCache()
enable_write_cache = _cache.enable_write_cache
disable_write_cache = _cache.disable_write_cache
get_status = _cache.get_status
add_good = _cache.add_good
add_bad = _cache.add_bad
add_ignored = _cache.add_ignored
set_project_name = _cache.set_project_name
set_project_path = _cache.set_project_path

# Make sure to cache the last time before exit
atexit.register(_cache._write_cache)
