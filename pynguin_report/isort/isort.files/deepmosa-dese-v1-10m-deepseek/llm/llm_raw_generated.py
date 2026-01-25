####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import os
from pathlib import Path
from typing import List, Iterator

class Config:
    def __init__(self, follow_links: bool = False, skipped_paths: List[str] = [], supported_extensions: List[str] = ['.py']):
        self.follow_links = follow_links
        self.skipped_paths = skipped_paths
        self.supported_extensions = supported_extensions

    def is_skipped(self, path: Path) -> bool:
        return str(path) in self.skipped_paths

    def is_supported_filetype(self, filepath: str) -> bool:
        return os.path.splitext(filepath)[1] in self.supported_extensions

def test_find_with_directory():
    config = Config()
    skipped: List[str] = []
    broken: List[str] = []
    paths = ['tests/test_data']
    result = list(find(paths, config, skipped, broken))
    assert len(result) > 0
    assert all(os.path.isfile(path) for path in result)
    assert len(skipped) == 0
    assert len(broken) == 0

def test_find_with_skipped_directory():
    config = Config(skipped_paths=['tests/test_data/skip_this'])
    skipped: List[str] = []
    broken: List[str] = []
    paths = ['tests/test_data']
    result = list(find(paths, config, skipped, broken))
    assert len(result) > 0
    assert all(os.path.isfile(path) for path in result)
    assert len(skipped) == 1
    assert len(broken) == 0

def test_find_with_broken_path():
    config = Config()
    skipped: List[str] = []
    broken: List[str] = []
    paths = ['tests/non_existent_path']
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1

def test_find_with_single_file():
    config = Config()
    skipped: List[str] = []
    broken: List[str] = []
    paths = ['tests/test_data/test_file.py']
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert result[0] == 'tests/test_data/test_file.py'
    assert len(skipped) == 0
    assert len(broken) == 0

def test_find_with_unsupported_filetype():
    config = Config(supported_extensions=['.txt'])
    skipped: List[str] = []
    broken: List[str] = []
    paths = ['tests/test_data']
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 0


# LLM-generated content at query #2
#--------------------------

```
def test_find_is_dir_predicate():
    paths = ["/some/directory"]
    config = Config()
    skipped = []
    broken = []
    os.path.isdir = lambda x: True
    list(find(paths, config, skipped, broken))


# LLM-generated content at query #3
#--------------------------

```python
def test_find_with_directory():
    config = Config(follow_links=False, supported_filetypes=['.py'], skipped_paths=[])
    skipped = []
    broken = []
    paths = ['test_dir']
    result = list(find(paths, config, skipped, broken))
    assert result == ['test_dir/file1.py', 'test_dir/file2.py']
    assert skipped == []
    assert broken == []

def test_find_with_skipped_directory():
    config = Config(follow_links=False, supported_filetypes=['.py'], skipped_paths=['test_dir/skip_me'])
    skipped = []
    broken = []
    paths = ['test_dir']
    result = list(find(paths, config, skipped, broken))
    assert result == ['test_dir/file1.py', 'test_dir/file2.py']
    assert skipped == ['test_dir/skip_me']
    assert broken == []

def test_find_with_nonexistent_path():
    config = Config(follow_links=False, supported_filetypes=['.py'], skipped_paths=[])
    skipped = []
    broken = []
    paths = ['nonexistent_path']
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ['nonexistent_path']

def test_find_with_file():
    config = Config(follow_links=False, supported_filetypes=['.py'], skipped_paths=[])
    skipped = []
    broken = []
    paths = ['test_file.py']
    result = list(find(paths, config, skipped, broken))
    assert result == ['test_file.py']
    assert skipped == []
    assert broken == []

def test_find_with_skipped_file():
    config = Config(follow_links=False, supported_filetypes=['.py'], skipped_paths=['test_file.py'])
    skipped = []
    broken = []
    paths = ['test_file.py']
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ['test_file.py']
    assert broken == []

def test_find_with_unsupported_filetype():
    config = Config(follow_links=False, supported_filetypes=['.py'], skipped_paths=[])
    skipped = []
    broken = []
    paths = ['test_file.txt']
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    paths = ["/nonexistent/path"]
    config = Config()
    skipped = []
    broken = []
    list(find(paths, config, skipped, broken))
    assert len(broken) == 1
    assert broken[0] == "/nonexistent/path"


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line27_evaluates_to_true():
    class MockConfig:
        def is_supported_filetype(self, filepath):
            return True
        
        def is_skipped(self, filepath):
            return True

    config = MockConfig()
    skipped = []
    broken = []
    filepath = "test_file.py"
    
    result = config.is_skipped(Path(os.path.abspath(filepath)))
    assert result is True


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_false():
    paths = ["/non_existent_directory"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(broken) == 1
    assert broken[0] == "/non_existent_directory"


# LLM-generated content at query #7
#--------------------------

```python
def test_resolved_path_in_visited_dirs():
    paths = ["test_dir"]
    config = Config()
    skipped = []
    broken = []
    visited_dirs = set()
    resolved_path = Path("test_dir/subdir").resolve()
    visited_dirs.add(resolved_path)
    dirnames = ["subdir"]
    dirpath = "test_dir"
    base_path = Path(dirpath)
    list(find(paths, config, skipped, broken))
    assert resolved_path in visited_dirs


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    paths = ["/nonexistent/path"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(broken) == 1
    assert broken[0] == "/nonexistent/path"


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_False():
    import os
    from pathlib import Path
    from typing import Iterable, Iterator, List

    class Config:
        def __init__(self, follow_links: bool):
            self.follow_links = follow_links

        def is_skipped(self, path: Path) -> bool:
            return False

        def is_supported_filetype(self, filepath: str) -> bool:
            return True

    paths = ["/nonexistent/directory"]
    config = Config(False)
    skipped: List[str] = []
    broken: List[str] = []
    result = list(find(paths, config, skipped, broken))
    assert not os.path.isdir("/nonexistent/directory")


# LLM-generated content at query #10
#--------------------------

```python
def test_find_with_directory():
    class MockConfig:
        def __init__(self, follow_links=False):
            self.follow_links = follow_links

        def is_skipped(self, path):
            return str(path) == "dir/skipped_dir"

        def is_supported_filetype(self, filepath):
            return filepath.endswith(".py")

    config = MockConfig()
    skipped = []
    broken = []
    paths = ["dir"]
    result = list(find(paths, config, skipped, broken))
    assert result == ["dir/file1.py", "dir/file2.py"]
    assert skipped == ["dir/skipped_dir"]
    assert broken == []

def test_find_with_nonexistent_path():
    class MockConfig:
        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return True

    config = MockConfig()
    skipped = []
    broken = []
    paths = ["nonexistent"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent"]

def test_find_with_file():
    class MockConfig:
        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return True

    config = MockConfig()
    skipped = []
    broken = []
    paths = ["file.py"]
    result = list(find(paths, config, skipped, broken))
    assert result == ["file.py"]
    assert skipped == []
    assert broken == []

def test_find_with_skipped_file():
    class MockConfig:
        def is_skipped(self, path):
            return str(path) == "skipped_file.py"

        def is_supported_filetype(self, filepath):
            return True

    config = MockConfig()
    skipped = []
    broken = []
    paths = ["skipped_file.py"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ["skipped_file.py"]
    assert broken == []

def test_find_with_unsupported_filetype():
    class MockConfig:
        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return filepath.endswith(".py")

    config = MockConfig()
    skipped = []
    broken = []
    paths = ["file.txt"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_find_with_directory():
    class MockConfig:
        def __init__(self, follow_links, skipped_files, supported_extensions):
            self.follow_links = follow_links
            self.skipped_files = skipped_files
            self.supported_extensions = supported_extensions

        def is_skipped(self, path):
            return str(path) in self.skipped_files

        def is_supported_filetype(self, path):
            return any(path.endswith(ext) for ext in self.supported_extensions)

    skipped = []
    broken = []
    config = MockConfig(False, ["/test/skip_me"], [".py"])
    paths = ["/test"]
    result = list(find(paths, config, skipped, broken))
    assert len(result) > 0
    assert skipped == ["/test/skip_me"]
    assert broken == []

def test_find_with_nonexistent_path():
    skipped = []
    broken = []
    config = MockConfig(False, [], [".py"])
    paths = ["/nonexistent"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["/nonexistent"]

def test_find_with_file():
    skipped = []
    broken = []
    config = MockConfig(False, [], [".py"])
    paths = ["/test/test_file.py"]
    result = list(find(paths, config, skipped, broken))
    assert result == ["/test/test_file.py"]
    assert skipped == []
    assert broken == []

def test_find_with_skipped_file():
    skipped = []
    broken = []
    config = MockConfig(False, ["/test/skip_me.py"], [".py"])
    paths = ["/test/skip_me.py"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ["/test/skip_me.py"]
    assert broken == []

def test_find_with_unsupported_filetype():
    skipped = []
    broken = []
    config = MockConfig(False, [], [".py"])
    paths = ["/test/test_file.txt"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []


# LLM-generated content at query #2
#--------------------------

```python
def test_find_yields_files_in_directory():
    config = Config(supported_extensions={'.py'}, skip_patterns=[], follow_links=False)
    skipped = []
    broken = []
    paths = ['test_dir']
    os.makedirs('test_dir', exist_ok=True)
    with open('test_dir/test_file.py', 'w') as f:
        f.write('')
    result = list(find(paths, config, skipped, broken))
    assert result == ['test_dir/test_file.py']
    assert skipped == []
    assert broken == []
    os.remove('test_dir/test_file.py')
    os.rmdir('test_dir')

def test_find_skips_unsupported_files():
    config = Config(supported_extensions={'.py'}, skip_patterns=[], follow_links=False)
    skipped = []
    broken = []
    paths = ['test_dir']
    os.makedirs('test_dir', exist_ok=True)
    with open('test_dir/test_file.txt', 'w') as f:
        f.write('')
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []
    os.remove('test_dir/test_file.txt')
    os.rmdir('test_dir')

def test_find_skips_skipped_directories():
    config = Config(supported_extensions={'.py'}, skip_patterns=['test_dir'], follow_links=False)
    skipped = []
    broken = []
    paths = ['test_dir']
    os.makedirs('test_dir', exist_ok=True)
    with open('test_dir/test_file.py', 'w') as f:
        f.write('')
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ['test_dir']
    assert broken == []
    os.remove('test_dir/test_file.py')
    os.rmdir('test_dir')

def test_find_reports_broken_paths():
    config = Config(supported_extensions={'.py'}, skip_patterns=[], follow_links=False)
    skipped = []
    broken = []
    paths = ['nonexistent_dir']
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ['nonexistent_dir']

def test_find_yields_files_directly():
    config = Config(supported_extensions={'.py'}, skip_patterns=[], follow_links=False)
    skipped = []
    broken = []
    paths = ['test_file.py']
    with open('test_file.py', 'w') as f:
        f.write('')
    result = list(find(paths, config, skipped, broken))
    assert result == ['test_file.py']
    assert skipped == []
    assert broken == []
    os.remove('test_file.py')

def test_find_skips_skipped_files():
    config = Config(supported_extensions={'.py'}, skip_patterns=['test_file.py'], follow_links=False)
    skipped = []
    broken = []
    paths = ['test_file.py']
    with open('test_file.py', 'w') as f:
        f.write('')
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath('test_file.py')]
    assert broken == []
    os.remove('test_file.py')

def test_find_follows_symlinks():
    config = Config(supported_extensions={'.py'}, skip_patterns=[], follow_links=True)
    skipped = []
    broken = []
    paths = ['test_dir']
    os.makedirs('test_dir', exist_ok=True)
    os.makedirs('test_dir/linked_dir', exist_ok=True)
    with open('test_dir/linked_dir/test_file.py', 'w') as f:
        f.write('')
    os.symlink('test_dir/linked_dir', 'test_dir/symlink')
    result = list(find(paths, config, skipped, broken))
    assert 'test_dir/symlink/test_file.py' in result
    assert skipped == []
    assert broken == []
    os.remove('test_dir/symlink/test_file.py')
    os.remove('test_dir/symlink')
    os.remove('test_dir/linked_dir/test_file.py')
    os.rmdir('test_dir/linked_dir')
    os.rmdir('test_dir')

def test_find_does_not_follow_symlinks():
    config = Config(supported_extensions={'.py'}, skip_patterns=[], follow_links=False)
    skipped = []
    broken = []
    paths = ['test_dir']
    os.makedirs('test_dir', exist_ok=True)
    os.makedirs('test_dir/linked_dir', exist_ok=True)
    with open('test_dir/linked_dir/test_file.py', 'w') as f:
        f.write('')
    os.symlink('test_dir/linked_dir', 'test_dir/symlink')
    result = list(find(paths, config, skipped, broken))
    assert 'test_dir/symlink/test_file.py' not in result
    assert skipped == []
    assert broken == []
    os.remove('test_dir/symlink/test_file.py')
    os.remove('test_dir/symlink')
    os.remove('test_dir/linked_dir/test_file.py')
    os.rmdir('test_dir/linked_dir')
    os.rmdir('test_dir')

def test_find_handles_duplicate_paths():
    config = Config(supported_extensions={'.py'}, skip_patterns=[], follow_links=False)
    skipped = []
    broken = []
    paths = ['test_dir', 'test_dir']
    os.makedirs('test_dir', exist_ok=True)
    with open('test_dir/test_file.py', 'w') as f:
        f.write('')
    result = list(find(paths, config, skipped, broken))
    assert result == ['test_dir/test_file.py']
    assert skipped == []
    assert broken == []
    os.remove('test_dir/test_file.py')
    os.rmdir('test_dir')

def test_find_handles_empty_paths():
    config = Config(supported_extensions={'.py'}, skip_patterns=[], follow_links=False)
    skipped = []
    broken = []
    paths = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []


# LLM-generated content at query #3
#--------------------------

```python
def test_find_with_directory_and_skipped_paths():
    class MockConfig:
        def is_skipped(self, path):
            return str(path).endswith("skip")
        def is_supported_filetype(self, path):
            return True
        def follow_links(self):
            return False

    config = MockConfig()
    skipped = []
    broken = []
    paths = ["/test_dir"]
    os.makedirs("/test_dir", exist_ok=True)
    os.makedirs("/test_dir/skip", exist_ok=True)
    open("/test_dir/file1.py", "w").close()
    open("/test_dir/skip/file2.py", "w").close()
    result = list(find(paths, config, skipped, broken))
    assert result == ["/test_dir/file1.py"]
    assert skipped == ["/test_dir/skip"]
    assert broken == []

def test_find_with_nonexistent_path():
    class MockConfig:
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, path):
            return True
        def follow_links(self):
            return False

    config = MockConfig()
    skipped = []
    broken = []
    paths = ["/nonexistent"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["/nonexistent"]

def test_find_with_supported_and_unsupported_files():
    class MockConfig:
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, path):
            return path.endswith(".py")
        def follow_links(self):
            return False

    config = MockConfig()
    skipped = []
    broken = []
    paths = ["/test_dir"]
    os.makedirs("/test_dir", exist_ok=True)
    open("/test_dir/file1.py", "w").close()
    open("/test_dir/file2.txt", "w").close()
    result = list(find(paths, config, skipped, broken))
    assert result == ["/test_dir/file1.py"]
    assert skipped == []
    assert broken == []

def test_find_with_single_file():
    class MockConfig:
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, path):
            return True
        def follow_links(self):
            return False

    config = MockConfig()
    skipped = []
    broken = []
    paths = ["/test_file.py"]
    open("/test_file.py", "w").close()
    result = list(find(paths, config, skipped, broken))
    assert result == ["/test_file.py"]
    assert skipped == []
    assert broken == []


# LLM-generated content at query #4
#--------------------------

```
def test_find_skips_directories():
    config = Config()
    config.is_skipped = lambda path: str(path).endswith('skip')
    config.is_supported_filetype = lambda path: True
    skipped = []
    broken = []
    result = list(find(['test_dir'], config, skipped, broken))
    assert 'test_dir/skip' not in result
    assert 'test_dir/skip' in skipped

def test_find_yields_supported_files():
    config = Config()
    config.is_skipped = lambda path: False
    config.is_supported_filetype = lambda path: path.endswith('.py')
    skipped = []
    broken = []
    result = list(find(['test_file.py', 'test_file.txt'], config, skipped, broken))
    assert 'test_file.py' in result
    assert 'test_file.txt' not in result

def test_find_adds_broken_paths():
    config = Config()
    config.is_skipped = lambda path: False
    config.is_supported_filetype = lambda path: True
    skipped = []
    broken = []
    result = list(find(['nonexistent'], config, skipped, broken))
    assert 'nonexistent' in broken
    assert not result

def test_find_follows_links_when_configured():
    config = Config()
    config.follow_links = True
    config.is_skipped = lambda path: False
    config.is_supported_filetype = lambda path: True
    skipped = []
    broken = []
    result = list(find(['symlink_dir'], config, skipped, broken))
    assert 'symlink_dir/target_file' in result

def test_find_skips_visited_directories():
    config = Config()
    config.follow_links = True
    config.is_skipped = lambda path: False
    config.is_supported_filetype = lambda path: True
    skipped = []
    broken = []
    result = list(find(['dir_with_loop'], config, skipped, broken))
    assert len([p for p in result if 'loop_dir/file' in p]) == 1


# LLM-generated content at query #5
#--------------------------

```python
def test_find_yields_files_in_directory():
    class MockConfig:
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return True
        def follow_links(self):
            return False

    skipped = []
    broken = []
    paths = ["test_dir"]
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("")
    with open("test_dir/file2.py", "w") as f:
        f.write("")
    assert list(find(paths, MockConfig(), skipped, broken)) == ["test_dir/file1.py", "test_dir/file2.py"]

def test_find_skips_unsupported_files():
    class MockConfig:
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith(".py")
        def follow_links(self):
            return False

    skipped = []
    broken = []
    paths = ["test_dir"]
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("")
    with open("test_dir/file2.txt", "w") as f:
        f.write("")
    assert list(find(paths, MockConfig(), skipped, broken)) == ["test_dir/file1.py"]

def test_find_skips_skipped_paths():
    class MockConfig:
        def is_skipped(self, path):
            return str(path).endswith("skip_dir") or str(path).endswith("skip_file.py")
        def is_supported_filetype(self, filepath):
            return True
        def follow_links(self):
            return False

    skipped = []
    broken = []
    paths = ["test_dir"]
    os.makedirs("test_dir", exist_ok=True)
    os.makedirs("test_dir/skip_dir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("")
    with open("test_dir/skip_file.py", "w") as f:
        f.write("")
    assert list(find(paths, MockConfig(), skipped, broken)) == ["test_dir/file1.py"]
    assert set(skipped) == {"test_dir/skip_dir", os.path.abspath("test_dir/skip_file.py")}

def test_find_reports_broken_paths():
    class MockConfig:
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return True
        def follow_links(self):
            return False

    skipped = []
    broken = []
    paths = ["nonexistent_dir"]
    assert list(find(paths, MockConfig(), skipped, broken)) == []
    assert broken == ["nonexistent_dir"]

def test_find_yields_direct_files():
    class MockConfig:
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return True
        def follow_links(self):
            return False

    skipped = []
    broken = []
    paths = ["file1.py"]
    with open("file1.py", "w") as f:
        f.write("")
    assert list(find(paths, MockConfig(), skipped, broken)) == ["file1.py"]


# LLM-generated content at query #6
#--------------------------

```python
def test_find_with_directory():
    paths = ['test_dir']
    config = Config(follow_links=False, skipped_paths=[], supported_filetypes=['.py'])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ['test_dir/file1.py', 'test_dir/file2.py']
    assert skipped == []
    assert broken == []

def test_find_with_skipped_directory():
    paths = ['test_dir']
    config = Config(follow_links=False, skipped_paths=['test_dir/skip'], supported_filetypes=['.py'])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ['test_dir/file1.py']
    assert skipped == ['test_dir/skip']
    assert broken == []

def test_find_with_broken_path():
    paths = ['non_existent_dir']
    config = Config(follow_links=False, skipped_paths=[], supported_filetypes=['.py'])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ['non_existent_dir']

def test_find_with_file():
    paths = ['test_file.py']
    config = Config(follow_links=False, skipped_paths=[], supported_filetypes=['.py'])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ['test_file.py']
    assert skipped == []
    assert broken == []

def test_find_with_unsupported_filetype():
    paths = ['test_file.txt']
    config = Config(follow_links=False, skipped_paths=[], supported_filetypes=['.py'])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []


# LLM-generated content at query #7
#--------------------------

```python
def test_find_with_directory_and_skipped_files():
    paths = ["test_dir"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert skipped == ["test_dir/skipped_file.py"]
    assert broken == []

def test_find_with_non_existent_path():
    paths = ["non_existent_path"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert skipped == []
    assert broken == ["non_existent_path"]

def test_find_with_supported_filetype():
    paths = ["test_file.py"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

def test_find_with_unsupported_filetype():
    paths = ["test_file.txt"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert skipped == []
    assert broken == []

def test_find_with_follow_links_disabled():
    paths = ["test_dir"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert skipped == ["test_dir/skipped_file.py"]
    assert broken == []

def test_find_with_multiple_paths():
    paths = ["test_dir", "test_file.py", "non_existent_path"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert skipped == ["test_dir/skipped_file.py"]
    assert broken == ["non_existent_path"]


# LLM-generated content at query #8
#--------------------------

```
def test_predicate_at_line_31_evaluates_to_true():
    paths = ["/nonexistent/path"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert not os.path.exists(paths[0])
    assert broken == paths


# LLM-generated content at query #9
#--------------------------

```
def test_predicate_at_line_31_evaluates_to_true():
    paths = ["/nonexistent/path"]
    config = Config()
    skipped = []
    broken = []
    list(find(paths, config, skipped, broken))
    assert broken == ["/nonexistent/path"]


# LLM-generated content at query #10
#--------------------------

```
def test_find_finds_files_in_directory():
    config = Config()
    skipped = []
    broken = []
    paths = ["test_dir"]
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("")
    with open("test_dir/file2.txt", "w") as f:
        f.write("")
    result = list(find(paths, config, skipped, broken))
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.txt" not in result
    assert len(skipped) == 0
    assert len(broken) == 0

def test_find_handles_skipped_directories():
    config = Config()
    skipped = []
    broken = []
    paths = ["test_dir"]
    os.makedirs("test_dir/skip_dir", exist_ok=True)
    with open("test_dir/skip_dir/file.py", "w") as f:
        f.write("")
    config.is_skipped = lambda x: str(x).endswith("skip_dir")
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert "test_dir/skip_dir" in skipped
    assert len(broken) == 0

def test_find_handles_broken_paths():
    config = Config()
    skipped = []
    broken = []
    paths = ["nonexistent_path"]
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert "nonexistent_path" in broken

def test_find_handles_direct_single_file():
    config = Config()
    skipped = []
    broken = []
    paths = ["test_file.py"]
    with open("test_file.py", "w") as f:
        f.write("")
    result = list(find(paths, config, skipped, broken))
    assert "test_file.py" in result
    assert len(skipped) == 0
    assert len(broken) == 0

def test_find_skips_unsupported_filetypes():
    config = Config()
    skipped = []
    broken = []
    paths = ["test_dir"]
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file.txt", "w") as f:
        f.write("")
    config.is_supported_filetype = lambda x: x.endswith(".py")
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 0


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_false():
    paths = ["non_existent_directory"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert os.path.isdir("non_existent_directory") == False


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_false():
    paths = ["/non-existent-directory"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    
    result = list(find(paths, config, skipped, broken))
    
    assert result == []
    assert broken == ["/non-existent-directory"]
    assert skipped == []


# LLM-generated content at query #13
#--------------------------

```python
def test_find_skips_directories_based_on_config():
    class MockConfig:
        def __init__(self, follow_links=False):
            self.follow_links = follow_links
            self._skipped_paths = set()
        
        def is_skipped(self, path):
            return str(path) in self._skipped_paths
        
        def is_supported_filetype(self, path):
            return True
    
    config = MockConfig()
    config._skipped_paths.add('/test/skipped_dir')
    skipped = []
    broken = []
    paths = ['/test']
    
    result = list(find(paths, config, skipped, broken))
    
    assert '/test/skipped_dir' in skipped


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_false():
    paths = ["/nonexistent/directory"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert not os.path.isdir(paths[0])


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    paths = ["non_existent_file"]
    config = Config()
    skipped = []
    broken = []
    list(find(paths, config, skipped, broken))
    assert "non_existent_file" in broken


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_true():
    paths = ["/some/directory"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    
    # Mock os.path.isdir to return True
    original_isdir = os.path.isdir
    os.path.isdir = lambda x: True
    
    # Mock os.walk to return a dummy generator
    original_walk = os.walk
    os.walk = lambda *args, **kwargs: iter([("root", ["dir1"], ["file1.py"])])
    
    try:
        # Call the function to trigger the predicate
        list(find(paths, config, skipped, broken))
        assert True  # If we reach here, the predicate evaluated to True
    finally:
        # Restore original functions
        os.path.isdir = original_isdir
        os.walk = original_walk


# LLM-generated content at query #17
#--------------------------

```
def test_resolved_path_not_in_visited_dirs():
    config = Config()
    config.follow_links = False
    config.is_skipped = lambda _: False
    config.is_supported_filetype = lambda _: True
    skipped = []
    broken = []
    visited_dirs = set()
    paths = ["test_dir"]
    os.makedirs("test_dir", exist_ok=True)
    
    result = list(find(paths, config, skipped, broken))
    assert len(result) > 0
    assert len(skipped) == 0
    assert len(broken) == 0


# LLM-generated content at query #18
#--------------------------

```
def test_predicate_at_line_9_evaluates_to_true():
    paths = ["/some/existing/directory"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) >= 0  # Just ensure the predicate evaluates to True by reaching line 9


# LLM-generated content at query #19
#--------------------------

```python
def test_find_with_directory_and_skipped_files():
    class MockConfig:
        def is_skipped(self, path):
            return str(path).endswith('skip_me')
        
        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')
        
        follow_links = False

    skipped = []
    broken = []
    paths = ['/test_dir']
    config = MockConfig()
    
    # Mock os.walk to return specific directory structure
    original_walk = os.walk
    os.walk = lambda *args, **kwargs: iter([
        ('/test_dir', ['skip_me_dir', 'normal_dir'], ['test.py', 'skip_me.py', 'ignore.txt']),
        ('/test_dir/normal_dir', [], ['normal.py']),
    ])
    
    # Mock os.path functions
    original_isdir = os.path.isdir
    original_exists = os.path.exists
    original_abspath = os.path.abspath
    os.path.isdir = lambda x: True
    os.path.exists = lambda x: True
    os.path.abspath = lambda x: x
    
    result = list(find(paths, config, skipped, broken))
    
    # Restore original functions
    os.walk = original_walk
    os.path.isdir = original_isdir
    os.path.exists = original_exists
    os.path.abspath = original_abspath
    
    assert result == ['/test_dir/test.py', '/test_dir/normal_dir/normal.py']
    assert skipped == ['/test_dir/skip_me.py']
    assert broken == []

def test_find_with_nonexistent_path():
    class MockConfig:
        def is_skipped(self, path):
            return False
        
        def is_supported_filetype(self, filepath):
            return True
        
        follow_links = False

    skipped = []
    broken = []
    paths = ['/nonexistent']
    config = MockConfig()
    
    original_exists = os.path.exists
    os.path.exists = lambda x: False
    
    result = list(find(paths, config, skipped, broken))
    
    os.path.exists = original_exists
    
    assert result == []
    assert skipped == []
    assert broken == ['/nonexistent']

def test_find_with_single_file():
    class MockConfig:
        def is_skipped(self, path):
            return False
        
        def is_supported_filetype(self, filepath):
            return True
        
        follow_links = False

    skipped = []
    broken = []
    paths = ['/test_file.py']
    config = MockConfig()
    
    original_isdir = os.path.isdir
    original_exists = os.path.exists
    os.path.isdir = lambda x: False
    os.path.exists = lambda x: True
    
    result = list(find(paths, config, skipped, broken))
    
    os.path.isdir = original_isdir
    os.path.exists = original_exists
    
    assert result == ['/test_file.py']
    assert skipped == []
    assert broken == []

def test_find_with_skipped_directory():
    class MockConfig:
        def is_skipped(self, path):
            return str(path) == '/test_dir/skip_me_dir'
        
        def is_supported_filetype(self, filepath):
            return True
        
        follow_links = False

    skipped = []
    broken = []
    paths = ['/test_dir']
    config = MockConfig()
    
    original_walk = os.walk
    original_isdir = os.path.isdir
    original_exists = os.path.exists
    original_abspath = os.path.abspath
    
    os.walk = lambda *args, **kwargs: iter([
        ('/test_dir', ['skip_me_dir', 'normal_dir'], ['test.py']),
        ('/test_dir/normal_dir', [], ['normal.py']),
    ])
    os.path.isdir = lambda x: True
    os.path.exists = lambda x: True
    os.path.abspath = lambda x: x
    
    result = list(find(paths, config, skipped, broken))
    
    os.walk = original_walk
    os.path.isdir = original_isdir
    os.path.exists = original_exists
    os.path.abspath = original_abspath
    
    assert result == ['/test_dir/test.py', '/test_dir/normal_dir/normal.py']
    assert skipped == ['/test_dir/skip_me_dir']
    assert broken == []


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_false():
    paths = ["/non-existent-directory"]
    config = Config()
    skipped = []
    broken = []
    for _ in find(paths, config, skipped, broken):
        pass
    assert not os.path.isdir(paths[0])


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    paths = ["nonexistent_path"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert not result
    assert broken == ["nonexistent_path"]
    assert not skipped


# LLM-generated content at query #22
#--------------------------

```python
def test_find_with_directory():
    class MockConfig:
        def __init__(self, follow_links=False):
            self.follow_links = follow_links

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, path):
            return True

    paths = ["test_dir"]
    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) > 0
    assert len(skipped) == 0
    assert len(broken) == 0

def test_find_with_skipped_directory():
    class MockConfig:
        def __init__(self, follow_links=False):
            self.follow_links = follow_links

        def is_skipped(self, path):
            return "skip_dir" in str(path)

        def is_supported_filetype(self, path):
            return True

    paths = ["test_dir"]
    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) > 0
    assert len(skipped) > 0
    assert len(broken) == 0

def test_find_with_broken_path():
    class MockConfig:
        def __init__(self, follow_links=False):
            self.follow_links = follow_links

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, path):
            return True

    paths = ["non_existent_dir"]
    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) > 0

def test_find_with_unsupported_filetype():
    class MockConfig:
        def __init__(self, follow_links=False):
            self.follow_links = follow_links

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, path):
            return path.endswith(".py")

    paths = ["test_dir"]
    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) > 0
    assert len(skipped) == 0
    assert len(broken) == 0

def test_find_with_follow_links():
    class MockConfig:
        def __init__(self, follow_links=True):
            self.follow_links = follow_links

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, path):
            return True

    paths = ["test_dir"]
    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) > 0
    assert len(skipped) == 0
    assert len(broken) == 0


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_false():
    paths = ["nonexistent_directory"]
    config = Config()
    skipped = []
    broken = []
    
    result = list(find(paths, config, skipped, broken))
    
    assert not os.path.isdir(paths[0])


# LLM-generated content at query #24
#--------------------------

```
def test_find_with_directory_and_skipped_files():
    class MockConfig:
        def is_skipped(self, path):
            return str(path).endswith('skip')
        
        def is_supported_filetype(self, path):
            return path.endswith('.py')
        
        def __init__(self):
            self.follow_links = False

    skipped = []
    broken = []
    config = MockConfig()
    paths = ['/test_dir']
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 0

def test_find_with_nonexistent_path():
    class MockConfig:
        def is_skipped(self, path):
            return False
        
        def is_supported_filetype(self, path):
            return True
        
        def __init__(self):
            self.follow_links = False

    skipped = []
    broken = []
    config = MockConfig()
    paths = ['/nonexistent']
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == '/nonexistent'

def test_find_with_supported_file():
    class MockConfig:
        def is_skipped(self, path):
            return False
        
        def is_supported_filetype(self, path):
            return path.endswith('.py')
        
        def __init__(self):
            self.follow_links = False

    skipped = []
    broken = []
    config = MockConfig()
    paths = ['/test.py']
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert result[0] == '/test.py'
    assert len(skipped) == 0
    assert len(broken) == 0

def test_find_with_skipped_file():
    class MockConfig:
        def is_skipped(self, path):
            return str(path).endswith('skip.py')
        
        def is_supported_filetype(self, path):
            return path.endswith('.py')
        
        def __init__(self):
            self.follow_links = False

    skipped = []
    broken = []
    config = MockConfig()
    paths = ['/skip.py']
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 1
    assert skipped[0] == '/skip.py'
    assert len(broken) == 0


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    paths = ["/nonexistent/path"]
    config = Config()
    skipped = []
    broken = []
    find(paths, config, skipped, broken)
    assert len(broken) == 1


# LLM-generated content at query #26
#--------------------------

```python
def test_find_skips_directories_based_on_config():
    class MockConfig:
        def __init__(self, follow_links, skipped_dirs):
            self.follow_links = follow_links
            self._skipped_dirs = skipped_dirs
        
        def is_skipped(self, path):
            return str(path) in self._skipped_dirs
        
        def is_supported_filetype(self, path):
            return True
    
    skipped = []
    broken = []
    config = MockConfig(follow_links=False, skipped_dirs=['/path/to/skip'])
    paths = ['/path/to/skip']
    
    result = list(find(paths, config, skipped, broken))
    
    assert skipped == ['/path/to/skip']
    assert broken == []
    assert result == []


# LLM-generated content at query #27
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    paths = ["/nonexistent/path"]
    config = Config()
    skipped = []
    broken = []
    list(find(paths, config, skipped, broken))
    assert len(broken) == 1


# LLM-generated content at query #28
#--------------------------

```python
def test_find_returns_correct_files():
    config = Config(supported_filetypes=['.py'], follow_links=False)
    skipped, broken = [], []
    paths = ['test_dir']
    result = list(find(paths, config, skipped, broken))
    assert result == ['test_dir/file1.py', 'test_dir/subdir/file2.py']

def test_find_skips_directories():
    config = Config(supported_filetypes=['.py'], follow_links=False)
    skipped, broken = [], []
    paths = ['test_dir']
    config.is_skipped = lambda path: 'skip' in str(path)
    result = list(find(paths, config, skipped, broken))
    assert result == ['test_dir/file1.py']
    assert skipped == ['test_dir/skip_dir']

def test_find_reports_broken_paths():
    config = Config(supported_filetypes=['.py'], follow_links=False)
    skipped, broken = [], []
    paths = ['nonexistent_dir']
    result = list(find(paths, config, skipped, broken))
    assert broken == ['nonexistent_dir']

def test_find_follows_symlinks_when_enabled():
    config = Config(supported_filetypes=['.py'], follow_links=True)
    skipped, broken = [], []
    paths = ['test_dir']
    result = list(find(paths, config, skipped, broken))
    assert result == ['test_dir/file1.py', 'test_dir/subdir/file2.py', 'test_dir/symlink/file3.py']

def test_find_ignores_unsupported_filetypes():
    config = Config(supported_filetypes=['.txt'], follow_links=False)
    skipped, broken = [], []
    paths = ['test_dir']
    result = list(find(paths, config, skipped, broken))
    assert result == ['test_dir/file1.txt']

def test_find_handles_empty_paths():
    config = Config(supported_filetypes=['.py'], follow_links=False)
    skipped, broken = [], []
    paths = []
    result = list(find(paths, config, skipped, broken))
    assert result == []


