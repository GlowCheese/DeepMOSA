####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_cache_saves_and_loads_from_file():
    import os
    import tempfile
    import pickle

    def dummy_func():
        return 42

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name

    cache_decorator = cache(temp_path, verbose=False)
    wrapped_func = cache_decorator(dummy_func)
    wrapped_func()
    assert os.path.exists(temp_path)

    with open(temp_path, "rb") as f:
        loaded_value = pickle.load(f)
    assert loaded_value == 42

    os.remove(temp_path)

def test_cache_does_not_save_when_path_is_none():
    def dummy_func():
        return 42

    cache_decorator = cache(None, verbose=False)
    wrapped_func = cache_decorator(dummy_func)
    result = wrapped_func()
    assert result == 42

def test_cache_loads_from_existing_file():
    import os
    import tempfile
    import pickle

    def dummy_func():
        return 42

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        pickle.dump(42, temp_file)

    cache_decorator = cache(temp_path, verbose=False)
    wrapped_func = cache_decorator(dummy_func)
    result = wrapped_func()
    assert result == 42

    os.remove(temp_path)

def test_cache_logs_when_verbose_is_true():
    import os
    import tempfile

    def dummy_func():
        return 42

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name

    cache_decorator = cache(temp_path, verbose=True, name="test")
    wrapped_func = cache_decorator(dummy_func)
    wrapped_func()

    os.remove(temp_path)


# LLM-generated content at query #2
#--------------------------

```python
def test_copy_tree_creates_destination_directory():
    src = "/tmp/source_dir"
    dst = "/tmp/destination_dir"
    os.makedirs(src, exist_ok=True)
    copy_tree(src, dst)
    assert os.path.exists(dst)

def test_copy_tree_copies_files():
    src = "/tmp/source_dir"
    dst = "/tmp/destination_dir"
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("test content")
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "test_file.txt"))

def test_copy_tree_overwrites_files_when_overwrite_is_true():
    src = "/tmp/source_dir"
    dst = "/tmp/destination_dir"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("new content")
    with open(os.path.join(dst, "test_file.txt"), "w") as f:
        f.write("old content")
    copy_tree(src, dst, overwrite=True)
    with open(os.path.join(dst, "test_file.txt"), "r") as f:
        assert f.read() == "new content"

def test_copy_tree_does_not_overwrite_files_when_overwrite_is_false():
    src = "/tmp/source_dir"
    dst = "/tmp/destination_dir"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("new content")
    with open(os.path.join(dst, "test_file.txt"), "w") as f:
        f.write("old content")
    copy_tree(src, dst, overwrite=False)
    with open(os.path.join(dst, "test_file.txt"), "r") as f:
        assert f.read() == "old content"

def test_copy_tree_copies_subdirectories():
    src = "/tmp/source_dir"
    dst = "/tmp/destination_dir"
    os.makedirs(src, exist_ok=True)
    os.makedirs(os.path.join(src, "subdir"), exist_ok=True)
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "subdir"))


# LLM-generated content at query #3
#--------------------------

```
def test_readable_size_bytes():
    assert readable_size(500) == "500.00"

def test_readable_size_kilobytes():
    assert readable_size(1024) == "1.00K"

def test_readable_size_megabytes():
    assert readable_size(1024 * 1024) == "1.00M"

def test_readable_size_gigabytes():
    assert readable_size(1024 * 1024 * 1024) == "1.00G"

def test_readable_size_terabytes():
    assert readable_size(1024 * 1024 * 1024 * 1024) == "1.00T"

def test_readable_size_petabytes():
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.00P"

def test_readable_size_custom_digits():
    assert readable_size(1024, n_digits=0) == "1K"

def test_readable_size_fractional():
    assert readable_size(1500) == "1.46K"

def test_readable_size_large_number():
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1024.00P"


# LLM-generated content at query #4
#--------------------------

```python
def test_copy_tree_creates_destination_directory():
    import tempfile
    import shutil
    import os
    src_dir = tempfile.mkdtemp()
    dst_dir = os.path.join(tempfile.mkdtemp(), "new_dir")
    copy_tree(src_dir, dst_dir)
    assert os.path.exists(dst_dir)
    shutil.rmtree(src_dir)
    shutil.rmtree(os.path.dirname(dst_dir))

def test_copy_tree_copies_files():
    import tempfile
    import shutil
    import os
    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()
    with open(os.path.join(src_dir, "test_file.txt"), "w") as f:
        f.write("test")
    copy_tree(src_dir, dst_dir)
    assert os.path.exists(os.path.join(dst_dir, "test_file.txt"))
    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)

def test_copy_tree_overwrites_files_when_overwrite_is_true():
    import tempfile
    import shutil
    import os
    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()
    with open(os.path.join(src_dir, "test_file.txt"), "w") as f:
        f.write("test")
    with open(os.path.join(dst_dir, "test_file.txt"), "w") as f:
        f.write("old_content")
    copy_tree(src_dir, dst_dir, overwrite=True)
    with open(os.path.join(dst_dir, "test_file.txt"), "r") as f:
        assert f.read() == "test"
    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)

def test_copy_tree_does_not_overwrite_files_when_overwrite_is_false():
    import tempfile
    import shutil
    import os
    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()
    with open(os.path.join(src_dir, "test_file.txt"), "w") as f:
        f.write("test")
    with open(os.path.join(dst_dir, "test_file.txt"), "w") as f:
        f.write("old_content")
    copy_tree(src_dir, dst_dir, overwrite=False)
    with open(os.path.join(dst_dir, "test_file.txt"), "r") as f:
        assert f.read() == "old_content"
    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)

def test_copy_tree_copies_subdirectories():
    import tempfile
    import shutil
    import os
    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(src_dir, "subdir"))
    with open(os.path.join(src_dir, "subdir", "test_file.txt"), "w") as f:
        f.write("test")
    copy_tree(src_dir, dst_dir)
    assert os.path.exists(os.path.join(dst_dir, "subdir", "test_file.txt"))
    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)


# LLM-generated content at query #5
#--------------------------

```python
def test_cache_no_path():
    @cache(None)
    def test_func():
        return 42
    assert test_func() == 42

def test_cache_with_path(tmpdir):
    cache_path = tmpdir.join("cache.pkl")
    @cache(str(cache_path))
    def test_func():
        return 42
    assert test_func() == 42
    assert cache_path.exists()
    with open(str(cache_path), "rb") as f:
        assert pickle.load(f) == 42

def test_cache_load_from_existing_file(tmpdir):
    cache_path = tmpdir.join("cache.pkl")
    with open(str(cache_path), "wb") as f:
        pickle.dump(42, f)
    @cache(str(cache_path))
    def test_func():
        return 0
    assert test_func() == 42

def test_cache_verbose_false(tmpdir):
    cache_path = tmpdir.join("cache.pkl")
    @cache(str(cache_path), verbose=False)
    def test_func():
        return 42
    assert test_func() == 42

def test_cache_custom_name(tmpdir):
    cache_path = tmpdir.join("cache.pkl")
    @cache(str(cache_path), name="custom")
    def test_func():
        return 42
    assert test_func() == 42


# LLM-generated content at query #6
#--------------------------

```python
def test_scandir_with_pathlib_path():
    path = Path("test_directory")
    result = list(scandir(path))
    assert len(result) == 2
    assert isinstance(result[0], Path)
    assert isinstance(result[1], Path)

def test_scandir_with_str_path():
    path = "test_directory"
    result = list(scandir(path))
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)

def test_scandir_empty_directory_with_pathlib_path():
    path = Path("empty_directory")
    result = list(scandir(path))
    assert len(result) == 0

def test_scandir_empty_directory_with_str_path():
    path = "empty_directory"
    result = list(scandir(path))
    assert len(result) == 0


# LLM-generated content at query #7
#--------------------------

```python
def test_scandir_yields_string_path_when_input_is_string():
    import os
    from tempfile import TemporaryDirectory
    from pathlib import Path

    with TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.touch()

        result = list(scandir(tmpdir))
        assert all(isinstance(path, str) for path in result)

def test_scandir_yields_path_object_when_input_is_path():
    import os
    from tempfile import TemporaryDirectory
    from pathlib import Path

    with TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.touch()

        result = list(scandir(Path(tmpdir)))
        assert all(isinstance(path, Path) for path in result)


# LLM-generated content at query #8
#--------------------------

```python
def test_scandir_path_type_check():
    path = Path('/some/directory')
    result = scandir(path)
    assert isinstance(next(result), Path)


# LLM-generated content at query #9
#--------------------------

```python
def test_scandir_with_str_path():
    path = "/non/existent/directory"
    result = list(scandir(path))
    assert not result


# LLM-generated content at query #10
#--------------------------

```python
def test_copy_tree_overwrite():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src)
    os.makedirs(dst)
    src_file = os.path.join(src, "file.txt")
    dst_file = os.path.join(dst, "file.txt")
    with open(src_file, "w") as f:
        f.write("source content")
    with open(dst_file, "w") as f:
        f.write("destination content")
    copy_tree(src, dst, overwrite=True)
    with open(dst_file, "r") as f:
        content = f.read()
    assert content == "source content"


# LLM-generated content at query #11
#--------------------------

```python
def test_copy_tree_overwrite_true():
    src = "/path/to/source"
    dst = "/path/to/destination"
    overwrite = True
    copy_tree(src, dst, overwrite=overwrite)
    assert os.path.exists(dst)

def test_copy_tree_overwrite_false():
    src = "/path/to/source"
    dst = "/path/to/destination"
    overwrite = False
    copy_tree(src, dst, overwrite=overwrite)
    assert os.path.exists(dst)


# LLM-generated content at query #12
#--------------------------

```
def test_scandir_with_str_path():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        result = list(scandir(tmpdir))
        assert len(result) == 1
        assert isinstance(result[0], str)
        assert os.path.basename(result[0]) == 'test.txt'

def test_scandir_with_pathlib_path():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / 'test.txt'
        with open(test_file, 'w') as f:
            f.write('test')
        result = list(scandir(Path(tmpdir)))
        assert len(result) == 1
        assert isinstance(result[0], Path)
        assert result[0].name == 'test.txt'


# LLM-generated content at query #13
#--------------------------

```python
def test_cache_with_path_and_verbose_true():
    import tempfile
    import os
    from flutes.fs import cache
    from flutes.log import log

    def dummy_func():
        return 42

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        cached_func = cache(temp_path, verbose=True, name="test")(dummy_func)
        result = cached_func()
        assert result == 42
        assert os.path.exists(temp_path)
        with open(temp_path, "rb") as f:
            cached_result = pickle.load(f)
            assert cached_result == 42
        os.unlink(temp_path)

def test_cache_with_path_and_verbose_false():
    import tempfile
    import os
    from flutes.fs import cache
    from flutes.log import log

    def dummy_func():
        return 42

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        cached_func = cache(temp_path, verbose=False, name="test")(dummy_func)
        result = cached_func()
        assert result == 42
        assert os.path.exists(temp_path)
        with open(temp_path, "rb") as f:
            cached_result = pickle.load(f)
            assert cached_result == 42
        os.unlink(temp_path)

def test_cache_with_path_none():
    from flutes.fs import cache

    def dummy_func():
        return 42

    cached_func = cache(None, verbose=True, name="test")(dummy_func)
    result = cached_func()
    assert result == 42

def test_cache_with_existing_cache_file():
    import tempfile
    import os
    from flutes.fs import cache

    def dummy_func():
        return 42

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        with open(temp_path, "wb") as f:
            pickle.dump(42, f)
        cached_func = cache(temp_path, verbose=True, name="test")(dummy_func)
        result = cached_func()
        assert result == 42
        assert os.path.exists(temp_path)
        with open(temp_path, "rb") as f:
            cached_result = pickle.load(f)
            assert cached_result == 42
        os.unlink(temp_path)


# LLM-generated content at query #14
#--------------------------

```python
def test_copy_tree_overwrite_true():
    src = "/tmp/test_src"
    dst = "/tmp/test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "file.txt"), "w") as f:
        f.write("src content")
    with open(os.path.join(dst, "file.txt"), "w") as f:
        f.write("dst content")
    copy_tree(src, dst, overwrite=True)
    with open(os.path.join(dst, "file.txt"), "r") as f:
        content = f.read()
    assert content == "src content"


# LLM-generated content at query #15
#--------------------------

```
def test_scandir_with_path_object():
    from pathlib import Path
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        result = list(scandir(path))
        assert all(isinstance(p, Path) for p in result)


# LLM-generated content at query #16
#--------------------------

```
def test_scandir_with_str_path():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        result = list(scandir(tmpdir))
        assert len(result) == 1
        assert isinstance(result[0], str)
        assert os.path.basename(result[0]) == 'test.txt'


# LLM-generated content at query #17
#--------------------------

```python
def test_scandir_with_pathlib_path():
    from pathlib import Path
    test_dir = Path(__file__).parent / "test_dir"
    test_dir.mkdir(exist_ok=True)
    (test_dir / "file1.txt").touch()
    (test_dir / "file2.txt").touch()
    result = list(scandir(test_dir))
    assert len(result) == 2
    assert all(isinstance(path, Path) for path in result)
    assert any(path.name == "file1.txt" for path in result)
    assert any(path.name == "file2.txt" for path in result)
    (test_dir / "file1.txt").unlink()
    (test_dir / "file2.txt").unlink()
    test_dir.rmdir()

def test_scandir_with_str_path():
    import os
    test_dir = os.path.join(os.path.dirname(__file__), "test_dir")
    os.makedirs(test_dir, exist_ok=True)
    file1 = os.path.join(test_dir, "file1.txt")
    file2 = os.path.join(test_dir, "file2.txt")
    open(file1, 'w').close()
    open(file2, 'w').close()
    result = list(scandir(test_dir))
    assert len(result) == 2
    assert all(isinstance(path, str) for path in result)
    assert any(os.path.basename(path) == "file1.txt" for path in result)
    assert any(os.path.basename(path) == "file2.txt" for path in result)
    os.remove(file1)
    os.remove(file2)
    os.rmdir(test_dir)


# LLM-generated content at query #18
#--------------------------

```
def test_copy_tree_overwrite_true_evaluates_to_true():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "test_file"), "w") as f:
        f.write("test")
    with open(os.path.join(dst, "test_file"), "w") as f:
        f.write("old")
    overwrite = True
    result = overwrite or not os.path.exists(os.path.join(dst, "test_file"))
    assert result == True

def test_copy_tree_file_not_exists_evaluates_to_true():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "test_file"), "w") as f:
        f.write("test")
    overwrite = False
    result = overwrite or not os.path.exists(os.path.join(dst, "test_file"))
    assert result == True


# LLM-generated content at query #19
#--------------------------

```python
def test_cache_with_file_and_verbose():
    import tempfile
    import os
    import pickle
    import functools
    from flutes.log import log

    def dummy_function():
        return 42

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        path = temp_file.name
        pickle.dump(123, temp_file)

    decorator = cache(path, verbose=True, name="test_cache")
    wrapped = decorator(dummy_function)
    result = wrapped()
    assert result == 123
    os.remove(path)

def test_cache_with_file_and_no_verbose():
    import tempfile
    import os
    import pickle
    import functools
    from flutes.log import log

    def dummy_function():
        return 42

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        path = temp_file.name
        pickle.dump(123, temp_file)

    decorator = cache(path, verbose=False, name="test_cache")
    wrapped = decorator(dummy_function)
    result = wrapped()
    assert result == 123
    os.remove(path)

def test_cache_with_no_file():
    import tempfile
    import os
    import pickle
    import functools
    from flutes.log import log

    def dummy_function():
        return 42

    path = tempfile.mktemp()
    decorator = cache(path, verbose=True, name="test_cache")
    wrapped = decorator(dummy_function)
    result = wrapped()
    assert result == 42
    assert os.path.exists(path)
    os.remove(path)

def test_cache_with_no_path():
    import functools
    from flutes.log import log

    def dummy_function():
        return 42

    decorator = cache(None, verbose=True, name="test_cache")
    wrapped = decorator(dummy_function)
    result = wrapped()
    assert result == 42


# LLM-generated content at query #20
#--------------------------

```python
def test_copy_tree_overwrite_false_destination_exists():
    src = "/path/to/source"
    dst = "/path/to/destination"
    overwrite = False
    assert not (overwrite or not os.path.exists(dst))


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_readable_size_bytes():
    assert readable_size(0) == "0.00"
    assert readable_size(500) == "500.00"
    assert readable_size(1023) == "1023.00"

def test_readable_size_kilobytes():
    assert readable_size(1024) == "1.00K"
    assert readable_size(2048) == "2.00K"
    assert readable_size(1536) == "1.50K"

def test_readable_size_megabytes():
    assert readable_size(1024 * 1024) == "1.00M"
    assert readable_size(2.5 * 1024 * 1024) == "2.50M"

def test_readable_size_gigabytes():
    assert readable_size(1024 * 1024 * 1024) == "1.00G"
    assert readable_size(3.75 * 1024 * 1024 * 1024) == "3.75G"

def test_readable_size_terabytes():
    assert readable_size(1024 * 1024 * 1024 * 1024) == "1.00T"
    assert readable_size(5.25 * 1024 * 1024 * 1024 * 1024) == "5.25T"

def test_readable_size_petabytes():
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.00P"
    assert readable_size(7.125 * 1024 * 1024 * 1024 * 1024 * 1024) == "7.12P"

def test_readable_size_custom_digits():
    assert readable_size(1024, n_digits=0) == "1K"
    assert readable_size(1536, n_digits=1) == "1.5K"
    assert readable_size(2048, n_digits=3) == "2.000K"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_cache_with_path_and_verbose():
    import os
    import tempfile
    from flutes.fs import cache

    def dummy_func():
        return 42

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cache.pkl")
        decorated_func = cache(path, verbose=True)(dummy_func)
        assert decorated_func() == 42
        assert os.path.exists(path)
        assert decorated_func() == 42

def test_cache_with_path_and_no_verbose():
    import os
    import tempfile
    from flutes.fs import cache

    def dummy_func():
        return 42

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cache.pkl")
        decorated_func = cache(path, verbose=False)(dummy_func)
        assert decorated_func() == 42
        assert os.path.exists(path)
        assert decorated_func() == 42

def test_cache_with_no_path():
    import os
    import tempfile
    from flutes.fs import cache

    def dummy_func():
        return 42

    with tempfile.TemporaryDirectory() as tmpdir:
        decorated_func = cache(None, verbose=True)(dummy_func)
        assert decorated_func() == 42
        assert not os.path.exists(os.path.join(tmpdir, "cache.pkl"))

def test_cache_with_name():
    import os
    import tempfile
    from flutes.fs import cache

    def dummy_func():
        return 42

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cache.pkl")
        decorated_func = cache(path, verbose=True, name="test")(dummy_func)
        assert decorated_func() == 42
        assert os.path.exists(path)
        assert decorated_func() == 42


# LLM-generated content at query #2
#--------------------------

```python
def test_readable_size_bytes():
    assert readable_size(0) == "0.00"
    assert readable_size(1) == "1.00"
    assert readable_size(1023) == "1023.00"
    assert readable_size(1023, 0) == "1023"

def test_readable_size_kilobytes():
    assert readable_size(1024) == "1.00K"
    assert readable_size(1024 * 1.5) == "1.50K"
    assert readable_size(1024 * 1024 - 1) == "1024.00K"
    assert readable_size(1024 * 1024 - 1, 0) == "1024K"

def test_readable_size_megabytes():
    assert readable_size(1024 * 1024) == "1.00M"
    assert readable_size(1024 * 1024 * 1.5) == "1.50M"
    assert readable_size(1024 * 1024 * 1024 - 1) == "1024.00M"
    assert readable_size(1024 * 1024 * 1024 - 1, 0) == "1024M"

def test_readable_size_gigabytes():
    assert readable_size(1024 * 1024 * 1024) == "1.00G"
    assert readable_size(1024 * 1024 * 1024 * 1.5) == "1.50G"
    assert readable_size(1024 * 1024 * 1024 * 1024 - 1) == "1024.00G"
    assert readable_size(1024 * 1024 * 1024 * 1024 - 1, 0) == "1024G"

def test_readable_size_terabytes():
    assert readable_size(1024 * 1024 * 1024 * 1024) == "1.00T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1.5) == "1.50T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 - 1) == "1024.00T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 - 1, 0) == "1024T"

def test_readable_size_petabytes():
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1.5) == "1.50P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1024.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024, 0) == "1024P"


# LLM-generated content at query #3
#--------------------------

```python
def test_scandir_with_pathlib_path():
    import tempfile
    import os
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir).joinpath('file1.txt').touch()
        Path(tmpdir).joinpath('file2.txt').touch()
        result = list(scandir(Path(tmpdir)))
        assert len(result) == 2
        assert all(isinstance(p, Path) for p in result)
        assert any(p.name == 'file1.txt' for p in result)
        assert any(p.name == 'file2.txt' for p in result)

def test_scandir_with_str_path():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, 'file1.txt'), 'w').close()
        open(os.path.join(tmpdir, 'file2.txt'), 'w').close()
        result = list(scandir(tmpdir))
        assert len(result) == 2
        assert all(isinstance(p, str) for p in result)
        assert any(os.path.basename(p) == 'file1.txt' for p in result)
        assert any(os.path.basename(p) == 'file2.txt' for p in result)


# LLM-generated content at query #4
#--------------------------

```python
def test_cache_file_exists():
    import os
    import tempfile
    import pickle

    @cache(tempfile.NamedTemporaryFile(delete=False).name)
    def dummy_func():
        return 42

    assert dummy_func() == 42
    assert dummy_func() == 42


# LLM-generated content at query #5
#--------------------------

```python
def test_copy_tree_new_directory():
    import os
    import shutil
    import tempfile

    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(src_dir, "subdir"))
    open(os.path.join(src_dir, "file1.txt"), "w").close()
    open(os.path.join(src_dir, "subdir", "file2.txt"), "w").close()

    copy_tree(src_dir, dst_dir)

    assert os.path.exists(os.path.join(dst_dir, "file1.txt"))
    assert os.path.exists(os.path.join(dst_dir, "subdir", "file2.txt"))
    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)


def test_copy_tree_overwrite_existing_files():
    import os
    import shutil
    import tempfile

    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(src_dir, "subdir"))
    open(os.path.join(src_dir, "file1.txt"), "w").write("src content")
    open(os.path.join(src_dir, "subdir", "file2.txt"), "w").write("src content")
    open(os.path.join(dst_dir, "file1.txt"), "w").write("dst content")
    os.makedirs(os.path.join(dst_dir, "subdir"))
    open(os.path.join(dst_dir, "subdir", "file2.txt"), "w").write("dst content")

    copy_tree(src_dir, dst_dir, overwrite=True)

    assert open(os.path.join(dst_dir, "file1.txt")).read() == "src content"
    assert open(os.path.join(dst_dir, "subdir", "file2.txt")).read() == "src content"
    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)


def test_copy_tree_skip_existing_files():
    import os
    import shutil
    import tempfile

    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(src_dir, "subdir"))
    open(os.path.join(src_dir, "file1.txt"), "w").write("src content")
    open(os.path.join(src_dir, "subdir", "file2.txt"), "w").write("src content")
    open(os.path.join(dst_dir, "file1.txt"), "w").write("dst content")
    os.makedirs(os.path.join(dst_dir, "subdir"))
    open(os.path.join(dst_dir, "subdir", "file2.txt"), "w").write("dst content")

    copy_tree(src_dir, dst_dir, overwrite=False)

    assert open(os.path.join(dst_dir, "file1.txt")).read() == "dst content"
    assert open(os.path.join(dst_dir, "subdir", "file2.txt")).read() == "dst content"
    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)


def test_copy_tree_empty_directory():
    import os
    import shutil
    import tempfile

    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()

    copy_tree(src_dir, dst_dir)

    assert os.path.exists(dst_dir)
    assert len(os.listdir(dst_dir)) == 0
    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)


