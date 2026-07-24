####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_cache_decorator_with_existing_file():
    test_path = "test_cache.pkl"
    test_data = {"key": "value"}
    with open(test_path, "wb") as f:
        pickle.dump(test_data, f)
    
    @cache(test_path)
    def test_func():
        return {"key": "different_value"}
    
    result = test_func()
    assert result == test_data
    os.remove(test_path)

def test_cache_decorator_with_nonexistent_file():
    test_path = "test_cache.pkl"
    if os.path.exists(test_path):
        os.remove(test_path)
    
    expected_data = {"key": "value"}
    
    @cache(test_path)
    def test_func():
        return expected_data
    
    result = test_func()
    assert result == expected_data
    assert os.path.exists(test_path)
    os.remove(test_path)

def test_cache_decorator_with_none_path():
    @cache(None)
    def test_func():
        return {"key": "value"}
    
    result = test_func()
    assert result == {"key": "value"}

def test_cache_decorator_verbose_false():
    test_path = "test_cache.pkl"
    if os.path.exists(test_path):
        os.remove(test_path)
    
    @cache(test_path, verbose=False)
    def test_func():
        return {"key": "value"}
    
    test_func()
    assert os.path.exists(test_path)
    os.remove(test_path)

def test_cache_decorator_with_custom_name():
    test_path = "test_cache.pkl"
    if os.path.exists(test_path):
        os.remove(test_path)
    
    @cache(test_path, name="custom")
    def test_func():
        return {"key": "value"}
    
    test_func()
    assert os.path.exists(test_path)
    os.remove(test_path)


# LLM-generated content at query #2
#--------------------------

```python
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
    assert readable_size(1024 * 1024, n_digits=0) == "1M"

def test_readable_size_fractional():
    assert readable_size(1500) == "1.46K"

def test_readable_size_large_number():
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1024.00P"

def test_readable_size_zero():
    assert readable_size(0) == "0.00"


# LLM-generated content at query #3
#--------------------------

```python
def test_cache_with_path_and_verbose():
    import tempfile
    import os
    import pickle
    test_data = {"key": "value"}
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(test_data, f)
        
        @cache(tmp_path, verbose=True, name="test")
        def test_func():
            return {"new": "data"}
        
        result = test_func()
        assert result == test_data
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def test_cache_with_path_and_no_verbose():
    import tempfile
    import os
    import pickle
    test_data = {"key": "value"}
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(test_data, f)
        
        @cache(tmp_path, verbose=False)
        def test_func():
            return {"new": "data"}
        
        result = test_func()
        assert result == test_data
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def test_cache_with_none_path():
    @cache(None)
    def test_func():
        return {"key": "value"}
    
    result = test_func()
    assert result == {"key": "value"}

def test_cache_with_nonexistent_path():
    import tempfile
    import os
    test_data = {"key": "value"}
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        tmp_path = tmp_file.name
    
    @cache(tmp_path, verbose=True, name="test")
    def test_func():
        return test_data
    
    result = test_func()
    assert result == test_data
    assert os.path.exists(tmp_path)
    try:
        with open(tmp_path, "rb") as f:
            loaded_data = pickle.load(f)
        assert loaded_data == test_data
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# LLM-generated content at query #4
#--------------------------

```python
def test_cache_with_path_and_verbose():
    import tempfile
    import os
    import pickle
    test_data = {"key": "value"}
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        path = tmp_file.name
    try:
        with open(path, "wb") as f:
            pickle.dump(test_data, f)
        
        @cache(path, verbose=True, name="test")
        def func():
            return {"should": "not be called"}
        
        result = func()
        assert result == test_data
    finally:
        if os.path.exists(path):
            os.unlink(path)

def test_cache_with_path_and_no_verbose():
    import tempfile
    import os
    import pickle
    test_data = {"key": "value"}
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        path = tmp_file.name
    try:
        with open(path, "wb") as f:
            pickle.dump(test_data, f)
        
        @cache(path, verbose=False, name="test")
        def func():
            return {"should": "not be called"}
        
        result = func()
        assert result == test_data
    finally:
        if os.path.exists(path):
            os.unlink(path)

def test_cache_without_path():
    @cache(None, verbose=True, name="test")
    def func():
        return {"key": "value"}
    
    result = func()
    assert result == {"key": "value"}

def test_cache_new_file_creation():
    import tempfile
    import os
    import pickle
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        path = tmp_file.name
    os.unlink(path)
    
    try:
        @cache(path, verbose=True, name="test")
        def func():
            return {"key": "value"}
        
        result = func()
        assert result == {"key": "value"}
        assert os.path.exists(path)
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded == {"key": "value"}
    finally:
        if os.path.exists(path):
            os.unlink(path)

def test_cache_function_execution_when_no_file():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile() as tmp_file:
        path = tmp_file.name
    assert not os.path.exists(path)
    
    @cache(path, verbose=True, name="test")
    def func():
        return {"key": "value"}
    
    result = func()
    assert result == {"key": "value"}
    assert os.path.exists(path)
    if os.path.exists(path):
        os.unlink(path)


# LLM-generated content at query #5
#--------------------------

```python
import os
import pickle

def test_cache_loads_from_existing_file():
    path = "test_cache.pkl"
    expected_result = "cached_data"
    with open(path, "wb") as f:
        pickle.dump(expected_result, f)
    
    @cache(path)
    def dummy_func():
        return "new_data"
    
    result = dummy_func()
    os.remove(path)
    assert result == expected_result

def test_cache_saves_to_file_when_not_exists():
    path = "test_cache.pkl"
    if os.path.exists(path):
        os.remove(path)
    
    @cache(path)
    def dummy_func():
        return "new_data"
    
    result = dummy_func()
    assert os.path.exists(path)
    with open(path, "rb") as f:
        loaded_result = pickle.load(f)
    os.remove(path)
    assert result == loaded_result

def test_cache_does_not_save_when_path_is_none():
    path = None
    
    @cache(path)
    def dummy_func():
        return "new_data"
    
    result = dummy_func()
    assert result == "new_data"

def test_cache_logs_loading_when_verbose():
    path = "test_cache.pkl"
    expected_result = "cached_data"
    with open(path, "wb") as f:
        pickle.dump(expected_result, f)
    
    @cache(path, verbose=True)
    def dummy_func():
        return "new_data"
    
    dummy_func()
    os.remove(path)

def test_cache_logs_saving_when_verbose():
    path = "test_cache.pkl"
    if os.path.exists(path):
        os.remove(path)
    
    @cache(path, verbose=True)
    def dummy_func():
        return "new_data"
    
    dummy_func()
    os.remove(path)

def test_cache_uses_custom_name_in_log():
    path = "test_cache.pkl"
    name = "CustomCache"
    expected_result = "cached_data"
    with open(path, "wb") as f:
        pickle.dump(expected_result, f)
    
    @cache(path, verbose=True, name=name)
    def dummy_func():
        return "new_data"
    
    dummy_func()
    os.remove(path)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import os
import pickle
import tempfile
import functools

@cache(None)
def test_func():
    return "test_value"

@cache("invalid_path")
def test_func_invalid_path():
    return "test_value"

def test_cache_no_path():
    result = test_func()
    assert result == "test_value"

def test_cache_invalid_path():
    result = test_func_invalid_path()
    assert result == "test_value"

def test_cache_valid_path():
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        path = tmp_file.name
        pickle.dump("cached_value", open(path, "wb"))

    @cache(path, verbose=False)
    def test_func_cached():
        return "new_value"

    result = test_func_cached()
    assert result == "cached_value"
    os.remove(path)

def test_cache_save_new():
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        path = tmp_file.name

    @cache(path, verbose=False)
    def test_func_save():
        return "new_value"

    result = test_func_save()
    assert result == "new_value"
    assert pickle.load(open(path, "rb")) == "new_value"
    os.remove(path)


# LLM-generated content at query #2
#--------------------------

```python
def test_readable_size_bytes():
    assert readable_size(0) == "0.00"
    assert readable_size(1) == "1.00"
    assert readable_size(1023) == "1023.00"

def test_readable_size_kilobytes():
    assert readable_size(1024) == "1.00K"
    assert readable_size(1536) == "1.50K"
    assert readable_size(1024 * 1023) == "1023.00K"

def test_readable_size_megabytes():
    assert readable_size(1024 * 1024) == "1.00M"
    assert readable_size(1024 * 1024 * 1.5) == "1.50M"
    assert readable_size(1024 * 1024 * 1023) == "1023.00M"

def test_readable_size_gigabytes():
    assert readable_size(1024 * 1024 * 1024) == "1.00G"
    assert readable_size(1024 * 1024 * 1024 * 2.5) == "2.50G"
    assert readable_size(1024 * 1024 * 1024 * 1023) == "1023.00G"

def test_readable_size_terabytes():
    assert readable_size(1024 * 1024 * 1024 * 1024) == "1.00T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 3.75) == "3.75T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1023) == "1023.00T"

def test_readable_size_petabytes():
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 4.125) == "4.12P"

def test_readable_size_custom_digits():
    assert readable_size(1536, 0) == "2K"
    assert readable_size(1536, 1) == "1.5K"
    assert readable_size(1536, 3) == "1.500K"
    assert readable_size(1024 * 1024 * 1.2345, 3) == "1.234M"


# LLM-generated content at query #3
#--------------------------

```python
def test_copy_tree_new_directory():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, "file1.txt"), "w") as f:
        f.write("Hello")
    copy_tree(src, dst)
    assert os.path.exists(dst)
    assert os.path.exists(os.path.join(dst, "file1.txt"))
    shutil.rmtree(src)
    shutil.rmtree(dst)

def test_copy_tree_existing_directory_no_overwrite():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "file1.txt"), "w") as f:
        f.write("Hello")
    with open(os.path.join(dst, "file1.txt"), "w") as f:
        f.write("World")
    copy_tree(src, dst)
    with open(os.path.join(dst, "file1.txt"), "r") as f:
        assert f.read() == "World"
    shutil.rmtree(src)
    shutil.rmtree(dst)

def test_copy_tree_existing_directory_with_overwrite():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "file1.txt"), "w") as f:
        f.write("Hello")
    with open(os.path.join(dst, "file1.txt"), "w") as f:
        f.write("World")
    copy_tree(src, dst, overwrite=True)
    with open(os.path.join(dst, "file1.txt"), "r") as f:
        assert f.read() == "Hello"
    shutil.rmtree(src)
    shutil.rmtree(dst)

def test_copy_tree_subdirectories():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(os.path.join(src, "subdir"), exist_ok=True)
    with open(os.path.join(src, "subdir", "file1.txt"), "w") as f:
        f.write("Hello")
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "subdir"))
    assert os.path.exists(os.path.join(dst, "subdir", "file1.txt"))
    shutil.rmtree(src)
    shutil.rmtree(dst)


# LLM-generated content at query #4
#--------------------------

```python
def test_cache_with_existing_file():
    import tempfile
    import os
    import pickle

    with tempfile.NamedTemporaryFile(delete=False) as f:
        test_data = {"key": "value"}
        pickle.dump(test_data, f)
        path = f.name

    @cache(path, verbose=False)
    def dummy_func():
        return {"key": "new_value"}

    result = dummy_func()
    os.remove(path)
    assert result == test_data


def test_cache_with_non_existing_file():
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
        os.remove(path)

    @cache(path, verbose=False)
    def dummy_func():
        return {"key": "new_value"}

    result = dummy_func()
    assert result == {"key": "new_value"}
    os.remove(path)


def test_cache_with_no_path():
    @cache(None, verbose=False)
    def dummy_func():
        return {"key": "new_value"}

    result = dummy_func()
    assert result == {"key": "new_value"}


def test_cache_verbose_logging():
    import tempfile
    import os
    import pickle
    from io import StringIO
    import sys

    with tempfile.NamedTemporaryFile(delete=False) as f:
        test_data = {"key": "value"}
        pickle.dump(test_data, f)
        path = f.name

    captured_output = StringIO()
    sys.stdout = captured_output

    @cache(path, verbose=True, name="test")
    def dummy_func():
        return {"key": "new_value"}

    dummy_func()
    os.remove(path)
    sys.stdout = sys.__stdout__
    assert "loaded from" in captured_output.getvalue()


# LLM-generated content at query #5
#--------------------------

```
def test_copy_tree_predicate_evaluates_false():
    src = "test_src"
    dst = "test_dst"
    overwrite = False
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "existing_file.txt"), "w") as f:
        f.write("content")
    assert not (overwrite or not os.path.exists(os.path.join(dst, "existing_file.txt")))


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_cache_with_path_and_verbose():
    import os
    import pickle
    import tempfile
    test_file = tempfile.NamedTemporaryFile(delete=False).name
    try:
        @cache(test_file, verbose=True, name="test")
        def test_func():
            return {"key": "value"}
        result = test_func()
        assert result == {"key": "value"}
        assert os.path.exists(test_file)
        with open(test_file, "rb") as f:
            assert pickle.load(f) == {"key": "value"}
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)

def test_cache_with_path_no_verbose():
    import os
    import pickle
    import tempfile
    test_file = tempfile.NamedTemporaryFile(delete=False).name
    try:
        @cache(test_file, verbose=False)
        def test_func():
            return {"key": "value"}
        result = test_func()
        assert result == {"key": "value"}
        assert os.path.exists(test_file)
        with open(test_file, "rb") as f:
            assert pickle.load(f) == {"key": "value"}
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)

def test_cache_with_none_path():
    @cache(None)
    def test_func():
        return {"key": "value"}
    result = test_func()
    assert result == {"key": "value"}

def test_cache_load_existing_file():
    import os
    import pickle
    import tempfile
    test_file = tempfile.NamedTemporaryFile(delete=False).name
    try:
        with open(test_file, "wb") as f:
            pickle.dump({"key": "cached_value"}, f)
        @cache(test_file, verbose=True, name="test")
        def test_func():
            return {"key": "new_value"}
        result = test_func()
        assert result == {"key": "cached_value"}
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


# LLM-generated content at query #2
#--------------------------

```python
def test_copy_tree_creates_destination_directory():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    copy_tree(src, dst)
    assert os.path.exists(dst)

def test_copy_tree_copies_files():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("test content")
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "test_file.txt"))

def test_copy_tree_overwrites_files():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("new content")
    with open(os.path.join(dst, "test_file.txt"), "w") as f:
        f.write("old content")
    copy_tree(src, dst, overwrite=True)
    with open(os.path.join(dst, "test_file.txt"), "r") as f:
        assert f.read() == "new content"

def test_copy_tree_does_not_overwrite_files():
    src = "test_src"
    dst = "test_dst"
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
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(os.path.join(src, "subdir"), exist_ok=True)
    with open(os.path.join(src, "subdir", "test_file.txt"), "w") as f:
        f.write("test content")
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "subdir", "test_file.txt"))


# LLM-generated content at query #3
#--------------------------

```python
def test_scandir_with_pathlib_path():
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        (tmp_path / 'file1.txt').touch()
        (tmp_path / 'file2.txt').touch()
        result = list(scandir(tmp_path))
        assert len(result) == 2
        assert all(isinstance(p, pathlib.Path) for p in result)
        assert any('file1.txt' in str(p) for p in result)
        assert any('file2.txt' in str(p) for p in result)

def test_scandir_with_str_path():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        (tmpdir + '/file1.txt').touch()
        (tmpdir + '/file2.txt').touch()
        result = list(scandir(tmpdir))
        assert len(result) == 2
        assert all(isinstance(p, str) for p in result)
        assert any('file1.txt' in p for p in result)
        assert any('file2.txt' in p for p in result)

def test_scandir_empty_directory():
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        result = list(scandir(tmp_path))
        assert len(result) == 0


# LLM-generated content at query #4
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


# LLM-generated content at query #5
#--------------------------

```python
def test_cache_with_path_and_verbose():
    path = "test_cache.pkl"
    data = {"key": "value"}
    with open(path, "wb") as f:
        pickle.dump(data, f)
    @cache(path, verbose=True, name="test")
    def dummy_func():
        return {"key": "value"}
    result = dummy_func()
    assert result == data
    os.remove(path)

def test_cache_with_path_and_no_verbose():
    path = "test_cache.pkl"
    data = {"key": "value"}
    with open(path, "wb") as f:
        pickle.dump(data, f)
    @cache(path, verbose=False, name="test")
    def dummy_func():
        return {"key": "value"}
    result = dummy_func()
    assert result == data
    os.remove(path)

def test_cache_without_path():
    @cache(None, verbose=True, name="test")
    def dummy_func():
        return {"key": "value"}
    result = dummy_func()
    assert result == {"key": "value"}

def test_cache_with_non_existing_path():
    path = "non_existing_cache.pkl"
    @cache(path, verbose=True, name="test")
    def dummy_func():
        return {"key": "value"}
    result = dummy_func()
    assert result == {"key": "value"}
    assert os.path.exists(path)
    os.remove(path)

def test_cache_with_non_existing_path_and_no_verbose():
    path = "non_existing_cache.pkl"
    @cache(path, verbose=False, name="test")
    def dummy_func():
        return {"key": "value"}
    result = dummy_func()
    assert result == {"key": "value"}
    assert os.path.exists(path)
    os.remove(path)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_cache_saves_and_loads_from_file(tmp_path):
    cache_file = tmp_path / "test_cache.pkl"
    data = {"key": "value"}

    @cache(cache_file, verbose=False)
    def test_func():
        return data

    result = test_func()
    assert result == data
    assert cache_file.exists()

    loaded_result = test_func()
    assert loaded_result == data

def test_cache_does_not_save_when_path_is_none():
    @cache(None, verbose=False)
    def test_func():
        return {"key": "value"}

    result = test_func()
    assert result == {"key": "value"}

def test_cache_logs_when_verbose_is_true(tmp_path, capsys):
    cache_file = tmp_path / "test_cache.pkl"
    data = {"key": "value"}

    @cache(cache_file, verbose=True)
    def test_func():
        return data

    test_func()
    captured = capsys.readouterr()
    assert f"Cache saved to '{cache_file}'" in captured.out

    test_func()
    captured = capsys.readouterr()
    assert f"Cache loaded from '{cache_file}'" in captured.out

def test_cache_uses_custom_name_in_logs(tmp_path, capsys):
    cache_file = tmp_path / "test_cache.pkl"
    data = {"key": "value"}

    @cache(cache_file, verbose=True, name="custom")
    def test_func():
        return data

    test_func()
    captured = capsys.readouterr()
    assert f"Custom saved to '{cache_file}'" in captured.out

    test_func()
    captured = capsys.readouterr()
    assert f"Custom loaded from '{cache_file}'" in captured.out


# LLM-generated content at query #2
#--------------------------

```python
def test_readable_size_bytes():
    assert readable_size(0) == "0.00"
    assert readable_size(1) == "1.00"
    assert readable_size(1023) == "1023.00"

def test_readable_size_kilobytes():
    assert readable_size(1024) == "1.00K"
    assert readable_size(1536) == "1.50K"
    assert readable_size(1024 * 1024 - 1) == "1024.00K"

def test_readable_size_megabytes():
    assert readable_size(1024 * 1024) == "1.00M"
    assert readable_size(1024 * 1024 * 1.5) == "1.50M"
    assert readable_size(1024 * 1024 * 1024 - 1) == "1024.00M"

def test_readable_size_gigabytes():
    assert readable_size(1024 * 1024 * 1024) == "1.00G"
    assert readable_size(1024 * 1024 * 1024 * 2.5) == "2.50G"
    assert readable_size(1024 * 1024 * 1024 * 1024 - 1) == "1024.00G"

def test_readable_size_terabytes():
    assert readable_size(1024 * 1024 * 1024 * 1024) == "1.00T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 3.75) == "3.75T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 - 1) == "1024.00T"

def test_readable_size_petabytes():
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 10.125) == "10.12P"

def test_readable_size_custom_digits():
    assert readable_size(1536, 0) == "2K"
    assert readable_size(1536, 1) == "1.5K"
    assert readable_size(1536, 3) == "1.500K"
    assert readable_size(1024 * 1024 * 1.23456, 4) == "1.2346M"


# LLM-generated content at query #3
#--------------------------

def test_cache_with_path_and_verbose():
    import tempfile
    import os
    test_file = tempfile.NamedTemporaryFile(delete=False).name
    try:
        @cache(test_file, verbose=True, name="test")
        def test_func():
            return {"key": "value"}
        result = test_func()
        assert result == {"key": "value"}
        assert os.path.exists(test_file)
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)

def test_cache_with_path_and_no_verbose():
    import tempfile
    import os
    test_file = tempfile.NamedTemporaryFile(delete=False).name
    try:
        @cache(test_file, verbose=False, name="test")
        def test_func():
            return {"key": "value"}
        result = test_func()
        assert result == {"key": "value"}
        assert os.path.exists(test_file)
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)

def test_cache_with_no_path():
    @cache(None, verbose=True, name="test")
    def test_func():
        return {"key": "value"}
    result = test_func()
    assert result == {"key": "value"}

def test_cache_loads_existing_file():
    import tempfile
    import os
    import pickle
    test_file = tempfile.NamedTemporaryFile(delete=False).name
    try:
        with open(test_file, "wb") as f:
            pickle.dump({"key": "cached_value"}, f)
        @cache(test_file, verbose=True, name="test")
        def test_func():
            return {"key": "value"}
        result = test_func()
        assert result == {"key": "cached_value"}
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


# LLM-generated content at query #4
#--------------------------

```python
def test_cache_with_path_and_verbose():
    @cache("test_cache.pkl", verbose=True, name="test")
    def test_func():
        return 42

    assert test_func() == 42
    assert os.path.exists("test_cache.pkl")
    os.remove("test_cache.pkl")


def test_cache_with_path_and_no_verbose():
    @cache("test_cache.pkl", verbose=False, name="test")
    def test_func():
        return 42

    assert test_func() == 42
    assert os.path.exists("test_cache.pkl")
    os.remove("test_cache.pkl")


def test_cache_without_path():
    @cache(None, verbose=True, name="test")
    def test_func():
        return 42

    assert test_func() == 42
    assert not os.path.exists("test_cache.pkl")


def test_cache_with_existing_cache_file():
    with open("test_cache.pkl", "wb") as f:
        pickle.dump(42, f)

    @cache("test_cache.pkl", verbose=True, name="test")
    def test_func():
        return 0

    assert test_func() == 42
    assert os.path.exists("test_cache.pkl")
    os.remove("test_cache.pkl")


# LLM-generated content at query #5
#--------------------------

```python
def test_cache_with_existing_file():
    import tempfile
    import os
    import pickle
    import functools

    def dummy_func():
        return 42

    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
        pickle.dump(100, f)

    @cache(path, verbose=False)
    def wrapped_func():
        return dummy_func()

    result = wrapped_func()
    os.remove(path)
    assert result == 100

def test_cache_with_non_existing_file():
    import tempfile
    import os
    import pickle
    import functools

    def dummy_func():
        return 42

    path = "non_existent_file.pkl"

    @cache(path, verbose=False)
    def wrapped_func():
        return dummy_func()

    result = wrapped_func()
    assert result == 42
    os.remove(path)

def test_cache_with_verbose():
    import tempfile
    import os
    import pickle
    import functools

    def dummy_func():
        return 42

    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
        pickle.dump(100, f)

    @cache(path, verbose=True, name="test")
    def wrapped_func():
        return dummy_func()

    result = wrapped_func()
    os.remove(path)
    assert result == 100

def test_cache_with_no_path():
    import tempfile
    import os
    import pickle
    import functools

    def dummy_func():
        return 42

    @cache(None, verbose=False)
    def wrapped_func():
        return dummy_func()

    result = wrapped_func()
    assert result == 42


# LLM-generated content at query #6
#--------------------------

```python
def test_scandir_with_path_object():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / 'file1.txt').touch()
        (Path(tmpdir) / 'file2.txt').touch()
        result = list(scandir(Path(tmpdir)))
        assert len(result) == 2
        assert all(isinstance(p, Path) for p in result)
        assert any('file1.txt' in str(p) for p in result)
        assert any('file2.txt' in str(p) for p in result)

def test_scandir_with_string_path():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        import os
        open(os.path.join(tmpdir, 'file1.txt'), 'w').close()
        open(os.path.join(tmpdir, 'file2.txt'), 'w').close()
        result = list(scandir(tmpdir))
        assert len(result) == 2
        assert all(isinstance(p, str) for p in result)
        assert any('file1.txt' in p for p in result)
        assert any('file2.txt' in p for p in result)

def test_scandir_empty_directory():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        result = list(scandir(Path(tmpdir)))
        assert len(result) == 0


# LLM-generated content at query #7
#--------------------------

```python
def test_scandir_with_string_path():
    path = "/some/string/path"
    result = list(scandir(path))
    assert not isinstance(path, Path)


# LLM-generated content at query #8
#--------------------------

```python
def test_scandir_with_pathlib_path():
    import os
    from pathlib import Path
    temp_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "temp_dir"
    temp_dir.mkdir(exist_ok=True)
    (temp_dir / "file1.txt").touch()
    (temp_dir / "file2.txt").touch()
    result = list(scandir(temp_dir))
    assert len(result) == 2
    assert all(isinstance(path, Path) for path in result)
    assert any("file1.txt" in str(path) for path in result)
    assert any("file2.txt" in str(path) for path in result)
    os.remove(temp_dir / "file1.txt")
    os.remove(temp_dir / "file2.txt")
    temp_dir.rmdir()

def test_scandir_with_str_path():
    import os
    from pathlib import Path
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_dir")
    os.makedirs(temp_dir, exist_ok=True)
    open(os.path.join(temp_dir, "file1.txt"), "w").close()
    open(os.path.join(temp_dir, "file2.txt"), "w").close()
    result = list(scandir(temp_dir))
    assert len(result) == 2
    assert all(isinstance(path, str) for path in result)
    assert any("file1.txt" in path for path in result)
    assert any("file2.txt" in path for path in result)
    os.remove(os.path.join(temp_dir, "file1.txt"))
    os.remove(os.path.join(temp_dir, "file2.txt"))
    os.rmdir(temp_dir)


# LLM-generated content at query #9
#--------------------------

```python
def test_scandir_with_pathlib_path():
    test_dir = Path("test_dir")
    test_dir.mkdir(exist_ok=True)
    (test_dir / "file1.txt").touch()
    (test_dir / "file2.txt").touch()
    result = list(scandir(test_dir))
    assert len(result) == 2
    assert isinstance(result[0], Path)
    assert isinstance(result[1], Path)
    assert any("file1.txt" in str(path) for path in result)
    assert any("file2.txt" in str(path) for path in result)
    test_dir.rmdir()

def test_scandir_with_str_path():
    test_dir = "test_dir"
    os.makedirs(test_dir, exist_ok=True)
    open(os.path.join(test_dir, "file1.txt"), "w").close()
    open(os.path.join(test_dir, "file2.txt"), "w").close()
    result = list(scandir(test_dir))
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)
    assert any("file1.txt" in path for path in result)
    assert any("file2.txt" in path for path in result)
    os.rmdir(test_dir)


# LLM-generated content at query #10
#--------------------------

```python
def test_scandir_with_path_instance():
    from pathlib import Path
    import os
    test_dir = Path(__file__).parent
    for path in scandir(test_dir):
        assert isinstance(path, Path)


# LLM-generated content at query #11
#--------------------------

```python
def test_cache_with_existing_file():
    import tempfile
    import os
    import pickle
    test_data = {"key": "value"}
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        pickle.dump(test_data, temp_file)
        temp_file_path = temp_file.name

    @cache(temp_file_path)
    def dummy_func():
        return {"key": "new_value"}

    result = dummy_func()
    os.remove(temp_file_path)
    assert result == test_data


def test_cache_with_non_existing_file():
    import tempfile
    import os
    test_data = {"key": "new_value"}
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name
        os.remove(temp_file_path)

    @cache(temp_file_path)
    def dummy_func():
        return test_data

    result = dummy_func()
    assert result == test_data
    with open(temp_file_path, "rb") as f:
        loaded_data = pickle.load(f)
    assert loaded_data == test_data
    os.remove(temp_file_path)


def test_cache_with_no_path():
    @cache(None)
    def dummy_func():
        return {"key": "value"}

    result = dummy_func()
    assert result == {"key": "value"}


def test_cache_with_custom_name():
    import tempfile
    import os
    test_data = {"key": "value"}
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        pickle.dump(test_data, temp_file)
        temp_file_path = temp_file.name

    @cache(temp_file_path, name="custom")
    def dummy_func():
        return {"key": "new_value"}

    result = dummy_func()
    os.remove(temp_file_path)
    assert result == test_data


def test_cache_with_verbose_false():
    import tempfile
    import os
    test_data = {"key": "value"}
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        pickle.dump(test_data, temp_file)
        temp_file_path = temp_file.name

    @cache(temp_file_path, verbose=False)
    def dummy_func():
        return {"key": "new_value"}

    result = dummy_func()
    os.remove(temp_file_path)
    assert result == test_data


# LLM-generated content at query #12
#--------------------------

```python
import os
from pathlib import Path

def test_scandir_with_pathlib_path():
    test_dir = Path("test_dir")
    test_dir.mkdir(exist_ok=True)
    (test_dir / "file1.txt").touch()
    (test_dir / "file2.txt").touch()
    result = list(scandir(test_dir))
    assert len(result) == 2
    assert isinstance(result[0], Path)
    assert isinstance(result[1], Path)
    os.remove(test_dir / "file1.txt")
    os.remove(test_dir / "file2.txt")
    test_dir.rmdir()

def test_scandir_with_str_path():
    test_dir = "test_dir"
    os.makedirs(test_dir, exist_ok=True)
    open(os.path.join(test_dir, "file1.txt"), "w").close()
    open(os.path.join(test_dir, "file2.txt"), "w").close()
    result = list(scandir(test_dir))
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)
    os.remove(os.path.join(test_dir, "file1.txt"))
    os.remove(os.path.join(test_dir, "file2.txt"))
    os.rmdir(test_dir)


# LLM-generated content at query #13
#--------------------------

```
def test_scandir_with_str_path():
    path = "test_dir"
    result = list(scandir(path))
    assert isinstance(result[0], str)


# LLM-generated content at query #14
#--------------------------

```python
def test_cache_predicate_evaluates_to_true():
    import os
    import tempfile
    import pickle

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name
        pickle.dump("test_data", tmp_file)

    # Define a simple function to decorate
    def test_func():
        return "new_data"

    # Apply the cache decorator
    decorated_func = cache(tmp_path)(test_func)

    # Call the decorated function - should load from cache
    result = decorated_func()

    # Clean up
    os.unlink(tmp_path)

    # Assert the predicate evaluated to True (file existed)
    assert result == "test_data"


# LLM-generated content at query #15
#--------------------------

```python
def test_copy_tree_creates_destination_directory():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    copy_tree(src, dst)
    assert os.path.exists(dst)

def test_copy_tree_copies_files():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("test content")
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "test_file.txt"))

def test_copy_tree_overwrites_files_when_overwrite_is_true():
    src = "test_src"
    dst = "test_dst"
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
    src = "test_src"
    dst = "test_dst"
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
    src = "test_src"
    dst = "test_dst"
    os.makedirs(os.path.join(src, "subdir"), exist_ok=True)
    with open(os.path.join(src, "subdir", "test_file.txt"), "w") as f:
        f.write("test content")
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "subdir", "test_file.txt"))


# LLM-generated content at query #16
#--------------------------

```python
def test_scandir_with_pathlib_path():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        result = list(scandir(tmp_path))
        assert len(result) == 2
        assert all(isinstance(p, Path) for p in result)

def test_scandir_with_str_path():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        (tmp_dir / "file1.txt").touch()
        (tmp_dir / "file2.txt").touch()
        result = list(scandir(tmp_dir))
        assert len(result) == 2
        assert all(isinstance(p, str) for p in result)


# LLM-generated content at query #17
#--------------------------

```python
def test_scandir_with_non_path_non_str_input():
    result = list(scandir(123))
    assert not result


# LLM-generated content at query #18
#--------------------------

```
def test_copy_tree_creates_destination_directory():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = os.path.join(tempfile.mkdtemp(), "new_dir")
    try:
        copy_tree(src, dst)
        assert os.path.exists(dst)
    finally:
        shutil.rmtree(src)
        shutil.rmtree(os.path.dirname(dst), ignore_errors=True)

def test_copy_tree_copies_files():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        with open(os.path.join(src, "test.txt"), "w") as f:
            f.write("test")
        copy_tree(src, dst)
        assert os.path.exists(os.path.join(dst, "test.txt"))
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_overwrites_files_when_enabled():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        with open(os.path.join(src, "test.txt"), "w") as f:
            f.write("new")
        with open(os.path.join(dst, "test.txt"), "w") as f:
            f.write("old")
        copy_tree(src, dst, overwrite=True)
        with open(os.path.join(dst, "test.txt"), "r") as f:
            assert f.read() == "new"
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_skips_files_when_not_overwriting():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        with open(os.path.join(src, "test.txt"), "w") as f:
            f.write("new")
        with open(os.path.join(dst, "test.txt"), "w") as f:
            f.write("old")
        copy_tree(src, dst, overwrite=False)
        with open(os.path.join(dst, "test.txt"), "r") as f:
            assert f.read() == "old"
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_copies_subdirectories():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(src, "subdir"))
        with open(os.path.join(src, "subdir", "test.txt"), "w") as f:
            f.write("test")
        copy_tree(src, dst)
        assert os.path.exists(os.path.join(dst, "subdir", "test.txt"))
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_preserves_file_stats():
    import tempfile
    import shutil
    import time
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        test_file = os.path.join(src, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        original_stat = os.stat(test_file)
        time.sleep(0.1)  # ensure timestamps are different
        copy_tree(src, dst)
        copied_stat = os.stat(os.path.join(dst, "test.txt"))
        assert abs(original_stat.st_mtime - copied_stat.st_mtime) < 0.1
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)


# LLM-generated content at query #19
#--------------------------

```python
import os
from pathlib import Path

def test_scandir_with_pathlib_path():
    test_dir = Path(os.path.dirname(__file__))
    result = list(scandir(test_dir))
    assert all(isinstance(path, Path) for path in result)

def test_scandir_with_str_path():
    test_dir = os.path.dirname(__file__)
    result = list(scandir(test_dir))
    assert all(isinstance(path, str) for path in result)

def test_scandir_returns_correct_paths():
    test_dir = Path(os.path.dirname(__file__))
    expected = [entry.path for entry in os.scandir(test_dir)]
    result = [str(path) for path in scandir(test_dir)]
    assert result == expected


# LLM-generated content at query #20
#--------------------------

```python
def test_cache_decorator_with_none_path():
    path = None
    verbose = True
    name = "test_cache"

    @cache(path, verbose, name)
    def dummy_func():
        return "test_value"

    result = dummy_func()
    assert result == "test_value"


# LLM-generated content at query #21
#--------------------------

```
def test_scandir_with_str_path():
    import tempfile
    import os
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        
        result = list(scandir(tmpdir))
        assert len(result) == 1
        assert isinstance(result[0], str)
        assert os.path.normpath(result[0]) == os.path.normpath(test_file)

def test_scandir_with_pathlib_path():
    import tempfile
    import os
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / 'test.txt'
        with open(test_file, 'w') as f:
            f.write('test')
        
        result = list(scandir(Path(tmpdir)))
        assert len(result) == 1
        assert isinstance(result[0], Path)
        assert result[0].absolute() == test_file.absolute()


# LLM-generated content at query #22
#--------------------------

```python
def test_scandir_with_str_path():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = list(scandir(tmpdir))
        assert all(isinstance(p, str) for p in result)

def test_scandir_with_pathlib_path():
    import tempfile
    import os
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        result = list(scandir(Path(tmpdir)))
        assert all(isinstance(p, Path) for p in result)


# LLM-generated content at query #23
#--------------------------

```python
def test_scandir_with_pathlib_path():
    from pathlib import Path
    import os
    test_dir = Path(os.path.dirname(__file__))
    result = list(scandir(test_dir))
    assert all(isinstance(item, Path) for item in result)

def test_scandir_with_str_path():
    import os
    test_dir = os.path.dirname(__file__)
    result = list(scandir(test_dir))
    assert all(isinstance(item, str) for item in result)

def test_scandir_returns_absolute_paths():
    from pathlib import Path
    import os
    test_dir = Path(os.path.dirname(__file__))
    result = list(scandir(test_dir))
    assert all(item.is_absolute() for item in result)

def test_scandir_returns_children():
    from pathlib import Path
    import os
    test_dir = Path(os.path.dirname(__file__))
    result = list(scandir(test_dir))
    assert len(result) > 0


# LLM-generated content at query #24
#--------------------------

```python
def test_copy_tree_creates_destination_directory():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    copy_tree(src, dst)
    assert os.path.exists(dst)
    shutil.rmtree(src)
    shutil.rmtree(dst)

def test_copy_tree_copies_files():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("test content")
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "test_file.txt"))
    shutil.rmtree(src)
    shutil.rmtree(dst)

def test_copy_tree_copies_subdirectories():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(os.path.join(src, "subdir"), exist_ok=True)
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "subdir"))
    shutil.rmtree(src)
    shutil.rmtree(dst)

def test_copy_tree_overwrites_files_when_overwrite_is_true():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("new content")
    with open(os.path.join(dst, "test_file.txt"), "w") as f:
        f.write("old content")
    copy_tree(src, dst, overwrite=True)
    with open(os.path.join(dst, "test_file.txt"), "r") as f:
        assert f.read() == "new content"
    shutil.rmtree(src)
    shutil.rmtree(dst)

def test_copy_tree_does_not_overwrite_files_when_overwrite_is_false():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("new content")
    with open(os.path.join(dst, "test_file.txt"), "w") as f:
        f.write("old content")
    copy_tree(src, dst, overwrite=False)
    with open(os.path.join(dst, "test_file.txt"), "r") as f:
        assert f.read() == "old content"
    shutil.rmtree(src)
    shutil.rmtree(dst)


# LLM-generated content at query #25
#--------------------------

```
def test_copy_tree_creates_destination_directory():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = os.path.join(tempfile.mkdtemp(), "new_dir")
    try:
        copy_tree(src, dst)
        assert os.path.exists(dst)
    finally:
        shutil.rmtree(src)
        shutil.rmtree(os.path.dirname(dst), ignore_errors=True)

def test_copy_tree_copies_files():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        with open(os.path.join(src, "test.txt"), "w") as f:
            f.write("hello")
        copy_tree(src, dst)
        assert os.path.exists(os.path.join(dst, "test.txt"))
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_copies_subdirectories():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(src, "subdir"))
        with open(os.path.join(src, "subdir", "test.txt"), "w") as f:
            f.write("hello")
        copy_tree(src, dst)
        assert os.path.exists(os.path.join(dst, "subdir", "test.txt"))
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_does_not_overwrite_by_default():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        with open(os.path.join(src, "test.txt"), "w") as f:
            f.write("src_content")
        with open(os.path.join(dst, "test.txt"), "w") as f:
            f.write("dst_content")
        copy_tree(src, dst)
        with open(os.path.join(dst, "test.txt"), "r") as f:
            assert f.read() == "dst_content"
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_overwrites_when_requested():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        with open(os.path.join(src, "test.txt"), "w") as f:
            f.write("src_content")
        with open(os.path.join(dst, "test.txt"), "w") as f:
            f.write("dst_content")
        copy_tree(src, dst, overwrite=True)
        with open(os.path.join(dst, "test.txt"), "r") as f:
            assert f.read() == "src_content"
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_copies_file_stats():
    import tempfile
    import shutil
    import time
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        test_file = os.path.join(src, "test.txt")
        with open(test_file, "w") as f:
            f.write("hello")
        expected_stat = os.stat(test_file)
        copy_tree(src, dst)
        actual_stat = os.stat(os.path.join(dst, "test.txt"))
        assert abs(expected_stat.st_mtime - actual_stat.st_mtime) < 1
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_copies_directory_stats():
    import tempfile
    import shutil
    import time
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        expected_stat = os.stat(src)
        copy_tree(src, dst)
        actual_stat = os.stat(dst)
        assert abs(expected_stat.st_mtime - actual_stat.st_mtime) < 1
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)


# LLM-generated content at query #26
#--------------------------

```
def test_copy_tree_overwrite_true():
    src = "/tmp/test_src"
    dst = "/tmp/test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    open(os.path.join(src, "test.txt"), "w").close()
    open(os.path.join(dst, "test.txt"), "w").close()
    copy_tree(src, dst, overwrite=True)
    assert os.path.exists(os.path.join(dst, "test.txt"))

def test_copy_tree_overwrite_false():
    src = "/tmp/test_src"
    dst = "/tmp/test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    open(os.path.join(src, "test.txt"), "w").close()
    copy_tree(src, dst, overwrite=False)
    assert os.path.exists(os.path.join(dst, "test.txt"))

def test_copy_tree_dst_not_exists():
    src = "/tmp/test_src"
    dst = "/tmp/test_dst"
    os.makedirs(src, exist_ok=True)
    open(os.path.join(src, "test.txt"), "w").close()
    copy_tree(src, dst, overwrite=False)
    assert os.path.exists(os.path.join(dst, "test.txt"))


# LLM-generated content at query #27
#--------------------------

```python
def test_copy_tree_creates_destination_directory_if_not_exists():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    copy_tree(src, dst)
    assert os.path.exists(dst)
    os.rmdir(dst)
    os.rmdir(src)

def test_copy_tree_copies_files_from_source_to_destination():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("test")
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "test_file.txt"))
    os.remove(os.path.join(dst, "test_file.txt"))
    os.remove(os.path.join(src, "test_file.txt"))
    os.rmdir(dst)
    os.rmdir(src)

def test_copy_tree_overwrites_files_if_overwrite_is_true():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("new_content")
    with open(os.path.join(dst, "test_file.txt"), "w") as f:
        f.write("old_content")
    copy_tree(src, dst, overwrite=True)
    with open(os.path.join(dst, "test_file.txt"), "r") as f:
        assert f.read() == "new_content"
    os.remove(os.path.join(dst, "test_file.txt"))
    os.remove(os.path.join(src, "test_file.txt"))
    os.rmdir(dst)
    os.rmdir(src)

def test_copy_tree_does_not_overwrite_files_if_overwrite_is_false():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("new_content")
    with open(os.path.join(dst, "test_file.txt"), "w") as f:
        f.write("old_content")
    copy_tree(src, dst, overwrite=False)
    with open(os.path.join(dst, "test_file.txt"), "r") as f:
        assert f.read() == "old_content"
    os.remove(os.path.join(dst, "test_file.txt"))
    os.remove(os.path.join(src, "test_file.txt"))
    os.rmdir(dst)
    os.rmdir(src)

def test_copy_tree_copies_subdirectories():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(os.path.join(src, "subdir"), exist_ok=True)
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "subdir"))
    os.rmdir(os.path.join(dst, "subdir"))
    os.rmdir(dst)
    os.rmdir(os.path.join(src, "subdir"))
    os.rmdir(src)


# LLM-generated content at query #28
#--------------------------

```python
import os
import pickle
from pathlib import Path

def test_cache_with_path_and_verbose():
    test_path = "test_cache.pkl"
    test_data = [1, 2, 3]

    @cache(test_path, verbose=True, name="test")
    def test_func():
        return test_data

    # First call should save to cache
    result = test_func()
    assert result == test_data
    assert os.path.exists(test_path)

    # Second call should load from cache
    result = test_func()
    assert result == test_data

    os.remove(test_path)

def test_cache_with_path_and_no_verbose():
    test_path = "test_cache.pkl"
    test_data = [1, 2, 3]

    @cache(test_path, verbose=False, name="test")
    def test_func():
        return test_data

    # First call should save to cache
    result = test_func()
    assert result == test_data
    assert os.path.exists(test_path)

    # Second call should load from cache
    result = test_func()
    assert result == test_data

    os.remove(test_path)

def test_cache_with_no_path():
    test_data = [1, 2, 3]

    @cache(None, verbose=True, name="test")
    def test_func():
        return test_data

    # Call should always execute the function
    result = test_func()
    assert result == test_data

def test_cache_with_non_existent_path():
    test_path = "non_existent_cache.pkl"
    test_data = [1, 2, 3]

    @cache(test_path, verbose=True, name="test")
    def test_func():
        return test_data

    # First call should save to cache
    result = test_func()
    assert result == test_data
    assert os.path.exists(test_path)

    # Second call should load from cache
    result = test_func()
    assert result == test_data

    os.remove(test_path)


# LLM-generated content at query #29
#--------------------------

```python
def test_copy_tree_overwrite_true():
    src = "test_src"
    dst = "test_dst"
    overwrite = True
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("test content")
    copy_tree(src, dst, overwrite=overwrite)
    assert os.path.exists(os.path.join(dst, "test_file.txt"))
    shutil.rmtree(src)
    shutil.rmtree(dst)

def test_copy_tree_overwrite_false():
    src = "test_src"
    dst = "test_dst"
    overwrite = False
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("test content")
    copy_tree(src, dst, overwrite=overwrite)
    assert os.path.exists(os.path.join(dst, "test_file.txt"))
    shutil.rmtree(src)
    shutil.rmtree(dst)


# LLM-generated content at query #30
#--------------------------

```python
def test_copy_tree_basic():
    import tempfile
    import os
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        with open(os.path.join(src, 'test.txt'), 'w') as f:
            f.write('test')
        os.mkdir(os.path.join(src, 'subdir'))
        with open(os.path.join(src, 'subdir', 'subtest.txt'), 'w') as f:
            f.write('subtest')
        
        copy_tree(src, dst)
        
        assert os.path.exists(os.path.join(dst, 'test.txt'))
        assert os.path.exists(os.path.join(dst, 'subdir'))
        assert os.path.exists(os.path.join(dst, 'subdir', 'subtest.txt'))
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_overwrite():
    import tempfile
    import os
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        with open(os.path.join(src, 'test.txt'), 'w') as f:
            f.write('new content')
        with open(os.path.join(dst, 'test.txt'), 'w') as f:
            f.write('old content')
        
        copy_tree(src, dst, overwrite=True)
        
        with open(os.path.join(dst, 'test.txt'), 'r') as f:
            assert f.read() == 'new content'
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_no_overwrite():
    import tempfile
    import os
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        with open(os.path.join(src, 'test.txt'), 'w') as f:
            f.write('new content')
        with open(os.path.join(dst, 'test.txt'), 'w') as f:
            f.write('old content')
        
        copy_tree(src, dst, overwrite=False)
        
        with open(os.path.join(dst, 'test.txt'), 'r') as f:
            assert f.read() == 'old content'
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_empty_src():
    import tempfile
    import os
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        copy_tree(src, dst)
        assert os.listdir(dst) == []
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_nonexistent_dst():
    import tempfile
    import os
    import shutil
    src = tempfile.mkdtemp()
    dst = os.path.join(tempfile.mkdtemp(), 'nonexistent')
    try:
        with open(os.path.join(src, 'test.txt'), 'w') as f:
            f.write('test')
        
        copy_tree(src, dst)
        
        assert os.path.exists(os.path.join(dst, 'test.txt'))
    finally:
        shutil.rmtree(src)
        shutil.rmtree(os.path.dirname(dst))


# LLM-generated content at query #31
#--------------------------

def test_cache_with_path_and_verbose():
    import tempfile
    import os
    import pickle
    test_data = {"key": "value"}
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        test_path = tmp_file.name
    with open(test_path, "wb") as f:
        pickle.dump(test_data, f)
    
    @cache(test_path, verbose=True, name="test")
    def test_func():
        return {"key": "different_value"}
    
    result = test_func()
    os.unlink(test_path)
    assert result == test_data

def test_cache_with_path_and_no_verbose():
    import tempfile
    import os
    import pickle
    test_data = {"key": "value"}
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        test_path = tmp_file.name
    with open(test_path, "wb") as f:
        pickle.dump(test_data, f)
    
    @cache(test_path, verbose=False, name="test")
    def test_func():
        return {"key": "different_value"}
    
    result = test_func()
    os.unlink(test_path)
    assert result == test_data

def test_cache_without_path():
    @cache(None, verbose=True, name="test")
    def test_func():
        return {"key": "value"}
    
    result = test_func()
    assert result == {"key": "value"}

def test_cache_with_nonexistent_path():
    import tempfile
    import os
    test_path = tempfile.mktemp()
    
    @cache(test_path, verbose=True, name="test")
    def test_func():
        return {"key": "value"}
    
    result = test_func()
    assert os.path.exists(test_path)
    os.unlink(test_path)
    assert result == {"key": "value"}

def test_cache_with_default_name():
    import tempfile
    import os
    test_path = tempfile.mktemp()
    
    @cache(test_path, verbose=True)
    def test_func():
        return {"key": "value"}
    
    result = test_func()
    assert os.path.exists(test_path)
    os.unlink(test_path)
    assert result == {"key": "value"}


# LLM-generated content at query #32
#--------------------------

```python
def test_get_worker_id_not_pool_worker():
    worker_id = get_worker_id()
    assert worker_id is None


# LLM-generated content at query #33
#--------------------------

def test_cache_with_path_and_verbose():
    import os
    import pickle
    from tempfile import NamedTemporaryFile
    test_path = NamedTemporaryFile(delete=False).name
    test_data = {"key": "value"}
    with open(test_path, "wb") as f:
        pickle.dump(test_data, f)

    @cache(test_path, verbose=True, name="test")
    def test_func():
        return {"new_key": "new_value"}

    result = test_func()
    assert result == test_data
    os.unlink(test_path)

def test_cache_with_path_and_no_verbose():
    import os
    import pickle
    from tempfile import NamedTemporaryFile
    test_path = NamedTemporaryFile(delete=False).name
    test_data = {"key": "value"}
    with open(test_path, "wb") as f:
        pickle.dump(test_data, f)

    @cache(test_path, verbose=False, name="test")
    def test_func():
        return {"new_key": "new_value"}

    result = test_func()
    assert result == test_data
    os.unlink(test_path)

def test_cache_with_none_path():
    @cache(None, verbose=True, name="test")
    def test_func():
        return {"key": "value"}

    result = test_func()
    assert result == {"key": "value"}

def test_cache_with_non_existent_path():
    import os
    from tempfile import NamedTemporaryFile
    test_path = NamedTemporaryFile(delete=False).name
    os.unlink(test_path)

    @cache(test_path, verbose=True, name="test")
    def test_func():
        return {"key": "value"}

    result = test_func()
    assert result == {"key": "value"}
    assert os.path.exists(test_path)
    os.unlink(test_path)

def test_cache_with_default_name():
    import os
    from tempfile import NamedTemporaryFile
    test_path = NamedTemporaryFile(delete=False).name
    os.unlink(test_path)

    @cache(test_path, verbose=True)
    def test_func():
        return {"key": "value"}

    result = test_func()
    assert result == {"key": "value"}
    assert os.path.exists(test_path)
    os.unlink(test_path)


# LLM-generated content at query #34
#--------------------------

def test_cache_path_is_none():
    path = None
    assert not (path is not None and os.path.exists(path))


# LLM-generated content at query #35
#--------------------------

```python
def test_scandir_yields_string_when_path_is_string():
    import os
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        
        result = list(scandir(tmpdir))
        assert isinstance(result[0], str)


# LLM-generated content at query #36
#--------------------------

```python
import os
import pickle
from flutes import log
from flutes.fs import cache
from pathlib import Path

def test_cache_with_existing_file():
    test_file = Path("test_cache.pkl")
    test_data = {"key": "value"}
    with open(test_file, "wb") as f:
        pickle.dump(test_data, f)

    @cache(test_file)
    def test_func():
        return {"key": "new_value"}

    result = test_func()
    assert result == test_data
    os.remove(test_file)

def test_cache_with_non_existing_file():
    test_file = Path("test_cache.pkl")
    expected_data = {"key": "value"}

    @cache(test_file)
    def test_func():
        return expected_data

    result = test_func()
    assert result == expected_data
    assert os.path.exists(test_file)
    os.remove(test_file)

def test_cache_with_no_path():
    expected_data = {"key": "value"}

    @cache(None)
    def test_func():
        return expected_data

    result = test_func()
    assert result == expected_data
    assert not os.path.exists("test_cache.pkl")

def test_cache_with_verbose_false():
    test_file = Path("test_cache.pkl")
    expected_data = {"key": "value"}

    @cache(test_file, verbose=False)
    def test_func():
        return expected_data

    result = test_func()
    assert result == expected_data
    assert os.path.exists(test_file)
    os.remove(test_file)

def test_cache_with_custom_name():
    test_file = Path("test_cache.pkl")
    expected_data = {"key": "value"}

    @cache(test_file, name="custom_cache")
    def test_func():
        return expected_data

    result = test_func()
    assert result == expected_data
    assert os.path.exists(test_file)
    os.remove(test_file)


# LLM-generated content at query #37
#--------------------------

```python
def test_copy_tree_creates_destination_directory_if_not_exists():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    copy_tree(src, dst)
    assert os.path.exists(dst)

def test_copy_tree_copies_files():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, "file.txt"), "w") as f:
        f.write("test")
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "file.txt"))

def test_copy_tree_copies_directories():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(os.path.join(src, "subdir"), exist_ok=True)
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "subdir"))

def test_copy_tree_overwrites_files_if_overwrite_is_true():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "file.txt"), "w") as f:
        f.write("new")
    with open(os.path.join(dst, "file.txt"), "w") as f:
        f.write("old")
    copy_tree(src, dst, overwrite=True)
    with open(os.path.join(dst, "file.txt"), "r") as f:
        assert f.read() == "new"

def test_copy_tree_does_not_overwrite_files_if_overwrite_is_false():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "file.txt"), "w") as f:
        f.write("new")
    with open(os.path.join(dst, "file.txt"), "w") as f:
        f.write("old")
    copy_tree(src, dst, overwrite=False)
    with open(os.path.join(dst, "file.txt"), "r") as f:
        assert f.read() == "old"

def test_copy_tree_preserves_file_stats():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, "file.txt"), "w") as f:
        f.write("test")
    copy_tree(src, dst)
    src_stat = os.stat(os.path.join(src, "file.txt"))
    dst_stat = os.stat(os.path.join(dst, "file.txt"))
    assert src_stat.st_mode == dst_stat.st_mode


# LLM-generated content at query #38
#--------------------------

```
def test_scandir_with_str_path():
    path = "/tmp"
    result = next(scandir(path))
    assert isinstance(result, str)
    assert not isinstance(result, Path)


# LLM-generated content at query #39
#--------------------------

```python
def test_predicate_at_line_8_evaluates_to_true():
    path = Path("/example/directory")
    assert isinstance(path, Path)


# LLM-generated content at query #40
#--------------------------

```python
def test_copy_tree_creates_destination_directory_if_not_exists():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = os.path.join(tempfile.mkdtemp(), "nonexistent")
    try:
        copy_tree(src, dst)
        assert os.path.exists(dst)
    finally:
        shutil.rmtree(src)
        if os.path.exists(dst):
            shutil.rmtree(dst)

def test_copy_tree_copies_files():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        with open(os.path.join(src, "test.txt"), "w") as f:
            f.write("test")
        copy_tree(src, dst)
        assert os.path.exists(os.path.join(dst, "test.txt"))
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_copies_subdirectories():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(src, "subdir"))
        with open(os.path.join(src, "subdir", "test.txt"), "w") as f:
            f.write("test")
        copy_tree(src, dst)
        assert os.path.exists(os.path.join(dst, "subdir", "test.txt"))
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_does_not_overwrite_existing_files_when_overwrite_false():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        with open(os.path.join(src, "test.txt"), "w") as f:
            f.write("src")
        with open(os.path.join(dst, "test.txt"), "w") as f:
            f.write("dst")
        copy_tree(src, dst, overwrite=False)
        with open(os.path.join(dst, "test.txt"), "r") as f:
            assert f.read() == "dst"
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_overwrites_existing_files_when_overwrite_true():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        with open(os.path.join(src, "test.txt"), "w") as f:
            f.write("src")
        with open(os.path.join(dst, "test.txt"), "w") as f:
            f.write("dst")
        copy_tree(src, dst, overwrite=True)
        with open(os.path.join(dst, "test.txt"), "r") as f:
            assert f.read() == "src"
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_cache_file_exists():
    path = "test_cache.pkl"
    with open(path, "wb") as f:
        pickle.dump("cached_value", f)
    @cache(path)
    def func():
        return "new_value"
    result = func()
    assert result == "cached_value"
    os.remove(path)

def test_cache_file_not_exists():
    path = "test_cache.pkl"
    @cache(path)
    def func():
        return "new_value"
    result = func()
    assert result == "new_value"
    os.remove(path)

def test_cache_no_path():
    @cache(None)
    def func():
        return "new_value"
    result = func()
    assert result == "new_value"

def test_cache_verbose():
    path = "test_cache.pkl"
    @cache(path, verbose=True)
    def func():
        return "new_value"
    func()
    os.remove(path)

def test_cache_name():
    path = "test_cache.pkl"
    @cache(path, name="custom_name")
    def func():
        return "new_value"
    func()
    os.remove(path)


# LLM-generated content at query #2
#--------------------------

```python
def test_copy_tree_creates_destination_directory():
    import tempfile
    import os
    import shutil
    src = tempfile.mkdtemp()
    dst = os.path.join(tempfile.mkdtemp(), 'new_dir')
    copy_tree(src, dst)
    assert os.path.exists(dst)

def test_copy_tree_copies_files():
    import tempfile
    import os
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    with open(os.path.join(src, 'test_file.txt'), 'w') as f:
        f.write('test content')
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, 'test_file.txt'))

def test_copy_tree_copies_directories():
    import tempfile
    import os
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    os.makedirs(os.path.join(src, 'sub_dir'))
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, 'sub_dir'))

def test_copy_tree_overwrites_files_when_overwrite_is_true():
    import tempfile
    import os
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    with open(os.path.join(src, 'test_file.txt'), 'w') as f:
        f.write('new content')
    with open(os.path.join(dst, 'test_file.txt'), 'w') as f:
        f.write('old content')
    copy_tree(src, dst, overwrite=True)
    with open(os.path.join(dst, 'test_file.txt'), 'r') as f:
        assert f.read() == 'new content'

def test_copy_tree_does_not_overwrite_files_when_overwrite_is_false():
    import tempfile
    import os
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    with open(os.path.join(src, 'test_file.txt'), 'w') as f:
        f.write('new content')
    with open(os.path.join(dst, 'test_file.txt'), 'w') as f:
        f.write('old content')
    copy_tree(src, dst, overwrite=False)
    with open(os.path.join(dst, 'test_file.txt'), 'r') as f:
        assert f.read() == 'old content'

def test_copy_tree_copies_file_statistics():
    import tempfile
    import os
    import shutil
    import time
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    with open(os.path.join(src, 'test_file.txt'), 'w') as f:
        f.write('test content')
    time.sleep(1)
    copy_tree(src, dst)
    src_stat = os.stat(src)
    dst_stat = os.stat(dst)
    assert src_stat.st_mtime == dst_stat.st_mtime


# LLM-generated content at query #3
#--------------------------

```python
def test_scandir_with_pathlib_path():
    import os
    from pathlib import Path
    test_dir = Path("test_dir")
    test_dir.mkdir(exist_ok=True)
    (test_dir / "file1.txt").touch()
    (test_dir / "file2.txt").touch()
    result = list(scandir(test_dir))
    assert len(result) == 2
    assert isinstance(result[0], Path)
    assert "file1.txt" in str(result[0]) or "file2.txt" in str(result[0])
    assert "file1.txt" in str(result[1]) or "file2.txt" in str(result[1])
    os.remove(test_dir / "file1.txt")
    os.remove(test_dir / "file2.txt")
    os.rmdir(test_dir)

def test_scandir_with_str_path():
    import os
    test_dir = "test_dir"
    os.makedirs(test_dir, exist_ok=True)
    open(os.path.join(test_dir, "file1.txt"), "w").close()
    open(os.path.join(test_dir, "file2.txt"), "w").close()
    result = list(scandir(test_dir))
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert "file1.txt" in result[0] or "file2.txt" in result[0]
    assert "file1.txt" in result[1] or "file2.txt" in result[1]
    os.remove(os.path.join(test_dir, "file1.txt"))
    os.remove(os.path.join(test_dir, "file2.txt"))
    os.rmdir(test_dir)


# LLM-generated content at query #4
#--------------------------

```python
def test_readable_size_bytes():
    assert readable_size(512) == "512.00"

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
    assert readable_size(1024, n_digits=3) == "1.000K"

def test_readable_size_fractional():
    assert readable_size(1500) == "1.46K"

def test_readable_size_zero():
    assert readable_size(0) == "0.00"

def test_readable_size_small_value():
    assert readable_size(0.5) == "0.50"


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_evaluates_to_false():
    path = "non_existent_directory"
    result = list(scandir(path))
    assert not result


# LLM-generated content at query #6
#--------------------------

```python
def test_scandir_with_path_instance():
    from pathlib import Path
    path = Path("some_directory")
    result = list(scandir(path))
    assert all(isinstance(p, Path) for p in result)


# LLM-generated content at query #7
#--------------------------

```python
def test_scandir_with_path_object():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        result = next(iter(scandir(path)))
        assert isinstance(result, Path)


# LLM-generated content at query #8
#--------------------------

```python
def test_scandir_path_type():
    path = Path('/some/directory')
    result = scandir(path)
    assert isinstance(next(result), Path)


# LLM-generated content at query #9
#--------------------------

```python
def test_cache_with_path_and_verbose():
    import os
    import tempfile
    import pickle

    def dummy_func():
        return "dummy_data"

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = os.path.join(temp_dir, "cache.pkl")
        cached_func = cache(cache_path, verbose=True, name="test_cache")(dummy_func)
        result = cached_func()
        assert result == "dummy_data"
        assert os.path.exists(cache_path)

def test_cache_with_path_and_verbose_existing_cache():
    import os
    import tempfile
    import pickle

    def dummy_func():
        return "dummy_data"

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = os.path.join(temp_dir, "cache.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump("cached_data", f)
        cached_func = cache(cache_path, verbose=True, name="test_cache")(dummy_func)
        result = cached_func()
        assert result == "cached_data"

def test_cache_with_path_and_not_verbose():
    import os
    import tempfile
    import pickle

    def dummy_func():
        return "dummy_data"

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = os.path.join(temp_dir, "cache.pkl")
        cached_func = cache(cache_path, verbose=False, name="test_cache")(dummy_func)
        result = cached_func()
        assert result == "dummy_data"
        assert os.path.exists(cache_path)

def test_cache_with_path_and_not_verbose_existing_cache():
    import os
    import tempfile
    import pickle

    def dummy_func():
        return "dummy_data"

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = os.path.join(temp_dir, "cache.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump("cached_data", f)
        cached_func = cache(cache_path, verbose=False, name="test_cache")(dummy_func)
        result = cached_func()
        assert result == "cached_data"

def test_cache_without_path():
    import os
    import tempfile

    def dummy_func():
        return "dummy_data"

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = os.path.join(temp_dir, "cache.pkl")
        cached_func = cache(None, verbose=True, name="test_cache")(dummy_func)
        result = cached_func()
        assert result == "dummy_data"
        assert not os.path.exists(cache_path)


# LLM-generated content at query #10
#--------------------------

```python
def test_scandir_returns_path_string_when_input_is_string():
    from pathlib import Path
    import os
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    with open(os.path.join(test_dir, "test_file.txt"), "w") as f:
        f.write("test")
    result = list(scandir(test_dir))
    assert all(isinstance(path, str) for path in result)
    os.remove(os.path.join(test_dir, "test_file.txt"))
    os.rmdir(test_dir)


# LLM-generated content at query #11
#--------------------------

```python
def test_cache_with_existing_file():
    import tempfile
    import pickle
    import os

    with tempfile.NamedTemporaryFile(delete=False) as f:
        pickle.dump("cached_value", f)
        path = f.name

    @cache(path)
    def dummy_func():
        return "new_value"

    result = dummy_func()
    assert result == "cached_value"
    os.remove(path)

def test_cache_with_non_existing_file():
    import tempfile
    import pickle
    import os

    path = tempfile.NamedTemporaryFile(delete=False).name
    os.remove(path)

    @cache(path)
    def dummy_func():
        return "new_value"

    result = dummy_func()
    assert result == "new_value"
    assert os.path.exists(path)
    os.remove(path)

def test_cache_with_no_path():
    @cache(None)
    def dummy_func():
        return "new_value"

    result = dummy_func()
    assert result == "new_value"

def test_cache_with_verbose_false():
    import tempfile
    import pickle
    import os

    path = tempfile.NamedTemporaryFile(delete=False).name
    os.remove(path)

    @cache(path, verbose=False)
    def dummy_func():
        return "new_value"

    result = dummy_func()
    assert result == "new_value"
    assert os.path.exists(path)
    os.remove(path)


# LLM-generated content at query #12
#--------------------------

def test_cache_with_path_and_verbose():
    import tempfile
    import os
    test_file = tempfile.NamedTemporaryFile(delete=False).name
    try:
        @cache(test_file, verbose=True, name="test")
        def test_func():
            return {"key": "value"}
        result = test_func()
        assert result == {"key": "value"}
        assert os.path.exists(test_file)
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)

def test_cache_with_path_no_verbose():
    import tempfile
    import os
    test_file = tempfile.NamedTemporaryFile(delete=False).name
    try:
        @cache(test_file, verbose=False, name="test")
        def test_func():
            return {"key": "value"}
        result = test_func()
        assert result == {"key": "value"}
        assert os.path.exists(test_file)
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)

def test_cache_without_path():
    @cache(None, verbose=True, name="test")
    def test_func():
        return {"key": "value"}
    result = test_func()
    assert result == {"key": "value"}

def test_cache_load_existing():
    import tempfile
    import os
    test_file = tempfile.NamedTemporaryFile(delete=False).name
    try:
        with open(test_file, "wb") as f:
            import pickle
            pickle.dump({"key": "cached_value"}, f)
        @cache(test_file, verbose=True, name="test")
        def test_func():
            return {"key": "value"}
        result = test_func()
        assert result == {"key": "cached_value"}
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


# LLM-generated content at query #13
#--------------------------

```
def test_scandir_with_str_path():
    import os
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        
        result = list(scandir(tmpdir))
        assert len(result) == 1
        assert isinstance(result[0], str)
        assert result[0] == test_file

def test_scandir_with_pathlib_path():
    import os
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / 'test.txt'
        with open(test_file, 'w') as f:
            f.write('test')
        
        result = list(scandir(Path(tmpdir)))
        assert len(result) == 1
        assert isinstance(result[0], Path)
        assert result[0] == test_file


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_13_evaluates_to_false():
    path = "non_existent_directory"
    result = next(scandir(path), None)
    assert result is None


# LLM-generated content at query #15
#--------------------------

```python
def test_scandir_with_path_object():
    path = Path("/some/directory")
    result = list(scandir(path))
    assert all(isinstance(p, Path) for p in result)

def test_scandir_with_string_path():
    path = "/some/directory"
    result = list(scandir(path))
    assert all(isinstance(p, str) for p in result)


# LLM-generated content at query #16
#--------------------------

```python
def test_cache_path_is_none():
    @cache(path=None)
    def dummy_func():
        return 42
    assert dummy_func() == 42


# LLM-generated content at query #17
#--------------------------

```python
def test_scandir_with_str_path():
    path = "/tmp"
    result = list(scandir(path))
    assert all(isinstance(item, str) for item in result)

def test_scandir_with_pathlib_path():
    from pathlib import Path
    path = Path("/tmp")
    result = list(scandir(path))
    assert all(isinstance(item, Path) for item in result)


# LLM-generated content at query #18
#--------------------------

```python
def test_scandir_with_non_path_type():
    path = "non_path_type_value"
    result = list(scandir(path))
    assert isinstance(result[0], str)


# LLM-generated content at query #19
#--------------------------

```python
def test_scandir_with_pathlib_path():
    from pathlib import Path
    directory = Path(__file__).parent
    result = list(scandir(directory))
    assert all(isinstance(item, Path) for item in result)

def test_scandir_with_str_path():
    directory = __file__[:-len(Path(__file__).name)]
    result = list(scandir(directory))
    assert all(isinstance(item, str) for item in result)

def test_scandir_empty_directory(tmpdir):
    result = list(scandir(tmpdir))
    assert len(result) == 0

def test_scandir_non_empty_directory(tmpdir):
    tmpdir.join("file1.txt").write("")
    tmpdir.join("file2.txt").write("")
    result = list(scandir(tmpdir))
    assert len(result) == 2


# LLM-generated content at query #20
#--------------------------

```python
def test_copy_tree_overwrite_false():
    src = "/path/to/src"
    dst = "/path/to/dst"
    file = "existing_file.txt"
    src_path = os.path.join(src, file)
    dst_path = os.path.join(dst, file)
    
    os.listdir = lambda x: [file]
    os.path.isdir = lambda x: False
    os.path.exists = lambda x: True
    overwrite = False
    
    assert not (overwrite or not os.path.exists(dst_path))


# LLM-generated content at query #21
#--------------------------

```python
def test_cache_with_path_and_verbose_true():
    path = "test_cache.pkl"
    data = {"key": "value"}
    with open(path, "wb") as f:
        pickle.dump(data, f)
    
    @cache(path, verbose=True, name="test")
    def test_func():
        return {"key": "new_value"}
    
    result = test_func()
    assert result == data
    os.remove(path)

def test_cache_with_path_and_verbose_false():
    path = "test_cache.pkl"
    data = {"key": "value"}
    with open(path, "wb") as f:
        pickle.dump(data, f)
    
    @cache(path, verbose=False, name="test")
    def test_func():
        return {"key": "new_value"}
    
    result = test_func()
    assert result == data
    os.remove(path)

def test_cache_without_path():
    @cache(None, verbose=True, name="test")
    def test_func():
        return {"key": "value"}
    
    result = test_func()
    assert result == {"key": "value"}

def test_cache_with_non_existing_path():
    path = "non_existing_cache.pkl"
    
    @cache(path, verbose=True, name="test")
    def test_func():
        return {"key": "value"}
    
    result = test_func()
    assert result == {"key": "value"}
    assert os.path.exists(path)
    os.remove(path)


# LLM-generated content at query #22
#--------------------------

```python
def test_cache_with_existing_file():
    import os
    import tempfile
    import pickle
    from flutes.log import log

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        test_data = "test_data"
        pickle.dump(test_data, temp_file)
        temp_path = temp_file.name

    @cache(temp_path, verbose=True, name="test")
    def dummy_func():
        return "new_data"

    result = dummy_func()
    os.remove(temp_path)
    assert result == test_data


def test_cache_with_non_existing_file():
    import tempfile
    from flutes.log import log

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        os.remove(temp_path)

    @cache(temp_path, verbose=True, name="test")
    def dummy_func():
        return "new_data"

    result = dummy_func()
    assert result == "new_data"
    os.remove(temp_path)


def test_cache_with_no_path():
    from flutes.log import log

    @cache(None, verbose=True, name="test")
    def dummy_func():
        return "new_data"

    result = dummy_func()
    assert result == "new_data"


def test_cache_with_verbose_false():
    import tempfile
    from flutes.log import log

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        os.remove(temp_path)

    @cache(temp_path, verbose=False, name="test")
    def dummy_func():
        return "new_data"

    result = dummy_func()
    assert result == "new_data"
    os.remove(temp_path)


def test_cache_with_name_none():
    import tempfile
    from flutes.log import log

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        os.remove(temp_path)

    @cache(temp_path, verbose=True, name=None)
    def dummy_func():
        return "new_data"

    result = dummy_func()
    assert result == "new_data"
    os.remove(temp_path)


def test_cache_with_existing_file_and_no_verbose():
    import os
    import tempfile
    import pickle
    from flutes.log import log

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        test_data = "test_data"
        pickle.dump(test_data, temp_file)
        temp_path = temp_file.name

    @cache(temp_path, verbose=False, name="test")
    def dummy_func():
        return "new_data"

    result = dummy_func()
    os.remove(temp_path)
    assert result == test_data


# LLM-generated content at query #23
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


# LLM-generated content at query #24
#--------------------------

```
def test_copy_tree_predicate_evaluates_false():
    src = "/path/to/src"
    dst = "/path/to/dst"
    file = "test_file.txt"
    overwrite = False
    os.listdir = lambda x: [file]
    os.path.isdir = lambda x: False
    os.path.exists = lambda x: True
    shutil.copy2 = lambda x, y: None
    copy_tree(src, dst, overwrite)


# LLM-generated content at query #25
#--------------------------

```python
def test_cache_predicate_evaluates_to_true():
    import os
    import tempfile
    import pickle

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name
        pickle.dump("test_data", tmp_file)

    # Define a simple function to decorate
    def test_func():
        return "new_data"

    # Apply the cache decorator with the temporary file path
    decorated_func = cache(tmp_path)(test_func)

    # Call the decorated function - should load from cache
    result = decorated_func()

    # Clean up
    os.unlink(tmp_path)

    # Verify the predicate evaluated to True (loaded from cache)
    assert result == "test_data"


# LLM-generated content at query #26
#--------------------------

```
def test_predicate_at_line_11_evaluates_to_true():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("test content")
    copy_tree(src, dst, overwrite=True)
    assert os.path.exists(os.path.join(dst, "test_file.txt"))


# LLM-generated content at query #27
#--------------------------

```python
def test_copy_tree_with_overwrite_true():
    src = "/tmp/test_src"
    dst = "/tmp/test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("test content")
    copy_tree(src, dst, overwrite=True)
    assert os.path.exists(os.path.join(dst, "test_file.txt"))


# LLM-generated content at query #28
#--------------------------

```python
def test_copy_tree_overwrite_false():
    src = "/test/src"
    dst = "/test/dst"
    file = "test_file.txt"
    os.listdir = lambda x: [file]
    os.path.isdir = lambda x: False
    os.path.exists = lambda x: x == os.path.join(dst, file)
    shutil.copy2 = lambda x, y: None
    copy_tree(src, dst, overwrite=False)

def test_copy_tree_overwrite_true():
    src = "/test/src"
    dst = "/test/dst"
    file = "test_file.txt"
    os.listdir = lambda x: [file]
    os.path.isdir = lambda x: False
    os.path.exists = lambda x: x == os.path.join(dst, file)
    shutil.copy2 = lambda x, y: None
    copy_tree(src, dst, overwrite=True)

def test_copy_tree_directory():
    src = "/test/src"
    dst = "/test/dst"
    file = "test_dir"
    os.listdir = lambda x: [file]
    os.path.isdir = lambda x: True
    copy_tree = lambda x, y, overwrite: None
    copy_tree(src, dst, overwrite=False)

def test_copy_tree_new_file():
    src = "/test/src"
    dst = "/test/dst"
    file = "test_file.txt"
    os.listdir = lambda x: [file]
    os.path.isdir = lambda x: False
    os.path.exists = lambda x: False
    shutil.copy2 = lambda x, y: None
    copy_tree(src, dst, overwrite=False)


# LLM-generated content at query #29
#--------------------------

```python
def test_copy_tree_new_directory():
    import tempfile
    import shutil
    import os

    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    os.rmdir(dst)
    
    os.makedirs(os.path.join(src, "subdir"))
    with open(os.path.join(src, "file1.txt"), "w") as f:
        f.write("test")
    with open(os.path.join(src, "subdir", "file2.txt"), "w") as f:
        f.write("test")

    copy_tree(src, dst)
    
    assert os.path.exists(dst)
    assert os.path.exists(os.path.join(dst, "file1.txt"))
    assert os.path.exists(os.path.join(dst, "subdir", "file2.txt"))

    shutil.rmtree(src)
    shutil.rmtree(dst)

def test_copy_tree_overwrite_existing_files():
    import tempfile
    import shutil
    import os

    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    
    with open(os.path.join(src, "file1.txt"), "w") as f:
        f.write("new_content")
    with open(os.path.join(dst, "file1.txt"), "w") as f:
        f.write("old_content")

    copy_tree(src, dst, overwrite=True)
    
    with open(os.path.join(dst, "file1.txt"), "r") as f:
        assert f.read() == "new_content"

    shutil.rmtree(src)
    shutil.rmtree(dst)

def test_copy_tree_do_not_overwrite_existing_files():
    import tempfile
    import shutil
    import os

    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    
    with open(os.path.join(src, "file1.txt"), "w") as f:
        f.write("new_content")
    with open(os.path.join(dst, "file1.txt"), "w") as f:
        f.write("old_content")

    copy_tree(src, dst, overwrite=False)
    
    with open(os.path.join(dst, "file1.txt"), "r") as f:
        assert f.read() == "old_content"

    shutil.rmtree(src)
    shutil.rmtree(dst)

def test_copy_tree_empty_source_directory():
    import tempfile
    import shutil
    import os

    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    
    copy_tree(src, dst)
    
    assert os.path.exists(dst)
    assert len(os.listdir(dst)) == 0

    shutil.rmtree(src)
    shutil.rmtree(dst)

def test_copy_tree_nested_directories():
    import tempfile
    import shutil
    import os

    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    
    os.makedirs(os.path.join(src, "subdir1", "subdir2"))
    with open(os.path.join(src, "subdir1", "subdir2", "file.txt"), "w") as f:
        f.write("test")

    copy_tree(src, dst)
    
    assert os.path.exists(os.path.join(dst, "subdir1", "subdir2", "file.txt"))

    shutil.rmtree(src)
    shutil.rmtree(dst)


# LLM-generated content at query #30
#--------------------------

```python
def test_cache_path_none():
    @cache(path=None)
    def dummy_func():
        return 42
    assert dummy_func() == 42


# LLM-generated content at query #31
#--------------------------

```python
def test_cache_path_exists():
    import os
    import tempfile
    import pickle

    def dummy_func():
        return 42

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        pickle.dump(123, temp_file)
        temp_path = temp_file.name

    @cache(temp_path)
    def decorated_func():
        return dummy_func()

    assert decorated_func() == 123
    os.remove(temp_path)


# LLM-generated content at query #32
#--------------------------

```python
def test_cache_path_is_none():
    @cache(path=None)
    def dummy_func():
        return 42
    assert dummy_func() == 42


# LLM-generated content at query #33
#--------------------------

```python
def test_cache_path_exists():
    path = "test_cache.pkl"
    with open(path, "wb") as f:
        pickle.dump("cached_data", f)
    
    @cache(path)
    def dummy_func():
        return "new_data"
    
    result = dummy_func()
    assert result == "cached_data"
    os.remove(path)


# LLM-generated content at query #34
#--------------------------

```python
def test_copy_tree_with_new_destination():
    import os
    import shutil
    import tempfile

    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()

    os.makedirs(os.path.join(src_dir, "subdir"))
    with open(os.path.join(src_dir, "file1.txt"), "w") as f:
        f.write("content1")
    with open(os.path.join(src_dir, "subdir", "file2.txt"), "w") as f:
        f.write("content2")

    copy_tree(src_dir, dst_dir)

    assert os.path.exists(os.path.join(dst_dir, "file1.txt"))
    assert os.path.exists(os.path.join(dst_dir, "subdir", "file2.txt"))
    assert os.path.isdir(os.path.join(dst_dir, "subdir"))

    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)

def test_copy_tree_with_existing_destination_and_overwrite():
    import os
    import shutil
    import tempfile

    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()

    with open(os.path.join(src_dir, "file1.txt"), "w") as f:
        f.write("new_content")
    with open(os.path.join(dst_dir, "file1.txt"), "w") as f:
        f.write("old_content")

    copy_tree(src_dir, dst_dir, overwrite=True)

    with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
        content = f.read()
    assert content == "new_content"

    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)

def test_copy_tree_with_existing_destination_and_no_overwrite():
    import os
    import shutil
    import tempfile

    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()

    with open(os.path.join(src_dir, "file1.txt"), "w") as f:
        f.write("new_content")
    with open(os.path.join(dst_dir, "file1.txt"), "w") as f:
        f.write("old_content")

    copy_tree(src_dir, dst_dir, overwrite=False)

    with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
        content = f.read()
    assert content == "old_content"

    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)

def test_copy_tree_with_nested_directories():
    import os
    import shutil
    import tempfile

    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()

    os.makedirs(os.path.join(src_dir, "subdir1", "subdir2"))
    with open(os.path.join(src_dir, "subdir1", "subdir2", "file.txt"), "w") as f:
        f.write("content")

    copy_tree(src_dir, dst_dir)

    assert os.path.exists(os.path.join(dst_dir, "subdir1", "subdir2", "file.txt"))
    assert os.path.isdir(os.path.join(dst_dir, "subdir1"))
    assert os.path.isdir(os.path.join(dst_dir, "subdir1", "subdir2"))

    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)

def test_copy_tree_with_empty_source():
    import os
    import shutil
    import tempfile

    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()

    copy_tree(src_dir, dst_dir)

    assert os.listdir(dst_dir) == []

    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)


# LLM-generated content at query #35
#--------------------------

```
def test_scandir_with_str_path():
    path = "/tmp"
    result = list(scandir(path))
    assert all(isinstance(p, str) for p in result)

def test_scandir_with_pathlib_path():
    from pathlib import Path
    path = Path("/tmp")
    result = list(scandir(path))
    assert all(isinstance(p, Path) for p in result)


# LLM-generated content at query #36
#--------------------------

```python
def test_cache_decorator_with_existing_file():
    test_path = "test_cache.pkl"
    test_data = {"key": "value"}
    with open(test_path, "wb") as f:
        pickle.dump(test_data, f)
    
    @cache(test_path)
    def dummy_func():
        return {"different": "data"}
    
    result = dummy_func()
    assert result == test_data
    os.remove(test_path)

def test_cache_decorator_with_nonexistent_file():
    test_path = "test_cache.pkl"
    
    @cache(test_path)
    def dummy_func():
        return {"key": "value"}
    
    result = dummy_func()
    assert result == {"key": "value"}
    assert os.path.exists(test_path)
    os.remove(test_path)

def test_cache_decorator_with_none_path():
    @cache(None)
    def dummy_func():
        return {"key": "value"}
    
    result = dummy_func()
    assert result == {"key": "value"}
    assert not os.path.exists("None")

def test_cache_decorator_with_verbose_false():
    test_path = "test_cache.pkl"
    
    @cache(test_path, verbose=False)
    def dummy_func():
        return {"key": "value"}
    
    result = dummy_func()
    assert result == {"key": "value"}
    os.remove(test_path)

def test_cache_decorator_with_custom_name():
    test_path = "test_cache.pkl"
    
    @cache(test_path, name="custom")
    def dummy_func():
        return {"key": "value"}
    
    dummy_func()
    os.remove(test_path)


# LLM-generated content at query #37
#--------------------------

```python
def test_predicate_at_line_13_evaluates_to_false():
    path = Path("/some/directory")
    result = next(scandir(path))
    assert not isinstance(result, str)


# LLM-generated content at query #38
#--------------------------

```python
def test_scandir_with_pathlib_path():
    import pathlib
    path = pathlib.Path(".")
    result = list(scandir(path))
    assert all(isinstance(item, pathlib.Path) for item in result)

def test_scandir_with_str_path():
    path = "."
    result = list(scandir(path))
    assert all(isinstance(item, str) for item in result)

def test_scandir_with_non_existent_path():
    import pathlib
    path = pathlib.Path("non_existent_directory")
    try:
        list(scandir(path))
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass

def test_scandir_with_empty_directory(tmpdir):
    import pathlib
    path = pathlib.Path(tmpdir)
    result = list(scandir(path))
    assert len(result) == 0


# LLM-generated content at query #39
#--------------------------

```python
def test_overwrite_true():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("source content")
    with open(os.path.join(dst, "test_file.txt"), "w") as f:
        f.write("destination content")
    copy_tree(src, dst, overwrite=True)
    with open(os.path.join(dst, "test_file.txt"), "r") as f:
        content = f.read()
    assert content == "source content"


# LLM-generated content at query #40
#--------------------------

```python
def test_copy_tree_overwrite_false():
    src = "existing_src_dir"
    dst = "existing_dst_dir"
    overwrite = False
    assert not (overwrite or not os.path.exists(os.path.join(dst, "existing_file")))


# LLM-generated content at query #41
#--------------------------

def test_cache_with_path_and_verbose():
    test_path = "test_cache.pkl"
    test_data = {"key": "value"}
    with open(test_path, "wb") as f:
        pickle.dump(test_data, f)
    
    @cache(test_path, verbose=True, name="test")
    def test_func():
        return {"key": "different_value"}
    
    result = test_func()
    assert result == test_data
    os.remove(test_path)

def test_cache_with_path_no_verbose():
    test_path = "test_cache.pkl"
    test_data = {"key": "value"}
    with open(test_path, "wb") as f:
        pickle.dump(test_data, f)
    
    @cache(test_path, verbose=False, name="test")
    def test_func():
        return {"key": "different_value"}
    
    result = test_func()
    assert result == test_data
    os.remove(test_path)

def test_cache_without_path():
    @cache(None, verbose=True, name="test")
    def test_func():
        return {"key": "value"}
    
    result = test_func()
    assert result == {"key": "value"}

def test_cache_save_new_file():
    test_path = "test_cache.pkl"
    if os.path.exists(test_path):
        os.remove(test_path)
    
    @cache(test_path, verbose=True, name="test")
    def test_func():
        return {"key": "value"}
    
    result = test_func()
    assert result == {"key": "value"}
    assert os.path.exists(test_path)
    with open(test_path, "rb") as f:
        loaded_data = pickle.load(f)
    assert loaded_data == {"key": "value"}
    os.remove(test_path)


# LLM-generated content at query #42
#--------------------------

def test_cache_decorator_with_existing_file():
    import tempfile
    import os
    import pickle

    test_data = {"key": "value"}
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        path = tmp_file.name
        pickle.dump(test_data, tmp_file)

    @cache(path=path, verbose=False)
    def dummy_func():
        return {"should": "not_be_called"}

    result = dummy_func()
    os.unlink(path)
    assert result == test_data


# LLM-generated content at query #43
#--------------------------

```python
def test_scandir_returns_string_path_when_input_is_string():
    path = "some_directory"
    result = list(scandir(path))
    assert all(isinstance(p, str) for p in result)

def test_scandir_returns_pathlib_path_when_input_is_pathlib_path():
    from pathlib import Path
    path = Path("some_directory")
    result = list(scandir(path))
    assert all(isinstance(p, Path) for p in result)


# LLM-generated content at query #44
#--------------------------

```python
def test_copy_tree_with_overwrite():
    import os
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as src_dir:
        with tempfile.TemporaryDirectory() as dst_dir:
            with open(os.path.join(src_dir, 'file1.txt'), 'w') as f:
                f.write('content1')
            with open(os.path.join(dst_dir, 'file1.txt'), 'w') as f:
                f.write('old_content1')
            copy_tree(src_dir, dst_dir, overwrite=True)
            with open(os.path.join(dst_dir, 'file1.txt'), 'r') as f:
                assert f.read() == 'content1'

def test_copy_tree_without_overwrite():
    import os
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as src_dir:
        with tempfile.TemporaryDirectory() as dst_dir:
            with open(os.path.join(src_dir, 'file1.txt'), 'w') as f:
                f.write('content1')
            with open(os.path.join(dst_dir, 'file1.txt'), 'w') as f:
                f.write('old_content1')
            copy_tree(src_dir, dst_dir, overwrite=False)
            with open(os.path.join(dst_dir, 'file1.txt'), 'r') as f:
                assert f.read() == 'old_content1'

def test_copy_tree_create_dst_directory():
    import os
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as src_dir:
        dst_dir = os.path.join(tempfile.mkdtemp(), 'new_dir')
        with open(os.path.join(src_dir, 'file1.txt'), 'w') as f:
            f.write('content1')
        copy_tree(src_dir, dst_dir)
        assert os.path.exists(dst_dir)
        assert os.path.isfile(os.path.join(dst_dir, 'file1.txt'))

def test_copy_tree_with_subdirectories():
    import os
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as src_dir:
        with tempfile.TemporaryDirectory() as dst_dir:
            os.makedirs(os.path.join(src_dir, 'subdir'))
            with open(os.path.join(src_dir, 'subdir', 'file1.txt'), 'w') as f:
                f.write('content1')
            copy_tree(src_dir, dst_dir)
            assert os.path.isdir(os.path.join(dst_dir, 'subdir'))
            assert os.path.isfile(os.path.join(dst_dir, 'subdir', 'file1.txt'))

def test_copy_tree_copystat():
    import os
    import shutil
    import tempfile
    import time

    with tempfile.TemporaryDirectory() as src_dir:
        with tempfile.TemporaryDirectory() as dst_dir:
            with open(os.path.join(src_dir, 'file1.txt'), 'w') as f:
                f.write('content1')
            time.sleep(1)  # Ensure different timestamps
            copy_tree(src_dir, dst_dir)
            src_stat = os.stat(src_dir)
            dst_stat = os.stat(dst_dir)
            assert src_stat.st_mode == dst_stat.st_mode
            assert src_stat.st_atime == dst_stat.st_atime
            assert src_stat.st_mtime == dst_stat.st_mtime


# LLM-generated content at query #45
#--------------------------

```python
def test_scandir_with_path_instance():
    path = Path("/example/path")
    result = list(scandir(path))
    assert all(isinstance(item, Path) for item in result)


# LLM-generated content at query #46
#--------------------------

def test_scandir_with_pathlib_path():
    path = Path("/tmp")
    result = list(scandir(path))
    assert all(isinstance(p, Path) for p in result)

def test_scandir_with_string_path():
    path = "/tmp"
    result = list(scandir(path))
    assert all(isinstance(p, str) for p in result)

def test_scandir_with_non_existent_path():
    path = "/non/existent/path"
    result = list(scandir(path))
    assert result == []


# LLM-generated content at query #47
#--------------------------

```python
def test_copy_tree_creates_destination_directory():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = os.path.join(tempfile.mkdtemp(), "new_dir")
    try:
        copy_tree(src, dst)
        assert os.path.exists(dst)
    finally:
        shutil.rmtree(src)
        if os.path.exists(dst):
            shutil.rmtree(dst)

def test_copy_tree_copies_files():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        with open(os.path.join(src, "test.txt"), "w") as f:
            f.write("test")
        copy_tree(src, dst)
        assert os.path.exists(os.path.join(dst, "test.txt"))
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_copies_subdirectories():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        os.mkdir(os.path.join(src, "subdir"))
        with open(os.path.join(src, "subdir", "test.txt"), "w") as f:
            f.write("test")
        copy_tree(src, dst)
        assert os.path.exists(os.path.join(dst, "subdir", "test.txt"))
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_does_not_overwrite_by_default():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        with open(os.path.join(src, "test.txt"), "w") as f:
            f.write("src")
        with open(os.path.join(dst, "test.txt"), "w") as f:
            f.write("dst")
        copy_tree(src, dst)
        with open(os.path.join(dst, "test.txt"), "r") as f:
            assert f.read() == "dst"
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_overwrites_when_requested():
    import tempfile
    import shutil
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        with open(os.path.join(src, "test.txt"), "w") as f:
            f.write("src")
        with open(os.path.join(dst, "test.txt"), "w") as f:
            f.write("dst")
        copy_tree(src, dst, overwrite=True)
        with open(os.path.join(dst, "test.txt"), "r") as f:
            assert f.read() == "src"
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)

def test_copy_tree_copies_file_attributes():
    import tempfile
    import shutil
    import time
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        test_file = os.path.join(src, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        expected_stat = os.stat(test_file)
        copy_tree(src, dst)
        dst_file = os.path.join(dst, "test.txt")
        actual_stat = os.stat(dst_file)
        assert abs(expected_stat.st_mtime - actual_stat.st_mtime) < 1
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)


# LLM-generated content at query #48
#--------------------------

```python
def test_copy_tree_creates_destination_directory():
    import shutil
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as src_dir:
        with tempfile.TemporaryDirectory() as dst_dir:
            shutil.rmtree(dst_dir)
            copy_tree(src_dir, dst_dir)
            assert os.path.exists(dst_dir)

def test_copy_tree_copies_files():
    import shutil
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as src_dir:
        with tempfile.TemporaryDirectory() as dst_dir:
            file_path = os.path.join(src_dir, "test.txt")
            with open(file_path, "w") as f:
                f.write("test")
            copy_tree(src_dir, dst_dir)
            assert os.path.exists(os.path.join(dst_dir, "test.txt"))
            with open(os.path.join(dst_dir, "test.txt"), "r") as f:
                assert f.read() == "test"

def test_copy_tree_overwrites_files_when_overwrite_is_true():
    import shutil
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as src_dir:
        with tempfile.TemporaryDirectory() as dst_dir:
            src_file_path = os.path.join(src_dir, "test.txt")
            dst_file_path = os.path.join(dst_dir, "test.txt")
            with open(src_file_path, "w") as f:
                f.write("new content")
            with open(dst_file_path, "w") as f:
                f.write("old content")
            copy_tree(src_dir, dst_dir, overwrite=True)
            with open(dst_file_path, "r") as f:
                assert f.read() == "new content"

def test_copy_tree_does_not_overwrite_files_when_overwrite_is_false():
    import shutil
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as src_dir:
        with tempfile.TemporaryDirectory() as dst_dir:
            src_file_path = os.path.join(src_dir, "test.txt")
            dst_file_path = os.path.join(dst_dir, "test.txt")
            with open(src_file_path, "w") as f:
                f.write("new content")
            with open(dst_file_path, "w") as f:
                f.write("old content")
            copy_tree(src_dir, dst_dir, overwrite=False)
            with open(dst_file_path, "r") as f:
                assert f.read() == "old content"

def test_copy_tree_copies_subdirectories():
    import shutil
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as src_dir:
        with tempfile.TemporaryDirectory() as dst_dir:
            subdir_path = os.path.join(src_dir, "subdir")
            os.makedirs(subdir_path)
            file_path = os.path.join(subdir_path, "test.txt")
            with open(file_path, "w") as f:
                f.write("test")
            copy_tree(src_dir, dst_dir)
            assert os.path.exists(os.path.join(dst_dir, "subdir"))
            assert os.path.exists(os.path.join(dst_dir, "subdir", "test.txt"))
            with open(os.path.join(dst_dir, "subdir", "test.txt"), "r") as f:
                assert f.read() == "test"

def test_copy_tree_copies_file_permissions():
    import shutil
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as src_dir:
        with tempfile.TemporaryDirectory() as dst_dir:
            file_path = os.path.join(src_dir, "test.txt")
            with open(file_path, "w") as f:
                f.write("test")
            os.chmod(file_path, 0o644)
            copy_tree(src_dir, dst_dir)
            assert os.stat(os.path.join(dst_dir, "test.txt")).st_mode == os.stat(file_path).st_mode


# LLM-generated content at query #49
#--------------------------

```
def test_copy_tree_creates_destination_directory():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    copy_tree(src, dst)
    assert os.path.exists(dst)
    os.rmdir(dst)
    os.rmdir(src)

def test_copy_tree_copies_files():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("test")
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "test_file.txt"))
    os.remove(os.path.join(dst, "test_file.txt"))
    os.rmdir(dst)
    os.remove(os.path.join(src, "test_file.txt"))
    os.rmdir(src)

def test_copy_tree_overwrites_files():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("new_content")
    with open(os.path.join(dst, "test_file.txt"), "w") as f:
        f.write("old_content")
    copy_tree(src, dst, overwrite=True)
    with open(os.path.join(dst, "test_file.txt"), "r") as f:
        assert f.read() == "new_content"
    os.remove(os.path.join(dst, "test_file.txt"))
    os.rmdir(dst)
    os.remove(os.path.join(src, "test_file.txt"))
    os.rmdir(src)

def test_copy_tree_does_not_overwrite_files_by_default():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "test_file.txt"), "w") as f:
        f.write("new_content")
    with open(os.path.join(dst, "test_file.txt"), "w") as f:
        f.write("old_content")
    copy_tree(src, dst)
    with open(os.path.join(dst, "test_file.txt"), "r") as f:
        assert f.read() == "old_content"
    os.remove(os.path.join(dst, "test_file.txt"))
    os.rmdir(dst)
    os.remove(os.path.join(src, "test_file.txt"))
    os.rmdir(src)

def test_copy_tree_copies_subdirectories():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(os.path.join(src, "subdir"), exist_ok=True)
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "subdir"))
    os.rmdir(os.path.join(dst, "subdir"))
    os.rmdir(dst)
    os.rmdir(os.path.join(src, "subdir"))
    os.rmdir(src)


# LLM-generated content at query #50
#--------------------------

```python
def test_overwrite_true_should_copy_file():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
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

def test_overwrite_false_should_not_copy_file():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    src_file = os.path.join(src, "file.txt")
    dst_file = os.path.join(dst, "file.txt")
    with open(src_file, "w") as f:
        f.write("source content")
    with open(dst_file, "w") as f:
        f.write("destination content")
    copy_tree(src, dst, overwrite=False)
    with open(dst_file, "r") as f:
        content = f.read()
    assert content == "destination content"

def test_overwrite_true_should_copy_directory():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    src_dir = os.path.join(src, "dir")
    dst_dir = os.path.join(dst, "dir")
    os.makedirs(src_dir, exist_ok=True)
    src_file = os.path.join(src_dir, "file.txt")
    with open(src_file, "w") as f:
        f.write("source content")
    copy_tree(src, dst, overwrite=True)
    dst_file = os.path.join(dst_dir, "file.txt")
    assert os.path.exists(dst_file)

def test_overwrite_false_should_copy_new_file():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    src_file = os.path.join(src, "file.txt")
    with open(src_file, "w") as f:
        f.write("source content")
    copy_tree(src, dst, overwrite=False)
    dst_file = os.path.join(dst, "file.txt")
    assert os.path.exists(dst_file)


# LLM-generated content at query #51
#--------------------------

```python
def test_copy_tree_overwrite_true():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    overwrite = True
    src_file = os.path.join(src, "file.txt")
    dst_file = os.path.join(dst, "file.txt")
    with open(src_file, "w") as f:
        f.write("src content")
    with open(dst_file, "w") as f:
        f.write("dst content")
    copy_tree(src, dst, overwrite=overwrite)
    with open(dst_file, "r") as f:
        content = f.read()
    assert content == "src content"
    os.remove(src_file)
    os.remove(dst_file)
    os.rmdir(src)
    os.rmdir(dst)


# LLM-generated content at query #52
#--------------------------

```python
def test_copy_tree_overwrite_false():
    src = "/path/to/src"
    dst = "/path/to/dst"
    overwrite = False
    os.listdir = lambda x: ["file1.txt"]
    os.path.isdir = lambda x: False
    os.path.exists = lambda x: True
    shutil.copy2 = lambda x, y: None
    copy_tree(src, dst, overwrite)


# LLM-generated content at query #53
#--------------------------

```python
def test_copy_tree_new_directory():
    import tempfile
    import shutil
    import os
    from pathlib import Path
    
    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()
    
    try:
        (Path(src_dir) / 'file1.txt').write_text('test1')
        (Path(src_dir) / 'subdir').mkdir()
        (Path(src_dir) / 'subdir' / 'file2.txt').write_text('test2')
        
        copy_tree(src_dir, dst_dir)
        
        assert os.path.exists(Path(dst_dir) / 'file1.txt')
        assert os.path.isdir(Path(dst_dir) / 'subdir')
        assert os.path.exists(Path(dst_dir) / 'subdir' / 'file2.txt')
    finally:
        shutil.rmtree(src_dir)
        shutil.rmtree(dst_dir)

def test_copy_tree_overwrite_existing():
    import tempfile
    import shutil
    import os
    from pathlib import Path
    
    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()
    
    try:
        (Path(src_dir) / 'file1.txt').write_text('new_content')
        (Path(dst_dir) / 'file1.txt').write_text('old_content')
        
        copy_tree(src_dir, dst_dir, overwrite=True)
        
        assert (Path(dst_dir) / 'file1.txt').read_text() == 'new_content'
    finally:
        shutil.rmtree(src_dir)
        shutil.rmtree(dst_dir)

def test_copy_tree_skip_existing():
    import tempfile
    import shutil
    import os
    from pathlib import Path
    
    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()
    
    try:
        (Path(src_dir) / 'file1.txt').write_text('new_content')
        (Path(dst_dir) / 'file1.txt').write_text('old_content')
        
        copy_tree(src_dir, dst_dir, overwrite=False)
        
        assert (Path(dst_dir) / 'file1.txt').read_text() == 'old_content'
    finally:
        shutil.rmtree(src_dir)
        shutil.rmtree(dst_dir)

def test_copy_tree_empty_directory():
    import tempfile
    import shutil
    import os
    from pathlib import Path
    
    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()
    
    try:
        copy_tree(src_dir, dst_dir)
        
        assert len(os.listdir(dst_dir)) == 0
    finally:
        shutil.rmtree(src_dir)
        shutil.rmtree(dst_dir)

def test_copy_tree_nested_directories():
    import tempfile
    import shutil
    import os
    from pathlib import Path
    
    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()
    
    try:
        (Path(src_dir) / 'level1').mkdir()
        (Path(src_dir) / 'level1' / 'level2').mkdir()
        (Path(src_dir) / 'level1' / 'level2' / 'file.txt').write_text('test')
        
        copy_tree(src_dir, dst_dir)
        
        assert os.path.isdir(Path(dst_dir) / 'level1')
        assert os.path.isdir(Path(dst_dir) / 'level1' / 'level2')
        assert os.path.exists(Path(dst_dir) / 'level1' / 'level2' / 'file.txt')
    finally:
        shutil.rmtree(src_dir)
        shutil.rmtree(dst_dir)


