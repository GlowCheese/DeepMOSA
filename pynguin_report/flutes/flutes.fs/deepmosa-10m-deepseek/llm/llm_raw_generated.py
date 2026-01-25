####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_readable_size_basic():
    assert readable_size(1023) == "1023.00"
    assert readable_size(1024) == "1.00K"
    assert readable_size(1024 * 1024) == "1.00M"
    assert readable_size(1024 * 1024 * 1024) == "1.00G"
    assert readable_size(1024 * 1024 * 1024 * 1024) == "1.00T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.00P"

def test_readable_size_custom_digits():
    assert readable_size(1023, n_digits=0) == "1023"
    assert readable_size(1024, n_digits=1) == "1.0K"
    assert readable_size(1024 * 1024, n_digits=3) == "1.000M"
    assert readable_size(1024 * 1024 * 1024, n_digits=4) == "1.0000G"

def test_readable_size_edge_cases():
    assert readable_size(0) == "0.00"
    assert readable_size(1) == "1.00"
    assert readable_size(1023.999) == "1023.99K"
    assert readable_size(1024 * 1024 - 1) == "1023.99K"

def test_readable_size_large_values():
    assert readable_size(1024 ** 6) == "1024.00P"
    assert readable_size(1024 ** 7) == "1048576.00P"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_readable_size_bytes():
    assert readable_size(0) == "0.00"
    assert readable_size(1) == "1.00"
    assert readable_size(1023) == "1023.00"
    assert readable_size(1023, 0) == "1023"

def test_readable_size_kilobytes():
    assert readable_size(1024) == "1.00K"
    assert readable_size(1025) == "1.00K"
    assert readable_size(1536) == "1.50K"
    assert readable_size(1024 * 1024 - 1) == "1024.00K"
    assert readable_size(1024 * 1024 - 1, 0) == "1024K"

def test_readable_size_megabytes():
    assert readable_size(1024 * 1024) == "1.00M"
    assert readable_size(1024 * 1024 * 1.5) == "1.50M"
    assert readable_size(1024 * 1024 * 1024 - 1) == "1024.00M"
    assert readable_size(1024 * 1024 * 1024 - 1, 0) == "1024M"

def test_readable_size_gigabytes():
    assert readable_size(1024 * 1024 * 1024) == "1.00G"
    assert readable_size(1024 * 1024 * 1024 * 2.5) == "2.50G"
    assert readable_size(1024 * 1024 * 1024 * 1024 - 1) == "1024.00G"
    assert readable_size(1024 * 1024 * 1024 * 1024 - 1, 0) == "1024G"

def test_readable_size_terabytes():
    assert readable_size(1024 * 1024 * 1024 * 1024) == "1.00T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 3.75) == "3.75T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 - 1) == "1024.00T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 - 1, 0) == "1024T"

def test_readable_size_petabytes():
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 10.125) == "10.12P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 100, 0) == "100P"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    assert readable_size(1024, 0) == "1K"

def test_readable_size_fractional():
    assert readable_size(1500) == "1.46K"

def test_readable_size_large_fractional():
    assert readable_size(1500 * 1024 * 1024) == "1.46G"

def test_readable_size_edge_case():
    assert readable_size(1023) == "1023.00"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_readable_size_zero_bytes():
    assert readable_size(0) == "0.00"

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
    assert readable_size(1024, 0) == "1K"

def test_readable_size_fractional():
    assert readable_size(1500) == "1.46K"

def test_readable_size_large_fractional():
    assert readable_size(1500 * 1024 * 1024) == "1.46G"


# LLM-generated content at query #2
#--------------------------

```python
def test_cache_decorator_with_valid_path():
    import os
    import pickle
    import tempfile

    def dummy_func():
        return 42

    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
        pickle.dump(123, f)

    cached_func = cache(path)(dummy_func)
    result = cached_func()
    os.remove(path)
    assert result == 123

def test_cache_decorator_with_nonexistent_path():
    import os
    import tempfile

    def dummy_func():
        return 42

    with tempfile.NamedTemporaryFile() as f:
        path = f.name

    cached_func = cache(path)(dummy_func)
    result = cached_func()
    assert result == 42
    assert os.path.exists(path)
    os.remove(path)

def test_cache_decorator_with_no_path():
    def dummy_func():
        return 42

    cached_func = cache(None)(dummy_func)
    result = cached_func()
    assert result == 42

def test_cache_decorator_with_verbose_false():
    import os
    import pickle
    import tempfile

    def dummy_func():
        return 42

    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
        pickle.dump(123, f)

    cached_func = cache(path, verbose=False)(dummy_func)
    result = cached_func()
    os.remove(path)
    assert result == 123

def test_cache_decorator_with_custom_name():
    import os
    import pickle
    import tempfile

    def dummy_func():
        return 42

    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
        pickle.dump(123, f)

    cached_func = cache(path, name="custom")(dummy_func)
    result = cached_func()
    os.remove(path)
    assert result == 123


# LLM-generated content at query #3
#--------------------------

```python
def test_cache_with_path_and_verbose():
    path = "test_cache.pkl"
    data = {"key": "value"}

    @cache(path, verbose=True, name="test")
    def test_func():
        return data

    ret = test_func()
    assert ret == data
    assert os.path.exists(path)

    ret_cached = test_func()
    assert ret_cached == data

    os.remove(path)

def test_cache_with_path_and_not_verbose():
    path = "test_cache.pkl"
    data = {"key": "value"}

    @cache(path, verbose=False, name="test")
    def test_func():
        return data

    ret = test_func()
    assert ret == data
    assert os.path.exists(path)

    ret_cached = test_func()
    assert ret_cached == data

    os.remove(path)

def test_cache_without_path():
    data = {"key": "value"}

    @cache(None, verbose=True, name="test")
    def test_func():
        return data

    ret = test_func()
    assert ret == data

def test_cache_with_nonexistent_path():
    path = "nonexistent_cache.pkl"
    data = {"key": "value"}

    @cache(path, verbose=True, name="test")
    def test_func():
        return data

    ret = test_func()
    assert ret == data
    assert os.path.exists(path)

    os.remove(path)


# LLM-generated content at query #4
#--------------------------

```python
import os
import pickle
import tempfile

def test_cache_with_existing_file():
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        test_data = {"key": "value"}
        pickle.dump(test_data, tmp_file)
        path = tmp_file.name

    @cache(path, verbose=False)
    def dummy_func():
        return {"key": "new_value"}

    result = dummy_func()
    assert result == test_data
    os.remove(path)

def test_cache_with_non_existing_file():
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        path = tmp_file.name
        os.remove(path)

    @cache(path, verbose=False)
    def dummy_func():
        return {"key": "value"}

    result = dummy_func()
    assert result == {"key": "value"}
    os.remove(path)

def test_cache_with_no_path():
    @cache(None, verbose=False)
    def dummy_func():
        return {"key": "value"}

    result = dummy_func()
    assert result == {"key": "value"}

def test_cache_with_verbose():
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        path = tmp_file.name
        os.remove(path)

    @cache(path, verbose=True, name="Test Cache")
    def dummy_func():
        return {"key": "value"}

    result = dummy_func()
    assert result == {"key": "value"}
    os.remove(path)


# LLM-generated content at query #5
#--------------------------

```python
def test_copy_tree_without_overwrite():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "file1.txt"), "w") as f:
        f.write("content1")
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "file1.txt"))
    assert open(os.path.join(dst, "file1.txt")).read() == "content1"

def test_copy_tree_with_overwrite():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "file1.txt"), "w") as f:
        f.write("content1")
    with open(os.path.join(dst, "file1.txt"), "w") as f:
        f.write("content2")
    copy_tree(src, dst, overwrite=True)
    assert os.path.exists(os.path.join(dst, "file1.txt"))
    assert open(os.path.join(dst, "file1.txt")).read() == "content1"

def test_copy_tree_without_overwrite_existing_file():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "file1.txt"), "w") as f:
        f.write("content1")
    with open(os.path.join(dst, "file1.txt"), "w") as f:
        f.write("content2")
    copy_tree(src, dst, overwrite=False)
    assert os.path.exists(os.path.join(dst, "file1.txt"))
    assert open(os.path.join(dst, "file1.txt")).read() == "content2"

def test_copy_tree_with_nested_directories():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(os.path.join(src, "subdir"), exist_ok=True)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "subdir", "file1.txt"), "w") as f:
        f.write("content1")
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "subdir", "file1.txt"))
    assert open(os.path.join(dst, "subdir", "file1.txt")).read() == "content1"

def test_copy_tree_with_non_existent_destination():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, "file1.txt"), "w") as f:
        f.write("content1")
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "file1.txt"))
    assert open(os.path.join(dst, "file1.txt")).read() == "content1"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_cache_with_path_and_verbose():
    import tempfile
    import os
    import pickle

    def test_func():
        return "test_value"

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        decorated = cache(tmp_path, verbose=True, name="test")(test_func)
        result = decorated()
        assert result == "test_value"
        assert os.path.exists(tmp_path)
        with open(tmp_path, "rb") as f:
            assert pickle.load(f) == "test_value"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_cache_with_path_no_verbose():
    import tempfile
    import os
    import pickle

    def test_func():
        return "test_value"

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        decorated = cache(tmp_path, verbose=False)(test_func)
        result = decorated()
        assert result == "test_value"
        assert os.path.exists(tmp_path)
        with open(tmp_path, "rb") as f:
            assert pickle.load(f) == "test_value"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_cache_with_existing_file():
    import tempfile
    import os
    import pickle

    def test_func():
        return "new_value"

    cached_value = "cached_value"
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name
        pickle.dump(cached_value, tmp_file)
    try:
        decorated = cache(tmp_path, verbose=True)(test_func)
        result = decorated()
        assert result == cached_value
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_cache_with_no_path():
    def test_func():
        return "test_value"

    decorated = cache(None, verbose=True)(test_func)
    result = decorated()
    assert result == "test_value"


# LLM-generated content at query #2
#--------------------------

```python
def test_cache_with_path_and_verbose():
    import tempfile
    import os
    import pickle

    def dummy_func():
        return "test_data"

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = os.path.join(temp_dir, "cache.pkl")
        cached_func = cache(cache_path, verbose=True, name="test")(dummy_func)
        result = cached_func()
        assert result == "test_data"
        assert os.path.exists(cache_path)
        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)
        assert cached_data == "test_data"

def test_cache_with_path_and_no_verbose():
    import tempfile
    import os
    import pickle

    def dummy_func():
        return "test_data"

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = os.path.join(temp_dir, "cache.pkl")
        cached_func = cache(cache_path, verbose=False, name="test")(dummy_func)
        result = cached_func()
        assert result == "test_data"
        assert os.path.exists(cache_path)
        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)
        assert cached_data == "test_data"

def test_cache_without_path():
    import tempfile
    import os

    def dummy_func():
        return "test_data"

    with tempfile.TemporaryDirectory() as temp_dir:
        cached_func = cache(None, verbose=True, name="test")(dummy_func)
        result = cached_func()
        assert result == "test_data"
        assert not os.path.exists(os.path.join(temp_dir, "cache.pkl"))

def test_cache_with_existing_cache():
    import tempfile
    import os
    import pickle

    def dummy_func():
        return "test_data"

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = os.path.join(temp_dir, "cache.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump("cached_data", f)
        cached_func = cache(cache_path, verbose=True, name="test")(dummy_func)
        result = cached_func()
        assert result == "cached_data"
        assert os.path.exists(cache_path)
        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)
        assert cached_data == "cached_data"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    assert readable_size(1024, 0) == "1K"
    assert readable_size(1024 * 1024, 3) == "1.000M"
    assert readable_size(1024 * 1024 * 1024, 1) == "1.0G"

def test_readable_size_fractional():
    assert readable_size(1536) == "1.50K"
    assert readable_size(1024 * 1024 * 1.5) == "1.50M"
    assert readable_size(1024 * 1024 * 1024 * 2.5) == "2.50G"

def test_readable_size_large_number():
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1024.00P"

def test_readable_size_zero():
    assert readable_size(0) == "0.00"


# LLM-generated content at query #2
#--------------------------

```python
def test_copy_tree_with_overwrite():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src)
    os.makedirs(dst)
    with open(os.path.join(src, "file.txt"), "w") as f:
        f.write("source content")
    with open(os.path.join(dst, "file.txt"), "w") as f:
        f.write("destination content")
    copy_tree(src, dst, overwrite=True)
    with open(os.path.join(dst, "file.txt"), "r") as f:
        assert f.read() == "source content"
    shutil.rmtree(src)
    shutil.rmtree(dst)

def test_copy_tree_without_overwrite():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src)
    os.makedirs(dst)
    with open(os.path.join(src, "file.txt"), "w") as f:
        f.write("source content")
    with open(os.path.join(dst, "file.txt"), "w") as f:
        f.write("destination content")
    copy_tree(src, dst, overwrite=False)
    with open(os.path.join(dst, "file.txt"), "r") as f:
        assert f.read() == "destination content"
    shutil.rmtree(src)
    shutil.rmtree(dst)

def test_copy_tree_new_destination():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src)
    with open(os.path.join(src, "file.txt"), "w") as f:
        f.write("source content")
    copy_tree(src, dst)
    assert os.path.exists(dst)
    with open(os.path.join(dst, "file.txt"), "r") as f:
        assert f.read() == "source content"
    shutil.rmtree(src)
    shutil.rmtree(dst)

def test_copy_tree_with_subdirectories():
    src = "test_src"
    dst = "test_dst"
    os.makedirs(src)
    os.makedirs(os.path.join(src, "subdir"))
    with open(os.path.join(src, "subdir", "file.txt"), "w") as f:
        f.write("subdir content")
    copy_tree(src, dst)
    assert os.path.exists(os.path.join(dst, "subdir", "file.txt"))
    with open(os.path.join(dst, "subdir", "file.txt"), "r") as f:
        assert f.read() == "subdir content"
    shutil.rmtree(src)
    shutil.rmtree(dst)


