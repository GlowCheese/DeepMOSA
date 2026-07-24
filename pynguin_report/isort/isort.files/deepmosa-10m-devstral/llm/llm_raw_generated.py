####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_find_with_empty_paths():
    result = list(find([], Config(), [], []))
    assert result == []

def test_find_with_single_file():
    with open("test_file.py", "w") as f:
        f.write("# test")
    config = Config()
    result = list(find(["test_file.py"], config, [], []))
    assert result == ["test_file.py"]
    os.remove("test_file.py")

def test_find_with_non_existent_path():
    skipped, broken = [], []
    result = list(find(["non_existent_path.py"], Config(), skipped, broken))
    assert result == []
    assert broken == ["non_existent_path.py"]

def test_find_with_directory_containing_python_files():
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# test2")
    config = Config()
    result = list(find(["test_dir"], config, [], []))
    assert len(result) == 2
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.py" in result
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.py")
    os.rmdir("test_dir")

def test_find_with_skipped_file():
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/skipped.py", "w") as f:
        f.write("# skipped")
    config = Config()
    skipped = []
    result = list(find(["test_dir"], config, skipped, []))
    assert result == []
    assert skipped == [os.path.abspath("test_dir/skipped.py")]
    os.remove("test_dir/skipped.py")
    os.rmdir("test_dir")

def test_find_with_broken_symlink():
    os.makedirs("test_dir", exist_ok=True)
    os.symlink("non_existent.py", "test_dir/link.py")
    config = Config(follow_links=True)
    skipped, broken = [], []
    result = list(find(["test_dir"], config, skipped, broken))
    assert result == []
    os.remove("test_dir/link.py")
    os.rmdir("test_dir")


# LLM-generated content at query #2
#--------------------------

```python
def test_find_with_empty_paths():
    result = list(find([], Config(), [], []))
    assert result == []

def test_find_with_nonexistent_path():
    broken = []
    result = list(find(["nonexistent_path.py"], Config(), [], broken))
    assert result == []
    assert broken == ["nonexistent_path.py"]

def test_find_with_single_file():
    with open("test_file.py", "w") as f:
        f.write("# test")
    result = list(find(["test_file.py"], Config(), [], []))
    assert result == ["test_file.py"]
    os.remove("test_file.py")

def test_find_with_skipped_file():
    with open("skipped_file.py", "w") as f:
        f.write("# test")
    config = Config()
    config.skip_patterns = ["skipped_*"]
    skipped = []
    result = list(find(["skipped_file.py"], config, skipped, []))
    assert result == []
    assert skipped == [os.path.abspath("skipped_file.py")]
    os.remove("skipped_file.py")

def test_find_with_directory():
    os.makedirs("test_dir")
    with open("test_dir/file1.py", "w") as f:
        f.write("# test")
    with open("test_dir/file2.txt", "w") as f:
        f.write("# test")
    config = Config()
    result = list(find(["test_dir"], config, [], []))
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.txt" not in result
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.txt")
    os.rmdir("test_dir")

def test_find_with_skipped_directory():
    os.makedirs("skipped_dir/subdir")
    with open("skipped_dir/file.py", "w") as f:
        f.write("# test")
    config = Config()
    config.skip_patterns = ["skipped_*"]
    skipped = []
    result = list(find(["skipped_dir"], config, skipped, []))
    assert result == []
    assert any("skipped_dir" in path for path in skipped)
    os.remove("skipped_dir/file.py")
    os.rmdir("skipped_dir/subdir")
    os.rmdir("skipped_dir")

def test_find_with_symlink():
    os.makedirs("real_dir")
    with open("real_dir/file.py", "w") as f:
        f.write("# test")
    os.symlink("real_dir", "symlink_dir")
    config = Config(follow_links=True)
    result = list(find(["symlink_dir"], config, [], []))
    assert "symlink_dir/file.py" in result
    os.remove("real_dir/file.py")
    os.rmdir("real_dir")
    os.remove("symlink_dir")

def test_find_with_broken_symlink():
    os.symlink("nonexistent", "broken_link")
    broken = []
    result = list(find(["broken_link"], Config(), [], broken))
    assert result == []
    assert broken == ["broken_link"]
    os.remove("broken_link")


# LLM-generated content at query #3
#--------------------------

```python
def test_isdir_predicate_evaluates_to_true():
    path = "test_directory"
    os.makedirs(path, exist_ok=True)
    assert os.path.isdir(path) is True
    os.rmdir(path)


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_at_line_31():
    paths = ["nonexistent_path"]
    config = Config()
    skipped = []
    broken = []

    list(find(paths, config, skipped, broken))

    assert broken == ["nonexistent_path"]


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    paths = ["nonexistent_file.py"]
    config = Config()
    skipped = []
    broken = []
    find(paths, config, skipped, broken)
    assert broken == ["nonexistent_file.py"]


# LLM-generated content at query #6
#--------------------------

```python
def test_find_with_empty_paths():
    result = list(find([], Config(), [], []))
    assert result == []

def test_find_with_nonexistent_path():
    broken = []
    result = list(find(["nonexistent_path"], Config(), [], broken))
    assert result == []
    assert broken == ["nonexistent_path"]

def test_find_with_single_file():
    with open("test_file.py", "w") as f:
        f.write("# test")
    result = list(find(["test_file.py"], Config(), [], []))
    assert result == ["test_file.py"]
    os.remove("test_file.py")

def test_find_with_directory_containing_python_files():
    os.makedirs("test_dir")
    with open("test_dir/file1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# test2")
    with open("test_dir/file3.txt", "w") as f:
        f.write("text")
    result = list(find(["test_dir"], Config(), [], []))
    assert set(result) == {"test_dir/file1.py", "test_dir/file2.py"}
    shutil.rmtree("test_dir")

def test_find_with_skipped_directory():
    os.makedirs("test_dir/skip_dir")
    with open("test_dir/skip_dir/file.py", "w") as f:
        f.write("# test")
    config = Config()
    config.skip = ["skip_dir"]
    skipped = []
    result = list(find(["test_dir"], config, skipped, []))
    assert result == []
    assert skipped == ["test_dir/skip_dir"]
    shutil.rmtree("test_dir")

def test_find_with_skipped_file():
    with open("test_file.py", "w") as f:
        f.write("# test")
    config = Config()
    config.skip = ["test_file.py"]
    skipped = []
    result = list(find(["test_file.py"], config, skipped, []))
    assert result == []
    assert skipped == [os.path.abspath("test_file.py")]
    os.remove("test_file.py")

def test_find_with_follow_links_disabled():
    os.makedirs("test_dir")
    os.makedirs("test_dir/subdir")
    os.symlink("subdir", "test_dir/link")
    with open("test_dir/subdir/file.py", "w") as f:
        f.write("# test")
    config = Config(follow_links=False)
    result = list(find(["test_dir"], config, [], []))
    assert "test_dir/link/file.py" not in result
    shutil.rmtree("test_dir")

def test_find_with_follow_links_enabled():
    os.makedirs("test_dir")
    os.makedirs("test_dir/subdir")
    os.symlink("subdir", "test_dir/link")
    with open("test_dir/subdir/file.py", "w") as f:
        f.write("# test")
    config = Config(follow_links=True)
    result = list(find(["test_dir"], config, [], []))
    assert "test_dir/link/file.py" in result
    shutil.rmtree("test_dir")

def test_find_with_unsupported_filetype():
    with open("test_file.txt", "w") as f:
        f.write("text")
    result = list(find(["test_file.txt"], Config(), [], []))
    assert result == []
    os.remove("test_file.txt")


# LLM-generated content at query #7
#--------------------------

```python
def test_isdir_predicate_evaluates_to_true():
    os.path.isdir.return_value = True
    assert os.path.isdir("some_path") is True


# LLM-generated content at query #8
#--------------------------

```python
def test_find_yields_python_files_in_directory():
    paths = ["test_dir"]
    config = Config(follow_links=False, skipped_paths=[], supported_filetypes=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_dir/file1.py" in result
    assert "test_dir/subdir/file2.py" in result

def test_find_skips_files_and_dirs_in_config():
    paths = ["test_dir"]
    config = Config(follow_links=False, skipped_paths=["test_dir/skip_dir", "test_dir/skip_file.py"], supported_filetypes=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_dir/skip_file.py" not in result
    assert "test_dir/skip_dir/file.py" not in result
    assert "test_dir/skip_dir" in skipped
    assert "test_dir/skip_file.py" in skipped

def test_find_handles_broken_paths():
    paths = ["nonexistent_path", "test_dir"]
    config = Config(follow_links=False, skipped_paths=[], supported_filetypes=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "nonexistent_path" in broken
    assert "test_dir/file1.py" in result

def test_find_yields_single_file_path():
    paths = ["test_file.py"]
    config = Config(follow_links=False, skipped_paths=[], supported_filetypes=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_file.py" in result


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_31():
    paths = ["nonexistent_path"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(broken) == 1
    assert broken[0] == "nonexistent_path"
    assert len(skipped) == 0
    assert len(result) == 0


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_at_line_31():
    paths = ["nonexistent_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(broken) == 1
    assert broken[0] == "nonexistent_file.py"


# LLM-generated content at query #11
#--------------------------

```python
def test_path_is_directory():
    assert os.path.isdir("/path/to/existing/directory") == True


# LLM-generated content at query #12
#--------------------------

```python
def test_isdir_predicate_evaluates_to_true():
    path = "test_directory"
    os.makedirs(path, exist_ok=True)
    assert os.path.isdir(path)


# LLM-generated content at query #13
#--------------------------

```python
def test_find_with_empty_paths():
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

def test_find_with_nonexistent_path():
    paths = ["nonexistent_path.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent_path.py"]

def test_find_with_single_file():
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

def test_find_with_directory_containing_python_files():
    paths = ["test_directory"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_directory/file1.py", "test_directory/file2.py"]
    assert skipped == []
    assert broken == []

def test_find_with_skipped_file():
    paths = ["test_directory"]
    config = Config()
    config.skip_patterns = ["file1.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_directory/file2.py"]
    assert skipped == ["test_directory/file1.py"]
    assert broken == []

def test_find_with_broken_symlink():
    paths = ["broken_symlink"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["broken_symlink"]


# LLM-generated content at query #14
#--------------------------

```python
def test_isdir_predicate_evaluates_to_true():
    import os
    from pathlib import Path
    from unittest.mock import Mock

    # Setup
    path = "test_dir"
    os.path.isdir = Mock(return_value=True)
    os.path.exists = Mock(return_value=True)
    os.walk = Mock(return_value=[])
    config = Mock()
    skipped = []
    broken = []

    # Exercise
    result = list(find([path], config, skipped, broken))

    # Verify
    assert os.path.isdir.called
    assert os.path.isdir.call_args[0][0] == path


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_evaluates_to_true():
    paths = ["nonexistent_path"]
    config = Config()
    skipped = []
    broken = []
    result = find(paths, config, skipped, broken)
    assert "nonexistent_path" in broken


# LLM-generated content at query #16
#--------------------------

```python
def test_find_with_empty_paths():
    paths = []
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

def test_find_with_nonexistent_path():
    paths = ["nonexistent_path.py"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent_path.py"]

def test_find_with_single_file():
    paths = ["test_file.py"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    with open("test_file.py", "w") as f:
        f.write("# test")
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []
    os.remove("test_file.py")

def test_find_with_directory_containing_python_files():
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# test2")
    with open("test_dir/file.txt", "w") as f:
        f.write("# not python")
    paths = ["test_dir"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert set(result) == {"test_dir/file1.py", "test_dir/file2.py"}
    assert skipped == []
    assert broken == []
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.py")
    os.remove("test_dir/file.txt")
    os.rmdir("test_dir")

def test_find_with_skipped_file():
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/skipped.py", "w") as f:
        f.write("# skipped")
    paths = ["test_dir"]
    config = Config(follow_links=False, skip_patterns=["skipped.py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath("test_dir/skipped.py")]
    assert broken == []
    os.remove("test_dir/skipped.py")
    os.rmdir("test_dir")

def test_find_with_skipped_directory():
    os.makedirs("test_dir/skipped_dir", exist_ok=True)
    with open("test_dir/skipped_dir/file.py", "w") as f:
        f.write("# skipped")
    paths = ["test_dir"]
    config = Config(follow_links=False, skip_patterns=["skipped_dir"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath("test_dir/skipped_dir")]
    assert broken == []
    os.remove("test_dir/skipped_dir/file.py")
    os.rmdir("test_dir/skipped_dir")
    os.rmdir("test_dir")

def test_find_with_symlink():
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file.py", "w") as f:
        f.write("# test")
    os.symlink("test_dir", "test_link")
    paths = ["test_link"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_link/file.py"]
    assert skipped == []
    assert broken == []
    os.remove("test_link")
    os.remove("test_dir/file.py")
    os.rmdir("test_dir")

def test_find_with_broken_symlink():
    os.symlink("nonexistent", "broken_link")
    paths = ["broken_link"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["broken_link"]
    os.remove("broken_link")


# LLM-generated content at query #17
#--------------------------

```python
def test_isdir_predicate_evaluates_to_true():
    paths = ["/existing_directory"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = find(paths, config, skipped, broken)
    assert list(result) == []


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_find_with_directory_and_supported_files():
    paths = ["test_dir"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.py" in result
    assert skipped == []
    assert broken == []

def test_find_with_skipped_directory():
    paths = ["test_dir"]
    config = Config(follow_links=False, skipped_dirs=["test_dir/skip_dir"], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "test_dir/file1.py" in result
    assert skipped == ["test_dir/skip_dir"]
    assert broken == []

def test_find_with_broken_path():
    paths = ["nonexistent_path"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent_path"]

def test_find_with_single_file():
    paths = ["test_file.py"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

def test_find_with_unsupported_file():
    paths = ["test_dir"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".txt"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []


# LLM-generated content at query #2
#--------------------------

```python
def test_predicate_at_line_7_evaluates_to_false():
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []


# LLM-generated content at query #3
#--------------------------

```python
def test_find_with_empty_paths():
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

def test_find_with_single_file():
    paths = ["test.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test.py"]
    assert skipped == []
    assert broken == []

def test_find_with_nonexistent_file():
    paths = ["nonexistent.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent.py"]

def test_find_with_directory():
    paths = ["src"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) > 0
    assert all(os.path.isfile(path) for path in result)
    assert skipped == []
    assert broken == []

def test_find_with_skipped_file():
    paths = ["test.py"]
    config = Config(skipped_files=["test.py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath("test.py")]
    assert broken == []

def test_find_with_skipped_directory():
    paths = ["src"]
    config = Config(skipped_dirs=["src/skipped_dir"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) > 0
    assert all("skipped_dir" not in path for path in result)
    assert skipped == [str(Path("src/skipped_dir").resolve())]
    assert broken == []

def test_find_with_broken_symlink():
    paths = ["broken_link"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["broken_link"]

def test_find_with_circular_symlink():
    paths = ["circular_link"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) > 0
    assert skipped == []
    assert broken == []


# LLM-generated content at query #4
#--------------------------

```python
def test_find_with_directory_and_valid_files():
    paths = ["test_dir"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py", "test_dir/subdir/file2.py"]
    assert skipped == []
    assert broken == []

def test_find_with_skipped_directory():
    paths = ["test_dir"]
    config = Config(follow_links=False, skipped_dirs=["test_dir/skip_me"], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py"]
    assert skipped == ["test_dir/skip_me"]
    assert broken == []

def test_find_with_broken_path():
    paths = ["non_existent_dir"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_dir"]

def test_find_with_single_file():
    paths = ["test_file.py"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

def test_find_with_unsupported_filetype():
    paths = ["test_dir"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".txt"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

def test_find_with_skipped_file():
    paths = ["test_dir/skip_me.py"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath("test_dir/skip_me.py")]
    assert broken == []


# LLM-generated content at query #5
#--------------------------

```python
def test_isdir_predicate_evaluates_to_true():
    import os
    from pathlib import Path
    from unittest.mock import Mock

    # Setup
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)

    # Test
    assert os.path.isdir(test_dir) is True

    # Cleanup
    os.rmdir(test_dir)


# LLM-generated content at query #6
#--------------------------

```python
def test_find_with_empty_paths():
    result = list(find([], Config(), [], []))
    assert result == []

def test_find_with_non_existent_path():
    broken = []
    result = list(find(["/non/existent/path"], Config(), [], broken))
    assert result == []
    assert broken == ["/non/existent/path"]

def test_find_with_single_file():
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(b"print('hello')")
        f.flush()
        result = list(find([f.name], Config(), [], []))
        assert result == [f.name]
        os.unlink(f.name)

def test_find_with_directory_containing_py_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = os.path.join(tmpdir, "test.py")
        with open(py_file, "w") as f:
            f.write("print('hello')")
        result = list(find([tmpdir], Config(), [], []))
        assert result == [py_file]

def test_find_with_skipped_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = os.path.join(tmpdir, "test.py")
        with open(py_file, "w") as f:
            f.write("print('hello')")
        config = Config()
        config.skip_patterns = ["test.py"]
        skipped = []
        result = list(find([tmpdir], config, skipped, []))
        assert result == []
        assert skipped == [os.path.abspath(py_file)]

def test_find_with_broken_symlink():
    with tempfile.TemporaryDirectory() as tmpdir:
        link_path = os.path.join(tmpdir, "broken_link")
        os.symlink("/non/existent/target", link_path)
        broken = []
        result = list(find([link_path], Config(), [], broken))
        assert result == []
        assert broken == [link_path]


# LLM-generated content at query #7
#--------------------------

```python
def test_find_with_empty_paths():
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

def test_find_with_nonexistent_path():
    paths = ["nonexistent_path"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent_path"]

def test_find_with_single_file():
    paths = ["existing_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["existing_file.py"]
    assert skipped == []
    assert broken == []

def test_find_with_directory():
    paths = ["existing_dir"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert all(os.path.isfile(path) for path in result)
    assert skipped == []
    assert broken == []

def test_find_with_skipped_file():
    paths = ["skipped_file.py"]
    config = Config()
    config.skip_patterns = ["skipped_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath("skipped_file.py")]
    assert broken == []

def test_find_with_skipped_directory():
    paths = ["skipped_dir"]
    config = Config()
    config.skip_patterns = ["skipped_dir"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == [str(Path("skipped_dir").resolve())]
    assert broken == []

def test_find_with_follow_links_disabled():
    paths = ["dir_with_symlink"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert all(os.path.isfile(path) for path in result)
    assert skipped == []
    assert broken == []

def test_find_with_follow_links_enabled():
    paths = ["dir_with_symlink"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert all(os.path.isfile(path) for path in result)
    assert skipped == []
    assert broken == []


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    paths = ["nonexistent_file.py"]
    config = Config()
    skipped = []
    broken = []
    find(paths, config, skipped, broken)
    assert len(broken) == 1
    assert broken[0] == "nonexistent_file.py"


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    paths = ["nonexistent_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(broken) == 1
    assert broken[0] == "nonexistent_file.py"


# LLM-generated content at query #10
#--------------------------

```python
def test_isdir_predicate_evaluates_to_true():
    path = "test_directory"
    os.makedirs(path, exist_ok=True)
    assert os.path.isdir(path)


# LLM-generated content at query #11
#--------------------------

```python
def test_find_with_directory_and_valid_files():
    paths = ["test_dir"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_dir/file1.py" in result
    assert "test_dir/subdir/file2.py" in result
    assert len(skipped) == 0
    assert len(broken) == 0

def test_find_with_skipped_directory():
    paths = ["test_dir"]
    config = Config(follow_links=False, skip_dirs=["subdir"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "test_dir/file1.py" in result
    assert "test_dir/subdir/file2.py" not in result
    assert len(skipped) == 1
    assert "test_dir/subdir" in skipped[0]
    assert len(broken) == 0

def test_find_with_nonexistent_path():
    paths = ["nonexistent_path"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert "nonexistent_path" in broken

def test_find_with_file_path():
    paths = ["test_dir/file1.py"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "test_dir/file1.py" in result
    assert len(skipped) == 0
    assert len(broken) == 0

def test_find_with_skipped_file():
    paths = ["test_dir"]
    config = Config(follow_links=False, skip_files=["file1.py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "test_dir/file1.py" not in result
    assert "test_dir/subdir/file2.py" in result
    assert len(skipped) == 1
    assert "test_dir/file1.py" in skipped[0]
    assert len(broken) == 0


# LLM-generated content at query #12
#--------------------------

```python
def test_find_with_empty_paths():
    result = list(find([], Config(), [], []))
    assert result == []

def test_find_with_nonexistent_path():
    broken = []
    result = list(find(["nonexistent_path.py"], Config(), [], broken))
    assert result == []
    assert broken == ["nonexistent_path.py"]

def test_find_with_single_file():
    with tempfile.NamedTemporaryFile(suffix=".py") as f:
        result = list(find([f.name], Config(), [], []))
        assert result == [f.name]

def test_find_with_skipped_file():
    with tempfile.NamedTemporaryFile(suffix=".py") as f:
        config = Config()
        config.add_skip_path(Path(f.name))
        skipped = []
        result = list(find([f.name], config, skipped, []))
        assert result == []
        assert skipped == [os.path.abspath(f.name)]

def test_find_with_directory_containing_python_files():
    with tempfile.TemporaryDirectory() as dirname:
        file1 = os.path.join(dirname, "file1.py")
        file2 = os.path.join(dirname, "file2.py")
        with open(file1, "w") as f:
            f.write("# test")
        with open(file2, "w") as f:
            f.write("# test")
        result = list(find([dirname], Config(), [], []))
        assert set(result) == {file1, file2}

def test_find_with_skipped_directory():
    with tempfile.TemporaryDirectory() as dirname:
        subdir = os.path.join(dirname, "subdir")
        os.makedirs(subdir)
        file_in_subdir = os.path.join(subdir, "file.py")
        with open(file_in_subdir, "w") as f:
            f.write("# test")
        config = Config()
        config.add_skip_path(Path(subdir))
        skipped = []
        result = list(find([dirname], config, skipped, []))
        assert result == []
        assert skipped == [os.path.abspath(subdir)]

def test_find_with_non_python_file():
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        result = list(find([f.name], Config(), [], []))
        assert result == []

def test_find_with_follow_links_disabled():
    with tempfile.TemporaryDirectory() as dirname:
        subdir = os.path.join(dirname, "subdir")
        os.makedirs(subdir)
        file_in_subdir = os.path.join(subdir, "file.py")
        with open(file_in_subdir, "w") as f:
            f.write("# test")
        config = Config(follow_links=False)
        result = list(find([dirname], config, [], []))
        assert file_in_subdir in result

def test_find_with_circular_symlink():
    with tempfile.TemporaryDirectory() as dirname:
        subdir = os.path.join(dirname, "subdir")
        os.makedirs(subdir)
        link = os.path.join(dirname, "link")
        os.symlink(subdir, link)
        file_in_subdir = os.path.join(subdir, "file.py")
        with open(file_in_subdir, "w") as f:
            f.write("# test")
        config = Config(follow_links=True)
        result = list(find([dirname], config, [], []))
        assert file_in_subdir in result


# LLM-generated content at query #13
#--------------------------

```python
def test_find_with_directory_and_supported_files():
    paths = ["test_dir"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py", "test_dir/file2.py"]
    assert skipped == []
    assert broken == []

def test_find_with_skipped_directory():
    paths = ["test_dir"]
    config = Config(follow_links=False, skipped_dirs=["test_dir/skip_dir"], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py"]
    assert skipped == ["test_dir/skip_dir"]
    assert broken == []

def test_find_with_broken_path():
    paths = ["nonexistent_path"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent_path"]

def test_find_with_file_path():
    paths = ["test_file.py"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

def test_find_with_unsupported_filetype():
    paths = ["test_dir"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".txt"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file.txt"]
    assert skipped == []
    assert broken == []


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_evaluates_to_true():
    paths = ["nonexistent_path"]
    config = Config()
    skipped = []
    broken = []
    find(paths, config, skipped, broken)
    assert "nonexistent_path" in broken


# LLM-generated content at query #15
#--------------------------

```python
def test_isdir_predicate_evaluates_to_true():
    path = "test_directory"
    os.makedirs(path, exist_ok=True)
    assert os.path.isdir(path)


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    paths = ["nonexistent_file.py"]
    config = Config()
    skipped = []
    broken = []
    find(paths, config, skipped, broken)
    assert len(broken) == 1
    assert broken[0] == "nonexistent_file.py"


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    paths = ["nonexistent_path"]
    config = Config()
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))

    assert len(broken) == 1
    assert broken[0] == "nonexistent_path"
    assert len(result) == 0
    assert len(skipped) == 0


# LLM-generated content at query #18
#--------------------------

```python
def test_isdir_predicate_evaluates_to_true():
    import os
    from pathlib import Path
    from unittest.mock import MagicMock

    # Setup
    paths = ["test_dir"]
    config = MagicMock()
    config.follow_links = False
    skipped = []
    broken = []
    os.path.isdir = MagicMock(return_value=True)
    os.walk = MagicMock(return_value=[])
    os.path.exists = MagicMock(return_value=True)

    # Exercise
    result = list(find(paths, config, skipped, broken))

    # Verify
    assert os.path.isdir.called
    assert os.path.isdir.call_args[0][0] == "test_dir"


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_true():
    paths = ["/existing_directory"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) > 0


# LLM-generated content at query #20
#--------------------------

```python
def test_find_with_directory_and_valid_files():
    paths = ["test_dir"]
    config = Config(follow_links=False, skipped_patterns=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py", "test_dir/subdir/file2.py"]
    assert skipped == []
    assert broken == []

def test_find_with_skipped_files():
    paths = ["test_dir"]
    config = Config(follow_links=False, skipped_patterns=["*skip*"], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py"]
    assert skipped == ["test_dir/skip_file.py"]
    assert broken == []

def test_find_with_broken_path():
    paths = ["nonexistent_path"]
    config = Config(follow_links=False, skipped_patterns=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent_path"]

def test_find_with_single_file():
    paths = ["test_dir/file1.py"]
    config = Config(follow_links=False, skipped_patterns=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py"]
    assert skipped == []
    assert broken == []

def test_find_with_unsupported_filetype():
    paths = ["test_dir"]
    config = Config(follow_links=False, skipped_patterns=[], supported_extensions=[".txt"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_evaluates_to_false():
    paths = ["nonexistent_dir"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    find_result = find(paths, config, skipped, broken)
    assert list(find_result) == []
    assert skipped == []
    assert broken == ["nonexistent_dir"]


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_false():
    paths = ["nonexistent_path"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert not os.path.isdir("nonexistent_path")


# LLM-generated content at query #23
#--------------------------

```python
def test_find_with_empty_paths():
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

def test_find_with_nonexistent_path():
    paths = ["nonexistent_path"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent_path"]

def test_find_with_single_file():
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

def test_find_with_directory_containing_python_files():
    paths = ["test_dir"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py", "test_dir/file2.py"]
    assert skipped == []
    assert broken == []

def test_find_with_skipped_directory():
    paths = ["test_dir"]
    config = Config(skip_dirs=["test_dir/skip_me"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py", "test_dir/file2.py"]
    assert skipped == ["test_dir/skip_me"]
    assert broken == []

def test_find_with_skipped_file():
    paths = ["test_dir"]
    config = Config(skip_files=["test_dir/skip_me.py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py", "test_dir/file2.py"]
    assert skipped == ["test_dir/skip_me.py"]
    assert broken == []

def test_find_with_broken_symlink():
    paths = ["broken_link"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["broken_link"]


# LLM-generated content at query #24
#--------------------------

```python
def test_find_with_empty_paths():
    assert list(find([], Config(), [], [])) == []

def test_find_with_non_existent_path():
    broken = []
    assert list(find(["/non/existent/path"], Config(), [], broken)) == []
    assert broken == ["/non/existent/path"]

def test_find_with_single_file():
    with open("test_file.py", "w") as f:
        f.write("# test")
    assert list(find(["test_file.py"], Config(), [], [])) == ["test_file.py"]
    os.remove("test_file.py")

def test_find_with_directory_containing_python_files():
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# test2")
    with open("test_dir/file3.txt", "w") as f:
        f.write("# not python")
    result = list(find(["test_dir"], Config(), [], []))
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.py" in result
    assert "test_dir/file3.txt" not in result
    shutil.rmtree("test_dir")

def test_find_with_skipped_directory():
    os.makedirs("test_dir/skip_me", exist_ok=True)
    with open("test_dir/skip_me/file.py", "w") as f:
        f.write("# test")
    config = Config()
    config.skip_patterns = ["skip_me"]
    skipped = []
    result = list(find(["test_dir"], config, skipped, []))
    assert result == []
    assert "skip_me" in skipped[0]
    shutil.rmtree("test_dir")

def test_find_with_skipped_file():
    with open("skip_me.py", "w") as f:
        f.write("# test")
    config = Config()
    config.skip_patterns = ["skip_me.py"]
    skipped = []
    result = list(find(["skip_me.py"], config, skipped, []))
    assert result == []
    assert "skip_me.py" in skipped[0]
    os.remove("skip_me.py")

def test_find_with_symlink_loop():
    os.makedirs("test_dir/subdir", exist_ok=True)
    os.symlink("../test_dir", "test_dir/subdir/loop_link")
    config = Config(follow_links=True)
    result = list(find(["test_dir"], config, [], []))
    assert len(result) == 0
    shutil.rmtree("test_dir")


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_false():
    paths = ["nonexistent_path"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "nonexistent_path"


# LLM-generated content at query #26
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_false():
    paths = ["test_file.txt"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert result[0] == "test_file.txt"


# LLM-generated content at query #27
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_false():
    paths = ["nonexistent_dir"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "nonexistent_dir"


# LLM-generated content at query #28
#--------------------------

```python
def test_os_path_isdir_evaluates_to_true():
    path = "/some/directory"
    assert os.path.isdir(path) is True


# LLM-generated content at query #29
#--------------------------

```python
def test_find_with_empty_paths():
    result = list(find([], Config(), [], []))
    assert result == []

def test_find_with_nonexistent_path():
    broken = []
    result = list(find(["nonexistent_path.py"], Config(), [], broken))
    assert result == []
    assert broken == ["nonexistent_path.py"]

def test_find_with_single_file():
    with open("test_file.py", "w") as f:
        f.write("# test")
    result = list(find(["test_file.py"], Config(), [], []))
    assert result == ["test_file.py"]
    os.remove("test_file.py")

def test_find_with_skipped_file():
    with open("skip_me.py", "w") as f:
        f.write("# skip")
    config = Config()
    config.skip_patterns = ["skip_me.py"]
    skipped = []
    result = list(find(["skip_me.py"], config, skipped, []))
    assert result == []
    assert skipped == [os.path.abspath("skip_me.py")]
    os.remove("skip_me.py")

def test_find_with_directory_containing_python_files():
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("# file1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# file2")
    with open("test_dir/readme.txt", "w") as f:
        f.write("# not python")
    result = list(find(["test_dir"], Config(), [], []))
    assert sorted(result) == ["test_dir/file1.py", "test_dir/file2.py"]
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.py")
    os.remove("test_dir/readme.txt")
    os.rmdir("test_dir")

def test_find_with_nested_directories():
    os.makedirs("parent/child", exist_ok=True)
    with open("parent/child/nested.py", "w") as f:
        f.write("# nested")
    result = list(find(["parent"], Config(), [], []))
    assert result == ["parent/child/nested.py"]
    os.remove("parent/child/nested.py")
    os.rmdir("parent/child")
    os.rmdir("parent")

def test_find_with_skipped_directory():
    os.makedirs("skip_dir/subdir", exist_ok=True)
    with open("skip_dir/subdir/file.py", "w") as f:
        f.write("# should be skipped")
    config = Config()
    config.skip_patterns = ["skip_dir"]
    skipped = []
    result = list(find(["skip_dir"], config, skipped, []))
    assert result == []
    assert skipped == [str(Path("skip_dir").resolve())]
    os.remove("skip_dir/subdir/file.py")
    os.rmdir("skip_dir/subdir")
    os.rmdir("skip_dir")

def test_find_with_symlink_loop():
    os.makedirs("link_dir", exist_ok=True)
    os.symlink("link_dir", "link_dir/link")
    with open("link_dir/file.py", "w") as f:
        f.write("# test")
    config = Config(follow_links=True)
    result = list(find(["link_dir"], config, [], []))
    assert result == ["link_dir/file.py"]
    os.remove("link_dir/file.py")
    os.remove("link_dir/link")
    os.rmdir("link_dir")


# LLM-generated content at query #30
#--------------------------

```python
def test_predicate_evaluates_to_true():
    paths = ["/valid/directory"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    find_result = find(paths, config, skipped, broken)
    assert list(find_result) == []


# LLM-generated content at query #31
#--------------------------

```python
def test_os_path_isdir_evaluates_to_true():
    path = "test_directory"
    assert os.path.isdir(path) is True


# LLM-generated content at query #32
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_true():
    paths = ["valid_directory"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    find_result = find(paths, config, skipped, broken)
    assert list(find_result) == ["valid_directory/file1.py", "valid_directory/file2.py"]


# LLM-generated content at query #33
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_false():
    paths = ["nonexistent_directory"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "nonexistent_directory"


# LLM-generated content at query #34
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_false():
    paths = ["nonexistent_directory"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "nonexistent_directory"


