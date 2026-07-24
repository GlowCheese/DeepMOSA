####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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

def test_find_with_directory():
    os.makedirs("test_dir")
    with open("test_dir/file1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# test2")
    result = list(find(["test_dir"], Config(), [], []))
    assert set(result) == {"test_dir/file1.py", "test_dir/file2.py"}
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.py")
    os.rmdir("test_dir")

def test_find_with_skipped_directory():
    os.makedirs("skip_dir")
    with open("skip_dir/file.py", "w") as f:
        f.write("# skip")
    config = Config()
    config.skip_patterns = ["skip_dir"]
    skipped = []
    result = list(find(["skip_dir"], config, skipped, []))
    assert result == []
    assert skipped == [str(Path("skip_dir").resolve())]
    os.remove("skip_dir/file.py")
    os.rmdir("skip_dir")

def test_find_with_nested_directories():
    os.makedirs("parent/child")
    with open("parent/file1.py", "w") as f:
        f.write("# parent")
    with open("parent/child/file2.py", "w") as f:
        f.write("# child")
    result = list(find(["parent"], Config(), [], []))
    assert set(result) == {"parent/file1.py", "parent/child/file2.py"}
    os.remove("parent/file1.py")
    os.remove("parent/child/file2.py")
    os.rmdir("parent/child")
    os.rmdir("parent")

def test_find_with_symlink():
    os.makedirs("real_dir")
    with open("real_dir/file.py", "w") as f:
        f.write("# real")
    os.symlink("real_dir", "link_dir")
    result = list(find(["link_dir"], Config(follow_links=True), [], []))
    assert result == ["link_dir/file.py"]
    os.remove("link_dir")
    os.remove("real_dir/file.py")
    os.rmdir("real_dir")

def test_find_with_unsupported_filetype():
    with open("test.txt", "w") as f:
        f.write("# not python")
    result = list(find(["test.txt"], Config(), [], []))
    assert result == []
    os.remove("test.txt")

def test_find_with_mixed_paths():
    with open("file.py", "w") as f:
        f.write("# file")
    os.makedirs("dir")
    with open("dir/file.py", "w") as f:
        f.write("# dir file")
    result = list(find(["file.py", "dir"], Config(), [], []))
    assert set(result) == {"file.py", "dir/file.py"}
    os.remove("file.py")
    os.remove("dir/file.py")
    os.rmdir("dir")


# LLM-generated content at query #2
#--------------------------

```python
def test_find_with_directory_and_valid_files():
    paths = ["valid_dir"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "valid_dir/file1.py" in result
    assert "valid_dir/file2.py" in result
    assert skipped == []
    assert broken == []

def test_find_with_skipped_directory():
    paths = ["dir_with_skipped"]
    config = Config(follow_links=False, skipped_dirs=["dir_with_skipped/skipped_subdir"], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "dir_with_skipped/valid_file.py" in result
    assert "dir_with_skipped/skipped_subdir" in skipped
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
    paths = ["single_file.py"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "single_file.py" in result
    assert skipped == []
    assert broken == []

def test_find_with_unsupported_filetype():
    paths = ["dir_with_unsupported"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "dir_with_unsupported/file.py" in result
    assert skipped == []
    assert broken == []

def test_find_with_symlink_loop():
    paths = ["dir_with_symlink"]
    config = Config(follow_links=True, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "dir_with_symlink/file.py" in result
    assert skipped == []
    assert broken == []


# LLM-generated content at query #3
#--------------------------

```python
def test_predicate_at_line_31():
    paths = ["nonexistent_path"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert broken == ["nonexistent_path"]
    assert result == []


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    assert not os.path.exists("non_existent_path")


# LLM-generated content at query #5
#--------------------------

```python
def test_find_with_directory_and_valid_files():
    paths = ["test_dir"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py", "test_dir/file2.py"]
    assert skipped == []
    assert broken == []

def test_find_with_directory_and_skipped_files():
    paths = ["test_dir"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py"]
    assert skipped == ["test_dir/file2.py"]
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

def test_find_with_file():
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

def test_find_with_symlink_and_follow_links_enabled():
    paths = ["test_symlink"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_symlink/file1.py", "test_symlink/file2.py"]
    assert skipped == []
    assert broken == []

def test_find_with_symlink_and_follow_links_disabled():
    paths = ["test_symlink"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []


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

def test_find_with_skipped_file():
    with open("skipped_file.py", "w") as f:
        f.write("# test")
    config = Config()
    config.skip_patterns = ["skipped_file.py"]
    skipped = []
    result = list(find(["skipped_file.py"], config, skipped, []))
    assert result == []
    assert skipped == [os.path.abspath("skipped_file.py")]
    os.remove("skipped_file.py")

def test_find_with_directory():
    os.makedirs("test_dir")
    with open("test_dir/file1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# test2")
    result = list(find(["test_dir"], Config(), [], []))
    assert set(result) == {"test_dir/file1.py", "test_dir/file2.py"}
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.py")
    os.rmdir("test_dir")

def test_find_with_skipped_directory():
    os.makedirs("test_dir/skipped_dir")
    with open("test_dir/skipped_dir/file.py", "w") as f:
        f.write("# test")
    config = Config()
    config.skip_patterns = ["skipped_dir"]
    skipped = []
    result = list(find(["test_dir"], config, skipped, []))
    assert result == []
    assert skipped == [str(Path("test_dir/skipped_dir").resolve())]
    os.remove("test_dir/skipped_dir/file.py")
    os.rmdir("test_dir/skipped_dir")
    os.rmdir("test_dir")

def test_find_with_follow_links_disabled():
    os.makedirs("test_dir")
    os.makedirs("test_dir/subdir")
    with open("test_dir/subdir/file.py", "w") as f:
        f.write("# test")
    os.symlink("test_dir/subdir", "test_dir/link")
    config = Config(follow_links=False)
    result = list(find(["test_dir"], config, [], []))
    assert "test_dir/link/file.py" not in result
    os.remove("test_dir/link")
    os.remove("test_dir/subdir/file.py")
    os.rmdir("test_dir/subdir")
    os.rmdir("test_dir")

def test_find_with_follow_links_enabled():
    os.makedirs("test_dir")
    os.makedirs("test_dir/subdir")
    with open("test_dir/subdir/file.py", "w") as f:
        f.write("# test")
    os.symlink("test_dir/subdir", "test_dir/link")
    config = Config(follow_links=True)
    result = list(find(["test_dir"], config, [], []))
    assert "test_dir/link/file.py" in result
    os.remove("test_dir/link")
    os.remove("test_dir/subdir/file.py")
    os.rmdir("test_dir/subdir")
    os.rmdir("test_dir")

def test_find_with_unsupported_filetype():
    with open("test_file.txt", "w") as f:
        f.write("# test")
    result = list(find(["test_file.txt"], Config(), [], []))
    assert result == []
    os.remove("test_file.txt")

def test_find_with_mixed_paths():
    with open("file.py", "w") as f:
        f.write("# test")
    os.makedirs("dir")
    with open("dir/file.py", "w") as f:
        f.write("# test")
    result = list(find(["file.py", "dir"], Config(), [], []))
    assert set(result) == {"file.py", "dir/file.py"}
    os.remove("file.py")
    os.remove("dir/file.py")
    os.rmdir("dir")

def test_find_with_circular_symlink():
    os.makedirs("test_dir")
    os.symlink("test_dir", "test_dir/link")
    config = Config(follow_links=True)
    result = list(find(["test_dir"], config, [], []))
    assert result == []
    os.remove("test_dir/link")
    os.rmdir("test_dir")


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert not os.path.exists("nonexistent_path")


# LLM-generated content at query #8
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
    assert os.path.isdir(test_dir)

    # Cleanup
    os.rmdir(test_dir)


# LLM-generated content at query #9
#--------------------------

```python
def test_isdir_predicate_evaluates_to_true():
    import os
    from pathlib import Path
    from unittest.mock import Mock

    # Setup
    paths = ["/existing_directory"]
    config = Mock()
    skipped = []
    broken = []

    # Mock os.path.isdir to return True for the given path
    os.path.isdir = Mock(return_value=True)

    # Mock os.walk to return a minimal iterator
    os.walk = Mock(return_value=[("/existing_directory", [], [])])

    # Mock other necessary methods
    config.follow_links = False
    config.is_skipped = Mock(return_value=False)
    config.is_supported_filetype = Mock(return_value=False)

    # Exercise
    result = list(find(paths, config, skipped, broken))

    # Verify
    assert os.path.isdir.called
    assert os.path.isdir.call_args[0][0] == "/existing_directory"


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_evaluates_to_true():
    paths = ["nonexistent_path"]
    config = Config()
    skipped = []
    broken = []
    find(paths, config, skipped, broken)
    assert "nonexistent_path" in broken


# LLM-generated content at query #11
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
    config = Config(follow_links=False, skipped_dirs=["test_dir/skip_me"], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.py" not in result
    assert skipped == ["test_dir/skip_me"]
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
    paths = ["test_dir/file1.py"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "test_dir/file1.py" in result
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


# LLM-generated content at query #12
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


# LLM-generated content at query #13
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


# LLM-generated content at query #14
#--------------------------

```python
def test_find_with_empty_paths():
    assert list(find([], Config(), [], [])) == []

def test_find_with_nonexistent_path():
    broken = []
    assert list(find(["nonexistent_path"], Config(), [], broken)) == []
    assert broken == ["nonexistent_path"]

def test_find_with_single_file():
    with open("test_file.py", "w") as f:
        f.write("# test")
    assert list(find(["test_file.py"], Config(), [], [])) == ["test_file.py"]
    os.remove("test_file.py")

def test_find_with_skipped_file():
    with open("test_file.py", "w") as f:
        f.write("# test")
    config = Config()
    config.skip_patterns = ["test_file.py"]
    skipped = []
    assert list(find(["test_file.py"], config, skipped, [])) == []
    assert skipped == [os.path.abspath("test_file.py")]
    os.remove("test_file.py")

def test_find_with_directory():
    os.makedirs("test_dir")
    with open("test_dir/test_file.py", "w") as f:
        f.write("# test")
    assert set(find(["test_dir"], Config(), [], [])) == {"test_dir/test_file.py"}
    os.remove("test_dir/test_file.py")
    os.rmdir("test_dir")

def test_find_with_skipped_directory():
    os.makedirs("test_dir")
    with open("test_dir/test_file.py", "w") as f:
        f.write("# test")
    config = Config()
    config.skip_patterns = ["test_dir"]
    skipped = []
    assert list(find(["test_dir"], config, skipped, [])) == []
    assert skipped == [str(Path("test_dir").resolve())]
    os.remove("test_dir/test_file.py")
    os.rmdir("test_dir")

def test_find_with_non_python_file():
    with open("test_file.txt", "w") as f:
        f.write("# test")
    assert list(find(["test_file.txt"], Config(), [], [])) == []
    os.remove("test_file.txt")

def test_find_with_follow_links_disabled():
    os.makedirs("test_dir")
    os.makedirs("test_dir/subdir")
    with open("test_dir/subdir/test_file.py", "w") as f:
        f.write("# test")
    os.symlink("test_dir/subdir", "test_dir/symlink")
    config = Config(follow_links=False)
    assert set(find(["test_dir"], config, [], [])) == {"test_dir/subdir/test_file.py"}
    os.remove("test_dir/symlink")
    os.remove("test_dir/subdir/test_file.py")
    os.rmdir("test_dir/subdir")
    os.rmdir("test_dir")

def test_find_with_circular_symlink():
    os.makedirs("test_dir")
    os.makedirs("test_dir/subdir")
    with open("test_dir/subdir/test_file.py", "w") as f:
        f.write("# test")
    os.symlink("../test_dir", "test_dir/subdir/circular_link")
    config = Config(follow_links=True)
    assert set(find(["test_dir"], config, [], [])) == {"test_dir/subdir/test_file.py"}
    os.remove("test_dir/subdir/circular_link")
    os.remove("test_dir/subdir/test_file.py")
    os.rmdir("test_dir/subdir")
    os.rmdir("test_dir")


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    paths = ["nonexistent_file.py"]
    config = Config()
    skipped = []
    broken = []

    find_result = list(find(paths, config, skipped, broken))

    assert len(broken) == 1
    assert broken[0] == "nonexistent_file.py"


# LLM-generated content at query #16
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

def test_find_with_directory():
    os.makedirs("test_dir")
    with open("test_dir/file1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# test2")
    result = list(find(["test_dir"], Config(), [], []))
    assert set(result) == {"test_dir/file1.py", "test_dir/file2.py"}
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.py")
    os.rmdir("test_dir")

def test_find_with_skipped_directory():
    os.makedirs("skip_dir")
    with open("skip_dir/file.py", "w") as f:
        f.write("# skip")
    config = Config()
    config.skip_patterns = ["skip_dir"]
    skipped = []
    result = list(find(["skip_dir"], config, skipped, []))
    assert result == []
    assert skipped == [str(Path("skip_dir").resolve())]
    os.remove("skip_dir/file.py")
    os.rmdir("skip_dir")

def test_find_with_follow_links_disabled():
    os.makedirs("link_dir")
    with open("link_dir/file.py", "w") as f:
        f.write("# test")
    config = Config(follow_links=False)
    result = list(find(["link_dir"], config, [], []))
    assert result == ["link_dir/file.py"]
    os.remove("link_dir/file.py")
    os.rmdir("link_dir")

def test_find_with_mixed_paths():
    os.makedirs("mixed_dir")
    with open("mixed_dir/valid.py", "w") as f:
        f.write("# valid")
    with open("invalid.txt", "w") as f:
        f.write("# invalid")
    result = list(find(["mixed_dir", "invalid.txt"], Config(), [], []))
    assert result == ["mixed_dir/valid.py"]
    os.remove("mixed_dir/valid.py")
    os.rmdir("mixed_dir")
    os.remove("invalid.txt")


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


# LLM-generated content at query #18
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
    result = list(find(["test_file.py"], Config(), [], []))
    assert result == ["test_file.py"]

def test_find_with_skipped_file():
    config = Config()
    config.skip_patterns = ["skip_*"]
    skipped = []
    result = list(find(["skip_me.py"], config, skipped, []))
    assert result == []
    assert skipped == [os.path.abspath("skip_me.py")]

def test_find_with_directory():
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("")
    with open("test_dir/file2.txt", "w") as f:
        f.write("")
    result = list(find(["test_dir"], Config(), [], []))
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.txt" not in result
    shutil.rmtree("test_dir")

def test_find_with_skipped_directory():
    os.makedirs("test_dir/skip_dir", exist_ok=True)
    with open("test_dir/skip_dir/file.py", "w") as f:
        f.write("")
    config = Config()
    config.skip_patterns = ["skip_*"]
    skipped = []
    result = list(find(["test_dir"], config, skipped, []))
    assert result == []
    assert any("skip_dir" in s for s in skipped)
    shutil.rmtree("test_dir")

def test_find_with_symlink_loop():
    os.makedirs("dir1", exist_ok=True)
    os.makedirs("dir2", exist_ok=True)
    os.symlink("dir2", "dir1/link_to_dir2")
    os.symlink("dir1", "dir2/link_to_dir1")
    result = list(find(["dir1"], Config(), [], []))
    assert result == []
    shutil.rmtree("dir1")
    shutil.rmtree("dir2")


# LLM-generated content at query #19
#--------------------------

```python
def test_os_path_isdir_evaluates_to_true():
    path = "test_directory"
    assert os.path.isdir(path) is True


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_false():
    paths = ["file.txt"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["file.txt"]


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_true():
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch

    # Setup mock objects
    config = Mock()
    config.follow_links = False
    config.is_skipped.return_value = False

    # Create a temporary directory structure
    with patch('os.walk') as mock_walk:
        mock_walk.return_value = [
            ('/test_dir', ['subdir'], ['file1.py'])
        ]

        # Mock the directory and file structure
        paths = ['/test_dir']
        skipped = []
        broken = []

        # Call the function
        result = list(find(paths, config, skipped, broken))

        # Verify os.walk was called with the correct arguments
        mock_walk.assert_called_once_with(
            '/test_dir',
            topdown=True,
            followlinks=config.follow_links
        )


# LLM-generated content at query #22
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


# LLM-generated content at query #23
#--------------------------

```python
def test_find_with_directory_and_supported_files():
    paths = ["test_dir"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_dir/file1.py" in result
    assert "test_dir/subdir/file2.py" in result
    assert skipped == []
    assert broken == []

def test_find_with_skipped_directory():
    paths = ["test_dir"]
    config = Config(follow_links=False, skip_dirs=["subdir"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "test_dir/file1.py" in result
    assert "test_dir/subdir/file2.py" not in result
    assert skipped == ["test_dir/subdir"]
    assert broken == []

def test_find_with_broken_path():
    paths = ["nonexistent_path"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert skipped == []
    assert broken == ["nonexistent_path"]

def test_find_with_single_file():
    paths = ["test_dir/file1.py"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "test_dir/file1.py" in result
    assert skipped == []
    assert broken == []

def test_find_with_skipped_file():
    paths = ["test_dir"]
    config = Config(follow_links=False, skip_files=["file1.py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "test_dir/file1.py" not in result
    assert "test_dir/subdir/file2.py" in result
    assert skipped == ["test_dir/file1.py"]
    assert broken == []

def test_find_with_unsupported_filetype():
    paths = ["test_dir"]
    config = Config(follow_links=False, supported_filetypes=[".txt"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert skipped == []
    assert broken == []


# LLM-generated content at query #24
#--------------------------

```python
def test_find_with_empty_paths():
    assert list(find([], Config(), [], [])) == []

def test_find_with_nonexistent_path():
    broken = []
    assert list(find(["nonexistent_path"], Config(), [], broken)) == []
    assert broken == ["nonexistent_path"]

def test_find_with_single_file():
    with open("test_file.py", "w") as f:
        f.write("# test")
    assert list(find(["test_file.py"], Config(), [], [])) == ["test_file.py"]
    os.remove("test_file.py")

def test_find_with_directory_containing_python_files():
    os.makedirs("test_dir")
    with open("test_dir/file1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/file2.txt", "w") as f:
        f.write("# test2")
    assert sorted(find(["test_dir"], Config(), [], [])) == ["test_dir/file1.py"]
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.txt")
    os.rmdir("test_dir")

def test_find_with_skipped_directory():
    os.makedirs("test_dir/skip_me")
    with open("test_dir/file.py", "w") as f:
        f.write("# test")
    config = Config()
    config.skip = ["skip_me"]
    skipped = []
    assert list(find(["test_dir"], config, skipped, [])) == ["test_dir/file.py"]
    assert skipped == ["test_dir/skip_me"]
    os.remove("test_dir/file.py")
    os.rmdir("test_dir/skip_me")
    os.rmdir("test_dir")

def test_find_with_skipped_file():
    with open("skip_me.py", "w") as f:
        f.write("# test")
    config = Config()
    config.skip = ["skip_me.py"]
    skipped = []
    assert list(find(["skip_me.py"], config, skipped, [])) == []
    assert skipped == [os.path.abspath("skip_me.py")]
    os.remove("skip_me.py")

def test_find_with_symlink_loop():
    os.makedirs("test_dir")
    os.makedirs("test_dir/subdir")
    os.symlink("subdir", "test_dir/link")
    with open("test_dir/subdir/file.py", "w") as f:
        f.write("# test")
    config = Config(follow_links=True)
    assert list(find(["test_dir"], config, [], [])) == ["test_dir/subdir/file.py"]
    os.remove("test_dir/subdir/file.py")
    os.remove("test_dir/link")
    os.rmdir("test_dir/subdir")
    os.rmdir("test_dir")


# LLM-generated content at query #25
#--------------------------

```python
def test_os_path_isdir_returns_true():
    os.path.isdir.return_value = True
    assert os.path.isdir("some_path") is True


# LLM-generated content at query #26
#--------------------------

```python
def test_os_path_isdir_evaluates_to_true():
    path = "test_directory"
    assert os.path.isdir(path) is True


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    result = list(find(["test_file.py"], Config(), [], []))
    assert result == ["test_file.py"]

def test_find_with_skipped_file():
    config = Config()
    config.skipped_patterns = ["skip_*"]
    skipped = []
    result = list(find(["skip_me.py"], config, skipped, []))
    assert result == []
    assert skipped == [os.path.abspath("skip_me.py")]

def test_find_with_directory_and_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "file1.py"), "w") as f:
            f.write("")
        with open(os.path.join(tmpdir, "subdir", "file2.py"), "w") as f:
            f.write("")
        result = list(find([tmpdir], Config(), [], []))
        assert len(result) == 2
        assert os.path.join(tmpdir, "file1.py") in result
        assert os.path.join(tmpdir, "subdir", "file2.py") in result

def test_find_with_skipped_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "skip_dir"))
        with open(os.path.join(tmpdir, "skip_dir", "file.py"), "w") as f:
            f.write("")
        config = Config()
        config.skipped_patterns = ["skip_*"]
        skipped = []
        result = list(find([tmpdir], config, skipped, []))
        assert result == []
        assert any("skip_dir" in s for s in skipped)

def test_find_with_broken_symlink():
    with tempfile.TemporaryDirectory() as tmpdir:
        link_path = os.path.join(tmpdir, "broken_link")
        os.symlink("nonexistent", link_path)
        broken = []
        result = list(find([link_path], Config(), [], broken))
        assert result == []
        assert broken == [link_path]

def test_find_with_follow_links_disabled():
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "file.py"), "w") as f:
            f.write("")
        link_path = os.path.join(tmpdir, "link")
        os.symlink(subdir, link_path)
        config = Config(follow_links=False)
        result = list(find([tmpdir], config, [], []))
        assert len(result) == 1
        assert os.path.join(subdir, "file.py") in result

def test_find_with_unsupported_filetype():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "file.txt"), "w") as f:
            f.write("")
        result = list(find([tmpdir], Config(), [], []))
        assert result == []


# LLM-generated content at query #2
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
    paths = ["/nonexistent/path"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["/nonexistent/path"]

def test_find_with_single_file():
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

def test_find_with_directory_skipped_files():
    paths = ["test_dir"]
    config = Config()
    config.skip_patterns = ["skip_*"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_dir/skip_file.py" not in result
    assert "skip_file.py" in skipped
    assert "test_dir/valid_file.py" in result

def test_find_with_directory_broken_links():
    paths = ["test_dir"]
    config = Config()
    config.follow_links = False
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_dir/broken_link.py" not in result
    assert "broken_link.py" in broken
    assert "test_dir/valid_file.py" in result

def test_find_with_nested_directories():
    paths = ["test_dir"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_dir/subdir/file.py" in result
    assert "test_dir/file.py" in result

def test_find_with_visited_directories():
    paths = ["test_dir", "test_dir"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == len(set(result))


# LLM-generated content at query #3
#--------------------------

```python
def test_isdir_predicate_evaluates_to_true():
    import os
    from pathlib import Path
    from unittest.mock import Mock

    # Setup
    path = "test_dir"
    os.makedirs(path, exist_ok=True)

    # Test
    assert os.path.isdir(path)

    # Cleanup
    os.rmdir(path)


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    paths = ["nonexistent_file.py"]
    config = Config()
    skipped = []
    broken = []
    find(paths, config, skipped, broken)
    assert broken == ["nonexistent_file.py"]


# LLM-generated content at query #5
#--------------------------

```python
def test_find_with_directory_and_valid_files():
    paths = ["valid_directory"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["valid_directory/file1.py", "valid_directory/file2.py"]
    assert skipped == []
    assert broken == []

def test_find_with_directory_and_skipped_files():
    paths = ["directory_with_skipped"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["directory_with_skipped/allowed.py"]
    assert skipped == ["directory_with_skipped/skipped.py"]
    assert broken == []

def test_find_with_nonexistent_path():
    paths = ["nonexistent_path"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent_path"]

def test_find_with_single_file():
    paths = ["single_file.py"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["single_file.py"]
    assert skipped == []
    assert broken == []

def test_find_with_symlinks_and_follow_links_enabled():
    paths = ["directory_with_symlinks"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["directory_with_symlinks/file.py", "directory_with_symlinks/link_target.py"]
    assert skipped == []
    assert broken == []

def test_find_with_symlinks_and_follow_links_disabled():
    paths = ["directory_with_symlinks"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["directory_with_symlinks/file.py"]
    assert skipped == []
    assert broken == []

def test_find_with_circular_symlinks():
    paths = ["circular_symlink_dir"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["circular_symlink_dir/file.py"]
    assert skipped == []
    assert broken == []


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_evaluates_to_true():
    paths = ["nonexistent_path"]
    config = Config()
    skipped = []
    broken = []
    find(paths, config, skipped, broken)
    assert "nonexistent_path" in broken


# LLM-generated content at query #7
#--------------------------

```python
def test_find_with_directory_and_valid_files():
    paths = ["test_dir"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py", "test_dir/subdir/file2.py"]
    assert skipped == []
    assert broken == []

def test_find_with_directory_and_skipped_files():
    paths = ["test_dir"]
    config = Config(follow_links=False, skip_patterns=["*skip*"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py"]
    assert skipped == ["test_dir/skip_file.py"]
    assert broken == []

def test_find_with_nonexistent_path():
    paths = ["nonexistent_path"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent_path"]

def test_find_with_file_path():
    paths = ["test_file.py"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

def test_find_with_symlink_and_follow_links_enabled():
    paths = ["symlink_dir"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["symlink_dir/file1.py", "symlink_dir/subdir/file2.py"]
    assert skipped == []
    assert broken == []

def test_find_with_symlink_and_follow_links_disabled():
    paths = ["symlink_dir"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_evaluates_to_true():
    paths = ["nonexistent_path"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(broken) == 1
    assert broken[0] == "nonexistent_path"


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    paths = ["nonexistent_path.txt"]
    config = Config()
    skipped = []
    broken = []
    find(paths, config, skipped, broken)
    assert len(broken) == 1
    assert broken[0] == "nonexistent_path.txt"


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    broken = []
    os.path.exists = lambda path: False
    find(["non_existent_file.py"], Config(), [], broken)
    assert broken == ["non_existent_file.py"]


# LLM-generated content at query #11
#--------------------------

```python
def test_find_with_directory_and_valid_files():
    paths = ["valid_dir"]
    config = Config(follow_links=False, skipped_patterns=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "valid_dir/file1.py" in result
    assert "valid_dir/file2.py" in result
    assert skipped == []
    assert broken == []

def test_find_with_directory_and_skipped_files():
    paths = ["dir_with_skipped"]
    config = Config(follow_links=False, skipped_patterns=["skip_*"], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "dir_with_skipped/keep.py" in result
    assert skipped == ["dir_with_skipped/skip_me.py"]
    assert broken == []

def test_find_with_nonexistent_path():
    paths = ["nonexistent_path"]
    config = Config(follow_links=False, skipped_patterns=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent_path"]

def test_find_with_file_directly():
    paths = ["single_file.py"]
    config = Config(follow_links=False, skipped_patterns=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["single_file.py"]
    assert skipped == []
    assert broken == []

def test_find_with_symlink_loop():
    paths = ["symlink_dir"]
    config = Config(follow_links=True, skipped_patterns=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "symlink_dir/real_file.py" in result
    assert skipped == []
    assert broken == []


# LLM-generated content at query #12
#--------------------------

```python
def test_find_with_directory_and_valid_files():
    paths = ["valid_dir"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["valid_dir/file1.py", "valid_dir/file2.py"]
    assert skipped == []
    assert broken == []

def test_find_with_skipped_directory():
    paths = ["dir_with_skipped"]
    config = Config(follow_links=False, skipped_dirs=["dir_with_skipped/skip_me"], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["dir_with_skipped/file.py"]
    assert skipped == ["dir_with_skipped/skip_me"]
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
    paths = ["file.py"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["file.py"]
    assert skipped == []
    assert broken == []

def test_find_with_unsupported_filetype():
    paths = ["dir_with_unsupported"]
    config = Config(follow_links=False, skipped_dirs=[], supported_extensions=[".py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []


# LLM-generated content at query #13
#--------------------------

```python
def test_isdir_predicate_evaluates_to_true():
    import os
    from pathlib import Path

    # Create a temporary directory for testing
    test_dir = Path("test_directory")
    test_dir.mkdir(exist_ok=True)

    # Verify that os.path.isdir evaluates to True for the created directory
    assert os.path.isdir(str(test_dir)) is True

    # Clean up
    test_dir.rmdir()


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_evaluates_to_true():
    paths = ["nonexistent_path"]
    config = Config()
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))

    assert len(broken) == 1
    assert broken[0] == "nonexistent_path"


# LLM-generated content at query #15
#--------------------------

```python
def test_find_with_empty_paths():
    result = list(find([], Config(), [], []))
    assert result == []

def test_find_with_non_existent_path():
    broken = []
    result = list(find(["/nonexistent/path"], Config(), [], broken))
    assert result == []
    assert broken == ["/nonexistent/path"]

def test_find_with_single_file():
    result = list(find(["test_file.py"], Config(), [], []))
    assert result == ["test_file.py"]

def test_find_with_skipped_file():
    config = Config()
    config.add_skip_pattern("skip_*")
    skipped = []
    result = list(find(["skip_me.py"], config, skipped, []))
    assert result == []
    assert skipped == [os.path.abspath("skip_me.py")]

def test_find_with_directory():
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("# test")
    result = list(find(["test_dir"], Config(), [], []))
    assert "test_dir/file1.py" in result
    shutil.rmtree("test_dir")

def test_find_with_skipped_directory():
    os.makedirs("test_dir/skip_dir", exist_ok=True)
    config = Config()
    config.add_skip_pattern("skip_*")
    skipped = []
    result = list(find(["test_dir"], config, skipped, []))
    assert "skip_dir" not in str(result)
    assert any("skip_dir" in s for s in skipped)
    shutil.rmtree("test_dir")

def test_find_with_follow_links_disabled():
    os.makedirs("test_dir", exist_ok=True)
    os.makedirs("test_dir/link_target", exist_ok=True)
    os.symlink("test_dir/link_target", "test_dir/link")
    config = Config(follow_links=False)
    result = list(find(["test_dir"], config, [], []))
    assert any("link" not in s for s in result)
    shutil.rmtree("test_dir")

def test_find_with_broken_symlink():
    os.makedirs("test_dir", exist_ok=True)
    os.symlink("nonexistent", "test_dir/broken_link")
    broken = []
    result = list(find(["test_dir"], config, [], broken))
    assert "broken_link" not in str(result)
    assert any("broken_link" in s for s in broken)
    shutil.rmtree("test_dir")


# LLM-generated content at query #16
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

def test_find_with_follow_links_enabled():
    paths = ["test_dir"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py", "test_dir/file2.py", "test_dir/link_file.py"]
    assert skipped == []
    assert broken == []

def test_find_with_follow_links_disabled():
    paths = ["test_dir"]
    config = Config(follow_links=False)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py", "test_dir/file2.py"]
    assert skipped == []
    assert broken == []


# LLM-generated content at query #17
#--------------------------

```python
def test_isdir_predicate():
    import os
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    assert os.path.isdir(test_dir)


