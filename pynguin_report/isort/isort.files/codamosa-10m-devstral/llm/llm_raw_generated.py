####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_find():
    # Test case 1: Single file path
    config = Config()
    paths = ["test_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 2: Directory with Python files
    config = Config()
    paths = ["test_dir"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) > 0
    assert all(file.endswith(".py") for file in result)
    assert skipped == []
    assert broken == []

    # Test case 3: Non-existent path
    config = Config()
    paths = ["non_existent_path.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path.py"]

    # Test case 4: Skipped file
    config = Config(skip=["skip_this.py"])
    paths = ["skip_this.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ["skip_this.py"]
    assert broken == []

    # Test case 5: Mixed paths (files and directories)
    config = Config()
    paths = ["test_file.py", "test_dir"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_file.py" in result
    assert len(result) > 1
    assert skipped == []
    assert broken == []


# LLM-generated content at query #2
#--------------------------

```python
def test_find():
    # Test case 1: Test with a single file path
    config = Config()
    paths = ["test_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 2: Test with a non-existent file path
    config = Config()
    paths = ["non_existent_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_file.py"]

    # Test case 3: Test with a directory containing Python files
    config = Config()
    paths = ["test_directory"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_directory/file1.py", "test_directory/file2.py"]
    assert skipped == []
    assert broken == []

    # Test case 4: Test with a directory containing skipped files
    config = Config(skip=["test_directory/skip_file.py"])
    paths = ["test_directory"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_directory/file1.py", "test_directory/file2.py"]
    assert skipped == ["test_directory/skip_file.py"]
    assert broken == []

    # Test case 5: Test with a directory containing symbolic links
    config = Config(follow_links=True)
    paths = ["test_directory_with_links"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_directory_with_links/file1.py", "test_directory_with_links/link_to_file.py"]
    assert skipped == []
    assert broken == []


# LLM-generated content at query #3
#--------------------------

```python
def test_find(tmp_path):
    # Setup test files and directories
    test_file = tmp_path / "test.py"
    test_file.write_text("# test content")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    subdir_file = subdir / "subfile.py"
    subdir_file.write_text("# subfile content")
    skipped_dir = tmp_path / "skipped_dir"
    skipped_dir.mkdir()
    skipped_file = skipped_dir / "skipped.py"
    skipped_file.write_text("# skipped content")
    non_py_file = tmp_path / "readme.txt"
    non_py_file.write_text("# not python")

    # Create config
    config = Config()
    config.skipped_paths = [str(skipped_dir)]

    # Test case 1: Find files in directory
    paths = [str(tmp_path)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert str(test_file) in result
    assert str(subdir_file) in result
    assert len(skipped) == 1
    assert str(skipped_file) in skipped[0]
    assert len(broken) == 0

    # Test case 2: Non-existent path
    paths = ["nonexistent/path"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert "nonexistent/path" in broken

    # Test case 3: Direct file path
    paths = [str(test_file)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert str(test_file) in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test case 4: Non-python file
    paths = [str(non_py_file)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 0


# LLM-generated content at query #4
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    config = Config()
    skipped = []
    broken = []
    result = list(find([], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(b"print('hello')")
    try:
        config = Config()
        skipped = []
        broken = []
        result = list(find([tmp_path], config, skipped, broken))
        assert result == [tmp_path]
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(tmp_path)

    # Test case 3: Directory with Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some Python files
        py_file1 = os.path.join(tmpdir, "test1.py")
        py_file2 = os.path.join(tmpdir, "test2.py")
        with open(py_file1, "w") as f:
            f.write("print('test1')")
        with open(py_file2, "w") as f:
            f.write("print('test2')")

        # Create a non-Python file
        non_py_file = os.path.join(tmpdir, "test.txt")
        with open(non_py_file, "w") as f:
            f.write("not python")

        config = Config()
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert set(result) == {py_file1, py_file2}
        assert skipped == []
        assert broken == []

    # Test case 4: Non-existent path
    config = Config()
    skipped = []
    broken = []
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["/nonexistent/path"]

    # Test case 5: Skipped files
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = os.path.join(tmpdir, "test.py")
        with open(py_file, "w") as f:
            f.write("print('test')")

        config = Config(skip=["test.py"])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert result == []
        assert skipped == [os.path.abspath(py_file)]
        assert broken == []

    # Test case 6: Mixed paths (files and directories)
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = os.path.join(tmpdir, "test.py")
        with open(py_file, "w") as f:
            f.write("print('test')")

        config = Config()
        skipped = []
        broken = []
        result = list(find([tmpdir, py_file], config, skipped, broken))
        assert py_file in result
        assert skipped == []
        assert broken == []


# LLM-generated content at query #5
#--------------------------

```python
def test_find():
    # Test case 1: Test with a directory containing Python files
    config = Config()
    paths = ["test_directory"]
    skipped = []
    broken = []
    # Assuming test_directory has files: test1.py, test2.py, and a subdirectory subdir with test3.py
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 3
    assert "test_directory/test1.py" in result
    assert "test_directory/test2.py" in result
    assert "test_directory/subdir/test3.py" in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test case 2: Test with a file path
    paths = ["test_directory/test1.py"]
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "test_directory/test1.py" in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test case 3: Test with a non-existent path
    paths = ["non_existent_path"]
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert "non_existent_path" in broken

    # Test case 4: Test with a skipped directory
    config = Config(skip=["test_directory/subdir"])
    paths = ["test_directory"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_directory/test1.py" in result
    assert "test_directory/test2.py" in result
    assert len(skipped) == 1
    assert "test_directory/subdir" in skipped[0]
    assert len(broken) == 0

    # Test case 5: Test with a skipped file
    config = Config(skip=["test_directory/test1.py"])
    paths = ["test_directory"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_directory/test2.py" in result
    assert "test_directory/subdir/test3.py" in result
    assert len(skipped) == 1
    assert "test_directory/test1.py" in skipped[0]
    assert len(broken) == 0


# LLM-generated content at query #6
#--------------------------

```python
def test_find():
    # Test case 1: Single file path
    config = Config()
    paths = ["test_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert result[0] == "test_file.py"
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test case 2: Directory with Python files
    config = Config()
    paths = ["test_dir"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) >= 1  # At least one Python file in test_dir
    assert all(".py" in file for file in result)
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test case 3: Non-existent path
    config = Config()
    paths = ["non_existent_path"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "non_existent_path"

    # Test case 4: Skipped file
    config = Config(skip=["skip_me.py"])
    paths = ["test_dir"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "skip_me.py" in skipped
    assert "skip_me.py" not in result

    # Test case 5: Mixed paths (files and directories)
    config = Config()
    paths = ["test_file.py", "test_dir"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_file.py" in result
    assert any(".py" in file for file in result)
    assert len(skipped) == 0
    assert len(broken) == 0


# LLM-generated content at query #7
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(b"print('hello')")
    try:
        paths = [tmp_path]
        config = Config()
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert result == [tmp_path]
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(tmp_path)

    # Test case 3: Non-existent path
    paths = ["nonexistent_path.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent_path.py"]

    # Test case 4: Directory with Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some Python files
        py_file1 = os.path.join(tmpdir, "file1.py")
        py_file2 = os.path.join(tmpdir, "file2.py")
        with open(py_file1, "w") as f:
            f.write("print('file1')")
        with open(py_file2, "w") as f:
            f.write("print('file2')")

        # Create a non-Python file
        non_py_file = os.path.join(tmpdir, "file.txt")
        with open(non_py_file, "w") as f:
            f.write("text file")

        paths = [tmpdir]
        config = Config()
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert set(result) == {py_file1, py_file2}
        assert skipped == []
        assert broken == []

    # Test case 5: Skipped files and directories
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory structure
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        py_file = os.path.join(subdir, "file.py")
        with open(py_file, "w") as f:
            f.write("print('file')")

        # Create a skipped directory
        skipped_dir = os.path.join(tmpdir, "skipped_dir")
        os.makedirs(skipped_dir)
        skipped_file = os.path.join(skipped_dir, "skipped.py")
        with open(skipped_file, "w") as f:
            f.write("print('skipped')")

        # Configure to skip the skipped_dir
        config = Config(skip=["skipped_dir"])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert result == [py_file]
        assert skipped_dir in skipped[0]
        assert broken == []

    # Test case 6: Symlinks (if supported by the system)
    if hasattr(os, "symlink"):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a real directory and a symlink to it
            real_dir = os.path.join(tmpdir, "real_dir")
            os.makedirs(real_dir)
            symlink_dir = os.path.join(tmpdir, "symlink_dir")
            os.symlink(real_dir, symlink_dir)

            # Create a Python file in the real directory
            py_file = os.path.join(real_dir, "file.py")
            with open(py_file, "w") as f:
                f.write("print('file')")

            # Test with followlinks=True
            config = Config(follow_links=True)
            skipped = []
            broken = []
            result = list(find([symlink_dir], config, skipped, broken))
            assert result == [py_file]
            assert skipped == []
            assert broken == []

            # Test with followlinks=False
            config = Config(follow_links=False)
            skipped = []
            broken = []
            result = list(find([symlink_dir], config, skipped, broken))
            assert result == []
            assert skipped == []
            assert broken == []


# LLM-generated content at query #8
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    with open(os.path.join(test_dir, "test.py"), "w") as f:
        f.write("# test")

    # Test with non-existent path
    non_existent_path = "non_existent.py"

    # Test with file path
    test_file = "test_file.py"
    with open(test_file, "w") as f:
        f.write("# test file")

    # Execute
    result = list(find([test_dir, non_existent_path, test_file], config, skipped, broken))

    # Verify
    assert len(result) == 2
    assert os.path.join(test_dir, "test.py") in result
    assert test_file in result
    assert non_existent_path in broken
    assert len(skipped) == 0

    # Cleanup
    os.remove(os.path.join(test_dir, "test.py"))
    os.rmdir(test_dir)
    os.remove(test_file)


# LLM-generated content at query #9
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/test.py", "w") as f:
        f.write("# test")
    with open(f"{test_dir}/test2.py", "w") as f:
        f.write("# test2")
    with open(f"{test_dir}/notpython.txt", "w") as f:
        f.write("not python")

    # Test with non-existent path
    non_existent_path = "non_existent_path.py"

    # Test with single file
    single_file = "single_file.py"
    with open(single_file, "w") as f:
        f.write("# single file")

    # Execute
    result = list(find([test_dir, non_existent_path, single_file], config, skipped, broken))

    # Assert
    assert len(result) == 3
    assert f"{test_dir}/test.py" in result
    assert f"{test_dir}/test2.py" in result
    assert single_file in result
    assert non_existent_path in broken
    assert len(skipped) == 0

    # Cleanup
    os.remove(f"{test_dir}/test.py")
    os.remove(f"{test_dir}/test2.py")
    os.remove(f"{test_dir}/notpython.txt")
    os.rmdir(test_dir)
    os.remove(single_file)


# LLM-generated content at query #10
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with a directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("# not python")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test")

        # Test find
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "/nonexistent/path"

    # Test with a single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# test")
        tmpfile.flush()

        result = list(find([tmpfile.name], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmpfile.name
        assert len(skipped) == 0
        assert len(broken) == 0

        os.unlink(tmpfile.name)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        skip_dir = os.path.join(tmpdir, "skip_me")
        os.makedirs(skip_dir)
        with open(os.path.join(skip_dir, "test.py"), "w") as f:
            f.write("# should be skipped")

        config.skip = [skip_dir]
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 0
        assert len(skipped) == 1
        assert skipped[0] == skip_dir
        assert len(broken) == 0


# LLM-generated content at query #11
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    config = Config()
    skipped = []
    broken = []
    result = list(find([], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(b"print('test')")
    try:
        config = Config()
        skipped = []
        broken = []
        result = list(find([tmp_path], config, skipped, broken))
        assert result == [tmp_path]
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(tmp_path)

    # Test case 3: Directory with Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        py_file = os.path.join(tmpdir, "test.py")
        with open(py_file, "w") as f:
            f.write("print('test')")
        non_py_file = os.path.join(tmpdir, "test.txt")
        with open(non_py_file, "w") as f:
            f.write("not python")

        config = Config()
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 1
        assert py_file in result
        assert non_py_file not in result
        assert skipped == []
        assert broken == []

    # Test case 4: Non-existent path
    config = Config()
    skipped = []
    broken = []
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["/nonexistent/path"]

    # Test case 5: Skipped file
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = os.path.join(tmpdir, "test.py")
        with open(py_file, "w") as f:
            f.write("print('test')")

        config = Config(skip=["test.py"])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert result == []
        assert skipped == [os.path.abspath(py_file)]
        assert broken == []

    # Test case 6: Multiple paths with mixed files and directories
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        # Create files in first directory
        py_file1 = os.path.join(tmpdir1, "test1.py")
        with open(py_file1, "w") as f:
            f.write("print('test1')")

        # Create files in second directory
        py_file2 = os.path.join(tmpdir2, "test2.py")
        with open(py_file2, "w") as f:
            f.write("print('test2')")

        # Create a standalone file
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
            standalone_file = tmp.name
            tmp.write(b"print('standalone')")

        try:
            config = Config()
            skipped = []
            broken = []
            result = list(find([tmpdir1, tmpdir2, standalone_file], config, skipped, broken))
            assert len(result) == 3
            assert py_file1 in result
            assert py_file2 in result
            assert standalone_file in result
            assert skipped == []
            assert broken == []
        finally:
            os.unlink(standalone_file)


# LLM-generated content at query #12
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    paths = ["test_dir"]
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/test.py", "w") as f:
        f.write("print('test')")

    # Test with non-existent path
    paths.append("non_existent_path.py")

    # Test with single file
    paths.append("single_file.py")
    with open("single_file.py", "w") as f:
        f.write("print('single')")

    # Execute
    result = list(find(paths, config, skipped, broken))

    # Verify
    assert "test_dir/test.py" in result
    assert "single_file.py" in result
    assert "non_existent_path.py" in broken
    assert len(skipped) == 0

    # Cleanup
    os.remove("test_dir/test.py")
    os.rmdir("test_dir")
    os.remove("single_file.py")


# LLM-generated content at query #13
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    with open("test_file.py", "w") as f:
        f.write("# test")
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []
    os.remove("test_file.py")

    # Test case 3: Directory with Python files
    os.makedirs("test_dir")
    with open("test_dir/file1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# test2")
    with open("test_dir/non_py.txt", "w") as f:
        f.write("text")
    paths = ["test_dir"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.py" in result
    assert skipped == []
    assert broken == []
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.py")
    os.remove("test_dir/non_py.txt")
    os.rmdir("test_dir")

    # Test case 4: Non-existent path
    paths = ["non_existent_path.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path.py"]

    # Test case 5: Skipped file
    os.makedirs("test_skip_dir")
    with open("test_skip_dir/skip_me.py", "w") as f:
        f.write("# skip")
    paths = ["test_skip_dir"]
    config = Config(skip=["skip_me.py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath("test_skip_dir/skip_me.py")]
    assert broken == []
    os.remove("test_skip_dir/skip_me.py")
    os.rmdir("test_skip_dir")

    # Test case 6: Mixed paths (files and directories)
    os.makedirs("test_mixed_dir")
    with open("test_mixed_dir/mixed.py", "w") as f:
        f.write("# mixed")
    with open("single_file.py", "w") as f:
        f.write("# single")
    paths = ["test_mixed_dir", "single_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_mixed_dir/mixed.py" in result
    assert "single_file.py" in result
    assert skipped == []
    assert broken == []
    os.remove("test_mixed_dir/mixed.py")
    os.rmdir("test_mixed_dir")
    os.remove("single_file.py")


# LLM-generated content at query #14
#--------------------------

```python
def test_find(tmp_path):
    # Setup test files and directories
    test_file = tmp_path / "test.py"
    test_file.write_text("# test")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    subdir_file = subdir / "subfile.py"
    subdir_file.write_text("# subfile")
    skipped_dir = tmp_path / "skipped"
    skipped_dir.mkdir()
    skipped_file = skipped_dir / "skipped.py"
    skipped_file.write_text("# skipped")
    broken_path = tmp_path / "nonexistent.py"

    # Create config
    config = Config()
    config.skipped_paths = [str(skipped_dir)]

    # Test case 1: Find files in directory
    paths = [str(tmp_path)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert str(test_file) in result
    assert str(subdir_file) in result
    assert len(skipped) == 1
    assert str(skipped_dir) in skipped[0]
    assert len(broken) == 0

    # Test case 2: Find single file
    paths = [str(test_file)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert str(test_file) in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test case 3: Non-existent path
    paths = [str(broken_path)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert str(broken_path) in broken

    # Test case 4: Mixed paths
    paths = [str(tmp_path), str(test_file), str(broken_path)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 3
    assert str(test_file) in result
    assert str(subdir_file) in result
    assert len(skipped) == 1
    assert str(skipped_dir) in skipped[0]
    assert len(broken) == 1
    assert str(broken_path) in broken


# LLM-generated content at query #15
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(b"print('hello')")
    try:
        paths = [tmp_path]
        config = Config()
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert result == [tmp_path]
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(tmp_path)

    # Test case 3: Directory with Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some Python files
        py_file1 = os.path.join(tmpdir, "file1.py")
        py_file2 = os.path.join(tmpdir, "file2.py")
        with open(py_file1, "w") as f:
            f.write("print('file1')")
        with open(py_file2, "w") as f:
            f.write("print('file2')")

        # Create a non-Python file
        txt_file = os.path.join(tmpdir, "file.txt")
        with open(txt_file, "w") as f:
            f.write("text file")

        paths = [tmpdir]
        config = Config()
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert sorted(result) == sorted([py_file1, py_file2])
        assert skipped == []
        assert broken == []

    # Test case 4: Non-existent path
    paths = ["/nonexistent/path"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["/nonexistent/path"]

    # Test case 5: Skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory structure
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        py_file = os.path.join(subdir, "file.py")
        with open(py_file, "w") as f:
            f.write("print('in subdir')")

        # Create a config that skips the subdir
        config = Config(skip=["subdir"])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert result == []
        assert skipped == [os.path.join(tmpdir, "subdir")]
        assert broken == []

    # Test case 6: Symlink handling
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory and a Python file
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        py_file = os.path.join(subdir, "file.py")
        with open(py_file, "w") as f:
            f.write("print('in subdir')")

        # Create a symlink to the subdir
        symlink = os.path.join(tmpdir, "symlink")
        os.symlink(subdir, symlink)

        # Test with follow_links=True
        config = Config(follow_links=True)
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert sorted(result) == sorted([py_file])

        # Test with follow_links=False
        config = Config(follow_links=False)
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert sorted(result) == sorted([py_file])


# LLM-generated content at query #16
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    paths = ["test_dir", "nonexistent_file.py", "single_file.py"]
    skipped = []
    broken = []

    # Create test directory structure
    os.makedirs("test_dir/subdir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("# test")
    with open("test_dir/subdir/file2.py", "w") as f:
        f.write("# test")
    with open("single_file.py", "w") as f:
        f.write("# test")

    # Test
    result = list(find(paths, config, skipped, broken))

    # Assertions
    assert len(result) == 3
    assert "test_dir/file1.py" in result
    assert "test_dir/subdir/file2.py" in result
    assert "single_file.py" in result
    assert len(skipped) == 0
    assert len(broken) == 1
    assert "nonexistent_file.py" in broken

    # Cleanup
    os.remove("test_dir/file1.py")
    os.remove("test_dir/subdir/file2.py")
    os.rmdir("test_dir/subdir")
    os.rmdir("test_dir")
    os.remove("single_file.py")


# LLM-generated content at query #17
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    with open("test_file.py", "w") as f:
        f.write("# test")
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []
    os.remove("test_file.py")

    # Test case 3: Directory with Python files
    os.makedirs("test_dir")
    with open("test_dir/file1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# test2")
    with open("test_dir/non_py.txt", "w") as f:
        f.write("text")
    paths = ["test_dir"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.py" in result
    assert skipped == []
    assert broken == []
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.py")
    os.remove("test_dir/non_py.txt")
    os.rmdir("test_dir")

    # Test case 4: Non-existent path
    paths = ["non_existent_path.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path.py"]

    # Test case 5: Skipped file
    os.makedirs("test_skip_dir")
    with open("test_skip_dir/skip_me.py", "w") as f:
        f.write("# skip")
    paths = ["test_skip_dir"]
    config = Config(skip=["skip_me.py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath("test_skip_dir/skip_me.py")]
    assert broken == []
    os.remove("test_skip_dir/skip_me.py")
    os.rmdir("test_skip_dir")

    # Test case 6: Mixed paths (files and directories)
    os.makedirs("test_mixed_dir")
    with open("test_mixed_dir/mixed.py", "w") as f:
        f.write("# mixed")
    with open("mixed_file.py", "w") as f:
        f.write("# mixed file")
    paths = ["test_mixed_dir", "mixed_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_mixed_dir/mixed.py" in result
    assert "mixed_file.py" in result
    assert skipped == []
    assert broken == []
    os.remove("test_mixed_dir/mixed.py")
    os.rmdir("test_mixed_dir")
    os.remove("mixed_file.py")


# LLM-generated content at query #18
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("not python")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test")

        # Test finding files
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    skipped.clear()
    broken.clear()
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "/nonexistent/path"

    # Test with single file
    skipped.clear()
    broken.clear()
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(b"# test")
        tmp_path = tmp.name

    try:
        result = list(find([tmp_path], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmp_path
        assert len(skipped) == 0
        assert len(broken) == 0
    finally:
        os.unlink(tmp_path)

    # Test with skipped directory
    skipped.clear()
    broken.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test structure
        os.makedirs(os.path.join(tmpdir, "skipme"))
        with open(os.path.join(tmpdir, "skipme", "test.py"), "w") as f:
            f.write("# should be skipped")

        config.skip = ["skipme"]
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 0
        assert len(skipped) == 1
        assert "skipme" in skipped[0]
        assert len(broken) == 0


# LLM-generated content at query #19
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 3: Non-existent file path
    paths = ["non_existent_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_file.py"]

    # Test case 4: Directory with Python files
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test")

        paths = [tmpdir]
        config = Config()
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert len(result) == 2
        assert os.path.join(tmpdir, "test1.py") in result
        assert os.path.join(tmpdir, "subdir", "test3.py") in result
        assert skipped == []
        assert broken == []

    # Test case 5: Skipped files and directories
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test files and directories
        os.makedirs(os.path.join(tmpdir, "skipdir"))
        with open(os.path.join(tmpdir, "skipfile.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "normal.py"), "w") as f:
            f.write("# test")

        paths = [tmpdir]
        config = Config(skip=["skipfile.py", "skipdir"])
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert result == [os.path.join(tmpdir, "normal.py")]
        assert len(skipped) == 2
        assert os.path.join(tmpdir, "skipfile.py") in skipped
        assert os.path.join(tmpdir, "skipdir") in skipped
        assert broken == []

    # Test case 6: Broken symlink (if applicable)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a broken symlink
        symlink_path = os.path.join(tmpdir, "broken_link")
        os.symlink("non_existent_target", symlink_path)

        paths = [symlink_path]
        config = Config(follow_links=True)
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert result == []
        assert skipped == []
        assert broken == [symlink_path]


# LLM-generated content at query #20
#--------------------------

```python
def test_find():
    # Test with a directory containing Python files
    config = Config()
    paths = ["test_dir"]
    skipped = []
    broken = []

    # Create a test directory with some Python files
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/test1.py", "w") as f:
        f.write("# test file 1")
    with open("test_dir/test2.py", "w") as f:
        f.write("# test file 2")
    with open("test_dir/skip_me.py", "w") as f:
        f.write("# skipped file")

    # Configure to skip skip_me.py
    config.skip = ["skip_me.py"]

    # Test the find function
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_dir/test1.py" in result
    assert "test_dir/test2.py" in result
    assert "test_dir/skip_me.py" not in result
    assert "test_dir/skip_me.py" in skipped
    assert len(broken) == 0

    # Clean up
    os.remove("test_dir/test1.py")
    os.remove("test_dir/test2.py")
    os.remove("test_dir/skip_me.py")
    os.rmdir("test_dir")

    # Test with a non-existent path
    paths = ["non_existent_path"]
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert "non_existent_path" in broken

    # Test with a single file path
    with open("single_file.py", "w") as f:
        f.write("# single file")
    paths = ["single_file.py"]
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "single_file.py" in result
    os.remove("single_file.py")


# LLM-generated content at query #21
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "subdir", "test2.py"), "w") as f:
            f.write("# test2")
        with open(os.path.join(tmpdir, "notpython.txt"), "w") as f:
            f.write("not python")

        # Test
        result = list(find([tmpdir], config, skipped, broken))

        # Assert
        assert len(result) == 2
        assert any("test.py" in path for path in result)
        assert any("test2.py" in path for path in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    skipped = []
    broken = []
    result = list(find(["nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "nonexistent/path"

    # Test with single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# test")
        tmpfile.flush()

        skipped = []
        broken = []
        result = list(find([tmpfile.name], config, skipped, broken))

        assert len(result) == 1
        assert result[0] == tmpfile.name
        assert len(skipped) == 0
        assert len(broken) == 0

        os.unlink(tmpfile.name)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test structure
        os.makedirs(os.path.join(tmpdir, "skipme"))
        with open(os.path.join(tmpdir, "skipme", "test.py"), "w") as f:
            f.write("# should be skipped")

        config = Config(skip=["skipme"])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))

        assert len(result) == 0
        assert len(skipped) == 1
        assert "skipme" in skipped[0]
        assert len(broken) == 0


# LLM-generated content at query #22
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with a directory containing Python files
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/test1.py", "w") as f:
        f.write("# test file")
    with open(f"{test_dir}/test2.py", "w") as f:
        f.write("# another test file")
    with open(f"{test_dir}/notpython.txt", "w") as f:
        f.write("not python")

    # Test with a single file
    single_file = "single_test.py"
    with open(single_file, "w") as f:
        f.write("# single test file")

    # Test with a non-existent file
    non_existent = "non_existent.py"

    # Test with a skipped directory
    skipped_dir = f"{test_dir}/skipped_dir"
    os.makedirs(skipped_dir, exist_ok=True)
    with open(f"{skipped_dir}/skipped.py", "w") as f:
        f.write("# skipped file")
    config.skip.append(skipped_dir)

    # Test with a skipped file
    skipped_file = f"{test_dir}/skipped_file.py"
    with open(skipped_file, "w") as f:
        f.write("# skipped file")
    config.skip.append(skipped_file)

    # Execute
    paths = [test_dir, single_file, non_existent, skipped_dir, skipped_file]
    result = list(find(paths, config, skipped, broken))

    # Assert
    assert len(result) == 3  # test1.py, test2.py, single_test.py
    assert f"{test_dir}/test1.py" in result
    assert f"{test_dir}/test2.py" in result
    assert single_file in result
    assert skipped_file in skipped
    assert non_existent in broken

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    os.remove(single_file)


# LLM-generated content at query #23
#--------------------------

```python
def test_find(tmp_path):
    # Setup test files and directories
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "file1.py").write_text("# Python file")
    (test_dir / "file2.txt").write_text("Not a Python file")
    sub_dir = test_dir / "sub_dir"
    sub_dir.mkdir()
    (sub_dir / "file3.py").write_text("# Another Python file")
    (sub_dir / "file4.py").write_text("# Yet another Python file")

    # Test with directory path
    config = Config()
    skipped = []
    broken = []
    result = list(find([str(test_dir)], config, skipped, broken))
    assert len(result) == 3
    assert all("file" in r and ".py" in r for r in result)
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test with non-existent path
    skipped = []
    broken = []
    result = list(find([str(tmp_path / "nonexistent")], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == str(tmp_path / "nonexistent")

    # Test with file path
    skipped = []
    broken = []
    result = list(find([str(test_dir / "file1.py")], config, skipped, broken))
    assert len(result) == 1
    assert result[0] == str(test_dir / "file1.py")
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test with skipped directory
    config = Config(skip=["sub_dir"])
    skipped = []
    broken = []
    result = list(find([str(test_dir)], config, skipped, broken))
    assert len(result) == 1
    assert "file1.py" in result[0]
    assert len(skipped) == 1
    assert "sub_dir" in skipped[0]
    assert len(broken) == 0


# LLM-generated content at query #24
#--------------------------

```python
def test_find(tmp_path):
    # Setup test files and directories
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    sub_dir = test_dir / "sub_dir"
    sub_dir.mkdir()
    (test_dir / "file1.py").write_text("# Python file")
    (test_dir / "file2.txt").write_text("# Text file")
    (sub_dir / "file3.py").write_text("# Python file in subdir")
    (sub_dir / "file4.py").write_text("# Another Python file")

    # Create a skipped directory
    skipped_dir = test_dir / "skipped_dir"
    skipped_dir.mkdir()
    (skipped_dir / "file5.py").write_text("# Should be skipped")

    # Create a broken path
    broken_path = str(tmp_path / "nonexistent.py")

    # Setup config
    config = Config()
    config.skipped_paths = [str(skipped_dir)]
    config.follow_links = False

    # Call function
    paths = [str(test_dir), str(test_dir / "file1.py"), broken_path]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))

    # Assertions
    assert len(result) == 3
    assert str(test_dir / "file1.py") in result
    assert str(sub_dir / "file3.py") in result
    assert str(sub_dir / "file4.py") in result

    assert len(skipped) == 1
    assert str(skipped_dir) in skipped[0]

    assert len(broken) == 1
    assert broken_path in broken


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_find():
    # Test case 1: Test with a directory containing Python files
    config = Config()
    paths = ["test_dir"]
    skipped = []
    broken = []

    # Create a test directory with Python files
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/test1.py", "w") as f:
        f.write("print('test1')")
    with open("test_dir/test2.py", "w") as f:
        f.write("print('test2')")
    with open("test_dir/non_python.txt", "w") as f:
        f.write("not python")

    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_dir/test1.py" in result
    assert "test_dir/test2.py" in result
    assert "test_dir/non_python.txt" not in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Clean up
    os.remove("test_dir/test1.py")
    os.remove("test_dir/test2.py")
    os.remove("test_dir/non_python.txt")
    os.rmdir("test_dir")

    # Test case 2: Test with a non-existent path
    paths = ["non_existent_path"]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert "non_existent_path" in broken

    # Test case 3: Test with a single file
    paths = ["single_file.py"]
    skipped = []
    broken = []

    with open("single_file.py", "w") as f:
        f.write("print('single')")

    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "single_file.py" in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Clean up
    os.remove("single_file.py")

    # Test case 4: Test with a skipped directory
    config = Config(skip=["skip_dir"])
    paths = ["test_dir"]
    skipped = []
    broken = []

    os.makedirs("test_dir/skip_dir", exist_ok=True)
    with open("test_dir/skip_dir/skipped.py", "w") as f:
        f.write("print('skipped')")

    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 1
    assert "skip_dir" in skipped[0]
    assert len(broken) == 0

    # Clean up
    os.remove("test_dir/skip_dir/skipped.py")
    os.rmdir("test_dir/skip_dir")
    os.rmdir("test_dir")


# LLM-generated content at query #2
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "test1.py").write_text("# test")
        (Path(tmpdir) / "test2.py").write_text("# test")
        (Path(tmpdir) / "test.txt").write_text("# not python")
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        (subdir / "test3.py").write_text("# test")

        # Test
        result = list(find([tmpdir], config, skipped, broken))

        # Assert
        assert len(result) == 3
        assert all("test" in f for f in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    skipped = []
    broken = []
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "/nonexistent/path"

    # Test with single file
    skipped = []
    broken = []
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# test")
        tmpfile_path = tmpfile.name

    try:
        result = list(find([tmpfile_path], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmpfile_path
        assert len(skipped) == 0
        assert len(broken) == 0
    finally:
        os.unlink(tmpfile_path)

    # Test with skipped directory
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "test.py").write_text("# test")
        skip_dir = Path(tmpdir) / "skipdir"
        skip_dir.mkdir()
        (skip_dir / "test2.py").write_text("# test")

        # Configure to skip directory
        config.skip = ["skipdir"]

        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 1
        assert "test.py" in result[0]
        assert len(skipped) == 1
        assert "skipdir" in skipped[0]
        assert len(broken) == 0


# LLM-generated content at query #3
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("not python")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test")

        # Test
        result = list(find([tmpdir], config, skipped, broken))

        # Assert
        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    skipped = []
    broken = []
    result = list(find(["nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "nonexistent/path"

    # Test with single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# test")
        tmpfile.flush()

        skipped = []
        broken = []
        result = list(find([tmpfile.name], config, skipped, broken))

        assert len(result) == 1
        assert result[0] == tmpfile.name
        assert len(skipped) == 0
        assert len(broken) == 0

        os.unlink(tmpfile.name)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test structure
        os.makedirs(os.path.join(tmpdir, "skipme"))
        with open(os.path.join(tmpdir, "skipme", "test.py"), "w") as f:
            f.write("# test")

        # Configure to skip "skipme" directory
        config.skip = ["skipme"]

        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))

        assert len(result) == 0
        assert len(skipped) == 1
        assert "skipme" in skipped[0]
        assert len(broken) == 0


# LLM-generated content at query #4
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(b"print('hello')")
        tmp_path = tmp.name
    paths = [tmp_path]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == [tmp_path]
    assert skipped == []
    assert broken == []
    os.unlink(tmp_path)

    # Test case 3: Directory with Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some Python files
        py_file1 = os.path.join(tmpdir, "file1.py")
        py_file2 = os.path.join(tmpdir, "file2.py")
        with open(py_file1, "w") as f:
            f.write("print('hello')")
        with open(py_file2, "w") as f:
            f.write("print('world')")

        # Create a non-Python file
        non_py_file = os.path.join(tmpdir, "file.txt")
        with open(non_py_file, "w") as f:
            f.write("not python")

        paths = [tmpdir]
        config = Config()
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert set(result) == {py_file1, py_file2}
        assert skipped == []
        assert broken == []

    # Test case 4: Non-existent path
    paths = ["/nonexistent/path"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["/nonexistent/path"]

    # Test case 5: Skipped files and directories
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory structure
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        py_file = os.path.join(subdir, "file.py")
        with open(py_file, "w") as f:
            f.write("print('hello')")

        # Create a file to skip
        skip_file = os.path.join(tmpdir, "skip.py")
        with open(skip_file, "w") as f:
            f.write("print('skip')")

        paths = [tmpdir]
        config = Config(skip=["skip.py"], skip_glob=["*/subdir/*"])
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert result == []
        assert set(skipped) == {os.path.abspath(skip_file), os.path.abspath(subdir)}
        assert broken == []


# LLM-generated content at query #5
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    with open(os.path.join(test_dir, "test.py"), "w") as f:
        f.write("# test")

    # Test with non-existent path
    non_existent_path = "non_existent.py"

    # Test with single file
    single_file = "single_file.py"
    with open(single_file, "w") as f:
        f.write("# single file")

    # Execute
    result = list(find([test_dir, non_existent_path, single_file], config, skipped, broken))

    # Assert
    assert len(result) == 2
    assert "test_directory/test.py" in result
    assert single_file in result
    assert non_existent_path in broken
    assert len(skipped) == 0

    # Cleanup
    os.remove(os.path.join(test_dir, "test.py"))
    os.rmdir(test_dir)
    os.remove(single_file)


# LLM-generated content at query #6
#--------------------------

```python
def test_find(tmp_path):
    # Test case 1: Single file
    test_file = tmp_path / "test.py"
    test_file.write_text("# test")
    paths = [str(test_file)]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == [str(test_file)]
    assert skipped == []
    assert broken == []

    # Test case 2: Directory with files
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    test_file1 = test_dir / "file1.py"
    test_file1.write_text("# test1")
    test_file2 = test_dir / "file2.py"
    test_file2.write_text("# test2")
    paths = [str(test_dir)]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert str(test_file1) in result
    assert str(test_file2) in result
    assert skipped == []
    assert broken == []

    # Test case 3: Non-existent path
    paths = ["non_existent_path.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path.py"]

    # Test case 4: Skipped file
    test_file = tmp_path / "skipped.py"
    test_file.write_text("# skipped")
    config = Config(skip=["skipped.py"])
    skipped = []
    broken = []
    result = list(find([str(test_file)], config, skipped, broken))
    assert result == []
    assert skipped == [str(test_file)]
    assert broken == []

    # Test case 5: Mixed paths
    test_file1 = tmp_path / "file1.py"
    test_file1.write_text("# test1")
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    test_file2 = test_dir / "file2.py"
    test_file2.write_text("# test2")
    paths = [str(test_file1), str(test_dir)]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert str(test_file1) in result
    assert str(test_file2) in result
    assert skipped == []
    assert broken == []


# LLM-generated content at query #7
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "subdir", "test2.py"), "w") as f:
            f.write("# test2")
        with open(os.path.join(tmpdir, "notpython.txt"), "w") as f:
            f.write("text")

        # Test
        result = list(find([tmpdir], config, skipped, broken))

        # Assert
        assert len(result) == 2
        assert any("test.py" in path for path in result)
        assert any("test2.py" in path for path in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    skipped.clear()
    broken.clear()
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "/nonexistent/path"

    # Test with single file
    skipped.clear()
    broken.clear()
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(b"# test")
        tmp_path = tmp.name

    try:
        result = list(find([tmp_path], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmp_path
        assert len(skipped) == 0
        assert len(broken) == 0
    finally:
        os.unlink(tmp_path)

    # Test with skipped directory
    skipped.clear()
    broken.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        skip_dir = os.path.join(tmpdir, "skipme")
        os.makedirs(skip_dir)
        with open(os.path.join(skip_dir, "test.py"), "w") as f:
            f.write("# test")

        config.skip = [skip_dir]
        result = list(find([tmpdir], config, skipped, broken))

        assert len(result) == 0
        assert len(skipped) == 1
        assert skipped[0] == skip_dir
        assert len(broken) == 0


# LLM-generated content at query #8
#--------------------------

```python
def test_find(tmp_path):
    # Setup test files and directories
    test_file = tmp_path / "test.py"
    test_file.write_text("# test content")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    subdir_file = subdir / "subfile.py"
    subdir_file.write_text("# subfile content")
    skipped_dir = tmp_path / "skipped_dir"
    skipped_dir.mkdir()
    skipped_file = skipped_dir / "skipped.py"
    skipped_file.write_text("# skipped content")
    non_python_file = tmp_path / "readme.md"
    non_python_file.write_text("# not python")

    # Create config
    config = Config()
    config.skip = ["skipped_dir"]
    config.follow_links = False

    # Test case 1: Find files in directory
    paths = [str(tmp_path)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert str(test_file) in result
    assert str(subdir_file) in result
    assert len(skipped) == 1
    assert str(skipped_dir) in skipped[0]
    assert len(broken) == 0

    # Test case 2: Non-existent path
    paths = [str(tmp_path / "nonexistent")]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert str(tmp_path / "nonexistent") in broken

    # Test case 3: Direct file path
    paths = [str(test_file)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert str(test_file) in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test case 4: Skipped file
    config.skip = ["subfile.py"]
    paths = [str(tmp_path)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert str(test_file) in result
    assert len(skipped) == 1
    assert str(subdir_file) in skipped[0]
    assert len(broken) == 0


# LLM-generated content at query #9
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test case 1: Single file path
    paths = ["test_file.py"]
    with open("test_file.py", "w") as f:
        f.write("# test")
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []
    os.remove("test_file.py")

    # Test case 2: Directory with Python files
    os.makedirs("test_dir")
    with open("test_dir/file1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# test2")
    with open("test_dir/non_py.txt", "w") as f:
        f.write("text")
    result = list(find(["test_dir"], config, skipped, broken))
    assert len(result) == 2
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.py" in result
    assert skipped == []
    assert broken == []
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.py")
    os.remove("test_dir/non_py.txt")
    os.rmdir("test_dir")

    # Test case 3: Non-existent path
    result = list(find(["nonexistent_path.py"], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent_path.py"]

    # Test case 4: Skipped file
    config.skip = ["skip_me.py"]
    with open("skip_me.py", "w") as f:
        f.write("# should be skipped")
    result = list(find(["skip_me.py"], config, skipped, broken))
    assert result == []
    assert skipped == ["skip_me.py"]
    assert broken == []
    os.remove("skip_me.py")

    # Test case 5: Mixed paths (files and directories)
    os.makedirs("mixed_dir")
    with open("mixed_dir/included.py", "w") as f:
        f.write("# included")
    with open("single_file.py", "w") as f:
        f.write("# single")
    result = list(find(["mixed_dir", "single_file.py"], config, skipped, broken))
    assert len(result) == 2
    assert "mixed_dir/included.py" in result
    assert "single_file.py" in result
    os.remove("mixed_dir/included.py")
    os.rmdir("mixed_dir")
    os.remove("single_file.py")


# LLM-generated content at query #10
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with a directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test directory structure
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("# not python")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test in subdir")

        # Test finding files
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)
        assert not any("test2.txt" in r for r in result)

    # Test with non-existent path
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert "/nonexistent/path" in broken

    # Test with a single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# test")
        tmpfile_path = tmpfile.name

    try:
        result = list(find([tmpfile_path], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmpfile_path
    finally:
        os.unlink(tmpfile_path)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "skipme"))
        with open(os.path.join(tmpdir, "skipme", "test.py"), "w") as f:
            f.write("# should be skipped")

        config.skip = ["skipme"]
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 0
        assert any("skipme" in s for s in skipped)


# LLM-generated content at query #11
#--------------------------

```python
def test_find():
    # Test case 1: Single file path
    config = Config()
    paths = ["test_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 2: Directory with Python files
    config = Config()
    paths = ["test_dir"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.py" in result
    assert skipped == []
    assert broken == []

    # Test case 3: Non-existent path
    config = Config()
    paths = ["non_existent_path"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path"]

    # Test case 4: Skipped file
    config = Config(skip=["test_dir/skip_file.py"])
    paths = ["test_dir"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_dir/skip_file.py" not in result
    assert "test_dir/file1.py" in result
    assert skipped == ["test_dir/skip_file.py"]
    assert broken == []

    # Test case 5: Follow links
    config = Config(follow_links=True)
    paths = ["test_dir_with_links"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_dir_with_links/link_file.py" in result
    assert skipped == []
    assert broken == []


# LLM-generated content at query #12
#--------------------------

```python
def test_find(tmp_path):
    # Setup test files and directories
    test_file = tmp_path / "test.py"
    test_file.write_text("# test file")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    subdir_file = subdir / "subfile.py"
    subdir_file.write_text("# subdir file")
    skipped_dir = tmp_path / "skipped_dir"
    skipped_dir.mkdir()
    skipped_file = skipped_dir / "skipped.py"
    skipped_file.write_text("# skipped file")
    non_py_file = tmp_path / "readme.md"
    non_py_file.write_text("# not python")

    # Create config
    config = Config()
    config.skipped_paths = [str(skipped_dir)]
    config.follow_links = False

    # Test case 1: Find files in directory
    skipped = []
    broken = []
    result = list(find([str(tmp_path)], config, skipped, broken))
    assert len(result) == 2
    assert str(test_file) in result
    assert str(subdir_file) in result
    assert len(skipped) == 1
    assert str(skipped_dir) in skipped[0]
    assert len(broken) == 0

    # Test case 2: Non-existent path
    skipped = []
    broken = []
    result = list(find(["nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert "nonexistent/path" in broken

    # Test case 3: Direct file path
    skipped = []
    broken = []
    result = list(find([str(test_file)], config, skipped, broken))
    assert len(result) == 1
    assert str(test_file) in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test case 4: Skipped file
    skipped = []
    broken = []
    result = list(find([str(skipped_file)], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 1
    assert str(skipped_file) in skipped[0]
    assert len(broken) == 0

    # Test case 5: Non-python file
    skipped = []
    broken = []
    result = list(find([str(non_py_file)], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 0


# LLM-generated content at query #13
#--------------------------

```python
def test_find():
    # Test with a directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("not python")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test")

        config = Config()
        skipped = []
        broken = []

        # Test finding files in directory
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    skipped = []
    broken = []
    result = list(find(["nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "nonexistent/path"

    # Test with single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# test")
        tmpfile.flush()

        skipped = []
        broken = []
        result = list(find([tmpfile.name], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmpfile.name
        assert len(skipped) == 0
        assert len(broken) == 0

        os.unlink(tmpfile.name)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "skipme"))
        with open(os.path.join(tmpdir, "skipme", "test.py"), "w") as f:
            f.write("# test")

        config = Config(skip=["skipme"])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 0
        assert len(skipped) == 1
        assert "skipme" in skipped[0]
        assert len(broken) == 0


# LLM-generated content at query #14
#--------------------------

```python
def test_find():
    # Test case 1: Test with a directory containing Python files
    config = Config()
    paths = ["test_directory"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_directory/file1.py" in result
    assert "test_directory/file2.py" in result
    assert len(skipped) == 1
    assert "test_directory/skipped_dir" in skipped[0]
    assert len(broken) == 0

    # Test case 2: Test with a non-existent path
    paths = ["non_existent_path"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert "non_existent_path" in broken

    # Test case 3: Test with a single file
    paths = ["test_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "test_file.py" in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test case 4: Test with a directory containing non-Python files
    config = Config()
    paths = ["test_directory_non_python"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 0


# LLM-generated content at query #15
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    with open("test_file.py", "w") as f:
        f.write("# test")
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []
    os.remove("test_file.py")

    # Test case 3: Directory with Python files
    os.makedirs("test_dir")
    with open("test_dir/file1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# test2")
    with open("test_dir/non_py.txt", "w") as f:
        f.write("text")
    paths = ["test_dir"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.py" in result
    assert skipped == []
    assert broken == []
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.py")
    os.remove("test_dir/non_py.txt")
    os.rmdir("test_dir")

    # Test case 4: Non-existent path
    paths = ["non_existent_path.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path.py"]

    # Test case 5: Skipped file
    os.makedirs("test_skip_dir")
    with open("test_skip_dir/skip_me.py", "w") as f:
        f.write("# skip")
    paths = ["test_skip_dir"]
    config = Config(skip=["skip_me.py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath("test_skip_dir/skip_me.py")]
    assert broken == []
    os.remove("test_skip_dir/skip_me.py")
    os.rmdir("test_skip_dir")


# LLM-generated content at query #16
#--------------------------

```python
def test_find(tmp_path):
    # Setup test files and directories
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "file1.py").write_text("# Python file")
    (test_dir / "file2.txt").write_text("# Not a Python file")
    (test_dir / "subdir").mkdir()
    (test_dir / "subdir" / "file3.py").write_text("# Python file in subdir")
    (test_dir / "skipped_dir").mkdir()
    (test_dir / "skipped_dir" / "file4.py").write_text("# Python file in skipped dir")

    # Setup config
    config = Config()
    config.skip = ["skipped_dir"]
    config.follow_links = False

    # Test case 1: Normal directory traversal
    skipped = []
    broken = []
    result = list(find([str(test_dir)], config, skipped, broken))
    assert len(result) == 2
    assert str(test_dir / "file1.py") in result
    assert str(test_dir / "subdir" / "file3.py") in result
    assert len(skipped) == 1
    assert str(test_dir / "skipped_dir") in skipped
    assert len(broken) == 0

    # Test case 2: Non-existent path
    skipped = []
    broken = []
    result = list(find(["nonexistent_path"], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert "nonexistent_path" in broken

    # Test case 3: Direct file path
    skipped = []
    broken = []
    result = list(find([str(test_dir / "file1.py")], config, skipped, broken))
    assert len(result) == 1
    assert str(test_dir / "file1.py") in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test case 4: Mixed paths (files and directories)
    skipped = []
    broken = []
    result = list(find([str(test_dir), str(test_dir / "file1.py")], config, skipped, broken))
    assert len(result) == 3
    assert str(test_dir / "file1.py") in result
    assert str(test_dir / "subdir" / "file3.py") in result
    assert len(skipped) == 1
    assert str(test_dir / "skipped_dir") in skipped
    assert len(broken) == 0


# LLM-generated content at query #17
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test")

        # Test
        result = list(find([tmpdir], config, skipped, broken))

        # Assert
        assert len(result) == 2
        assert any("test1.py" in path for path in result)
        assert any("test3.py" in path for path in result)
        assert not any("test2.txt" in path for path in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    skipped = []
    broken = []
    result = list(find(["nonexistent_path"], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "nonexistent_path"

    # Test with single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# test")
        tmpfile_path = tmpfile.name

    try:
        skipped = []
        broken = []
        result = list(find([tmpfile_path], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmpfile_path
        assert len(skipped) == 0
        assert len(broken) == 0
    finally:
        os.unlink(tmpfile_path)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "skipme"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "skipme", "test2.py"), "w") as f:
            f.write("# test")

        # Configure to skip "skipme" directory
        config.skip = ["skipme"]

        # Test
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 1
        assert "test1.py" in result[0]
        assert len(skipped) == 1
        assert "skipme" in skipped[0]
        assert len(broken) == 0


# LLM-generated content at query #18
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(b"print('hello')")
    try:
        paths = [tmp_path]
        config = Config()
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert result == [tmp_path]
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(tmp_path)

    # Test case 3: Directory with Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some Python files
        py_file1 = os.path.join(tmpdir, "test1.py")
        py_file2 = os.path.join(tmpdir, "test2.py")
        with open(py_file1, "w") as f:
            f.write("print('test1')")
        with open(py_file2, "w") as f:
            f.write("print('test2')")

        # Create a non-Python file
        non_py_file = os.path.join(tmpdir, "test.txt")
        with open(non_py_file, "w") as f:
            f.write("not python")

        paths = [tmpdir]
        config = Config()
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert set(result) == {py_file1, py_file2}
        assert skipped == []
        assert broken == []

    # Test case 4: Non-existent path
    paths = ["/nonexistent/path"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["/nonexistent/path"]

    # Test case 5: Skipped file
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = os.path.join(tmpdir, "test.py")
        with open(py_file, "w") as f:
            f.write("print('test')")

        config = Config(skip=["test.py"])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert result == []
        assert skipped == [os.path.abspath(py_file)]
        assert broken == []

    # Test case 6: Symlink handling
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        py_file = os.path.join(subdir, "test.py")
        with open(py_file, "w") as f:
            f.write("print('test')")

        link_dir = os.path.join(tmpdir, "linkdir")
        os.symlink(subdir, link_dir)

        config = Config(follow_links=True)
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert py_file in result
        assert skipped == []
        assert broken == []

    # Test case 7: Multiple paths
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        py_file1 = os.path.join(tmpdir1, "test1.py")
        py_file2 = os.path.join(tmpdir2, "test2.py")
        with open(py_file1, "w") as f:
            f.write("print('test1')")
        with open(py_file2, "w") as f:
            f.write("print('test2')")

        paths = [tmpdir1, tmpdir2]
        config = Config()
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert set(result) == {py_file1, py_file2}
        assert skipped == []
        assert broken == []


# LLM-generated content at query #19
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    with open("test_file.py", "w") as f:
        f.write("# test")
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []
    os.remove("test_file.py")

    # Test case 3: Non-existent file path
    paths = ["non_existent_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_file.py"]

    # Test case 4: Directory with Python files
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# test2")
    with open("test_dir/file3.txt", "w") as f:
        f.write("# test3")

    paths = ["test_dir"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.py" in result
    assert skipped == []
    assert broken == []

    # Clean up
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.py")
    os.remove("test_dir/file3.txt")
    os.rmdir("test_dir")

    # Test case 5: Skipped file
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/skipped_file.py", "w") as f:
        f.write("# skipped")

    paths = ["test_dir"]
    config = Config(skip=["skipped_file.py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath("test_dir/skipped_file.py")]
    assert broken == []

    # Clean up
    os.remove("test_dir/skipped_file.py")
    os.rmdir("test_dir")

    # Test case 6: Broken symlink
    paths = ["broken_link"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["broken_link"]


# LLM-generated content at query #20
#--------------------------

```python
def test_find():
    # Test case 1: Test with a directory containing Python files
    config = Config()
    paths = ["test_directory"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2  # Assuming test_directory contains 2 Python files
    assert skipped == []
    assert broken == []

    # Test case 2: Test with a non-existent path
    paths = ["non_existent_path"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path"]

    # Test case 3: Test with a file path
    paths = ["test_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 4: Test with a skipped directory
    config = Config(skip=["skip_directory"])
    paths = ["test_directory"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1  # Assuming test_directory contains 2 Python files and one is skipped
    assert skipped == ["test_directory/skip_directory"]
    assert broken == []

    # Test case 5: Test with a skipped file
    config = Config(skip=["skip_file.py"])
    paths = ["test_directory"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1  # Assuming test_directory contains 2 Python files and one is skipped
    assert skipped == ["test_directory/skip_file.py"]
    assert broken == []


# LLM-generated content at query #21
#--------------------------

```python
def test_find(tmp_path):
    # Setup test files and directories
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "file1.py").write_text("print('hello')")
    (test_dir / "file2.txt").write_text("not python")
    sub_dir = test_dir / "sub_dir"
    sub_dir.mkdir()
    (sub_dir / "file3.py").write_text("print('world')")

    # Test with valid directory
    config = Config()
    paths = [str(test_dir)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert any("file1.py" in r for r in result)
    assert any("file3.py" in r for r in result)
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test with non-existent path
    paths = [str(tmp_path / "nonexistent")]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == str(tmp_path / "nonexistent")

    # Test with single file
    paths = [str(test_dir / "file1.py")]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "file1.py" in result[0]
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test with skipped directory
    config = Config(skip=["sub_dir"])
    paths = [str(test_dir)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "file1.py" in result[0]
    assert len(skipped) == 1
    assert "sub_dir" in skipped[0]
    assert len(broken) == 0


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/file1.py", "w") as f:
        f.write("# Python file")
    with open(f"{test_dir}/file2.txt", "w") as f:
        f.write("Not a Python file")

    # Test with non-existent path
    non_existent_path = "non_existent_path"

    # Test with single file
    single_file = "single_file.py"
    with open(single_file, "w") as f:
        f.write("# Single Python file")

    # Execute
    result = list(find([test_dir, non_existent_path, single_file], config, skipped, broken))

    # Verify
    assert len(result) == 2
    assert f"{test_dir}/file1.py" in result
    assert single_file in result
    assert non_existent_path in broken
    assert len(skipped) == 0

    # Cleanup
    os.remove(f"{test_dir}/file1.py")
    os.remove(f"{test_dir}/file2.txt")
    os.rmdir(test_dir)
    os.remove(single_file)


# LLM-generated content at query #2
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/test1.py", "w") as f:
        f.write("# test file 1")
    with open(f"{test_dir}/test2.py", "w") as f:
        f.write("# test file 2")
    with open(f"{test_dir}/not_python.txt", "w") as f:
        f.write("not python")

    # Test with non-existent path
    non_existent_path = "non_existent_path.py"

    # Test with single file
    single_file = "single_file.py"
    with open(single_file, "w") as f:
        f.write("# single file")

    # Execute
    result = list(find([test_dir, non_existent_path, single_file], config, skipped, broken))

    # Assert
    assert len(result) == 3
    assert f"{test_dir}/test1.py" in result
    assert f"{test_dir}/test2.py" in result
    assert single_file in result
    assert non_existent_path in broken
    assert len(skipped) == 0

    # Cleanup
    os.remove(f"{test_dir}/test1.py")
    os.remove(f"{test_dir}/test2.py")
    os.remove(f"{test_dir}/not_python.txt")
    os.rmdir(test_dir)
    os.remove(single_file)


# LLM-generated content at query #3
#--------------------------

```python
def test_find():
    # Test case 1: Test with a directory containing Python files
    config = Config()
    paths = ["test_directory"]
    skipped = []
    broken = []
    # Assuming test_directory contains test_file1.py and test_file2.py
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_file1.py" in result
    assert "test_file2.py" in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test case 2: Test with a non-existent path
    paths = ["non_existent_path"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "non_existent_path"

    # Test case 3: Test with a file path
    paths = ["test_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert result[0] == "test_file.py"
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test case 4: Test with a skipped directory
    config = Config(skip=["skip_directory"])
    paths = ["test_directory"]
    skipped = []
    broken = []
    # Assuming test_directory contains skip_directory with test_file3.py
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_file1.py" in result
    assert "test_file2.py" in result
    assert len(skipped) == 1
    assert "skip_directory" in skipped[0]
    assert len(broken) == 0


# LLM-generated content at query #4
#--------------------------

```python
def test_find():
    # Test with a directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("not python")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test")

        # Test basic functionality
        config = Config()
        skipped = []
        broken = []
        paths = [tmpdir]
        result = list(find(paths, config, skipped, broken))
        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    config = Config()
    skipped = []
    broken = []
    paths = ["/nonexistent/path"]
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "/nonexistent/path"

    # Test with single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(b"# test")
        tmp_path = tmp.name

    try:
        config = Config()
        skipped = []
        broken = []
        paths = [tmp_path]
        result = list(find(paths, config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmp_path
        assert len(skipped) == 0
        assert len(broken) == 0
    finally:
        os.unlink(tmp_path)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test structure
        os.makedirs(os.path.join(tmpdir, "skipme"))
        with open(os.path.join(tmpdir, "skipme", "test.py"), "w") as f:
            f.write("# should be skipped")

        config = Config(skip=["skipme"])
        skipped = []
        broken = []
        paths = [tmpdir]
        result = list(find(paths, config, skipped, broken))
        assert len(result) == 0
        assert len(skipped) == 1
        assert "skipme" in skipped[0]
        assert len(broken) == 0


# LLM-generated content at query #5
#--------------------------

```python
def test_find():
    # Test case 1: Test with a directory containing Python files
    config = Config()
    paths = ["test_directory"]
    skipped = []
    broken = []

    # Create a test directory with some Python files
    os.makedirs("test_directory", exist_ok=True)
    with open("test_directory/test1.py", "w") as f:
        f.write("# test file 1")
    with open("test_directory/test2.py", "w") as f:
        f.write("# test file 2")
    with open("test_directory/skip_me.py", "w") as f:
        f.write("# skipped file")

    # Configure to skip a file
    config.skip = ["skip_me.py"]

    # Call the function
    result = list(find(paths, config, skipped, broken))

    # Assertions
    assert len(result) == 2
    assert "test_directory/test1.py" in result
    assert "test_directory/test2.py" in result
    assert "test_directory/skip_me.py" not in result
    assert len(skipped) == 1
    assert "test_directory/skip_me.py" in skipped[0]
    assert len(broken) == 0

    # Cleanup
    os.remove("test_directory/test1.py")
    os.remove("test_directory/test2.py")
    os.remove("test_directory/skip_me.py")
    os.rmdir("test_directory")

    # Test case 2: Test with a non-existent path
    paths = ["non_existent_path"]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))

    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert "non_existent_path" in broken

    # Test case 3: Test with a single file path
    paths = ["test_file.py"]
    skipped = []
    broken = []

    # Create a test file
    with open("test_file.py", "w") as f:
        f.write("# test file")

    result = list(find(paths, config, skipped, broken))

    assert len(result) == 1
    assert "test_file.py" in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Cleanup
    os.remove("test_file.py")

    # Test case 4: Test with a directory containing a symlink
    paths = ["test_symlink_dir"]
    skipped = []
    broken = []

    # Create a test directory with a symlink
    os.makedirs("test_symlink_dir", exist_ok=True)
    with open("test_symlink_dir/target.py", "w") as f:
        f.write("# target file")
    os.symlink("test_symlink_dir/target.py", "test_symlink_dir/link.py")

    config.follow_links = True

    result = list(find(paths, config, skipped, broken))

    assert len(result) == 2
    assert "test_symlink_dir/target.py" in result
    assert "test_symlink_dir/link.py" in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Cleanup
    os.remove("test_symlink_dir/target.py")
    os.remove("test_symlink_dir/link.py")
    os.rmdir("test_symlink_dir")


# LLM-generated content at query #6
#--------------------------

```python
def test_find():
    # Test case 1: Single file path
    config = Config()
    skipped = []
    broken = []
    paths = ["test_file.py"]
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 2: Directory with Python files
    config = Config()
    skipped = []
    broken = []
    paths = ["test_dir"]
    result = list(find(paths, config, skipped, broken))
    assert all(path.endswith(".py") for path in result)
    assert skipped == []
    assert broken == []

    # Test case 3: Non-existent path
    config = Config()
    skipped = []
    broken = []
    paths = ["non_existent_path"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path"]

    # Test case 4: Skipped file
    config = Config()
    skipped = []
    broken = []
    paths = ["skipped_file.py"]
    config.is_skipped = lambda path: True
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ["skipped_file.py"]
    assert broken == []

    # Test case 5: Mixed paths (files, directories, non-existent)
    config = Config()
    skipped = []
    broken = []
    paths = ["test_file.py", "test_dir", "non_existent_path"]
    result = list(find(paths, config, skipped, broken))
    assert "test_file.py" in result
    assert any(path.endswith(".py") for path in result)
    assert skipped == []
    assert broken == ["non_existent_path"]


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    config = Config()
    skipped = []
    broken = []
    result = list(find([], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(b"print('hello')")
        tmp_path = tmp.name
    try:
        config = Config()
        skipped = []
        broken = []
        result = list(find([tmp_path], config, skipped, broken))
        assert result == [tmp_path]
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(tmp_path)

    # Test case 3: Non-existent file path
    config = Config()
    skipped = []
    broken = []
    result = list(find(["/non/existent/path.py"], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["/non/existent/path.py"]

    # Test case 4: Directory with Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some Python files
        py_file1 = os.path.join(tmpdir, "file1.py")
        py_file2 = os.path.join(tmpdir, "file2.py")
        with open(py_file1, "w") as f:
            f.write("print('hello')")
        with open(py_file2, "w") as f:
            f.write("print('world')")

        # Create a non-Python file
        non_py_file = os.path.join(tmpdir, "file.txt")
        with open(non_py_file, "w") as f:
            f.write("not python")

        config = Config()
        skipped = []
        broken = []
        result = sorted(find([tmpdir], config, skipped, broken))
        assert result == sorted([py_file1, py_file2])
        assert skipped == []
        assert broken == []

    # Test case 5: Skipped files and directories
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory structure
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        py_file = os.path.join(subdir, "file.py")
        with open(py_file, "w") as f:
            f.write("print('hello')")

        # Create a skipped file
        skipped_file = os.path.join(tmpdir, "skipped.py")
        with open(skipped_file, "w") as f:
            f.write("print('skipped')")

        config = Config(skip=["skipped.py", "subdir"])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert result == []
        assert sorted(skipped) == sorted([os.path.abspath(skipped_file), os.path.abspath(subdir)])
        assert broken == []

    # Test case 6: Follow links configuration
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a symlink to a directory
        target_dir = os.path.join(tmpdir, "target")
        os.makedirs(target_dir)
        link_dir = os.path.join(tmpdir, "link")
        os.symlink(target_dir, link_dir)

        # Create a Python file in the target directory
        py_file = os.path.join(target_dir, "file.py")
        with open(py_file, "w") as f:
            f.write("print('hello')")

        # Test with follow_links=True
        config = Config(follow_links=True)
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert py_file in result
        assert skipped == []
        assert broken == []

        # Test with follow_links=False
        config = Config(follow_links=False)
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert py_file not in result
        assert skipped == []
        assert broken == []


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test case 1: Empty paths
    paths = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
        tmp.write(b'test')
        tmp_path = tmp.name
    try:
        paths = [tmp_path]
        result = list(find(paths, config, skipped, broken))
        assert result == [tmp_path]
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(tmp_path)

    # Test case 3: Directory with Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        py_file = os.path.join(tmpdir, 'test.py')
        with open(py_file, 'w') as f:
            f.write('test')

        non_py_file = os.path.join(tmpdir, 'test.txt')
        with open(non_py_file, 'w') as f:
            f.write('test')

        # Create subdirectory with Python file
        subdir = os.path.join(tmpdir, 'subdir')
        os.makedirs(subdir)
        sub_py_file = os.path.join(subdir, 'subtest.py')
        with open(sub_py_file, 'w') as f:
            f.write('test')

        paths = [tmpdir]
        result = list(find(paths, config, skipped, broken))
        assert len(result) == 2
        assert py_file in result
        assert sub_py_file in result
        assert skipped == []
        assert broken == []

    # Test case 4: Non-existent path
    paths = ['/nonexistent/path']
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ['/nonexistent/path']

    # Test case 5: Skipped file
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = os.path.join(tmpdir, 'test.py')
        with open(py_file, 'w') as f:
            f.write('test')

        config = Config(skip=['test.py'])
        paths = [tmpdir]
        result = list(find(paths, config, skipped, broken))
        assert result == []
        assert skipped == [os.path.abspath(py_file)]
        assert broken == []

    # Test case 6: Mixed paths (files and directories)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a Python file in the temp directory
        py_file = os.path.join(tmpdir, 'test.py')
        with open(py_file, 'w') as f:
            f.write('test')

        # Create another Python file outside the directory
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
            tmp.write(b'test')
            tmp_path = tmp.name
        try:
            paths = [tmpdir, tmp_path]
            result = list(find(paths, config, skipped, broken))
            assert len(result) == 2
            assert py_file in result
            assert tmp_path in result
            assert skipped == []
            assert broken == []
        finally:
            os.unlink(tmp_path)


# LLM-generated content at query #2
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test case 1: Directory with Python files
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/file1.py", "w") as f:
        f.write("# Python file")
    with open(f"{test_dir}/file2.txt", "w") as f:
        f.write("Not a Python file")

    # Test case 2: Non-existent path
    non_existent_path = "non_existent_path.py"

    # Test case 3: Single Python file
    single_file = "single_file.py"
    with open(single_file, "w") as f:
        f.write("# Single Python file")

    # Execute
    results = list(find([test_dir, non_existent_path, single_file], config, skipped, broken))

    # Assert
    assert len(results) == 2
    assert f"{test_dir}/file1.py" in results
    assert single_file in results
    assert non_existent_path in broken
    assert len(skipped) == 0

    # Cleanup
    os.remove(f"{test_dir}/file1.py")
    os.remove(f"{test_dir}/file2.txt")
    os.rmdir(test_dir)
    os.remove(single_file)


# LLM-generated content at query #3
#--------------------------

```python
def test_find(tmp_path):
    # Setup test files and directories
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    sub_dir = test_dir / "sub_dir"
    sub_dir.mkdir()
    skipped_dir = test_dir / "skipped_dir"
    skipped_dir.mkdir()

    # Create some test files
    (test_dir / "file1.py").write_text("# test")
    (test_dir / "file2.py").write_text("# test")
    (sub_dir / "file3.py").write_text("# test")
    (skipped_dir / "file4.py").write_text("# test")
    (test_dir / "not_python.txt").write_text("not python")

    # Create a broken path
    broken_path = str(tmp_path / "nonexistent.py")

    # Setup config
    config = Config()
    config.skipped_path = ["skipped_dir"]
    config.follow_links = False

    # Test the find function
    paths = [str(test_dir), broken_path]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))

    # Assertions
    assert len(result) == 3  # file1.py, file2.py, file3.py
    assert str(test_dir / "file1.py") in result
    assert str(test_dir / "file2.py") in result
    assert str(sub_dir / "file3.py") in result
    assert str(skipped_dir / "file4.py") not in result
    assert str(test_dir / "not_python.txt") not in result

    assert len(skipped) == 1
    assert str(skipped_dir) in skipped[0]

    assert len(broken) == 1
    assert broken_path in broken

def test_find_single_file():
    # Test with a single file path
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(b"# test")
        tmp_path = tmp.name

    config = Config()
    skipped = []
    broken = []
    result = list(find([tmp_path], config, skipped, broken))

    assert len(result) == 1
    assert tmp_path in result
    assert len(skipped) == 0
    assert len(broken) == 0

    os.unlink(tmp_path)

def test_find_nonexistent_path():
    # Test with a non-existent path
    config = Config()
    skipped = []
    broken = []
    result = list(find(["nonexistent/path.py"], config, skipped, broken))

    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert "nonexistent/path.py" in broken


# LLM-generated content at query #2
#--------------------------

```python
def test_find():
    # Test with a directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "test1.py").write_text("# Python file")
        (Path(tmpdir) / "test2.py").write_text("# Another Python file")
        (Path(tmpdir) / "test.txt").write_text("# Not a Python file")
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        (subdir / "test3.py").write_text("# Python file in subdir")

        # Create config
        config = Config()
        skipped = []
        broken = []

        # Test find function
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 3
        assert all(path.endswith(".py") for path in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    skipped = []
    broken = []
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "/nonexistent/path"

    # Test with single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# Python file")
        tmpfile_path = tmpfile.name

    try:
        skipped = []
        broken = []
        result = list(find([tmpfile_path], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmpfile_path
        assert len(skipped) == 0
        assert len(broken) == 0
    finally:
        os.unlink(tmpfile_path)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "test1.py").write_text("# Python file")
        skip_dir = Path(tmpdir) / "skipdir"
        skip_dir.mkdir()
        (skip_dir / "test2.py").write_text("# Python file in skipdir")

        # Create config that skips "skipdir"
        config = Config(skip=["skipdir"])
        skipped = []
        broken = []

        # Test find function
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 1
        assert result[0].endswith("test1.py")
        assert len(skipped) == 1
        assert skipped[0].endswith("skipdir")
        assert len(broken) == 0


# LLM-generated content at query #4
#--------------------------

```python
def test_find():
    # Test with a directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("# not python")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test")

        config = Config()
        paths = [tmpdir]
        skipped = []
        broken = []

        result = list(find(paths, config, skipped, broken))

        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    config = Config()
    paths = ["/nonexistent/path"]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))

    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "/nonexistent/path"

    # Test with a single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# test")
        tmpfile_path = tmpfile.name

    try:
        config = Config()
        paths = [tmpfile_path]
        skipped = []
        broken = []

        result = list(find(paths, config, skipped, broken))

        assert len(result) == 1
        assert result[0] == tmpfile_path
        assert len(skipped) == 0
        assert len(broken) == 0
    finally:
        os.unlink(tmpfile_path)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "skipme"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "skipme", "test2.py"), "w") as f:
            f.write("# test")

        config = Config(skip=["skipme"])
        paths = [tmpdir]
        skipped = []
        broken = []

        result = list(find(paths, config, skipped, broken))

        assert len(result) == 1
        assert "test1.py" in result[0]
        assert len(skipped) == 1
        assert "skipme" in skipped[0]
        assert len(broken) == 0


# LLM-generated content at query #3
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    with open("test_file.py", "w") as f:
        f.write("# test")
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []
    os.remove("test_file.py")

    # Test case 3: Directory with Python files
    os.makedirs("test_dir")
    with open("test_dir/file1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# test2")
    with open("test_dir/non_py.txt", "w") as f:
        f.write("not python")
    paths = ["test_dir"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.py" in result
    assert skipped == []
    assert broken == []
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.py")
    os.remove("test_dir/non_py.txt")
    os.rmdir("test_dir")

    # Test case 4: Non-existent path
    paths = ["non_existent_path.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path.py"]

    # Test case 5: Skipped files and directories
    os.makedirs("skip_dir")
    with open("skip_dir/skip_file.py", "w") as f:
        f.write("# skip")
    paths = ["skip_dir"]
    config = Config(skip=["skip_dir/skip_file.py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath("skip_dir/skip_file.py")]
    assert broken == []
    os.remove("skip_dir/skip_file.py")
    os.rmdir("skip_dir")

    # Test case 6: Symlinks (if supported)
    if hasattr(os, "symlink"):
        os.makedirs("real_dir")
        with open("real_dir/real_file.py", "w") as f:
            f.write("# real")
        os.symlink("real_dir", "symlink_dir")
        paths = ["symlink_dir"]
        config = Config(follow_links=True)
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert len(result) == 1
        assert "real_dir/real_file.py" in result[0] or "symlink_dir/real_file.py" in result[0]
        assert skipped == []
        assert broken == []
        os.remove("symlink_dir")
        os.remove("real_dir/real_file.py")
        os.rmdir("real_dir")


# LLM-generated content at query #5
#--------------------------

```python
def test_find():
    # Test with a directory containing Python files
    config = Config()
    paths = ["test_dir"]
    skipped = []
    broken = []

    # Create a test directory with Python files
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/test1.py", "w") as f:
        f.write("# test file 1")
    with open("test_dir/test2.py", "w") as f:
        f.write("# test file 2")
    with open("test_dir/ignored.txt", "w") as f:
        f.write("not a Python file")

    # Test the find function
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_dir/test1.py" in result
    assert "test_dir/test2.py" in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Clean up
    os.remove("test_dir/test1.py")
    os.remove("test_dir/test2.py")
    os.remove("test_dir/ignored.txt")
    os.rmdir("test_dir")

    # Test with a non-existent path
    paths = ["non_existent_path"]
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert "non_existent_path" in broken

    # Test with a single file path
    with open("single_file.py", "w") as f:
        f.write("# single file")
    paths = ["single_file.py"]
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "single_file.py" in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Clean up
    os.remove("single_file.py")


# LLM-generated content at query #4
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(b"test")
    try:
        paths = [tmp_path]
        config = Config()
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert result == [tmp_path]
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(tmp_path)

    # Test case 3: Directory with Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        py_file = os.path.join(tmpdir, "test.py")
        with open(py_file, "w") as f:
            f.write("test")
        non_py_file = os.path.join(tmpdir, "test.txt")
        with open(non_py_file, "w") as f:
            f.write("test")

        paths = [tmpdir]
        config = Config()
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert result == [py_file]
        assert skipped == []
        assert broken == []

    # Test case 4: Non-existent path
    paths = ["/non/existent/path"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["/non/existent/path"]

    # Test case 5: Skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        py_file = os.path.join(tmpdir, "test.py")
        with open(py_file, "w") as f:
            f.write("test")

        # Create a directory to skip
        skip_dir = os.path.join(tmpdir, "skip_me")
        os.makedirs(skip_dir)
        skipped_py_file = os.path.join(skip_dir, "skipped.py")
        with open(skipped_py_file, "w") as f:
            f.write("test")

        paths = [tmpdir]
        config = Config(skip=["skip_me"])
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert result == [py_file]
        assert skipped == [os.path.abspath(skip_dir)]
        assert broken == []


# LLM-generated content at query #5
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    config = Config()
    skipped = []
    broken = []
    result = list(find([], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(b"print('hello')")
    try:
        config = Config()
        skipped = []
        broken = []
        result = list(find([tmp_path], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmp_path
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(tmp_path)

    # Test case 3: Directory with Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some Python files
        py_file1 = os.path.join(tmpdir, "file1.py")
        py_file2 = os.path.join(tmpdir, "file2.py")
        with open(py_file1, "w") as f:
            f.write("print('file1')")
        with open(py_file2, "w") as f:
            f.write("print('file2')")

        # Create a non-Python file
        txt_file = os.path.join(tmpdir, "file.txt")
        with open(txt_file, "w") as f:
            f.write("text file")

        config = Config()
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 2
        assert py_file1 in result
        assert py_file2 in result
        assert txt_file not in result
        assert skipped == []
        assert broken == []

    # Test case 4: Non-existent path
    config = Config()
    skipped = []
    broken = []
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert len(broken) == 1
    assert broken[0] == "/nonexistent/path"

    # Test case 5: Skipped files and directories
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory structure
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        py_file = os.path.join(subdir, "file.py")
        with open(py_file, "w") as f:
            f.write("print('file')")

        # Create a skipped file
        skipped_file = os.path.join(tmpdir, "skipped.py")
        with open(skipped_file, "w") as f:
            f.write("print('skipped')")

        config = Config(skip=["skipped.py"])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == py_file
        assert len(skipped) == 1
        assert skipped[0] == os.path.abspath(skipped_file)
        assert broken == []

    # Test case 6: Symlinks (if supported)
    if hasattr(os, "symlink"):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory with a symlink
            target_dir = os.path.join(tmpdir, "target")
            os.makedirs(target_dir)
            py_file = os.path.join(target_dir, "file.py")
            with open(py_file, "w") as f:
                f.write("print('file')")

            link_dir = os.path.join(tmpdir, "link")
            os.symlink(target_dir, link_dir)

            config = Config(follow_links=True)
            skipped = []
            broken = []
            result = list(find([link_dir], config, skipped, broken))
            assert len(result) == 1
            assert result[0] == py_file
            assert skipped == []
            assert broken == []


# LLM-generated content at query #6
#--------------------------

```python
def test_find(tmp_path):
    # Setup test files and directories
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "file1.py").write_text("# Python file")
    (test_dir / "file2.txt").write_text("Not a Python file")
    (test_dir / "subdir").mkdir()
    (test_dir / "subdir" / "file3.py").write_text("# Python file in subdir")

    # Create a skipped directory
    skipped_dir = test_dir / "skipped_dir"
    skipped_dir.mkdir()
    (skipped_dir / "file4.py").write_text("# Should be skipped")

    # Create a config that skips the skipped_dir
    config = Config(skip=["skipped_dir"])

    # Test the find function
    paths = [str(test_dir)]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))

    # Assertions
    assert len(result) == 2
    assert str(test_dir / "file1.py") in result
    assert str(test_dir / "subdir" / "file3.py") in result
    assert str(test_dir / "file2.txt") not in result
    assert str(skipped_dir / "file4.py") not in result
    assert len(skipped) == 1
    assert str(skipped_dir) in skipped[0]
    assert len(broken) == 0

    # Test with non-existent path
    paths = [str(test_dir), "/non/existent/path"]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))

    assert len(result) == 2
    assert len(broken) == 1
    assert "/non/existent/path" in broken

    # Test with a single file path
    paths = [str(test_dir / "file1.py")]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))

    assert len(result) == 1
    assert str(test_dir / "file1.py") in result


# LLM-generated content at query #7
#--------------------------

```python
def test_find():
    # Test case 1: Single file path
    config = Config()
    paths = ["test_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 2: Directory with Python files
    config = Config()
    paths = ["test_directory"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) > 0
    assert all(file.endswith(".py") for file in result)
    assert skipped == []
    assert broken == []

    # Test case 3: Non-existent path
    config = Config()
    paths = ["non_existent_path"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path"]

    # Test case 4: Skipped file
    config = Config(skip=["test_skip.py"])
    paths = ["test_skip.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ["test_skip.py"]
    assert broken == []

    # Test case 5: Mixed paths
    config = Config()
    paths = ["test_file.py", "test_directory", "non_existent_path"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_file.py" in result
    assert any(file.endswith(".py") for file in result)
    assert skipped == []
    assert broken == ["non_existent_path"]


# LLM-generated content at query #6
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    config = Config()
    skipped = []
    broken = []
    result = list(find([], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(b"print('hello')")
        tmp_path = tmp.name
    try:
        config = Config()
        skipped = []
        broken = []
        result = list(find([tmp_path], config, skipped, broken))
        assert result == [tmp_path]
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(tmp_path)

    # Test case 3: Directory with Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some Python files
        py_file1 = os.path.join(tmpdir, "test1.py")
        py_file2 = os.path.join(tmpdir, "test2.py")
        non_py_file = os.path.join(tmpdir, "test.txt")
        with open(py_file1, "w") as f:
            f.write("print('test1')")
        with open(py_file2, "w") as f:
            f.write("print('test2')")
        with open(non_py_file, "w") as f:
            f.write("not python")

        config = Config()
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert set(result) == {py_file1, py_file2}
        assert skipped == []
        assert broken == []

    # Test case 4: Non-existent path
    config = Config()
    skipped = []
    broken = []
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["/nonexistent/path"]

    # Test case 5: Skipped files and directories
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory structure
        os.makedirs(os.path.join(tmpdir, "skip_me"))
        os.makedirs(os.path.join(tmpdir, "keep_me"))
        py_file = os.path.join(tmpdir, "keep_me", "test.py")
        with open(py_file, "w") as f:
            f.write("print('test')")

        config = Config(skip=["skip_me"])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert result == [py_file]
        assert len(skipped) == 1
        assert skipped[0].endswith("skip_me")
        assert broken == []

    # Test case 6: Follow links configuration
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create symlink
        real_dir = os.path.join(tmpdir, "real_dir")
        os.makedirs(real_dir)
        link_dir = os.path.join(tmpdir, "link_dir")
        os.symlink(real_dir, link_dir)

        py_file = os.path.join(real_dir, "test.py")
        with open(py_file, "w") as f:
            f.write("print('test')")

        # Test with follow_links=True
        config = Config(follow_links=True)
        skipped = []
        broken = []
        result = list(find([link_dir], config, skipped, broken))
        assert result == [py_file]
        assert skipped == []
        assert broken == []

        # Test with follow_links=False
        config = Config(follow_links=False)
        skipped = []
        broken = []
        result = list(find([link_dir], config, skipped, broken))
        assert result == []
        assert skipped == []
        assert broken == []


# LLM-generated content at query #7
#--------------------------

```python
def test_find():
    # Test case 1: Single file path
    config = Config()
    paths = ["test_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 2: Directory with Python files
    config = Config()
    paths = ["test_dir"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.py" in result
    assert skipped == []
    assert broken == []

    # Test case 3: Non-existent path
    config = Config()
    paths = ["non_existent_path"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path"]

    # Test case 4: Skipped file
    config = Config(skip=["test_dir/skip_file.py"])
    paths = ["test_dir"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_dir/file1.py" in result
    assert "test_dir/skip_file.py" not in result
    assert "test_dir/skip_file.py" in skipped
    assert broken == []

    # Test case 5: Mixed paths (files and directories)
    config = Config()
    paths = ["test_file.py", "test_dir"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_file.py" in result
    assert "test_dir/file1.py" in result
    assert skipped == []
    assert broken == []


# LLM-generated content at query #8
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test case 1: Empty paths
    paths = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Non-existent path
    paths = ["non_existent_path.py"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path.py"]

    # Test case 3: Single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(b"print('hello')")
        tmp_path = tmp.name

    paths = [tmp_path]
    result = list(find(paths, config, skipped, broken))
    assert result == [tmp_path]
    assert skipped == []
    assert broken == []
    os.unlink(tmp_path)

    # Test case 4: Directory with Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files
        py_file = os.path.join(tmpdir, "test.py")
        with open(py_file, "w") as f:
            f.write("print('hello')")

        non_py_file = os.path.join(tmpdir, "test.txt")
        with open(non_py_file, "w") as f:
            f.write("not python")

        # Create a subdirectory with a Python file
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        sub_py_file = os.path.join(subdir, "sub_test.py")
        with open(sub_py_file, "w") as f:
            f.write("print('sub')")

        paths = [tmpdir]
        result = list(find(paths, config, skipped, broken))
        assert len(result) == 2
        assert py_file in result
        assert sub_py_file in result
        assert skipped == []
        assert broken == []

    # Test case 5: Skipped files and directories
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file that should be skipped
        skip_file = os.path.join(tmpdir, "skip.py")
        with open(skip_file, "w") as f:
            f.write("print('skip')")

        # Create a directory that should be skipped
        skip_dir = os.path.join(tmpdir, "skip_dir")
        os.makedirs(skip_dir)
        skip_sub_file = os.path.join(skip_dir, "skip_sub.py")
        with open(skip_sub_file, "w") as f:
            f.write("print('skip sub')")

        # Configure to skip these
        config.skip = ["skip.py", "skip_dir"]

        paths = [tmpdir]
        result = list(find(paths, config, skipped, broken))
        assert result == []
        assert len(skipped) == 2
        assert skip_file in skipped
        assert skip_dir in skipped
        assert broken == []


# LLM-generated content at query #8
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    paths = ["test_dir", "nonexistent_file.py", "single_file.py"]
    skipped = []
    broken = []

    # Create test directory structure
    os.makedirs("test_dir/subdir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("# test")
    with open("test_dir/subdir/file2.py", "w") as f:
        f.write("# test")
    with open("single_file.py", "w") as f:
        f.write("# test")

    # Test
    result = list(find(paths, config, skipped, broken))

    # Assertions
    assert len(result) == 3
    assert "test_dir/file1.py" in result
    assert "test_dir/subdir/file2.py" in result
    assert "single_file.py" in result
    assert len(broken) == 1
    assert "nonexistent_file.py" in broken
    assert len(skipped) == 0

    # Cleanup
    os.remove("test_dir/file1.py")
    os.remove("test_dir/subdir/file2.py")
    os.rmdir("test_dir/subdir")
    os.rmdir("test_dir")
    os.remove("single_file.py")


# LLM-generated content at query #9
#--------------------------

```python
def test_find(tmp_path):
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Create test files and directories
    py_file = tmp_path / "test.py"
    py_file.write_text("# test")
    non_py_file = tmp_path / "test.txt"
    non_py_file.write_text("text")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    sub_py_file = subdir / "sub.py"
    sub_py_file.write_text("# sub")
    skipped_dir = tmp_path / "skipped"
    skipped_dir.mkdir()
    skipped_py = skipped_dir / "skipped.py"
    skipped_py.write_text("# skipped")

    # Test with valid directory
    result = list(find([str(tmp_path)], config, skipped, broken))
    assert len(result) == 2
    assert str(py_file) in result
    assert str(sub_py_file) in result
    assert len(skipped) == 0

    # Test with skipped directory
    config.skip = [str(skipped_dir)]
    result = list(find([str(tmp_path)], config, skipped, broken))
    assert len(result) == 2
    assert str(py_file) in result
    assert str(sub_py_file) in result
    assert str(skipped_py) in skipped

    # Test with non-existent path
    broken_path = str(tmp_path / "nonexistent.py")
    result = list(find([broken_path], config, skipped, broken))
    assert len(result) == 0
    assert broken_path in broken

    # Test with direct file path
    result = list(find([str(py_file)], config, skipped, broken))
    assert len(result) == 1
    assert str(py_file) in result

    # Test with non-Python file
    result = list(find([str(non_py_file)], config, skipped, broken))
    assert len(result) == 0


# LLM-generated content at query #9
#--------------------------

```python
def test_find():
    # Test with a directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("# not python")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test")

        config = Config()
        paths = [tmpdir]
        skipped = []
        broken = []

        result = list(find(paths, config, skipped, broken))
        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    config = Config()
    paths = ["nonexistent/path"]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "nonexistent/path"

    # Test with single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# test")
        tmpfile_path = tmpfile.name

    try:
        config = Config()
        paths = [tmpfile_path]
        skipped = []
        broken = []

        result = list(find(paths, config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmpfile_path
        assert len(skipped) == 0
        assert len(broken) == 0
    finally:
        os.unlink(tmpfile_path)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "skipme"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "skipme", "test2.py"), "w") as f:
            f.write("# test")

        config = Config(skip=["skipme"])
        paths = [tmpdir]
        skipped = []
        broken = []

        result = list(find(paths, config, skipped, broken))
        assert len(result) == 1
        assert "test1.py" in result[0]
        assert len(skipped) == 1
        assert "skipme" in skipped[0]
        assert len(broken) == 0


# LLM-generated content at query #10
#--------------------------

```python
def test_find():
    # Test case 1: Test with a directory containing Python files
    config = Config()
    paths = ["test_dir"]
    skipped = []
    broken = []
    # Create a test directory with Python files
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/test1.py", "w") as f:
        f.write("# test")
    with open("test_dir/test2.py", "w") as f:
        f.write("# test")
    # Create a skipped directory
    os.makedirs("test_dir/skipped_dir", exist_ok=True)
    with open("test_dir/skipped_dir/test3.py", "w") as f:
        f.write("# test")
    config.skip = ["skipped_dir"]
    # Test the function
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_dir/test1.py" in result
    assert "test_dir/test2.py" in result
    assert "test_dir/skipped_dir/test3.py" not in result
    assert len(skipped) == 1
    assert "test_dir/skipped_dir" in skipped[0]
    # Clean up
    os.remove("test_dir/test1.py")
    os.remove("test_dir/test2.py")
    os.rmdir("test_dir/skipped_dir")
    os.rmdir("test_dir")

    # Test case 2: Test with a non-existent path
    paths = ["non_existent_path"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(broken) == 1
    assert "non_existent_path" in broken

    # Test case 3: Test with a single file
    paths = ["test_file.py"]
    skipped = []
    broken = []
    with open("test_file.py", "w") as f:
        f.write("# test")
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "test_file.py" in result
    # Clean up
    os.remove("test_file.py")

    # Test case 4: Test with a directory containing non-Python files
    paths = ["test_dir_non_py"]
    skipped = []
    broken = []
    os.makedirs("test_dir_non_py", exist_ok=True)
    with open("test_dir_non_py/test.txt", "w") as f:
        f.write("# test")
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    # Clean up
    os.remove("test_dir_non_py/test.txt")
    os.rmdir("test_dir_non_py")


# LLM-generated content at query #10
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test case 1: Single file path
    paths = ["test_file.py"]
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 2: Non-existent file path
    paths = ["non_existent_file.py"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_file.py"]

    # Test case 3: Directory with Python files
    # Create a temporary directory with a Python file
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a Python file in the temporary directory
        test_file = Path(tmpdir) / "test_module.py"
        test_file.write_text("# test")

        paths = [tmpdir]
        result = list(find(paths, config, skipped, broken))
        assert str(test_file) in result
        assert skipped == []
        assert broken == []

    # Test case 4: Skipped file
    # Create a temporary directory with a skipped Python file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a Python file in the temporary directory
        test_file = Path(tmpdir) / "skip_me.py"
        test_file.write_text("# skip me")

        # Configure to skip the file
        config.skip = ["skip_me.py"]

        paths = [tmpdir]
        result = list(find(paths, config, skipped, broken))
        assert result == []
        assert str(test_file) in skipped
        assert broken == []

    # Test case 5: Mixed paths (files and directories)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a Python file in the temporary directory
        test_file = Path(tmpdir) / "test_module.py"
        test_file.write_text("# test")

        paths = ["test_file.py", tmpdir, "non_existent_file.py"]
        result = list(find(paths, config, skipped, broken))
        assert "test_file.py" in result
        assert str(test_file) in result
        assert skipped == []
        assert broken == ["non_existent_file.py"]


# LLM-generated content at query #11
#--------------------------

```python
def test_find(tmp_path):
    # Setup test files and directories
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "file1.py").write_text("# Python file")
    (test_dir / "file2.txt").write_text("Not a Python file")
    (test_dir / "file3.py").write_text("# Another Python file")

    sub_dir = test_dir / "sub_dir"
    sub_dir.mkdir()
    (sub_dir / "file4.py").write_text("# Python file in subdirectory")

    skipped_dir = test_dir / "skipped_dir"
    skipped_dir.mkdir()
    (skipped_dir / "file5.py").write_text("# Should be skipped")

    # Create config
    config = Config()
    config.skip = ["skipped_dir"]

    # Test parameters
    paths = [str(test_dir)]
    skipped = []
    broken = []

    # Call function
    result = list(find(paths, config, skipped, broken))

    # Assertions
    assert len(result) == 3
    assert str(test_dir / "file1.py") in result
    assert str(test_dir / "file3.py") in result
    assert str(sub_dir / "file4.py") in result
    assert str(test_dir / "file2.txt") not in result
    assert str(skipped_dir / "file5.py") not in result
    assert len(skipped) == 1
    assert str(skipped_dir) in skipped[0]
    assert len(broken) == 0

def test_find_nonexistent_path():
    # Test with non-existent path
    paths = ["nonexistent_path.py"]
    config = Config()
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))

    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert "nonexistent_path.py" in broken

def test_find_single_file():
    # Test with single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(b"# Test file")
        tmp_path = tmp.name

    try:
        paths = [tmp_path]
        config = Config()
        skipped = []
        broken = []

        result = list(find(paths, config, skipped, broken))

        assert len(result) == 1
        assert tmp_path in result
        assert len(skipped) == 0
        assert len(broken) == 0
    finally:
        os.unlink(tmp_path)

def test_find_with_follow_links():
    # Test with symlinks
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create real directory and file
        real_dir = Path(tmpdir) / "real_dir"
        real_dir.mkdir()
        (real_dir / "real_file.py").write_text("# Real file")

        # Create symlink
        symlink_dir = Path(tmpdir) / "symlink_dir"
        symlink_dir.symlink_to(real_dir)

        # Test with follow_links=True
        config = Config(follow_links=True)
        paths = [str(symlink_dir)]
        skipped = []
        broken = []

        result = list(find(paths, config, skipped, broken))

        assert len(result) == 1
        assert str(real_dir / "real_file.py") in result
        assert len(skipped) == 0
        assert len(broken) == 0


# LLM-generated content at query #11
#--------------------------

```python
def test_find(tmp_path):
    # Setup test files and directories
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    test_file = test_dir / "test.py"
    test_file.write_text("# test file")
    skipped_file = test_dir / "skipped.py"
    skipped_file.write_text("# skipped file")
    non_python_file = test_dir / "readme.md"
    non_python_file.write_text("# readme")
    broken_path = tmp_path / "broken_path"

    # Setup config
    config = Config()
    config.skipped_paths = [str(skipped_file)]
    config.follow_links = False

    # Test case 1: Find files in directory
    paths = [str(test_dir)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert str(test_file) in result
    assert str(skipped_file) in skipped
    assert len(broken) == 0

    # Test case 2: Non-existent path
    paths = [str(broken_path)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert str(broken_path) in broken

    # Test case 3: Direct file path
    paths = [str(test_file)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert str(test_file) in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test case 4: Non-Python file
    paths = [str(non_python_file)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 0


# LLM-generated content at query #12
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with a directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("# not python")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test")

        # Test
        result = list(find([tmpdir], config, skipped, broken))

        # Assert
        assert len(result) == 2
        assert os.path.join(tmpdir, "test1.py") in result
        assert os.path.join(tmpdir, "subdir", "test3.py") in result
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    skipped = []
    broken = []
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "/nonexistent/path"

    # Test with a single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(b"# test")
        tmp_path = tmp.name

    try:
        skipped = []
        broken = []
        result = list(find([tmp_path], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmp_path
        assert len(skipped) == 0
        assert len(broken) == 0
    finally:
        os.unlink(tmp_path)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "skipme"))
        with open(os.path.join(tmpdir, "skipme", "test.py"), "w") as f:
            f.write("# test")

        # Configure to skip "skipme" directory
        config.skip = ["skipme"]

        # Test
        result = list(find([tmpdir], config, skipped, broken))

        # Assert
        assert len(result) == 0
        assert len(skipped) == 1
        assert "skipme" in skipped[0]
        assert len(broken) == 0


# LLM-generated content at query #12
#--------------------------

```python
def test_find(tmp_path, mocker):
    # Setup
    config = mocker.Mock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Create test files and directories
    test_file = tmp_path / "test.py"
    test_file.write_text("# test")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    subdir_file = subdir / "sub.py"
    subdir_file.write_text("# sub")

    # Test with single file
    result = list(find([str(test_file)], config, [], []))
    assert result == [str(test_file)]

    # Test with directory
    result = list(find([str(tmp_path)], config, [], []))
    assert str(test_file) in result
    assert str(subdir_file) in result

    # Test with non-existent path
    broken = []
    list(find(["nonexistent"], config, [], broken))
    assert broken == ["nonexistent"]

    # Test with skipped file
    config.is_skipped.return_value = True
    skipped = []
    list(find([str(test_file)], config, skipped, []))
    assert str(test_file) in skipped

    # Test with unsupported file type
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = False
    result = list(find([str(test_file)], config, [], []))
    assert result == []

    # Test with symlink (if supported)
    if hasattr(os, 'symlink'):
        link = tmp_path / "link"
        link.symlink_to(subdir)
        config.follow_links = True
        result = list(find([str(link)], config, [], []))
        assert str(subdir_file) in result


# LLM-generated content at query #13
#--------------------------

```python
def test_find():
    # Test case 1: Test with a directory containing Python files
    config = Config()
    paths = ["test_directory"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_directory/file1.py" in result
    assert "test_directory/file2.py" in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test case 2: Test with a file that doesn't exist
    paths = ["nonexistent_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert "nonexistent_file.py" in broken

    # Test case 3: Test with a skipped directory
    config = Config(skip=["test_directory/skipped_dir"])
    paths = ["test_directory"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "test_directory/file1.py" in result
    assert len(skipped) == 1
    assert "test_directory/skipped_dir" in skipped[0]
    assert len(broken) == 0

    # Test case 4: Test with a single file
    paths = ["test_directory/file1.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "test_directory/file1.py" in result
    assert len(skipped) == 0
    assert len(broken) == 0


# LLM-generated content at query #13
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    paths = ["test_dir", "nonexistent_file.py", "single_file.py"]
    skipped = []
    broken = []

    # Create test directory structure
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("# test")
    with open("test_dir/subdir/file2.py", "w") as f:
        f.write("# test")
    with open("single_file.py", "w") as f:
        f.write("# test")

    # Test
    result = list(find(paths, config, skipped, broken))

    # Assertions
    assert len(result) == 3
    assert "test_dir/file1.py" in result
    assert "test_dir/subdir/file2.py" in result
    assert "single_file.py" in result
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "nonexistent_file.py"

    # Cleanup
    os.remove("test_dir/file1.py")
    os.remove("test_dir/subdir/file2.py")
    os.rmdir("test_dir/subdir")
    os.rmdir("test_dir")
    os.remove("single_file.py")


# LLM-generated content at query #14
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "test1.py").write_text("# test")
        (Path(tmpdir) / "test2.py").write_text("# test")
        (Path(tmpdir) / "test.txt").write_text("# not python")

        # Test
        result = list(find([tmpdir], config, skipped, broken))

        # Assert
        assert len(result) == 2
        assert all("test1.py" in r or "test2.py" in r for r in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    skipped = []
    broken = []
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "/nonexistent/path"

    # Test with single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# test")
        tmpfile_path = tmpfile.name

    try:
        skipped = []
        broken = []
        result = list(find([tmpfile_path], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmpfile_path
        assert len(skipped) == 0
        assert len(broken) == 0
    finally:
        os.unlink(tmpfile_path)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test structure
        test_dir = Path(tmpdir) / "test_dir"
        test_dir.mkdir()
        (test_dir / "test.py").write_text("# test")

        # Configure to skip test_dir
        config.skip = ["test_dir"]

        # Test
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 0
        assert len(skipped) == 1
        assert "test_dir" in skipped[0]
        assert len(broken) == 0


# LLM-generated content at query #14
#--------------------------

```python
def test_find():
    # Test with a directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("# not python")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test")

        # Create config
        config = Config()
        skipped = []
        broken = []

        # Test finding files
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)

    # Test with non-existent path
    skipped = []
    broken = []
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert len(broken) == 1
    assert broken[0] == "/nonexistent/path"

    # Test with single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# test")
        tmpfile.flush()
        tmpfile_name = tmpfile.name

    try:
        skipped = []
        broken = []
        result = list(find([tmpfile_name], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmpfile_name
    finally:
        os.unlink(tmpfile_name)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test structure
        os.makedirs(os.path.join(tmpdir, "skipme"))
        with open(os.path.join(tmpdir, "skipme", "test.py"), "w") as f:
            f.write("# should be skipped")

        # Create config that skips "skipme"
        config = Config(skip=["skipme"])
        skipped = []
        broken = []

        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 0
        assert len(skipped) == 1
        assert "skipme" in skipped[0]


# LLM-generated content at query #15
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 3: Non-existent file path
    paths = ["non_existent_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_file.py"]

    # Test case 4: Directory with Python files
    paths = ["test_directory"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_directory/file1.py", "test_directory/file2.py"]
    assert skipped == []
    assert broken == []

    # Test case 5: Directory with skipped files
    paths = ["test_directory"]
    config = Config(skip=["test_directory/skip_file.py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_directory/file1.py", "test_directory/file2.py"]
    assert skipped == ["test_directory/skip_file.py"]
    assert broken == []

    # Test case 6: Directory with broken symlink
    paths = ["test_directory"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_directory/file1.py", "test_directory/file2.py"]
    assert skipped == []
    assert broken == ["test_directory/broken_symlink.py"]


# LLM-generated content at query #15
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with a directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a Python file
        py_file = os.path.join(tmpdir, "test.py")
        with open(py_file, "w") as f:
            f.write("print('hello')")

        # Create a non-Python file
        non_py_file = os.path.join(tmpdir, "test.txt")
        with open(non_py_file, "w") as f:
            f.write("not python")

        # Test
        result = list(find([tmpdir], config, skipped, broken))

        # Assert
        assert len(result) == 1
        assert result[0] == py_file
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with a non-existent path
    skipped = []
    broken = []
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "/nonexistent/path"

    # Test with a single file
    skipped = []
    broken = []
    result = list(find([py_file], config, skipped, broken))
    assert len(result) == 1
    assert result[0] == py_file
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test with a skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory structure
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        py_file = os.path.join(subdir, "test.py")
        with open(py_file, "w") as f:
            f.write("print('hello')")

        # Configure to skip the subdir
        config.skip = ["subdir"]

        # Test
        result = list(find([tmpdir], config, skipped, broken))

        # Assert
        assert len(result) == 0
        assert len(skipped) == 1
        assert skipped[0] == os.path.abspath(subdir)
        assert len(broken) == 0


# LLM-generated content at query #16
#--------------------------

```python
def test_find(tmp_path):
    # Create test directory structure
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    sub_dir = test_dir / "sub_dir"
    sub_dir.mkdir()
    skipped_dir = test_dir / "skipped_dir"
    skipped_dir.mkdir()

    # Create test files
    (test_dir / "file1.py").write_text("# Python file")
    (sub_dir / "file2.py").write_text("# Python file")
    (test_dir / "file3.txt").write_text("# Not Python")
    (skipped_dir / "file4.py").write_text("# Skipped Python file")

    # Create config
    config = Config()
    config.skip = ["skipped_dir"]

    # Test find function
    paths = [str(test_dir)]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))

    # Assertions
    assert len(result) == 2
    assert str(test_dir / "file1.py") in result
    assert str(sub_dir / "file2.py") in result
    assert str(test_dir / "file3.txt") not in result
    assert str(skipped_dir / "file4.py") not in result

    assert len(skipped) == 1
    assert str(skipped_dir) in skipped[0]

    assert len(broken) == 0

def test_find_nonexistent_path():
    config = Config()
    paths = ["nonexistent_path"]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))

    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert "nonexistent_path" in broken

def test_find_single_file():
    config = Config()
    paths = ["single_file.py"]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))

    assert len(result) == 1
    assert "single_file.py" in result
    assert len(skipped) == 0
    assert len(broken) == 0

def test_find_skipped_file():
    config = Config()
    config.skip = ["skip_me.py"]
    paths = ["skip_me.py", "include_me.py"]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))

    assert len(result) == 1
    assert "include_me.py" in result
    assert len(skipped) == 1
    assert "skip_me.py" in skipped[0]
    assert len(broken) == 0


# LLM-generated content at query #16
#--------------------------

```python
def test_find():
    # Test case 1: Single file path
    config = Config()
    paths = ["test_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 2: Directory with Python files
    config = Config()
    paths = ["test_dir"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) > 0
    assert all(file.endswith(".py") for file in result)
    assert skipped == []
    assert broken == []

    # Test case 3: Non-existent path
    config = Config()
    paths = ["non_existent_path.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path.py"]

    # Test case 4: Skipped file
    config = Config(skip=["skip_this.py"])
    paths = ["skip_this.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ["skip_this.py"]
    assert broken == []

    # Test case 5: Mixed paths (files and directories)
    config = Config()
    paths = ["test_file.py", "test_dir"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_file.py" in result
    assert any(file.endswith(".py") for file in result)
    assert skipped == []
    assert broken == []


# LLM-generated content at query #17
#--------------------------

```python
def test_find():
    # Test case 1: Test with a directory containing Python files
    config = Config()
    paths = ["test_directory"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2  # Assuming test_directory has 2 Python files
    assert skipped == []
    assert broken == []

    # Test case 2: Test with a non-existent path
    config = Config()
    paths = ["non_existent_path"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert skipped == []
    assert broken == ["non_existent_path"]

    # Test case 3: Test with a file path
    config = Config()
    paths = ["test_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert result[0] == "test_file.py"
    assert skipped == []
    assert broken == []

    # Test case 4: Test with a skipped directory
    config = Config(skip=["skip_directory"])
    paths = ["test_directory"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1  # Assuming skip_directory has 1 Python file
    assert skipped == ["test_directory/skip_directory"]
    assert broken == []


# LLM-generated content at query #17
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test case 1: Single file path
    paths = ["test_file.py"]
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 2: Non-existent file path
    paths = ["non_existent_file.py"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_file.py"]

    # Test case 3: Directory with Python files
    paths = ["test_directory"]
    result = list(find(paths, config, skipped, broken))
    assert "test_directory/file1.py" in result
    assert "test_directory/file2.py" in result
    assert skipped == []
    assert broken == []

    # Test case 4: Directory with skipped files
    paths = ["test_directory_with_skipped"]
    result = list(find(paths, config, skipped, broken))
    assert "test_directory_with_skipped/file1.py" in result
    assert "test_directory_with_skipped/skipped_file.py" not in result
    assert "test_directory_with_skipped/skipped_file.py" in skipped
    assert broken == []

    # Test case 5: Mixed paths (files and directories)
    paths = ["test_file.py", "test_directory"]
    result = list(find(paths, config, skipped, broken))
    assert "test_file.py" in result
    assert "test_directory/file1.py" in result
    assert "test_directory/file2.py" in result
    assert skipped == []
    assert broken == []


# LLM-generated content at query #18
#--------------------------

```python
def test_find(tmp_path):
    # Setup test files and directories
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "file1.py").write_text("# Python file")
    (test_dir / "file2.txt").write_text("Not a Python file")
    sub_dir = test_dir / "subdir"
    sub_dir.mkdir()
    (sub_dir / "file3.py").write_text("# Python file in subdir")
    (sub_dir / "skipped_file.py").write_text("# Should be skipped")

    # Create a config that skips files with "skipped" in the name
    config = Config(skip=["skipped_file.py"])
    skipped = []
    broken = []

    # Test with directory path
    paths = [str(test_dir)]
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2  # file1.py and file3.py
    assert str(test_dir / "file1.py") in result
    assert str(sub_dir / "file3.py") in result
    assert str(sub_dir / "skipped_file.py") not in result
    assert len(skipped) == 1
    assert str(sub_dir / "skipped_file.py") in skipped[0]
    assert len(broken) == 0

    # Test with non-existent path
    paths = [str(test_dir / "nonexistent.py")]
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(broken) == 1
    assert str(test_dir / "nonexistent.py") in broken

    # Test with direct file path
    paths = [str(test_dir / "file1.py")]
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert str(test_dir / "file1.py") in result
    assert len(skipped) == 1  # from previous test
    assert len(broken) == 1  # from previous test

    # Test with symlink (if supported)
    if hasattr(os, 'symlink'):
        symlink_dir = tmp_path / "symlink_dir"
        symlink_dir.mkdir()
        symlink = symlink_dir / "symlink"
        symlink.symlink_to(test_dir)
        config.follow_links = True
        paths = [str(symlink)]
        result = list(find(paths, config, skipped, broken))
        assert len(result) == 2  # file1.py and file3.py through symlink


# LLM-generated content at query #18
#--------------------------

```python
def test_find():
    # Test case 1: Single file path
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(b"print('hello')")
        tmp_path = tmp.name

    config = Config()
    skipped = []
    broken = []
    result = list(find([tmp_path], config, skipped, broken))
    assert len(result) == 1
    assert result[0] == tmp_path
    assert len(skipped) == 0
    assert len(broken) == 0
    os.unlink(tmp_path)

    # Test case 2: Non-existent file path
    config = Config()
    skipped = []
    broken = []
    result = list(find(["nonexistent.py"], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "nonexistent.py"

    # Test case 3: Directory with Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some Python files
        py_file1 = os.path.join(tmpdir, "file1.py")
        py_file2 = os.path.join(tmpdir, "file2.py")
        with open(py_file1, "w") as f:
            f.write("print('file1')")
        with open(py_file2, "w") as f:
            f.write("print('file2')")

        # Create a non-Python file
        txt_file = os.path.join(tmpdir, "file.txt")
        with open(txt_file, "w") as f:
            f.write("not python")

        config = Config()
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 2
        assert py_file1 in result
        assert py_file2 in result
        assert txt_file not in result
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test case 4: Skipped files and directories
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory structure
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        py_file = os.path.join(subdir, "file.py")
        with open(py_file, "w") as f:
            f.write("print('file')")

        # Create a file to be skipped
        skip_file = os.path.join(tmpdir, "skip.py")
        with open(skip_file, "w") as f:
            f.write("print('skip')")

        config = Config(skip=[skip_file])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 1
        assert py_file in result
        assert skip_file not in result
        assert len(skipped) == 1
        assert skipped[0] == skip_file
        assert len(broken) == 0


# LLM-generated content at query #19
#--------------------------

```python
def test_find():
    # Test with a directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "subdir", "test2.py"), "w") as f:
            f.write("# test2")
        with open(os.path.join(tmpdir, "ignore.txt"), "w") as f:
            f.write("ignore")

        # Test basic functionality
        config = Config()
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 2
        assert any("test.py" in r for r in result)
        assert any("test2.py" in r for r in result)

    # Test with skipped files
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "skipdir"))
        with open(os.path.join(tmpdir, "skip.py"), "w") as f:
            f.write("# skip")
        with open(os.path.join(tmpdir, "skipdir", "skip2.py"), "w") as f:
            f.write("# skip2")

        config = Config(skip=["skip.py", "skipdir"])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 0
        assert len(skipped) == 2
        assert any("skip.py" in s for s in skipped)
        assert any("skipdir" in s for s in skipped)

    # Test with non-existent path
    config = Config()
    skipped = []
    broken = []
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert len(broken) == 1
    assert broken[0] == "/nonexistent/path"

    # Test with single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# single file")
        tmpfile_path = tmpfile.name

    try:
        config = Config()
        skipped = []
        broken = []
        result = list(find([tmpfile_path], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmpfile_path
    finally:
        os.unlink(tmpfile_path)

    # Test with unsupported file type
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmpfile:
        tmpfile.write(b"# not python")
        tmpfile_path = tmpfile.name

    try:
        config = Config()
        skipped = []
        broken = []
        result = list(find([tmpfile_path], config, skipped, broken))
        assert len(result) == 0
    finally:
        os.unlink(tmpfile_path)


# LLM-generated content at query #20
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    config = Config()
    skipped = []
    broken = []
    result = list(find([], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Non-existent path
    config = Config()
    skipped = []
    broken = []
    result = list(find(["/non/existent/path"], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["/non/existent/path"]

    # Test case 3: Single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(b"print('hello')")
        tmp_path = tmp.name

    config = Config()
    skipped = []
    broken = []
    result = list(find([tmp_path], config, skipped, broken))
    assert result == [tmp_path]
    assert skipped == []
    assert broken == []
    os.unlink(tmp_path)

    # Test case 4: Directory with Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some Python files
        py_file1 = os.path.join(tmpdir, "file1.py")
        py_file2 = os.path.join(tmpdir, "file2.py")
        with open(py_file1, "w") as f:
            f.write("print('file1')")
        with open(py_file2, "w") as f:
            f.write("print('file2')")

        # Create a non-Python file
        txt_file = os.path.join(tmpdir, "file.txt")
        with open(txt_file, "w") as f:
            f.write("text file")

        config = Config()
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert set(result) == {py_file1, py_file2}
        assert skipped == []
        assert broken == []

    # Test case 5: Skipped files and directories
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory structure
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        py_file = os.path.join(subdir, "file.py")
        with open(py_file, "w") as f:
            f.write("print('file')")

        # Create a skipped file
        skipped_file = os.path.join(tmpdir, "skipped.py")
        with open(skipped_file, "w") as f:
            f.write("print('skipped')")

        config = Config(skip=["skipped.py"])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert result == [py_file]
        assert skipped == [os.path.abspath(skipped_file)]
        assert broken == []

    # Test case 6: Mixed paths (files and directories)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a Python file in the directory
        py_file = os.path.join(tmpdir, "file.py")
        with open(py_file, "w") as f:
            f.write("print('file')")

        # Create another Python file outside the directory
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
            tmp.write(b"print('outside')")
            outside_file = tmp.name

        config = Config()
        skipped = []
        broken = []
        result = list(find([tmpdir, outside_file], config, skipped, broken))
        assert set(result) == {py_file, outside_file}
        assert skipped == []
        assert broken == []
        os.unlink(outside_file)


# LLM-generated content at query #19
#--------------------------

```python
def test_find(tmp_path):
    # Setup test files and directories
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "file1.py").write_text("# Python file")
    (test_dir / "file2.txt").write_text("Not a Python file")
    (test_dir / "skipped_file.py").write_text("# Should be skipped")
    sub_dir = test_dir / "sub_dir"
    sub_dir.mkdir()
    (sub_dir / "file3.py").write_text("# Python file in subdirectory")

    # Create a symlink to test_dir to check for cycles
    symlink_dir = tmp_path / "symlink_dir"
    symlink_dir.symlink_to(test_dir)

    # Create config
    config = Config()
    config.skip = ["skipped_file.py"]
    config.follow_links = True

    # Test find function
    paths = [str(test_dir), str(symlink_dir), str(test_dir / "file1.py"), "nonexistent_file.py"]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))

    # Assertions
    assert len(result) == 3
    assert str(test_dir / "file1.py") in result
    assert str(sub_dir / "file3.py") in result
    assert str(test_dir / "file1.py") in result  # Direct file path
    assert str(test_dir / "skipped_file.py") not in result
    assert str(test_dir / "file2.txt") not in result
    assert len(skipped) == 1
    assert str(test_dir / "skipped_file.py") in skipped
    assert len(broken) == 1
    assert "nonexistent_file.py" in broken


# LLM-generated content at query #20
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with a directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("not python")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test")

        # Test find function
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)
        assert not any("test2.txt" in r for r in result)

    # Test with non-existent path
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert "/nonexistent/path" in broken

    # Test with direct file path
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# test")
        tmpfile.flush()
        result = list(find([tmpfile.name], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmpfile.name
        os.unlink(tmpfile.name)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        skip_dir = os.path.join(tmpdir, "skipme")
        os.makedirs(skip_dir)
        with open(os.path.join(skip_dir, "test.py"), "w") as f:
            f.write("# test")

        config.skip = [skip_dir]
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 0
        assert any(skip_dir in s for s in skipped)


# LLM-generated content at query #21
#--------------------------

```python
def test_find(tmp_path):
    # Setup test files and directories
    test_file = tmp_path / "test.py"
    test_file.write_text("# test content")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    subdir_file = subdir / "subfile.py"
    subdir_file.write_text("# subfile content")
    skipped_dir = tmp_path / "skipped_dir"
    skipped_dir.mkdir()
    skipped_file = skipped_dir / "skipped.py"
    skipped_file.write_text("# skipped content")
    broken_path = tmp_path / "nonexistent.py"

    # Create config
    config = Config()
    config.skip = ["skipped_dir"]

    # Test case 1: Find files in directory
    paths = [str(tmp_path)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert str(test_file) in result
    assert str(subdir_file) in result
    assert len(skipped) == 1
    assert str(skipped_dir) in skipped[0]
    assert len(broken) == 0

    # Test case 2: Find single file
    paths = [str(test_file)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert str(test_file) in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test case 3: Non-existent path
    paths = [str(broken_path)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert str(broken_path) in broken[0]

    # Test case 4: Mixed paths
    paths = [str(tmp_path), str(test_file), str(broken_path)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 3
    assert str(test_file) in result
    assert str(subdir_file) in result
    assert len(skipped) == 1
    assert str(skipped_dir) in skipped[0]
    assert len(broken) == 1
    assert str(broken_path) in broken[0]


# LLM-generated content at query #21
#--------------------------

```python
def test_find():
    # Test case 1: Test with a directory containing Python files
    config = Config()
    paths = ["test_directory"]
    skipped = []
    broken = []

    # Create a test directory with Python files
    os.makedirs("test_directory", exist_ok=True)
    with open("test_directory/test1.py", "w") as f:
        f.write("# test file")
    with open("test_directory/test2.py", "w") as f:
        f.write("# test file 2")
    with open("test_directory/non_python.txt", "w") as f:
        f.write("not a python file")

    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_directory/test1.py" in result
    assert "test_directory/test2.py" in result
    assert "test_directory/non_python.txt" not in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Clean up
    os.remove("test_directory/test1.py")
    os.remove("test_directory/test2.py")
    os.remove("test_directory/non_python.txt")
    os.rmdir("test_directory")

    # Test case 2: Test with a non-existent path
    paths = ["non_existent_path"]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert "non_existent_path" in broken

    # Test case 3: Test with a single file path
    paths = ["test_file.py"]
    skipped = []
    broken = []

    with open("test_file.py", "w") as f:
        f.write("# test file")

    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "test_file.py" in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Clean up
    os.remove("test_file.py")

    # Test case 4: Test with a skipped directory
    config = Config(skip=["skip_me"])
    paths = ["test_directory"]
    skipped = []
    broken = []

    os.makedirs("test_directory", exist_ok=True)
    os.makedirs("test_directory/skip_me", exist_ok=True)
    with open("test_directory/skip_me/test.py", "w") as f:
        f.write("# test file")
    with open("test_directory/test.py", "w") as f:
        f.write("# test file")

    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "test_directory/test.py" in result
    assert len(skipped) == 1
    assert "skip_me" in skipped[0]
    assert len(broken) == 0

    # Clean up
    os.remove("test_directory/skip_me/test.py")
    os.rmdir("test_directory/skip_me")
    os.remove("test_directory/test.py")
    os.rmdir("test_directory")


# LLM-generated content at query #22
#--------------------------

```python
def test_find():
    # Test case 1: Test with a directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some Python files
        (Path(tmpdir) / "file1.py").write_text("# Python file 1")
        (Path(tmpdir) / "file2.py").write_text("# Python file 2")
        (Path(tmpdir) / "subdir").mkdir()
        (Path(tmpdir) / "subdir" / "file3.py").write_text("# Python file 3")
        (Path(tmpdir) / "non_python.txt").write_text("# Not a Python file")

        config = Config()
        skipped = []
        broken = []

        result = list(find([tmpdir], config, skipped, broken))

        assert len(result) == 3
        assert all(path.endswith(".py") for path in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test case 2: Test with a non-existent path
    config = Config()
    skipped = []
    broken = []

    result = list(find(["/non/existent/path"], config, skipped, broken))

    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "/non/existent/path"

    # Test case 3: Test with a single file
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "single_file.py"
        file_path.write_text("# Single Python file")

        config = Config()
        skipped = []
        broken = []

        result = list(find([str(file_path)], config, skipped, broken))

        assert len(result) == 1
        assert result[0] == str(file_path)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test case 4: Test with a skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory structure with a skipped directory
        (Path(tmpdir) / "file1.py").write_text("# Python file 1")
        (Path(tmpdir) / "skipped_dir").mkdir()
        (Path(tmpdir) / "skipped_dir" / "file2.py").write_text("# Should be skipped")
        (Path(tmpdir) / "normal_dir").mkdir()
        (Path(tmpdir) / "normal_dir" / "file3.py").write_text("# Should be included")

        config = Config(skip=["skipped_dir"])
        skipped = []
        broken = []

        result = list(find([tmpdir], config, skipped, broken))

        assert len(result) == 2
        assert len(skipped) == 1
        assert "skipped_dir" in skipped[0]
        assert len(broken) == 0


# LLM-generated content at query #23
#--------------------------

```python
def test_find():
    # Test case 1: Single file path
    config = Config()
    paths = ["test_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 2: Directory with Python files
    config = Config()
    paths = ["test_dir"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) > 0
    assert all(file.endswith(".py") for file in result)
    assert skipped == []
    assert broken == []

    # Test case 3: Non-existent path
    config = Config()
    paths = ["non_existent_path.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path.py"]

    # Test case 4: Skipped file
    config = Config(skip=["test_skip.py"])
    paths = ["test_skip.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ["test_skip.py"]
    assert broken == []

    # Test case 5: Mixed paths (files and directories)
    config = Config()
    paths = ["test_file.py", "test_dir"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_file.py" in result
    assert any(file.endswith(".py") for file in result)
    assert skipped == []
    assert broken == []


# LLM-generated content at query #22
#--------------------------

```python
def test_find():
    # Test with a directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("text")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test")

        config = Config()
        paths = [tmpdir]
        skipped = []
        broken = []

        result = list(find(paths, config, skipped, broken))
        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    config = Config()
    paths = ["nonexistent_path"]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "nonexistent_path"

    # Test with a single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# test")
        tmpfile_path = tmpfile.name

    try:
        config = Config()
        paths = [tmpfile_path]
        skipped = []
        broken = []

        result = list(find(paths, config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmpfile_path
        assert len(skipped) == 0
        assert len(broken) == 0
    finally:
        os.unlink(tmpfile_path)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "skipdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "skipdir", "test2.py"), "w") as f:
            f.write("# test")

        config = Config(skip=["skipdir"])
        paths = [tmpdir]
        skipped = []
        broken = []

        result = list(find(paths, config, skipped, broken))
        assert len(result) == 1
        assert "test1.py" in result[0]
        assert len(skipped) == 1
        assert "skipdir" in skipped[0]
        assert len(broken) == 0


# LLM-generated content at query #23
#--------------------------

```python
def test_find(tmp_path):
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Create test files and directories
    test_file = tmp_path / "test.py"
    test_file.write_text("# test")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    sub_file = subdir / "sub.py"
    sub_file.write_text("# sub")
    skipped_dir = tmp_path / "skipped"
    skipped_dir.mkdir()
    skipped_file = skipped_dir / "skipped.py"
    skipped_file.write_text("# skipped")
    nonexistent = tmp_path / "nonexistent.py"

    # Mock config methods
    config.is_skipped = lambda path: "skipped" in str(path)
    config.is_supported_filetype = lambda path: path.endswith(".py")
    config.follow_links = False

    # Test
    result = list(find([str(tmp_path), str(nonexistent)], config, skipped, broken))

    # Assertions
    assert str(test_file) in result
    assert str(sub_file) in result
    assert str(skipped_file) not in result
    assert str(skipped_dir) in skipped
    assert str(nonexistent) in broken


# LLM-generated content at query #24
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 3: Non-existent file path
    paths = ["non_existent_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_file.py"]

    # Test case 4: Directory with Python files
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("not python")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test in subdir")

        paths = [tmpdir]
        config = Config()
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)
        assert skipped == []
        assert broken == []

    # Test case 5: Skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory structure with a skipped directory
        os.makedirs(os.path.join(tmpdir, "skip_me"))
        with open(os.path.join(tmpdir, "test.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "skip_me", "ignored.py"), "w") as f:
            f.write("# should be skipped")

        paths = [tmpdir]
        config = Config(skip=["skip_me"])
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert len(result) == 1
        assert "test.py" in result[0]
        assert len(skipped) == 1
        assert "skip_me" in skipped[0]
        assert broken == []

    # Test case 6: Mixed paths with some broken and some valid
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a valid file
        with open(os.path.join(tmpdir, "valid.py"), "w") as f:
            f.write("# valid")

        paths = [tmpdir, "broken_path.py", os.path.join(tmpdir, "valid.py")]
        config = Config()
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert len(result) == 2
        assert any("valid.py" in r for r in result)
        assert broken == ["broken_path.py"]
        assert skipped == []


# LLM-generated content at query #25
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 3: Directory with Python files
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("# not python")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test")

        paths = [tmpdir]
        config = Config()
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)
        assert skipped == []
        assert broken == []

    # Test case 4: Non-existent path
    paths = ["nonexistent_path.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent_path.py"]

    # Test case 5: Skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "skipdir"))
        with open(os.path.join(tmpdir, "skipdir", "test.py"), "w") as f:
            f.write("# test")

        paths = [tmpdir]
        config = Config(skip=["skipdir"])
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert result == []
        assert len(skipped) == 1
        assert "skipdir" in skipped[0]
        assert broken == []

    # Test case 6: Symlink handling
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "dir1"))
        os.makedirs(os.path.join(tmpdir, "dir2"))
        with open(os.path.join(tmpdir, "dir1", "test.py"), "w") as f:
            f.write("# test")

        # Create symlink
        symlink_path = os.path.join(tmpdir, "symlink")
        os.symlink(os.path.join(tmpdir, "dir1"), symlink_path)

        paths = [tmpdir]
        config = Config(follow_links=True)
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert len(result) == 1
        assert "test.py" in result[0]
        assert skipped == []
        assert broken == []


# LLM-generated content at query #24
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test case 1: Single file path
    paths = ["test_file.py"]
    with open("test_file.py", "w") as f:
        f.write("# test")
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []
    os.remove("test_file.py")

    # Test case 2: Directory with Python files
    os.makedirs("test_dir")
    with open("test_dir/file1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# test2")
    with open("test_dir/non_python.txt", "w") as f:
        f.write("text")
    paths = ["test_dir"]
    result = list(find(paths, config, skipped, broken))
    assert set(result) == {"test_dir/file1.py", "test_dir/file2.py"}
    assert skipped == []
    assert broken == []
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.py")
    os.remove("test_dir/non_python.txt")
    os.rmdir("test_dir")

    # Test case 3: Non-existent path
    paths = ["non_existent_path.py"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path.py"]

    # Test case 4: Skipped file
    config.skip = ["skip_me.py"]
    paths = ["skip_me.py"]
    with open("skip_me.py", "w") as f:
        f.write("# skip")
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ["skip_me.py"]
    assert broken == []
    os.remove("skip_me.py")
    config.skip = []

    # Test case 5: Mixed paths (files and directories)
    os.makedirs("test_mixed")
    with open("test_mixed/mixed.py", "w") as f:
        f.write("# mixed")
    with open("single.py", "w") as f:
        f.write("# single")
    paths = ["test_mixed", "single.py"]
    result = list(find(paths, config, skipped, broken))
    assert set(result) == {"test_mixed/mixed.py", "single.py"}
    assert skipped == []
    assert broken == []
    os.remove("test_mixed/mixed.py")
    os.rmdir("test_mixed")
    os.remove("single.py")


# LLM-generated content at query #26
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    paths = [".", "test_dir"]
    skipped = []
    broken = []

    # Create test directory and files
    os.makedirs("test_dir", exist_ok=True)
    with open("test_file.py", "w") as f:
        f.write("# test")
    with open("test_dir/test_file.py", "w") as f:
        f.write("# test")
    with open("test_dir/skipped_file.py", "w") as f:
        f.write("# skipped")

    # Mock config methods
    config.is_skipped = lambda path: "skipped" in str(path)
    config.is_supported_filetype = lambda path: path.endswith(".py")
    config.follow_links = False

    # Test
    result = list(find(paths, config, skipped, broken))

    # Assertions
    assert "test_file.py" in result
    assert "test_dir/test_file.py" in result
    assert "test_dir/skipped_file.py" not in result
    assert "test_dir/skipped_file.py" in skipped
    assert len(broken) == 0

    # Cleanup
    os.remove("test_file.py")
    os.remove("test_dir/test_file.py")
    os.remove("test_dir/skipped_file.py")
    os.rmdir("test_dir")


# LLM-generated content at query #25
#--------------------------

```python
def test_find():
    # Test case 1: Single file path
    config = Config()
    paths = ["test_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert result[0] == "test_file.py"
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test case 2: Directory with Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("# not python")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test in subdir")

        config = Config()
        paths = [tmpdir]
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test case 3: Non-existent path
    config = Config()
    paths = ["nonexistent_path.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "nonexistent_path.py"

    # Test case 4: Skipped file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file that should be skipped
        skip_file = os.path.join(tmpdir, "skip_me.py")
        with open(skip_file, "w") as f:
            f.write("# should be skipped")

        config = Config(skip=["skip_me.py"])
        paths = [tmpdir]
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert len(result) == 0
        assert len(skipped) == 1
        assert skipped[0] == skip_file
        assert len(broken) == 0

    # Test case 5: Mixed paths (files and directories)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        with open(os.path.join(tmpdir, "file1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "file2.py"), "w") as f:
            f.write("# test")

        config = Config()
        paths = [os.path.join(tmpdir, "file1.py"), tmpdir]
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert len(result) == 2
        assert any("file1.py" in r for r in result)
        assert any("file2.py" in r for r in result)
        assert len(skipped) == 0
        assert len(broken) == 0


# LLM-generated content at query #27
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with empty paths
    result = list(find([], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test with non-existent path
    result = list(find(["non_existent_path.py"], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert "non_existent_path.py" in broken

    # Test with single file
    with open("test_file.py", "w") as f:
        f.write("# test")
    result = list(find(["test_file.py"], config, skipped, broken))
    assert "test_file.py" in result
    assert skipped == []
    assert broken == []
    os.remove("test_file.py")

    # Test with directory
    os.makedirs("test_dir")
    with open("test_dir/test_file.py", "w") as f:
        f.write("# test")
    result = list(find(["test_dir"], config, skipped, broken))
    assert "test_dir/test_file.py" in result
    assert skipped == []
    assert broken == []
    os.remove("test_dir/test_file.py")
    os.rmdir("test_dir")

    # Test with skipped file
    config = Config(skip=["skip_me.py"])
    with open("skip_me.py", "w") as f:
        f.write("# test")
    result = list(find(["skip_me.py"], config, skipped, broken))
    assert result == []
    assert "skip_me.py" in skipped
    assert broken == []
    os.remove("skip_me.py")

    # Test with mixed paths
    os.makedirs("test_dir2")
    with open("test_dir2/test_file.py", "w") as f:
        f.write("# test")
    with open("test_file2.py", "w") as f:
        f.write("# test")
    result = list(find(["test_dir2", "test_file2.py", "non_existent.py"], config, skipped, broken))
    assert "test_dir2/test_file.py" in result
    assert "test_file2.py" in result
    assert "non_existent.py" in broken
    os.remove("test_dir2/test_file.py")
    os.rmdir("test_dir2")
    os.remove("test_file2.py")


# LLM-generated content at query #28
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/test.py", "w") as f:
        f.write("# test file")
    with open(f"{test_dir}/test.txt", "w") as f:
        f.write("not a python file")

    # Test with non-existent path
    non_existent_path = "non_existent_path"

    # Test with single file
    single_file = "single_file.py"
    with open(single_file, "w") as f:
        f.write("# single file")

    # Execute
    result = list(find([test_dir, non_existent_path, single_file], config, skipped, broken))

    # Verify
    assert len(result) == 2
    assert f"{test_dir}/test.py" in result
    assert single_file in result
    assert non_existent_path in broken
    assert len(skipped) == 0

    # Cleanup
    os.remove(f"{test_dir}/test.py")
    os.remove(f"{test_dir}/test.txt")
    os.rmdir(test_dir)
    os.remove(single_file)


# LLM-generated content at query #29
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    with open("test_file.py", "w") as f:
        f.write("# Test file")
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []
    os.remove("test_file.py")

    # Test case 3: Directory with Python files
    os.makedirs("test_dir")
    with open("test_dir/file1.py", "w") as f:
        f.write("# Test file 1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# Test file 2")
    with open("test_dir/non_python.txt", "w") as f:
        f.write("# Not a Python file")
    paths = ["test_dir"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.py" in result
    assert skipped == []
    assert broken == []
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.py")
    os.remove("test_dir/non_python.txt")
    os.rmdir("test_dir")

    # Test case 4: Non-existent path
    paths = ["non_existent_path.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path.py"]

    # Test case 5: Skipped file
    os.makedirs("test_skip_dir")
    with open("test_skip_dir/skip_me.py", "w") as f:
        f.write("# Should be skipped")
    paths = ["test_skip_dir"]
    config = Config(skip=["skip_me.py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath("test_skip_dir/skip_me.py")]
    assert broken == []
    os.remove("test_skip_dir/skip_me.py")
    os.rmdir("test_skip_dir")

    # Test case 6: Mixed paths (files and directories)
    os.makedirs("test_mixed_dir")
    with open("test_mixed_dir/mixed_file.py", "w") as f:
        f.write("# Mixed file")
    with open("single_file.py", "w") as f:
        f.write("# Single file")
    paths = ["test_mixed_dir", "single_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_mixed_dir/mixed_file.py" in result
    assert "single_file.py" in result
    assert skipped == []
    assert broken == []
    os.remove("test_mixed_dir/mixed_file.py")
    os.rmdir("test_mixed_dir")
    os.remove("single_file.py")


# LLM-generated content at query #30
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with a directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("not python")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test")

        # Test finding files
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)
        assert not any("test2.txt" in r for r in result)

    # Test with non-existent path
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert len(broken) == 1
    assert "/nonexistent/path" in broken

    # Test with direct file path
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# test")
        tmpfile.flush()
        result = list(find([tmpfile.name], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmpfile.name
        os.unlink(tmpfile.name)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        skip_dir = os.path.join(tmpdir, "skipme")
        os.makedirs(skip_dir)
        with open(os.path.join(skip_dir, "test.py"), "w") as f:
            f.write("# test")

        config.skip = [skip_dir]
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 0
        assert len(skipped) == 1
        assert skip_dir in skipped[0]


# LLM-generated content at query #26
#--------------------------

```python
def test_find():
    # Test case 1: Single file path
    config = Config()
    paths = ["test_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 2: Directory with Python files
    config = Config()
    paths = ["test_dir"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py", "test_dir/file2.py"]
    assert skipped == []
    assert broken == []

    # Test case 3: Skipped file
    config = Config(skip=["test_skip.py"])
    paths = ["test_skip.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ["test_skip.py"]
    assert broken == []

    # Test case 4: Non-existent path
    config = Config()
    paths = ["non_existent.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent.py"]

    # Test case 5: Mixed paths
    config = Config(skip=["test_skip_dir"])
    paths = ["test_file.py", "test_dir", "test_skip_dir", "non_existent.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py", "test_dir/file1.py", "test_dir/file2.py"]
    assert skipped == ["test_skip_dir"]
    assert broken == ["non_existent.py"]


# LLM-generated content at query #27
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    paths = ["test_dir"]
    skipped = []
    broken = []

    # Create test directory structure
    os.makedirs("test_dir/subdir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("# test")
    with open("test_dir/subdir/file2.py", "w") as f:
        f.write("# test")
    with open("test_dir/skip_me.py", "w") as f:
        f.write("# test")
    with open("test_dir/non_python.txt", "w") as f:
        f.write("not python")

    # Test with default config (should skip nothing)
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 3  # file1.py, file2.py, skip_me.py
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test with skip pattern
    config.skip = ["skip_me.py"]
    skipped = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2  # file1.py, file2.py
    assert len(skipped) == 1
    assert skipped[0].endswith("skip_me.py")

    # Test with broken path
    broken = []
    result = list(find(["nonexistent_path"], config, skipped, broken))
    assert len(result) == 0
    assert len(broken) == 1
    assert broken[0] == "nonexistent_path"

    # Test with single file path
    result = list(find(["test_dir/file1.py"], config, skipped, broken))
    assert len(result) == 1
    assert result[0].endswith("file1.py")

    # Cleanup
    import shutil
    shutil.rmtree("test_dir")


# LLM-generated content at query #28
#--------------------------

```python
def test_find():
    # Test with a directory containing Python files
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/test1.py", "w") as f:
        f.write("# test file 1")
    with open(f"{test_dir}/test2.py", "w") as f:
        f.write("# test file 2")
    with open(f"{test_dir}/test.txt", "w") as f:
        f.write("# not a Python file")

    config = Config()
    skipped = []
    broken = []

    result = list(find([test_dir], config, skipped, broken))
    assert len(result) == 2
    assert all("test_directory" in path for path in result)
    assert all(path.endswith(".py") for path in result)
    assert len(skipped) == 0
    assert len(broken) == 0

    # Clean up
    os.remove(f"{test_dir}/test1.py")
    os.remove(f"{test_dir}/test2.py")
    os.remove(f"{test_dir}/test.txt")
    os.rmdir(test_dir)

    # Test with a non-existent path
    broken = []
    result = list(find(["non_existent_path"], config, skipped, broken))
    assert len(result) == 0
    assert len(broken) == 1
    assert broken[0] == "non_existent_path"

    # Test with a single file
    test_file = "test_file.py"
    with open(test_file, "w") as f:
        f.write("# test file")
    result = list(find([test_file], config, skipped, broken))
    assert len(result) == 1
    assert result[0] == test_file
    os.remove(test_file)


# LLM-generated content at query #29
#--------------------------

```python
def test_find(tmp_path):
    # Setup test files and directories
    test_file = tmp_path / "test.py"
    test_file.write_text("# test file")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    subdir_file = subdir / "subfile.py"
    subdir_file.write_text("# subdir file")
    skipped_dir = tmp_path / "skipped_dir"
    skipped_dir.mkdir()
    skipped_file = skipped_dir / "skipped.py"
    skipped_file.write_text("# skipped file")
    non_python_file = tmp_path / "readme.txt"
    non_python_file.write_text("# not python")

    # Create config
    config = Config()
    config.skipped_paths = [str(skipped_dir)]
    config.follow_links = False

    # Test case 1: Find files in directory
    paths = [str(tmp_path)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert str(test_file) in result
    assert str(subdir_file) in result
    assert len(skipped) == 1
    assert str(skipped_dir) in skipped[0]
    assert len(broken) == 0

    # Test case 2: Non-existent path
    paths = [str(tmp_path / "nonexistent")]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert str(tmp_path / "nonexistent") in broken

    # Test case 3: Single file path
    paths = [str(test_file)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert str(test_file) in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Test case 4: Non-python file
    paths = [str(non_python_file)]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 0


# LLM-generated content at query #31
#--------------------------

```python
def test_find():
    # Test with a directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("not python")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test")

        # Test basic functionality
        config = Config()
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    skipped = []
    broken = []
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "/nonexistent/path"

    # Test with single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# test")
        tmpfile.flush()
        tmpfile_name = tmpfile.name

    try:
        skipped = []
        broken = []
        result = list(find([tmpfile_name], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmpfile_name
        assert len(skipped) == 0
        assert len(broken) == 0
    finally:
        os.unlink(tmpfile_name)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        skip_dir = os.path.join(tmpdir, "skipme")
        os.makedirs(skip_dir)
        with open(os.path.join(skip_dir, "test.py"), "w") as f:
            f.write("# test")

        config = Config(skip=["skipme"])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 0
        assert len(skipped) == 1
        assert "skipme" in skipped[0]
        assert len(broken) == 0


# LLM-generated content at query #32
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 3: Single directory path
    paths = ["test_dir"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert all("test_dir" in path for path in result)
    assert skipped == []
    assert broken == []

    # Test case 4: Non-existent path
    paths = ["non_existent_path.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path.py"]

    # Test case 5: Skipped file
    paths = ["skipped_file.py"]
    config = Config(skip=["skipped_file.py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ["skipped_file.py"]
    assert broken == []

    # Test case 6: Mixed paths
    paths = ["test_file.py", "test_dir", "non_existent_path.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_file.py" in result
    assert any("test_dir" in path for path in result)
    assert skipped == []
    assert broken == ["non_existent_path.py"]


# LLM-generated content at query #30
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with empty paths
    result = list(find([], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test with non-existent path
    result = list(find(["nonexistent_path.py"], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent_path.py"]

    # Test with single file
    with open("test_file.py", "w") as f:
        f.write("# test")
    result = list(find(["test_file.py"], config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []
    os.remove("test_file.py")

    # Test with directory containing Python files
    os.makedirs("test_dir")
    with open("test_dir/test1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/test2.py", "w") as f:
        f.write("# test2")
    with open("test_dir/ignore.txt", "w") as f:
        f.write("ignore")

    result = list(find(["test_dir"], config, skipped, broken))
    assert len(result) == 2
    assert "test_dir/test1.py" in result
    assert "test_dir/test2.py" in result
    assert skipped == []
    assert broken == []

    # Cleanup
    os.remove("test_dir/test1.py")
    os.remove("test_dir/test2.py")
    os.remove("test_dir/ignore.txt")
    os.rmdir("test_dir")

    # Test with skipped file
    config.skip = ["skip_me.py"]
    with open("skip_me.py", "w") as f:
        f.write("# skip")
    result = list(find(["skip_me.py"], config, skipped, broken))
    assert result == []
    assert skipped == ["skip_me.py"]
    assert broken == []
    os.remove("skip_me.py")

    # Test with symlink (if supported)
    try:
        os.makedirs("link_dir")
        with open("link_dir/target.py", "w") as f:
            f.write("# target")
        os.symlink("link_dir", "symlink_dir")

        config.follow_links = True
        result = list(find(["symlink_dir"], config, skipped, broken))
        assert len(result) == 1
        assert "symlink_dir/target.py" in result

        # Cleanup
        os.remove("symlink_dir/target.py")
        os.remove("symlink_dir")
        os.rmdir("link_dir")
    except OSError:
        pass  # Skip if symlinks not supported


# LLM-generated content at query #33
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    paths = ["test_dir", "nonexistent_file.py", "single_file.py"]
    skipped = []
    broken = []

    # Create test directory and files
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("# test")
    with open("test_dir/subdir/file2.py", "w") as f:
        f.write("# test")
    with open("single_file.py", "w") as f:
        f.write("# test")

    # Test
    result = list(find(paths, config, skipped, broken))

    # Assertions
    assert len(result) == 3
    assert "test_dir/file1.py" in result
    assert "test_dir/subdir/file2.py" in result
    assert "single_file.py" in result
    assert len(skipped) == 0
    assert len(broken) == 1
    assert "nonexistent_file.py" in broken

    # Cleanup
    os.remove("test_dir/file1.py")
    os.remove("test_dir/subdir/file2.py")
    os.rmdir("test_dir/subdir")
    os.rmdir("test_dir")
    os.remove("single_file.py")


# LLM-generated content at query #34
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/test1.py", "w") as f:
        f.write("# test")
    with open(f"{test_dir}/test2.py", "w") as f:
        f.write("# test")
    with open(f"{test_dir}/test.txt", "w") as f:
        f.write("not python")

    # Test with file
    test_file = "test_file.py"
    with open(test_file, "w") as f:
        f.write("# test")

    # Test with non-existent path
    non_existent = "non_existent.py"

    # Test execution
    result = list(find([test_dir, test_file, non_existent], config, skipped, broken))

    # Assertions
    assert len(result) == 3
    assert f"{test_dir}/test1.py" in result
    assert f"{test_dir}/test2.py" in result
    assert test_file in result
    assert non_existent in broken
    assert len(skipped) == 0

    # Cleanup
    os.remove(f"{test_dir}/test1.py")
    os.remove(f"{test_dir}/test2.py")
    os.remove(f"{test_dir}/test.txt")
    os.rmdir(test_dir)
    os.remove(test_file)


# LLM-generated content at query #35
#--------------------------

```python
def test_find():
    # Test case 1: Test with a directory containing Python files
    config = Config()
    paths = ["test_directory"]
    skipped = []
    broken = []

    # Create a test directory with Python files
    os.makedirs("test_directory", exist_ok=True)
    with open("test_directory/file1.py", "w") as f:
        f.write("# Python file 1")
    with open("test_directory/file2.py", "w") as f:
        f.write("# Python file 2")
    with open("test_directory/file3.txt", "w") as f:
        f.write("# Not a Python file")

    # Call the function
    result = list(find(paths, config, skipped, broken))

    # Assertions
    assert len(result) == 2
    assert "test_directory/file1.py" in result
    assert "test_directory/file2.py" in result
    assert "test_directory/file3.txt" not in result
    assert len(skipped) == 0
    assert len(broken) == 0

    # Clean up
    os.remove("test_directory/file1.py")
    os.remove("test_directory/file2.py")
    os.remove("test_directory/file3.txt")
    os.rmdir("test_directory")

    # Test case 2: Test with a non-existent path
    paths = ["non_existent_path"]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))

    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert "non_existent_path" in broken

    # Test case 3: Test with a file path
    paths = ["test_file.py"]
    skipped = []
    broken = []

    with open("test_file.py", "w") as f:
        f.write("# Python file")

    result = list(find(paths, config, skipped, broken))

    assert len(result) == 1
    assert "test_file.py" in result
    assert len(skipped) == 0
    assert len(broken) == 0

    os.remove("test_file.py")

    # Test case 4: Test with a skipped directory
    paths = ["test_skipped_directory"]
    skipped = []
    broken = []

    os.makedirs("test_skipped_directory", exist_ok=True)
    with open("test_skipped_directory/file.py", "w") as f:
        f.write("# Python file")

    config = Config(skip=["test_skipped_directory"])

    result = list(find(paths, config, skipped, broken))

    assert len(result) == 0
    assert len(skipped) == 1
    assert "test_skipped_directory" in skipped[0]
    assert len(broken) == 0

    os.remove("test_skipped_directory/file.py")
    os.rmdir("test_skipped_directory")


# LLM-generated content at query #31
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    with open("test_file.py", "w") as f:
        f.write("# test")
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []
    os.remove("test_file.py")

    # Test case 3: Directory with Python files
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# test2")
    with open("test_dir/non_python.txt", "w") as f:
        f.write("text")
    paths = ["test_dir"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.py" in result
    assert skipped == []
    assert broken == []
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.py")
    os.remove("test_dir/non_python.txt")
    os.rmdir("test_dir")

    # Test case 4: Non-existent path
    paths = ["non_existent_path.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path.py"]

    # Test case 5: Skipped file
    os.makedirs("test_skip_dir", exist_ok=True)
    with open("test_skip_dir/skip_me.py", "w") as f:
        f.write("# skip")
    paths = ["test_skip_dir"]
    config = Config(skip=["skip_me.py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath("test_skip_dir/skip_me.py")]
    assert broken == []
    os.remove("test_skip_dir/skip_me.py")
    os.rmdir("test_skip_dir")

    # Test case 6: Follow links
    os.makedirs("test_link_dir", exist_ok=True)
    with open("test_link_dir/target.py", "w") as f:
        f.write("# target")
    os.symlink("test_link_dir", "test_link")
    paths = ["test_link"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert "test_link/target.py" in result
    assert skipped == []
    assert broken == []
    os.remove("test_link")
    os.remove("test_link_dir/target.py")
    os.rmdir("test_link_dir")


# LLM-generated content at query #36
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/test1.py", "w") as f:
        f.write("# test")
    with open(f"{test_dir}/test2.py", "w") as f:
        f.write("# test")
    with open(f"{test_dir}/test.txt", "w") as f:
        f.write("# test")

    # Test directory traversal
    results = list(find([test_dir], config, skipped, broken))
    assert len(results) == 2
    assert all("test_directory" in r for r in results)
    assert all(r.endswith(".py") for r in results)

    # Test with non-existent path
    broken = []
    list(find(["non_existent_path"], config, skipped, broken))
    assert "non_existent_path" in broken

    # Test with single file
    results = list(find([f"{test_dir}/test1.py"], config, skipped, broken))
    assert len(results) == 1
    assert results[0].endswith("test1.py")

    # Test with skipped directory
    config.skip = ["test_directory"]
    skipped = []
    list(find([test_dir], config, skipped, broken))
    assert len(skipped) == 1
    assert test_dir in skipped[0]

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


# LLM-generated content at query #32
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# test")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("# not python")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# test")

        # Test find function
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    result = list(find(["nonexistent_path"], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "nonexistent_path"

    # Test with single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# test")
        tmpfile_path = tmpfile.name

    try:
        result = list(find([tmpfile_path], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == tmpfile_path
        assert len(skipped) == 0
        assert len(broken) == 0
    finally:
        os.unlink(tmpfile_path)

    # Test with skipped file
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "skip_me.py"), "w") as f:
            f.write("# test")

        config_skip = Config(skip=["skip_me.py"])
        result = list(find([tmpdir], config_skip, skipped, broken))
        assert len(result) == 0
        assert len(skipped) == 1
        assert "skip_me.py" in skipped[0]
        assert len(broken) == 0


# LLM-generated content at query #37
#--------------------------

```python
def test_find(tmp_path):
    # Setup test files and directories
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "file1.py").write_text("# Python file")
    (test_dir / "file2.txt").write_text("Not a Python file")
    (test_dir / "file3.py").write_text("# Another Python file")

    sub_dir = test_dir / "sub_dir"
    sub_dir.mkdir()
    (sub_dir / "file4.py").write_text("# Python file in subdirectory")

    skipped_dir = test_dir / "skipped_dir"
    skipped_dir.mkdir()
    (skipped_dir / "file5.py").write_text("# Python file in skipped directory")

    # Test with default config
    config = Config()
    paths = [str(test_dir)]
    skipped = []
    broken = []

    # Test finding files
    found_files = list(find(paths, config, skipped, broken))
    assert len(found_files) == 3
    assert str(test_dir / "file1.py") in found_files
    assert str(test_dir / "file3.py") in found_files
    assert str(sub_dir / "file4.py") in found_files

    # Test skipped files
    assert len(skipped) == 0

    # Test broken paths
    assert len(broken) == 0

    # Test with skipped directory
    config = Config(skip=["skipped_dir"])
    skipped = []
    found_files = list(find(paths, config, skipped, broken))
    assert len(found_files) == 3
    assert str(skipped_dir / "file5.py") not in found_files
    assert len(skipped) == 1
    assert str(skipped_dir) in skipped[0]

    # Test with non-existent path
    broken = []
    paths = [str(test_dir), str(tmp_path / "non_existent")]
    found_files = list(find(paths, config, skipped, broken))
    assert len(found_files) == 3
    assert len(broken) == 1
    assert str(tmp_path / "non_existent") in broken

    # Test with single file path
    paths = [str(test_dir / "file1.py")]
    found_files = list(find(paths, config, skipped, broken))
    assert len(found_files) == 1
    assert str(test_dir / "file1.py") in found_files


# LLM-generated content at query #33
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    with open("test_file.py", "w") as f:
        f.write("# test")
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []
    os.remove("test_file.py")

    # Test case 3: Directory with Python files
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file1.py", "w") as f:
        f.write("# test1")
    with open("test_dir/file2.py", "w") as f:
        f.write("# test2")
    with open("test_dir/non_py.txt", "w") as f:
        f.write("text")
    paths = ["test_dir"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.py" in result
    assert skipped == []
    assert broken == []
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.py")
    os.remove("test_dir/non_py.txt")
    os.rmdir("test_dir")

    # Test case 4: Non-existent path
    paths = ["non_existent_path.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path.py"]

    # Test case 5: Skipped files
    os.makedirs("test_skip_dir", exist_ok=True)
    with open("test_skip_dir/skip_me.py", "w") as f:
        f.write("# skip")
    paths = ["test_skip_dir"]
    config = Config(skip=["skip_me.py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath("test_skip_dir/skip_me.py")]
    assert broken == []
    os.remove("test_skip_dir/skip_me.py")
    os.rmdir("test_skip_dir")

    # Test case 6: Symlinks (if supported)
    if hasattr(os, 'symlink'):
        os.makedirs("test_link_dir", exist_ok=True)
        with open("test_link_dir/target.py", "w") as f:
            f.write("# target")
        os.symlink("test_link_dir/target.py", "test_link_dir/link.py")
        paths = ["test_link_dir"]
        config = Config(follow_links=True)
        skipped = []
        broken = []
        result = list(find(paths, config, skipped, broken))
        assert len(result) == 2
        assert "test_link_dir/target.py" in result
        assert "test_link_dir/link.py" in result
        assert skipped == []
        assert broken == []
        os.remove("test_link_dir/link.py")
        os.remove("test_link_dir/target.py")
        os.rmdir("test_link_dir")


# LLM-generated content at query #38
#--------------------------

```python
def test_find():
    # Test with a directory containing Python files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "test1.py"), "w") as f:
            f.write("# Test file 1")
        with open(os.path.join(tmpdir, "test2.txt"), "w") as f:
            f.write("Not a Python file")
        with open(os.path.join(tmpdir, "subdir", "test3.py"), "w") as f:
            f.write("# Test file 3")

        config = Config()
        paths = [tmpdir]
        skipped = []
        broken = []

        result = list(find(paths, config, skipped, broken))

        assert len(result) == 2
        assert any("test1.py" in r for r in result)
        assert any("test3.py" in r for r in result)
        assert len(skipped) == 0
        assert len(broken) == 0

    # Test with non-existent path
    config = Config()
    paths = ["/nonexistent/path"]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))

    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "/nonexistent/path"

    # Test with a single file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(b"# Single file test")
        tmpfile_path = tmpfile.name

    try:
        config = Config()
        paths = [tmpfile_path]
        skipped = []
        broken = []

        result = list(find(paths, config, skipped, broken))

        assert len(result) == 1
        assert result[0] == tmpfile_path
        assert len(skipped) == 0
        assert len(broken) == 0
    finally:
        os.unlink(tmpfile_path)

    # Test with skipped directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test structure
        os.makedirs(os.path.join(tmpdir, "skipme"))
        with open(os.path.join(tmpdir, "skipme", "test.py"), "w") as f:
            f.write("# Should be skipped")
        with open(os.path.join(tmpdir, "included.py"), "w") as f:
            f.write("# Should be included")

        config = Config(skip=["skipme"])
        paths = [tmpdir]
        skipped = []
        broken = []

        result = list(find(paths, config, skipped, broken))

        assert len(result) == 1
        assert "included.py" in result[0]
        assert len(skipped) == 1
        assert "skipme" in skipped[0]
        assert len(broken) == 0


# LLM-generated content at query #34
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/test1.py", "w") as f:
        f.write("# test")
    with open(f"{test_dir}/test2.py", "w") as f:
        f.write("# test")
    with open(f"{test_dir}/test.txt", "w") as f:
        f.write("# test")

    # Test with non-existent path
    non_existent_path = "non_existent.py"

    # Test with single file
    single_file = "single_test.py"
    with open(single_file, "w") as f:
        f.write("# test")

    # Execute
    results = list(find([test_dir, non_existent_path, single_file], config, skipped, broken))

    # Verify
    assert len(results) == 3
    assert f"{test_dir}/test1.py" in results
    assert f"{test_dir}/test2.py" in results
    assert single_file in results
    assert non_existent_path in broken
    assert len(skipped) == 0

    # Cleanup
    os.remove(f"{test_dir}/test1.py")
    os.remove(f"{test_dir}/test2.py")
    os.remove(f"{test_dir}/test.txt")
    os.rmdir(test_dir)
    os.remove(single_file)


# LLM-generated content at query #39
#--------------------------

```python
def test_find():
    # Setup
    config = Config()
    skipped = []
    broken = []

    # Test with directory containing Python files
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/test1.py", "w") as f:
        f.write("# test")
    with open(f"{test_dir}/test2.py", "w") as f:
        f.write("# test")
    with open(f"{test_dir}/ignore.txt", "w") as f:
        f.write("# ignore")

    # Test directory traversal
    results = list(find([test_dir], config, skipped, broken))
    assert len(results) == 2
    assert all("test_directory" in r for r in results)
    assert all(r.endswith(".py") for r in results)

    # Test with skipped file
    config.skip = ["test_directory/test1.py"]
    skipped = []
    results = list(find([test_dir], config, skipped, broken))
    assert len(results) == 1
    assert "test_directory/test2.py" in results[0]
    assert len(skipped) == 1
    assert "test_directory/test1.py" in skipped[0]

    # Test with non-existent path
    broken = []
    results = list(find(["nonexistent_path"], config, skipped, broken))
    assert len(results) == 0
    assert len(broken) == 1
    assert "nonexistent_path" in broken

    # Test with single file
    results = list(find(["test_directory/test2.py"], config, skipped, broken))
    assert len(results) == 1
    assert "test_directory/test2.py" in results[0]

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


# LLM-generated content at query #35
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 3: Directory with Python files
    paths = ["test_dir"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_dir/file1.py", "test_dir/file2.py"]
    assert skipped == []
    assert broken == []

    # Test case 4: Non-existent path
    paths = ["non_existent_path"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_path"]

    # Test case 5: Skipped directory
    paths = ["skipped_dir"]
    config = Config(skip=["skipped_dir"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ["skipped_dir"]
    assert broken == []

    # Test case 6: Skipped file
    paths = ["skipped_file.py"]
    config = Config(skip=["skipped_file.py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ["skipped_file.py"]
    assert broken == []

    # Test case 7: Mixed paths
    paths = ["test_file.py", "test_dir", "non_existent_path", "skipped_dir"]
    config = Config(skip=["skipped_dir"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py", "test_dir/file1.py", "test_dir/file2.py"]
    assert skipped == ["skipped_dir"]
    assert broken == ["non_existent_path"]


# LLM-generated content at query #40
#--------------------------

```python
def test_find():
    # Test case 1: Empty paths
    paths = []
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

    # Test case 2: Single file path
    paths = ["test_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == ["test_file.py"]
    assert skipped == []
    assert broken == []

    # Test case 3: Non-existent file path
    paths = ["non_existent_file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent_file.py"]

    # Test case 4: Directory with Python files
    paths = ["test_directory"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert all(os.path.isfile(path) and path.endswith(".py") for path in result)
    assert skipped == []
    assert broken == []

    # Test case 5: Directory with skipped files
    paths = ["test_directory"]
    config = Config(skip=["test_directory/skipped_file.py"])
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_directory/skipped_file.py" not in result
    assert "test_directory/skipped_file.py" in skipped
    assert broken == []

    # Test case 6: Mixed paths (files and directories)
    paths = ["test_file.py", "test_directory"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert "test_file.py" in result
    assert all(os.path.isfile(path) and path.endswith(".py") for path in result if path != "test_file.py")
    assert skipped == []
    assert broken == []

    # Test case 7: Symlinks (if applicable)
    paths = ["symlink_directory"]
    config = Config(follow_links=True)
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert all(os.path.isfile(path) and path.endswith(".py") for path in result)
    assert skipped == []
    assert broken == []


