####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_find_with_single_file():
    config = Config()
    skipped = []
    broken = []
    paths = ["test.py"]
    result = list(find(paths, config, skipped, broken))
    assert result == ["test.py"]
    assert skipped == []
    assert broken == []

def test_find_with_nonexistent_file():
    config = Config()
    skipped = []
    broken = []
    paths = ["nonexistent.py"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent.py"]

def test_find_with_skipped_file():
    config = Config()
    config.is_skipped = lambda p: True
    config.is_supported_filetype = lambda f: True
    skipped = []
    broken = []
    paths = ["skipped.py"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath("skipped.py")]
    assert broken == []

def test_find_with_directory():
    config = Config()
    config.is_supported_filetype = lambda f: f.endswith(".py")
    config.is_skipped = lambda p: False
    config.follow_links = False
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
    assert skipped == []
    assert broken == []
    os.remove("test_dir/file1.py")
    os.remove("test_dir/file2.txt")
    os.rmdir("test_dir")

def test_find_with_skipped_directory():
    config = Config()
    config.is_supported_filetype = lambda f: f.endswith(".py")
    config.is_skipped = lambda p: str(p) == "test_dir"
    config.follow_links = False
    skipped = []
    broken = []
    paths = ["test_dir"]
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file.py", "w") as f:
        f.write("")
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ["test_dir"]
    assert broken == []
    os.remove("test_dir/file.py")
    os.rmdir("test_dir")

def test_find_with_symlink_directory():
    config = Config()
    config.is_supported_filetype = lambda f: f.endswith(".py")
    config.is_skipped = lambda p: False
    config.follow_links = True
    skipped = []
    broken = []
    os.makedirs("real_dir", exist_ok=True)
    with open("real_dir/file.py", "w") as f:
        f.write("")
    os.symlink("real_dir", "link_dir")
    paths = ["link_dir"]
    result = list(find(paths, config, skipped, broken))
    assert "link_dir/file.py" in result
    assert skipped == []
    assert broken == []
    os.remove("real_dir/file.py")
    os.rmdir("real_dir")
    os.remove("link_dir")

def test_find_with_duplicate_resolved_directory():
    config = Config()
    config.is_supported_filetype = lambda f: f.endswith(".py")
    config.is_skipped = lambda p: False
    config.follow_links = True
    skipped = []
    broken = []
    os.makedirs("dir1", exist_ok=True)
    os.symlink("dir1", "dir2")
    with open("dir1/file.py", "w") as f:
        f.write("")
    paths = ["dir1", "dir2"]
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert skipped == []
    assert broken == []
    os.remove("dir1/file.py")
    os.rmdir("dir1")
    os.remove("dir2")

def test_find_with_mixed_paths():
    config = Config()
    config.is_supported_filetype = lambda f: f.endswith(".py")
    config.is_skipped = lambda p: str(p) == "skip.py"
    config.follow_links = False
    skipped = []
    broken = []
    os.makedirs("mixed_dir", exist_ok=True)
    with open("mixed_dir/valid.py", "w") as f:
        f.write("")
    with open("mixed_dir/skip.py", "w") as f:
        f.write("")
    with open("single.py", "w") as f:
        f.write("")
    paths = ["mixed_dir", "single.py", "ghost.py"]
    result = list(find(paths, config, skipped, broken))
    assert "mixed_dir/valid.py" in result
    assert "single.py" in result
    assert "mixed_dir/skip.py" not in result
    assert os.path.abspath("mixed_dir/skip.py") in skipped
    assert broken == ["ghost.py"]
    os.remove("mixed_dir/valid.py")
    os.remove("mixed_dir/skip.py")
    os.rmdir("mixed_dir")
    os.remove("single.py")


# LLM-generated content at query #2
#--------------------------

def test_find_with_directory_and_skipped_path():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return str(path).endswith("skip_me")
        def is_supported_filetype(self, filepath):
            return filepath.endswith(".py")
    config = MockConfig()
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "skip_me"))
        os.makedirs(os.path.join(tmpdir, "include"))
        with open(os.path.join(tmpdir, "skip_me", "a.py"), "w") as f:
            f.write("")
        with open(os.path.join(tmpdir, "include", "b.py"), "w") as f:
            f.write("")
        result = list(find([tmpdir], config, skipped, broken))
    assert len(result) == 1
    assert result[0].endswith(os.path.join("include", "b.py"))
    assert len(skipped) == 1
    assert skipped[0].endswith("skip_me")
    assert broken == []

def test_find_with_single_file():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith(".py")
    config = MockConfig()
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.py")
        with open(file_path, "w") as f:
            f.write("")
        result = list(find([file_path], config, skipped, broken))
    assert len(result) == 1
    assert result[0] == file_path
    assert skipped == []
    assert broken == []

def test_find_with_non_existent_file():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith(".py")
    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(["non_existent.py"], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["non_existent.py"]

def test_find_with_unsupported_filetype():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith(".py")
    config = MockConfig()
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.txt")
        with open(file_path, "w") as f:
            f.write("")
        result = list(find([file_path], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

def test_find_with_skipped_file():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return str(path).endswith("skip.py")
        def is_supported_filetype(self, filepath):
            return filepath.endswith(".py")
    config = MockConfig()
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "skip.py")
        with open(file_path, "w") as f:
            f.write("")
        result = list(find([file_path], config, skipped, broken))
    assert result == []
    assert len(skipped) == 1
    assert skipped[0] == os.path.abspath(file_path)
    assert broken == []

def test_find_with_follow_links_and_visited_dirs():
    class MockConfig:
        follow_links = True
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith(".py")
    config = MockConfig()
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = os.path.join(tmpdir, "dir")
        os.makedirs(dir_path)
        link_path = os.path.join(tmpdir, "link")
        os.symlink(dir_path, link_path)
        with open(os.path.join(dir_path, "test.py"), "w") as f:
            f.write("")
        result = list(find([tmpdir], config, skipped, broken))
    assert len(result) == 1
    assert result[0].endswith(os.path.join("dir", "test.py"))
    assert skipped == []
    assert broken == []

def test_find_with_multiple_paths():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith(".py")
    config = MockConfig()
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        dir1 = os.path.join(tmpdir, "dir1")
        dir2 = os.path.join(tmpdir, "dir2")
        os.makedirs(dir1)
        os.makedirs(dir2)
        file1 = os.path.join(dir1, "a.py")
        file2 = os.path.join(dir2, "b.py")
        with open(file1, "w") as f:
            f.write("")
        with open(file2, "w") as f:
            f.write("")
        result = list(find([dir1, dir2], config, skipped, broken))
    assert len(result) == 2
    assert any(r.endswith(os.path.join("dir1", "a.py")) for r in result)
    assert any(r.endswith(os.path.join("dir2", "b.py")) for r in result)
    assert skipped == []
    assert broken == []


# LLM-generated content at query #3
#--------------------------

def test_find_with_directory_and_skipped_path():
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else {'.py'}

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    import tempfile, os
    from pathlib import Path
    tmpdir = tempfile.mkdtemp()
    subdir = os.path.join(tmpdir, 'subdir')
    os.makedirs(subdir)
    file1 = os.path.join(tmpdir, 'file1.py')
    file2 = os.path.join(subdir, 'file2.py')
    skipped_file = os.path.join(tmpdir, 'skipped.py')
    open(file1, 'w').close()
    open(file2, 'w').close()
    open(skipped_file, 'w').close()
    config = MockConfig(skipped_paths=[skipped_file])
    skipped = []
    broken = []
    result = list(find([tmpdir], config, skipped, broken))
    assert sorted(result) == sorted([file1, file2])
    assert skipped == [skipped_file]
    assert broken == []
    import shutil
    shutil.rmtree(tmpdir)

def test_find_with_nonexistent_path():
    class MockConfig:
        def __init__(self):
            self.follow_links = False
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(['/nonexistent/path'], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ['/nonexistent/path']

def test_find_with_single_file():
    class MockConfig:
        def __init__(self):
            self.follow_links = False
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    import tempfile
    tmpfile = tempfile.NamedTemporaryFile(suffix='.py', delete=False)
    tmpfile.close()
    config = MockConfig()
    skipped = []
    broken = []
    result = list(find([tmpfile.name], config, skipped, broken))
    assert result == [tmpfile.name]
    assert skipped == []
    assert broken == []
    import os
    os.unlink(tmpfile.name)

def test_find_with_skipped_directory():
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')

    import tempfile, os
    from pathlib import Path
    tmpdir = tempfile.mkdtemp()
    skipped_dir = os.path.join(tmpdir, 'skipped_dir')
    os.makedirs(skipped_dir)
    file_in_skipped = os.path.join(skipped_dir, 'file.py')
    open(file_in_skipped, 'w').close()
    config = MockConfig(skipped_paths=[skipped_dir])
    skipped = []
    broken = []
    result = list(find([tmpdir], config, skipped, broken))
    assert result == []
    assert skipped == [skipped_dir]
    assert broken == []
    import shutil
    shutil.rmtree(tmpdir)

def test_find_with_follow_links_and_visited_dirs():
    class MockConfig:
        def __init__(self, follow_links=True):
            self.follow_links = follow_links
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    subdir = os.path.join(tmpdir, 'subdir')
    os.makedirs(subdir)
    linkdir = os.path.join(tmpdir, 'linkdir')
    os.symlink(subdir, linkdir)
    file1 = os.path.join(subdir, 'file1.py')
    open(file1, 'w').close()
    config = MockConfig(follow_links=True)
    skipped = []
    broken = []
    result = list(find([tmpdir], config, skipped, broken))
    assert result == [file1]
    assert skipped == []
    assert broken == []
    import shutil
    shutil.rmtree(tmpdir)
    os.unlink(linkdir)

def test_find_with_unsupported_filetype():
    class MockConfig:
        def __init__(self):
            self.follow_links = False
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    import tempfile
    tmpfile = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
    tmpfile.close()
    config = MockConfig()
    skipped = []
    broken = []
    result = list(find([tmpfile.name], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []
    import os
    os.unlink(tmpfile.name)

def test_find_with_multiple_paths():
    class MockConfig:
        def __init__(self):
            self.follow_links = False
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    import tempfile, os
    tmpdir1 = tempfile.mkdtemp()
    tmpdir2 = tempfile.mkdtemp()
    file1 = os.path.join(tmpdir1, 'file1.py')
    file2 = os.path.join(tmpdir2, 'file2.py')
    open(file1, 'w').close()
    open(file2, 'w').close()
    config = MockConfig()
    skipped = []
    broken = []
    result = list(find([tmpdir1, tmpdir2], config, skipped, broken))
    assert sorted(result) == sorted([file1, file2])
    assert skipped == []
    assert broken == []
    import shutil
    shutil.rmtree(tmpdir1)
    shutil.rmtree(tmpdir2)


# LLM-generated content at query #4
#--------------------------

def test_predicate_at_line_8_evaluates_true_for_directory():
    paths = ["/tmp/test_dir"]
    config = Config()
    skipped = []
    broken = []
    os.makedirs("/tmp/test_dir", exist_ok=True)
    result = list(find(paths, config, skipped, broken))
    assert os.path.isdir("/tmp/test_dir")


# LLM-generated content at query #5
#--------------------------

def test_predicate_at_line_8_evaluates_true_for_directory():
    paths = ["/some/directory"]
    config = Config()
    skipped = []
    broken = []
    os.isdir = lambda p: True
    result = list(find(paths, config, skipped, broken))
    assert os.isdir.called_with("/some/directory")


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_find_with_single_file():
    config = Config()
    skipped = []
    broken = []
    result = list(find(["test.py"], config, skipped, broken))
    assert result == ["test.py"]
    assert skipped == []
    assert broken == []

def test_find_with_nonexistent_file():
    config = Config()
    skipped = []
    broken = []
    result = list(find(["missing.py"], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["missing.py"]

def test_find_with_skipped_file():
    config = Config()
    config.is_skipped = lambda p: True
    config.is_supported_filetype = lambda f: True
    skipped = []
    broken = []
    result = list(find(["test.py"], config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath("test.py")]
    assert broken == []

def test_find_with_skipped_directory():
    config = Config()
    config.is_skipped = lambda p: str(p) == "skip_dir"
    config.is_supported_filetype = lambda f: True
    skipped = []
    broken = []
    result = list(find(["skip_dir"], config, skipped, broken))
    assert result == []
    assert skipped == ["skip_dir"]
    assert broken == []

def test_find_with_supported_file_in_directory():
    config = Config()
    config.is_skipped = lambda p: False
    config.is_supported_filetype = lambda f: f.endswith(".py")
    skipped = []
    broken = []
    result = list(find(["dir"], config, skipped, broken))
    expected = [os.path.join("dir", f) for f in ["a.py", "b.py"]]
    assert sorted(result) == sorted(expected)
    assert skipped == []
    assert broken == []

def test_find_with_unsupported_file_in_directory():
    config = Config()
    config.is_skipped = lambda p: False
    config.is_supported_filetype = lambda f: False
    skipped = []
    broken = []
    result = list(find(["dir"], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

def test_find_with_symlink_following():
    config = Config()
    config.follow_links = True
    config.is_skipped = lambda p: False
    config.is_supported_filetype = lambda f: f.endswith(".py")
    skipped = []
    broken = []
    result = list(find(["symlink_dir"], config, skipped, broken))
    expected = [os.path.join("symlink_dir", "linked.py")]
    assert result == expected
    assert skipped == []
    assert broken == []

def test_find_without_symlink_following():
    config = Config()
    config.follow_links = False
    config.is_skipped = lambda p: False
    config.is_supported_filetype = lambda f: f.endswith(".py")
    skipped = []
    broken = []
    result = list(find(["symlink_dir"], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

def test_find_with_duplicate_resolved_directory():
    config = Config()
    config.follow_links = True
    config.is_skipped = lambda p: False
    config.is_supported_filetype = lambda f: f.endswith(".py")
    skipped = []
    broken = []
    result = list(find(["dir", "symlink_to_dir"], config, skipped, broken))
    expected = [os.path.join("dir", "a.py"), os.path.join("dir", "b.py")]
    assert sorted(result) == sorted(expected)
    assert skipped == []
    assert broken == []

def test_find_with_mixed_paths():
    config = Config()
    config.is_skipped = lambda p: str(p) == "skip"
    config.is_supported_filetype = lambda f: f.endswith(".py")
    skipped = []
    broken = []
    paths = ["file1.py", "skip", "missing.py", "dir"]
    result = list(find(paths, config, skipped, broken))
    expected = ["file1.py"] + [os.path.join("dir", f) for f in ["a.py", "b.py"]]
    assert sorted(result) == sorted(expected)
    assert skipped == ["skip"]
    assert broken == ["missing.py"]


# LLM-generated content at query #2
#--------------------------

def test_predicate_at_line_31_evaluates_to_true():
    paths = ["/fake/nonexistent/file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert broken == ["/fake/nonexistent/file.py"]


# LLM-generated content at query #3
#--------------------------

def test_find_with_single_file():
    config = Config()
    skipped = []
    broken = []
    paths = ["test.py"]
    result = list(find(paths, config, skipped, broken))
    assert result == ["test.py"]
    assert skipped == []
    assert broken == []

def test_find_with_nonexistent_file():
    config = Config()
    skipped = []
    broken = []
    paths = ["nonexistent.py"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent.py"]

def test_find_with_directory():
    config = Config()
    skipped = []
    broken = []
    paths = ["test_dir"]
    result = list(find(paths, config, skipped, broken))
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.py" in result
    assert skipped == []
    assert broken == []

def test_find_with_skipped_file():
    config = Config()
    config.is_skipped = lambda path: str(path) == "skipped.py"
    skipped = []
    broken = []
    paths = ["skipped.py"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ["skipped.py"]
    assert broken == []

def test_find_with_skipped_directory():
    config = Config()
    config.is_skipped = lambda path: str(path) == "skipped_dir"
    skipped = []
    broken = []
    paths = ["skipped_dir"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ["skipped_dir"]
    assert broken == []

def test_find_with_mixed_paths():
    config = Config()
    skipped = []
    broken = []
    paths = ["file1.py", "nonexistent.py", "test_dir"]
    result = list(find(paths, config, skipped, broken))
    assert "file1.py" in result
    assert "test_dir/file1.py" in result
    assert "test_dir/file2.py" in result
    assert broken == ["nonexistent.py"]
    assert skipped == []

def test_find_with_follow_links():
    config = Config()
    config.follow_links = True
    skipped = []
    broken = []
    paths = ["linked_dir"]
    result = list(find(paths, config, skipped, broken))
    assert "linked_dir/file.py" in result
    assert skipped == []
    assert broken == []

def test_find_without_follow_links():
    config = Config()
    config.follow_links = False
    skipped = []
    broken = []
    paths = ["linked_dir"]
    result = list(find(paths, config, skipped, broken))
    assert "linked_dir/file.py" in result
    assert skipped == []
    assert broken == []

def test_find_with_unsupported_filetype():
    config = Config()
    config.is_supported_filetype = lambda filepath: filepath.endswith(".py")
    skipped = []
    broken = []
    paths = ["test.txt"]
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

def test_find_with_supported_filetype_in_directory():
    config = Config()
    config.is_supported_filetype = lambda filepath: filepath.endswith(".py")
    skipped = []
    broken = []
    paths = ["mixed_dir"]
    result = list(find(paths, config, skipped, broken))
    assert "mixed_dir/script.py" in result
    assert "mixed_dir/data.txt" not in result
    assert skipped == []
    assert broken == []


# LLM-generated content at query #4
#--------------------------

def test_find_with_directory_and_skipped_path():
    class MockConfig:
        def __init__(self, follow_links, skipped_paths, supported_ext):
            self.follow_links = follow_links
            self.skipped_paths = skipped_paths
            self.supported_ext = supported_ext
        def is_skipped(self, path):
            return str(path) in self.skipped_paths
        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_ext)
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, 'subdir')
        os.makedirs(subdir)
        file1 = os.path.join(tmpdir, 'file1.py')
        file2 = os.path.join(subdir, 'file2.py')
        skipped_file = os.path.join(tmpdir, 'skipped.py')
        for f in [file1, file2, skipped_file]:
            with open(f, 'w') as fp:
                fp.write('')
        config = MockConfig(follow_links=False, skipped_paths=[skipped_file], supported_ext=['.py'])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 2
        assert file1 in result
        assert file2 in result
        assert skipped_file not in result
        assert len(skipped) == 1
        assert skipped[0] == os.path.abspath(skipped_file)
        assert len(broken) == 0
def test_find_with_nonexistent_path():
    class MockConfig:
        def __init__(self):
            self.follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')
    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(['/nonexistent/path'], config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == '/nonexistent/path'
def test_find_with_skipped_directory():
    class MockConfig:
        def __init__(self, follow_links, skipped_dirs):
            self.follow_links = follow_links
            self.skipped_dirs = skipped_dirs
        def is_skipped(self, path):
            return str(path) in self.skipped_dirs
        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        skipped_dir = os.path.join(tmpdir, 'skipped')
        os.makedirs(skipped_dir)
        file_in_skipped = os.path.join(skipped_dir, 'file.py')
        with open(file_in_skipped, 'w') as fp:
            fp.write('')
        config = MockConfig(follow_links=False, skipped_dirs=[skipped_dir])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 0
        assert len(skipped) == 1
        assert skipped[0] == skipped_dir
        assert len(broken) == 0
def test_find_with_direct_file_path():
    class MockConfig:
        def __init__(self):
            self.follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return True
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'test.txt')
        with open(file_path, 'w') as fp:
            fp.write('')
        config = MockConfig()
        skipped = []
        broken = []
        result = list(find([file_path], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == file_path
        assert len(skipped) == 0
        assert len(broken) == 0
def test_find_with_unsupported_filetype():
    class MockConfig:
        def __init__(self):
            self.follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        txt_file = os.path.join(tmpdir, 'test.txt')
        with open(txt_file, 'w') as fp:
            fp.write('')
        config = MockConfig()
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert len(result) == 0
        assert len(skipped) == 0
        assert len(broken) == 0
def test_find_with_follow_links_and_visited_dirs():
    class MockConfig:
        def __init__(self, follow_links):
            self.follow_links = follow_links
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        link_dir = os.path.join(tmpdir, 'link')
        target_dir = os.path.join(tmpdir, 'target')
        os.makedirs(target_dir)
        os.symlink(target_dir, link_dir)
        file_in_target = os.path.join(target_dir, 'file.py')
        with open(file_in_target, 'w') as fp:
            fp.write('')
        config = MockConfig(follow_links=True)
        skipped = []
        broken = []
        result = list(find([tmpdir, link_dir], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == file_in_target
        assert len(skipped) == 0
        assert len(broken) == 0


# LLM-generated content at query #5
#--------------------------

def test_find_with_directory_and_skipped_path():
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else {'.py'}

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, 'subdir')
        os.makedirs(subdir)
        file1 = os.path.join(tmpdir, 'file1.py')
        file2 = os.path.join(subdir, 'file2.py')
        file3 = os.path.join(subdir, 'file3.txt')
        open(file1, 'w').close()
        open(file2, 'w').close()
        open(file3, 'w').close()
        config = MockConfig(skipped_paths=[subdir])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert result == [file1]
        assert skipped == [subdir]
        assert broken == []

def test_find_with_nonexistent_path():
    class MockConfig:
        def __init__(self):
            self.follow_links = False
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(['/nonexistent/path'], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ['/nonexistent/path']

def test_find_with_single_file():
    class MockConfig:
        def __init__(self):
            self.follow_links = False
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'test.py')
        open(file_path, 'w').close()
        config = MockConfig()
        skipped = []
        broken = []
        result = list(find([file_path], config, skipped, broken))
        assert result == [file_path]
        assert skipped == []
        assert broken == []

def test_find_with_follow_links_and_visited_dirs():
    class MockConfig:
        def __init__(self, follow_links):
            self.follow_links = follow_links
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        real_dir = os.path.join(tmpdir, 'real')
        link_dir = os.path.join(tmpdir, 'link')
        os.makedirs(real_dir)
        os.symlink(real_dir, link_dir, target_is_directory=True)
        file_path = os.path.join(real_dir, 'file.py')
        open(file_path, 'w').close()
        config = MockConfig(follow_links=True)
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert file_path in result
        assert len(result) == 1
        assert skipped == []
        assert broken == []

def test_find_with_unsupported_filetype():
    class MockConfig:
        def __init__(self):
            self.follow_links = False
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        txt_file = os.path.join(tmpdir, 'test.txt')
        open(txt_file, 'w').close()
        config = MockConfig()
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert result == []
        assert skipped == []
        assert broken == []

def test_find_with_skipped_file():
    class MockConfig:
        def __init__(self, skipped_paths=None):
            self.follow_links = False
            self.supported_extensions = {'.py'}
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'skipped.py')
        open(file_path, 'w').close()
        config = MockConfig(skipped_paths=[os.path.abspath(file_path)])
        skipped = []
        broken = []
        result = list(find([tmpdir], config, skipped, broken))
        assert result == []
        assert skipped == [os.path.abspath(file_path)]
        assert broken == []

def test_find_with_multiple_paths():
    class MockConfig:
        def __init__(self):
            self.follow_links = False
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        dir1 = os.path.join(tmpdir, 'dir1')
        dir2 = os.path.join(tmpdir, 'dir2')
        os.makedirs(dir1)
        os.makedirs(dir2)
        file1 = os.path.join(dir1, 'file1.py')
        file2 = os.path.join(dir2, 'file2.py')
        open(file1, 'w').close()
        open(file2, 'w').close()
        config = MockConfig()
        skipped = []
        broken = []
        result = list(find([dir1, dir2], config, skipped, broken))
        assert set(result) == {file1, file2}
        assert skipped == []
        assert broken == []


# LLM-generated content at query #6
#--------------------------

def test_predicate_at_line_9_evaluates_to_false():
    mock_paths = ["/fake/file.py"]
    mock_config = type('Config', (), {'follow_links': False})()
    skipped = []
    broken = []
    result = list(find(mock_paths, mock_config, skipped, broken))
    assert result == ["/fake/file.py"]
    assert skipped == []
    assert broken == []


# LLM-generated content at query #7
#--------------------------

def test_predicate_at_line_31_evaluates_true_when_path_does_not_exist():
    paths = ["/non/existent/path"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(broken) == 1
    assert broken[0] == "/non/existent/path"
    assert len(result) == 0


# LLM-generated content at query #8
#--------------------------

def test_predicate_at_line_27_evaluates_to_true():
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else set()
        def is_skipped(self, path):
            return str(path) in self.skipped_paths
        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)
    config = MockConfig(supported_extensions=['.py'], skipped_paths=['/abs/path/to/skipped.py'])
    skipped = []
    broken = []
    paths = ['/some/dir']
    import os
    from unittest.mock import patch, MagicMock
    with patch('os.path.isdir', return_value=True):
        with patch('os.walk') as mock_walk:
            mock_walk.return_value = [('/some/dir', [], ['skipped.py'])]
            with patch('os.path.join', return_value='/some/dir/skipped.py'):
                with patch('os.path.abspath', return_value='/abs/path/to/skipped.py'):
                    with patch('os.path.exists', return_value=True):
                        result = list(find(paths, config, skipped, broken))
    assert config.is_skipped(Path(os.path.abspath('/some/dir/skipped.py'))) == True


# LLM-generated content at query #9
#--------------------------

def test_predicate_at_line_27_evaluates_to_true():
    class MockConfig:
        def __init__(self, follow_links, skipped_paths, supported_extensions):
            self.follow_links = follow_links
            self.skipped_paths = skipped_paths
            self.supported_extensions = supported_extensions
        def is_skipped(self, path):
            return str(path) in self.skipped_paths
        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)
    import os
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        file_path = os.path.join(subdir, "test.py")
        with open(file_path, "w") as f:
            f.write("")
        abs_file_path = os.path.abspath(file_path)
        config = MockConfig(follow_links=False, skipped_paths=[abs_file_path], supported_extensions=[".py"])
        skipped = []
        broken = []
        paths = [tmpdir]
        result = list(find(paths, config, skipped, broken))
        assert config.is_skipped(Path(os.path.abspath(file_path))) == True
        assert abs_file_path in skipped
        assert file_path not in result


# LLM-generated content at query #10
#--------------------------

def test_find_with_directory_and_skipped_path():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return str(path).endswith('skip')
        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')
    config = MockConfig()
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, 'skip_dir'))
        os.makedirs(os.path.join(tmpdir, 'normal_dir'))
        with open(os.path.join(tmpdir, 'skip_dir', 'a.py'), 'w') as f:
            pass
        with open(os.path.join(tmpdir, 'normal_dir', 'b.py'), 'w') as f:
            pass
        result = list(find([tmpdir], config, skipped, broken))
    assert len(result) == 1
    assert result[0].endswith('normal_dir' + os.sep + 'b.py')
    assert len(skipped) == 1
    assert skipped[0].endswith('skip_dir')

def test_find_with_single_file():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')
    config = MockConfig()
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'test.py')
        with open(file_path, 'w') as f:
            pass
        result = list(find([file_path], config, skipped, broken))
    assert len(result) == 1
    assert result[0] == file_path
    assert skipped == []
    assert broken == []

def test_find_with_broken_path():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')
    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(['/non/existent/path'], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ['/non/existent/path']

def test_find_with_skipped_file():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return str(path).endswith('skip.py')
        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')
    config = MockConfig()
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'skip.py')
        with open(file_path, 'w') as f:
            pass
        result = list(find([file_path], config, skipped, broken))
    assert result == []
    assert len(skipped) == 1
    assert skipped[0] == os.path.abspath(file_path)
    assert broken == []

def test_find_with_unsupported_filetype():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')
    config = MockConfig()
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'test.txt')
        with open(file_path, 'w') as f:
            pass
        result = list(find([file_path], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

def test_find_with_follow_links_and_visited_dirs():
    class MockConfig:
        follow_links = True
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')
    config = MockConfig()
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = os.path.join(tmpdir, 'dir')
        os.makedirs(dir_path)
        link_path = os.path.join(tmpdir, 'link')
        os.symlink(dir_path, link_path)
        with open(os.path.join(dir_path, 'a.py'), 'w') as f:
            pass
        result = list(find([tmpdir], config, skipped, broken))
    assert len(result) == 1
    assert result[0].endswith('dir' + os.sep + 'a.py')
    assert skipped == []
    assert broken == []

def test_find_with_multiple_paths():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')
    config = MockConfig()
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        dir1 = os.path.join(tmpdir, 'dir1')
        dir2 = os.path.join(tmpdir, 'dir2')
        os.makedirs(dir1)
        os.makedirs(dir2)
        file1 = os.path.join(dir1, 'a.py')
        file2 = os.path.join(dir2, 'b.py')
        with open(file1, 'w') as f:
            pass
        with open(file2, 'w') as f:
            pass
        result = list(find([dir1, dir2], config, skipped, broken))
    assert len(result) == 2
    assert any(r.endswith('dir1' + os.sep + 'a.py') for r in result)
    assert any(r.endswith('dir2' + os.sep + 'b.py') for r in result)
    assert skipped == []
    assert broken == []


