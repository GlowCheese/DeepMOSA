####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_find_with_directory_and_skipped_path():
    import os
    import tempfile
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else {'.py'}
        def is_skipped(self, path):
            return str(path) in self.skipped_paths
        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)
    config = MockConfig(skipped_paths=[])
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, 'subdir')
        os.makedirs(subdir)
        file1 = os.path.join(tmpdir, 'file1.py')
        file2 = os.path.join(subdir, 'file2.py')
        open(file1, 'w').close()
        open(file2, 'w').close()
        paths = [tmpdir]
        result = list(find(paths, config, skipped, broken))
    assert len(result) == 2
    assert file1 in result
    assert file2 in result
    assert skipped == []
    assert broken == []

def test_find_with_skipped_directory():
    import os
    import tempfile
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else {'.py'}
        def is_skipped(self, path):
            return str(path) in self.skipped_paths
        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)
    config = MockConfig(skipped_paths=[])
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, 'subdir')
        os.makedirs(subdir)
        file1 = os.path.join(tmpdir, 'file1.py')
        file2 = os.path.join(subdir, 'file2.py')
        open(file1, 'w').close()
        open(file2, 'w').close()
        config.skipped_paths.add(subdir)
        paths = [tmpdir]
        result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert file1 in result
    assert file2 not in result
    assert len(skipped) == 1
    assert skipped[0] == os.path.abspath(file2)
    assert broken == []

def test_find_with_skipped_file():
    import os
    import tempfile
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else {'.py'}
        def is_skipped(self, path):
            return str(path) in self.skipped_paths
        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)
    config = MockConfig(skipped_paths=[])
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, 'file1.py')
        file2 = os.path.join(tmpdir, 'file2.py')
        open(file1, 'w').close()
        open(file2, 'w').close()
        config.skipped_paths.add(os.path.abspath(file1))
        paths = [tmpdir]
        result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert file2 in result
    assert file1 not in result
    assert len(skipped) == 1
    assert skipped[0] == os.path.abspath(file1)
    assert broken == []

def test_find_with_broken_path():
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else {'.py'}
        def is_skipped(self, path):
            return str(path) in self.skipped_paths
        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)
    config = MockConfig(skipped_paths=[])
    skipped = []
    broken = []
    paths = ['/nonexistent/path']
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert skipped == []
    assert broken == ['/nonexistent/path']

def test_find_with_file_path():
    import os
    import tempfile
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else {'.py'}
        def is_skipped(self, path):
            return str(path) in self.skipped_paths
        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)
    config = MockConfig(skipped_paths=[])
    skipped = []
    broken = []
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
        filepath = f.name
    paths = [filepath]
    result = list(find(paths, config, skipped, broken))
    os.unlink(filepath)
    assert len(result) == 1
    assert result[0] == filepath
    assert skipped == []
    assert broken == []

def test_find_with_unsupported_filetype():
    import os
    import tempfile
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else {'.py'}
        def is_skipped(self, path):
            return str(path) in self.skipped_paths
        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)
    config = MockConfig(skipped_paths=[])
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, 'file1.txt')
        open(file1, 'w').close()
        paths = [tmpdir]
        result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert skipped == []
    assert broken == []

def test_find_with_follow_links():
    import os
    import tempfile
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else {'.py'}
        def is_skipped(self, path):
            return str(path) in self.skipped_paths
        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)
    config = MockConfig(follow_links=True, skipped_paths=[])
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, 'subdir')
        os.maked


# LLM-generated content at query #2
#--------------------------

def test_predicate_at_line_7_evaluates_to_false():
    config = Config()
    config.follow_links = False
    config.is_skipped = lambda path: False
    config.is_supported_filetype = lambda filepath: True
    paths = ["/non_existent_file.py"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert broken == ["/non_existent_file.py"]
    assert skipped == []


# LLM-generated content at query #3
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
        with open(os.path.join(tmpdir, 'normal_dir', 'a.py'), 'w') as f:
            pass
        with open(os.path.join(tmpdir, 'skip_dir', 'b.py'), 'w') as f:
            pass
        result = list(find([tmpdir], config, skipped, broken))
    assert len(result) == 1
    assert result[0].endswith('a.py')
    assert any('skip_dir' in s for s in skipped)
    assert broken == []

def test_find_with_nonexistent_path():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')
    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(['/nonexistent/path'], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ['/nonexistent/path']

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
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
        fname = f.name
    try:
        result = list(find([fname], config, skipped, broken))
        assert result == [fname]
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(fname)

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
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        fname = f.name
    try:
        result = list(find([fname], config, skipped, broken))
        assert result == []
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(fname)

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
    with tempfile.NamedTemporaryFile(suffix='skip.py', delete=False) as f:
        fname = f.name
    try:
        result = list(find([fname], config, skipped, broken))
        assert result == []
        assert skipped == [os.path.abspath(fname)]
        assert broken == []
    finally:
        os.unlink(fname)

def test_find_with_follow_links_and_cycle():
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
        link_dir = os.path.join(tmpdir, 'link')
        os.symlink(tmpdir, link_dir)
        result = list(find([tmpdir], config, skipped, broken))
        assert result == []
        assert broken == []


# LLM-generated content at query #4
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

    import os
    import tempfile
    tmpdir = tempfile.mkdtemp()
    subdir = os.path.join(tmpdir, 'subdir')
    os.makedirs(subdir)
    open(os.path.join(tmpdir, 'file1.py'), 'w').close()
    open(os.path.join(subdir, 'file2.py'), 'w').close()
    open(os.path.join(tmpdir, 'file3.txt'), 'w').close()
    config = MockConfig(skipped_paths=[subdir])
    skipped = []
    broken = []
    result = list(find([tmpdir], config, skipped, broken))
    assert len(result) == 1
    assert any('file1.py' in r for r in result)
    assert len(skipped) == 1
    assert subdir in skipped[0]
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
    os.unlink(tmpfile.name)

def test_find_with_skipped_file():
    class MockConfig:
        def __init__(self, skipped_paths=None):
            self.follow_links = False
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    import tempfile
    tmpfile = tempfile.NamedTemporaryFile(suffix='.py', delete=False)
    tmpfile.close()
    config = MockConfig(skipped_paths=[tmpfile.name])
    skipped = []
    broken = []
    result = list(find([tmpfile.name], config, skipped, broken))
    assert result == []
    assert len(skipped) == 1
    assert tmpfile.name in skipped[0]
    assert broken == []
    os.unlink(tmpfile.name)

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
    os.unlink(tmpfile.name)

def test_find_with_follow_links_and_visited_dirs():
    class MockConfig:
        def __init__(self, follow_links=False):
            self.follow_links = follow_links
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    import os
    import tempfile
    tmpdir = tempfile.mkdtemp()
    subdir = os.path.join(tmpdir, 'subdir')
    os.makedirs(subdir)
    linkdir = os.path.join(tmpdir, 'linkdir')
    os.symlink(subdir, linkdir)
    open(os.path.join(subdir, 'file.py'), 'w').close()
    config = MockConfig(follow_links=True)
    skipped = []
    broken = []
    result = list(find([tmpdir], config, skipped, broken))
    assert len(result) == 1
    assert 'file.py' in result[0]
    config2 = MockConfig(follow_links=False)
    skipped2 = []
    broken2 = []
    result2 = list(find([tmpdir], config2, skipped2, broken2))
    assert len(result2) == 1
    assert 'file.py' in result2[0]
    os.unlink(linkdir)
    import shutil
    shutil.rmtree(tmpdir)


# LLM-generated content at query #5
#--------------------------

def test_predicate_at_line_8_evaluates_true_for_directory():
    paths = ["/some/directory"]
    config = Config()
    skipped = []
    broken = []
    os.path.isdir = lambda p: True
    result = list(find(paths, config, skipped, broken))
    assert os.path.isdir.called_with("/some/directory")


# LLM-generated content at query #6
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

    import os
    import tempfile
    tmpdir = tempfile.mkdtemp()
    subdir = os.path.join(tmpdir, 'subdir')
    os.makedirs(subdir)
    file1 = os.path.join(tmpdir, 'file1.py')
    file2 = os.path.join(subdir, 'file2.py')
    open(file1, 'w').close()
    open(file2, 'w').close()
    config = MockConfig(skipped_paths=[subdir])
    skipped = []
    broken = []
    result = list(find([tmpdir], config, skipped, broken))
    expected_files = [file1]
    assert sorted(result) == sorted(expected_files)
    assert skipped == [subdir]
    assert broken == []
    import shutil
    shutil.rmtree(tmpdir)

def test_find_with_nonexistent_path():
    class MockConfig:
        def __init__(self):
            self.follow_links = False
            self.skipped_paths = set()
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')

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
            self.skipped_paths = set()
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')

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
    os.unlink(tmpfile.name)

def test_find_with_unsupported_filetype():
    class MockConfig:
        def __init__(self):
            self.follow_links = False
            self.skipped_paths = set()
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')

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
    os.unlink(tmpfile.name)

def test_find_with_follow_links_and_cycle():
    class MockConfig:
        def __init__(self, follow_links):
            self.follow_links = follow_links
            self.skipped_paths = set()
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')

    import os
    import tempfile
    tmpdir = tempfile.mkdtemp()
    linkdir = os.path.join(tmpdir, 'link')
    os.symlink(tmpdir, linkdir)
    config = MockConfig(follow_links=True)
    skipped = []
    broken = []
    result = list(find([tmpdir], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []
    os.remove(linkdir)
    os.rmdir(tmpdir)

def test_find_with_skipped_file():
    class MockConfig:
        def __init__(self, skipped_paths=None):
            self.follow_links = False
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')

    import os
    import tempfile
    tmpfile = tempfile.NamedTemporaryFile(suffix='.py', delete=False)
    tmpfile.close()
    config = MockConfig(skipped_paths=[os.path.abspath(tmpfile.name)])
    skipped = []
    broken = []
    result = list(find([tmpfile.name], config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath(tmpfile.name)]
    assert broken == []
    os.unlink(tmpfile.name)

def test_find_with_multiple_paths():
    class MockConfig:
        def __init__(self):
            self.follow_links = False
            self.skipped_paths = set()
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return False

        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')

    import os
    import tempfile
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
    expected_files = [file1, file2]
    assert sorted(result) == sorted(expected_files)
    assert skipped == []
    assert broken == []
    import shutil
    shutil.rmtree(tmpdir1)
    shutil.rmtree(tmpdir2)


# LLM-generated content at query #7
#--------------------------

def test_predicate_at_line_8_evaluates_true_for_directory():
    paths = ["/some/directory"]
    config = Config()
    skipped = []
    broken = []
    os.path.isdir = lambda p: True
    result = list(find(paths, config, skipped, broken))
    assert os.path.isdir.called_with("/some/directory")


# LLM-generated content at query #8
#--------------------------

def test_find_with_single_file():
    config = Config()
    config.is_supported_filetype = lambda x: x.endswith('.py')
    config.is_skipped = lambda x: False
    config.follow_links = False
    skipped = []
    broken = []
    paths = ['/tmp/test.py']
    result = list(find(paths, config, skipped, broken))
    assert result == ['/tmp/test.py']
    assert skipped == []
    assert broken == []

def test_find_with_nonexistent_file():
    config = Config()
    config.is_supported_filetype = lambda x: x.endswith('.py')
    config.is_skipped = lambda x: False
    config.follow_links = False
    skipped = []
    broken = []
    paths = ['/tmp/nonexistent.py']
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ['/tmp/nonexistent.py']

def test_find_with_skipped_file():
    config = Config()
    config.is_supported_filetype = lambda x: x.endswith('.py')
    config.is_skipped = lambda x: str(x) == '/tmp/skipped.py'
    config.follow_links = False
    skipped = []
    broken = []
    paths = ['/tmp/skipped.py']
    result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ['/tmp/skipped.py']
    assert broken == []

def test_find_with_directory():
    config = Config()
    config.is_supported_filetype = lambda x: x.endswith('.py')
    config.is_skipped = lambda x: False
    config.follow_links = False
    skipped = []
    broken = []
    paths = ['/tmp/dir']
    with patch('os.walk') as mock_walk:
        mock_walk.return_value = [('/tmp/dir', [], ['file1.py', 'file2.txt'])]
        result = list(find(paths, config, skipped, broken))
    assert result == ['/tmp/dir/file1.py']
    assert skipped == []
    assert broken == []

def test_find_with_skipped_directory():
    config = Config()
    config.is_supported_filetype = lambda x: x.endswith('.py')
    config.is_skipped = lambda x: str(x) == '/tmp/dir/skip'
    config.follow_links = False
    skipped = []
    broken = []
    paths = ['/tmp/dir']
    with patch('os.walk') as mock_walk:
        mock_walk.return_value = [('/tmp/dir', ['skip'], []), ('/tmp/dir/skip', [], ['file.py'])]
        result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == ['/tmp/dir/skip']
    assert broken == []

def test_find_with_symlink_following():
    config = Config()
    config.is_supported_filetype = lambda x: x.endswith('.py')
    config.is_skipped = lambda x: False
    config.follow_links = True
    skipped = []
    broken = []
    paths = ['/tmp/dir']
    with patch('os.walk') as mock_walk:
        mock_walk.return_value = [('/tmp/dir', [], ['link.py'])]
        result = list(find(paths, config, skipped, broken))
    assert result == ['/tmp/dir/link.py']
    assert skipped == []
    assert broken == []

def test_find_with_duplicate_resolved_path():
    config = Config()
    config.is_supported_filetype = lambda x: x.endswith('.py')
    config.is_skipped = lambda x: False
    config.follow_links = False
    skipped = []
    broken = []
    paths = ['/tmp/dir']
    visited_dirs = set()
    with patch('os.walk') as mock_walk, patch('pathlib.Path.resolve') as mock_resolve:
        mock_walk.return_value = [('/tmp/dir', ['subdir'], [])]
        mock_resolve.return_value = Path('/tmp/real')
        visited_dirs.add(Path('/tmp/real'))
        result = list(find(paths, config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

def test_find_with_multiple_paths():
    config = Config()
    config.is_supported_filetype = lambda x: x.endswith('.py')
    config.is_skipped = lambda x: False
    config.follow_links = False
    skipped = []
    broken = []
    paths = ['/tmp/file1.py', '/tmp/dir', '/tmp/missing.py']
    with patch('os.walk') as mock_walk:
        mock_walk.return_value = [('/tmp/dir', [], ['file2.py'])]
        result = list(find(paths, config, skipped, broken))
    assert result == ['/tmp/file1.py', '/tmp/dir/file2.py']
    assert skipped == []
    assert broken == ['/tmp/missing.py']

def test_find_with_skipped_file_in_directory():
    config = Config()
    config.is_supported_filetype = lambda x: x.endswith('.py')
    config.is_skipped = lambda x: str(x) == '/tmp/dir/skipped.py'
    config.follow_links = False
    skipped = []
    broken = []
    paths = ['/tmp/dir']
    with patch('os.walk') as mock_walk:
        mock_walk.return_value = [('/tmp/dir', [], ['skipped.py', 'included.py'])]
        result = list(find(paths, config, skipped, broken))
    assert result == ['/tmp/dir/included.py']
    assert skipped == ['/tmp/dir/skipped.py']
    assert broken == []


# LLM-generated content at query #9
#--------------------------

def test_predicate_at_line_8_true_for_directory():
    config = Config()
    config.follow_links = False
    config.is_skipped = lambda p: False
    config.is_supported_filetype = lambda p: True
    skipped = []
    broken = []
    paths = ["/tmp/test_dir"]
    os.makedirs("/tmp/test_dir", exist_ok=True)
    result = list(find(paths, config, skipped, broken))
    assert os.path.isdir("/tmp/test_dir")


# LLM-generated content at query #10
#--------------------------

def test_predicate_at_line_8_evaluates_to_true():
    import os
    class Config:
        follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return True
    config = Config()
    skipped = []
    broken = []
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    paths = [test_dir]
    result = list(find(paths, config, skipped, broken))
    assert os.path.isdir(paths[0]) == True
    os.rmdir(test_dir)


# LLM-generated content at query #11
#--------------------------

def test_predicate_at_line_8_evaluates_to_true_for_directory():
    paths = ["/some/directory"]
    config = Config()
    skipped = []
    broken = []
    os.path.isdir = lambda p: True
    result = list(find(paths, config, skipped, broken))
    assert os.path.isdir.called_with("/some/directory")


# LLM-generated content at query #12
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
    result = list(find(["nonexistent.py"], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent.py"]

def test_find_with_skipped_file():
    config = Config()
    config.is_skipped = lambda p: True
    skipped = []
    broken = []
    result = list(find(["test.py"], config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath("test.py")]
    assert broken == []

def test_find_with_unsupported_filetype():
    config = Config()
    config.is_supported_filetype = lambda f: False
    skipped = []
    broken = []
    result = list(find(["test.txt"], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

def test_find_with_directory():
    config = Config()
    config.is_supported_filetype = lambda f: f.endswith(".py")
    config.is_skipped = lambda p: False
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "file1.py")
        file2 = os.path.join(tmpdir, "file2.py")
        open(file1, "w").close()
        open(file2, "w").close()
        result = list(find([tmpdir], config, skipped, broken))
        assert set(result) == {file1, file2}
        assert skipped == []
        assert broken == []

def test_find_with_skipped_directory():
    config = Config()
    config.is_supported_filetype = lambda f: f.endswith(".py")
    config.is_skipped = lambda p: True
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        file1 = os.path.join(subdir, "file1.py")
        open(file1, "w").close()
        result = list(find([tmpdir], config, skipped, broken))
        assert result == []
        assert skipped == [subdir]
        assert broken == []

def test_find_with_symlink_following():
    config = Config()
    config.follow_links = True
    config.is_supported_filetype = lambda f: f.endswith(".py")
    config.is_skipped = lambda p: False
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        real_dir = os.path.join(tmpdir, "real")
        os.makedirs(real_dir)
        link_dir = os.path.join(tmpdir, "link")
        os.symlink(real_dir, link_dir)
        file1 = os.path.join(real_dir, "file1.py")
        open(file1, "w").close()
        result = list(find([link_dir], config, skipped, broken))
        assert result == [file1]
        assert skipped == []
        assert broken == []

def test_find_with_duplicate_resolved_path():
    config = Config()
    config.follow_links = True
    config.is_supported_filetype = lambda f: f.endswith(".py")
    config.is_skipped = lambda p: False
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        real_dir = os.path.join(tmpdir, "real")
        os.makedirs(real_dir)
        link_dir = os.path.join(tmpdir, "link")
        os.symlink(real_dir, link_dir)
        file1 = os.path.join(real_dir, "file1.py")
        open(file1, "w").close()
        result = list(find([real_dir, link_dir], config, skipped, broken))
        assert result == [file1]
        assert skipped == []
        assert broken == []


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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
        with open(os.path.join(tmpdir, 'normal_dir', 'a.py'), 'w') as f:
            f.write('')
        with open(os.path.join(tmpdir, 'skip_dir', 'b.py'), 'w') as f:
            f.write('')
        result = list(find([tmpdir], config, skipped, broken))
    assert len(result) == 1
    assert result[0].endswith('normal_dir/a.py')
    assert len(skipped) == 1
    assert skipped[0].endswith('skip_dir')
    assert broken == []

def test_find_with_nonexistent_path():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')
    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(['/nonexistent/path'], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ['/nonexistent/path']

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
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
        f.write(b'')
        filepath = f.name
    try:
        result = list(find([filepath], config, skipped, broken))
        assert result == [filepath]
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(filepath)

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
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b'')
        filepath = f.name
    try:
        result = list(find([filepath], config, skipped, broken))
        assert result == []
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(filepath)

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
    with tempfile.NamedTemporaryFile(suffix='skip.py', delete=False) as f:
        f.write(b'')
        filepath = f.name
    try:
        result = list(find([filepath], config, skipped, broken))
        assert result == []
        assert skipped == [os.path.abspath(filepath)]
        assert broken == []
    finally:
        os.unlink(filepath)

def test_find_with_follow_links_and_cycle():
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
        dir_a = os.path.join(tmpdir, 'a')
        dir_b = os.path.join(tmpdir, 'b')
        os.makedirs(dir_a)
        os.makedirs(dir_b)
        os.symlink(dir_a, os.path.join(dir_b, 'link_to_a'))
        os.symlink(dir_b, os.path.join(dir_a, 'link_to_b'))
        with open(os.path.join(dir_a, 'file.py'), 'w') as f:
            f.write('')
        result = list(find([tmpdir], config, skipped, broken))
    assert len(result) == 1
    assert result[0].endswith('file.py')
    assert skipped == []
    assert broken == []


# LLM-generated content at query #2
#--------------------------

def test_predicate_at_line_7_evaluates_to_true():
    paths = ["/some/directory"]
    config = Config()
    config.follow_links = True
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert True


# LLM-generated content at query #3
#--------------------------

def test_find_with_directory_and_skipped_path():
    class MockConfig:
        def __init__(self, follow_links=False):
            self.follow_links = follow_links
            self.skipped_paths = set()
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    config = MockConfig()
    config.skipped_paths.add('/test/skip_dir')
    skipped = []
    broken = []
    paths = ['/test']
    result = list(find(paths, config, skipped, broken))
    assert skipped == ['/test/skip_dir']
    assert broken == []
    assert result == []

def test_find_with_single_file():
    class MockConfig:
        def __init__(self, follow_links=False):
            self.follow_links = follow_links
            self.skipped_paths = set()
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    config = MockConfig()
    skipped = []
    broken = []
    paths = ['/test/file.py']
    result = list(find(paths, config, skipped, broken))
    assert skipped == []
    assert broken == []
    assert result == ['/test/file.py']

def test_find_with_nonexistent_path():
    class MockConfig:
        def __init__(self, follow_links=False):
            self.follow_links = follow_links
            self.skipped_paths = set()
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    config = MockConfig()
    skipped = []
    broken = []
    paths = ['/nonexistent']
    result = list(find(paths, config, skipped, broken))
    assert skipped == []
    assert broken == ['/nonexistent']
    assert result == []

def test_find_with_skipped_file():
    class MockConfig:
        def __init__(self, follow_links=False):
            self.follow_links = follow_links
            self.skipped_paths = set()
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    config = MockConfig()
    config.skipped_paths.add('/test/skipped.py')
    skipped = []
    broken = []
    paths = ['/test/skipped.py']
    result = list(find(paths, config, skipped, broken))
    assert skipped == ['/test/skipped.py']
    assert broken == []
    assert result == []

def test_find_with_unsupported_filetype():
    class MockConfig:
        def __init__(self, follow_links=False):
            self.follow_links = follow_links
            self.skipped_paths = set()
            self.supported_extensions = {'.py'}

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    config = MockConfig()
    skipped = []
    broken = []
    paths = ['/test/file.txt']
    result = list(find(paths, config, skipped, broken))
    assert skipped == []
    assert broken == []
    assert result == []


# LLM-generated content at query #4
#--------------------------

def test_predicate_at_line_8_evaluates_to_true():
    import os
    import tempfile
    from pathlib import Path
    from typing import Iterable, Iterator

    class Config:
        def __init__(self, follow_links=False):
            self.follow_links = follow_links
            self._skipped_paths = set()
            self._supported_extensions = {'.py'}

        def is_skipped(self, path):
            return str(path) in self._skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self._supported_extensions)

        def add_skipped_path(self, path):
            self._skipped_paths.add(str(path))

    def find(paths: Iterable[str], config: Config, skipped: list[str], broken: list[str]) -> Iterator[str]:
        visited_dirs: set[Path] = set()
        for path in paths:
            if os.path.isdir(path):
                for dirpath, dirnames, filenames in os.walk(path, topdown=True, followlinks=config.follow_links):
                    base_path = Path(dirpath)
                    for dirname in list(dirnames):
                        full_path = base_path / dirname
                        resolved_path = full_path.resolve()
                        if config.is_skipped(full_path):
                            skipped.append(str(full_path))
                            dirnames.remove(dirname)
                        else:
                            if resolved_path in visited_dirs:
                                dirnames.remove(dirname)
                        visited_dirs.add(resolved_path)
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if config.is_supported_filetype(filepath):
                            if config.is_skipped(Path(os.path.abspath(filepath))):
                                skipped.append(os.path.abspath(filepath))
                            else:
                                yield filepath
            elif not os.path.exists(path):
                broken.append(path)
            else:
                yield path

    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = os.path.join(tmpdir, "test_dir")
        os.makedirs(dir_path)
        config = Config()
        skipped = []
        broken = []
        result = list(find([dir_path], config, skipped, broken))
        assert os.path.isdir(dir_path)


# LLM-generated content at query #5
#--------------------------

def test_predicate_at_line_31_evaluates_to_true():
    paths = ["/fake/nonexistent/file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "/fake/nonexistent/file.py"


# LLM-generated content at query #6
#--------------------------

def test_predicate_at_line_31_evaluates_to_true():
    paths = ["/non/existent/file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert broken == ["/non/existent/file.py"]


# LLM-generated content at query #7
#--------------------------

def test_predicate_at_line_8_evaluates_to_true():
    import os
    from pathlib import Path

    class Config:
        def __init__(self, follow_links=False):
            self.follow_links = follow_links
            self._skipped_paths = set()
            self._supported_extensions = {'.py'}

        def is_skipped(self, path):
            return str(path) in self._skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self._supported_extensions)

        def add_skipped_path(self, path):
            self._skipped_paths.add(str(path))

    def find(paths, config, skipped, broken):
        visited_dirs = set()
        for path in paths:
            if os.path.isdir(path):
                for dirpath, dirnames, filenames in os.walk(path, topdown=True, followlinks=config.follow_links):
                    base_path = Path(dirpath)
                    for dirname in list(dirnames):
                        full_path = base_path / dirname
                        resolved_path = full_path.resolve()
                        if config.is_skipped(full_path):
                            skipped.append(str(full_path))
                            dirnames.remove(dirname)
                        else:
                            if resolved_path in visited_dirs:
                                dirnames.remove(dirname)
                        visited_dirs.add(resolved_path)
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if config.is_supported_filetype(filepath):
                            if config.is_skipped(Path(os.path.abspath(filepath))):
                                skipped.append(os.path.abspath(filepath))
                            else:
                                yield filepath
            elif not os.path.exists(path):
                broken.append(path)
            else:
                yield path

    test_dir = 'test_directory'
    os.makedirs(test_dir, exist_ok=True)
    config = Config()
    skipped = []
    broken = []
    result = list(find([test_dir], config, skipped, broken))
    assert os.path.isdir(test_dir) == True
    os.rmdir(test_dir)


# LLM-generated content at query #8
#--------------------------

def test_predicate_at_line_8_evaluates_true_for_directory():
    paths = ["/some/directory"]
    config = Config()
    skipped = []
    broken = []
    os.path.isdir = lambda x: True
    result = list(find(paths, config, skipped, broken))
    assert os.path.isdir.called_with("/some/directory")


# LLM-generated content at query #9
#--------------------------

def test_predicate_at_line_31_evaluates_to_true():
    import os
    from pathlib import Path
    from typing import Iterable, Iterator
    class Config:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else set()
        def is_skipped(self, path):
            return str(path) in self.skipped_paths or Path(path).resolve() in map(Path.resolve, map(Path, self.skipped_paths))
        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)
    def find(paths: Iterable[str], config: Config, skipped: list[str], broken: list[str]) -> Iterator[str]:
        visited_dirs: set[Path] = set()
        for path in paths:
            if os.path.isdir(path):
                for dirpath, dirnames, filenames in os.walk(path, topdown=True, followlinks=config.follow_links):
                    base_path = Path(dirpath)
                    for dirname in list(dirnames):
                        full_path = base_path / dirname
                        resolved_path = full_path.resolve()
                        if config.is_skipped(full_path):
                            skipped.append(str(full_path))
                            dirnames.remove(dirname)
                        else:
                            if resolved_path in visited_dirs:
                                dirnames.remove(dirname)
                        visited_dirs.add(resolved_path)
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if config.is_supported_filetype(filepath):
                            if config.is_skipped(Path(os.path.abspath(filepath))):
                                skipped.append(os.path.abspath(filepath))
                            else:
                                yield filepath
            elif not os.path.exists(path):
                broken.append(path)
            else:
                yield path
    config = Config(supported_extensions=['.py'])
    skipped = []
    broken = []
    test_dir = 'test_nonexistent_dir'
    result = list(find([test_dir], config, skipped, broken))
    assert not os.path.exists(test_dir)
    assert broken == [test_dir]
    assert result == []


# LLM-generated content at query #10
#--------------------------

def test_predicate_at_line_31_evaluates_to_true():
    paths = ["/non/existent/file.py"]
    config = Config()
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(skipped) == 0
    assert len(broken) == 1
    assert broken[0] == "/non/existent/file.py"


# LLM-generated content at query #11
#--------------------------

def test_find_with_directory_and_skipped():
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else {'.py'}

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    import os
    import tempfile
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
    expected_files = [file1, file2]
    assert sorted(result) == sorted(expected_files)
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
    os.unlink(tmpfile.name)

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

    import os
    import tempfile
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

def test_find_with_follow_links():
    class MockConfig:
        def __init__(self, follow_links=True, skipped_paths=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')

    import os
    import tempfile
    tmpdir = tempfile.mkdtemp()
    link_dir = os.path.join(tmpdir, 'link_dir')
    os.makedirs(link_dir)
    target_dir = os.path.join(tmpdir, 'target_dir')
    os.makedirs(target_dir)
    file_in_target = os.path.join(target_dir, 'file.py')
    open(file_in_target, 'w').close()
    os.symlink(target_dir, link_dir, target_is_directory=True)
    config = MockConfig(follow_links=True)
    skipped = []
    broken = []
    result = list(find([link_dir], config, skipped, broken))
    expected_files = [file_in_target]
    assert sorted(result) == sorted(expected_files)
    assert skipped == []
    assert broken == []
    import shutil
    shutil.rmtree(tmpdir)

def test_find_with_duplicate_resolved_path():
    class MockConfig:
        def __init__(self, follow_links=True, skipped_paths=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')

    import os
    import tempfile
    tmpdir = tempfile.mkdtemp()
    dir1 = os.path.join(tmpdir, 'dir1')
    os.makedirs(dir1)
    dir2 = os.path.join(tmpdir, 'dir2')
    os.makedirs(dir2)
    file1 = os.path.join(dir1, 'file.py')
    file2 = os.path.join(dir2, 'file.py')
    open(file1, 'w').close()
    open(file2, 'w').close()
    os.symlink(dir1, os.path.join(dir2, 'link'), target_is_directory=True)
    config = MockConfig(follow_links=True)
    skipped = []
    broken = []
    result = list(find([tmpdir], config, skipped, broken))
    expected_files = [file1, file2]
    assert sorted(result) == sorted(expected_files)
    assert skipped == []
    assert broken == []
    import shutil
    shutil.rmtree(tmpdir)


# LLM-generated content at query #12
#--------------------------

def test_predicate_at_line_9_evaluates_to_true():
    import os
    class Config:
        def __init__(self, follow_links=False):
            self.follow_links = follow_links
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return True
    config = Config()
    skipped = []
    broken = []
    test_dir = "test_dir"
    os.makedirs(test_dir, exist_ok=True)
    paths = [test_dir]
    result = list(find(paths, config, skipped, broken))
    assert os.path.isdir(paths[0]) == True
    os.rmdir(test_dir)


# LLM-generated content at query #13
#--------------------------

def test_predicate_at_line_9_evaluates_to_false():
    class Config:
        follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return True
    config = Config()
    skipped = []
    broken = []
    paths = ["/non_existent_directory"]
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 0
    assert len(broken) == 1
    assert broken[0] == "/non_existent_directory"


# LLM-generated content at query #14
#--------------------------

def test_find_with_directory_and_skipped_path():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return str(path).endswith("skip")
        def is_supported_filetype(self, filepath):
            return filepath.endswith(".py")
    config = MockConfig()
    skipped = []
    broken = []
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "skip_dir"))
        os.makedirs(os.path.join(tmpdir, "keep_dir"))
        with open(os.path.join(tmpdir, "skip_dir", "a.py"), "w") as f:
            f.write("")
        with open(os.path.join(tmpdir, "keep_dir", "b.py"), "w") as f:
            f.write("")
        result = list(find([tmpdir], config, skipped, broken))
    assert len(result) == 1
    assert result[0].endswith(os.path.join("keep_dir", "b.py"))
    assert len(skipped) == 1
    assert skipped[0].endswith("skip_dir")
    assert broken == []

def test_find_with_nonexistent_path():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith(".py")
    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(["/nonexistent/path"], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["/nonexistent/path"]

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
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(b"")
        filepath = f.name
    try:
        result = list(find([filepath], config, skipped, broken))
        assert result == [filepath]
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(filepath)

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
    with tempfile.NamedTemporaryFile(suffix="skip.py", delete=False) as f:
        f.write(b"")
        filepath = f.name
    try:
        result = list(find([filepath], config, skipped, broken))
        assert result == []
        assert skipped == [os.path.abspath(filepath)]
        assert broken == []
    finally:
        os.unlink(filepath)

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
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"")
        filepath = f.name
    try:
        result = list(find([filepath], config, skipped, broken))
        assert result == []
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(filepath)

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
        real_dir = os.path.join(tmpdir, "real")
        os.makedirs(real_dir)
        link_dir = os.path.join(tmpdir, "link")
        os.symlink(real_dir, link_dir)
        with open(os.path.join(real_dir, "a.py"), "w") as f:
            f.write("")
        result = list(find([tmpdir], config, skipped, broken))
    assert len(result) == 1
    assert result[0].endswith(os.path.join("real", "a.py"))
    assert skipped == []
    assert broken == []


# LLM-generated content at query #15
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
    result = list(find(["nonexistent.py"], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ["nonexistent.py"]

def test_find_with_skipped_directory():
    config = Config()
    config.is_skipped = lambda path: str(path).endswith("skip")
    skipped = []
    broken = []
    result = list(find(["skip"], config, skipped, broken))
    assert result == []
    assert len(skipped) == 1
    assert broken == []

def test_find_with_supported_file_in_directory():
    config = Config()
    config.is_supported_filetype = lambda filepath: filepath.endswith(".py")
    config.is_skipped = lambda path: False
    config.follow_links = False
    skipped = []
    broken = []
    result = list(find(["dir"], config, skipped, broken))
    assert "dir/test.py" in result
    assert skipped == []
    assert broken == []

def test_find_with_skipped_file():
    config = Config()
    config.is_supported_filetype = lambda filepath: filepath.endswith(".py")
    config.is_skipped = lambda path: str(path).endswith("skip.py")
    config.follow_links = False
    skipped = []
    broken = []
    result = list(find(["skip.py"], config, skipped, broken))
    assert result == []
    assert len(skipped) == 1
    assert broken == []

def test_find_with_mixed_paths():
    config = Config()
    config.is_supported_filetype = lambda filepath: filepath.endswith(".py")
    config.is_skipped = lambda path: False
    config.follow_links = False
    skipped = []
    broken = []
    result = list(find(["file.py", "nonexistent.py", "dir"], config, skipped, broken))
    assert "file.py" in result
    assert "dir/test.py" in result
    assert broken == ["nonexistent.py"]
    assert skipped == []


# LLM-generated content at query #16
#--------------------------

def test_predicate_at_line_9_evaluates_to_true():
    import os
    from pathlib import Path
    from typing import Iterable, Iterator

    class Config:
        def __init__(self, follow_links=False):
            self.follow_links = follow_links
            self._skipped_paths = set()
            self._supported_extensions = {'.py'}

        def is_skipped(self, path):
            return str(path) in self._skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self._supported_extensions)

        def add_skipped_path(self, path):
            self._skipped_paths.add(str(path))

    def find(paths: Iterable[str], config: Config, skipped: list[str], broken: list[str]) -> Iterator[str]:
        visited_dirs: set[Path] = set()
        for path in paths:
            if os.path.isdir(path):
                for dirpath, dirnames, filenames in os.walk(path, topdown=True, followlinks=config.follow_links):
                    base_path = Path(dirpath)
                    for dirname in list(dirnames):
                        full_path = base_path / dirname
                        resolved_path = full_path.resolve()
                        if config.is_skipped(full_path):
                            skipped.append(str(full_path))
                            dirnames.remove(dirname)
                        else:
                            if resolved_path in visited_dirs:
                                dirnames.remove(dirname)
                        visited_dirs.add(resolved_path)
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if config.is_supported_filetype(filepath):
                            if config.is_skipped(Path(os.path.abspath(filepath))):
                                skipped.append(os.path.abspath(filepath))
                            else:
                                yield filepath
            elif not os.path.exists(path):
                broken.append(path)
            else:
                yield path

    test_dir = 'test_directory'
    os.makedirs(test_dir, exist_ok=True)
    config = Config()
    skipped = []
    broken = []
    result = list(find([test_dir], config, skipped, broken))
    assert os.path.isdir(test_dir) == True
    os.rmdir(test_dir)


# LLM-generated content at query #17
#--------------------------

def test_predicate_at_line_9_evaluates_to_false():
    import os
    class Config:
        follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return True
    config = Config()
    skipped = []
    broken = []
    paths = ["/non_existent_directory"]
    result = list(find(paths, config, skipped, broken))
    assert not os.path.isdir(paths[0])
    assert "non_existent_directory" in broken


# LLM-generated content at query #18
#--------------------------

def test_find_with_directory_and_skipped():
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
        os.makedirs(os.path.join(tmpdir, 'keep_dir'))
        with open(os.path.join(tmpdir, 'skip_dir', 'a.py'), 'w') as f:
            f.write('')
        with open(os.path.join(tmpdir, 'keep_dir', 'b.py'), 'w') as f:
            f.write('')
        result = list(find([tmpdir], config, skipped, broken))
    assert len(result) == 1
    assert result[0].endswith(os.path.join('keep_dir', 'b.py'))
    assert len(skipped) == 1
    assert skipped[0].endswith('skip_dir')

def test_find_with_nonexistent_path():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')
    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(['/nonexistent/path'], config, skipped, broken))
    assert len(result) == 0
    assert len(broken) == 1
    assert broken[0] == '/nonexistent/path'

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
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
        f.write(b'')
        filepath = f.name
    try:
        result = list(find([filepath], config, skipped, broken))
        assert len(result) == 1
        assert result[0] == filepath
        assert len(skipped) == 0
        assert len(broken) == 0
    finally:
        os.unlink(filepath)

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
    with tempfile.NamedTemporaryFile(suffix='skip.py', delete=False) as f:
        f.write(b'')
        filepath = f.name
    try:
        result = list(find([filepath], config, skipped, broken))
        assert len(result) == 0
        assert len(skipped) == 1
        assert skipped[0] == os.path.abspath(filepath)
        assert len(broken) == 0
    finally:
        os.unlink(filepath)

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
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b'')
        filepath = f.name
    try:
        result = list(find([filepath], config, skipped, broken))
        assert len(result) == 0
        assert len(skipped) == 0
        assert len(broken) == 0
    finally:
        os.unlink(filepath)

def test_find_with_follow_links():
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
        real_dir = os.path.join(tmpdir, 'real')
        link_dir = os.path.join(tmpdir, 'link')
        os.makedirs(real_dir)
        os.symlink(real_dir, link_dir)
        with open(os.path.join(real_dir, 'a.py'), 'w') as f:
            f.write('')
        result = list(find([link_dir], config, skipped, broken))
    assert len(result) == 1
    assert result[0].endswith(os.path.join('real', 'a.py'))

def test_find_with_visited_dirs_loop():
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
        dir_a = os.path.join(tmpdir, 'a')
        dir_b = os.path.join(tmpdir, 'b')
        os.makedirs(dir_a)
        os.makedirs(dir_b)
        os.symlink(dir_a, os.path.join(dir_b, 'link_to_a'))
        with open(os.path.join(dir_a, 'file.py'), 'w') as f:
            f.write('')
        result = list(find([tmpdir], config, skipped, broken))
    assert len([p for p in result if p.endswith('file.py')]) == 1


# LLM-generated content at query #19
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
        with open(os.path.join(tmpdir, 'normal_dir', 'a.py'), 'w') as f:
            f.write('')
        with open(os.path.join(tmpdir, 'skip_dir', 'b.py'), 'w') as f:
            f.write('')
        result = list(find([tmpdir], config, skipped, broken))
    assert len(result) == 1
    assert result[0].endswith(os.path.join('normal_dir', 'a.py'))
    assert len(skipped) == 1
    assert skipped[0].endswith('skip_dir')
    assert broken == []

def test_find_with_nonexistent_path():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')
    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(['/nonexistent/path'], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ['/nonexistent/path']

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
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
        f.write(b'')
        filepath = f.name
    try:
        result = list(find([filepath], config, skipped, broken))
        assert result == [filepath]
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(filepath)

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
    with tempfile.NamedTemporaryFile(suffix='skip.py', delete=False) as f:
        f.write(b'')
        filepath = f.name
    try:
        result = list(find([filepath], config, skipped, broken))
        assert result == []
        assert skipped == [os.path.abspath(filepath)]
        assert broken == []
    finally:
        os.unlink(filepath)

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
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b'')
        filepath = f.name
    try:
        result = list(find([filepath], config, skipped, broken))
        assert result == []
        assert skipped == []
        assert broken == []
    finally:
        os.unlink(filepath)

def test_find_with_follow_links_and_cycle():
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
        link_target = os.path.join(tmpdir, 'target')
        os.makedirs(link_target)
        link_source = os.path.join(tmpdir, 'link')
        os.symlink(link_target, link_source)
        with open(os.path.join(link_target, 'a.py'), 'w') as f:
            f.write('')
        os.symlink(link_source, os.path.join(link_target, 'loop'))
        result = list(find([tmpdir], config, skipped, broken))
    assert len(result) == 1
    assert result[0].endswith(os.path.join('target', 'a.py'))
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
        with open(os.path.join(dir1, 'a.py'), 'w') as f:
            f.write('')
        with open(os.path.join(dir2, 'b.py'), 'w') as f:
            f.write('')
        result = list(find([dir1, dir2], config, skipped, broken))
    assert len(result) == 2
    assert any(r.endswith(os.path.join('dir1', 'a.py')) for r in result)
    assert any(r.endswith(os.path.join('dir2', 'b.py')) for r in result)
    assert skipped == []
    assert broken == []

def test_find_with_empty_paths():
    class MockConfig:
        follow_links = False
        def is_skipped(self, path):
            return False
        def is_supported_filetype(self, filepath):
            return filepath.endswith('.py')
    config = MockConfig()
    skipped = []
    broken = []
    result = list(find([], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []


# LLM-generated content at query #20
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

    config = MockConfig(skipped_paths=['/test/skip_dir'])
    skipped = []
    broken = []
    result = list(find(['/test'], config, skipped, broken))
    assert result == []
    assert skipped == ['/test/skip_dir']
    assert broken == []

def test_find_with_supported_file():
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else {'.py'}

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(['/test/file.py'], config, skipped, broken))
    assert result == ['/test/file.py']
    assert skipped == []
    assert broken == []

def test_find_with_broken_path():
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else {'.py'}

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(['/nonexistent'], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ['/nonexistent']

def test_find_with_unsupported_filetype():
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else {'.py'}

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(['/test/file.txt'], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

def test_find_with_follow_links_and_visited_dirs():
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else {'.py'}

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    config = MockConfig(follow_links=True)
    skipped = []
    broken = []
    result = list(find(['/test'], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

def test_find_with_skipped_file():
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else {'.py'}

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    config = MockConfig(skipped_paths=['/test/skip_file.py'])
    skipped = []
    broken = []
    result = list(find(['/test/skip_file.py'], config, skipped, broken))
    assert result == []
    assert skipped == ['/test/skip_file.py']
    assert broken == []

def test_find_with_multiple_paths():
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else {'.py'}

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    config = MockConfig()
    skipped = []
    broken = []
    result = list(find(['/test/file1.py', '/test/file2.py'], config, skipped, broken))
    assert result == ['/test/file1.py', '/test/file2.py']
    assert skipped == []
    assert broken == []

def test_find_with_mixed_paths():
    class MockConfig:
        def __init__(self, follow_links=False, skipped_paths=None, supported_extensions=None):
            self.follow_links = follow_links
            self.skipped_paths = set(skipped_paths) if skipped_paths else set()
            self.supported_extensions = set(supported_extensions) if supported_extensions else {'.py'}

        def is_skipped(self, path):
            return str(path) in self.skipped_paths

        def is_supported_filetype(self, filepath):
            return any(filepath.endswith(ext) for ext in self.supported_extensions)

    config = MockConfig(skipped_paths=['/test/skip.py'])
    skipped = []
    broken = []
    result = list(find(['/test/file.py', '/test/skip.py', '/nonexistent'], config, skipped, broken))
    assert result == ['/test/file.py']
    assert skipped == ['/test/skip.py']
    assert broken == ['/nonexistent']


# LLM-generated content at query #21
#--------------------------

def test_find_with_directory_and_skipped():
    from unittest.mock import Mock
    config = Mock()
    config.follow_links = False
    config.is_skipped = Mock(side_effect=lambda p: str(p).endswith('skip'))
    config.is_supported_filetype = Mock(return_value=True)
    skipped = []
    broken = []
    result = list(find(['test_dir'], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

def test_find_with_supported_file_in_directory():
    import os
    from unittest.mock import Mock, patch
    config = Mock()
    config.follow_links = False
    config.is_skipped = Mock(return_value=False)
    config.is_supported_filetype = Mock(return_value=True)
    skipped = []
    broken = []
    with patch('os.path.isdir', return_value=True):
        with patch('os.walk') as mock_walk:
            mock_walk.return_value = [('test_dir', [], ['file.py'])]
            result = list(find(['test_dir'], config, skipped, broken))
    assert result == [os.path.join('test_dir', 'file.py')]
    assert skipped == []
    assert broken == []

def test_find_with_skipped_file_in_directory():
    import os
    from unittest.mock import Mock, patch
    config = Mock()
    config.follow_links = False
    config.is_skipped = Mock(side_effect=lambda p: str(p).endswith('skip.py'))
    config.is_supported_filetype = Mock(return_value=True)
    skipped = []
    broken = []
    with patch('os.path.isdir', return_value=True):
        with patch('os.walk') as mock_walk:
            mock_walk.return_value = [('test_dir', [], ['skip.py'])]
            result = list(find(['test_dir'], config, skipped, broken))
    assert result == []
    assert skipped == [os.path.abspath(os.path.join('test_dir', 'skip.py'))]
    assert broken == []

def test_find_with_skipped_directory():
    from unittest.mock import Mock, patch
    config = Mock()
    config.follow_links = False
    config.is_skipped = Mock(side_effect=lambda p: str(p).endswith('skip_dir'))
    config.is_supported_filetype = Mock(return_value=True)
    skipped = []
    broken = []
    with patch('os.path.isdir', return_value=True):
        with patch('os.walk') as mock_walk:
            mock_walk.return_value = [('test_dir', ['skip_dir'], [])]
            result = list(find(['test_dir'], config, skipped, broken))
    assert result == []
    assert skipped == ['skip_dir']
    assert broken == []

def test_find_with_broken_path():
    from unittest.mock import Mock, patch
    config = Mock()
    config.follow_links = False
    config.is_skipped = Mock()
    config.is_supported_filetype = Mock()
    skipped = []
    broken = []
    with patch('os.path.isdir', return_value=False):
        with patch('os.path.exists', return_value=False):
            result = list(find(['broken_path'], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == ['broken_path']

def test_find_with_direct_file_path():
    from unittest.mock import Mock, patch
    config = Mock()
    config.follow_links = False
    config.is_skipped = Mock()
    config.is_supported_filetype = Mock()
    skipped = []
    broken = []
    with patch('os.path.isdir', return_value=False):
        with patch('os.path.exists', return_value=True):
            result = list(find(['file.py'], config, skipped, broken))
    assert result == ['file.py']
    assert skipped == []
    assert broken == []

def test_find_with_unsupported_filetype():
    from unittest.mock import Mock, patch
    config = Mock()
    config.follow_links = False
    config.is_skipped = Mock(return_value=False)
    config.is_supported_filetype = Mock(return_value=False)
    skipped = []
    broken = []
    with patch('os.path.isdir', return_value=True):
        with patch('os.walk') as mock_walk:
            mock_walk.return_value = [('test_dir', [], ['file.txt'])]
            result = list(find(['test_dir'], config, skipped, broken))
    assert result == []
    assert skipped == []
    assert broken == []

def test_find_with_follow_links_and_visited_dir():
    import os
    from unittest.mock import Mock, patch
    config = Mock()
    config.follow_links = True
    config.is_skipped = Mock(return_value=False)
    config.is_supported_filetype = Mock(return_value=True)
    skipped = []
    broken = []
    with patch('os.path.isdir', return_value=True):
        with patch('os.walk') as mock_walk:
            mock_walk.return_value = [('test_dir', ['subdir'], ['file.py'])]
            with patch('pathlib.Path.resolve', return_value=Path('/resolved/subdir')):
                result = list(find(['test_dir'], config, skipped, broken))
    assert result == [os.path.join('test_dir', 'file.py')]
    assert skipped == []
    assert broken == []

def test_find_with_multiple_paths():
    import os
    from unittest.mock import Mock, patch
    config = Mock()
    config.follow_links = False
    config.is_skipped = Mock(return_value=False)
    config.is_supported_filetype = Mock(return_value=True)
    skipped = []
    broken = []
    with patch('os.path.isdir', side_effect=[True, False, False]):
        with patch('os.path.exists', side_effect=[True, True]):
            with patch('os.walk') as mock_walk:
                mock_walk.return_value = [('dir1', [], ['f1.py'])]
                result = list(find(['dir1', 'file2.py', 'broken'], config, skipped, broken))
    expected = [os.path.join('dir1', 'f1.py'), 'file2.py']
    assert result == expected
    assert skipped == []
    assert broken == ['broken']


