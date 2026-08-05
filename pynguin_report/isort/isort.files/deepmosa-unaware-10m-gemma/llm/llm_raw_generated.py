####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Setup Mocks
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True
    
    skipped = []
    broken = []
    paths = ["/mock/dir", "/mock/file.py", "/non/existent"]

    # Mocking os.path and filesystem structure
    # Structure:
    # /mock/dir/
    #    sub/
    #       target.py
    #    skip_me/ (should be skipped)
    #       hidden.py
    # /mock/file.py (exists as file)
    # /non/existent (does not exist)

    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.path.abspath") as mock_abspath, \
         patch("os.walk") as mock_walk:
        
        def side_effect_isdir(path):
            return path == "/mock/dir"

        def side_effect_exists(path):
            return path in ["/mock/dir", "/mock/file.py"]

        mock_isdir.side_effect = side_effect_isdir
        mock_exists.side_effect = side_effect_exists
        mock_abspath.side_effect = lambda x: x
        
        # Setup os.walk behavior for /mock/dir
        # First iteration: root is /mock/dir, contains 'sub' and 'skip_me'
        # Second iteration (inside sub): root is /mock/dir/sub, contains 'target.py'
        # Third iteration (inside skip_me): root is /mock/dir/skip_me, contains 'hidden.py'
        mock_walk.return_value = [
            ("/mock/dir", ["sub", "skip_me"], []),
            ("/mock/dir/sub", [], ["target.py"]),
            ("/mock/dir/skip_me", [], ["hidden.py"]),
        ]

        # Define skip logic: skip 'skip_me' directory and its contents
        def is_skipped(path):
            return "skip_me" in str(path)
        
        config.is_skipped.side_effect = is_skipped

        # Execution
        result = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. /mock/file.py should be yielded directly (it's a file)
        assert "/mock/file.py" in result
        
        # 2. target.py should be yielded because it is in an unskipped directory
        assert "/mock/dir/sub/target.py" in result

        # 3. /non/existent should be in broken list
        assert "/non/existent" in broken

        # 4. skip_me and its contents should be in skipped list
        assert any("skip_me" in s for s in skipped)
        
        # 5. hidden.py should NOT be in result because its parent was skipped
        assert "/mock/dir/skip_me/hidden.py" not in result

        # 6. Verify is_supported_filetype was checked for found files
        assert config.is_supported_filetype.called
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    skipped = []
    broken = []

    # Test Case 1: Single file path that exists
    with patch("os.path.isdir", return_value=False), \
         patch("os.path.exists", return_value=True):
        paths = ["file.py"]
        results = list(find(paths, config, skipped, broken))
        assert results == ["file.py"]

    # Test Case 2: Path does not exist
    with patch("os.path.isdir", return_value=False), \
         patch("os.path.exists", return_value=False):
        paths = ["nonexistent.py"]
        results = list(find(paths, config, skipped, broken))
        assert results == []
        assert "nonexistent.py" in broken

    # Test Case 3: Directory traversal with files and skips
    # Setup a mock filesystem structure via os.walk
    # dir/
    #   file1.py (supported)
    #   skipped_dir/ (should be skipped)
    #     file2.py
    #   file3.txt (not supported)
    
    with patch("os.path.isdir", return_value=True), \
         patch("os.walk") as mock_walk, \
         patch("os.path.exists", return_value=True), \
         patch("pathlib.Path.resolve") as mock_resolve:
        
        # Mocking os.walk behavior
        mock_walk.return_value = [
            ("/root", ["skipped_dir", "subdir"], ["file1.py", "file3.txt"]),
            ("/root/skipped_dir", [], ["file2.py"]),
            ("/root/subdir", [], ["file4.py"]),
        ]

        # Mocking path resolution to prevent actual FS access
        mock_resolve.side_effect = lambda: MagicMock()

        # Setup config behavior for skipping a specific directory
        def is_skipped_side_effect(path):
            return "skipped_dir" in str(path)
        
        config.is_skipped.side_effect = is_skipped_side_effect
        
        # Setup supported filetypes (only .py)
        def is_supported_side_effect(filepath):
            return filepath.endswith(".py")
        
        config.is_supported_filetype.side_effect = is_supported_side_effect

        paths = ["/root"]
        # Clear lists for clean test
        skipped.clear()
        broken.clear()

        results = list(find(paths, config, skipped, broken))

        # file1.py is in root and supported -> yield
        # file3.txt is in root but not supported -> ignore
        # skipped_dir is skipped -> add to skipped, don't traverse
        # subdir is in root and supported -> traverse
        # file4.py is in subdir and supported -> yield (if walk reached it)

        # Note: In the logic, if dirnames.remove(dirname) happens, 
        # os.walk won't enter that directory in the next iteration of the loop 
        # provided by the mock_walk setup.
        
        assert "file1.py" in results
        assert any("skipped_dir" in s for s in skipped)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    skipped = []
    broken = []

    # Use tmp_path for real filesystem interaction during tests
    base_dir = tmp_path / "project"
    base_dir.mkdir()
    
    src_dir = base_dir / "src"
    src_dir.mkdir()
    
    py_file = src_dir / "main.py"
    py_file.write_text("print('hello')")
    
    readme_file = base_dir / "README.md"
    readme_file.write_text("docs")
    
    # Setup skip directory
    skip_dir = src_dir / "venv"
    skip_dir.mkdir()
    (skip_dir / "lib.py").write_text("")

    # Define paths to search
    paths = [str(base_dir), "non_existent_path", str(py_file)]

    with patch("os.walk") as mock_walk:
        # Mocking os.walk behavior for the directory 'base_dir'
        # We simulate finding src, README.md, and venv
        mock_walk.return_value = [
            (str(base_dir), ["src"], ["README.md"]),
            (str(src_dir), ["venv"], ["main.py", "README.md"]),
            (str(skip_dir), [], ["lib.py"]),
        ]

        # Configure is_skipped to trigger on 'venv'
        def side_effect_is_skipped(path):
            return "venv" in str(path)
        config.is_skipped.side_effect = side_effect_is_skipped

        # Configure is_supported_filetype to only allow .py files
        def side_effect_is_supported(path):
            return path.endswith(".py")
        config.is_supported_filetype.side_effect = side_effect_is_supported

        # Execute function
        results = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. 'non_existent_path' should be in broken
        assert "non_existent_path" in broken
        
        # 2. 'venv' directory was skipped, so it should be in skipped list
        assert any("venv" in s for s in skipped)

        # 3. Results should include the single file path provided directly and the found .py file
        # Note: README.md is filtered out by is_supported_filetype
        assert str(py_file) in results
        assert str(py_file) in results # It appears via the direct path and via walk simulation
        
        # 4. Check that broken list contains the invalid path
        assert "non_existent_path" in broken

def test_find_direct_file():
    config = MagicMock(spec=Config)
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True
    
    tmp_file = tmp_path / "single.py"
    tmp_file.write_text("")
    
    skipped = []
    broken = []
    paths = [str(tmp_file)]
    
    results = list(find(paths, config, skipped, broken))
    assert results == [str(tmp_file)]

def test_find_with_skipping_logic():
    config = MagicMock(spec=Config)
    config.follow_links = False
    # Simulate that all files are skipped
    config.is_skipped.return_value = True
    config.is_supported_filetype.return_value = True

    tmp_dir = tmp_path / "skip_test"
    tmp_dir.mkdir()
    file_path = tmp_dir / "test.py"
    file_path.write_text("")

    skipped = []
    broken = []
    paths = [str(tmp_dir)]

    results = list(find(paths, config, skipped, broken))
    
    assert len(results) == 0
    assert str(file_path) in skipped
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup paths and tracking lists
    paths = ["/valid/file.py", "/valid/dir", "/non/existent", "/skip/me.py"]
    skipped = []
    broken = []

    # Create a fake directory structure using patch
    # We simulate:
    # /valid/file.py (direct file)
    # /valid/dir/inner.py (file in dir)
    # /valid/dir/sub_dir (directory)
    # /skip/me.py (skipped file)
    
    with patch("os.path.exists") as mock_exists, \
         patch("os.path.isdir") as mock_isdir, \
         patch("os.walk") as mock_walk, \
         patch("os.path.abspath") as mock_abspath, \
         patch("pathlib.Path.resolve") as mock_resolve:

        def side_effect_exists(p):
            return p in ["/valid/file.py", "/valid/dir", "/skip/me.py"]
        
        def side_effect_isdir(p):
            return p == "/valid/dir"

        mock_exists.side_effect = side_effect_exists
        mock_isdir.side_effect = side_effect_isdir
        mock_abspath.side_effect = lambda x: x
        mock_resolve.side_effect = lambda: MagicMock() # simplified

        # Mock os.walk for the directory "/valid/dir"
        # dirpath, dirnames, filenames
        mock_walk.return_value = [
            ("/valid/dir", ["sub_dir"], ["inner.py"]),
        ]
        
        # Configure is_skipped behavior
        def is_skipped_side_effect(p):
            return str(p) == "/skip/me.py"
        config.is_skipped.side_effect = is_skipped_side_effect

        # Run the function
        result = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. /valid/file.py should be yielded (direct file)
        # 2. /valid/dir/inner.py should be yielded (found via walk)
        # 3. /non/existent should be in broken
        # 4. /skip/me.py should be in skipped
        
        assert "/valid/file.py" in result
        assert "/valid/dir/inner.py" in result
        assert "/non/existent" in broken
        assert "/skip/me.py" in skipped
        assert len(result) == 2
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Test data
    paths = ["/valid/file.py", "/non/existent/path", "/valid/dir"]
    skipped = []
    broken = []

    # Setup filesystem structure using patch
    # We need to mock os.path.isdir, os.path.exists, os.walk, and os.path.abspath
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.walk") as mock_walk, \
         patch("os.path.abspath") as mock_abspath:

        # Define behavior for paths
        def isdir_side_effect(p):
            return p == "/valid/dir"
        
        def exists_side_effect(p):
            if p == "/non/existent/path":
                return False
            return True

        mock_isdir.side_ext = isdir_side_effect
        mock_isdir.side_effect = isdir_side_effect
        mock_exists.side_effect = exists_side_effect
        mock_abspath.side_effect = lambda x: x

        # Define behavior for os.walk in /valid/dir
        # dirpath, dirnames, filenames
        mock_walk.return_value = [
            ("/valid/dir", ["subdir"], ["file1.py"]),
            ("/valid/dir/subdir", [], ["file2.py"])
        ]

        # Define behavior for config.is_skipped
        # Let's make 'subdir' a skipped directory
        def is_skipped_side_effect(path):
            return "subdir" in str(path)
        
        config.is_skipped.side_effect = is_skipped_side_effect

        # Execute function
        result = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. Valid file (direct path) should be in result
        assert "/valid/file.py" in result
        
        # 2. Non-existent path should be in broken
        assert "/non/existent/path" in broken
        
        # 3. Supported file in dir should be in result
        assert "/valid/dir/file1.py" in result

        # 4. Skipped directory content should not be in result, and subdir should be in skipped
        assert "/valid/dir/subdir" in skipped
        assert "/valid/dir/subdir/file2.py" not in result

        # 5. Check that dirnames.remove was called (implied by skip logic)
        assert len(result) == 2 # /valid/file.py and /valid/dir/file1.py
```


# LLM-generated content at query #6
#--------------------------

```python
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock


def test_find(tmp_path):
    # Setup mock config
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Create directory structure
    # root/
    #   file1.py
    #   subdir/
    #     file2.py
    #     skipped_dir/ (should be skipped)
    #     file3.txt (not supported)
    # non_existent_path
    
    root = tmp_path / "src"
    root.mkdir()
    
    file1 = root / "file1.py"
    file1.write_text("print(1)")
    
    subdir = root / "subdir"
    subdir.mkdir()
    file2 = subdir / "file2.py"
    file2.write_text("print(2)")
    
    skipped_dir = root / "skipped_dir"
    skipped_dir.mkdir()
    file_in_skipped = skipped_dir / "hidden.py"
    file_in_skipped.write_text("print(3)")
    
    file3 = subdir / "file3.txt"
    file3.write_text("not python")

    # Configure mock behavior for specific paths
    def side_effect_is_skipped(path):
        return "skipped_dir" in str(path)
    
    config.is_skipped.side_effect = side_effect_is_skipped

    def side_effect_is_supported(path):
        return path.endswith(".py")
    
    config.is_supported_filetype.side_effect = side_effect_is_supported

    # Test inputs
    paths = [str(root), "non_existent_path", str(file1)]
    skipped = []
    broken = []

    # Execution
    results = list(find(paths, config, skipped, broken))

    # Assertions
    # 1. Check found files (file1.py and file2.py)
    # Note: order might vary depending on os.walk, but contents should be correct
    assert str(file1) in results
    assert str(file2) in results
    assert str(file3) not in results  # Unsupported extension
    
    # 2. Check skipped items
    assert any("skipped_dir" in s for s in skipped)
    
    # 3. Check broken paths
    assert "non_existent_path" in broken

    # 4. Check direct file path input
    assert str(file1) in results

def test_find_visited_dirs_logic(tmp_path):
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Create symlink to same directory to test visited_dirs logic
    root = tmp_path / "root"
    root.mkdir()
    file1 = root / "file1.py"
    file1.write_text("content")
    
    link_dir = tmp_path / "link_dir"
    os.symlink(root, link_dir)

    paths = [str(root), str(link_dir)]
    skipped = []
    broken = []

    results = list(find(paths, config, skipped, broken))

    # file1.py should only be yielded once despite being in both paths
    assert results.count(str(file1)) == 1
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    skipped = []
    broken = []

    # Setup a temporary directory structure using tmp_path fixture
    # root/
    #   file1.py
    #   subdir/
    #     file2.py
    #   skip_me/ (should be skipped)
    #     file3.py
    #   broken_link (non-existent)

    root = tmp_path / "test_dir"
    root.mkdir()
    
    file1 = root / "file1.py"
    file1.write_text("print(1)")
    
    subdir = root / "subdir"
    subdir.mkdir()
    file2 = subdir / "file2.py"
    file2.write_text("print(2)")
    
    skip_dir = root / "skip_me"
    skip_dir.mkdir()
    file3 = skip_dir / "file3.py"
    file3.write_text("print(3)")

    # Define behavior for is_skipped: skip 'skip_me' directory
    def side_effect_is_skipped(path):
        return "skip_me" in str(path)
    config.is_skipped.side_effect = side_effect_is_skipped

    # Define behavior for unsupported files (e.g., ignore .txt)
    def side_effect_supported(filepath):
        return not filepath.endswith(".txt")
    config.is_supported_filetype.side_effect = side_effect_supported

    # Add a txt file that should be ignored by is_supported_filetype
    file_txt = root / "ignore.txt"
    file_txt.write_text("ignore me")

    # Path to a non-existent directory/file
    broken_path = str(root / "non_existent_path")
    
    paths = [str(root), broken_path, str(file1)]

    # Execution
    result = list(find(paths, config, skipped, broken))

    # Assertions
    # 1. Check that all valid python files were found
    assert str(file1) in result
    assert str(file2) in result
    assert str(file_txt) not in result
    
    # 2. Check that file1 was yielded twice (once as part of root, once explicitly)
    # Note: The implementation yields 'path' directly if it exists and is not a dir.
    # Since file1 is inside root, it's found via walk AND via explicit path.
    assert result.count(str(file1)) == 2

    # 3. Check skipped logic
    assert any("skip_me" in s for s in skipped)
    assert str(file3) not in result # file3 is inside a skipped dir

    # 4. Check broken paths
    assert broken_path in broken

    # 5. Check that the txt file was filtered by supported_filetype
    assert any(f.endswith(".txt") for f in result) == False
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup temporary directory structure
    tmp_dir = Path("test_root")
    tmp_dir.mkdir(exist_ok=True)
    subdir = tmp_dir / "subdir"
    subdir.mkdir(exist_ok=True)
    file1 = tmp_dir / "file1.py"
    file2 = subdir / "file2.py"
    file1.write_text("print(1)")
    file2.write_text("print(2)")
    skipped_file = tmp_dir / "skipped.py"
    skipped_file.write_text("print(3)")

    try:
        # Case 1: Basic finding of files in directory and subdirectories
        paths = [str(tmp_dir)]
        skipped = []
        broken = []
        
        results = list(find(paths, config, skipped, broken))
        
        assert len(results) == 2
        assert any("file1.py" in r for r in results)
        assert any("file2.py" in r for r in results)

        # Case 2: Handling non-existent paths (broken)
        paths = ["non_existent_path"]
        skipped = []
        broken = []
        list(find(paths, config, skipped, broken))
        assert "non_existent_path" in broken

        # Case 3: Handling single files passed directly
        paths = [str(file1)]
        skipped = []
        broken = []
        results = list(find(paths, config, skipped, broken))
        assert results == [str(file1)]

        # Case 4: Skipping files via config
        config.is_skipped.side_effect = lambda p: "skipped.py" in str(p)
        paths = [str(tmp_dir)]
        skipped = []
        broken = []
        results = list(find(paths, config, skipped, broken))
        
        # Should not contain the skipped file in yield, but should be in skipped list
        assert len(results) == 1 # Only file2.py remains if subdir/file2 is not skipped
        assert any("skipped.py" in s for s in skipped)

        # Case 5: Skipping directories via config
        config.is_skipped.side_effect = lambda p: "subdir" in str(p)
        paths = [str(tmp_dir)]
        skipped = []
        broken = []
        results = list(find(paths, config, skipped, broken))
        # Should only find file1.py because subdir is skipped
        assert len(results) == 1
        assert "file1.py" in results[0]
        assert any("subdir" in s for s in skipped)

    finally:
        # Cleanup
        import shutil
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Setup mock config
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup temporary directory structure
    # root/
    #   file1.py
    #   subdir/
    #     file2.py
    #   skipped_dir/ (to be skipped)
    #   broken_path (non-existent)
    
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.walk") as mock_walk, \
         patch("os.path.abspath") as mock_abspath, \
         patch("pathlib.Path.resolve") as mock_resolve:

        # Define behaviors for paths
        # Path 1: A directory
        # Path 2: A single file
        # Path 3: A non-existent path
        mock_isdir.side_effect = lambda p: p == "root" or p == "root/subdir"
        mock_exists.side_effect = lambda p: p != "non_existent"
        mock_abspath.side_effect = lambda p: p

        # Mock os.walk for the 'root' directory
        # dirpath, dirnames, filenames
        mock_walk.return_value = [
            ("root", ["subdir", "skipped_dir"], ["file1.py"]),
            ("root/subdir", [], ["file2.py"]),
            ("root/skipped_dir", [], ["file3.py"]),
        ]

        # Mock resolution for symlink detection logic
        mock_resolve.side_effect = lambda: MagicMock()

        # Configure is_skipped to skip 'skipped_dir' and its contents
        def is_skipped_side_effect(path):
            return "skipped_dir" in str(path)
        config.is_skipped.side_effect = is_skipped_side_effect

        paths = ["root", "root/file1.py", "non_existent"]
        skipped = []
        broken = []

        # Execute function
        result = list(find(paths, config, skipped, broken))

        # Assertions
        # file1.py should be found (from root)
        # file2.py should be found (from subdir)
        # root/file1.py was passed as a direct path (yielded directly)
        assert "root/file1.py" in result
        assert "root/file1.py" in result # From the explicit path in 'paths'
        assert "root/subdir/file2.py" in result
        
        # Check broken paths
        assert "non_existent" in broken

        # Check skipped paths
        # The directory 'skipped_dir' was identified as skipped during walk
        assert any("skipped_dir" in s for s in skipped)
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

def test_find(tmp_path):
    # Setup directory structure
    # root/
    #   file1.py
    #   file2.txt (unsupported)
    #   subdir/
    #     file3.py
    #     skipped_dir/
    #       file4.py
    #   broken_path (non-existent)

    root = tmp_path / "root"
    root.mkdir()
    
    file1 = root / "file1.py"
    file1.write_text("content")
    
    file2 = root / "file2.txt"
    file2.write_text("content")
    
    subdir = root / "subdir"
    subdir.mkdir()
    file3 = subdir / "file3.py"
    file3.write_text("content")
    
    skipped_dir = root / "skipped_dir"
    skipped_dir.mkdir()
    file4 = skipped_dir / "file4.py"
    file4.write_text("content")

    broken_path = str(tmp_path / "non_existent_path")
    single_file = root / "single.py"
    single_file.write_text("content")

    # Mock Config
    config = MagicMock()
    def is_skipped(p):
        return "skipped_dir" in str(p)
    
    def is_supported_filetype(p):
        return p.endswith(".py")

    config.is_skipped.side_effect = is_skipped
    config.is_supported_filetype.side_effect = is_supported_filetype
    config.follow_links = False

    paths = [str(root), broken_path, str(single_file)]
    skipped = []
    broken = []

    # Execution
    results = list(find(paths, config, skipped, broken))

    # Assertions
    # Expected files: root/file1.py, root/subdir/file3.py, root/single.py
    # Note: order depends on os.walk, but content should match
    assert len(results) == 3
    assert str(file1) in results
    assert str(file3) in results
    assert str(single_file) in results
    assert str(file2) not in results  # Unsupported type
    
    # Check skipped logic
    assert any("skipped_dir" in s for s in skipped)
    assert str(skipped_dir) in skipped
    
    # Check broken logic
    assert broken_path in broken
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Setup common mocks
    config = MagicMock()
    skipped = []
    broken = []
    
    # Case 1: Path does not exist
    with patch("os.path.exists", return_value=False):
        results = list(find(["non_existent"], config, skipped, broken))
        assert results == []
        assert "non_existent" in broken

    # Reset state
    skipped = []
    broken = []

    # Case 2: Path is a single file (exists, not a directory)
    with patch("os.path.isdir", return_value=False), \
         patch("os.path.exists", return_value=True):
        results = list(find(["file.py"], config, skipped, broken))
        assert results == ["file.py"]

    # Case 3: Path is a directory with files and logic for skipping/walking
    # We simulate os.walk behavior
    with patch("os.path.isdir", return_value=True), \
         patch("os.walk") as mock_walk, \
         patch("os.path.exists", return_value=True):
        
        # Mocking os.walk to return: 
        # dirpath='root', dirnames=['subdir'], filenames=['a.py', 'b.txt']
        mock_walk.return_value = [
            ('root', ['subdir'], ['a.py', 'b.txt']),
            ('root/subdir', [], ['c.py'])
        ]
        
        # Mock Config behaviors
        # is_supported_filetype: only .py files
        config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")
        # is_skipped: skip 'b.txt' (already filtered by type) and 'subdir'
        config.is_skipped.side_effect = lambda p: str(p).endswith("subdir")
        config.follow_links = False

        # Mock Path.resolve to avoid real filesystem dependency during dirname logic
        with patch("pathlib.Path.resolve", return_value=Path("root/subdir").resolve()):
            results = list(find(["root"], config, skipped, broken))

        # a.py should be yielded
        # b.txt should be ignored by is_supported_filetype
        # c.py should be yielded because subdir was removed from dirnames via is_skipped
        assert "root/a.py" in results
        assert "root/b.txt" not in results
        assert "root/subdir/c.py" not in results # Because 'subdir' was skipped and removed from dirnames
        assert any("root/subdir" in s for s in skipped)

    # Case 4: Testing duplicate directory detection (visited_dirs)
    skipped = []
    with patch("os.path.isdir", return_value=True), \
         patch("os.walk") as mock_walk, \
         patch("os.path.exists", return_value=True):
        
        # Two walks that hit the same directory via different paths (simulated)
        mock_walk.return_value = [
            ('root', ['subdir'], ['a.py']),
            ('root/subdir', [], ['c.py'])
        ]
        config.is_supported_filetype.return_value = True
        config.is_skipped.return_value = False
        config.follow_links = False

        # Simulate that root/subdir is already in visited_dirs by making resolve return same path
        with patch("pathlib.Path.resolve") as mock_resolve:
            mock_resolve.side_effect = [Path("root/subdir").resolve(), Path("root/subdir").resolve()]
            results = list(find(["root"], config, skipped, broken))
            
            # The second iteration (the subdir) should be skipped because it's in visited_dirs
            # But the first file 'a.py' is found before the dirnames loop processes the sub-folder logic
            assert "root/a.py" in results
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

def test_find(tmp_path):
    # Setup configuration mock
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Create directory structure
    # root/
    #   file1.py
    #   subdir/
    #     file2.py
    #     skipped_dir/ (to be skipped)
    #       file3.py
    #   broken_path (non-existent)
    
    root = tmp_path / "root"
    root.mkdir()
    
    f1 = root / "file1.py"
    f1.write_text("print(1)")
    
    subdir = root / "subdir"
    subdir.mkdir()
    f2 = subdir / "file2.py"
    f2.write_text("print(2)")
    
    skip_dir = root / "skipped_dir"
    skip_dir.mkdir()
    f3 = skip_dir / "file3.py"
    f3.write_text("print(3)")

    non_existent = str(tmp_path / "not_here")
    single_file = str(root / "file1.py")
    
    # Define behavior for skipping
    def is_skipped_side_effect(path):
        return "skipped_dir" in str(path)
    
    config.is_skipped.side_effect = is_skipped_side_effect

    paths = [str(root), single_file, non_existent]
    skipped = []
    broken = []

    # Execute function
    results = list(find(paths, config, skipped, broken))

    # Assertions
    # 1. Check yielded files (should find file1 and file2, but not inside skipped_dir)
    # Note: order depends on os.walk, but we check existence in result
    assert str(f1) in results
    assert str(f2) in results
    assert str(f3) not in results

    # 2. Check single file path (yielded directly)
    assert single_file in results

    # 3. Check skipped list
    assert any("skipped_dir" in s for s in skipped)

    # 4. Check broken list
    assert non_existent in broken

def test_find_with_symlinks(tmp_path):
    config = MagicMock()
    config.follow_links = True
    config.is_skipped.return_value = False
    config.is_supported_filetype.returnative = True
    
    # Setup symlink scenario
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target"
    target.mkdir()
    f_target = target / "target.py"
    f_target.write_text("target")
    
    link_dir = root / "link_dir"
    link_dir.mkdir()
    
    # Create a symlink to the target directory
    symlink_path = root / "link_to_target"
    os.symlink(target, symlink_path)

    paths = [str(root)]
    skipped = []
    broken = []
    
    # We use a patch to ensure we can control the walk or just rely on os.walk 
    # with follow_links=True
    results = list(find(paths, config, skipped, broken))
    
    # If follow_links is True, it might find files via the link and the real dir
    # but visited_dirs logic should prevent infinite loops or duplicates
    assert str(f_target) in results

def test_find_unsupported_filetype(tmp_path):
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    # Only .py files are supported
    config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")

    root = tmp_path / "root"
    root.mkdir()
    f_py = root / "test.py"
    f_py.write_text("content")
    f_txt = root / "test.txt"
    f_txt.write_text("content")

    paths = [str(root)]
    skipped = []
    broken = []

    results = list(find(paths, config, skipped, broken))

    assert str(f_py) in results
    assert str(f_txt) not in results
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    skipped = []
    broken = []

    # Setup temporary directory structure using patch
    # We will mock os.path, os.walk, and os.path.isdir to avoid actual filesystem IO
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.walk") as mock_walk, \
         patch("os.path.abspath") as mock_abspath, \
         patch("pathlib.Path.resolve") as mock_resolve:

        # Case 1: Path is a single file (not a directory)
        mock_isdir.side_effect = lambda p: p == "file.py"
        mock_exists.side_effect = lambda p: p == "file.py" or p == "dir_path"
        mock_abspath.side_effect = lambda p: p
        
        paths = ["file.py"]
        results = list(find(paths, config, skipped, broken))
        assert results == ["file.py"]
        assert broken == []

        # Case 2: Path does not exist
        paths = ["non_existent.py"]
        mock_isdir.side_effect = lambda p: False
        mock_exists.side_effect = lambda p: False
        results = list(find(paths, config, skipped, broken))
        assert results == []
        assert "non_existent.py" in broken

        # Case 3: Path is a directory with files and subdirectories
        # Reset state
        skipped = []
        broken = []
        mock_isdir.side_effect = lambda p: p == "dir_path"
        mock_exists.side_effect = lambda p: True
        # Mock walk to return one level of directory structure
        # dirpath, dirnames, filenames
        mock_walk.return_value = [
            ("dir_path", ["subdir"], ["script1.py", "readme.txt"]),
        ]
        # Path.resolve logic for visited_dirs check
        mock_resolve.side_effect = lambda: MagicMock() 
        # Mocking is_supported_filetype to only allow .py
        config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")
        
        # Simulate 'subdir' being skipped via config
        def is_skipped_side_effect(path):
            return "subdir" in str(path)
        config.is_skip_side_effect = is_skipped_side_effect # Note: implementation uses config.is_skipped
        config.is_skipped.side_effect = is_skipped_side_effect

        paths = ["dir_path"]
        results = list(find(paths, config, skipped, broken))

        # Check results
        assert "dir_path/script1.py" in results
        assert "dir_path/readme.txt" not in results # because is_supported_filetype returns False
        assert any("subdir" in s for s in skipped)

    # Case 4: Testing the directory removal logic (visited_dirs / symlinks simulation)
    skipped = []
    broken = []
    mock_isdir.side_effect = lambda p: p == "dir_path"
    mock_walk.return_value = [
        ("dir_path", ["sub1", "sub2"], ["file.py"]),
        ("dir_path/sub1", [], ["file2.py"]),
    ]
    # Simulate that sub2 is a duplicate/visited via resolve
    # We need to control Path(full_path).resolve()
    with patch("pathlib.Path") as mock_path_class:
        resolved_path = MagicMock()
        mock_path_instance = MagicMock()
        mock_path_class.return_value = mock_path_instance
        mock_path_instance.resolve.return_value = resolved_path
        
        # First time sub1 is encountered, it's fine. Second time (if walk returned it), it would be removed.
        # Here we just test that files in the main walk are yielded.
        results = list(find(["dir_path"], config, skipped, broken))
        assert "dir_path/file.py" in results

```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup paths and tracking lists
    paths = ["/tmp/valid_file.py", "/tmp/valid_dir", "/tmp/non_existent"]
    skipped = []
    broken = []

    # Mocking os.path and filesystem structure
    # We will simulate:
    # 1. /tmp/valid_file.py (File) -> yields path
    # 2. /tmp/non_existent (Missing) -> adds to broken
    # 3. /tmp/valid_dir (Dir) containing:
    #    - /tmp/valid_dir/sub/file.py (Supported) -> yields path
    #    - /tmp/valid_dir/skipped_dir (Skipped by config) -> adds to skipped, stops recursion
    #    - /tmp/valid_dir/ignored.txt (Unsupported type) -> ignored

    with patch("os.path.exists") \
            .patch("os.path.isdir") \
            .patch("os.walk") \
            .patch("os.path.abspath") \
            .patch("pathlib.Path.resolve") as mock_resolve:
        
        # Define behavior for existence and type checks
        def exists_side_effect(p):
            return p != "/tmp/non_existent"
        
        def isdir_side_effect(p):
            return p == "/tmp/valid_dir"

        def abs_path_side_effect(p):
            return p

        os.path.exists.side_effect = exists_side_effect
        os.path.isdir.side_effect = isdir_side_effect
        os.path.abspath.side_effect = abs_path_side_effect
        mock_resolve.side_effect = lambda: Path("/tmp/dummy")

        # Define os.walk behavior
        # dirpath, dirnames, filenames
        os.walk.return_value = [
            ("/tmp/valid_dir", ["sub", "skipped_dir"], ["ignored.txt"]),
            ("/tmp/valid_dir/sub", [], ["file.py"]),
            ("/tmp/valid_dir/skipped_dir", [], []),
        ]

        # Define config side effects for skipping logic
        def is_skipped_side_effect(p):
            return "skipped_dir" in str(p)
        
        def is_supported_side_effect(p):
            return p.endswith(".py")

        config.is_skipped.side_effect = is_skipped_side_effect
        config.is_supported_filetype.side_effect = is_supported_side_effect

        # Execution
        result = list(find(paths, config, skipped, broken))

        # Assertions
        assert "/tmp/valid_file.py" in result
        assert "/tmp/valid_dir/sub/file.py" in result
        assert "/tmp/ignored.txt" not in result # Not supported
        assert "/tmp/non_existent" in broken
        assert any("skipped_dir" in s for s in skipped)
        assert "/tmp/non_existent" not in result
```


# LLM-generated content at query #15
#--------------------------

```python
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock


def test_find(tmp_path):
    # Setup directory structure
    # root/
    #   file1.py
    #   subdir/
    #     file2.py
    #     skipped_dir/
    #       file3.py
    #   skipped_file.py
    #   broken_path (non-existent)

    root = tmp_path / "root"
    root.mkdir()
    
    file1 = root / "file1.py"
    file1.write_text("print(1)")
    
    subdir = root / "subdir"
    subdir.mkdir()
    file2 = subdir / "file2.py"
    file2.write_text("print(2)")
    
    skipped_dir = root / "skipped_dir"
    skipped_dir.mkdir()
    file3 = skipped_dir / "file3.py"
    file3.write_text("print(3)")
    
    skipped_file = root / "skipped_file.py"
    skipped_file.write_text("print(4)")

    broken_path = str(tmp_path / "non_existent_folder")

    # Mock Config
    config = MagicMock()
    def is_supported_filetype(path):
        return path.endswith(".py")
    def is_skipped(path):
        return "skipped" in str(path)

    config.is_supported_filetype.side_effect = is_supported_filetype
    config.is_skipped.side_effect = is_skipped
    config.follow_links = False

    # Inputs
    paths = [str(root), broken_path]
    skipped = []
    broken = []

    # Execution
    results = list(find(paths, config, skipped, broken))

    # Assertions
    # Should find file1.py and file2.py
    assert str(file1) in results
    assert str(file2) in results
    # Should not find file3.py because subdir/skipped_dir is skipped
    assert str(file3) not in results
    # Should not find skipped_file.py
    assert str(skipped_file) not in results

    # Check skipped list (contains directories and files that were explicitly skipped)
    # Note: the implementation adds full_path to skipped if config.is_skipped(full_path)
    assert any("skipped_dir" in s for s in skipped)
    assert str(os.path.abspath(skipped_file)) in skipped

    # Check broken list
    assert broken_path in broken

    # Check that all found files are correct type
    for r in results:
        assert r.endswith(".py")
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

def test_find():
    # Mock Config object
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup temporary directory structure
    tmp_dir = Path("test_root")
    src_dir = tmp_dir / "src"
    sub_dir = tmp_dir / "sub"
    skipped_dir = tmp_dir / "skipped_folder"
    file1 = src_dir / "file1.py"
    file2 = sub_dir / "file2.py"
    file3 = skipped_dir / "file3.py"
    broken_path = "non_existent_path"

    tmp_dir.mkdir(exist_ok=True)
    src_dir.mkdir(exist_ok=True)
    sub_dir.mkdir(exist_ok=True)
    skipped_dir.mkdir(exist_ok=True)
    file1.write_text("print(1)")
    file2.write_exists = True # Just to ensure exists
    file2.touch()
    file3.touch()

    try:
        # Test Case 1: Find files in directories and handle skipped/broken paths
        # We need to control is_skipped for the specific directory
        def side_effect_is_skipped(path):
            return "skipped_folder" in str(path)
        
        config.is_skipped.side_effect = side_effect_is_skipped
        
        paths = [str(src_dir), str(sub_dir), str(skipped_dir), str(tmp_dir / "single.py"), broken_path]
        # Create single file for testing direct path yield
        single_file = tmp_dir / "single.py"
        single_file.touch()
        # Add the broken path to paths list (already done)

        skipped = []
        broken = []
        
        results = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. Check found files (must be absolute or relative as yielded by os.path.join/yield path)
        # Note: find yields filepath from os.walk or the path itself if it's a file
        assert any(str(file1) in r for r in results)
        assert any(str(file2) in r for r in results)
        assert any("single.py" in r for r in results)

        # 2. Check skipped directories/files
        assert any("skipped_folder" in s for s in skipped)
        
        # 3. Check broken paths
        assert broken_path in broken

    finally:
        # Cleanup
        import shutil
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

def test_find_visited_dirs_logic():
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    tmp_dir = Path("test_visited")
    dir1 = tmp_dir / "dir1"
    dir2 = tmp_dir / "dir2"
    dir1.mkdir(parents=True, exist_ok=True)
    dir2.mkdir(parents=True, exist_ok=True)
    
    file1 = dir1 / "file1.py"
    file1.touch()

    # Simulate a symlink/duplicate directory logic via mock if necessary, 
    # but here we test the basic flow of visited_dirs
    try:
        skipped = []
        broken = []
        results = list(find([str(tmp_dir)], config, skipped, broken))
        assert len(results) >= 1
    finally:
        import shutil
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup temporary directory structure using tmp_path fixture (implicitly available in pytest)
    # Note: Since we cannot add imports, we assume 'tmp_path' is passed or handled via standard pytest mechanics
    # For a standalone test function, we will mock os/path behavior to avoid dependency on filesystem state
    
    paths = ["/valid/file.py", "/valid/dir", "/nonexistent", "/skipped/file.py"]
    skipped = []
    broken = []

    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.walk") as mock_walk, \
         patch("os.path.abspath") as mock_abspath, \
         patch("pathlib.Path.resolve") as mock_resolve:

        # Define behaviors
        def isdir_side_effect(p):
            return p == "/valid/dir"
        
        def exists_side_effect(p):
            return p != "/nonexistent"

        mock_isdir.side_effect = isdir_side_effect
        mock_exists.side_effect = exists_side_effect
        mock_abspath.side_effect = lambda x: x # simplified
        
        # Mock os.walk for the directory case
        # dirpath, dirnames, filenames
        mock_walk.return_value = [
            ("/valid/dir", ["subdir"], ["file1.py"]),
            ("/valid/dir/subdir", [], ["file2.py"])
        ]
        
        # Mock Path.resolve to prevent issues with actual filesystem resolution during testing
        mock_resolve.return_value = MagicMock(spec=Path)

        # Case: Specific file type skip logic
        def is_skipped_side_effect(p):
            return str(p) == "/skipped/file.py"
        config.is_skipped.side_effect = is_skipped_side_effect

        # Execute
        result = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. /valid/file.py (direct file) -> yielded
        # 2. /valid/dir (directory) -> yields files inside via walk
        # 3. /nonexistent -> added to broken
        # 4. /skipped/file.py -> added to skipped
        
        assert "/valid/file.py" in result
        assert "/valid/dir/file1.py" in result
        assert "/valid/dir/subdir/file2.py" in result
        assert "/nonexistent" in broken
        assert any("/skipped/file.py" in s for s in skipped)

    # Test directory skipping logic
    config.is_skipped.side_effect = None
    config.is_skipped.return_value = True
    skipped = []
    mock_walk.return_value = [("/valid/dir", ["subdir"], ["file1.py"])]
    
    result = list(find(["/valid/dir"], config, skipped, broken))
    # Since subdir is skipped, it shouldn't yield file2.py from inside it
    assert not any("subdir" in r for r in result)
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    skipped = []
    broken = []

    # Test Case 1: Single file path that exists
    with patch("os.path.isdir", return_value=False), \
         patch("os.path.exists", return_value=True):
        paths = ["test.py"]
        result = list(find(paths, config, skipped, broken))
        assert result == ["test.py"]

    # Test Case 2: Path does not exist (broken)
    with patch("os.path.isdir", return_value=False), \
         patch("os.path.exists", return_value=False):
        paths = ["non_existent.py"]
        result = list(find(paths, config, skipped, broken))
        assert result == []
        assert "non_existent.py" in broken

    # Test Case 3: Directory traversal with files and skips
    # We simulate a structure:
    # root/
    #   file1.py
    #   subdir/
    #     file2.py
    #   skipped_dir/ (to be skipped)
    #   skipped_file.py (to be skipped)

    with patch("os.path.isdir", return_value=True), \
         patch("os.walk") as mock_walk, \
         patch("os.path.exists", return_value=True), \
         patch("pathlib.Path.resolve", side_effect=lambda x: x):
        
        # Mocking os.walk behavior
        # 1st iteration: root/
        # 2nd iteration: root/subdir/
        # 3rd iteration: root/skipped_dir/
        mock_walk.return_value = [
            ("root", ["subdir", "skipped_dir"], ["file1.py", "skipped_file.py"]),
            ("root/subdir", [], ["file2.py"]),
            ("root/skipped_dir", [], ["hidden.py"]),
        ]

        # Define side effects for config methods
        def is_skipped_side_effect(path):
            return "skipped" in str(path)

        config.is_skipped.side_effect = is_skipped_side_effect
        
        # Mocking filetype support: only .py files are supported
        def is_supported_side_effect(filepath):
            return filepath.endswith(".py")
        config.is_supported_filetype.side_effect = is_supported_side_effect

        paths = ["root"]
        result = list(find(paths, config, skipped, broken))

        # Assertions for files found
        # Note: os.path.join/abspath might vary by OS, but we assume standard behavior here
        # file1.py is in root (not skipped)
        # file2.py is in subdir (not skipped)
        assert any("file1.py" in f for f in result)
        assert any("file2.py" in f for f in result)
        
        # Assertions for skipped items
        assert any("skipped_file.py" in s for s in skipped)
        assert any("skipped_dir" in s for s in skipped)

        # Ensure unsupported files (if any were added to walk but not .py) are not in result
        for r in result:
            assert config.is_supported_filetype(r) is True
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    skipped = []
    broken = []

    # Test case 1: Path does not exist
    with patch("os.path.exists", return_value=False):
        paths = ["non_existent_path"]
        results = list(find(paths, config, skipped, broken))
        assert results == []
        assert "non_existent_path" in broken

    # Test case 2: Path is a direct file
    with patch("os.path.exists", return_value=True), \
         patch("os.path.isdir", return_value=False):
        paths = ["existing_file.py"]
        results = list(find(paths, config, skipped, broken))
        assert results == ["existing_file.py"]

    # Test case 3: Path is a directory with files and skips
    # Setup mock filesystem structure using os.walk side effects
    # We simulate one directory containing one file and one skipped directory
    with patch("os.path.exists", return_value=True), \
         patch("os.path.isdir", return_value=True), \
         patch("os.walk") as mock_walk:
        
        # Mock os.walk to return: dirpath, [dirnames], [filenames]
        mock_walk.return_value = [
            ("/root", ["sub_dir", "skip_me"], ["file1.py", "file2.txt"]),
        ]
        
        # Configure config behavior
        def is_skipped_side_effect(path):
            return "skip_me" in str(path)
        
        def is_supported_side_effect(filepath):
            return filepath.endswith(".py")

        config.is_skipped.side_effect = is_skipped_side_effect
        config.is_supported_filetype.side_effect = is_supported_side_effect
        
        # Mock Path.resolve to avoid actual filesystem dependency in logic
        with patch("pathlib.Path.resolve", return_value=Path("/root/sub_dir")):
            paths = ["/root"]
            results = list(find(paths, config, skipped, broken))

            # Check if file2.txt was filtered out by supported_filetype
            assert "file1.py" in results or "/root/file1.py" in results
            assert not any("file2.txt" in r for r in results)
            
            # Check if skip_me was added to skipped list
            assert any("skip_me" in s for s in skipped)

    # Test case 4: Broken paths accumulation
    skipped = []
    broken = []
    with patch("os.path.exists", return_value=False):
        paths = ["bad1", "bad2"]
        list(find(paths, config, skipped, broken))
        assert "bad1" in broken
        assert "bad2" in broken
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    skipped = []
    broken = []

    # Test Case 1: Single existing file path
    with patch("os.path.exists", return_value=True), \
         patch("os.path.isdir", return_value=False):
        paths = ["file.py"]
        result = list(find(paths, config, skipped, broken))
        assert result == ["file.py"]

    # Test Case 2: Non-existent path (Broken)
    with patch("os.path.exists", return_value=False), \
         patch("os.path.isdir", return_value=False):
        paths = ["non_existent.py"]
        result = list(find(paths, config, skipped, broken))
        assert result == []
        assert "non_existent.py" in broken

    # Test Case 3: Directory traversal with files and skips
    # Setup a fake directory structure using os.walk mock
    # Structure:
    # root/
    #   a.py (supported)
    #   b.txt (unsupported)
    #   skipped_dir/ (skipped)
    #     c.py
    #   root/ (symlink/duplicate check simulation)
    
    with patch("os.path.isdir", return_value=True), \
         patch("os.walk") as mock_walk, \
         patch("os.path.exists", return_value=True), \
         patch("pathlib.Path.resolve") as mock_resolve:
        
        # Mock os.walk behavior
        # 1st call: root directory
        # 2nd call: inside skipped_dir
        mock_walk.return_value = [
            ("/root", ["skipped_dir"], ["a.py", "b.txt"]),
            ("/root/skipped_dir", [], ["c.py"]),
        ]
        
        # Mock path resolution to simulate visited dirs
        mock_resolve.side_effect = lambda: MagicMock() 
        
        # Configure Config behavior for specific files
        def is_skipped_side_effect(path):
            return "skipped_dir" in str(path)
        config.is_skipped.side_effect = is_skipped_side_effect
        
        def is_supported_side_effect(filepath):
            return filepath.endswith(".py")
        config.is_supported_filetype.side_effect = is_supported_side_effect

        paths = ["/root"]
        result = list(find(paths, config, skipped, broken))

        # a.py should be yielded
        assert "/root/a.py" in result
        # b.txt should NOT be yielded (unsupported)
        assert "/root/b.txt" not in result
        # c.py should NOT be yielded because its parent dir was skipped
        assert "/root/skipped_dir/c.py" not in result
        # The directory itself should be in skipped list
        assert any("skipped_dir" in s for s in skipped)

    # Test Case 4: Broken paths accumulation
    broken = []
    with patch("os.path.isdir", return_value=False), \
         patch("os.path.exists", return_value=False):
        paths = ["missing1.py", "missing2.py"]
        list(find(paths, config, skipped, broken))
        assert "missing1.py" in broken
        assert "missing2.py" in broken

    # Test Case 5: Absolute path handling for supported files
    with patch("os.path.isdir", return_value=False), \
         patch("os.path.exists", return_value=True), \
         patch("os.path.abspath", side_effect=lambda x: x):
        # Mocking a single file that is supported and not skipped
        paths = ["test.py"]
        config.is_skipped.return_value = False
        config.is_supported_filetype.return_value = True
        result = list(find(paths, config, [], []))
        assert result == ["test.py"]
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    skipped = []
    broken = []

    # Test Case 1: File path that does not exist
    with patch("os.path.exists", return_value=False):
        paths = ["/non/existent/path"]
        results = list(find(paths, config, skipped, broken))
        assert results == []
        assert "/non/existent/path" in broken

    # Test Case 2: Direct file path that exists
    with patch("os.path.exists", return_value=True), \
         patch("os.path.isdir", return_value=False):
        paths = ["/existing/file.py"]
        results = list(find(paths, config, skipped, broken))
        assert results == ["/existing/file.py"]

    # Test Case 3: Directory traversal
    # Setup a mock directory structure
    # /root/a.py
    # /root/subdir/b.py
    # /root/skipped_dir/ (should be skipped)
    # /root/unsupported.txt (should be filtered by is_supported_filetype)
    
    with patch("os.path.isdir", return_value=True), \
         patch("os.walk") as mock_walk, \
         patch("os.path.exists", return_value=True), \
         patch("pathlib.Path.resolve", side_effect=lambda x: x):
        
        mock_walk.return_value = [
            ("/root", ["subdir", "skipped_dir"], ["a.py", "unsupported.txt"]),
            ("/root/subdir", [], ["b.py"]),
            ("/root/skipped_dir", [], ["hidden.py"])
        ]

        # Define behavior for is_skipped
        def side_effect_is_skipped(path):
            return "skipped_dir" in str(path)
        
        config.is_skipped.side_effect = side_effect_is_skipped
        
        # Define behavior for supported filetypes
        def side_effect_supported(filepath):
            return filepath.endswith(".py")
        
        config.is_supported_filetype.side_effect = side_effect_supported

        paths = ["/root"]
        results = list(find(paths, config, skipped, broken))

        # Verify findings
        # a.py is found, unsupported.txt is filtered out by type, b.py is found
        # skipped_dir is added to skipped and its contents are not walked
        assert "/root/a.py" in results
        assert "/root/subdir/b.py" in results
        assert "unsupported.txt" not in results
        
        # Verify skipped list contains the directory that was skipped
        assert any("skipped_dir" in s for s in skipped)

    # Test Case 4: Visited directories (avoiding infinite loops/duplicates)
    with patch("os.path.isdir", return_value=True), \
         patch("os.walk") as mock_walk, \
         patch("os.path.exists", return_value=True), \
         patch("pathlib.Path.resolve") as mock_resolve:
        
        # Simulate a symlink/duplicate path via resolve
        mock_resolve.side_effect = [
            pytest.importorskip("pathlib").Path("/root/dir1"), # for dirnames loop 1
            pytest.importorskip("pathlib").Path("/root/dir2"), # for dirnames loop 2
            pytest.importorskip("pathlib").Path("/root/dir1")  # duplicate resolve
        ]
        
        mock_walk.return_value = [
            ("/root", ["dir1", "dir2"], ["file1.py"]),
            ("/root/dir1", [], ["file1.py"]),
            ("/root/dir2", [], ["file2.py"])
        ]

        paths = ["/root"]
        results = list(find(paths, config, skipped, broken))
        
        # The second directory should have been skipped because its resolved path was already visited
        assert "/root/dir2/file2.py" not in results
```


# LLM-generated content at query #22
#--------------------------

```python
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock


def test_find(tmp_path):
    # Setup directory structure
    # root/
    #   file1.py
    #   dir1/
    #     file2.py
    #     skipped_dir/
    #       file3.py
    #   dir2/
    #     file4.txt (unsupported)
    #   broken_path (does not exist)

    root = tmp_path / "root"
    root.mkdir()
    
    file1 = root / "file1.py"
    file1.write_text("print(1)")
    
    dir1 = root / "dir1"
    dir1.mkdir()
    file2 = dir1 / "file2.py"
    file2.write_text("print(2)")
    
    skip_dir = dir1 / "skipped_dir"
    skip_dir.mkdir()
    file3 = skip_dir / "file3.py"
    file3.write_text("print(3)")
    
    dir2 = root / "dir2"
    dir2.mkdir()
    file4 = dir2 / "file4.txt"
    file4.write_text("not python")

    broken_path = str(tmp_path / "non_existent_path")

    # Mock Config
    config = MagicMock()
    # Assume .py files are supported, others are not
    config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")
    # Assume 'skipped_dir' is skipped
    config.is_skipped.side_effect = lambda p: "skipped_dir" in str(p)
    config.follow_links = False

    paths = [str(root), broken_path]
    skipped = []
    broken = []

    # Execute
    results = list(find(paths, config, skipped, broken))

    # Assertions
    # file1.py and file2.py should be found
    # file3.py is in a skipped directory
    # file4.txt is not a supported filetype
    # broken_path should be in broken list
    
    expected_files = [str(file1.absolute()), str(file2.absolute())]
    assert sorted(results) == sorted(expected_files)
    
    # Check skipped: the directory itself was skipped, so file3 path should not appear in results
    # but since config.is_skipped is called on the dir, it adds to skipped list
    assert any("skipped_dir" in s for s in skipped)
    
    # Check broken
    assert broken == [broken_path]

def test_find_single_file(tmp_path):
    file_path = tmp_path / "standalone.py"
    file_path.write_text("pass")
    
    config = MagicMock()
    config.is_supported_filetype.return_value = True
    config.is_skipped.return_value = False
    config.follow_links = False

    results = list(find([str(file_path)], config, [], []))
    assert results == [str(file_path.absolute())]

def test_find_empty_paths(config_mock):
    # Verification of behavior with empty input
    results = list(find([], config_mock, [], []))
    assert results == []
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup temporary directory structure
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.walk") as mock_walk, \
         patch("pathlib.Path.resolve") as mock_resolve:

        # Case 1: Path is a file that exists
        mock_isdir.return_value = False
        mock_exists.return_value = True
        paths = ["file.py"]
        skipped = []
        broken = []
        
        result = list(find(paths, config, skipped, broken))
        assert result == ["file.py"]

        # Case 2: Path does not exist (Broken)
        mock_exists.return_value = False
        paths = ["nonexistent.py"]
        skipped = []
        broken = []
        
        result = list(find(paths, config, skipped, broken))
        assert result == []
        assert "nonexistent.py" in broken

        # Case 3: Path is a directory with files and subdirectories
        mock_isdir.return_value = True
        mock_exists.return_value = True
        # Simulate os.walk: root, [subdirs], [files]
        mock_walk.return_value = [
            ("/root", ["subdir"], ["file1.py", "ignore.txt"]),
            ("/root/subdir", [], ["file2.py"])
        ]
        
        # Mocking Path behavior for dirnames and resolved paths
        # We need to simulate that /root/subdir is NOT a duplicate
        mock_resolve.side_effect = lambda: MagicMock() 
        
        # Configure config behaviors for specific files
        def side_effect_is_skipped(path):
            return str(path) == "/root/skip_me"

        def side_effect_is_supported(filepath):
            return filepath.endswith(".py")

        config.is_skipped.side_effect = side_effect_is_skipped
        config.is_supported_filetype.side_effect = side_effect_is_supported

        # Add a skipped directory to the walk simulation
        mock_walk.return_value = [
            ("/root", ["skip_me"], ["file1.py"]),
            ("/root/skip_me", [], ["hidden.py"]),
            ("/root/subdir", [], ["file2.py"])
        ]

        paths = ["/root"]
        skipped = []
        broken = []
        
        result = list(find(paths, config, skipped, broken))
        
        # file1.py is found, skip_me directory is added to skipped and its contents ignored
        # subdir/file2.py is found because it's not skipped
        assert "file1.py" in result or "/root/file1.py" in result # depends on os.path.join behavior
        # Checking logic of skips:
        assert any("skip_me" in s for s in skipped)

    # Case 4: Testing filetype filtering
    mock_isdir.return_value = True
    mock_walk.return_value = [("/root", [], ["file1.py", "image.png"])]
    config.is_supported_filetype.side_effect = lambda x: x.endswith(".py")
    
    paths = ["/root"]
    skipped = []
    broken = []
    result = list(find(paths, config, skipped, broken))
    assert len(result) == 1
    assert result[0].endswith("file1.py")

def test_find_visited_dirs():
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.walk") as mock_walk, \
         patch("pathlib.Path.resolve") as mock_resolve:

        mock_isdir.return_value = True
        mock_exists.return_value = True
        
        # Simulate two different paths pointing to the same resolved directory
        path1 = "/dir1"
        path2 = "/dir2"
        mock_walk.side_effect = [
            ("/dir1", ["sub"], ["f1.py"]),
            ("/dir2", ["sub"], ["f2.py"])
        ]
        
        # Both paths resolve to the same physical location
        resolved_path = MagicMock()
        mock_resolve.return_value = resolved_path

        paths = [path1, path2]
        skipped = []
        broken = []
        
        result = list(find(paths, config, skipped, broken))
        
        # The second walk should skip 'sub' because it was already visited via /dir1/sub
        # But f1.py and f2.py are inside the dirpaths themselves
        assert len(result) >= 2
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

def test_find():
    # Mock Config object
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup temporary directory structure
    tmp_dir = Path("test_root")
    tmp_dir.mkdir(exist_ok=True)
    subdir = tmp_dir / "subdir"
    subdir.mkdir(exist_ok=True)
    file1 = tmp_dir / "file1.py"
    file2 = subdir / "file2.py"
    file1.write_text("print(1)")
    file2.write_text("print(2)")
    
    skipped = []
    broken = []

    try:
        # Test Case 1: Single file path
        results = list(find([str(file1)], config, skipped, broken))
        assert results == [str(file1)]

        # Test Case 2: Directory traversal
        results = list(find([str(tmp_dir)], config, skipped, broken))
        # Note: order depends on os.walk, but should contain both files
        assert str(file1) in results
        assert str(file2) in results

        # Test Case 3: Broken path
        broken_path = "non_existent_path"
        list(find([broken_path], config, skipped, broken))
        assert broken_path in broken

        # Test Case 4: Skipped file
        config.is_skipped.side_effect = lambda p: str(p) == str(file1)
        skipped.clear()
        results = list(find([str(tmp_dir)], config, skipped, broken))
        assert str(file1) in skipped
        assert str(file2) in results

        # Test Case 5: Unsupported file type
        config.is_skipped.side_effect = lambda p: False
        config.is_supported_filetype.side_effect = lambda p: "py" in p
        
        txt_file = tmp_dir / "notes.txt"
        txt_file.write_text("hello")
        results = list(find([str(tmp_dir)], config, skipped, broken))
        assert str(txt_file) not in results

    finally:
        # Cleanup
        for p in [file1, file2, tmp_dir, subdir]:
            if p.exists():
                if p.is_file():
                    p.unlink()
                else:
                    import shutil
                    shutil.rmtree(p)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Setup Config mock
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup paths, skipped, and broken lists
    paths = ["dir1", "file1.py", "non_existent"]
    skipped = []
    broken = []

    # Mocking os/path filesystem structure
    # We simulate:
    # dir1/ (directory)
    #   dir1/subdir (directory, to be skipped)
    #   dir1/file2.py (supported file)
    #   dir1/ignore.txt (unsupported file type)
    # file1.py (exists as a file)
    # non_existent (does not exist)

    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.walk") as mock_walk, \
         patch("os.path.abspath") as mock_abspath:

        # Define behavior for path existence/type
        def side_effect_isdir(p):
            return p == "dir1"
        mock_isdir.side_effect = side_effect_isdir

        def side_effect_exists(p):
            return p in ["dir1", "file1.py"]
        mock_exists.side_effect = side_effect_exists

        # Define behavior for os.walk
        # We'll simulate one walk for 'dir1'
        # dirnames contains 'subdir', filenames contains 'file2.py' and 'ignore.txt'
        mock_walk.return_value = [
            ("dir1", ["subdir"], ["file2.py", "ignore.txt"]),
        ]

        # Define behavior for config.is_skipped
        # Let's say 'dir1/subdir' is skipped
        def side_effect_is_skipped(p):
            return str(p).endswith("subdir")
        config.is_skipped.side_effect = side_effect_is_skipped

        # Define behavior for config.is_supported_filetype
        # Let's say '.txt' is not supported
        def side_effect_is_supported(p):
            return not p.endswith(".txt")
        config.is_supported_filetype.side_effect = side_effect_is_supported

        # Mock abspath to return the path as is for simplicity
        mock_abspath.side_effect = lambda x: x

        # Execute function
        result = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. 'file1.py' should be yielded (it exists and is a file)
        # 2. 'dir1/file2.py' should be yielded (it is supported)
        # 3. 'non_existent' should be in broken
        # 4. 'dir1/subdir' should be in skipped
        # 5. 'dir1/ignore.txt' should NOT be in result because it's not a supported filetype
        
        assert "file1.py" in result
        assert any("file2.py" in r for r in result)
        assert "non_existent" in broken
        assert any("subdir" in s for s in skipped)
        assert not any(r.endswith(".txt") for r in result)
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Setup Config Mock
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup paths and tracking lists
    paths = ["/fake/dir", "/fake/file.py", "/non/existent"]
    skipped = []
    broken = []

    # Mocking os/path behaviors
    # We need to simulate: 
    # 1. /fake/dir is a directory
    # 2. /fake/file.py is a file
    # 3. /non/existent does not exist
    
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.walk") as mock_walk, \
         patch("os.path.abspath") as mock_abspath, \
         patch("pathlib.Path.resolve") as mock_resolve:

        # Define side effects for os.path checks
        def isdir_side_effect(p):
            return p == "/fake/dir"
        mock_isdir.side_effect = isdir_side_effect

        def exists_side_effect(p):
            return p in ["/fake/dir", "/fake/file.py"]
        mock_exists.side_effect = exists_side_effect

        # Mock os.walk for the directory
        # dirpath, dirnames, filenames
        mock_walk.return_value = [
            ("/fake/dir", ["subdir"], ["file1.py", "ignore.txt"]),
        ]
        
        # Simulate subdir being skipped via config
        # We'll make is_skipped return True ONLY for the subdir
        def is_skipped_side_effect(p):
            return str(p) == "/fake/dir/subdir"
        config.is_skipped.side_effect = is_skipped_side_effect
        
        # Mock supported filetype (only .py files)
        def is_supported_side_effect(p):
            return p.endswith(".py")
        config.is_supported_filetype.side_effect = is_supported_side_effect

        # Mock absolute path and resolution to prevent real filesystem access
        mock_abspath.side_effect = lambda x: x
        mock_resolve.return_value = MagicMock(spec=Path)

        # Execute the function
        result = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. Check yielded files (file.py from paths list + file1.py from walk)
        assert "/fake/file.py" in result
        assert "/fake/dir/file1.py" in result
        assert "ignore.txt" not in result # Because is_supported_filetype returns False for it

        # 2. Check broken paths (non-existent path)
        assert "/non/existent" in broken

        # 3. Check skipped paths (the subdir was identified as skipped)
        assert "/fake/dir/subdir" in skipped

        # 4. Verify the logic of dirnames removal (ensure walk didn't enter subdir)
        # Since we mocked os.walk manually, we check if is_skipped was called on the directory
        config.is_skipped.assert_any_call(Path("/fake/dir/subdir"))
```


# LLM-generated content at query #5
#--------------------------

```python
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock


def test_find(tmp_path):
    # Setup temporary file structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    sub_dir = src_dir / "subdir"
    sub_dir.mkdir()
    
    file1 = src_dir / "main.py"
    file1.write_text("print('hello')")
    file2 = sub_dir / "utils.py"
    file2.write_text("def utils(): pass")
    file3 = src_dir / "ignored.txt"
    file3.write_text("ignore me")
    
    broken_path = "/non/existent/path"
    skipped_dir_name = "skipped_dir"
    skipped_dir = src_dir / skipped_dir_name
    skipped_dir.mkdir()
    file4 = skipped_dir / "hidden.py"
    file4.write_text("secret")

    # Mock Config object
    config = MagicMock()
    # Simulate is_supported_filetype: only .py files
    config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")
    # Simulate is_skipped: skip the 'skipped_dir' and specific files
    def is_skipped_logic(path):
        return str(path).endswith(skipped_dir_name) or "ignored.txt" in str(path)
    config.is_skipped.side_effect = is_skipped_logic
    config.follow_links = False

    paths = [str(src_dir), broken_path, str(file1)]
    skipped = []
    broken = []

    # Execution
    result = list(find(paths, config, skipped, broken))

    # Assertions
    # 1. Check yielded files (should be main.py and utils.py)
    assert os.path.abspath(str(file1)) in [os.path.abspath(r) for r in result]
    assert os.path.abspath(str(file2)) in [os.path.abspath(r) for r in result]
    assert len(result) == 2

    # 2. Check skipped list (the directory and the txt file)
    # Note: The implementation adds the full_path of the dir to skipped if is_skipped returns True
    assert any(skipped_dir_name in s for s in skipped)
    assert any("ignored.txt" in s for s in skipped)

    # 3. Check broken list
    assert broken_path in broken

    # 4. Verify single file path input works
    single_file_result = list(find([str(file1)], config, [], []))
    assert len(single_file_result) == 1
    assert os.path.abspath(single_file_result[0]) == os.path.abspath(str(file1))
```


# LLM-generated content at query #6
#--------------------------

```python
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

def test_find(tmp_path):
    # Setup directory structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    sub_dir = src_dir / "subdir"
    sub_dir.mkdir()
    
    file1 = src_dir / "main.py"
    file1.write_text("print(1)")
    file2 = sub_dir / "utils.py"
    file2.write_text("print(2)")
    file3 = src_dir / "data.txt"
    file3.write_text("not python")
    
    ignored_dir = src_dir / ".ignore_me"
    ignored_dir.mkdir()
    ignored_file = ignored_dir / "secret.py"
    ignored_file.write_text("secret")

    # Mock Config
    config = MagicMock()
    # Simulate is_supported_filetype: only .py files
    config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")
    # Simulate is_skipped: skip anything in ".ignore_me" directory
    config.is_skipped.side_mock = lambda p: ".ignore_me" in str(p)
    config.is_skipped.side_effect = lambda p: ".ignore_me" in str(p)
    config.follow_links = False

    paths = [str(src_dir), "non_existent_path"]
    skipped = []
    broken = []

    # Execute
    results = list(find(paths, config, skipped, broken))

    # Assertions
    # 1. Check yielded files (only supported and not skipped)
    assert str(file1) in results
    assert str(file2) in results
    assert str(file3) not in results  # Not a .py file
    
    # 2. Check skipped files/dirs
    assert any(".ignore_me" in s for s in skipped)
    assert str(ignored_file) in skipped or any("secret.py" in s for s in skipped)

    # 3. Check broken paths
    assert "non_existent_path" in broken

    # 4. Verify all yielded files are absolute/correct paths
    for res in results:
        assert os.path.exists(res)
```


# LLM-generated content at query #7
#--------------------------

```python
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

def test_find(tmp_path):
    # Setup directory structure
    # root/
    #   file1.py
    #   subdir/
    #     file2.py
    #     skipped_dir/ (to be skipped)
    #       file3.py
    #   broken_path (non-existent)
    #   not_a_python_file.txt
    
    root = tmp_path / "root"
    root.mkdir()
    
    file1 = root / "file1.py"
    file1.write_text("content")
    
    subdir = root / "subdir"
    subdir.mkdir()
    file2 = subdir / "file2.py"
    file2.write_text("content")
    
    skipped_dir = root / "skipped_dir"
    skipped_dir.mkdir()
    file3 = skipped_dir / "file3.py"
    file3.write_text("content")
    
    txt_file = root / "not_a_python_file.txt"
    txt_file.write_text("content")
    
    broken_path = str(tmp_path / "non_existent_path")
    
    # Mock Config
    config = MagicMock()
    def is_skipped(path):
        return "skipped_dir" in str(path)
    
    def is_supported_filetype(path):
        return path.endswith(".py")

    config.is_skipped.side_effect = is_skipped
    config.is_supported_filetype.side_effect = is_supported_filetype
    config.follow_links = False

    paths = [str(root), broken_path]
    skipped = []
    broken = []

    # Execution
    results = list(find(paths, config, skipped, broken))

    # Assertions
    # Should find file1.py and file2.py
    assert str(file1) in results
    assert str(file2) in results
    assert str(txt_file) not in results
    
    # Should identify skipped directory contents as skipped
    assert any("skipped_dir" in s for s in skipped)
    assert str(skipped_dir) in skipped
    
    # Should identify broken path
    assert broken_path in broken

def test_find_single_file():
    config = MagicMock()
    config.is_supported_filetype.return_value = True
    config.is_skipped.return_value = False
    
    path = "/tmp/test_file.py"
    # Mocking os.path behaviors for a single file input
    with pytest.MonkeyPatch.context() as m:
        m.setattr(os, "path", MagicMock())
        m.setattr(os, "isdir", lambda x: False)
        m.setattr(os, "exists", lambda x: True)
        
        results = list(find([path], config, [], []))
        assert path in results

def test_find_with_visited_dirs(tmp_path):
    # Test that resolved paths prevent infinite loops/re-visiting
    root = tmp_path / "root"
    root.mkdir()
    subdir = root / "subdir"
    subdir.mkdir()
    file1 = root / "file1.py"
    file1.write_text("content")
    
    # Create a symlink to the same directory (if OS allows)
    # Since we cannot guarantee symlink permissions in all CI, 
    # we rely on the logic of the function's visited_dirs set.
    
    config = MagicMock()
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True
    config.follow_links = True

    paths = [str(root)]
    skipped = []
    broken = []

    results = list(find(paths, config, skipped, broken))
    assert str(file1) in results
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    skipped = []
    broken = []

    # Test Case 1: Path does not exist (Broken)
    with patch("os.path.exists", return_value=False):
        paths = ["/non/existent/path"]
        results = list(find(paths, config, skipped, broken))
        assert results == []
        assert "/non/existent/path" in broken

    # Test Case 2: Path is a file (Direct yield)
    with patch("os.path.exists", return_value=True), \
         patch("os.path.isdir", return_value=False):
        paths = ["/existing/file.py"]
        results = list(find(paths, config, skipped, broken))
        assert results == ["/existing/mock/file.py"] # Note: logic uses path directly if file

    # Test Case 3: Path is a directory (Walking)
    # We mock os.walk and os.path.isdir to simulate a directory structure
    with patch("os.path.exists", return_value=True), \
         patch("os.path.isdir", return_value=True), \
         patch("os.walk") as mock_walk:
        
        # Setup mock walk: one dir, one file, one skipped file
        # dirpath, dirnames, filenames
        mock_walk.return_value = [
            ("/root", ["subdir"], ["file1.py", "file2.txt"]),
        ]
        
        # Mock behavior for subdirectories and files
        def side_effect_is_skipped(path):
            if "file2.txt" in str(path):
                return True
            return False
        
        config.is_skipped.side_effect = side_effect_is_skipped
        config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")

        # Setup Path.resolve to return a stable value for visited_dirs logic
        with patch("pathlib.Path.resolve", return_value=Path("/root/subdir")):
            paths = ["/root"]
            results = list(find(paths, config, skipped, broken))

        # file1.py is supported and not skipped
        assert "/root/file1.py" in results
        # file2.txt is supported but skipped (if logic dictates) or unsupported
        # In our side_effect: file2.txt is skipped via config.is_skipped
        assert any("file2.txt" in s for s in skipped)

    # Test Case 4: Directory skipping logic
    with patch("os.path.exists", return_value=True), \
         patch("os.path.isdir", return_value=True), \
         patch("os.walk") as mock_walk, \
         patch("pathlib.Path.resolve", return_value=Path("/root/skipped_dir")):
        
        mock_walk.return_value = [
            ("/root", ["skipped_dir"], ["file1.py"]),
        ]
        config.is_skipped.side_effect = lambda p: "skipped_dir" in str(p)
        config.is_supported_filetype.return_value = True

        paths = ["/root"]
        results = list(find(paths, config, skipped, broken))

        assert "/root/file1.py" in results
        assert any("skipped_dir" in s for s in skipped)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup temporary directory structure using pytest's tmp_path
    root = tmp_path / "project"
    root.mkdir()
    
    src_dir = root / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("print('hello')")
    (src_dir / "utils.py").write_text("pass")
    
    test_dir = root / "tests"
    test_dir.mkdir()
    (test_dir / "test_main.py").write_text("test")
    
    skipped_dir = root / "ignored"
    skipped_dir.mkdir()
    (skipped_dir / "secret.py").write_text("hidden")

    # Define paths to search
    paths = [str(root), str(tmp_path / "non_existent")]
    
    # Configure is_skipped for the 'ignored' directory
    def side_effect_is_skipped(p):
        return "ignored" in str(p)
    config.is_skipped.side_with = side_effect_is_skipped
    config.is_skipped.side_effect = side_effect_is_skipped

    # Define file type support (only .py files)
    def side_effect_supported(p):
        return p.endswith(".py")
    config.is_supported_filetype.side_effect = side_effect_supported

    skipped = []
    broken = []

    # Execute function
    result = list(find(paths, config, skipped, broken))

    # Assertions
    # Expected files: src/main.py, src/utils.py, tests/test_main.py
    # Note: 'ignored/secret.py' should be in skipped because its parent dir is skipped
    expected_files = {
        str(src_dir / "main.py"),
        str(src_dir / "utils.py"),
        str(test_dir / "test_main.py")
    }
    
    assert set(result) == expected_files
    assert str(skipped_dir) in skipped
    assert str(tmp_path / "non_existent") in broken

def test_find_single_file():
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    temp_file = tmp_path / "single.py"
    temp_file.write_text("content")
    
    paths = [str(temp_file)]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))

    assert result == [str(temp_file)]
    assert len(skipped) == 0
    assert len(broken) == 0

def test_find_with_symlinks():
    config = MagicMock(spec=Config)
    config.follow_links = True
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    root = tmp_path / "symlink_test"
    root.mkdir()
    target = root / "target"
    target.mkdir()
    (target / "file.py").write_text("data")
    
    link_dir = root / "link"
    os.symlink(target, link_dir)

    paths = [str(root)]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))
    
    # Because follow_links=True, it might traverse the symlinked dir.
    # The implementation uses visited_dirs to prevent infinite loops/re-visiting.
    assert str(target / "file.py") in result
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Setup Mock Config
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup paths and tracking lists
    paths = ["/root", "/nonexistent", "/file.py"]
    skipped = []
    broken = []

    # Mocking os.path and filesystem structure
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.walk") as mock_walk, \
         patch("os.path.abspath") as mock_abspath, \
         patch("pathlib.Path.resolve") as mock_resolve:

        # Scenario Setup
        # /root is a directory
        # /nonexistent does not exist
        # /file.py is a direct file
        mock_isdir.side_effect = lambda p: p == "/root"
        mock_exists.side_effect = lambda p: p != "/nonexistent"
        mock_abspath.side_effect = lambda p: p

        # Mock os.walk for /root
        # dirpath, dirnames, filenames
        mock_walk.return_value = [
            ("/root", ["subdir", "ignored_dir"], ["script.py", "README.md"]),
            ("/root/subdir", [], ["sub_script.py"])
        ]

        # Mock behavior for directory skipping logic
        # Let's make 'ignored_dir' skipped
        def is_skipped_side_effect(path):
            return "ignored_dir" in str(path)
        config.is_skipped.side_effect = is_skipped_side_effect

        # Mock behavior for supported filetypes
        # Let's make 'README.md' unsupported (not a python file)
        def is_supported_side_effect(filepath):
            return filepath.endswith(".py")
        config.is_supported_filetype.side_effect = is_supported_side_effect

        # Mock resolve for visited_dirs logic
        mock_resolve.side_effect = lambda: MagicMock()

        # Execute function
        results = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. Check yielded files (script.py and sub_script.py from /root/subdir, plus direct file.py)
        assert "/root/script.py" in results
        assert "/root/subdir/sub_script.py" in results
        assert "/file.py" in results
        # README.md should be filtered out by is_supported_filetype
        assert "/root/README.md" not in results

        # 2. Check broken paths
        assert "/nonexistent" in broken

        # 3. Check skipped paths (the directory 'ignored_dir' was skipped)
        # Since it was a directory, the loop adds it to skipped if config.is_skipped returns True
        assert any("ignored_dir" in s for s in skipped)

        # 4. Verify logic flow
        assert len(results) == 3  # script.py, sub_script.py, file.py
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Setup Mock Config
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup paths and tracking lists
    paths = ["dir1", "file1.py", "non_existent"]
    skipped = []
    broken = []

    # Create a temporary directory structure for testing
    with pytest.raises(SystemError): # Placeholder to allow logic flow if we can't use real FS
        pass 

    # We will use patch to simulate the file system without creating actual files
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.walk") as mock_walk, \
         patch("os.path.abspath") as mock_abspath, \
         patch("pathlib.Path.resolve") as mock_resolve:

        # Define behavior for paths
        # "dir1" is a directory, "file1.py" is a file, "non_existent" does not exist
        def side_effect_isdir(path):
            return path == "dir1"
        mock_isdir.side_ext_effect = side_effect_isdir
        mock_isdir.side_effect = side_effect_isdir

        def side_effect_exists(path):
            return path != "non_existent"
        mock_exists.side_effect = side_effect_exists

        # Mock os.walk for 'dir1'
        # dirpath, dirnames, filenames
        mock_walk.return_value = [
            ("dir1", ["subdir"], ["src.py"]),
            ("dir1/subdir", [], ["module.py"])
        ]

        # Mock absolute paths and resolution
        mock_abspath.side_effect = lambda x: x
        mock_resolve.side_effect = lambda: MagicMock() # Simplified for structure

        # Scenario 1: Simple file yield
        results = list(find(["file1.py"], config, skipped, broken))
        assert "file1.py" in results
        assert "non_existent" in broken

        # Scenario 2: Directory walking and skipping
        skipped = []
        broken = []
        # Mocking is_skipped to skip 'subdir'
        config.is_skipped.side_effect = lambda p: str(p) == "dir1/subdir"
        
        results = list(find(["dir1"], config, skipped, broken))
        
        # 'src.py' should be found because it's in dir1
        # 'module.py' is in subdir, but subdir was skipped
        assert "dir1/src.py" in results
        assert "dir1/subdir/module.py" not in results
        assert "dir1/subdir" in skipped

        # Scenario 3: Unsupported filetype
        skipped = []
        broken = []
        config.is_skipped.return_value = False
        config.is_supported_filetype.side_effect = lambda x: x != "dir1/src.txt"
        
        # Manually inject a txt file into the walk mock for this specific test run
        mock_walk.return_value = [("dir1", [], ["src.py", "src.txt"])]
        
        results = list(find(["dir1"], config, skipped, broken))
        assert "dir1/src.py" in results
        assert "dir1/src.txt" not in results

        # Scenario 4: File is skipped via config
        skipped = []
        broken = []
        config.is_supported_filetype.return_value = True
        config.is_skipped.side_effect = lambda p: str(p) == os.path.abspath("dir1/skip.py")
        mock_walk.return_value = [("dir1", [], ["skip.py"])]
        
        results = list(find(["dir1"], config, skipped, broken))
        assert len(results) == 0
        assert os.path.abspath("dir1/skip.py") in skipped
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup temporary directory structure
    tmp_dir = Path("test_root")
    tmp_dir.mkdir(exist_ok=True)
    sub_dir = tmp_dir / "subdir"
    sub_dir.mkdir(exist_ok=True)
    file1 = tmp_dir / "file1.py"
    file2 = sub_dir / "file2.py"
    file1.write_text("content")
    file2.write_text("content")
    
    skipped_dir = tmp_dir / "skipped_dir"
    skipped_dir.mkdir(exist_ok=True)
    skipped_file = skipped_dir / "skip.py"
    skipped_file.write_text("content")

    broken_path = "non_existent_path"
    single_file = tmp_dir / "single.py"
    single_file.write_text("content")

    try:
        # Test Case 1: Basic finding and directory walking
        paths = [str(tmp_dir)]
        skipped = []
        broken = []
        
        results = list(find(paths, config, skipped, broken))
        
        assert str(file1) in results
        assert str(file2) in results
        assert str(single_file) in results
        assert len(results) >= 3

        # Test Case 2: Broken paths
        paths_with_broken = [str(tmp_dir), broken_path]
        skipped = []
        broken = []
        list(find(paths_with_broken, config, skipped, broken))
        assert broken_path in broken

        # Test Case 3: Skipping files via config
        config.is_skipped.side_effect = lambda p: "skip.py" in str(p)
        skipped = []
        broken = []
        results = list(find([str(tmp_dir)], config, skipped, broken))
        
        # The file inside skipped_dir should be in skipped list and not yielded
        assert any("skip.py" in s for s in skipped)
        assert not any("skip.py" in r for r in results)

        # Test Case 4: Unsupported filetypes
        config.is_skipped.side_effect = lambda p: False
        config.is_supported_filetype.side_effect = lambda p: ".py" in p
        bad_file = tmp_dir / "test.txt"
        bad_file.write_text("content")
        
        results = list(find([str(tmp_dir)], config, [], []))
        assert str(bad_file) not in results

    finally:
        # Cleanup
        import shutil
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Setup Mocks
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True
    
    skipped = []
    broken = []
    paths = ["/fake/path/dir", "/fake/path/file.py", "/non/existent/path"]

    # Create a temporary directory structure for os.walk simulation
    # We will patch os.path, os.walk, and os.path.exists to control the environment
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.walk") as mock_walk, \
         patch("os.path.abspath") as mock_abspath, \
         patch("pathlib.Path.resolve") as mock_resolve:

        # Define behavior for os.path.isdir
        def isdir_side_effect(p):
            return p == "/fake/path/dir"
        mock_isdir.side_with = isdir_side_effect
        mock_isdir.side_effect = isdir_side_effect

        # Define behavior for os.path.exists
        def exists_side_effect(p):
            return p != "/non/existent/path"
        mock_exists.side_effect = exists_side_effect

        # Define behavior for os.walk (simulating one directory with files and a sub-dir)
        # dirpath, dirnames, filenames
        mock_walk.return_value = [
            ("/fake/path/dir", ["subdir"], ["file1.py", "ignore.txt"]),
            ("/fake/path/dir/subdir", [], ["file2.py"])
        ]

        # Define behavior for abspath
        mock_abspath.side_effect = lambda p: p

        # Define behavior for resolve (to handle visited_dirs logic)
        # We'll make it return the path itself to avoid complex symlink mocking
        mock_resolve.side_effect = lambda p: p

        # Configure config.is_supported_filetype for specific files
        def supported_side_effect(p):
            return p.endswith(".py")
        config.is_supported_filetype.side_effect = supported_side_effect

        # Configure config.is_skipped for a specific directory
        def skipped_side_effect(p):
            return str(p) == "/fake/path/dir/subdir"
        config.is_skipped.side_effect = skipped_side_effect

        # Execute function
        results = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. Check found files (file.py from paths input + file1.py + file2.py is skipped)
        assert "/fake/path/file.py" in results
        assert "/fake/path/dir/file1.py" in results
        # Note: file2.py should not be yielded because its parent 'subdir' was skipped
        assert "/fake/path/dir/subdir/file2.py" not in results

        # 2. Check broken paths
        assert "/non/existent/path" in broken

        # 3. Check skipped paths (the directory itself)
        assert "/fake/path/dir/subdir" in skipped

        # 4. Check that ignore.txt was filtered by supported_filetype
        assert "/fake/path/dir/ignore.txt" not in results
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    skipped = []
    broken = []

    # Create a temporary directory structure for testing
    with pytest.raises(Exception): # Should not happen, used to trigger cleanup logic if needed
        pass 

    with tmp_path.as_posix() as root:
        # Setup file structure:
        # root/
        #   file1.py
        #   subdir/
        #     file2.py
        #     skipped_dir/ (should be skipped)
        #       file3.py
        #   broken_path_not_exists
        #   file_unsupported.txt (should not yield)

        dir_root = tmp_path / "test_root"
        dir_root.mkdir()
        
        file1 = dir_root / "file1.py"
        file1.write_text("content")
        
        subdir = dir_root / "subdir"
        subdir.mkdir()
        file2 = subdir / "file2.py"
        file2.write_text("content")
        
        skip_dir = dir_root / "skipped_dir"
        skip_dir.mkdir()
        file3 = skip_dir / "file3.py"
        file3.write_text("content")
        
        unsupported = dir_root / "file_unsupported.txt"
        unsupported.write_text("content")

        # Define behavior for is_skipped: skip 'skipped_dir'
        def side_effect_is_skipped(path):
            return "skipped_dir" in str(path)
        config.is_skipped.side_effect = side_effect_is_skipped

        # Define behavior for is_supported_filetype: only .py files
        def side_effect_is_supported(path):
            return path.endswith(".py")
        config.is_supported_filetype.side_effect = side_effect_is_supported

        paths = [str(dir_root), "non_existent_path"]

        # Execute function
        result = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. Check that valid python files are yielded
        assert str(file1) in result
        assert str(file2) in result
        assert str(unsupported) not in result
        
        # 2. Check that the unsupported file was filtered by config
        assert any(".txt" in r for r in result) == False

        # 3. Check broken paths
        assert "non_existent_path" in broken

        # 4. Check skipped directories/files
        # The 'skipped_dir' directory itself or its contents should be in skipped list
        assert any("skipped_dir" in s for s in skipped)
        assert str(file3) not in result
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Setup Mock Config
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup paths and tracking lists
    paths = ["/valid/dir", "/single/file.py", "/non/existent"]
    skipped = []
    broken = []

    # Mocking os structure
    # We will mock os.path, os.walk, and os.path.exists to control the filesystem behavior
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.walk") as mock_walk, \
         patch("os.path.abspath") as mock_abspath, \
         patch("pathlib.Path.resolve") as mock_resolve:

        # Define behavior for path existence and type
        def side_effect_isdir(p):
            return p == "/valid/dir"
        mock_isdir.side_equal = side_effect_isdir
        mock_isdir.side_effect = side_effect_isdir

        def side_effect_exists(p):
            if p == "/non/existent":
                return False
            return True
        mock_exists.side_effect = side_effect_exists

        # Define behavior for os.walk
        # dirpath, dirnames, filenames
        mock_walk.return_value = [
            ("/valid/dir", ["subdir", "ignored_dir"], ["file1.py", "ignore.txt"]),
        ]

        # Setup mock for directory walking details
        # We need to simulate the logic inside the loop
        # For 'subdir', we'll make it resolved as a new path
        # For 'ignored_dir', we'll make config.is_skipped return True
        def side_effect_is_skipped(p):
            return "ignored_dir" in str(p)

        config.is_skipped.side_effect = side_effect_is_skipped
        
        # Mock abspath to return the same string for simplicity
        mock_abspath.side_effect = lambda x: x
        
        # Mock resolve to prevent real filesystem access during dir comparison
        mock_resolve.side_effect = lambda: MagicMock()

        # Execute function
        result = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. Check if the single file was yielded
        assert "/single/file.py" in result
        
        # 2. Check if valid files from walk were yielded
        # Note: 'file1.py' and 'ignore.txt' are in filenames. 
        # config.is_supported_filetype is True for all by default in our mock setup.
        assert "/valid/dir/file1.py" in result
        assert "/valid/dir/ignore.txt" in result

        # 3. Check if the broken path was captured
        assert "/non/existent" in broken

        # 4. Check if skipped directories were captured
        assert any("ignored_dir" in s for s in skipped)

    # Test logic for supported filetype filtering
    config.is_supported_filetype.return_value = False
    skipped = []
    broken = []
    result = list(find(["/valid/dir"], config, skipped, broken))
    # Since is_supported_filetype is False, no files from the walk should be yielded
    # (But /single/file.py was not in this specific run's paths)
    assert len(result) == 0

def test_find_visited_dirs():
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    paths = ["/root"]
    skipped = []
    broken = []

    with patch("os.path.isdir", return_value=True), \
         patch("os.path.exists", return_value=True), \
         patch("os.walk") as mock_walk, \
         patch("pathlib.Path.resolve") as mock_resolve:

        # Simulate a structure where two different paths resolve to the same directory
        mock_walk.return_value = [
            ("/root", ["dir1"], ["file1.py"]),
            ("/root/dir1", ["dir2"], ["file2.py"]),
        ]
        
        # Mock resolve to return a constant, simulating symlink collision
        resolved_path = MagicMock()
        mock_resolve.return_value = resolved_path

        result = list(find(paths, config, skipped, broken))

        # 'dir2' should be skipped because its resolved path is already in visited_dirs
        # However, the function yields files found in the current walk iteration.
        # The logic for dirnames.remove(dirname) affects subsequent iterations of the loop.
        assert "/root/dir1/file1.py" in result
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

def test_find():
    # Mock Config object
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    skipped = []
    broken = []

    # Create a temporary directory structure for testing
    with pytest.raises(Exception): # Placeholder to ensure we use tmp_path
        pass 

    tmp_dir = Path("test_root")
    tmp_dir.mkdir(exist_ok=True)
    subdir = tmp_dir / "subdir"
    subdir.mkdir(exist_ok=True)
    file1 = tmp_dir / "file1.py"
    file1.write_text("print(1)")
    file2 = subdir / "file2.py"
    file2.write_text("print(2)")
    unsupported = tmp_dir / "readme.txt"
    unsupported.write_text("text")

    # Setup config behavior for specific files
    def is_skipped_side_effect(path):
        if "skip_me" in str(path):
            return True
        return False

    def is_supported_side_effect(path):
        return path.endswith(".py")

    config.is_skipped.side_effect = is_skipped_side_effect
    config.is_supported_filetype.side_effect = is_supported_side_effect

    # Create a skipped directory
    skip_dir = tmp_dir / "skip_me"
    skip_dir.mkdir(exist_ok=True)
    (skip_dir / "hidden.py").write_text("hidden")

    try:
        # Test Case 1: Find files in directory
        paths = [str(tmp_dir)]
        results = list(find(paths, config, skipped, broken))
        
        # Assertions for valid files
        assert str(file1) in results
        assert str(file2) in results
        # Unsupported file should not be in results
        assert str(unsupported) not in results
        # Skipped directory content should not be in results
        assert any("skip_me" in r for r in results) is False
        # Skipped dir should be in skipped list
        assert any(str(skip_dir) in s for s in skipped)

        # Test Case 2: Single file path
        skipped = []
        broken = []
        results = list(find([str(file1)], config, skipped, broken))
        assert results == [str(file1)]

        # Test Case 3: Broken path
        skipped = []
        broken = []
        paths = ["/non/existent/path"]
        list(find(paths, config, skipped, broken))
        assert "/non/existent/path" in broken

        # Test Case 4: Mixed paths (file and dir)
        skipped = []
        broken = []
        paths = [str(file1), str(file2)]
        results = list(find(paths, config, skipped, broken))
        assert str(file1) in results
        assert str(file2) in results

    finally:
        # Cleanup
        import shutil
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock

def test_find(tmp_path):
    # Setup directory structure
    # root/
    #   file1.py
    #   dir1/
    #     file2.py
    #     skipped_dir/ (to be skipped)
    #       file3.py
    #   broken_path (does not exist)
    
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    
    file1 = src_dir / "file1.py"
    file1.write_text("print(1)")
    
    dir1 = src_dir / "dir1"
    dir1.mkdir()
    file2 = dir1 / "file2.py"
    file2.write_text("print(2)")
    
    skipped_dir = dir1 / "skipped_dir"
    skipped_dir.mkdir()
    file3 = skipped_dir / "file3.py"
    file3.write_text("print(3)")
    
    unsupported_file = src_dir / "readme.txt"
    unsupported_file.write_text("text")

    broken_path = str(tmp_path / "non_existent")
    
    # Mock Config
    config = MagicMock()
    # Assume only .py files are supported
    config.is_supported_filetype.side_effect = lambda x: x.endswith(".py")
    # Define skip logic: skip 'skipped_dir'
    config.is_skipped.side_effect = lambda p: "skipped_dir" in str(p)
    config.follow_links = False

    paths = [str(src_dir), broken_path]
    skipped = []
    broken = []

    # Execution
    results = list(find(paths, config, skipped, broken))

    # Assertions
    # File 1 and File 2 should be found
    assert os.path.abspath(str(file1)) in [os.path.abspath(r) for r in results]
    assert os.path.abspath(str(file2)) in [os.path.abspath(r) for r in results]
    
    # File 3 should be skipped because its parent dir is skipped
    assert any("skipped_dir" in s for s in skipped)
    assert os.path.abspath(str(file3)) not in [os.path.abspath(r) for r in results]

    # Unsupported file type should not be yielded
    assert os.path.abspath(str(unsupported_file)) not in [os.path.abspath(r) for r in results]

    # Broken path should be recorded
    assert broken_path in broken

    # Verify single file path input works
    single_file_path = str(file1)
    results_single = list(find([single_file_path], config, [], []))
    assert os.path.abspath(str(file1)) in [os.path.abspath(r) for r in results_single]

def test_find_visited_dirs_prevention(tmp_path):
    # Test that visited directories are not re-scanned (prevent infinite loops/redundancy)
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    
    sub_dir = src_dir / "sub"
    sub_dir.mkdir()
    file_in_sub = sub_dir / "sub.py"
    file_in_sub.write_text("content")

    config = MagicMock()
    config.is_supported_filetype.return_value = True
    config.is_skipped.return_value = False
    config.follow_links = False

    # We manually trigger the logic where a directory is added to visited_dirs 
    # via the first path, and then check if it's ignored in subsequent paths
    paths = [str(src_dir), str(sub_dir)]
    skipped = []
    broken = []

    results = list(find(paths, config, skipped, broken))
    
    # Even though sub_dir is passed twice (once via src_dir and once directly), 
    # the file should only be yielded once because of visited_dirs.
    assert len([r for r in results if "sub.py" in r]) == 1
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    skipped = []
    broken = []

    # Setup dummy file structure using patch
    # Case 1: path is a file (yielded)
    # Case 2: path is non-existent (added to broken)
    # Case 3: path is a directory with files and subdirs
    
    with patch("os.path.exists") as mock_exists, \
         patch("os.path.isdir") as mock_isdir, \
         patch("os.walk") as mock_walk, \
         patch("os.path.abspath") as mock_abspath, \
         patch("pathlib.Path.resolve") as mock_resolve:

        # Define behaviors for the mocks
        # We will simulate one directory structure:
        # root/
        #   file1.py (supported)
        #   file2.txt (unsupported)
        #   subdir/ (skipped via config)
        #   subfile.py (supported)

        mock_exists.side_effect = lambda p: p in ["valid_file.py", "valid_dir", "non_existent"]
        mock_isdir.side_effect = lambda p: p == "valid_dir"
        mock_abspath.side_effect = lambda p: p # Return as is for simplicity
        mock_resolve.side_effect = lambda: MagicMock() # Dummy resolve

        # Setup os.walk for 'valid_dir'
        # 1st iteration: root/
        # 2nd iteration: root/subdir/ (but we will trigger skip)
        mock_walk.return_value = [
            ("valid_dir", ["subdir"], ["file1.py", "file2.txt"]),
            ("valid_dir/subdir", [], ["subfile.py"]),
        ]

        # Define config behavior for specific paths
        def is_skipped_side_effect(path):
            return str(path) == "valid_dir/subdir"

        def is_supported_side_effect(filepath):
            return filepath.endswith(".py")

        config.is_skipped.side_effect = is_skipped_side_effect
        config.is_supported_filetype.side_effect = is_supported_side_effect

        # Execute function with different types of paths
        paths = ["valid_file.py", "non_existent", "valid_dir"]
        result = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. valid_file.py (direct file) should be yielded
        assert "valid_file.py" in result
        
        # 2. non_existent should be in broken
        assert "non_existent" in broken
        
        # 3. valid_dir contents:
        # file1.py is supported and not skipped -> yield
        # file2.txt is unsupported -> skip
        # subdir is skipped -> add to skipped, do not traverse
        assert "valid_dir/file1.py" in result
        assert "valid_dir/file2.txt" not in result
        assert "valid_dir/subdir" in skipped
        assert "valid_dir/subdir/subfile.py" not in result # Because subdir was skipped

        # 4. Check total results
        # Only 'valid_file.py' and 'valid_dir/file1.py' should be yielded
        assert len(result) == 2
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    skipped = []
    broken = []

    # Create a temporary directory structure for testing
    with pytest.io.tmpdir.as_dir() as tmpdir:
        # 1. Setup valid file
        valid_file = tmpdir.join("test_file.py")
        valid_file.write("print('hello')")
        
        # 2. Setup a directory with a supported file and a skipped file
        subdir = tmpdir.mkdir("subdir")
        sub_file = subdir.join("sub_file.py")
        sub_file.write("print('sub')")
        
        skipped_file = subdir.join("skipped.py")
        skipped_file.write("pass")

        # 3. Setup a directory to be skipped entirely
        skip_dir = tmpdir.mkdir("skip_me")
        inside_skip_dir = skip_dir.join("hidden.py")
        inside_skip_dir.write("pass")

        # Logic for is_skipped mock
        def is_skipped_side_effect(path):
            return "skip_me" in str(path) or "skipped.py" in str(path)
        
        config.is_skipped.side_effect = is_compatible_skipping = is_skipped_side_effect

        # Test Case 1: Valid path (file)
        paths_file = [str(valid_file)]
        results = list(find(paths_file, config, skipped, broken))
        assert str(valid_file) in results
        assert len(broken) == 0

        # Test Case 2: Valid path (directory)
        paths_dir = [str(tmpdir)]
        results = list(find(paths_dir, config, skipped, broken))
        
        # Should find valid_file and sub_file.py
        # Should NOT find skipped.py (it's in skipped list)
        # Should NOT find anything inside skip_me/
        assert str(valid_file) in results
        assert str(sub_file) in results
        assert str(skipped_file) in skipped
        assert not any("skip_me" in r for r in results)

        # Test Case 3: Broken path (non-existent)
        broken_path = str(tmpdir.join("does_not_exist.py"))
        paths_broken = [broken_path]
        list(find(paths_broken, config, skipped, broken))
        assert broken_path in broken

        # Test Case 4: Mixed paths
        paths_mixed = [str(valid_file), broken_path]
        results_mixed = list(find(paths_mixed, config, skipped, broken))
        assert str(valid_file) in results_mixed
        assert broken_path in broken

    # Test Case 5: Unsupported filetype
    config.is_supported_filetype.return_value = False
    unsupported_file = tmpdir.join("unsupported.txt")
    unsupported_file.write("text")
    results_unsupported = list(find([str(tmpdir)], config, [], []))
    # Only valid_file and sub_file.py were supported in previous logic 
    # but now everything is unsupported
    assert not any(".txt" in r for r in results_unsupported)
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config object
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    skipped = []
    broken = []

    # Use tmp_path for real filesystem interaction in a controlled environment
    root = tmp_path / "project"
    root.mkdir()
    
    dir_a = root / "dir_a"
    dir_a.mkdir()
    file_a = dir_a / "file_a.py"
    file_a.write_text("print('hello')")

    dir_b = root / "dir_b"
    dir_b.mkdir()
    file_b = dir_b / "file_b.py"
    file_b.write_text("print('world')")

    skipped_dir = root / "skipped_dir"
    skipped_dir.mkdir()
    file_skip = skipped_dir / "skip.py"
    file_skip.write_text("skip")

    non_python_file = dir_a / "readme.txt"
    non_python_file.write_text("text")

    # Setup scenario 1: Valid files and traversal
    config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")
    
    paths = [str(root), "non_existent_path"]
    
    # We need to mock is_skipped for the specific directory skip test
    def side_effect_skip(path):
        return "skipped_dir" in str(path)
    config.is_skipped.side_effect = side_effect_skip

    results = list(find([str(root), "non_existent_path"], config, skipped, broken))

    # Assertions
    assert str(file_a) in results
    assert str(file_b) in results
    assert "non_existent_path" in broken
    assert any("skipped_dir" in s for s in skipped)
    assert str(file_skip) not in results # Should be skipped via config

    # Setup scenario 2: Single file path (not a directory)
    results_single = list(find([str(file_a)], config, [], []))
    assert results_single == [str(file_a)]

    # Setup scenario 3: Broken paths
    broken_list = []
    list(find(["invalid_path_123"], config, [], broken_list))
    assert "invalid_path_123" in broken_list

    # Setup scenario 4: Filetype filtering
    config.is_supported_filetype.side_effect = lambda p: not p.endswith(".txt")
    results_filtered = list(find([str(dir_a)], config, [], []))
    assert str(file_a) in results_filtered
    assert str(non_python_file) not in results_filtered
```


# LLM-generated content at query #21
#--------------------------

```python
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock


def test_find(tmp_path):
    # Setup directory structure
    # root/
    #   file1.py
    #   subdir/
    #     file2.py
    #     skipped_dir/
    #       file3.py
    #   ignored_file.txt
    # broken_path_dir/ (non-existent)
    
    root = tmp_path / "root"
    root.mkdir()
    
    file1 = root / "file1.py"
    file1.write_text("print('hello')")
    
    subdir = root / "subdir"
    subdir.mkdir()
    file2 = subdir / "file2.py"
    file2.write_text("print('world')")
    
    skipped_dir = root / "skipped_dir"
    skipped_dir.mkdir()
    file3 = skipped_dir / "file3.py"
    file3.write_text("print('skip me')")
    
    ignored_file = root / "ignored_file.txt"
    ignored_file.write_text("not python")

    broken_path = str(tmp_path / "non_existent_dir")

    # Mock Config
    config = MagicMock()
    # Simulate is_supported_filetype: only .py files are supported
    config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")
    # Simulate is_skipped: skip 'skipped_dir'
    config.is_skipped.side_effect = lambda p: "skipped_dir" in str(p)
    # Mock follow_links as False
    config.follow_links = False

    paths = [str(root), broken_path]
    skipped = []
    broken = []

    results = list(find(paths, config, skipped, broken))

    # Assertions
    # 1. Check found files (should only be file1.py and file2.py)
    assert len(results) == 2
    assert str(file1) in results
    assert str(file2) in results
    assert str(ignored_file) not in results

    # 2. Check skipped items
    # The function adds the directory path to skipped if is_skipped returns True during dir traversal
    assert any("skipped_dir" in s for s in skipped)
    
    # 3. Check broken paths
    assert broken_path in broken

    # 4. Verify file2 logic: check that file3 was never even reached because its parent was skipped
    assert str(file3) not in results
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Setup Mock Config
    config = MagicMock(spec=Config)
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup Paths and tracking lists
    paths = ["/valid/dir", "/valid/file.py", "/non/existent"]
    skipped = []
    broken = []

    # Mocking file system structure
    # /valid/dir contains:
    #   - subdir_to_skip (will be skipped)
    #   - subdir_normal (contains file1.py)
    #   - file2.py (supported)
    #   - file3.txt (unsupported)

    mock_tree = {
        "/valid/dir": {
            "dirnames": ["subdir_to_skip", "subdir_normal"],
            "filenames": ["file2.py", "file3.txt"]
        },
        "/valid/dir/subdir_normal": {
            "dirnames": [],
            "filenames": ["file1.py"]
        }
    }

    def mock_isdir(path):
        return path in mock_tree or path == "/valid/file.py"

    def mock_exists(path):
        return path != "/non/existent"

    def mock_walk(top, topdown, followlinks):
        if top == "/valid/dir":
            yield "/valid/dir", ["subdir_to_skip", "subdir_normal"], ["file2.py", "file3.txt"]
        elif top == "/valid/dir/subdir_normal":
            yield "/valid/dir/subdir_normal", [], ["file1.py"]

    # Configure is_skipped behavior
    def side_effect_is_skipped(path):
        return "subdir_to_skip" in str(path)

    config.is_skipped.side_effect = side_effect_is_skipped
    config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")

    with patch("os.path.isdir", side_effect=mock_isdir), \
         patch("os.path.exists", side_effect=mock_exists), \
         patch("os.walk", side_effect=mock_walk), \
         patch("os.path.abspath", side_effect=lambda x: x), \
         patch("pathlib.Path.resolve", side_effect=lambda: MagicMock()):

        # Execute function
        results = list(find(paths, config, skipped, broken))

        # Assertions
        # Expected files: /valid/file.py (direct file), /valid/dir/file2.py, /valid/dir/subdir_normal/file1.py
        # Note: file3.txt is filtered by is_supported_filetype
        assert "/valid/file.py" in results
        assert "/valid/dir/file2.py" in results
        assert "/valid/dir/subdir_normal/file1.py" in results
        assert len(results) == 3

        # Assert broken paths
        assert "/non/existent" in broken

        # Assert skipped paths (the directory itself was identified as skipped)
        assert any("subdir_to_skip" in s for s in skipped)
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

def test_find():
    # Mock Config object
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup temporary directory structure
    with pytest.raises(SystemError):  # Placeholder to allow scope logic, but we use tmp_path
        pass 

    # Use a real temp directory for os.walk and path logic
    tmp_dir = Path("test_root")
    tmp_dir.mkdir(exist_ok=True)
    subdir = tmp_dir / "subdir"
    subdir.mkdir(exist_ok=True)
    file1 = tmp_dir / "file1.py"
    file1.write_text("print(1)")
    file2 = subdir / "file2.py"
    file2.write_text("print(2)")
    skip_dir = tmp_dir / "skipped_dir"
    skip_dir.mkdir(exist_ok=True)
    skip_file = skip_dir / "skip.py"
    skip_file.write_text("skip")

    try:
        # Case 1: Standard files in paths
        paths = [str(tmp_dir)]
        skipped = []
        broken = []
        
        # Configure mock to skip the specific skip_dir
        def is_skipped_side_effect(path):
            return "skipped_dir" in str(path)
        config.is_skipped.side_effect = is_skipped_side_effect

        results = list(find(paths, config, skipped, broken))

        # Assertions for Case 1
        assert str(file1) in results
        assert str(file2) in results
        assert str(skip_dir) in skipped
        assert str(skip_file) not in results
        assert len(broken) == 0

        # Case 2: Broken paths
        paths_with_broken = [str(tmp_dir), "non_existent_path"]
        skipped_2 = []
        broken_2 = []
        results_2 = list(find(paths_with_broken, config, skipped_2, broken_2))

        assert str(file1) in results_2
        assert "non_existent_path" in broken_2

        # Case 3: Single file path (not a directory)
        paths_single = [str(file1)]
        skipped_3 = []
        broken_3 = []
        results_3 = list(find(paths_single, config, skipped_3, broken_3))
        assert results_3 == [str(file1)]

        # Case 4: Unsupported filetype
        config.is_supported_filetype.side_effect = lambda p: not p.endswith(".txt")
        txt_file = tmp_dir / "test.txt"
        txt_file.write_text("text")
        
        results_4 = list(find([str(tmp_dir)], config, [], []))
        assert str(txt_file) not in results_4

    finally:
        # Cleanup
        import shutil
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
```


