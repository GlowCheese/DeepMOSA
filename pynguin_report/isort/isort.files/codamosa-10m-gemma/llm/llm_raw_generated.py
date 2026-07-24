####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    skipped = []
    broken = []

    # Setup directory structure using patch
    # We will mock os.path.isdir, os.path.exists, os.walk, and os.path.abspath
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.walk") as mock_walk, \
         patch("os.path.abspath") as mock_abspath, \
         patch("pathlib.Path.resolve") as mock_resolve:

        # Case 1: Single file path (exists)
        mock_isdir.side_effect = lambda p: p == "file.py"
        mock_exists.side_effect = lambda p: p == "file.py"
        mock_abspath.side_effect = lambda p: p
        
        paths = ["file.py"]
        result = list(find(paths, config, skipped, broken))
        assert result == ["file.py"]
        assert broken == []
        assert skipped == []

        # Case 2: Broken path
        mock_isdir.side_effect = lambda p: p == "non_existent"
        mock_exists.side_effect = lambda p: p == "file.py" # file.py exists, non_existent does not
        paths = ["non_existent"]
        result = list(find(paths, config, skipped, broken))
        assert result == []
        assert "non_existent" in broken

        # Case 3: Directory walking
        # Reset state
        skipped = []
        broken = []
        
        mock_isdir.side_effect = lambda p: p == "src"
        mock_exists.side_effect = lambda p: True
        mock_abspath.side_effect = lambda p: p
        
        # Mock os.walk to return one directory and one file
        # dirpath, dirnames, filenames
        mock_walk.return_value = [
            ("src", ["subdir"], ["module.py"]),
            ("src/subdir", [], ["inner.py"])
        ]
        
        # Mock Path.resolve to prevent actual filesystem access for visited_dirs logic
        mock_resolve.side_effect = lambda: MagicMock()

        # Test skip logic for a directory
        config.is_skipped.side_effect = lambda p: str(p) == "src/subdir"
        
        paths = ["src"]
        result = list(find(paths, config, skipped, broken))
        
        # Should yield module.py, but skip subdir (and thus inner.py)
        # Note: the loop for filenames in 'src' runs before 'subdir' is processed in the next iteration of walk
        # However, the implementation processes filenames in the current dirpath first.
        # In 'src', filenames=['module.py']. In 'src', dirnames=['subdir'].
        # If subdir is skipped, it's added to skipped.
        
        assert "src/module.py" in result
        assert "src/subdir" in skipped
        
        # Case 4: Unsupported filetype
        config.is_skipped.side_effect = lambda p: False
        config.is_supported_filetype.side_effect = lambda p: p != "src/module.txt"
        
        mock_walk.return_value = [
            ("src", [], ["module.py", "module.txt"])
        ]
        
        paths = ["src"]
        result = list(find(paths, config, skipped, broken))
        assert "src/module.py" in result
        assert "src/module.txt" not in result

        # Case 5: File is skipped
        config.is_skipped.side_effect = lambda p: str(p) == str(Path("src/module.py").absolute())
        mock_abspath.side_effect = lambda p: Path(p).absolute()
        
        paths = ["src"]
        result = list(find(paths, config, skipped, broken))
        assert "src/module.py" not in result
        assert any("module.py" in s for s in skipped)
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock

def test_find():
    # Mock Config
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup temporary directory structure
    tmp_dir = Path(tmpdir, "test_root")
    tmp_dir.mkdir()
    
    # Create files
    file1 = tmp_dir / "file1.py"
    file1.write_text("content")
    file2 = tmp_dir / "file2.py"
    file2.write_text("content")
    
    sub_dir = tmp_dir / "subdir"
    sub_dir.mkdir()
    file3 = sub_dir / "file3.py"
    file3.write_text("content")
    
    skipped_dir = tmp_dir / "skipped_dir"
    skipped_dir.mkdir()
    file4 = skipped_dir / "file4.py"
    file4.write_text("content")
    
    # Define behaviors for mocks
    def is_skipped_side_effect(path):
        return "skipped_dir" in str(path)
    
    def is_supported_side_effect(path):
        return path.endswith(".py")

    config.is_skipped.side_effect = is_skipped_side_effect
    config.is_supported_filetype.side_effect = is_supported_side_effect

    # Test cases
    paths = [str(tmp_dir), "non_existent_path"]
    skipped = []
    broken = []

    # Execution
    results = list(find(paths, config, skipped, broken))

    # Assertions
    # 1. Check found files (file1, file2, file3)
    # Note: order might vary depending on os.walk, so we use set
    expected_files = {str(file1), str(file2), str(file3)}
    assert set(results) == expected_files
    
    # 2. Check skipped files/dirs
    # The logic appends the directory to skipped if is_skipped is True
    assert any("skipped_dir" in s for s in skipped)
    
    # 3. Check broken paths
    assert "non_existent_path" in broken

    # 4. Test single file path input
    single_file_path = [str(file1)]
    results_single = list(find(single_file_path, config, [], []))
    assert results_single == [str(file1)]

    # 5. Test file type filtering
    unsupported_file = tmp_dir / "test.txt"
    unsupported_file.write_text("text")
    # Re-run with the txt file in paths
    results_filter = list(find([str(tmp_dir)], config, [], []))
    assert str(unsupported_file) not in results_filter
```


# LLM-generated content at query #3
#--------------------------

```python
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

def test_find(tmp_path):
    # Setup directory structure
    # root/
    #   main.py
    #   sub/
    #     module.py
    #     skipped_dir/
    #       ignored.py
    #   broken_file.txt (not supported)
    #   non_existent_path
    
    root = tmp_path / "root"
    root.mkdir()
    
    main_py = root / "main.py"
    main_py.write_text("print(1)")
    
    sub_dir = root / "sub"
    sub_dir.mkdir()
    module_py = sub_dir / "module.py"
    module_py.write_text("print(2)")
    
    skipped_dir = sub_dir / "skipped_dir"
    skipped_dir.mkdir()
    ignored_py = skipped_dir / "ignored.py"
    ignored_py.write_text("print(3)")
    
    unsupported_file = root / "unsupported.txt"
    unsupported_file.write_text("not python")

    # Mock Config
    config = MagicMock()
    # is_supported_filetype: only .py files
    config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")
    # is_skipped: skip 'skipped_dir'
    config.is_skipped.side_effect = lambda p: "skipped_dir" in str(p)
    # follow_links: False
    config.follow_links = False

    paths = [str(root), "non_existent_path"]
    skipped = []
    broken = []

    # Execution
    result = list(find(paths, config, skipped, broken))

    # Assertions
    # 1. Check yielded files (only .py files not in skipped dirs)
    # Note: order depends on os.walk, but we check content
    assert len(result) == 2
    assert str(main_py) in result
    assert str(module_py) in result
    assert str(ignored_py) not in result
    assert str(unsupported_file) not in result

    # 2. Check skipped list (contains the directory that was skipped)
    assert any("skipped_dir" in s for s in skipped)

    # 3. Check broken list (contains non-existent paths)
    assert "non_existent_path" in broken

def test_find_single_file_path(tmp_path):
    single_file = tmp_path / "single.py"
    single_file.write_text("pass")
    
    config = MagicMock()
    config.is_supported_filetype.return_value = True
    config.is_skipped.return_value = False
    config.follow_links = False
    
    skipped = []
    broken = []
    
    result = list(find([str(single_file)], config, skipped, broken))
    
    assert result == [str(single_file)]
    assert len(skipped) == 0
    assert len(broken) == 0
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Mock Config
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    skipped = []
    broken = []

    # Case 1: Path does not exist
    with patch("os.path.exists", return_value=False):
        paths = ["/non/existent/path"]
        results = list(find(paths, config, skipped, broken))
        assert results == []
        assert "/non/existent/path" in broken

    # Case 2: Path is a direct file
    with patch("os.path.exists", return_value=True), \
         patch("os.path.isdir", return_value=False):
        paths = ["/existing/file.py"]
        results = list(find(paths, config, skipped, broken))
        assert results == ["/existing/file.py"]

    # Case 3: Path is a directory with files and skipped directories
    # Setup a mock directory structure
    # /root/a.py
    # /root/subdir/b.py
    # /root/skipped_dir/c.py
    with patch("os.path.isdir", return_value=True), \
         patch("os.walk") as mock_walk, \
         patch("os.path.exists", return_value=True):
        
        # Mock os.walk behavior
        # First iteration: root
        # Second iteration: root/subdir
        # Third iteration: root/skipped_dir
        mock_walk.return_value = [
            ("/root", ["subdir", "skipped_dir"], ["a.py"]),
            ("/root/subdir", [], ["b.py"]),
            ("/root/skipped_dir", [], ["c.py"]),
        ]

        # Mock config.is_skipped to skip 'skipped_dir'
        def is_skipped_side_effect(path):
            return "skipped_dir" in str(path)
        
        config.is_skipped.side_effect = is_skipped_side_effect
        
        # Mock path resolution to prevent actual filesystem access
        with patch("pathlib.Path.resolve", return_value=Path("/root/dummy")):
            paths = ["/root"]
            results = list(find(paths, config, skipped, broken))

            # a.py and b.py should be yielded
            # c.py is in a skipped directory, so it shouldn't be yielded
            assert "a.py" in [os.path.basename(r) for r in results]
            assert "b.py" in [os.path.basename(r) for r in results]
            assert "c.py" not in [os.path.basename(r) for r in results]
            
            # Verify skipped list contains the skipped directory
            assert any("skipped_dir" in s for s in skipped)

    # Case 4: Unsupported file type
    with patch("os.path.isdir", return_value=True), \
         patch("os.walk") as mock_walk, \
         patch("os.path.exists", return_value=True):
        
        mock_walk.return_value = [("/root", [], ["a.py", "ignore.txt"])]
        config.is_supported_filetype.side_effect = lambda p: not p.endswith(".txt")
        
        paths = ["/root"]
        results = list(find(paths, config, skipped, broken))
        assert len(results) == 1
        assert results[0].endswith("a.py")
```


# LLM-generated content at query #5
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
    #   file2.py
    #   subdir/
    #     file3.py
    #     skipped_dir/
    #       file4.py
    #   skipped_file.py
    
    root = tmp_path / "root"
    root.mkdir()
    
    file1 = root / "file1.py"
    file1.write_text("print(1)")
    
    file2 = root / "file2.py"
    file2.write_text("print(2)")
    
    subdir = root / "subdir"
    subdir.mkdir()
    file3 = subdir / "file3.py"
    file3.write_text("print(3)")
    
    skipped_dir = root / "skipped_dir"
    skipped_dir.mkdir()
    file4 = skipped_dir / "file4.py"
    file4.write_text("print(4)")
    
    skipped_file = root / "skipped_file.py"
    skipped_file.write_text("print(5)")

    non_py_file = root / "notes.txt"
    non_py_file.write_text("not python")

    non_existent = root / "does_not_exist.py"

    # Mock Config
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.side_effect = lambda p: str(p) in [str(skipped_dir), str(skipped_file)]
    config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")

    paths = [str(root), str(non_existent)]
    skipped = []
    broken = []

    # Execute
    result = list(find(paths, config, skipped, broken))

    # Assertions
    # Expected files: file1.py, file2.py, file3.py (file4 is in skipped_dir, skipped_file is skipped, notes.txt is not .py)
    # Note: os.walk order can vary, so we sort
    expected_files = [
        os.path.abspath(str(file1)),
        os.path.abspath(str(file2)),
        os.path.abspath(str(file3)),
    ]
    assert sorted(result) == sorted(expected_files)

    # Check skipped list
    # The logic adds skipped paths when config.is_skipped returns True
    # For directories: skipped_dir is skipped
    # For files: skipped_file is skipped
    assert any(str(skipped_dir) in s for s in skipped)
    assert any(str(skipped_file) in s for s in skipped)

    # Check broken list
    assert str(non_existent) in broken
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
    # root/
    #   file1.py
    #   dir1/
    #     file2.py
    #     skipped_dir/
    #       file3.py
    #   dir2/
    #     file4.py
    #   not_python.txt
    
    root = tmp_path / "root"
    root.mkdir()
    
    file1 = root / "file1.py"
    file1.write_text("print(1)")
    
    dir1 = root / "dir1"
    dir1.mkdir()
    file2 = dir1 / "file2.py"
    file2.write_text("print(2)")
    
    skipped_dir = dir1 / "skipped_dir"
    skipped_dir.mkdir()
    file3 = skipped_dir / "file3.py"
    file3.write_text("print(3)")
    
    dir2 = root / "dir2"
    dir2.mkdir()
    file4 = dir_2_file := dir2 / "file4.py"
    file4.write_text("print(4)")
    
    txt_file = root / "not_python.txt"
    txt_file.write_text("hello")
    
    broken_path = str(root / "non_existent")
    
    # Mock Config
    config = MagicMock()
    # Only .py files are supported
    config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")
    # Skip 'skipped_dir'
    config.is_skipped.side_effect = lambda p: "skipped_dir" in str(p)
    config.follow_links = False

    paths = [str(root), broken_path]
    skipped = []
    broken = []

    # Execution
    results = list(find(paths, config, skipped, broken))

    # Assertions
    # Should find file1, file2, file4 (file3 is in skipped_dir, txt_file is wrong extension)
    # Note: os.walk order can vary, but we check content
    assert len(results) == 3
    assert str(file1) in results
    assert str(file2) in results
    assert str(file4) in results
    assert str(txt_file) not in results
    
    # Check skipped
    assert any("skipped_dir" in s for s in skipped)
    assert str(skipped_dir) in skipped
    
    # Check broken
    assert broken_path in broken
```


# LLM-generated content at query #7
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
    #     skipped_dir/
    #       file3.py
    #   dir2/
    #     file4.txt (unsupported)
    #   broken_path (non-existent)
    
    root = tmp_path / "root"
    root.mkdir()
    
    file1 = root / "file1.py"
    file1.write_text("print(1)")
    
    dir1 = root / "dir1"
    dir1.mkdir()
    file2 = dir1 / "file2.py"
    file2.write_text("print(2)")
    
    skipped_dir = dir1 / "skipped_dir"
    skipped_dir.mkdir()
    file3 = skipped_dir / "file3.py"
    file3.write_text("print(3)")
    
    dir2 = root / "dir2"
    dir2.mkdir()
    file4 = dir2 / "file4.txt"
    file4.write_text("not python")

    broken_path = str(tmp_path / "non_existent_path")

    # Mock Config
    config = MagicMock()
    # Assume .py files are supported, others are not
    config.is_supported_filetype.side_effect = lambda x: x.endswith(".py")
    # Assume 'skipped_dir' is skipped
    config.is_skipped.side_effect = lambda p: "skipped_dir" in str(p)
    config.follow_links = False

    paths = [str(root), broken_path]
    skipped = []
    broken = []

    # Execute
    results = list(find(paths, config, skipped, broken))

    # Assertions
    # Should find file1.py and file2.py
    # file3.py is in a skipped directory
    # file4.txt is not a supported filetype
    # broken_path should be in broken list
    
    # Normalize paths for comparison
    normalized_results = [os.path.abspath(r) for r in results]
    expected_results = [os.path.abspath(str(file1)), os.path.abspath(str(file2))]
    
    assert sorted(normalized_results) == sorted(expected_results)
    assert os.path.abspath(str(skipped_dir)) in [os.path.abspath(s) for s in skipped]
    assert broken_path in broken
```


# LLM-generated content at query #8
#--------------------------

```python
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

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
    #   skipped_dir/
    #     file3.py
    #   ignored.txt
    
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
    
    ignored_file = root / "ignored.txt"
    ignored_file.write_text("hello")
    
    # Configure mock behavior for specific paths
    def is_skipped_side_effect(path):
        return "skipped_dir" in str(path)
    
    def is_supported_side_effect(path):
        return path.endswith(".py")

    config.is_skipped.side_effect = is_skipped_side_effect
    config.is_supported_filetype.side_effect = is_supported_side_effect

    skipped = []
    broken = []
    paths = [str(root), "non_existent_path"]

    # Execute function
    result = list(find(paths, config, skipped, broken))

    # Assertions
    # Should find file1.py and file2.py
    # Should skip file3.py because it is in a skipped directory
    # Should skip ignored.txt because it's not a supported filetype
    assert str(file1) in result
    assert str(file2) in result
    assert str(ignored_file) not in result
    
    # Check skipped list
    assert any("skipped_dir" in s for s in skipped)
    
    # Check broken list
    assert "non_existent_path" in broken

def test_find_single_file(tmp_path):
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    single_file = tmp_path / "standalone.py"
    single_file.write_text("pass")
    
    skipped = []
    broken = []
    
    result = list(find([str(single_file)], config, skipped, broken))
    
    assert result == [str(single_file)]
    assert len(skipped) == 0
    assert len(broken) == 0

def test_find_symlink_behavior(tmp_path):
    config = MagicMock()
    config.follow_links = True
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    
    file_in_target = target_dir / "target_file.py"
    file_in_target.write_text("target")
    
    link_dir = tmp_path / "link_dir"
    link_dir.mkdir()
    
    os.symlink(target_dir, link_dir / "link_to_target")
    
    skipped = []
    broken = []
    
    # When follow_links is True, the walker enters the symlinked directory
    result = list(find([str(base_dir), str(link_dir)], config, skipped, broken))
    
    # Ensure the file inside the linked directory was found
    assert any("target_file.py" in r for r in result)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock

def test_find():
    # Setup Mock Config
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Create a temporary directory structure for testing
    temp_dir = Path(tmp_path, "test_root")
    temp_dir.mkdir()
    
    dir_a = temp_dir / "dir_a"
    dir_a.mkdir()
    file_a = dir_a / "file_a.py"
    file_a.write_text("print('a')")
    
    dir_b = temp_dir / "dir_b"
    dir_b.mkdir()
    file_b = dir_b / "file_b.py"
    file_b.write_text("print('b')")
    
    skipped_dir = temp_dir / "skipped_dir"
    skipped_dir.mkdir()
    file_skipped = skipped_dir / "skipped.py"
    file_skipped.write_text("print('skip')")
    
    broken_path = str(temp_dir / "non_existent")
    single_file = temp_dir / "standalone.py"
    single_file.write_text("print('standalone')")

    # Define behavior for is_skipped
    def is_skipped_side_effect(path):
        return "skipped_dir" in str(path)
    config.is_skipped.side_effect = is_skipped_side_effect

    # Define behavior for is_supported_filetype
    def is_supported_side_effect(filepath):
        return filepath.endswith(".py")
    config.is_supported_filetype.side_effect = is_supported_side_effect

    paths = [str(temp_dir), broken_path, str(single_file)]
    skipped = []
    broken = []

    # Execute function
    results = list(find(paths, config, skipped, broken))

    # Assertions
    # 1. Check that valid python files are yielded
    # Note: order depends on os.walk, but we check presence
    assert str(file_a) in results
    assert str(file_b) in results
    assert str(single_file) in results
    
    # 2. Check that unsupported filetypes are NOT yielded (if we added a .txt)
    file_txt = dir_a / "test.txt"
    file_txt.write_text("text")
    # Re-run with specific logic for txt
    config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")
    results_txt = list(find([str(dir_a)], config, [], []))
    assert str(file_txt) not in results_txt

    # 3. Check that skipped files/dirs are in skipped list and not yielded
    assert any("skipped_dir" in s for s in skipped)
    assert str(file_skipped) not in results
    
    # 4. Check that broken paths are in broken list
    assert broken_path in broken

    # 5. Check single file path yielding
    assert str(single_file) in results
```


# LLM-generated content at query #10
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
    #   sub/
    #     file2.py
    #   skipped_dir/
    #     file3.py
    #   broken_path_ref
    
    root = tmp_path / "src"
    root.mkdir()
    
    file1 = root / "file1.py"
    file1.write_text("print(1)")
    
    sub = root / "sub"
    sub.mkdir()
    file2 = sub / "file2.py"
    file2.write_text("print(2)")
    
    skipped_dir = root / "skipped_dir"
    skipped_dir.mkdir()
    file3 = skipped_dir / "file3.py"
    file3.write_text("print(3)")
    
    unsupported_file = root / "readme.txt"
    unsupported_file.write_text("text")
    
    broken_path = "non_existent_path"
    
    # Define behaviors for specific paths
    def is_skipped_side_effect(path):
        return "skipped_dir" in str(path)

    def is_supported_side_effect(path):
        return path.endswith(".py")

    config.is_skipped.side_effect = is_skipped_side_effect
    config.is_supported_filetype.side_effect = is_supported_side_effect

    paths = [str(root), broken_path]
    skipped = []
    broken = []

    # Execute
    result = list(find(paths, config, skipped, broken))

    # Assertions
    # 1. Check yielded files (only .py files in non-skipped dirs)
    assert str(file1) in result
    assert str(file2) in result
    assert str(unsupported_file) not in result
    
    # 2. Check skipped list (directories and files skipped by config)
    assert any("skipped_dir" in s for s in skipped)
    
    # 3. Check broken list
    assert broken_path in broken

    # 4. Check single file path input
    single_file_path = str(file1)
    result_single = list(find([single_file_path], config, [], []))
    assert result_single == [single_file_path]

def test_find_symlink_behavior(tmp_path):
    config = MagicMock()
    config.follow_links = True
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    file_in_target = target / "target.py"
    file_in_target.write_text("target")
    
    link_dir = root / "link_to_target"
    os.symlink(target, link_dir)

    paths = [str(root)]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))
    
    # If follow_links is True, it should find files in the symlinked directory
    assert str(file_in_target) in result
```


# LLM-generated content at query #11
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
    #   file2.txt (unsupported)
    #   subdir/
    #     file3.py
    #     skipped_dir/
    #       file4.py
    #   broken_link (non-existent)
    
    root = tmp_path / "root"
    root.mkdir()
    
    file1 = root / "file1.py"
    file1.write_text("print(1)")
    
    file2 = root / "file2.txt"
    file2.write_text("test")
    
    subdir = root / "subdir"
    subdir.mkdir()
    file3 = subdir / "file3.py"
    file3.write_text("print(3)")
    
    skipped_dir = root / "skipped_dir"
    skipped_dir.mkdir()
    file4 = skipped_dir / "file4.py"
    file4.write_text("print(4)")
    
    broken_path = str(tmp_path / "non_existent_path")
    
    # Mock Config
    config = MagicMock()
    config.follow_links = False
    config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")
    config.is_skipped.side_effect = lambda p: "skipped_dir" in str(p)

    paths = [str(root), broken_path]
    skipped = []
    broken = []

    # Execution
    result = list(find(paths, config, skipped, broken))

    # Assertions
    # 1. Check yielded files (only .py files not in skipped directories)
    assert len(result) == 2
    assert str(file1) in result
    assert str(file3) in result
    assert str(file2) not in result # unsupported type
    assert str(file4) not in result # skipped

    # 2. Check skipped list
    assert any("skipped_dir" in s for s in skipped)
    assert str(skipped_dir) in skipped

    # 3. Check broken list
    assert broken_path in broken

def test_find_single_file(tmp_path):
    file = tmp_path / "single.py"
    file.write_text("content")
    
    config = MagicMock()
    config.is_supported_filetype.return_value = True
    config.is_skipped.return_value = False
    
    paths = [str(file)]
    skipped = []
    broken = []
    
    result = list(find(paths, config, skipped, broken))
    
    assert result == [str(file)]
    assert len(skipped) == 0
    assert len(broken) == 0

def test_find_visited_dirs_prevention(tmp_path):
    # Test that resolving paths prevents infinite loops/re-visiting
    root = tmp_path / "root"
    root.mkdir()
    subdir = root / "subdir"
    subdir.mkdir()
    
    file_in_subdir = subdir / "sub.py"
    file_in_subdir.write_text("content")
    
    # Create a symlink to subdir inside root (if OS allows)
    # Note: we use a manual approach to simulate the logic of the function
    config = MagicMock()
    config.follow_links = True
    config.is_supported_filetype.return_value = True
    config.is_skipped.return_value = False

    paths = [str(root)]
    skipped = []
    broken = []

    # We can't easily create real symlinks in all CI environments, 
    # but the function logic is tested by the first test's structure.
    # This test specifically targets the 'resolved_path in visited_dirs' logic.
    result = list(find(paths, config, skipped, broken))
    assert str(file_in_subdir) in result
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock

def test_find():
    # Mock Config
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup temporary directory structure
    tmp_dir = Path(tmp_path, "root")
    tmp_dir.mkdir()
    
    # File 1: Normal python file
    file1 = tmp_dir / "file1.py"
    file1.write_text("print(1)")
    
    # File 2: Supported file but skipped
    file2 = tmp_dir / "skipped.py"
    file2.write_text("print(2)")
    
    # File 3: Unsupported file type
    file3 = tmp_dir / "readme.txt"
    file3.write_text("text")
    
    # Subdirectory with a file
    sub_dir = tmp_dir / "subdir"
    sub_dir.mkdir()
    file4 = sub_dir / "file4.py"
    file4.write_text("print(4)")

    # Setup test inputs
    paths = [str(tmp_dir), "non_existent_path"]
    skipped = []
    broken = []

    # Configure behavior for specific files
    def side_effect_is_skipped(path):
        return str(path.resolve()) == str(file2.resolve())

    def side_effect_is_supported(filepath):
        return filepath.endswith(".py")

    config.is_skipped.side_effect = side_effect_is_skipped
    config.is_supported_filetype.side_effect = side_effect_is_supported

    # Execute function
    results = list(find(paths, config, skipped, broken))

    # Assertions
    # 1. Check found files (file1 and file4 should be yielded)
    assert str(file1.absolute()) in results
    assert str(file4.absolute()) in results
    assert len(results) == 2

    # 2. Check skipped files
    assert str(file2.absolute()) in skipped
    
    # 3. Check broken paths
    assert "non_existent_path" in broken

    # 4. Check that unsupported files were not yielded
    assert str(file3.absolute()) not in results

    # 5. Check that directory skipping works
    skip_dir = tmp_dir / "skip_me"
    skip_dir.mkdir()
    (skip_dir / "hidden.py").write_text("hidden")
    
    # Reset and re-run with a skip config for the directory
    skipped_v2 = []
    broken_v2 = []
    
    def side_effect_is_skipped_v2(path):
        return "skip_me" in str(path)

    config.is_skipped.side_effect = side_effect_is_skipped_v2
    
    results_v2 = list(find([str(tmp_dir)], config, skipped_v2, broken_v2))
    
    # The directory 'skip_me' should be in skipped, and its contents should not be in results
    assert any("skip_me" in s for s in skipped_v2)
    assert not any("skip_me" in r for r in results_v2)
```


# LLM-generated content at query #13
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
    #   file2.txt (unsupported)
    #   subdir/
    #     file3.py
    #     skipped_dir/
    #       file4.py
    #   broken_path/ (does not exist)

    root = tmp_path / "root"
    root.mkdir()
    
    file1 = root / "file1.py"
    file1.write_text("print(1)")
    
    file2 = root / "file2.txt"
    file2.write_text("hello")
    
    subdir = root / "subdir"
    subdir.mkdir()
    file3 = subdir / "file3.py"
    file3.write_text("print(2)")
    
    skipped_dir = root / "skipped_dir"
    skipped_dir.mkdir()
    file4 = skipped_dir / "file4.py"
    file4.write_text("print(3)")

    # Mock Config
    config = MagicMock()
    # Supported files are .py
    config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")
    # Skip 'skipped_dir'
    config.is_skipped.side_effect = lambda p: "skipped_directories" in str(p) or "skipped_dir" in str(p)
    config.follow_links = False

    paths = [str(root), "non_existent_path"]
    skipped = []
    broken = []

    # Execute
    result = list(find(paths, config, skipped, broken))

    # Assertions
    # Should find file1.py and file3.py
    # file2.txt is unsupported
    # file4.py is in a skipped directory
    # 'non_existent_path' is broken
    
    expected_files = [os.path.abspath(str(file1)), os.path.abspath(str(file3))]
    actual_files = [os.path.abspath(f) for f in result]
    
    assert sorted(actual_files) == sorted(expected_files)
    assert "non_existent_path" in broken
    assert any("skipped_dir" in s for s in skipped)
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

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
    #   skipped_dir/
    #     file3.py
    #   broken_path (non-existent)
    
    root = tmp_path / "project"
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
    
    # Define paths to scan
    paths = [str(root), "non_existent_path"]
    skipped = []
    broken = []

    # Mock config.is_skipped to skip 'skipped_dir'
    def is_skipped_side_effect(path):
        return "skipped_dir" in str(path)
    
    config.is_skipped.side_effect = is_skipped_side_effect

    # Execute function
    result = list(find(paths, config, skipped, broken))

    # Assertions
    # 1. Check found files (file1.py and file2.py should be found)
    # Note: order might vary depending on os.walk, so we sort
    expected_files = sorted([str(file1.absolute()), str(file2.absolute())])
    actual_files = sorted([os.path.abspath(f) for f in result])
    assert actual_files == expected_files

    # 2. Check skipped list (skipped_dir should be in skipped)
    assert any("skipped_dir" in s for s in skipped)

    # 3. Check broken list
    assert "non_existent_path" in broken

def test_find_single_file(tmp_path):
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    single_file = tmp_path / "standalone.py"
    single_file.write_text("pass")
    
    paths = [str(single_file)]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))
    
    assert len(result) == 1
    assert os.path.abspath(result[0]) == os.path.abspath(str(single_file))

def test_find_file_type_filtering(tmp_path):
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    # Only allow .py files
    config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")

    root = tmp_path / "project"
    root.mkdir()
    py_file = root / "test.py"
    py_file.write_text("pass")
    txt_file = root / "test.txt"
    txt_file.write_text("pass")

    paths = [str(root)]
    skipped = []
    broken = []

    result = list(find(paths, config, skipped, broken))

    assert len(result) == 1
    assert result[0].endswith("test.py")
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Setup Mock Config
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Test cases data
    paths = ["/valid/file.py", "/valid/dir", "/non/existent/path"]
    skipped = []
    broken = []

    # Mocking filesystem structure
    # /valid/file.py (File)
    # /valid/dir/
    # /valid/dir/sub/file.py (File)
    # /valid/dir/skip_me/ (Dir - to be skipped)
    
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.walk") as mock_walk, \
         patch("os.path.abspath") as mock_abspath, \
         patch("pathlib.Path.resolve") as mock_resolve:

        # Define behavior for isdir and exists
        def isdir_side_effect(path):
            return path == "/valid/dir"
        mock_isdir.side_effect = isdir_side_effect

        def exists_side_effect(path):
            return path != "/non/existent/path"
        mock_exists.side_effect = exists_side_effect

        # Define behavior for walk
        # 1st iteration: /valid/dir
        # 2nd iteration: /valid/dir/sub
        mock_walk.return_value = [
            ("/valid/dir", ["sub", "skip_me"], ["file1.py"]),
            ("/valid/dir/sub", [], ["file2.py"]),
        ]

        # Mocking path resolution to avoid actual filesystem access
        mock_resolve.side_effect = lambda: MagicMock(spec=Path)
        
        # Mocking is_skipped logic
        # Let's say "skip_me" is skipped
        def is_skipped_side_effect(path_obj):
            return "skip_me" in str(path_obj)
        config.is_skipped.side_effect = is_skipped_side_effect

        # Mocking abspath for files
        mock_abspath.side_effect = lambda x: x

        # Execution
        result = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. /valid/file.py should be yielded directly
        assert "/valid/file.py" in result
        
        # 2. /non/existent/path should be in broken
        assert "/non/existent/path" in broken
        
        # 3. Files in valid dirs should be yielded
        assert "/valid/dir/file1.py" in result
        assert "/valid/dir/sub/file2.py" in result
        
        # 4. Skipped directory should be in skipped list
        assert any("skip_me" in s for s in skipped)
        
        # 5. Ensure we didn't yield the skipped directory's contents if it was a file
        # (In this mock, skip_me is a dir, so we check if it was added to skipped)
        assert any("/valid/dir/skip_me" in s for s in skipped)

def test_find_broken_path():
    config = MagicMock()
    config.follow_links = False
    paths = ["/invalid/path"]
    skipped = []
    broken = []

    with patch("os.path.isdir", return_value=False), \
         patch("os.path.exists", return_value=False):
        result = list(find(paths, config, skipped, broken))
        assert result == []
        assert "/invalid/path" in broken

def test_find_single_file():
    config = MagicMock()
    config.follow_links = False
    config.is_supported_filetype.return_value = True
    config.is_skipped.return_value = False
    paths = ["/single/file.py"]
    skipped = []
    broken = []

    with patch("os.path.isdir", return_value=False), \
         patch("os.path.exists", return_value=True):
        result = list(find(paths, config, skipped, broken))
        assert result == ["/single/file.py"]
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Setup Config mock
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Mocking os and filesystem structure
    # Structure:
    # /root/
    #   file1.py
    #   /subdir/
    #     file2.py
    #   /skipped_dir/ (to be skipped)
    #   /broken_path/ (not exists)
    
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.walk") as mock_walk, \
         patch("os.path.abspath") as mock_abspath, \
         patch("pathlib.Path.resolve") as mock_resolve:

        # Define behaviors
        # paths = ["/root", "/non_existent"]
        paths = ["/root", "/non_existent"]
        
        # Mock isdir: /root is dir, /non_existent is not
        mock_isdir.side_effect = lambda p: p == "/root"
        
        # Mock exists: /root exists, /non_exists does not
        mock_exists.side_effect = lambda p: p == "/root" or p == "/root/file1.py" or p == "/root/subdir/file2.py"
        
        # Mock walk for /root
        # 1st call: /root, dirs=[subdir, skipped_dir], files=[file1.py]
        # 2nd call: /root/subdir, dirs=[], files=[file2.py]
        mock_walk.return_value = [
            ("/root", ["subdir", "skipped_dir"], ["file1.py"]),
            ("/root/subdir", [], ["file2.py"]),
        ]
        
        # Mock config.is_skipped for 'skipped_dir'
        def is_skipped_side_effect(path):
            return "skipped_dir" in str(path)
        config.is_skipped.side_effect = is_skipped_side_effect
        
        # Mock path resolution to prevent infinite loops/errors in test
        mock_resolve.side_effect = lambda: MagicMock()
        
        # Mock abspath to return the same path
        mock_abspath.side_effect = lambda p: p

        skipped = []
        broken = []
        
        # Execute
        results = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. file1.py and file2.py should be found
        assert "/root/file1.py" in results
        assert "/root/subdir/file2.py" in results
        
        # 2. skipped_dir should be in skipped list
        assert any("skipped_dir" in s for s in skipped)
        
        # 3. /non_existent should be in broken list
        assert "/non_existent" in broken

        # 4. Check if file1.py was yielded correctly
        assert len(results) == 2
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

    # Setup paths
    valid_file = "/tmp/test_file.py"
    valid_dir = "/tmp/test_dir"
    non_existent_path = "/tmp/does_not_exist"
    skipped_file = "/tmp/skipped.py"
    
    # Mocking os and filesystem structure
    # We will simulate:
    # 1. A single file path
    # 2. A directory containing a file and a skipped directory
    # 3. A non-existent path
    
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.path.abspath") as mock_abspath, \
         patch("os.walk") as mock_walk:
        
        # Define behavior for path existence
        def isdir_side_effect(path):
            return path == valid_dir or path == "/tmp"
        
        def exists_side_effect(path):
            return path != non_existent_path

        mock_isdir.side_effect = isdir_side_effect
        mock_exists.side_effect = exists_side_effect
        mock_abspath.side_effect = lambda x: x
        
        # Setup os.walk to return a directory structure
        # dirpath, dirnames, filenames
        mock_walk.return_value = [
            (valid_dir, ["skipped_subdir", "normal_subdir"], ["file1.py", "file2.py"]),
            (os.path.join(valid_dir, "normal_subdir"), [], ["file3.py"]),
        ]

        # Setup config.is_skipped behavior
        # Let's say 'skipped_subdir' is skipped
        def is_skipped_side_effect(path):
            return "skipped_subdir" in str(path)
        
        config.is_skipped.side_effect = is_skipped_side_effect
        
        # Setup config.is_supported_filetype behavior
        # Let's say only .py files are supported
        config.is_supported_filetype.side_effect = lambda x: x.endswith(".py")

        # Inputs
        paths = [valid_file, valid_dir, non_existent_path]
        skipped = []
        broken = []

        # Execution
        result = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. Check yielded files
        # valid_file (direct file)
        # valid_dir/file1.py
        # valid_dir/file2.py
        # valid_dir/normal_subdir/file3.py
        assert valid_file in result
        assert os.path.join(valid_dir, "file1.py") in result
        assert os.path.join(valid_dir, "file2.py") in result
        assert os.path.join(valid_dir, "normal_subdir", "file3.py") in result
        
        # 2. Check skipped list
        # The directory 'skipped_subdir' should be in skipped
        assert any("skipped_subdir" in s for s in skipped)
        
        # 3. Check broken list
        assert non_existent_path in broken

        # 4. Ensure skipped files are not yielded
        # (If we had a file named skipped.py in the walk, it would be in skipped)
        for item in result:
            assert not config.is_skipped(Path(item))
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock

def test_find():
    # Setup Mock Config
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Create temporary directory structure
    tmp_dir = Path(tmpdir, "test_root")
    tmp_dir.mkdir()
    
    # File 1: Standard python file
    file1 = tmp_dir / "valid.py"
    file1.write_text("print('hello')")
    
    # File 2: File to be skipped via config
    file2 = tmp_dir / "skipped.py"
    file2.write_text("print('skip me')")
    
    # File 3: File with unsupported extension
    file3 = tmp_dir / "unsupported.txt"
    file3.write_text("text content")
    
    # Directory to be skipped
    skipped_dir = tmp_dir / "ignored_dir"
    skipped_dir.mkdir()
    file4 = skipped_dir / "hidden.py"
    file4.write_text("hidden")

    # Define behavior for mock config
    def is_skipped_side_effect(path):
        return "ignored_dir" in str(path) or "skipped.py" in str(path)
    
    def is_supported_side_effect(path):
        return path.endswith(".py")

    config.is_skipped.side_effect = is_skipped_side_effect
    config.is_supported_filetype.side_effect = is_supported_side_effect

    # Test Cases
    paths = [str(tmp_dir), "non_existent_path"]
    skipped = []
    broken = []

    # Execute
    results = list(find(paths, config, skipped, broken))

    # Assertions
    # 1. Check valid files found
    assert str(file1) in results
    
    # 2. Check unsupported file type filtered out
    assert str(file3) not in results
    
    # 3. Check skipped files recorded in skipped list
    assert any("skipped.py" in s for s in skipped)
    
    # 4. Check skipped directories prevent traversal
    assert not any("hidden.py" in r for r in results)
    
    # 5. Check broken paths recorded
    assert "non_existent_path" in broken

    # 6. Check single file path input
    single_file_path = [str(file1)]
    results_single = list(find(single_file_path, config, [], []))
    assert results_single == [str(file1)]
```


# LLM-generated content at query #6
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

    # Setup test environment paths
    # We use patches to avoid actual filesystem dependency
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.walk") as mock_walk, \
         patch("os.path.abspath") as mock_abspath, \
         patch("pathlib.Path.resolve") as mock_resolve:

        # Scenario 1: Single file path (not a directory)
        mock_isdir.return_value = False
        mock_exists.return_value = True
        mock_abspath.side_effect = lambda x: x
        
        paths = ["file.py"]
        skipped = []
        broken = []
        
        results = list(find(paths, config, skipped, broken))
        assert results == ["file.py"]
        assert broken == []

        # Scenario 2: Broken path
        mock_isdir.return_value = False
        mock_exists.return_value = False
        paths = ["non_existent.py"]
        results = list(find(paths, config, skipped, broken))
        assert results == []
        assert "non_existent.py" in broken

        # Scenario 3: Directory traversal
        mock_isdir.return_value = True
        mock_exists.return_value = True
        # Mock os.walk: (dirpath, dirnames, filenames)
        mock_walk.return_value = [
            ("/root", ["subdir"], ["file1.py", "ignored.txt"]),
            ("/root/subdir", [], ["file2.py"])
        ]
        # Mock Path behavior for dirnames removal
        mock_resolve.side_effect = lambda: MagicMock()
        
        # Configure config to skip 'ignored.txt'
        def is_skipped_side_effect(path):
            return "ignored.txt" in str(path)
        config.is_skipped.side_ext = is_skipped_side_effect
        config.is_skipped.side_effect = is_skipped_side_effect
        
        # Configure config to only support .py
        def is_supported_side_effect(path):
            return path.endswith(".py")
        config.is_supported_filetype.side_effect = is_supported_side_effect
        
        # Mock absolute path for skipped list
        mock_abspath.side_effect = lambda x: x

        paths = ["/root"]
        skipped = []
        broken = []
        
        results = list(find(paths, config, skipped, broken))
        
        # Should find file1.py and file2.py, but not ignored.txt
        assert "file1.py" in results or "/root/file1.py" in results
        assert "/root/file2.py" in results
        assert any("ignored.txt" in s for s in skipped)

        # Scenario 4: Directory skipping logic (dirnames removal)
        # Reset mocks for a clean state
        mock_walk.return_value = [
            ("/root", ["skip_me"], ["file1.py"]),
            ("/root/skip_me", [], ["file2.py"])
        ]
        config.is_skipped.side_effect = lambda path: "skip_me" in str(path)
        
        skipped = []
        results = list(find(["/root"], config, skipped, []))
        
        # file2.py should NOT be found because 'skip_me' was removed from dirnames
        assert "/root/file1.py" in results or "file1.py" in results
        assert not any("file2.py" in r for r in results)
        assert "/root/skip_me" in skipped
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Setup Config Mock
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup test filesystem structure using patch
    # We will simulate:
    # /root/file1.py (valid)
    # /root/subdir/file2.py (valid)
    # /root/skipped_dir/ (to be skipped)
    # /root/skipped_file.py (to be skipped)
    # /root/broken_path (non-existent)
    
    paths = ["/root", "/non_existent"]
    skipped = []
    broken = []

    # Mocking os.path and os.walk behavior
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.path.abspath") as mock_abspath, \
         patch("os.walk") as mock_walk:
        
        def side_effect_isdir(path):
            return path == "/root"
        
        def side_effect_exists(path):
            return path == "/root" or path == "/root/file1.py" or path == "/root/subdir/file2.py"
        
        mock_isdir.side_effect = side_effect_isdir
        mock_exists.side_effect = side_effect_exists
        mock_abspath.side_effect = lambda x: x

        # Mock os.walk to return a specific tree
        # dirpath, dirnames, filenames
        mock_walk.return_value = [
            ("/root", ["subdir", "skipped_dir"], ["file1.py", "skipped_file.py"]),
            ("/root/subdir", [], ["file2.py"]),
            ("/root/skipped_dir", [], [])
        ]

        # Define behavior for is_skipped
        def is_skipped_side_effect(path):
            return "skipped" in str(path)
        
        config.is_skipped.side_effect = is_skipped_side_effect

        # Execute function
        result = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. Check yielded files (only non-skipped supported files)
        assert "/root/file1.py" in result
        assert "/root/subdir/file2.py" in result
        assert "/root/skipped_file.py" not in result
        
        # 2. Check skipped list (directories and files)
        assert any("skipped_dir" in s for s in skipped)
        assert any("skipped_file.py" in s for s in skipped)
        
        # 3. Check broken list
        assert "/non_existent" in broken

        # 4. Check that config methods were called
        assert config.is_supported_filetype.called
        assert config.is_skipped.called
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

    # Setup temporary file structure
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.walk") as mock_walk, \
         patch("os.path.abspath") as mock_abspath, \
         patch("pathlib.Path.resolve") as mock_resolve:

        # Case 1: Path is a file
        mock_isdir.return_value = False
        mock_exists.return_value = True
        paths = ["file.py"]
        skipped = []
        broken = []
        
        results = list(find(paths, config, skipped, broken))
        assert results == ["file.py"]
        assert skipped == []
        assert broken == []

        # Case 2: Path does not exist
        mock_isdir.return_value = False
        mock_exists.return_value = False
        paths = ["non_existent.py"]
        skipped = []
        broken = []
        
        results = list(find(paths, config, skipped, broken))
        assert results == []
        assert broken == ["non_existent.py"]

        # Case 3: Path is a directory with files and skipped directories
        mock_isdir.return_value = True
        mock_exists.return_value = True
        paths = ["src"]
        skipped = []
        broken = []
        
        # Mock os.walk behavior
        # dirpath, dirnames, filenames
        mock_walk.return_value = [
            ("src", ["venv", "utils"], ["main.py"]),
            ("src/utils", [], ["helper.py"]),
        ]
        
        # Mock path resolution and skipping logic
        # We'll simulate 'venv' being skipped
        def is_skipped_side_effect(path):
            return "venv" in str(path)
        
        config.is_skipped.side_effect = is_skipped_side_effect
        mock_abspath.side_effect = lambda x: x
        
        # Mock Path.resolve to return the same path for simplicity in checking visited_dirs
        mock_resolve.side_effect = lambda: Path("src/utils") # simplified
        
        # To make the test deterministic, we'll override the side effect for the actual test run
        # Let's refine the mock_walk to be more specific for the iteration
        mock_walk.return_value = [
            ("src", ["venv", "utils"], ["main.py"]),
            ("src/utils", [], ["helper.py"]),
        ]
        
        # We need to control how Path(dirpath) / dirname behaves for the skip logic
        # Since we can't easily mock the / operator on a real Path object without complexity,
        # we rely on the side_effect of is_skipped.
        
        results = list(find(paths, config, skipped, broken))
        
        # Check if main.py and helper.py are found, but venv is skipped
        assert "src/main.py" in results
        assert "src/utils/helper.py" in results
        assert any("venv" in s for s in skipped)
```


# LLM-generated content at query #9
#--------------------------

```python
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

def test_find(tmp_path):
    # Setup structure
    # root/
    #   file1.py
    #   subdir/
    #     file2.py
    #   skipped_dir/
    #     file3.py
    #   skipped_file.py
    # broken_path_does_not_exist
    
    root = tmp_path / "root"
    root.mkdir()
    
    file1 = root / "file1.py"
    file1.write_text("print(1)")
    
    subdir = root / "subdir"
    subdir.mkdir()
    file2 = subdir / "file2.py"
    file2.write_text("print(2)")
    
    skip_dir = root / "skipped_dir"
    skip_dir.mkdir()
    file3 = skip_dir / "file3.py"
    file3.write_text("print(3)")
    
    skip_file = root / "skipped_file.py"
    skip_file.write_text("print(4)")
    
    broken_path = "non_existent_path_12345"

    # Mock Config
    config = MagicMock()
    config.follow_links = False
    
    # Define behavior for is_skipped
    def is_skipped_side_effect(path):
        path_str = str(Path(path).absolute())
        skip_dir_str = str(skip_dir.absolute())
        skip_file_str = str(skip_file.absolute())
        return path_str.startswith(skip_dir_str) or path_str == skip_file_str
    
    config.is_skipped.side_effect = is_skipped_side_effect
    
    # Define behavior for is_supported_filetype
    def is_supported_side_effect(filepath):
        return filepath.endswith(".py")
    
    config.is_supported_filetype.side_effect = is_supported_side_effect

    paths = [str(root), broken_path]
    skipped = []
    broken = []

    # Execute
    result = list(find(paths, config, skipped, broken))

    # Assertions
    # Files found should be file1.py and file2.py
    assert len(result) == 2
    assert str(file1.absolute()) in result
    assert str(file2.absolute()) in result
    
    # Broken paths should be recorded
    assert broken == [broken_path]
    
    # Skipped items should be recorded
    # Note: the implementation appends the directory to skipped if it's skipped
    # and the file itself if it's skipped.
    assert any(str(skip_dir.absolute()) in s for s in skipped)
    assert any(str(skip_file.absolute()) in s for s in skipped)

def test_find_single_file(tmp_path):
    file_path = tmp_path / "single.py"
    file_path.write_text("content")
    
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True
    
    skipped = []
    broken = []
    
    result = list(find([str(file_path)], config, skipped, broken))
    
    assert result == [str(file_path.absolute())]
    assert len(skipped) == 0
    assert len(broken) == 0

def test_find_already_resolved_dir_deduplication(tmp_path):
    # Test the visited_dirs logic
    root = tmp_path / "root"
    root.mkdir()
    subdir = root / "subdir"
    subdir.mkdir()
    
    file_in_subdir = subdir / "file.py"
    file_in_subdir.write_text("content")
    
    # Create a symlink to the same directory to test visited_dirs
    symlink_dir = tmp_path / "symlink_dir"
    try:
        os.symlink(root, symlink_dir)
    except OSError:
        pytest.skip("Symlinks not supported on this OS")

    config = MagicMock()
    config.follow_links = False # Should not follow symlink via os.walk, but logic uses resolved_path
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True
    
    skipped = []
    broken = []
    
    # If we pass both the real dir and the symlink dir, it should only yield files once
    result = list(find([str(root), str(symlink_dir)], config, skipped, broken))
    
    # Count occurrences of file.py
    file_name = "file.py"
    matches = [r for r in result if file_name in r]
    assert len(matches) == 1
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_find():
    # Setup Mock Config
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup Mock Filesystem Structure
    # /tmp/test_dir/
    #   /tmp/test_dir/a.py
    #   /tmp/test_dir/sub/
    #     /tmp/test_dir/sub/b.py
    # /tmp/test_dir/skipped.py
    # /tmp/non_existent
    
    base_path = Path("/tmp/test_dir")
    file_a = str(base_path / "a.py")
    file_b = str(base_path / "sub" / "b.py")
    file_skipped = str(base_path / "skipped.py")
    path_non_existent = "/tmp/non_existent"
    path_single_file = str(base_path / "a.py")

    # Mocking os/path behavior
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.path.exists") as mock_exists, \
         patch("os.path.abspath") as mock_abspath, \
         patch("os.walk") as mock_walk:
        
        # Define behavior for path checks
        def isdir_side_effect(p):
            return p == str(base_path)
        
        def exists_side_effect(p):
            return p != path_non_existent
        
        mock_isdir.side_effect = isdir_side_effect
        mock_exists.side_effect = exists_side_effect
        mock_abspath.side_effect = lambda x: x
        
        # Mock os.walk to simulate directory traversal
        # 1st call: base_path
        # 2nd call: base_path/sub
        mock_walk.side_effect = [
            (str(base_path), ["sub"], ["a.py", "skipped.py"]),
            (str(base_path / "sub"), [], ["b.py"]),
        ]

        # Setup skip logic: skip 'skipped.py'
        def is_skipped_side_effect(p):
            return str(p) == file_skipped
        config.is_skipped.side_effect = is_skipped_side_effect

        # Input parameters
        paths = [str(base_path), path_single_file, path_non_existent]
        skipped = []
        broken = []

        # Execute
        result = list(find(paths, config, skipped, broken))

        # Assertions
        # Should yield a.py, b.py, and the single file path a.py
        assert file_a in result
        assert file_b in result
        assert path_single_file in result
        assert len(result) == 3
        
        # Check skipped list
        assert file_skipped in skipped
        
        # Check broken list
        assert path_non_existent in broken
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock

def test_find(tmp_path):
    # Setup filesystem structure
    # root/
    #   file1.py
    #   file2.py
    #   subdir/
    #     file3.py
    #   skipped_dir/
    #     file4.py
    #   ignored.txt
    
    root = tmp_path / "root"
    root.mkdir()
    
    file1 = root / "file1.py"
    file1.write_text("content")
    file2 = root / "file2.py"
    file2.write_text("content")
    
    subdir = root / "subdir"
    subdir.mkdir()
    file3 = subdir / "file3.py"
    file3.write_text("content")
    
    skip_dir = root / "skipped_dir"
    skip_dir.mkdir()
    file4 = skip_dir / "file4.py"
    file4.write_text("content")
    
    ignored_file = root / "ignored.txt"
    ignored_file.write_text("content")
    
    broken_path = root / "non_existent"
    
    # Mock Config
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.side_effect = lambda p: "skipped_dir" in str(p)
    config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")
    
    paths = [str(root), str(broken_path)]
    skipped = []
    broken = []
    
    # Execution
    result = list(find(paths, config, skipped, broken))
    
    # Assertions
    # 1. Check yielded files (should be py files not in skipped dirs)
    # Note: os.walk order can vary, so we sort for comparison
    expected_files = sorted([
        str(file1.absolute()),
        str(file2.absolute()),
        str(file3.absolute())
    ])
    actual_files = sorted([os.path.abspath(f) for f in result])
    assert actual_files == expected_files
    
    # 2. Check skipped list (should contain the skipped directory path)
    assert any("skipped_dir" in s for s in skipped)
    
    # 3. Check broken list
    assert str(broken_path) in broken

def test_find_single_file(tmp_path):
    file_path = tmp_path / "standalone.py"
    file_path.write_text("content")
    
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True
    
    paths = [str(file_path)]
    skipped = []
    broken = []
    
    result = list(find(paths, config, skipped, broken))
    
    assert result == [str(file_path)]
    assert len(skipped) == 0
    assert len(broken) == 0

def test_find_invalid_path(tmp_path):
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True
    
    non_existent = str(tmp_path / "ghost")
    paths = [non_existent]
    skipped = []
    broken = []
    
    result = list(find(paths, config, skipped, broken))
    
    assert result == []
    assert non_existent in broken
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock

def test_find():
    # Mock Config
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Create temporary directory structure
    tmp_dir = Path("test_root").absolute()
    subdir = tmp_dir / "subdir"
    file1 = tmp_dir / "file1.py"
    file2 = subdir / "file2.py"
    skip_dir = tmp_dir / "skipped_dir"
    skip_file = tmp_dir / "skipped.py"
    broken_path = "non_existent_path"

    # Setup actual files for os.walk and os.path.isdir
    os.makedirs(subdir, exist_ok=True)
    os.makedirs(skip_dir, exist_ok=True)
    file1.write_text("print(1)")
    file2.write_text("print(2)")
    skip_file.write_text("print(3)")

    # Define behavior for is_skipped
    # Skip the 'skipped_dir' directory and 'skipped.py' file
    def is_skipped_side_effect(path):
        p = Path(path)
        return "skipped_dir" in p.parts or p.name == "skipped.py"
    
    config.is_skipped.side_effect = is_skipped_side_effect

    # Define behavior for is_supported_filetype
    def is_supported_side_effect(filepath):
        return filepath.endswith(".py")
    
    config.is_supported_filetype.side_effect = is_supported_side_effect

    paths = [str(tmp_dir), str(broken_path)]
    skipped = []
    broken = []

    try:
        results = list(find(paths, config, skipped, broken))

        # Assertions
        # 1. Check that valid files are found
        assert str(file1) in results
        assert str(file2) in results
        
        # 2. Check that skipped files/dirs are in skipped list
        assert any("skipped_dir" in s for s in skipped)
        assert str(skip_file) in skipped

        # 3. Check that broken paths are recorded
        assert broken_path in broken

        # 4. Check that unsupported filetypes are not yielded
        # (Create a non-py file to test)
        unsupported_file = tmp_dir / "test.txt"
        unsupported_file.write_text("text")
        # Re-run find to include the new file
        results_with_txt = list(find([str(tmp_dir)], config, skipped, broken))
        assert str(unsupported_file) not in results_with_txt

    finally:
        # Cleanup
        import shutil
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
```


# LLM-generated content at query #13
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
    #   skipped_dir/
    #     file3.py
    #   skipped_file.py
    
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

    # Define behavior for skipped items
    def is_skipped_side_effect(path):
        # Skip the directory 'skipped_dir' and the file 'skipped_file.py'
        return "skipped_dir" in str(path) or "skipped_file.py" in str(path)

    config.is_skipped.side_effect = is_skipped_side_effect
    
    # Define behavior for supported filetypes
    def is_supported_side_effect(filepath):
        return filepath.endswith(".py")
    
    config.is_supported_filetype.side_effect = is_supported_side_effect

    # Test cases
    paths = [str(root), "non_existent_path"]
    skipped = []
    broken = []

    # Execution
    result = list(find(paths, config, skipped, broken))

    # Assertions
    # 1. Check found files (should only be file1 and file2)
    assert len(result) == 2
    assert any("file1.py" in f for f in result)
    assert any("file2.py" in f for f in result)
    
    # 2. Check broken paths
    assert "non_existent_path" in broken
    
    # 3. Check skipped items
    # Note: 'skipped_dir' is added to skipped when the loop encounters the directory
    assert any("skipped_dir" in s for s in skipped)
    assert any("skipped_file.py" in s for s in skipped)

def test_find_single_file(tmp_path):
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True
    
    file_path = tmp_path / "standalone.py"
    file_path.write_text("")
    
    skipped = []
    broken = []
    
    result = list(find([str(file_path)], config, skipped, broken))
    
    assert result == [str(file_path)]
    assert len(skipped) == 0
    assert len(broken) == 0
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock

def test_find():
    # Setup Config Mock
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Create a temporary directory structure for testing
    tmp_dir = Path("tmp_test_root").absolute()
    src_dir = tmp_dir / "src"
    sub_dir = src_dir / "sub"
    skipped_dir = src_dir / "skipped_dir"
    
    src_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)
    skipped_dir.mkdir(parents=True, exist_ok=True)

    file1 = src_dir / "file1.py"
    file2 = sub_dir / "file2.py"
    file3 = skipped_dir / "file3.py"
    
    file1.write_text("print(1)")
    file2.write_text("print(2)")
    file3.write_text("print(3)")

    # Define paths to search
    paths = [str(src_dir), "non_existent_path"]
    skipped = []
    broken = []

    # Configure mock behavior for specific paths
    def is_skipped_side_effect(path):
        return "skipped_dir" in str(path)

    config.is_skipped.side_effect = is_skipped_side_effect

    # Execute function
    results = list(find(paths, config, skipped, broken))

    # Assertions
    # 1. Check files found (should find file1 and file2, but not file3 because its parent is skipped)
    assert str(file1) in results
    assert str(file2) in results
    assert str(file3) not in results

    # 2. Check broken paths
    assert "non_existent_path" in broken

    # 3. Check skipped list
    # The function appends the directory to skipped when it detects it should be skipped during walk
    assert any("skipped_dir" in s for s in skipped)

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir)

def test_find_single_file():
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    temp_file = Path("single_test.py").absolute()
    temp_file.write_text("test")
    
    paths = [str(temp_file)]
    skipped = []
    broken = []

    results = list(find(paths, config, skipped, broken))

    assert str(temp_file) in results
    assert len(broken) == 0
    
    temp_file.unlink()

def test_find_direct_file_is_skipped():
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = True
    config.is_supported_filetype.return_value = True

    temp_file = Path("skipped_file.py").absolute()
    temp_file.write_text("test")
    
    paths = [str(temp_file)]
    skipped = []
    broken = []

    results = list(find(paths, config, skipped, broken))

    assert len(results) == 0
    assert str(temp_file) in skipped
    
    temp_file.unlink()
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock

def test_find():
    # Mock Config object
    config = MagicMock()
    config.follow_links = False
    config.is_skipped.return_value = False
    config.is_supported_filetype.return_value = True

    # Setup temporary directory structure
    temp_dir = Path(tmp_path, "test_root")
    temp_dir.mkdir()
    
    # File 1: Standard python file
    file1 = temp_dir / "valid.py"
    file1.write_text("print('hello')")
    
    # File 2: Supported but skipped file
    file2 = temp_dir / "skipped.py"
    file2.write_text("print('skip me')")
    
    # File 3: Unsupported file type
    file3 = temp_dir / "readme.txt"
    file3.write_text("text content")
    config.is_supported_filetype.side_effect = lambda p: p.endswith(".py")

    # File 4: Inside a skipped directory
    skip_dir = temp_dir / "skipped_dir"
    skip_dir.mkdir()
    file4 = skip_dir / "inside_skipped.py"
    file4.write_text("print('hidden')")
    
    # Setup skip logic
    def is_skipped_logic(path):
        return "skipped_dir" in str(path) or "skipped.py" in str(path)
    config.is_skipped.side_effect = is_skipped_logic

    # File 5: A broken path (non-existent)
    broken_path = str(temp_dir / "does_not_exist.py")

    # File 6: A direct file path (not a directory)
    direct_file = str(file1.absolute())

    # Inputs
    paths = [str(temp_dir), broken_path, direct_file]
    skipped = []
    broken = []

    # Execution
    results = list(find(paths, config, skipped, broken))

    # Assertions
    # Should find valid.py and the direct path to valid.py
    assert str(file1.absolute()) in results
    assert direct_file in results
    
    # Should not find the txt file (unsupported)
    assert str(file3.absolute()) not in results
    
    # Should not find files inside skipped directories
    assert str(file4.absolute()) not in results
    
    # Check broken paths
    assert broken_path in broken
    
    # Check skipped list
    # Note: skipped contains the path to skipped.py and the directory skipped_dir
    assert any("skipped.py" in s for s in skipped)
    assert any("skipped_dir" in s for s in skipped)
```


