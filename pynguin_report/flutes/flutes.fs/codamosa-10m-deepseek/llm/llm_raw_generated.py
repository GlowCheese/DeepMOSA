####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function scandir
def test_scandir(): 
    # Test with a Path object
    path = Path('.')
    for child in scandir(path):
        assert isinstance(child, Path)
        assert child.exists()
    
    # Test with a string
    path = '.'
    for child in scandir(path):
        assert isinstance(child, str)
        assert os.path.exists(child)
    
    print("All tests passed!")

if __name__ == "__main__":
    test_scandir()


# LLM-generated content at query #2
#--------------------------

# Unit test for function cache
def test_cache():  
    import tempfile
    import os
    import pickle

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, 'test_cache.pkl')

        # Define a simple function to cache
        @cache(cache_file, verbose=False)
        def expensive_computation(x):
            return x * 2

        # First call should compute and cache
        result1 = expensive_computation(5)
        assert result1 == 10, f"Expected 10, got {result1}"

        # Check that cache file was created
        assert os.path.exists(cache_file), "Cache file should exist"

        # Load cache file to verify content
        with open(cache_file, 'rb') as f:
            cached_result = pickle.load(f)
        assert cached_result == 10, f"Cached value should be 10, got {cached_result}"

        # Modify the cache file to simulate corrupted cache
        with open(cache_file, 'wb') as f:
            pickle.dump(20, f)  # Corrupt cache with wrong value

        # Second call should still load from cache (even if corrupted)
        result2 = expensive_computation(5)
        assert result2 == 20, f"Expected 20 from corrupted cache, got {result2}"

        # Remove cache file to test recomputation
        os.remove(cache_file)
        result3 = expensive_computation(5)
        assert result3 == 10, f"Expected recomputation to return 10, got {result3}"

        # Test with path=None (no caching)
        @cache(None, verbose=False)
        def no_cache_func(x):
            return x + 1

        result4 = no_cache_func(5)
        assert result4 == 6, f"Expected 6, got {result4}"
        # Ensure no cache file was created
        assert not os.path.exists(os.path.join(tmpdir, 'nonexistent.pkl')), "No cache file should be created"

        print("All tests passed!")

if __name__ == "__main__":
    test_cache()


# LLM-generated content at query #3
#--------------------------

# Unit test for function cache
def test_cache(): 
    import tempfile
    import os
    import pickle

    # Create a temporary directory for cache files
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "test_cache.pkl")

        # Define a simple function to cache
        @cache(cache_file, verbose=False)
        def compute_value(x):
            return x * 2

        # First call should compute and cache
        result1 = compute_value(5)
        assert result1 == 10, f"Expected 10, got {result1}"

        # Second call should load from cache
        result2 = compute_value(5)
        assert result2 == 10, f"Expected 10, got {result2}"

        # Verify cache file exists and contains correct data
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
        assert cached_data == 10, f"Expected cached data to be 10, got {cached_data}"

        # Test with different arguments
        result3 = compute_value(7)
        assert result3 == 14, f"Expected 14, got {result3}"

        # Verify cache file updated
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
        assert cached_data == 14, f"Expected cached data to be 14, got {cached_data}"

        print("All cache tests passed.")



# LLM-generated content at query #4
#--------------------------

# Unit test for function scandir
def test_scandir(): 
    # Create a temporary directory with some files and directories
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files and directories
        os.makedirs(os.path.join(tmpdir, "dir1"))
        os.makedirs(os.path.join(tmpdir, "dir2"))
        with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
            f.write("Hello")
        with open(os.path.join(tmpdir, "file2.txt"), "w") as f:
            f.write("World")
        with open(os.path.join(tmpdir, "dir1", "file3.txt"), "w") as f:
            f.write("Foo")
        with open(os.path.join(tmpdir, "dir2", "file4.txt"), "w") as f:
            f.write("Bar")
        
        # Test with Path objects
        path = Path(tmpdir)
        children = list(scandir(path))
        assert len(children) == 4
        assert Path(os.path.join(tmpdir, "dir1")) in children
        assert Path(os.path.join(tmpdir, "dir2")) in children
        assert Path(os.path.join(tmpdir, "file1.txt")) in children
        assert Path(os.path.join(tmpdir, "file2.txt")) in children
        
        # Test with string paths
        children = list(scandir(tmpdir))
        assert len(children) == 4
        assert os.path.join(tmpdir, "dir1") in children
        assert os.path.join(tmpdir, "dir2") in children
        assert os.path.join(tmpdir, "file1.txt") in children
        assert os.path.join(tmpdir, "file2.txt") in children
        
        # Test with nested directory
        children = list(scandir(os.path.join(tmpdir, "dir1")))
        assert len(children) == 1
        assert os.path.join(tmpdir, "dir1", "file3.txt") in children
        
        # Test with empty directory
        os.makedirs(os.path.join(tmpdir, "empty_dir"))
        children = list(scandir(os.path.join(tmpdir, "empty_dir")))
        assert len(children) == 0
        
        print("All tests passed!")

if __name__ == "__main__":
    test_scandir()


# LLM-generated content at query #5
#--------------------------

# Unit test for function copy_tree
def test_copy_tree(): 
    import tempfile
    import filecmp
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a source directory
        src = Path(tmpdir) / "src"
        src.mkdir()
        # Create a destination directory
        dst = Path(tmpdir) / "dst"
        dst.mkdir()
        # Create a file in the source directory
        (src / "file.txt").write_text("Hello, world!")
        # Create a subdirectory in the source directory
        (src / "subdir").mkdir()
        # Create a file in the subdirectory
        (src / "subdir" / "file2.txt").write_text("Hello, world!")
        # Copy the source directory to the destination directory
        copy_tree(src, dst, overwrite=True)
        # Check that the destination directory contains the same files as the source directory
        assert filecmp.dircmp(src, dst).diff_files == []
        # Check that the destination directory contains the same subdirectories as the source directory
        assert filecmp.dircmp(src, dst).common_dirs == ["subdir"]
        # Check that the destination directory contains the same files in the subdirectory as the source directory
        assert filecmp.dircmp(src / "subdir", dst / "subdir").diff_files == []
        # Check that the destination directory contains the same files as the source directory when overwrite is False
        copy_tree(src, dst, overwrite=False)
        assert filecmp.dircmp(src, dst).diff_files == []
        # Check that the destination directory contains the same subdirectories as the source directory when overwrite is False
        assert filecmp.dircmp(src, dst).common_dirs == ["subdir"]
        # Check that the destination directory contains the same files in the subdirectory as the source directory when overwrite is False
        assert filecmp.dircmp(src / "subdir", dst / "subdir").diff_files == []
        # Check that the destination directory contains the same files as the source directory when overwrite is True and the destination directory already exists
        copy_tree(src, dst, overwrite=True)
        assert filecmp.dircmp(src, dst).diff_files == []
        # Check that the destination directory contains the same subdirectories as the source directory when overwrite is True and the destination directory already exists
        assert filecmp.dircmp(src, dst).common_dirs == ["subdir"]
        # Check that the destination directory contains the same files in the subdirectory as the source directory when overwrite is True and the destination directory already exists
        assert filecmp.dircmp(src / "subdir", dst / "subdir").diff_files == []
        # Check that the destination directory contains the same files as the source directory when overwrite is False and the destination directory already exists
        copy_tree(src, dst, overwrite=False)
        assert filecmp.dircmp(src, dst).diff_files == []
        # Check that the destination directory contains the same subdirectories as the source directory when overwrite is False and the destination directory already exists
        assert filecmp.dircmp(src, dst).common_dirs == ["subdir"]
        # Check that the destination directory contains the same files in the subdirectory as the source directory when overwrite is False and the destination directory already exists
        assert filecmp.dircmp(src / "subdir", dst / "subdir").diff_files == []
        # Check that the destination directory contains the same files as the source directory when overwrite is True and the destination directory already exists and contains a file with the same name
        (dst / "file.txt").write_text("Goodbye, world!")
        copy_tree(src, dst, overwrite=True)
        assert filecmp.dircmp(src, dst).diff_files == []
        # Check that the destination directory contains the same subdirectories as the source directory when overwrite is True and the destination directory already exists and contains a file with the same name
        assert filecmp.dircmp(src, dst).common_dirs == ["subdir"]
        # Check that the destination directory contains the same files in the subdirectory as the source directory when overwrite is True and the destination directory already exists and contains a file with the same name
        assert filecmp.dircmp(src / "subdir", dst / "subdir").diff_files == []
        # Check that the destination directory contains the same files as the source directory when overwrite is False and the destination directory already exists and contains a file with the same name
        (dst / "file.txt").write_text("Goodbye, world!")
        copy_tree(src, dst, overwrite=False)
        assert filecmp.dircmp(src, dst).diff_files == ["file.txt"]
        # Check that the destination directory contains the same subdirectories as the source directory when overwrite is False and the destination directory already exists and contains a file with the same name
        assert filecmp.dircmp(src, dst).common_dirs == ["subdir"]
        # Check that the destination directory contains the same files in the subdirectory as the source directory when overwrite is False and the destination directory already exists and contains a file with the same name
        assert filecmp.dircmp(src / "subdir", dst / "subdir").diff_files == []
        # Check that the destination directory contains the same files as the source directory when overwrite is True and the destination directory already exists and contains a subdirectory with the same name
        (dst / "subdir").mkdir()
        copy_tree(src, dst, overwrite=True)
        assert filecmp.dircmp(src, dst).diff_files == []
        # Check that the destination directory contains the same subdirectories as the source directory when overwrite is True and the destination directory already exists and contains a subdirectory with the same name
        assert filecmp.dircmp(src, dst).common_dirs == ["subdir"]
        # Check that the destination directory contains the same files in the subdirectory as the source directory when overwrite is True and the destination directory already exists and contains a subdirectory with the same name
        assert filecmp.dircmp(src / "subdir", dst / "subdir").diff_files == []
        # Check that the destination directory contains the same files as the source directory when overwrite is False and the destination directory already exists and contains a subdirectory with the same name
        (dst / "subdir").mkdir()
        copy_tree(src, dst, overwrite=False)
        assert filecmp.dircmp(src, dst).diff_files == []
        # Check that the destination directory contains the same subdirectories as the source directory when overwrite is False and the destination directory already exists and contains a subdirectory with the same name
        assert filecmp.dircmp(src, dst).common_dirs == ["subdir"]
        # Check that the destination directory contains the same files in the subdirectory as the source directory when overwrite is False and the destination directory already exists and contains a subdirectory with the same name
        assert filecmp.dircmp(src / "subdir", dst / "subdir").diff_files == []
        # Check that the destination directory contains the same files as the source directory when overwrite is True and the destination directory already exists and contains a file with the same name in a subdirectory
        (dst / "subdir" / "file2.txt").write_text("Goodbye, world!")
        copy_tree(src, dst, overwrite=True)
        assert filecmp.dircmp(src, dst).diff_files == []
        # Check that the destination directory contains the same subdirectories as the source directory when overwrite is True and the destination directory already exists and contains a file with the same name in a subdirectory
        assert filecmp.dircmp(src, dst).common_dirs == ["subdir"]
        # Check that the destination directory contains the same files in the subdirectory as the source directory when overwrite is True and the destination directory already exists and contains a file with the same name in a subdirectory
        assert filecmp.dircmp(src / "subdir", dst / "subdir").diff_files == []
        # Check that the destination directory contains the same files as the source directory when overwrite is False and the destination directory already exists and contains a file with the same name in a subdirectory
        (dst / "subdir" / "file2.txt").write_text("Goodbye, world!")
        copy_tree(src, dst, overwrite=False)
        assert filecmp.dircmp(src, dst).diff_files == []
        # Check that the destination directory contains the same subdirectories as the source directory when overwrite is False and the destination directory already exists and contains a file with the same name in a subdirectory
        assert filecmp.dircmp(src, dst).common_dirs == ["subdir"]
        # Check that the destination directory contains the same files in the subdirectory as the source directory when overwrite is False and the destination directory already exists and contains a file with the same name in a subdirectory
        assert filecmp.dircmp(src / "subdir", dst / "subdir").diff_files == ["file2.txt"]
        # Check that the destination directory contains the same files as the source directory when overwrite is True and the destination directory already exists and contains a subdirectory with the same name and a file with the same name in that subdirectory
        (dst / "subdir").mkdir()
        (dst / "subdir" / "file2.txt").write_text("Goodbye, world!")
        copy_tree(src, dst, overwrite=True)
        assert filecmp.dircmp(src, dst).diff_files == []
       


# LLM-generated content at query #6
#--------------------------

# Unit test for function cache
def test_cache():  # pragma: no cover
    import tempfile
    import time

    @cache(None)
    def func1():
        time.sleep(0.1)
        return 42

    @cache(tempfile.mktemp())
    def func2():
        time.sleep(0.1)
        return 43

    @cache(tempfile.mktemp())
    def func3():
        time.sleep(0.1)
        return 44

    assert func1() == 42
    assert func2() == 43
    assert func3() == 44

    # Test that the cache is used
    start = time.time()
    assert func2() == 43
    assert time.time() - start < 0.05

    # Test that the cache is not used
    start = time.time()
    assert func1() == 42
    assert time.time() - start >= 0.1

    print("All tests passed.")


if __name__ == "__main__":  # pragma: no cover
    test_cache()


# LLM-generated content at query #7
#--------------------------

# Unit test for function scandir
def test_scandir(): 
    # Test with Path object
    path = Path(".")
    for child in scandir(path):
        assert isinstance(child, Path)
        assert child.exists()
    # Test with string
    path = "."
    for child in scandir(path):
        assert isinstance(child, str)
        assert os.path.exists(child)
    print("All tests passed.")

if __name__ == "__main__":
    test_scandir()


# LLM-generated content at query #8
#--------------------------

# Unit test for function cache
def test_cache(): 
    import tempfile
    import os
    import pickle
    from .log import log

    # Create a temporary directory for the cache file
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, 'test_cache.pkl')
        
        # Define a simple function to cache
        @cache(cache_path, verbose=False, name='test')
        def expensive_computation(x):
            return x * 2
        
        # First call should compute and cache
        result1 = expensive_computation(5)
        assert result1 == 10
        
        # Check that cache file was created
        assert os.path.exists(cache_path)
        
        # Load cache file to verify content
        with open(cache_path, 'rb') as f:
            cached_result = pickle.load(f)
        assert cached_result == 10
        
        # Modify the cache file to simulate corrupted cache
        with open(cache_path, 'wb') as f:
            pickle.dump(20, f)  # Different value
        
        # Second call should load from cache (even though it's different)
        result2 = expensive_computation(5)
        assert result2 == 20  # Should load corrupted cache
        
        # Test with path=None (no caching)
        @cache(None, verbose=False)
        def no_cache_func(x):
            return x * 3
        
        result3 = no_cache_func(5)
        assert result3 == 15
        # No cache file should be created
        assert not os.path.exists(os.path.join(tmpdir, 'nonexistent.pkl'))
        
        print("All cache tests passed!")



# LLM-generated content at query #9
#--------------------------

# Unit test for function copy_tree
def test_copy_tree(): 
    import tempfile
    import filecmp
    import os
    import shutil
    import sys
    import unittest
    import flutes

    class TestCopyTree(unittest.TestCase):
        def setUp(self):
            # Create a temporary directory
            self.test_dir = tempfile.mkdtemp()

        def tearDown(self):
            # Remove the directory after the test
            shutil.rmtree(self.test_dir)

        def test_copy_tree(self):
            # Create a source directory with some files and subdirectories
            src_dir = os.path.join(self.test_dir, 'src')
            os.makedirs(src_dir)
            with open(os.path.join(src_dir, 'file1.txt'), 'w') as f:
                f.write('Hello, world!')
            os.makedirs(os.path.join(src_dir, 'subdir'))
            with open(os.path.join(src_dir, 'subdir', 'file2.txt'), 'w') as f:
                f.write('Another file')

            # Create a destination directory
            dst_dir = os.path.join(self.test_dir, 'dst')
            os.makedirs(dst_dir)

            # Copy the source directory to the destination directory
            flutes.copy_tree(src_dir, dst_dir)

            # Check that the destination directory contains the same files as the source directory
            self.assertTrue(filecmp.dircmp(src_dir, dst_dir).same_files)

        def test_copy_tree_overwrite(self):
            # Create a source directory with a file
            src_dir = os.path.join(self.test_dir, 'src')
            os.makedirs(src_dir)
            with open(os.path.join(src_dir, 'file.txt'), 'w') as f:
                f.write('Source file')

            # Create a destination directory with a file with the same name but different content
            dst_dir = os.path.join(self.test_dir, 'dst')
            os.makedirs(dst_dir)
            with open(os.path.join(dst_dir, 'file.txt'), 'w') as f:
                f.write('Destination file')

            # Copy the source directory to the destination directory with overwrite=True
            flutes.copy_tree(src_dir, dst_dir, overwrite=True)

            # Check that the destination file has been overwritten with the source file content
            with open(os.path.join(dst_dir, 'file.txt'), 'r') as f:
                self.assertEqual(f.read(), 'Source file')

        def test_copy_tree_no_overwrite(self):
            # Create a source directory with a file
            src_dir = os.path.join(self.test_dir, 'src')
            os.makedirs(src_dir)
            with open(os.path.join(src_dir, 'file.txt'), 'w') as f:
                f.write('Source file')

            # Create a destination directory with a file with the same name but different content
            dst_dir = os.path.join(self.test_dir, 'dst')
            os.makedirs(dst_dir)
            with open(os.path.join(dst_dir, 'file.txt'), 'w') as f:
                f.write('Destination file')

            # Copy the source directory to the destination directory with overwrite=False
            flutes.copy_tree(src_dir, dst_dir, overwrite=False)

            # Check that the destination file has not been overwritten
            with open(os.path.join(dst_dir, 'file.txt'), 'r') as f:
                self.assertEqual(f.read(), 'Destination file')

    # Run the tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCopyTree)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == "__main__":
    test_copy_tree()


# LLM-generated content at query #10
#--------------------------

# Unit test for function readable_size
def test_readable_size(): 
    assert readable_size(1024 * 1024) == "1.00M"
    assert readable_size(1024 * 1024 * 1024) == "1.00G"
    assert readable_size(1024 * 1024 * 1024 * 1024) == "1.00T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1024.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1048576.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1073741824.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1099511627776.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1125899906842624.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1152921504606846976.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1180591620717411303424.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1208925819614629174706176.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1237940039285380274899124224.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1267650600228229401496703205376.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1298074214633706907132624082305024.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1329227995784915872903807060280344576.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1361129467683753853853498429727072845824.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1393796574908163946345982392040522594123776.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1427247692705959881058285969449495136382746624.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1461501637330902918203684832716283019655932542976.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1496577676626844588240573268701473812127674924007424.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1532495540865888858358347027150309183618739122183602176.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1569275433846670190958947355801916604025588861116008628224.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1606938044258990275541962092341162602522202993782792835301376.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1645504557321206042154969182557350504982735865633579863348609024.00P"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024) == "1684996666696914987166688442938726917102321526403515786552695619584.00P"



# LLM-generated content at query #11
#--------------------------

# Unit test for function cache
def test_cache(): 
    import tempfile
    import os
    import pickle

    # Create a temporary directory for the cache file
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, 'test_cache.pkl')

        # Define a function to be cached
        @cache(cache_file, verbose=False, name='test')
        def expensive_computation(x):
            return x * 2

        # Test 1: Cache miss - function should be executed
        result = expensive_computation(5)
        assert result == 10, f"Expected 10, got {result}"
        assert os.path.exists(cache_file), "Cache file should exist after first call"

        # Test 2: Cache hit - function should not be executed, result loaded from cache
        # Modify the cache file to simulate a different result
        with open(cache_file, 'wb') as f:
            pickle.dump(20, f)  # Change cached value to 20

        result = expensive_computation(5)
        assert result == 20, f"Expected 20 (from cache), got {result}"

        # Test 3: Cache miss with different arguments
        result = expensive_computation(10)
        assert result == 20, f"Expected 20 (from cache for arg 5), got {result}"
        # Note: The cache decorator does not differentiate by arguments, so this is expected behavior

        # Test 4: No cache path provided
        @cache(None, verbose=False, name='test')
        def no_cache_func(x):
            return x * 3

        result = no_cache_func(5)
        assert result == 15, f"Expected 15, got {result}"
        assert not os.path.exists(os.path.join(tmpdir, 'nonexistent.pkl')), "No cache file should be created"

        print("All tests passed!")



# LLM-generated content at query #12
#--------------------------

# Unit test for function cache
def test_cache():  
    import tempfile  
    import os  
    import pickle  
    from flutes import log  
  
    # Create a temporary directory for cache files  
    with tempfile.TemporaryDirectory() as tmpdir:  
        cache_file = os.path.join(tmpdir, 'test_cache.pkl')  
  
        # Define a simple function to cache  
        @cache(cache_file, verbose=True, name='test')  
        def expensive_computation(x):  
            log(f"Computing for {x}")  
            return x * x  
  
        # First call should compute and cache  
        result1 = expensive_computation(5)  
        assert result1 == 25, f"Expected 25, got {result1}"  
  
        # Second call should load from cache  
        result2 = expensive_computation(5)  
        assert result2 == 25, f"Expected 25, got {result2}"  
  
        # Verify cache file exists  
        assert os.path.exists(cache_file), "Cache file should exist"  
  
        # Load cache file directly to verify content  
        with open(cache_file, 'rb') as f:  
            cached_result = pickle.load(f)  
        assert cached_result == 25, f"Cached result should be 25, got {cached_result}"  
  
        log("All cache tests passed!")  
  


# LLM-generated content at query #13
#--------------------------

# Unit test for function scandir
def test_scandir(): 
    # Test with Path object
    path = Path(".")
    for child in scandir(path):
        assert isinstance(child, Path)
        assert child.exists()
    # Test with string
    path = "."
    for child in scandir(path):
        assert isinstance(child, str)
        assert os.path.exists(child)
    print("All tests passed!")

if __name__ == "__main__":
    test_scandir()


# LLM-generated content at query #14
#--------------------------

# Unit test for function cache
def test_cache():  
    import tempfile
    import os
    import pickle

    # Create a temporary directory for the cache file
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, 'test_cache.pkl')

        # Define a simple function to cache
        @cache(cache_file, verbose=False)
        def compute_value(x):
            return x * 2

        # First call should compute and cache the result
        result1 = compute_value(5)
        assert result1 == 10, f"Expected 10, got {result1}"

        # Check that the cache file was created
        assert os.path.exists(cache_file), "Cache file was not created"

        # Load the cache file to verify its content
        with open(cache_file, 'rb') as f:
            cached_result = pickle.load(f)
        assert cached_result == 10, f"Expected cached result 10, got {cached_result}"

        # Modify the cache file to simulate a different cached result
        with open(cache_file, 'wb') as f:
            pickle.dump(20, f)

        # Second call should load from cache (modified value)
        result2 = compute_value(5)
        assert result2 == 20, f"Expected 20 from cache, got {result2}"

        # Test with path=None (no caching)
        @cache(None, verbose=False)
        def compute_another(x):
            return x + 1

        result3 = compute_another(5)
        assert result3 == 6, f"Expected 6, got {result3}"
        # Ensure no cache file was created
        assert not os.path.exists(os.path.join(tmpdir, 'nonexistent.pkl')), "Cache file should not exist"

        print("All tests passed!")

# Run the unit test
if __name__ == "__main__":
    test_cache()


# LLM-generated content at query #15
#--------------------------

# Unit test for function scandir
def test_scandir(): 
    # Test with a Path object
    path = Path(".")
    for child in scandir(path):
        assert isinstance(child, Path)
        print(child)
    # Test with a string
    path = "."
    for child in scandir(path):
        assert isinstance(child, str)
        print(child)
    print("All tests passed!")

if __name__ == "__main__":
    test_scandir()


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a subdirectory and files
        sub_dir = Path(temp_dir) / "subdir"
        sub_dir.mkdir()

        file1 = sub_dir / "file1.txt"
        file1.touch()

        file2 = sub_dir / "file2.txt"
        file2.touch()

        # Test scandir with Path
        paths = list(scandir(sub_dir))
        assert len(paths) == 2
        assert Path(file1) in paths
        assert Path(file2) in paths

        # Test scandir with str
        paths = list(scandir(str(sub_dir)))
        assert len(paths) == 2
        assert str(file1) in paths
        assert str(file2) in paths


# LLM-generated content at query #2
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import shutil
    import time

    cache_dir = tempfile.mkdtemp()
    cache_path = os.path.join(cache_dir, 'cache.pkl')

    @cache(cache_path)
    def expensive_computation():
        time.sleep(2)
        return 42

    # First call should take time
    start_time = time.time()
    result = expensive_computation()
    elapsed_time = time.time() - start_time
    assert result == 42
    assert elapsed_time >= 2

    # Second call should be instant due to cache
    start_time = time.time()
    result = expensive_computation()
    elapsed_time = time.time() - start_time
    assert result == 42
    assert elapsed_time < 0.1

    shutil.rmtree(cache_dir)


# LLM-generated content at query #3
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import pytest

    @cache(None)
    def func_none():
        return "func_none"

    assert func_none() == "func_none"

    with tempfile.NamedTemporaryFile() as tmpfile:
        @cache(tmpfile.name)
        def func_cache():
            return "func_cache"

        assert func_cache() == "func_cache"
        assert func_cache() == "func_cache"

    with tempfile.NamedTemporaryFile() as tmpfile:
        @cache(tmpfile.name, verbose=False)
        def func_cache_verbose():
            return "func_cache_verbose"

        assert func_cache_verbose() == "func_cache_verbose"
        assert func_cache_verbose() == "func_cache_verbose"

    with pytest.raises(TypeError):
        @cache(123)
        def func_cache_invalid():
            return "func_cache_invalid"

        func_cache_invalid()


# LLM-generated content at query #4
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create some files and directories
        os.makedirs(os.path.join(tmpdirname, "dir1"))
        os.makedirs(os.path.join(tmpdirname, "dir2"))
        with open(os.path.join(tmpdirname, "file1.txt"), "w") as f:
            f.write("test")
        with open(os.path.join(tmpdirname, "file2.txt"), "w") as f:
            f.write("test")

        # Test scandir with Path
        paths = list(scandir(Path(tmpdirname)))
        assert len(paths) == 4
        assert Path(os.path.join(tmpdirname, "dir1")) in paths
        assert Path(os.path.join(tmpdirname, "dir2")) in paths
        assert Path(os.path.join(tmpdirname, "file1.txt")) in paths
        assert Path(os.path.join(tmpdirname, "file2.txt")) in paths

        # Test scandir with string
        paths = list(scandir(tmpdirname))
        assert len(paths) == 4
        assert os.path.join(tmpdirname, "dir1") in paths
        assert os.path.join(tmpdirname, "dir2") in paths
        assert os.path.join(tmpdirname, "file1.txt") in paths
        assert os.path.join(tmpdirname, "file2.txt") in paths


# LLM-generated content at query #5
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src"
        dst_path = Path(tmp) / "dst"

        # Create source directory and files
        src_path.mkdir()
        (src_path / "file1.txt").write_text("file1")
        (src_path / "file2.txt").write_text("file2")
        (src_path / "subdir").mkdir()
        (src_path / "subdir" / "file3.txt").write_text("file3")

        # Test copying with overwrite
        copy_tree(src_path, dst_path, overwrite=True)
        assert (dst_path / "file1.txt").read_text() == "file1"
        assert (dst_path / "file2.txt").read_text() == "file2"
        assert (dst_path / "subdir" / "file3.txt").read_text() == "file3"

        # Modify a file in source and copy without overwrite
        (src_path / "file1.txt").write_text("modified")
        copy_tree(src_path, dst_path, overwrite=False)
        assert (dst_path / "file1.txt").read_text() == "file1"

        # Test copying with overwrite again
        copy_tree(src_path, dst_path, overwrite=True)
        assert (dst_path / "file1.txt").read_text() == "modified"



# LLM-generated content at query #6
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os
    import shutil

    # Temporary directory for testing
    temp_dir = tempfile.mkdtemp()

    try:
        # Test function to cache
        def test_func():
            return {"key": "value"}

        # Cache file path
        cache_file = os.path.join(temp_dir, "test_cache.pkl")

        # Decorate the function with cache
        cached_func = cache(cache_file)(test_func)

        # First call, function should be executed and cache should be saved
        result = cached_func()
        assert result == {"key": "value"}
        assert os.path.exists(cache_file)

        # Second call, function should not be executed and cache should be loaded
        result = cached_func()
        assert result == {"key": "value"}

        # Decorate with verbose=False to suppress logging
        cached_func_verbose = cache(cache_file, verbose=False)(test_func)

        # Call function, cache should be loaded without logging
        result = cached_func_verbose()
        assert result == {"key": "value"}

        # Decorate with None path, function should always be executed
        cached_func_none = cache(None)(test_func)

        # Call function, it should be executed without caching
        result = cached_func_none()
        assert result == {"key": "value"}
        assert not os.path.exists(os.path.join(temp_dir, "none.pkl"))

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


# LLM-generated content at query #7
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os
    import pickle

    # Test with path specified
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, 'test_cache.pkl')
        test_value = {'key': 'value'}

        @cache(cache_file)
        def func():
            return test_value

        # First call, function should be executed
        result = func()
        assert result == test_value, "Function output does not match expected value"

        # Second call, cache should be loaded
        result = func()
        assert result == test_value, "Cached output does not match expected value"

        # Modify cache file and ensure it is not reloaded
        with open(cache_file, 'wb') as f:
            pickle.dump({'wrong_key': 'wrong_value'}, f)

        result = func()
        assert result == test_value, "Cached output should not be reloaded"

    # Test with path as None
    @cache(None)
    def func():
        return test_value

    result = func()
    assert result == test_value, "Function output does not match expected value with cache disabled"


# LLM-generated content at query #8
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import filecmp

    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, 'src')
        dst = os.path.join(tmpdir, 'dst')

        # Create source directory structure
        os.makedirs(src)
        os.makedirs(os.path.join(src, 'subdir'))
        with open(os.path.join(src, 'file1.txt'), 'w') as f:
            f.write('file1')
        with open(os.path.join(src, 'subdir', 'file2.txt'), 'w') as f:
            f.write('file2')

        # Copy source to destination
        copy_tree(src, dst)

        # Check if destination directory structure matches source
        assert os.path.exists(dst)
        assert os.path.exists(os.path.join(dst, 'subdir'))
        assert os.path.exists(os.path.join(dst, 'file1.txt'))
        assert os.path.exists(os.path.join(dst, 'subdir', 'file2.txt'))

        # Check if file contents are the same
        assert filecmp.cmp(os.path.join(src, 'file1.txt'), os.path.join(dst, 'file1.txt'))
        assert filecmp.cmp(os.path.join(src, 'subdir', 'file2.txt'), os.path.join(dst, 'subdir', 'file2.txt'))

        # Test overwrite
        with open(os.path.join(src, 'file1.txt'), 'w') as f:
            f.write('new content')
        copy_tree(src, dst, overwrite=True)
        assert filecmp.cmp(os.path.join(src, 'file1.txt'), os.path.join(dst, 'file1.txt'))

        # Test no overwrite
        with open(os.path.join(src, 'file1.txt'), 'w') as f:
            f.write('another content')
        copy_tree(src, dst, overwrite=False)
        assert not filecmp.cmp(os.path.join(src, 'file1.txt'), os.path.join(dst, 'file1.txt'))


# LLM-generated content at query #9
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import shutil

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create some test files and directories
        test_files = ["file1.txt", "file2.txt"]
        test_dirs = ["dir1", "dir2"]
        for file in test_files:
            with open(os.path.join(temp_dir, file), "w") as f:
                f.write("test")
        for dir in test_dirs:
            os.makedirs(os.path.join(temp_dir, dir))

        # Test with Path input
        path_obj = Path(temp_dir)
        results = list(scandir(path_obj))
        assert len(results) == 4
        for result in results:
            assert isinstance(result, Path)

        # Test with string input
        results = list(scandir(temp_dir))
        assert len(results) == 4
        for result in results:
            assert isinstance(result, str)

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #10
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import os
    import pathlib

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files and directories
        os.makedirs(os.path.join(tmpdir, "subdir1"))
        os.makedirs(os.path.join(tmpdir, "subdir2"))
        with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
            f.write("test")
        with open(os.path.join(tmpdir, "file2.txt"), "w") as f:
            f.write("test")

        # Test with str path
        entries = list(scandir(tmpdir))
        assert len(entries) == 4
        assert any("subdir1" in str(p) for p in entries)
        assert any("subdir2" in str(p) for p in entries)
        assert any("file1.txt" in str(p) for p in entries)
        assert any("file2.txt" in str(p) for p in entries)

        # Test with Path path
        entries = list(scandir(pathlib.Path(tmpdir)))
        assert len(entries) == 4
        assert any("subdir1" in str(p) for p in entries)
        assert any("subdir2" in str(p) for p in entries)
        assert any("file1.txt" in str(p) for p in entries)
        assert any("file2.txt" in str(p) for p in entries)


# LLM-generated content at query #11
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import shutil

    # Create temporary directories
    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()

    # Create files in source directory
    with open(os.path.join(src_dir, 'test_file1.txt'), 'w') as f:
        f.write('test content')
    os.makedirs(os.path.join(src_dir, 'subdir'))
    with open(os.path.join(src_dir, 'subdir', 'test_file2.txt'), 'w') as f:
        f.write('test content')

    # Perform the copy operation
    copy_tree(src_dir, dst_dir, overwrite=True)

    # Verify the copy operation
    assert os.path.exists(os.path.join(dst_dir, 'test_file1.txt'))
    assert os.path.exists(os.path.join(dst_dir, 'subdir', 'test_file2.txt'))

    # Clean up
    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)


# LLM-generated content at query #12
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    from pathlib import Path
    from flutes import cache

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_file = Path(tmp_dir) / "test_cache.pkl"

        # Test function
        @cache(cache_file)
        def test_func():
            return "test_value"

        # First call, cache file should not exist
        assert test_func() == "test_value"
        assert cache_file.exists()

        # Second call, cache file should exist and value should be loaded from cache
        assert test_func() == "test_value"

        # Modify cache file to test if the function returns the cached value
        with open(cache_file, "wb") as f:
            pickle.dump("modified_value", f)

        assert test_func() == "modified_value"

        # Test with path=None, no cache should be used
        @cache(None)
        def test_func_no_cache():
            return "test_value"

        assert test_func_no_cache() == "test_value"
        assert not (Path(tmp_dir) / "test_cache.pkl").exists()


# LLM-generated content at query #13
#--------------------------

# Unit test for function scandir
def test_scandir():
    # Create a temporary directory with some files and subdirectories
    temp_dir = Path("temp_dir")
    temp_dir.mkdir(exist_ok=True)
    (temp_dir / "file1.txt").touch()
    (temp_dir / "file2.txt").touch()
    (temp_dir / "sub_dir").mkdir()
    (temp_dir / "sub_dir" / "file3.txt").touch()

    # Test scandir with Path input
    paths = list(scandir(temp_dir))
    assert len(paths) == 3
    assert Path("temp_dir/file1.txt") in paths
    assert Path("temp_dir/file2.txt") in paths
    assert Path("temp_dir/sub_dir") in paths

    # Test scandir with str input
    paths = list(scandir(str(temp_dir)))
    assert len(paths) == 3
    assert str(Path("temp_dir/file1.txt")) in paths
    assert str(Path("temp_dir/file2.txt")) in paths
    assert str(Path("temp_dir/sub_dir")) in paths

    # Clean up
    shutil.rmtree(temp_dir)


# LLM-generated content at query #14
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os

    def foo(x):
        return x + 1

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = os.path.join(temp_dir, "cache.pkl")
        cached_foo = cache(cache_file)(foo)
        assert cached_foo(1) == 2
        assert os.path.exists(cache_file)
        assert cached_foo(1) == 2


# LLM-generated content at query #15
#--------------------------

# Unit test for function scandir
def test_scandir():
    # Create a temporary directory with files and subdirectories
    import tempfile
    import shutil
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        (tmpdir / "file1").touch()
        (tmpdir / "file2").touch()
        (tmpdir / "subdir").mkdir()
        (tmpdir / "subdir" / "file3").touch()

        # Test scandir with Path
        results = list(scandir(tmpdir))
        assert len(results) == 3
        assert any(str(file).endswith("file1") for file in results)
        assert any(str(file).endswith("file2") for file in results)
        assert any(str(file).endswith("subdir") for file in results)

        # Test scandir with str
        results = list(scandir(str(tmpdir)))
        assert len(results) == 3
        assert any(file.endswith("file1") for file in results)
        assert any(file.endswith("file2") for file in results)
        assert any(file.endswith("subdir") for file in results)

        # Clean up
        shutil.rmtree(tmpdirname)

# Run the unit test
test_scandir()


# LLM-generated content at query #16
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import os
    import pathlib

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files and directories
        os.makedirs(os.path.join(tmpdir, "dir1"))
        os.makedirs(os.path.join(tmpdir, "dir2"))
        with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
            f.write("test")
        with open(os.path.join(tmpdir, "file2.txt"), "w") as f:
            f.write("test")

        # Test with string path
        entries = list(scandir(tmpdir))
        assert len(entries) == 4
        assert any("dir1" in str(entry) for entry in entries)
        assert any("dir2" in str(entry) for entry in entries)
        assert any("file1.txt" in str(entry) for entry in entries)
        assert any("file2.txt" in str(entry) for entry in entries)

        # Test with Path object
        path = pathlib.Path(tmpdir)
        entries = list(scandir(path))
        assert len(entries) == 4
        assert any("dir1" in str(entry) for entry in entries)
        assert any("dir2" in str(entry) for entry in entries)
        assert any("file1.txt" in str(entry) for entry in entries)
        assert any("file2.txt" in str(entry) for entry in entries)

    print("test_scandir passed")


# LLM-generated content at query #17
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    src = Path("test_src")
    dst = Path("test_dst")
    try:
        src.mkdir()
        (src / "file1.txt").write_text("file1")
        (src / "file2.txt").write_text("file2")
        (src / "dir1").mkdir()
        (src / "dir1" / "file3.txt").write_text("file3")
        (src / "dir1" / "file4.txt").write_text("file4")
        (src / "dir2").mkdir()
        (src / "dir2" / "file5.txt").write_text("file5")
        (src / "dir2" / "file6.txt").write_text("file6")
        copy_tree(src, dst)
        assert (dst / "file1.txt").read_text() == "file1"
        assert (dst / "file2.txt").read_text() == "file2"
        assert (dst / "dir1" / "file3.txt").read_text() == "file3"
        assert (dst / "dir1" / "file4.txt").read_text() == "file4"
        assert (dst / "dir2" / "file5.txt").read_text() == "file5"
        assert (dst / "dir2" / "file6.txt").read_text() == "file6"
    finally:
        shutil.rmtree(src)
        shutil.rmtree(dst)


# LLM-generated content at query #18
#--------------------------

# Unit test for function readable_size
def test_readable_size():
    assert readable_size(1024) == "1.00K"
    assert readable_size(1024 * 1024) == "1.00M"
    assert readable_size(1024 * 1024 * 1024) == "1.00G"
    assert readable_size(1024 * 1024 * 1024 * 1024) == "1.00T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.00P"
    assert readable_size(0) == "0.00"
    assert readable_size(1) == "1.00"
    assert readable_size(999) == "999.00"
    assert readable_size(1000) == "1000.00"
    assert readable_size(1023) == "1023.00"
    assert readable_size(1025) == "1.00K"
    assert readable_size(2048) == "2.00K"
    assert readable_size(2048, n_digits=0) == "2K"
    assert readable_size(2048, n_digits=1) == "2.0K"
    assert readable_size(2048, n_digits=3) == "2.000K"


# LLM-generated content at query #19
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os
    import pickle

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Define a cache path
        cache_path = os.path.join(tmpdirname, "cache.pkl")

        # Define a function to cache
        @cache(cache_path)
        def add(a, b):
            return a + b

        # First call: should compute and cache
        result = add(2, 3)
        assert result == 5
        assert os.path.exists(cache_path)

        # Second call: should load from cache
        cached_result = add(2, 3)
        assert cached_result == 5

        # Verify cache content
        with open(cache_path, "rb") as f:
            cached_value = pickle.load(f)
        assert cached_value == 5

        # Modify cache file
        with open(cache_path, "wb") as f:
            pickle.dump(10, f)

        # Third call: should load modified cache
        modified_result = add(2, 3)
        assert modified_result == 10

        # Test with no cache path
        @cache(None)
        def subtract(a, b):
            return a - b

        # Should compute every time
        result = subtract(5, 3)
        assert result == 2
        assert not os.path.exists(os.path.join(tmpdirname, "None.pkl"))


# LLM-generated content at query #20
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    src = Path("test_src")
    dst = Path("test_dst")
    src.mkdir(exist_ok=True)
    dst.mkdir(exist_ok=True)
    (src / "file1.txt").write_text("file1")
    (src / "file2.txt").write_text("file2")
    (src / "subdir").mkdir()
    (src / "subdir" / "file3.txt").write_text("file3")
    copy_tree(src, dst)
    assert (dst / "file1.txt").read_text() == "file1"
    assert (dst / "file2.txt").read_text() == "file2"
    assert (dst / "subdir" / "file3.txt").read_text() == "file3"
    shutil.rmtree(src)
    shutil.rmtree(dst)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import os
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files and directories
        os.makedirs(os.path.join(tmpdir, "subdir1"))
        os.makedirs(os.path.join(tmpdir, "subdir2"))
        with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
            f.write("test")
        with open(os.path.join(tmpdir, "file2.txt"), "w") as f:
            f.write("test")

        # Test with Path input
        path = Path(tmpdir)
        entries = list(scandir(path))
        assert len(entries) == 4
        assert any(entry.name == "subdir1" for entry in entries)
        assert any(entry.name == "subdir2" for entry in entries)
        assert any(entry.name == "file1.txt" for entry in entries)
        assert any(entry.name == "file2.txt" for entry in entries)

        # Test with str input
        entries = list(scandir(tmpdir))
        assert len(entries) == 4
        assert any(os.path.basename(entry) == "subdir1" for entry in entries)
        assert any(os.path.basename(entry) == "subdir2" for entry in entries)
        assert any(os.path.basename(entry) == "file1.txt" for entry in entries)
        assert any(os.path.basename(entry) == "file2.txt" for entry in entries)


# LLM-generated content at query #2
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = os.path.join(temp_dir, "test_cache.pkl")

        @cache(cache_file)
        def test_function():
            return {"key": "value"}

        # First call, cache should be created
        result = test_function()
        assert result == {"key": "value"}
        assert os.path.exists(cache_file)

        # Second call, cache should be loaded
        result = test_function()
        assert result == {"key": "value"}

        # Test with verbose=False
        @cache(cache_file, verbose=False)
        def test_function_verbose():
            return {"key": "value"}

        result = test_function_verbose()
        assert result == {"key": "value"}

        # Test with path=None
        @cache(None)
        def test_function_no_cache():
            return {"key": "value"}

        result = test_function_no_cache()
        assert result == {"key": "value"}
        assert not os.path.exists(os.path.join(temp_dir, "none_cache.pkl"))


# LLM-generated content at query #3
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os
    import pickle

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        cache_file = os.path.join(tmpdirname, "cache.pkl")

        # Define a function to be cached
        @cache(cache_file)
        def add(a, b):
            return a + b

        # Test the cache
        assert add(1, 2) == 3  # Function should be executed
        assert add(1, 2) == 3  # Function should not be executed, result should be loaded from cache

        # Verify cache file exists and contains the correct data
        with open(cache_file, "rb") as f:
            assert pickle.load(f) == 3

        # Test with different arguments
        assert add(2, 3) == 5  # Function should be executed again
        assert add(2, 3) == 5  # Function should not be executed, result should be loaded from cache

        # Verify cache file has been updated with new data
        with open(cache_file, "rb") as f:
            assert pickle.load(f) == 5

        # Test with no cache path
        @cache(None)
        def subtract(a, b):
            return a - b

        assert subtract(5, 3) == 2  # Function should be executed
        assert subtract(5, 3) == 2  # Function should be executed again, no cache

        # Test with verbose False
        @cache(cache_file, verbose=False)
        def multiply(a, b):
            return a * b

        assert multiply(2, 3) == 6  # Function should be executed
        assert multiply(2, 3) == 6  # Function should not be executed, result should be loaded from cache

        # Verify cache file has been updated with new data
        with open(cache_file, "rb") as f:
            assert pickle.load(f) == 6


# LLM-generated content at query #4
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir).joinpath("file1.txt").touch()
        Path(tmpdir).joinpath("file2.txt").touch()
        Path(tmpdir).joinpath("subdir").mkdir()
        Path(tmpdir).joinpath("subdir/file3.txt").touch()

        entries = list(scandir(tmpdir))
        assert len(entries) == 3
        assert Path(tmpdir).joinpath("file1.txt") in entries
        assert Path(tmpdir).joinpath("file2.txt") in entries
        assert Path(tmpdir).joinpath("subdir") in entries

        entries = list(scandir(str(tmpdir)))
        assert len(entries) == 3
        assert str(Path(tmpdir).joinpath("file1.txt")) in entries
        assert str(Path(tmpdir).joinpath("file2.txt")) in entries
        assert str(Path(tmpdir).joinpath("subdir")) in entries


# LLM-generated content at query #5
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import shutil
    import os

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        src_dir = os.path.join(tmpdir, "src")
        dst_dir = os.path.join(tmpdir, "dst")

        # Create source directory structure
        os.makedirs(src_dir)
        os.makedirs(os.path.join(src_dir, "subdir"))
        with open(os.path.join(src_dir, "file1.txt"), "w") as f:
            f.write("file1 content")
        with open(os.path.join(src_dir, "subdir", "file2.txt"), "w") as f:
            f.write("file2 content")

        # Test copying without overwrite
        copy_tree(src_dir, dst_dir, overwrite=False)
        assert os.path.exists(os.path.join(dst_dir, "file1.txt"))
        assert os.path.exists(os.path.join(dst_dir, "subdir", "file2.txt"))

        # Modify file in source
        with open(os.path.join(src_dir, "file1.txt"), "w") as f:
            f.write("modified content")

        # Test copying without overwrite (should not overwrite)
        copy_tree(src_dir, dst_dir, overwrite=False)
        with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
            assert f.read() == "file1 content"

        # Test copying with overwrite
        copy_tree(src_dir, dst_dir, overwrite=True)
        with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
            assert f.read() == "modified content"

        # Clean up
        shutil.rmtree(src_dir)
        shutil.rmtree(dst_dir)


# LLM-generated content at query #6
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import os
    import pathlib

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some files and directories
        os.makedirs(os.path.join(temp_dir, "subdir1"))
        os.makedirs(os.path.join(temp_dir, "subdir2"))
        with open(os.path.join(temp_dir, "file1.txt"), "w") as f:
            f.write("test")
        with open(os.path.join(temp_dir, "file2.txt"), "w") as f:
            f.write("test")

        # Test with str path
        entries = list(scandir(temp_dir))
        assert len(entries) == 4
        assert any("subdir1" in str(p) for p in entries)
        assert any("subdir2" in str(p) for p in entries)
        assert any("file1.txt" in str(p) for p in entries)
        assert any("file2.txt" in str(p) for p in entries)

        # Test with Path path
        entries = list(scandir(pathlib.Path(temp_dir)))
        assert len(entries) == 4
        assert any("subdir1" in str(p) for p in entries)
        assert any("subdir2" in str(p) for p in entries)
        assert any("file1.txt" in str(p) for p in entries)
        assert any("file2.txt" in str(p) for p in entries)


# LLM-generated content at query #7
#--------------------------

# Unit test for function cache
def test_cache():
    @cache("test_cache.pkl", verbose=False)
    def expensive_computation():
        import time
        time.sleep(2)
        return 42

    result1 = expensive_computation()
    assert result1 == 42

    result2 = expensive_computation()
    assert result2 == 42

    os.remove("test_cache.pkl")


# LLM-generated content at query #8
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import time
    import pickle

    def dummy_function(x):
        time.sleep(1)
        return x * 2

    # Test with cache file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        cache_path = tmp_file.name
        cached_function = cache(cache_path)(dummy_function)

        # First call should take time
        start_time = time.time()
        assert cached_function(2) == 4
        assert time.time() - start_time >= 1

        # Second call should be instant
        start_time = time.time()
        assert cached_function(2) == 4
        assert time.time() - start_time < 1

        # Modify cache file and test again
        with open(cache_path, "wb") as f:
            pickle.dump(8, f)
        start_time = time.time()
        assert cached_function(2) == 8
        assert time.time() - start_time < 1

    # Test without cache file
    uncached_function = cache(None)(dummy_function)
    start_time = time.time()
    assert uncached_function(2) == 4
    assert time.time() - start_time >= 1
    start_time = time.time()
    assert uncached_function(2) == 4
    assert time.time() - start_time >= 1


# LLM-generated content at query #9
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import pytest
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        (tmpdir_path / "file1").touch()
        (tmpdir_path / "file2").touch()
        (tmpdir_path / "dir1").mkdir()
        (tmpdir_path / "dir1" / "file3").touch()
        (tmpdir_path / "dir2").mkdir()

        paths = list(scandir(tmpdir_path))
        assert len(paths) == 4
        assert set(paths) == {tmpdir_path / "file1", tmpdir_path / "file2", tmpdir_path / "dir1", tmpdir_path / "dir2"}

        paths = list(scandir(str(tmpdir_path)))
        assert len(paths) == 4
        assert set(paths) == {str(tmpdir_path / "file1"), str(tmpdir_path / "file2"), str(tmpdir_path / "dir1"), str(tmpdir_path / "dir2")}

        with pytest.raises(FileNotFoundError):
            list(scandir(tmpdir_path / "nonexistent"))


# LLM-generated content at query #10
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import os
    import pathlib

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files and directories
        os.makedirs(os.path.join(tmpdir, "subdir1"))
        os.makedirs(os.path.join(tmpdir, "subdir2"))
        with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
            f.write("test")
        with open(os.path.join(tmpdir, "file2.txt"), "w") as f:
            f.write("test")

        # Test with str path
        entries = list(scandir(tmpdir))
        assert len(entries) == 4
        assert any("subdir1" in str(p) for p in entries)
        assert any("subdir2" in str(p) for p in entries)
        assert any("file1.txt" in str(p) for p in entries)
        assert any("file2.txt" in str(p) for p in entries)

        # Test with Path path
        entries = list(scandir(pathlib.Path(tmpdir)))
        assert len(entries) == 4
        assert any("subdir1" in str(p) for p in entries)
        assert any("subdir2" in str(p) for p in entries)
        assert any("file1.txt" in str(p) for p in entries)
        assert any("file2.txt" in str(p) for p in entries)


# LLM-generated content at query #11
#--------------------------

# Unit test for function scandir
def test_scandir():
    # Create a temporary directory and files for testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files and directories inside the temporary directory
        os.makedirs(os.path.join(tmpdir, "subdir1"))
        os.makedirs(os.path.join(tmpdir, "subdir2"))
        with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
            f.write("test content")
        with open(os.path.join(tmpdir, "file2.txt"), "w") as f:
            f.write("test content")
        
        # List of expected paths
        expected_paths = [
            os.path.join(tmpdir, "subdir1"),
            os.path.join(tmpdir, "subdir2"),
            os.path.join(tmpdir, "file1.txt"),
            os.path.join(tmpdir, "file2.txt")
        ]
        
        # Collect paths from scandir
        paths = list(scandir(tmpdir))
        
        # Sort both lists to ensure order does not matter
        expected_paths.sort()
        paths.sort()
        
        # Assert that the paths match
        assert paths == expected_paths, f"Expected {expected_paths}, but got {paths}"


# LLM-generated content at query #12
#--------------------------

# Unit test for function cache
def test_cache():
    def factorial(n):
        if n == 1:
            return 1
        else:
            return n * factorial(n - 1)

    factorial = cache("test_cache.pkl")(factorial)
    assert factorial(5) == 120
    assert factorial(5) == 120
    assert factorial(6) == 720
    assert factorial(6) == 720
    os.remove("test_cache.pkl")


# LLM-generated content at query #13
#--------------------------

# Unit test for function readable_size
def test_readable_size():
    assert readable_size(1024) == "1.00K"
    assert readable_size(1024 * 1024) == "1.00M"
    assert readable_size(1024 * 1024 * 1024) == "1.00G"
    assert readable_size(1024 * 1024 * 1024 * 1024) == "1.00T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.00P"
    assert readable_size(0) == "0.00"
    assert readable_size(1) == "1.00"
    assert readable_size(1023) == "1023.00"
    assert readable_size(1025) == "1.00K"
    assert readable_size(1024 * 1024 - 1) == "1024.00K"
    assert readable_size(1024 * 1024 + 1) == "1.00M"
    assert readable_size(1024 * 1024 * 1024 - 1) == "1024.00M"
    assert readable_size(1024 * 1024 * 1024 + 1) == "1.00G"
    assert readable_size(1024 * 1024 * 1024 * 1024 - 1) == "1024.00G"
    assert readable_size(1024 * 1024 * 1024 * 1024 + 1) == "1.00T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 - 1) == "1024.00T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024 + 1) == "1.00P"



# LLM-generated content at query #14
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import shutil
    import os

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as src_dir:
        with tempfile.TemporaryDirectory() as dst_dir:
            # Create a file in the source directory
            src_file = os.path.join(src_dir, "test.txt")
            with open(src_file, "w") as f:
                f.write("test")

            # Create a subdirectory in the source directory
            src_subdir = os.path.join(src_dir, "subdir")
            os.makedirs(src_subdir)

            # Create a file in the subdirectory
            src_subfile = os.path.join(src_subdir, "subtest.txt")
            with open(src_subfile, "w") as f:
                f.write("subtest")

            # Copy the source directory to the destination directory
            copy_tree(src_dir, dst_dir)

            # Check that the files were copied
            dst_file = os.path.join(dst_dir, "test.txt")
            assert os.path.exists(dst_file)
            with open(dst_file, "r") as f:
                assert f.read() == "test"

            dst_subdir = os.path.join(dst_dir, "subdir")
            assert os.path.exists(dst_subdir)

            dst_subfile = os.path.join(dst_subdir, "subtest.txt")
            assert os.path.exists(dst_subfile)
            with open(dst_subfile, "r") as f:
                assert f.read() == "subtest"

            # Test overwrite
            with open(src_file, "w") as f:
                f.write("test2")

            copy_tree(src_dir, dst_dir, overwrite=True)

            with open(dst_file, "r") as f:
                assert f.read() == "test2"

            # Test no overwrite
            with open(src_file, "w") as f:
                f.write("test3")

            copy_tree(src_dir, dst_dir, overwrite=False)

            with open(dst_file, "r") as f:
                assert f.read() == "test2"

    print("test_copy_tree passed")


# LLM-generated content at query #15
#--------------------------

# Unit test for function readable_size
def test_readable_size():
    assert readable_size(1024) == "1.00K"
    assert readable_size(1024 * 1024) == "1.00M"
    assert readable_size(1024 * 1024 * 1024) == "1.00G"
    assert readable_size(1024 * 1024 * 1024 * 1024) == "1.00T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.00P"
    assert readable_size(1024, n_digits=0) == "1K"
    assert readable_size(1024, n_digits=1) == "1.0K"
    assert readable_size(1024, n_digits=3) == "1.000K"



# LLM-generated content at query #16
#--------------------------

# Unit test for function scandir
def test_scandir():
    # Create a temporary directory and some files
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a few files and directories
        with open(os.path.join(tmpdirname, "file1.txt"), "w") as f:
            f.write("file1")
        os.mkdir(os.path.join(tmpdirname, "dir1"))
        with open(os.path.join(tmpdirname, "dir1", "file2.txt"), "w") as f:
            f.write("file2")
        
        # Test scandir with Path
        paths = list(scandir(Path(tmpdirname)))
        assert len(paths) == 2
        assert Path(os.path.join(tmpdirname, "file1.txt")) in paths
        assert Path(os.path.join(tmpdirname, "dir1")) in paths
        
        # Test scandir with str
        paths = list(scandir(tmpdirname))
        assert len(paths) == 2
        assert os.path.join(tmpdirname, "file1.txt") in paths
        assert os.path.join(tmpdirname, "dir1") in paths

        # Test scandir on nested directory
        paths = list(scandir(Path(os.path.join(tmpdirname, "dir1"))))
        assert len(paths) == 1
        assert Path(os.path.join(tmpdirname, "dir1", "file2.txt")) in paths

        # Test scandir on nested directory with str
        paths = list(scandir(os.path.join(tmpdirname, "dir1")))
        assert len(paths) == 1
        assert os.path.join(tmpdirname, "dir1", "file2.txt") in paths



# LLM-generated content at query #17
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import os

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files and subdirectories
        os.makedirs(os.path.join(tmpdir, "subdir1"))
        os.makedirs(os.path.join(tmpdir, "subdir2"))
        with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
            f.write("test")
        with open(os.path.join(tmpdir, "file2.txt"), "w") as f:
            f.write("test")

        # Test with Path object
        path_obj = Path(tmpdir)
        entries = list(scandir(path_obj))
        assert len(entries) == 4
        assert any(entry.name == "subdir1" for entry in entries)
        assert any(entry.name == "subdir2" for entry in entries)
        assert any(entry.name == "file1.txt" for entry in entries)
        assert any(entry.name == "file2.txt" for entry in entries)

        # Test with string path
        str_path = tmpdir
        entries = list(scandir(str_path))
        assert len(entries) == 4
        assert any(os.path.basename(entry) == "subdir1" for entry in entries)
        assert any(os.path.basename(entry) == "subdir2" for entry in entries)
        assert any(os.path.basename(entry) == "file1.txt" for entry in entries)
        assert any(os.path.basename(entry) == "file2.txt" for entry in entries)


# LLM-generated content at query #18
#--------------------------

# Unit test for function scandir
def test_scandir(): 
    path = "test_dir"
    os.makedirs(path, exist_ok=True)
    file1 = os.path.join(path, "file1.txt")
    file2 = os.path.join(path, "file2.txt")
    sub_dir = os.path.join(path, "sub_dir")
    os.makedirs(sub_dir, exist_ok=True)
    with open(file1, "w") as f:
        f.write("test")
    with open(file2, "w") as f:
        f.write("test")
    files = list(scandir(path))
    assert len(files) == 3
    assert str(files[0]) == file1
    assert str(files[1]) == file2
    assert str(files[2]) == sub_dir
    shutil.rmtree(path)


# LLM-generated content at query #19
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import os
    import pathlib

    # Test with str path
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some files and directories
        file1 = os.path.join(temp_dir, "file1.txt")
        file2 = os.path.join(temp_dir, "file2.txt")
        dir1 = os.path.join(temp_dir, "dir1")
        os.makedirs(dir1)
        with open(file1, "w") as f:
            f.write("test")
        with open(file2, "w") as f:
            f.write("test")

        # Test scandir with str path
        entries = list(scandir(temp_dir))
        assert len(entries) == 3
        assert file1 in entries
        assert file2 in entries
        assert dir1 in entries

    # Test with Path path
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        # Create some files and directories
        file1 = temp_path / "file1.txt"
        file2 = temp_path / "file2.txt"
        dir1 = temp_path / "dir1"
        os.makedirs(dir1)
        with open(file1, "w") as f:
            f.write("test")
        with open(file2, "w") as f:
            f.write("test")

        # Test scandir with Path path
        entries = list(scandir(temp_path))
        assert len(entries) == 3
        assert file1 in entries
        assert file2 in entries
        assert dir1 in entries


# LLM-generated content at query #20
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import os

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        src_dir = os.path.join(temp_dir, "src")
        dst_dir = os.path.join(temp_dir, "dst")

        # Create source directory and add some files
        os.makedirs(src_dir)
        with open(os.path.join(src_dir, "file1.txt"), "w") as f:
            f.write("file1")
        with open(os.path.join(src_dir, "file2.txt"), "w") as f:
            f.write("file2")
        os.makedirs(os.path.join(src_dir, "subdir"))
        with open(os.path.join(src_dir, "subdir", "file3.txt"), "w") as f:
            f.write("file3")

        # Copy contents of src_dir to dst_dir
        copy_tree(src_dir, dst_dir)

        # Check if files were copied correctly
        assert os.path.exists(os.path.join(dst_dir, "file1.txt"))
        assert os.path.exists(os.path.join(dst_dir, "file2.txt"))
        assert os.path.exists(os.path.join(dst_dir, "subdir", "file3.txt"))

        # Check if contents of files are correct
        with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
            assert f.read() == "file1"
        with open(os.path.join(dst_dir, "file2.txt"), "r") as f:
            assert f.read() == "file2"
        with open(os.path.join(dst_dir, "subdir", "file3.txt"), "r") as f:
            assert f.read() == "file3"

        # Test overwrite
        with open(os.path.join(src_dir, "file1.txt"), "w") as f:
            f.write("new_file1")
        copy_tree(src_dir, dst_dir, overwrite=True)
        with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
            assert f.read() == "new_file1"

        # Test no overwrite
        with open(os.path.join(src_dir, "file1.txt"), "w") as f:
            f.write("another_new_file1")
        copy_tree(src_dir, dst_dir, overwrite=False)
        with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
            assert f.read() == "new_file1"


# LLM-generated content at query #21
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files and directories
        os.makedirs(os.path.join(tmpdir, 'dir1'))
        os.makedirs(os.path.join(tmpdir, 'dir2'))
        with open(os.path.join(tmpdir, 'file1.txt'), 'w') as f:
            f.write('test')
        with open(os.path.join(tmpdir, 'file2.txt'), 'w') as f:
            f.write('test')
        
        # Test scandir with Path type
        paths = list(scandir(Path(tmpdir)))
        assert len(paths) == 4
        assert any(str(path).endswith('dir1') for path in paths)
        assert any(str(path).endswith('dir2') for path in paths)
        assert any(str(path).endswith('file1.txt') for path in paths)
        assert any(str(path).endswith('file2.txt') for path in paths)
        
        # Test scandir with str type
        paths = list(scandir(tmpdir))
        assert len(paths) == 4
        assert any(path.endswith('dir1') for path in paths)
        assert any(path.endswith('dir2') for path in paths)
        assert any(path.endswith('file1.txt') for path in paths)
        assert any(path.endswith('file2.txt') for path in paths)


# LLM-generated content at query #22
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os

    # Test with a simple function
    @cache("test_cache.pkl")
    def simple_func():
        return {"key": "value"}

    # First call should execute the function
    result = simple_func()
    assert result == {"key": "value"}
    assert os.path.exists("test_cache.pkl")

    # Second call should load from cache
    result = simple_func()
    assert result == {"key": "value"}

    # Clean up
    os.remove("test_cache.pkl")

    # Test with None path (no caching)
    @cache(None)
    def no_cache_func():
        return {"key": "no_cache"}

    result = no_cache_func()
    assert result == {"key": "no_cache"}
    assert not os.path.exists("None")

    # Test with verbose=False
    @cache("test_cache_quiet.pkl", verbose=False)
    def quiet_func():
        return {"key": "quiet"}

    result = quiet_func()
    assert result == {"key": "quiet"}
    assert os.path.exists("test_cache_quiet.pkl")

    # Clean up
    os.remove("test_cache_quiet.pkl")

    # Test with a custom name
    @cache("test_cache_named.pkl", name="custom")
    def named_func():
        return {"key": "named"}

    result = named_func()
    assert result == {"key": "named"}
    assert os.path.exists("test_cache_named.pkl")

    # Clean up
    os.remove("test_cache_named.pkl")


# LLM-generated content at query #23
#--------------------------

# Unit test for function cache
def test_cache():
    @cache("test_cache.pkl")
    def test_func():
        return {"key": "value"}
    
    # First call should save the cache
    result = test_func()
    assert result == {"key": "value"}
    
    # Second call should load from cache
    result = test_func()
    assert result == {"key": "value"}
    
    # Clean up
    os.remove("test_cache.pkl")


# LLM-generated content at query #24
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os
    import pickle

    # Test function to be cached
    def test_func(x):
        return x * 2

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "cache.pkl")

        # Test cache decorator
        cached_func = cache(cache_file)(test_func)

        # First call, function should be executed and result cached
        result = cached_func(10)
        assert result == 20
        assert os.path.exists(cache_file)

        # Second call, function should not be executed, result should be loaded from cache
        result = cached_func(10)
        assert result == 20

        # Test with a different argument, function should be executed and result cached
        result = cached_func(5)
        assert result == 10

        # Test cache decorator with verbose=False
        cached_func = cache(cache_file, verbose=False)(test_func)
        result = cached_func(10)
        assert result == 20

        # Test cache decorator with path=None
        cached_func = cache(None)(test_func)
        result = cached_func(10)
        assert result == 20
        assert not os.path.exists(cache_file)

        # Test cache decorator with name
        cached_func = cache(cache_file, name="test")(test_func)
        result = cached_func(10)
        assert result == 20


# LLM-generated content at query #25
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import shutil
    import os

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as src_dir:
        # Create a subdirectory and a file in the source directory
        os.makedirs(os.path.join(src_dir, "subdir"))
        with open(os.path.join(src_dir, "subdir", "file.txt"), "w") as f:
            f.write("test content")

        # Create a destination directory
        with tempfile.TemporaryDirectory() as dst_dir:
            # Test copying without overwrite
            copy_tree(src_dir, dst_dir)
            assert os.path.exists(os.path.join(dst_dir, "subdir", "file.txt"))

            # Modify the file in the source directory
            with open(os.path.join(src_dir, "subdir", "file.txt"), "w") as f:
                f.write("modified content")

            # Test copying with overwrite
            copy_tree(src_dir, dst_dir, overwrite=True)
            with open(os.path.join(dst_dir, "subdir", "file.txt"), "r") as f:
                assert f.read() == "modified content"

            # Test copying to an existing directory
            copy_tree(src_dir, dst_dir)
            assert os.path.exists(os.path.join(dst_dir, "subdir", "file.txt"))


# LLM-generated content at query #26
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    with tempfile.TemporaryDirectory() as src_dir:
        with tempfile.TemporaryDirectory() as dst_dir:
            # Create a test file in the source directory
            test_file_path = os.path.join(src_dir, 'test.txt')
            with open(test_file_path, 'w') as f:
                f.write('test content')
            # Create a test subdirectory in the source directory
            test_subdir_path = os.path.join(src_dir, 'test_subdir')
            os.makedirs(test_subdir_path)
            # Create a test file in the subdirectory
            test_subdir_file_path = os.path.join(test_subdir_path, 'subdir_test.txt')
            with open(test_subdir_file_path, 'w') as f:
                f.write('subdir test content')
            # Copy the contents of the source directory to the destination directory
            copy_tree(src_dir, dst_dir)
            # Check if the test file was copied
            copied_test_file_path = os.path.join(dst_dir, 'test.txt')
            assert os.path.exists(copied_test_file_path)
            with open(copied_test_file_path, 'r') as f:
                assert f.read() == 'test content'
            # Check if the test subdirectory was copied
            copied_test_subdir_path = os.path.join(dst_dir, 'test_subdir')
            assert os.path.exists(copied_test_subdir_path)
            # Check if the test file in the subdirectory was copied
            copied_test_subdir_file_path = os.path.join(copied_test_subdir_path, 'subdir_test.txt')
            assert os.path.exists(copied_test_subdir_file_path)
            with open(copied_test_subdir_file_path, 'r') as f:
                assert f.read() == 'subdir test content'
            # Create a test file in the destination directory with the same name as the one in the source directory
            with open(copied_test_file_path, 'w') as f:
                f.write('overwritten content')
            # Copy the contents of the source directory to the destination directory with overwrite=False
            copy_tree(src_dir, dst_dir, overwrite=False)
            # Check if the test file was not overwritten
            with open(copied_test_file_path, 'r') as f:
                assert f.read() == 'overwritten content'
            # Copy the contents of the source directory to the destination directory with overwrite=True
            copy_tree(src_dir, dst_dir, overwrite=True)
            # Check if the test file was overwritten
            with open(copied_test_file_path, 'r') as f:
                assert f.read() == 'test content'


# LLM-generated content at query #27
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    # Create a temporary source directory
    src_dir = Path("test_src")
    src_dir.mkdir(exist_ok=True)
    (src_dir / "file1.txt").write_text("Hello, World!")
    (src_dir / "subdir").mkdir()
    (src_dir / "subdir" / "file2.txt").write_text("Another file")

    # Create a temporary destination directory
    dst_dir = Path("test_dst")
    dst_dir.mkdir(exist_ok=True)

    # Copy the source directory to the destination directory
    copy_tree(src_dir, dst_dir)

    # Verify that the files were copied correctly
    assert (dst_dir / "file1.txt").read_text() == "Hello, World!"
    assert (dst_dir / "subdir" / "file2.txt").read_text() == "Another file"

    # Clean up the temporary directories
    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)



# LLM-generated content at query #28
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files and directories
        tmpdir = Path(tmpdir)
        (tmpdir / "file1").touch()
        (tmpdir / "file2").touch()
        (tmpdir / "dir1").mkdir()
        (tmpdir / "dir2").mkdir()
        
        # Test scandir
        paths = list(scandir(tmpdir))
        assert len(paths) == 4
        assert any(p.name == "file1" for p in paths)
        assert any(p.name == "file2" for p in paths)
        assert any(p.name == "dir1" for p in paths)
        assert any(p.name == "dir2" for p in paths)
        
        # Test with string path
        paths = list(scandir(str(tmpdir)))
        assert len(paths) == 4
        assert any(os.path.basename(p) == "file1" for p in paths)
        assert any(os.path.basename(p) == "file2" for p in paths)
        assert any(os.path.basename(p) == "dir1" for p in paths)
        assert any(os.path.basename(p) == "dir2" for p in paths)



# LLM-generated content at query #29
#--------------------------

# Unit test for function scandir
def test_scandir():
    # Create a temporary directory
    import tempfile
    import shutil
    tmpdir = tempfile.mkdtemp()

    # Create some files and directories within the temporary directory
    open(os.path.join(tmpdir, 'file1.txt'), 'w').close()
    open(os.path.join(tmpdir, 'file2.txt'), 'w').close()
    os.mkdir(os.path.join(tmpdir, 'dir1'))
    open(os.path.join(tmpdir, 'dir1', 'file3.txt'), 'w').close()

    # Test scandir with Path objects
    paths = list(scandir(Path(tmpdir)))
    assert len(paths) == 3
    assert any('file1.txt' in str(p) for p in paths)
    assert any('file2.txt' in str(p) for p in paths)
    assert any('dir1' in str(p) for p in paths)

    # Test scandir with string paths
    paths = list(scandir(tmpdir))
    assert len(paths) == 3
    assert any('file1.txt' in p for p in paths)
    assert any('file2.txt' in p for p in paths)
    assert any('dir1' in p for p in paths)

    # Clean up the temporary directory
    shutil.rmtree(tmpdir)


# LLM-generated content at query #30
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import pathlib

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create some files and directories in the temporary directory
        (pathlib.Path(tmpdirname) / "file1.txt").touch()
        (pathlib.Path(tmpdirname) / "file2.txt").touch()
        (pathlib.Path(tmpdirname) / "subdir").mkdir()

        # Test scandir with Path input
        paths_from_path = list(scandir(pathlib.Path(tmpdirname)))
        assert len(paths_from_path) == 3
        assert any(p.name == "file1.txt" for p in paths_from_path)
        assert any(p.name == "file2.txt" for p in paths_from_path)
        assert any(p.name == "subdir" for p in paths_from_path)

        # Test scandir with str input
        paths_from_str = list(scandir(tmpdirname))
        assert len(paths_from_str) == 3
        assert any(p.endswith("file1.txt") for p in paths_from_str)
        assert any(p.endswith("file2.txt") for p in paths_from_str)
        assert any(p.endswith("subdir") for p in paths_from_str)


# LLM-generated content at query #31
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        # Create files and subdirectories
        (tmpdir / "file1.txt").touch()
        (tmpdir / "file2.txt").touch()
        subdir = tmpdir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").touch()

        # Test scandir with Path input
        paths = list(scandir(tmpdir))
        assert len(paths) == 3
        assert any(p.name == "file1.txt" for p in paths)
        assert any(p.name == "file2.txt" for p in paths)
        assert any(p.name == "subdir" for p in paths)

        # Test scandir with str input
        paths = list(scandir(str(tmpdir)))
        assert len(paths) == 3
        assert any(Path(p).name == "file1.txt" for p in paths)
        assert any(Path(p).name == "file2.txt" for p in paths)
        assert any(Path(p).name == "subdir" for p in paths)

        # Test scandir on subdirectory
        paths = list(scandir(subdir))
        assert len(paths) == 1
        assert paths[0].name == "file3.txt"



# LLM-generated content at query #32
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import shutil
    import os

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        src_dir = os.path.join(tmp_dir, "src")
        dst_dir = os.path.join(tmp_dir, "dst")

        # Create source directory structure
        os.makedirs(src_dir)
        os.makedirs(os.path.join(src_dir, "subdir"))
        with open(os.path.join(src_dir, "file1.txt"), "w") as f:
            f.write("file1")
        with open(os.path.join(src_dir, "subdir", "file2.txt"), "w") as f:
            f.write("file2")

        # Test copying without overwrite
        copy_tree(src_dir, dst_dir)
        assert os.path.exists(os.path.join(dst_dir, "file1.txt"))
        assert os.path.exists(os.path.join(dst_dir, "subdir", "file2.txt"))

        # Modify source file
        with open(os.path.join(src_dir, "file1.txt"), "w") as f:
            f.write("modified")

        # Test copying without overwrite (should not overwrite)
        copy_tree(src_dir, dst_dir)
        with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
            assert f.read() == "file1"

        # Test copying with overwrite
        copy_tree(src_dir, dst_dir, overwrite=True)
        with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
            assert f.read() == "modified"

        # Clean up
        shutil.rmtree(src_dir)
        shutil.rmtree(dst_dir)


# LLM-generated content at query #33
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import filecmp

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        src_dir = os.path.join(tmpdir, "src")
        dst_dir = os.path.join(tmpdir, "dst")

        # Create source directory structure
        os.makedirs(src_dir)
        os.makedirs(os.path.join(src_dir, "subdir"))
        with open(os.path.join(src_dir, "file1.txt"), "w") as f:
            f.write("test file 1")
        with open(os.path.join(src_dir, "subdir", "file2.txt"), "w") as f:
            f.write("test file 2")

        # Test copying without overwrite
        copy_tree(src_dir, dst_dir)
        assert os.path.exists(dst_dir)
        assert os.path.exists(os.path.join(dst_dir, "file1.txt"))
        assert os.path.exists(os.path.join(dst_dir, "subdir", "file2.txt"))
        assert filecmp.cmp(os.path.join(src_dir, "file1.txt"), os.path.join(dst_dir, "file1.txt"))
        assert filecmp.cmp(os.path.join(src_dir, "subdir", "file2.txt"), os.path.join(dst_dir, "subdir", "file2.txt"))

        # Modify source file
        with open(os.path.join(src_dir, "file1.txt"), "w") as f:
            f.write("modified file 1")

        # Test copying with overwrite=False (should not overwrite)
        copy_tree(src_dir, dst_dir, overwrite=False)
        with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
            assert f.read() == "test file 1"

        # Test copying with overwrite=True (should overwrite)
        copy_tree(src_dir, dst_dir, overwrite=True)
        with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
            assert f.read() == "modified file 1"

        # Clean up
        shutil.rmtree(src_dir)
        shutil.rmtree(dst_dir)


# LLM-generated content at query #34
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import os
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files and directories in the temporary directory
        os.makedirs(os.path.join(tmpdir, "subdir1"))
        os.makedirs(os.path.join(tmpdir, "subdir2"))
        with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
            f.write("test")
        with open(os.path.join(tmpdir, "file2.txt"), "w") as f:
            f.write("test")

        # Test with Path input
        path = Path(tmpdir)
        entries = list(scandir(path))
        assert len(entries) == 4
        assert any(entry.name == "subdir1" for entry in entries)
        assert any(entry.name == "subdir2" for entry in entries)
        assert any(entry.name == "file1.txt" for entry in entries)
        assert any(entry.name == "file2.txt" for entry in entries)

        # Test with str input
        entries = list(scandir(tmpdir))
        assert len(entries) == 4
        assert any(os.path.basename(entry) == "subdir1" for entry in entries)
        assert any(os.path.basename(entry) == "subdir2" for entry in entries)
        assert any(os.path.basename(entry) == "file1.txt" for entry in entries)
        assert any(os.path.basename(entry) == "file2.txt" for entry in entries)


# LLM-generated content at query #35
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import itertools
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        file_names = ["test1.txt", "test2.txt", "test3.txt"]
        dir_names = ["dir1", "dir2", "dir3"]
        for file_name in file_names:
            (tmpdir_path / file_name).touch()
        for dir_name in dir_names:
            (tmpdir_path / dir_name).mkdir()

        # Test with Path input
        scandir_path = scandir(tmpdir_path)
        paths = list(scandir_path)
        assert len(paths) == len(file_names + dir_names)
        for path in paths:
            assert isinstance(path, Path)
            assert path.exists()

        # Test with str input
        scandir_str = scandir(str(tmpdir_path))
        paths = list(scandir_str)
        assert len(paths) == len(file_names + dir_names)
        for path in paths:
            assert isinstance(path, str)
            assert os.path.exists(path)



# LLM-generated content at query #36
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import shutil

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as src_dir:
        with tempfile.TemporaryDirectory() as dst_dir:
            # Create a file in the source directory
            src_file = os.path.join(src_dir, "test_file.txt")
            with open(src_file, "w") as f:
                f.write("test content")

            # Create a subdirectory in the source directory
            src_subdir = os.path.join(src_dir, "subdir")
            os.makedirs(src_subdir)
            src_subfile = os.path.join(src_subdir, "subfile.txt")
            with open(src_subfile, "w") as f:
                f.write("sub content")

            # Copy the contents of the source directory to the destination directory
            copy_tree(src_dir, dst_dir)

            # Check that the file was copied
            dst_file = os.path.join(dst_dir, "test_file.txt")
            assert os.path.exists(dst_file)
            with open(dst_file, "r") as f:
                assert f.read() == "test content"

            # Check that the subdirectory was copied
            dst_subdir = os.path.join(dst_dir, "subdir")
            assert os.path.exists(dst_subdir)
            dst_subfile = os.path.join(dst_subdir, "subfile.txt")
            assert os.path.exists(dst_subfile)
            with open(dst_subfile, "r") as f:
                assert f.read() == "sub content"

            # Create a file in the destination directory that has the same name as a file in the source directory
            dst_conflict_file = os.path.join(dst_dir, "test_file.txt")
            with open(dst_conflict_file, "w") as f:
                f.write("conflict content")

            # Copy the contents of the source directory to the destination directory again, with overwrite=False
            copy_tree(src_dir, dst_dir, overwrite=False)

            # Check that the file was not overwritten
            with open(dst_conflict_file, "r") as f:
                assert f.read() == "conflict content"

            # Copy the contents of the source directory to the destination directory again, with overwrite=True
            copy_tree(src_dir, dst_dir, overwrite=True)

            # Check that the file was overwritten
            with open(dst_conflict_file, "r") as f:
                assert f.read() == "test content"


# LLM-generated content at query #37
#--------------------------

# Unit test for function scandir
def test_scandir():
    test_dir = Path("test_dir")
    test_dir.mkdir(exist_ok=True)
    (test_dir / "file1.txt").touch()
    (test_dir / "file2.txt").touch()
    (test_dir / "subdir").mkdir()
    (test_dir / "subdir" / "file3.txt").touch()

    try:
        paths = list(scandir(test_dir))
        assert len(paths) == 3, "scandir did not return the correct number of paths"
        assert any("file1.txt" in str(p) for p in paths), "file1.txt not found"
        assert any("file2.txt" in str(p) for p in paths), "file2.txt not found"
        assert any("subdir" in str(p) for p in paths), "subdir not found"

        subdir_path = next(p for p in paths if "subdir" in str(p))
        subdir_files = list(scandir(subdir_path))
        assert len(subdir_files) == 1, "scandir did not return the correct number of subdir paths"
        assert "file3.txt" in str(subdir_files[0]), "file3.txt not found in subdir"

    finally:
        shutil.rmtree(test_dir)


# LLM-generated content at query #38
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os
    import pickle

    @cache('test_cache.pkl', verbose=False)
    def test_function():
        return [1, 2, 3]

    result = test_function()
    assert result == [1, 2, 3]
    assert os.path.exists('test_cache.pkl')

    with open('test_cache.pkl', 'rb') as f:
        cached_result = pickle.load(f)
    assert cached_result == [1, 2, 3]

    os.remove('test_cache.pkl')


# LLM-generated content at query #39
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    from .log import log

    def dummy_function(x):
        return x * 2

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "cache.pkl")
        cached_function = cache(cache_path)(dummy_function)
        assert cached_function(2) == 4
        assert os.path.exists(cache_path)
        with open(cache_path, "rb") as f:
            assert pickle.load(f) == 4
        assert cached_function(3) == 4
        assert cached_function(3) == 4

        cached_function = cache(None)(dummy_function)
        assert cached_function(3) == 6
        assert not os.path.exists(cache_path)


# LLM-generated content at query #40
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os
    import pickle

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "test_cache.pkl")

        # Test with a simple function
        @cache(cache_file)
        def test_func(x):
            return x * 2

        # First call should compute and cache
        assert test_func(2) == 4
        assert os.path.exists(cache_file)

        # Second call should load from cache
        with open(cache_file, "rb") as f:
            cached_value = pickle.load(f)
        assert cached_value == 4

        # Test with verbose=False
        @cache(cache_file, verbose=False)
        def test_func_quiet(x):
            return x * 3

        assert test_func_quiet(3) == 9

        # Test with no cache
        @cache(None)
        def test_func_no_cache(x):
            return x * 4

        assert test_func_no_cache(4) == 16
        assert not os.path.exists(os.path.join(tmpdir, "nonexistent.pkl"))

        # Test with name parameter
        @cache(cache_file, name="custom_name")
        def test_func_named(x):
            return x * 5

        assert test_func_named(5) == 25

    print("All cache tests passed.")

test_cache()


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function scandir
def test_scandir():
    # Create a temporary directory with some files and subdirectories
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files and subdirectories
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        file1 = os.path.join(tmpdir, "file1.txt")
        file2 = os.path.join(subdir, "file2.txt")
        with open(file1, "w") as f:
            f.write("test")
        with open(file2, "w") as f:
            f.write("test")

        # Test scandir with str path
        paths = list(scandir(tmpdir))
        assert len(paths) == 2
        assert file1 in paths
        assert subdir in paths

        # Test scandir with Path path
        paths = list(scandir(Path(tmpdir)))
        assert len(paths) == 2
        assert Path(file1) in paths
        assert Path(subdir) in paths

        # Test scandir on empty directory
        empty_dir = os.path.join(tmpdir, "empty")
        os.makedirs(empty_dir)
        assert len(list(scandir(empty_dir))) == 0


# LLM-generated content at query #2
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import os
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files and subdirectories
        Path(tmpdir, "file1.txt").touch()
        Path(tmpdir, "file2.txt").touch()
        Path(tmpdir, "subdir1").mkdir()
        Path(tmpdir, "subdir1", "file3.txt").touch()
        Path(tmpdir, "subdir2").mkdir()

        # Test with Path input
        paths = list(scandir(Path(tmpdir)))
        assert len(paths) == 4
        assert any(str(p).endswith("file1.txt") for p in paths)
        assert any(str(p).endswith("file2.txt") for p in paths)
        assert any(str(p).endswith("subdir1") for p in paths)
        assert any(str(p).endswith("subdir2") for p in paths)

        # Test with str input
        paths = list(scandir(tmpdir))
        assert len(paths) == 4
        assert any(p.endswith("file1.txt") for p in paths)
        assert any(p.endswith("file2.txt") for p in paths)
        assert any(p.endswith("subdir1") for p in paths)
        assert any(p.endswith("subdir2") for p in paths)

    print("test_scandir passed")


# LLM-generated content at query #3
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os

    # Test with a simple function
    @cache("test_cache.pkl")
    def simple_func():
        return {"key": "value"}

    # First call should execute the function and save to cache
    result = simple_func()
    assert result == {"key": "value"}
    assert os.path.exists("test_cache.pkl")

    # Second call should load from cache
    result = simple_func()
    assert result == {"key": "value"}

    # Clean up
    os.remove("test_cache.pkl")

    # Test with None path (no caching)
    @cache(None)
    def no_cache_func():
        return {"key": "value"}

    # Should always execute the function
    result = no_cache_func()
    assert result == {"key": "value"}
    assert not os.path.exists("None")

    # Test with verbose=False
    @cache("test_cache_quiet.pkl", verbose=False)
    def quiet_func():
        return {"key": "value"}

    result = quiet_func()
    assert result == {"key": "value"}
    assert os.path.exists("test_cache_quiet.pkl")

    # Clean up
    os.remove("test_cache_quiet.pkl")

    # Test with a custom name
    @cache("test_cache_named.pkl", name="custom")
    def named_func():
        return {"key": "value"}

    result = named_func()
    assert result == {"key": "value"}
    assert os.path.exists("test_cache_named.pkl")

    # Clean up
    os.remove("test_cache_named.pkl")

    # Test with a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = os.path.join(temp_dir, "temp_cache.pkl")

        @cache(cache_path)
        def temp_func():
            return {"key": "value"}

        result = temp_func()
        assert result == {"key": "value"}
        assert os.path.exists(cache_path)

        # Second call should load from cache
        result = temp_func()
        assert result == {"key": "value"}

    # The temporary directory and cache file should be cleaned up automatically


# LLM-generated content at query #4
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    from pathlib import Path
    from typing import List

    def func(lst: List[int]) -> List[int]:
        return [x + 1 for x in lst]

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_path = Path(tmp_dir) / "cache.pkl"
        cached_func = cache(cache_path)(func)
        lst = [1, 2, 3]
        result = cached_func(lst)
        assert result == [2, 3, 4]
        assert cache_path.exists()
        cached_func(lst)  # Should load from cache
        assert cached_func(lst) == [2, 3, 4]

        # Test with no cache path
        cached_func_no_path = cache(None)(func)
        result = cached_func_no_path(lst)
        assert result == [2, 3, 4]
        assert not (Path(tmp_dir) / "cache.pkl").exists()

        # Test with different name
        cached_func_name = cache(cache_path, name="test")(func)
        cached_func_name(lst)  # Should load from cache
        assert cached_func_name(lst) == [2, 3, 4]


# LLM-generated content at query #5
#--------------------------

# Unit test for function readable_size
def test_readable_size():
    assert readable_size(1024) == "1.00K"
    assert readable_size(1024 * 1024) == "1.00M"
    assert readable_size(1024 * 1024 * 1024) == "1.00G"
    assert readable_size(1024 * 1024 * 1024 * 1024) == "1.00T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.00P"
    assert readable_size(512) == "512.00"
    assert readable_size(512, n_digits=0) == "512"
    assert readable_size(512, n_digits=4) == "512.0000"



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import os
    import pathlib

    # Test with str path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files and directories
        os.makedirs(os.path.join(tmpdir, "dir1"))
        os.makedirs(os.path.join(tmpdir, "dir2"))
        with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
            f.write("test")
        with open(os.path.join(tmpdir, "file2.txt"), "w") as f:
            f.write("test")

        # Test scandir with str path
        entries = list(scandir(tmpdir))
        assert len(entries) == 4
        assert any("dir1" in str(p) for p in entries)
        assert any("dir2" in str(p) for p in entries)
        assert any("file1.txt" in str(p) for p in entries)
        assert any("file2.txt" in str(p) for p in entries)

    # Test with Path path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files and directories
        pathlib.Path(tmpdir).joinpath("dir1").mkdir()
        pathlib.Path(tmpdir).joinpath("dir2").mkdir()
        pathlib.Path(tmpdir).joinpath("file1.txt").write_text("test")
        pathlib.Path(tmpdir).joinpath("file2.txt").write_text("test")

        # Test scandir with Path path
        entries = list(scandir(pathlib.Path(tmpdir)))
        assert len(entries) == 4
        assert any("dir1" in str(p) for p in entries)
        assert any("dir2" in str(p) for p in entries)
        assert any("file1.txt" in str(p) for p in entries)
        assert any("file2.txt" in str(p) for p in entries)


# LLM-generated content at query #2
#--------------------------

# Unit test for function scandir
def test_scandir():
    # Create a temporary directory structure
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create some files and directories
        os.makedirs(os.path.join(tmpdirname, 'dir1'))
        os.makedirs(os.path.join(tmpdirname, 'dir2'))
        with open(os.path.join(tmpdirname, 'file1.txt'), 'w') as f:
            f.write('file1')
        with open(os.path.join(tmpdirname, 'file2.txt'), 'w') as f:
            f.write('file2')
        with open(os.path.join(tmpdirname, 'dir1', 'file3.txt'), 'w') as f:
            f.write('file3')

        # Test scandir with Path input
        paths = list(scandir(Path(tmpdirname)))
        assert len(paths) == 3
        assert Path(os.path.join(tmpdirname, 'dir1')) in paths
        assert Path(os.path.join(tmpdirname, 'dir2')) in paths
        assert Path(os.path.join(tmpdirname, 'file1.txt')) in paths

        # Test scandir with str input
        paths = list(scandir(tmpdirname))
        assert len(paths) == 3
        assert os.path.join(tmpdirname, 'dir1') in paths
        assert os.path.join(tmpdirname, 'dir2') in paths
        assert os.path.join(tmpdirname, 'file1.txt') in paths


# LLM-generated content at query #3
#--------------------------

# Unit test for function cache
def test_cache():
    # Test cache function
    @cache('test_cache.pkl')
    def test_func():
        return 42

    # Ensure the function returns the correct value
    assert test_func() == 42

    # Ensure the cache file exists and contains the correct value
    with open('test_cache.pkl', 'rb') as f:
        assert pickle.load(f) == 42

    # Clean up
    os.remove('test_cache.pkl')


# LLM-generated content at query #4
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os

    @cache(None)
    def func_no_cache():
        return 42

    assert func_no_cache() == 42

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        cache_file_path = temp_file.name

        @cache(cache_file_path)
        def func_with_cache():
            return 42

        assert func_with_cache() == 42
        assert os.path.exists(cache_file_path)

        @cache(cache_file_path)
        def func_with_cache():
            return 43

        assert func_with_cache() == 42
        os.unlink(cache_file_path)



# LLM-generated content at query #5
#--------------------------

# Unit test for function readable_size
def test_readable_size():
    assert readable_size(1024) == "1.00K"
    assert readable_size(1024 * 1024) == "1.00M"
    assert readable_size(1024 * 1024 * 1024) == "1.00G"
    assert readable_size(1024 * 1024 * 1024 * 1024) == "1.00T"
    assert readable_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.00P"
    assert readable_size(512) == "512.00"
    assert readable_size(1536) == "1.50K"
    assert readable_size(1536, n_digits=0) == "2K"
    assert readable_size(1536, n_digits=1) == "1.5K"


# LLM-generated content at query #6
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os

    @cache("test_cache.pkl")
    def test_function():
        return "test"

    assert test_function() == "test"
    assert os.path.exists("test_cache.pkl")
    os.remove("test_cache.pkl")


# LLM-generated content at query #7
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import os
    import shutil
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        src_dir = os.path.join(temp_dir, "src")
        dst_dir = os.path.join(temp_dir, "dst")
        os.makedirs(src_dir)
        os.makedirs(dst_dir)

        # Create files in src_dir
        with open(os.path.join(src_dir, "file1.txt"), "w") as f:
            f.write("file1")
        with open(os.path.join(src_dir, "file2.txt"), "w") as f:
            f.write("file2")
        os.makedirs(os.path.join(src_dir, "subdir"))
        with open(os.path.join(src_dir, "subdir", "file3.txt"), "w") as f:
            f.write("file3")

        # Copy files from src_dir to dst_dir
        copy_tree(src_dir, dst_dir)

        # Check if files are copied correctly
        assert os.path.exists(os.path.join(dst_dir, "file1.txt"))
        assert os.path.exists(os.path.join(dst_dir, "file2.txt"))
        assert os.path.exists(os.path.join(dst_dir, "subdir", "file3.txt"))

        # Check if the contents of the files are correct
        with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
            assert f.read() == "file1"
        with open(os.path.join(dst_dir, "file2.txt"), "r") as f:
            assert f.read() == "file2"
        with open(os.path.join(dst_dir, "subdir", "file3.txt"), "r") as f:
            assert f.read() == "file3"

        # Test overwrite
        with open(os.path.join(src_dir, "file1.txt"), "w") as f:
            f.write("file1_updated")
        copy_tree(src_dir, dst_dir, overwrite=True)
        with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
            assert f.read() == "file1_updated"

        # Test no overwrite
        with open(os.path.join(src_dir, "file2.txt"), "w") as f:
            f.write("file2_updated")
        copy_tree(src_dir, dst_dir, overwrite=False)
        with open(os.path.join(dst_dir, "file2.txt"), "r") as f:
            assert f.read() == "file2"



# LLM-generated content at query #8
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import filecmp

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a source directory and add some files
        src_dir = Path(tmp_dir) / "src"
        src_dir.mkdir()
        (src_dir / "file1.txt").write_text("Hello")
        (src_dir / "file2.txt").write_text("World")
        (src_dir / "subdir").mkdir()
        (src_dir / "subdir" / "file3.txt").write_text("File 3")

        # Create a destination directory
        dst_dir = Path(tmp_dir) / "dst"
        dst_dir.mkdir()

        # Copy the source directory to the destination directory
        copy_tree(src_dir, dst_dir)

        # Verify that the destination directory contains the same files as the source directory
        assert filecmp.cmp(src_dir / "file1.txt", dst_dir / "file1.txt")
        assert filecmp.cmp(src_dir / "file2.txt", dst_dir / "file2.txt")
        assert filecmp.cmp(src_dir / "subdir" / "file3.txt", dst_dir / "subdir" / "file3.txt")

        # Test overwrite behavior
        (src_dir / "file1.txt").write_text("Updated")
        copy_tree(src_dir, dst_dir, overwrite=True)
        assert filecmp.cmp(src_dir / "file1.txt", dst_dir / "file1.txt")

        # Test non-overwrite behavior
        (src_dir / "file1.txt").write_text("New Update")
        copy_tree(src_dir, dst_dir, overwrite=False)
        assert not filecmp.cmp(src_dir / "file1.txt", dst_dir / "file1.txt")

        # Verify that the destination directory contains the same files as the source directory
        assert filecmp.cmp(src_dir / "file2.txt", dst_dir / "file2.txt")
        assert filecmp.cmp(src_dir / "subdir" / "file3.txt", dst_dir / "subdir" / "file3.txt")



# LLM-generated content at query #9
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import shutil
    import os

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as src_dir:
        # Create a subdirectory and a file in the source directory
        os.makedirs(os.path.join(src_dir, "subdir"))
        with open(os.path.join(src_dir, "file1.txt"), "w") as f:
            f.write("test content")
        with open(os.path.join(src_dir, "subdir", "file2.txt"), "w") as f:
            f.write("test content 2")

        # Create a destination directory
        with tempfile.TemporaryDirectory() as dst_dir:
            # Test copying without overwrite
            copy_tree(src_dir, dst_dir)
            assert os.path.exists(os.path.join(dst_dir, "file1.txt"))
            assert os.path.exists(os.path.join(dst_dir, "subdir", "file2.txt"))

            # Modify the source file
            with open(os.path.join(src_dir, "file1.txt"), "w") as f:
                f.write("modified content")

            # Test copying with overwrite
            copy_tree(src_dir, dst_dir, overwrite=True)
            with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
                assert f.read() == "modified content"

            # Test copying without overwrite (should not update the file)
            with open(os.path.join(src_dir, "file1.txt"), "w") as f:
                f.write("new content")
            copy_tree(src_dir, dst_dir, overwrite=False)
            with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
                assert f.read() == "modified content"

    print("test_copy_tree passed")


# LLM-generated content at query #10
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os
    import pickle

    @cache(None)
    def func():
        return 42

    assert func() == 42

    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name

    @cache(path)
    def func2():
        return 42

    assert func2() == 42
    assert os.path.exists(path)
    with open(path, "rb") as f:
        assert pickle.load(f) == 42

    os.remove(path)


# LLM-generated content at query #11
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os
    import pickle
    from .log import log

    # Test basic functionality
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = os.path.join(temp_dir, "cache.pkl")
        @cache(cache_file)
        def add(a, b):
            return a + b

        # First call, should execute function and save to cache
        assert add(1, 2) == 3
        assert os.path.exists(cache_file)

        # Second call, should load from cache
        assert add(1, 2) == 3

    # Test verbose logging
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = os.path.join(temp_dir, "cache.pkl")
        @cache(cache_file, verbose=True)
        def add(a, b):
            return a + b

        # First call, should execute function and save to cache
        assert add(1, 2) == 3
        assert os.path.exists(cache_file)

        # Second call, should load from cache
        assert add(1, 2) == 3

    # Test with name parameter
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = os.path.join(temp_dir, "cache.pkl")
        @cache(cache_file, verbose=True, name="custom_name")
        def add(a, b):
            return a + b

        # First call, should execute function and save to cache
        assert add(1, 2) == 3
        assert os.path.exists(cache_file)

        # Second call, should load from cache
        assert add(1, 2) == 3

    # Test with path=None
    @cache(None)
    def add(a, b):
        return a + b

    assert add(1, 2) == 3

    # Test overwriting cache
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = os.path.join(temp_dir, "cache.pkl")
        @cache(cache_file)
        def add(a, b):
            return a + b

        # First call, should execute function and save to cache
        assert add(1, 2) == 3
        assert os.path.exists(cache_file)

        # Modify cache file
        with open(cache_file, "wb") as f:
            pickle.dump(10, f)

        # Second call, should load from cache
        assert add(1, 2) == 10

    # Test cache file does not exist initially
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = os.path.join(temp_dir, "cache.pkl")
        @cache(cache_file)
        def add(a, b):
            return a + b

        # First call, should execute function and save to cache
        assert add(1, 2) == 3
        assert os.path.exists(cache_file)

        # Remove cache file
        os.remove(cache_file)

        # Second call, should execute function and save to cache
        assert add(1, 2) == 3
        assert os.path.exists(cache_file)


# LLM-generated content at query #12
#--------------------------

# Unit test for function scandir
def test_scandir():
    # Create a temporary directory with some files and subdirectories
    tmp_dir = Path("tmp_dir")
    tmp_dir.mkdir(exist_ok=True)
    (tmp_dir / "file1.txt").touch()
    (tmp_dir / "file2.txt").touch()
    (tmp_dir / "subdir").mkdir()
    (tmp_dir / "subdir" / "file3.txt").touch()

    # Test scandir with Path input
    paths = list(scandir(tmp_dir))
    assert len(paths) == 3
    assert any(path.name == "file1.txt" for path in paths)
    assert any(path.name == "file2.txt" for path in paths)
    assert any(path.name == "subdir" for path in paths)

    # Test scandir with str input
    paths = list(scandir(str(tmp_dir)))
    assert len(paths) == 3
    assert any(Path(path).name == "file1.txt" for path in paths)
    assert any(Path(path).name == "file2.txt" for path in paths)
    assert any(Path(path).name == "subdir" for path in paths)

    # Clean up
    shutil.rmtree(tmp_dir)


# LLM-generated content at query #13
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "test_cache.pkl")

        # Test with a simple function
        @cache(cache_file)
        def simple_func():
            return {"key": "value"}

        # First call - should execute the function and save to cache
        result1 = simple_func()
        assert result1 == {"key": "value"}
        assert os.path.exists(cache_file)

        # Second call - should load from cache
        result2 = simple_func()
        assert result2 == {"key": "value"}

        # Test with a function that takes arguments
        @cache(cache_file)
        def func_with_args(a, b):
            return a + b

        # Should load from cache (same file) and ignore arguments
        result3 = func_with_args(1, 2)
        assert result3 == {"key": "value"}

        # Test with no cache path
        @cache(None)
        def no_cache_func():
            return "no_cache"

        # Should execute each time
        result4 = no_cache_func()
        assert result4 == "no_cache"
        result5 = no_cache_func()
        assert result5 == "no_cache"

        # Test with verbose=False
        @cache(cache_file, verbose=False)
        def non_verbose_func():
            return "non_verbose"

        # Should work without printing
        result6 = non_verbose_func()
        assert result6 == {"key": "value"}

        # Clean up
        os.remove(cache_file)


# LLM-generated content at query #14
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import os
    import pathlib

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files and directories
        os.makedirs(os.path.join(tmpdir, "subdir1"))
        os.makedirs(os.path.join(tmpdir, "subdir2"))
        with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
            f.write("test")
        with open(os.path.join(tmpdir, "file2.txt"), "w") as f:
            f.write("test")

        # Test with str path
        paths = list(scandir(tmpdir))
        assert len(paths) == 4
        assert any("subdir1" in str(p) for p in paths)
        assert any("subdir2" in str(p) for p in paths)
        assert any("file1.txt" in str(p) for p in paths)
        assert any("file2.txt" in str(p) for p in paths)

        # Test with Path path
        paths = list(scandir(pathlib.Path(tmpdir)))
        assert len(paths) == 4
        assert any("subdir1" in str(p) for p in paths)
        assert any("subdir2" in str(p) for p in paths)
        assert any("file1.txt" in str(p) for p in paths)
        assert any("file2.txt" in str(p) for p in paths)


# LLM-generated content at query #15
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    # Create a temporary directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        src = os.path.join(tmpdirname, 'src')
        dst = os.path.join(tmpdirname, 'dst')
        os.makedirs(src)
        os.makedirs(dst)
        # Create a file in the source directory
        with open(os.path.join(src, 'test.txt'), 'w') as f:
            f.write('test')
        # Copy the contents of the source directory to the destination directory
        copy_tree(src, dst)
        # Check that the file was copied
        assert os.path.exists(os.path.join(dst, 'test.txt'))
        # Check that the file contents are the same
        with open(os.path.join(dst, 'test.txt'), 'r') as f:
            assert f.read() == 'test'
        # Check that the destination directory was not overwritten
        with open(os.path.join(dst, 'test.txt'), 'w') as f:
            f.write('test2')
        copy_tree(src, dst, overwrite=False)
        with open(os.path.join(dst, 'test.txt'), 'r') as f:
            assert f.read() == 'test2'
        # Check that the destination directory was overwritten
        copy_tree(src, dst, overwrite=True)
        with open(os.path.join(dst, 'test.txt'), 'r') as f:
            assert f.read() == 'test'


# LLM-generated content at query #16
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os
    from flutes import log
    
    @cache("test_cache_file.pkl", verbose=False)
    def func():
        return 42
    
    assert func() == 42
    assert os.path.exists("test_cache_file.pkl")
    
    @cache("test_cache_file.pkl", verbose=False)
    def func():
        return 43
    
    assert func() == 42
    
    os.remove("test_cache_file.pkl")
    
    @cache(None, verbose=False)
    def func():
        return 44
    
    assert func() == 44
    assert not os.path.exists("test_cache_file.pkl")


# LLM-generated content at query #17
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import filecmp
    import shutil

    # Create a temporary directory
    tmp_dir = tempfile.mkdtemp()
    src_dir = os.path.join(tmp_dir, "src")
    dst_dir = os.path.join(tmp_dir, "dst")
    os.makedirs(src_dir)
    os.makedirs(dst_dir)

    # Create some files and directories in the source directory
    with open(os.path.join(src_dir, "file1.txt"), "w") as f:
        f.write("file1")
    os.makedirs(os.path.join(src_dir, "subdir"))
    with open(os.path.join(src_dir, "subdir", "file2.txt"), "w") as f:
        f.write("file2")

    # Copy the source directory to the destination directory
    copy_tree(src_dir, dst_dir)

    # Check that the destination directory contains the same files and directories as the source directory
    assert filecmp.dircmp(src_dir, dst_dir).diff_files == []

    # Clean up
    shutil.rmtree(tmp_dir)



# LLM-generated content at query #18
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import filecmp

    # Setup
    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()
    sub_dir = os.path.join(src_dir, "subdir")
    os.makedirs(sub_dir)
    file1 = os.path.join(src_dir, "file1.txt")
    file2 = os.path.join(sub_dir, "file2.txt")
    with open(file1, "w") as f:
        f.write("file1")
    with open(file2, "w") as f:
        f.write("file2")

    # Test copy without overwrite
    copy_tree(src_dir, dst_dir)
    assert os.path.exists(os.path.join(dst_dir, "file1.txt"))
    assert os.path.exists(os.path.join(dst_dir, "subdir"))
    assert os.path.exists(os.path.join(dst_dir, "subdir", "file2.txt"))
    assert filecmp.cmp(file1, os.path.join(dst_dir, "file1.txt"))
    assert filecmp.cmp(file2, os.path.join(dst_dir, "subdir", "file2.txt"))

    # Test copy with overwrite
    with open(file1, "w") as f:
        f.write("file1_modified")
    copy_tree(src_dir, dst_dir, overwrite=True)
    assert filecmp.cmp(file1, os.path.join(dst_dir, "file1.txt"))

    # Cleanup
    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)


# LLM-generated content at query #19
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import shutil
    import os

    # Create a temporary directory
    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()

    try:
        # Create some files and directories in the source directory
        os.makedirs(os.path.join(src_dir, "subdir"))
        with open(os.path.join(src_dir, "file1.txt"), "w") as f:
            f.write("test1")
        with open(os.path.join(src_dir, "subdir", "file2.txt"), "w") as f:
            f.write("test2")

        # Test copying without overwrite
        copy_tree(src_dir, dst_dir, overwrite=False)
        assert os.path.exists(os.path.join(dst_dir, "file1.txt"))
        assert os.path.exists(os.path.join(dst_dir, "subdir", "file2.txt"))

        # Modify a file in the source directory
        with open(os.path.join(src_dir, "file1.txt"), "w") as f:
            f.write("modified")

        # Test copying with overwrite
        copy_tree(src_dir, dst_dir, overwrite=True)
        with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
            assert f.read() == "modified"

    finally:
        # Clean up
        shutil.rmtree(src_dir)
        shutil.rmtree(dst_dir)


# LLM-generated content at query #20
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import shutil
    import os

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create some files and directories in the temporary directory
        os.makedirs(os.path.join(temp_dir, "subdir1"))
        os.makedirs(os.path.join(temp_dir, "subdir2"))
        with open(os.path.join(temp_dir, "file1.txt"), "w") as f:
            f.write("test")
        with open(os.path.join(temp_dir, "file2.txt"), "w") as f:
            f.write("test")

        # Test with Path object
        path_obj = Path(temp_dir)
        entries = list(scandir(path_obj))
        assert len(entries) == 4
        assert any(entry.name == "subdir1" for entry in entries)
        assert any(entry.name == "subdir2" for entry in entries)
        assert any(entry.name == "file1.txt" for entry in entries)
        assert any(entry.name == "file2.txt" for entry in entries)

        # Test with string path
        entries = list(scandir(temp_dir))
        assert len(entries) == 4
        assert any(os.path.basename(entry) == "subdir1" for entry in entries)
        assert any(os.path.basename(entry) == "subdir2" for entry in entries)
        assert any(os.path.basename(entry) == "file1.txt" for entry in entries)
        assert any(os.path.basename(entry) == "file2.txt" for entry in entries)

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #21
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import uuid
    tmp_dir = tempfile.mkdtemp()
    dir_name = str(uuid.uuid4())
    dir_path = os.path.join(tmp_dir, dir_name)
    os.mkdir(dir_path)
    file_name = str(uuid.uuid4())
    file_path = os.path.join(dir_path, file_name)
    with open(file_path, 'w') as f:
        f.write('test')
    try:
        assert sorted(scandir(dir_path)) == sorted([file_path])
    finally:
        shutil.rmtree(tmp_dir)


# LLM-generated content at query #22
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os
    import pickle

    # Test with a simple function
    @cache("test_cache.pkl")
    def simple_func():
        return {"key": "value"}

    # Remove the cache file if it exists
    if os.path.exists("test_cache.pkl"):
        os.remove("test_cache.pkl")

    # First call should execute the function and save the result
    result = simple_func()
    assert result == {"key": "value"}
    assert os.path.exists("test_cache.pkl")

    # Second call should load from cache
    result = simple_func()
    assert result == {"key": "value"}

    # Clean up
    os.remove("test_cache.pkl")

    # Test with a function that takes arguments
    @cache("test_cache_args.pkl")
    def func_with_args(a, b):
        return a + b

    # Remove the cache file if it exists
    if os.path.exists("test_cache_args.pkl"):
        os.remove("test_cache_args.pkl")

    # First call should execute the function and save the result
    result = func_with_args(1, 2)
    assert result == 3
    assert os.path.exists("test_cache_args.pkl")

    # Second call should load from cache, but arguments are ignored
    result = func_with_args(10, 20)
    assert result == 3

    # Clean up
    os.remove("test_cache_args.pkl")

    # Test with path=None (no caching)
    @cache(None)
    def no_cache_func():
        return "no_cache"

    result = no_cache_func()
    assert result == "no_cache"
    assert not os.path.exists("None")

    # Test with verbose=False
    @cache("test_cache_verbose.pkl", verbose=False)
    def verbose_func():
        return "verbose"

    # Remove the cache file if it exists
    if os.path.exists("test_cache_verbose.pkl"):
        os.remove("test_cache_verbose.pkl")

    result = verbose_func()
    assert result == "verbose"
    assert os.path.exists("test_cache_verbose.pkl")

    # Clean up
    os.remove("test_cache_verbose.pkl")

    # Test with name parameter
    @cache("test_cache_name.pkl", name="named_cache")
    def named_func():
        return "named"

    # Remove the cache file if it exists
    if os.path.exists("test_cache_name.pkl"):
        os.remove("test_cache_name.pkl")

    result = named_func()
    assert result == "named"
    assert os.path.exists("test_cache_name.pkl")

    # Clean up
    os.remove("test_cache_name.pkl")


# LLM-generated content at query #23
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    # Test copying a tree with overwrite
    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()
    src_sub_dir = tempfile.mkdtemp(dir=src_dir)
    dst_sub_dir = tempfile.mkdtemp(dir=dst_dir)
    src_file = tempfile.NamedTemporaryFile(dir=src_dir, delete=False)
    dst_file = tempfile.NamedTemporaryFile(dir=dst_dir, delete=False)
    src_sub_file = tempfile.NamedTemporaryFile(dir=src_sub_dir, delete=False)
    dst_sub_file = tempfile.NamedTemporaryFile(dir=dst_sub_dir, delete=False)
    try:
        copy_tree(src_dir, dst_dir, overwrite=True)
        assert os.path.exists(dst_dir)
        assert os.path.exists(dst_sub_dir)
        assert os.path.exists(os.path.join(dst_dir, os.path.basename(src_file.name)))
        assert os.path.exists(os.path.join(dst_sub_dir, os.path.basename(src_sub_file.name)))
    finally:
        shutil.rmtree(src_dir)
        shutil.rmtree(dst_dir)

    # Test copying a tree without overwrite
    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()
    src_sub_dir = tempfile.mkdtemp(dir=src_dir)
    dst_sub_dir = tempfile.mkdtemp(dir=dst_dir)
    src_file = tempfile.NamedTemporaryFile(dir=src_dir, delete=False)
    dst_file = tempfile.NamedTemporaryFile(dir=dst_dir, delete=False)
    src_sub_file = tempfile.NamedTemporaryFile(dir=src_sub_dir, delete=False)
    dst_sub_file = tempfile.NamedTemporaryFile(dir=dst_sub_dir, delete=False)
    try:
        copy_tree(src_dir, dst_dir)
        assert os.path.exists(dst_dir)
        assert os.path.exists(dst_sub_dir)
        assert os.path.exists(os.path.join(dst_dir, os.path.basename(src_file.name)))
        assert os.path.exists(os.path.join(dst_sub_dir, os.path.basename(src_sub_file.name)))
    finally:
        shutil.rmtree(src_dir)
        shutil.rmtree(dst_dir)

    # Test copying a tree with empty destination
    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()
    src_sub_dir = tempfile.mkdtemp(dir=src_dir)
    src_file = tempfile.NamedTemporaryFile(dir=src_dir, delete=False)
    src_sub_file = tempfile.NamedTemporaryFile(dir=src_sub_dir, delete=False)
    try:
        shutil.rmtree(dst_dir)
        copy_tree(src_dir, dst_dir)
        assert os.path.exists(dst_dir)
        assert os.path.exists(os.path.join(dst_dir, os.path.basename(src_sub_dir)))
        assert os.path.exists(os.path.join(dst_dir, os.path.basename(src_file.name)))
        assert os.path.exists(os.path.join(dst_dir, os.path.basename(src_sub_dir), os.path.basename(src_sub_file.name)))
    finally:
        shutil.rmtree(src_dir)
        shutil.rmtree(dst_dir)



# LLM-generated content at query #24
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import os
    import pathlib

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files and directories
        os.makedirs(os.path.join(tmpdir, "dir1"))
        os.makedirs(os.path.join(tmpdir, "dir2"))
        with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
            f.write("test")
        with open(os.path.join(tmpdir, "file2.txt"), "w") as f:
            f.write("test")

        # Test with str path
        entries = list(scandir(tmpdir))
        assert len(entries) == 4
        assert any("dir1" in str(e) for e in entries)
        assert any("dir2" in str(e) for e in entries)
        assert any("file1.txt" in str(e) for e in entries)
        assert any("file2.txt" in str(e) for e in entries)

        # Test with Path path
        entries = list(scandir(pathlib.Path(tmpdir)))
        assert len(entries) == 4
        assert any("dir1" in str(e) for e in entries)
        assert any("dir2" in str(e) for e in entries)
        assert any("file1.txt" in str(e) for e in entries)
        assert any("file2.txt" in str(e) for e in entries)


# LLM-generated content at query #25
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a few files and directories
        file1 = os.path.join(temp_dir, "file1.txt")
        file2 = os.path.join(temp_dir, "file2.txt")
        dir1 = os.path.join(temp_dir, "dir1")
        dir2 = os.path.join(temp_dir, "dir2")
        os.makedirs(dir1)
        os.makedirs(dir2)
        with open(file1, "w") as f:
            f.write("test")
        with open(file2, "w") as f:
            f.write("test")

        # Test with Path input
        paths = list(scandir(Path(temp_dir)))
        assert len(paths) == 4
        assert Path(file1) in paths
        assert Path(file2) in paths
        assert Path(dir1) in paths
        assert Path(dir2) in paths

        # Test with str input
        paths = list(scandir(temp_dir))
        assert len(paths) == 4
        assert file1 in paths
        assert file2 in paths
        assert dir1 in paths
        assert dir2 in paths


# LLM-generated content at query #26
#--------------------------

# Unit test for function scandir
def test_scandir():
    # Create a temporary directory with some files
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files and directories
        Path(tmpdir).joinpath("file1.txt").touch()
        Path(tmpdir).joinpath("file2.txt").touch()
        Path(tmpdir).joinpath("subdir").mkdir()
        Path(tmpdir).joinpath("subdir/file3.txt").touch()

        # Test scandir with Path input
        paths = list(scandir(Path(tmpdir)))
        assert len(paths) == 3
        assert any("file1.txt" in str(p) for p in paths)
        assert any("file2.txt" in str(p) for p in paths)
        assert any("subdir" in str(p) for p in paths)

        # Test scandir with str input
        paths = list(scandir(tmpdir))
        assert len(paths) == 3
        assert any("file1.txt" in p for p in paths)
        assert any("file2.txt" in p for p in paths)
        assert any("subdir" in p for p in paths)

        # Test scandir with empty directory
        empty_dir = Path(tmpdir).joinpath("empty")
        empty_dir.mkdir()
        assert len(list(scandir(empty_dir))) == 0

    print("All scandir tests passed!")

if __name__ == "__main__":
    test_scandir()


# LLM-generated content at query #27
#--------------------------

# Unit test for function scandir
def test_scandir():
    # Create a temporary directory with some files and subdirectories
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files and subdirectories
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        file1 = os.path.join(tmpdir, "file1.txt")
        file2 = os.path.join(subdir, "file2.txt")
        with open(file1, "w") as f:
            f.write("test")
        with open(file2, "w") as f:
            f.write("test")

        # Test scandir with str path
        paths = list(scandir(tmpdir))
        assert len(paths) == 2
        assert file1 in paths
        assert subdir in paths

        # Test scandir with Path path
        paths = list(scandir(Path(tmpdir)))
        assert len(paths) == 2
        assert Path(file1) in paths
        assert Path(subdir) in paths

    print("test_scandir passed")


# LLM-generated content at query #28
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import shutil
    import os

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as src_dir:
        # Create a subdirectory and a file in the source directory
        os.makedirs(os.path.join(src_dir, "subdir"))
        with open(os.path.join(src_dir, "file1.txt"), "w") as f:
            f.write("test content")
        with open(os.path.join(src_dir, "subdir", "file2.txt"), "w") as f:
            f.write("test content 2")

        # Create a destination directory
        with tempfile.TemporaryDirectory() as dst_dir:
            # Test copying without overwrite
            copy_tree(src_dir, dst_dir)
            assert os.path.exists(os.path.join(dst_dir, "file1.txt"))
            assert os.path.exists(os.path.join(dst_dir, "subdir", "file2.txt"))

            # Modify a file in the source directory
            with open(os.path.join(src_dir, "file1.txt"), "w") as f:
                f.write("modified content")

            # Test copying with overwrite
            copy_tree(src_dir, dst_dir, overwrite=True)
            with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
                assert f.read() == "modified content"

            # Test copying without overwrite (should not modify the file)
            with open(os.path.join(src_dir, "file1.txt"), "w") as f:
                f.write("new content")
            copy_tree(src_dir, dst_dir)
            with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
                assert f.read() == "modified content"


# LLM-generated content at query #29
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os
    import pickle
    from unittest.mock import patch

    # Test case 1: Cache file exists
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        cache_file = tmp.name
        test_data = {"key": "value"}
        pickle.dump(test_data, tmp)

    @cache(cache_file)
    def func1():
        return {"key": "new_value"}

    result = func1()
    assert result == test_data, "Cache should load existing file"

    os.unlink(cache_file)

    # Test case 2: Cache file doesn't exist
    @cache(cache_file)
    def func2():
        return {"key": "new_value"}

    result = func2()
    assert result == {"key": "new_value"}, "Function should execute when cache doesn't exist"
    assert os.path.exists(cache_file), "Cache file should be created"

    with open(cache_file, "rb") as f:
        cached_data = pickle.load(f)
    assert cached_data == {"key": "new_value"}, "Cache file should contain function output"

    os.unlink(cache_file)

    # Test case 3: No cache path provided
    @cache(None)
    def func3():
        return {"key": "no_cache"}

    result = func3()
    assert result == {"key": "no_cache"}, "Function should execute when no cache path is provided"
    assert not os.path.exists(cache_file), "No cache file should be created"

    # Test case 4: Verbose logging
    with patch('flutes.log.log') as mock_log:
        @cache(cache_file, verbose=True, name="test")
        def func4():
            return {"key": "verbose"}

        # First call - should save
        func4()
        assert mock_log.call_count == 1
        assert "saved" in mock_log.call_args[0][0]

        # Second call - should load
        func4()
        assert mock_log.call_count == 2
        assert "loaded" in mock_log.call_args[0][0]

    os.unlink(cache_file)


# LLM-generated content at query #30
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import shutil
    import os

    # Create a temporary directory
    src_dir = tempfile.mkdtemp()
    dst_dir = tempfile.mkdtemp()

    # Create a file in the source directory
    src_file = os.path.join(src_dir, "test.txt")
    with open(src_file, "w") as f:
        f.write("test")

    # Copy the directory
    copy_tree(src_dir, dst_dir)

    # Check if the file was copied
    dst_file = os.path.join(dst_dir, "test.txt")
    assert os.path.exists(dst_file)

    # Clean up
    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)



# LLM-generated content at query #31
#--------------------------

# Unit test for function scandir
def test_scandir():
    import tempfile
    import os
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files and directories
        os.makedirs(os.path.join(tmpdir, "subdir1"))
        os.makedirs(os.path.join(tmpdir, "subdir2"))
        with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
            f.write("test")
        with open(os.path.join(tmpdir, "file2.txt"), "w") as f:
            f.write("test")

        # Test with Path input
        path = Path(tmpdir)
        entries = list(scandir(path))
        assert len(entries) == 4
        assert any(entry.name == "subdir1" for entry in entries)
        assert any(entry.name == "subdir2" for entry in entries)
        assert any(entry.name == "file1.txt" for entry in entries)
        assert any(entry.name == "file2.txt" for entry in entries)

        # Test with str input
        entries = list(scandir(tmpdir))
        assert len(entries) == 4
        assert any(os.path.basename(entry) == "subdir1" for entry in entries)
        assert any(os.path.basename(entry) == "subdir2" for entry in entries)
        assert any(os.path.basename(entry) == "file1.txt" for entry in entries)
        assert any(os.path.basename(entry) == "file2.txt" for entry in entries)


# LLM-generated content at query #32
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import shutil

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        src_dir = os.path.join(temp_dir, "src")
        dst_dir = os.path.join(temp_dir, "dst")

        # Create source directory and add some files
        os.makedirs(src_dir)
        with open(os.path.join(src_dir, "file1.txt"), "w") as f:
            f.write("file1")
        os.makedirs(os.path.join(src_dir, "subdir"))
        with open(os.path.join(src_dir, "subdir", "file2.txt"), "w") as f:
            f.write("file2")

        # Test copying without overwrite
        copy_tree(src_dir, dst_dir)
        assert os.path.exists(os.path.join(dst_dir, "file1.txt"))
        assert os.path.exists(os.path.join(dst_dir, "subdir", "file2.txt"))

        # Modify source files
        with open(os.path.join(src_dir, "file1.txt"), "w") as f:
            f.write("file1_modified")
        with open(os.path.join(src_dir, "subdir", "file2.txt"), "w") as f:
            f.write("file2_modified")

        # Test copying without overwrite (should not overwrite)
        copy_tree(src_dir, dst_dir)
        with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
            assert f.read() == "file1"
        with open(os.path.join(dst_dir, "subdir", "file2.txt"), "r") as f:
            assert f.read() == "file2"

        # Test copying with overwrite (should overwrite)
        copy_tree(src_dir, dst_dir, overwrite=True)
        with open(os.path.join(dst_dir, "file1.txt"), "r") as f:
            assert f.read() == "file1_modified"
        with open(os.path.join(dst_dir, "subdir", "file2.txt"), "r") as f:
            assert f.read() == "file2_modified"

        # Clean up
        shutil.rmtree(dst_dir)


# LLM-generated content at query #33
#--------------------------

# Unit test for function scandir
def test_scandir():
    # Create a temporary directory structure
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a file within the directory
        with open(os.path.join(tmpdirname, 'file.txt'), 'w') as f:
            f.write('test')
        # Create a subdirectory within the directory
        os.makedirs(os.path.join(tmpdirname, 'subdir'))
        # Test scandir with Path object
        paths = list(scandir(Path(tmpdirname)))
        assert len(paths) == 2
        # Test scandir with string
        paths = list(scandir(tmpdirname))
        assert len(paths) == 2



# LLM-generated content at query #34
#--------------------------

# Unit test for function copy_tree
def test_copy_tree():
    import tempfile
    import filecmp

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        src_dir = os.path.join(tmp_dir, 'src')
        dst_dir = os.path.join(tmp_dir, 'dst')

        # Create source directory and files
        os.makedirs(src_dir)
        with open(os.path.join(src_dir, 'file1.txt'), 'w') as f:
            f.write('file1')
        os.makedirs(os.path.join(src_dir, 'subdir'))
        with open(os.path.join(src_dir, 'subdir', 'file2.txt'), 'w') as f:
            f.write('file2')

        # Perform the copy
        copy_tree(src_dir, dst_dir)

        # Verify that the directories are identical
        assert filecmp.dircmp(src_dir, dst_dir).diff_files == []



# LLM-generated content at query #35
#--------------------------

# Unit test for function cache
def test_cache():
    import tempfile
    import os
    import pickle

    # Test with path=None
    @cache(None)
    def func1():
        return 42

    assert func1() == 42

    # Test with path
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name

    try:
        @cache(path)
        def func2():
            return 42

        assert func2() == 42
        assert os.path.exists(path)
        with open(path, "rb") as f:
            assert pickle.load(f) == 42

        # Test cache loading
        @cache(path)
        def func3():
            return 43

        assert func3() == 42
    finally:
        if os.path.exists(path):
            os.unlink(path)


