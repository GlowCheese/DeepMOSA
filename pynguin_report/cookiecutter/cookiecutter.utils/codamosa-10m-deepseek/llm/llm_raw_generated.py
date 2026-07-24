####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        with work_in(temp_dir):
            assert os.getcwd() == os.path.realpath(temp_dir)
        assert os.getcwd() == original_dir

    with work_in(None):
        assert os.getcwd() == original_dir


# LLM-generated content at query #2
#--------------------------

# Unit test for function work_in
def test_work_in():
    current_directory = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == current_directory


# LLM-generated content at query #3
#--------------------------

# Unit test for function simple_filter
def test_simple_filter():
    """Test the simple_filter decorator."""
    @simple_filter
    def test_filter(value):
        return value.upper()

    assert test_filter.__name__ == 'test_filter'
    assert isinstance(test_filter, type)
    assert issubclass(test_filter, Extension)


# LLM-generated content at query #4
#--------------------------

# Unit test for function work_in
def test_work_in():
    # Save the current working directory
    original_dir = os.getcwd()
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use the work_in context manager to change the working directory
        with work_in(temp_dir):
            # Verify that the working directory has been changed
            assert os.getcwd() == os.path.abspath(temp_dir)
        
        # Verify that the working directory has been restored
        assert os.getcwd() == original_dir


# LLM-generated content at query #5
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    new_dir = tempfile.mkdtemp()
    
    with work_in(new_dir):
        assert os.getcwd() == new_dir
    
    assert os.getcwd() == original_dir
    
    # Cleanup
    shutil.rmtree(new_dir)


# LLM-generated content at query #6
#--------------------------

# Unit test for function work_in
def test_work_in():
    import os
    from tempfile import mkdtemp
    from shutil import rmtree

    original_dir = os.getcwd()
    temp_dir = mkdtemp()

    try:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


# LLM-generated content at query #7
#--------------------------

# Unit test for function work_in
def test_work_in():
    current_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tempdir:
        with work_in(tempdir):
            assert os.getcwd() == tempdir
        assert os.getcwd() == current_dir


# LLM-generated content at query #8
#--------------------------

# Unit test for function work_in
def test_work_in():
    import os
    original_dir = os.getcwd()
    test_dir = "/tmp"
    
    with work_in(test_dir):
        assert os.getcwd() == test_dir
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #9
#--------------------------

# Unit test for function work_in
def test_work_in():
    import tempfile
    import os

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        original_dir = os.getcwd()
        
        # Use the work_in context manager
        with work_in(tmpdirname):
            assert os.getcwd() == tmpdirname
        
        # Ensure the directory is restored after exiting the context
        assert os.getcwd() == original_dir



# LLM-generated content at query #10
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        with work_in(temp_dir):
            assert os.getcwd() == os.path.realpath(temp_dir)
        assert os.getcwd() == original_dir


# LLM-generated content at query #11
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == original_dir
    finally:
        os.rmdir(temp_dir)


# LLM-generated content at query #12
#--------------------------

# Unit test for function work_in
def test_work_in():        
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        with work_in(temp_dir):
            assert os.getcwd() == os.path.abspath(temp_dir)
        assert os.getcwd() == original_dir


# LLM-generated content at query #13
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        with work_in(temp_dir):
            assert os.getcwd() == os.path.abspath(temp_dir)
        assert os.getcwd() == original_dir


# LLM-generated content at query #14
#--------------------------

# Unit test for function work_in
def test_work_in():
    import os
    original_dir = os.getcwd()
    with work_in('/tmp'):
        assert os.getcwd() == '/tmp'
    assert os.getcwd() == original_dir



# LLM-generated content at query #15
#--------------------------

# Unit test for function work_in
def test_work_in():
    # Test changing to a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = os.getcwd()
        with work_in(tmpdir):
            assert os.getcwd() == os.path.realpath(tmpdir)
        assert os.getcwd() == original_dir

    # Test not changing directory
    original_dir = os.getcwd()
    with work_in(None):
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir


# LLM-generated content at query #16
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        with work_in(temp_dir):
            assert os.getcwd() == os.path.realpath(temp_dir)
        assert os.getcwd() == original_dir


# LLM-generated content at query #17
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    try:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(temp_dir)



# LLM-generated content at query #18
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tempdir:
        with work_in(tempdir):
            assert os.getcwd() == tempdir
        assert os.getcwd() == original_dir


# LLM-generated content at query #19
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == original_dir
    finally:
        os.rmdir(temp_dir)


# LLM-generated content at query #20
#--------------------------

# Unit test for function simple_filter
def test_simple_filter():
    @simple_filter
    def custom_filter(value):
        return value.upper()

    env = StrictEnvironment()
    ext = custom_filter(env)
    assert ext.__name__ == "custom_filter"
    assert "custom_filter" in env.filters
    assert env.filters["custom_filter"]("test") == "TEST"


# LLM-generated content at query #21
#--------------------------

# Unit test for function simple_filter
def test_simple_filter():
    """Test the simple_filter decorator."""
    @simple_filter
    def test_filter(value):
        return value.upper()

    env = StrictEnvironment()
    ext = test_filter(env)
    assert ext.__name__ == 'test_filter'
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('test') == 'TEST'


# LLM-generated content at query #22
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    with work_in('/tmp'):
        assert os.getcwd() == '/tmp'
    assert os.getcwd() == original_dir


# LLM-generated content at query #23
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
    assert os.getcwd() == original_dir


# LLM-generated content at query #24
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        with work_in(temp_dir):
            assert os.getcwd() == os.path.realpath(temp_dir)
        assert os.getcwd() == original_dir

    # Test with None
    with work_in(None):
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir


# LLM-generated content at query #25
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == original_dir


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()
    
    with work_in(test_dir):
        assert os.getcwd() == test_dir
    
    assert os.getcwd() == original_dir
    
    shutil.rmtree(test_dir)


# LLM-generated content at query #2
#--------------------------

# Unit test for function simple_filter
def test_simple_filter():
    """Test the simple_filter function."""
    def test_func(value):
        return value.upper()

    filter_ext = simple_filter(test_func)
    assert filter_ext.__name__ == 'test_func'
    assert issubclass(filter_ext, Extension)


# LLM-generated content at query #3
#--------------------------

# Unit test for function simple_filter
def test_simple_filter():
    """Test the simple_filter function."""
    @simple_filter
    def test_filter(value):
        return value.upper()

    env = StrictEnvironment()
    ext = test_filter(env)
    assert ext.__name__ == 'test_filter'
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('test') == 'TEST'


# LLM-generated content at query #4
#--------------------------

# Unit test for function work_in
def test_work_in():
    # Save the current working directory
    original_dir = os.getcwd()

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use the work_in context manager
        with work_in(temp_dir):
            # Verify that the current working directory has changed
            assert os.getcwd() == temp_dir

        # Verify that the current working directory has been restored
        assert os.getcwd() == original_dir

    # Verify that the temporary directory has been cleaned up
    assert not os.path.exists(temp_dir)


# LLM-generated content at query #5
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == original_dir
    finally:
        os.rmdir(temp_dir)


# LLM-generated content at query #6
#--------------------------

# Unit test for function work_in
def test_work_in():
    import os
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    try:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == original_dir
    finally:
        os.rmdir(temp_dir)


# LLM-generated content at query #7
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
    assert os.getcwd() == original_dir


# LLM-generated content at query #8
#--------------------------

# Unit test for function simple_filter
def test_simple_filter():
    @simple_filter
    def custom_filter(value):
        return value.upper()

    environment = StrictEnvironment()
    extension = custom_filter(environment)
    assert 'custom_filter' in environment.filters
    assert environment.filters['custom_filter']('test') == 'TEST'


# LLM-generated content at query #9
#--------------------------

# Unit test for function simple_filter
def test_simple_filter():
    @simple_filter
    def custom_filter(value):
        return value.upper()

    env = StrictEnvironment()
    ext = custom_filter(env)
    assert ext.__name__ == 'custom_filter'
    assert env.filters.get('custom_filter') is not None


# LLM-generated content at query #10
#--------------------------

# Unit test for function work_in
def test_work_in():
    # Save the current directory
    original_dir = os.getcwd()
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use the work_in context manager to change to the temporary directory
        with work_in(temp_dir):
            # Assert that the current directory is the temporary directory
            assert os.getcwd() == temp_dir
        
        # Assert that the current directory is back to the original directory
        assert os.getcwd() == original_dir



# LLM-generated content at query #11
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == original_dir
        
        # Test with None
        with work_in(None):
            assert os.getcwd() == original_dir
    finally:
        os.rmdir(temp_dir)


# LLM-generated content at query #12
#--------------------------

# Unit test for function simple_filter
def test_simple_filter():
    """Test the simple_filter function."""

    @simple_filter
    def test_filter(value):
        return value.upper()

    env = StrictEnvironment()
    ext = test_filter(env)
    assert ext.__name__ == 'test_filter'
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('test') == 'TEST'


# LLM-generated content at query #13
#--------------------------

# Unit test for function simple_filter
def test_simple_filter():
    @simple_filter
    def add_one(number):
        return number + 1

    assert add_one.__name__ == 'add_one'


# LLM-generated content at query #14
#--------------------------

# Unit test for function work_in
def test_work_in():
    """Test the work_in context manager."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(temp_dir)


# LLM-generated content at query #15
#--------------------------

# Unit test for function simple_filter
def test_simple_filter():
    @simple_filter
    def custom_filter(value):
        return value.upper()

    env = StrictEnvironment()
    extension = custom_filter(env)
    assert extension.__name__ == "custom_filter"
    assert "custom_filter" in env.filters
    assert env.filters["custom_filter"]("test") == "TEST"


# LLM-generated content at query #16
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        with work_in(temp_dir):
            assert os.getcwd() == os.path.abspath(temp_dir)
        assert os.getcwd() == original_dir

    with work_in(None):
        assert os.getcwd() == original_dir


# LLM-generated content at query #17
#--------------------------

# Unit test for function work_in
def test_work_in():
    # Test that the function changes the working directory and reverts it back
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()

    with work_in(temp_dir):
        assert os.getcwd() == temp_dir

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(temp_dir)


# LLM-generated content at query #18
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == original_dir



# LLM-generated content at query #19
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(temp_dir)


# LLM-generated content at query #20
#--------------------------

# Unit test for function work_in
def test_work_in():
    # Save the current working directory
    original_dir = os.getcwd()
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use the work_in context manager to change to the temporary directory
        with work_in(temp_dir):
            # Verify that the current working directory has changed
            assert os.getcwd() == temp_dir
        # Verify that the current working directory has been restored
        assert os.getcwd() == original_dir


# LLM-generated content at query #21
#--------------------------

# Unit test for function simple_filter
def test_simple_filter():
    """Test the simple_filter decorator."""
    @simple_filter
    def test_filter(value):
        return value.upper()

    env = StrictEnvironment()
    ext = test_filter(env)
    assert ext.__name__ == 'test_filter'
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('test') == 'TEST'


# LLM-generated content at query #22
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        with work_in(temp_dir):
            assert os.getcwd() == os.path.realpath(temp_dir)
        assert os.getcwd() == original_dir


# LLM-generated content at query #23
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    with work_in(tempfile.mkdtemp()):
        assert os.getcwd() != original_dir
    assert os.getcwd() == original_dir


# LLM-generated content at query #24
#--------------------------

# Unit test for function simple_filter
def test_simple_filter():
    @simple_filter
    def custom_filter(value):
        return value.upper()

    assert custom_filter.__name__ == "custom_filter"


# LLM-generated content at query #25
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()
    try:
        with work_in(test_dir):
            assert os.getcwd() == test_dir
        assert os.getcwd() == original_dir
    finally:
        rmtree(test_dir)


# LLM-generated content at query #26
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == original_dir
    finally:
        os.rmdir(temp_dir)


# LLM-generated content at query #27
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    try:
        with work_in(test_dir):
            assert os.getcwd() == test_dir
        assert os.getcwd() == original_dir
    finally:
        os.rmdir(test_dir)


# LLM-generated content at query #28
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == original_dir



# LLM-generated content at query #29
#--------------------------

# Unit test for function simple_filter
def test_simple_filter():
    """Test the simple_filter decorator."""
    @simple_filter
    def custom_filter(value):
        return value.upper()

    env = StrictEnvironment()
    ext = custom_filter(env)

    assert ext.__name__ == 'custom_filter'
    assert 'custom_filter' in env.filters
    assert env.filters['custom_filter']('test') == 'TEST'


# LLM-generated content at query #30
#--------------------------

# Unit test for function simple_filter
def test_simple_filter():
    """Test the simple_filter decorator."""
    @simple_filter
    def test_filter(value):
        return value.upper()

    env = StrictEnvironment()
    ext = test_filter(env)
    assert ext.__name__ == 'test_filter'
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('test') == 'TEST'


# LLM-generated content at query #31
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    with work_in('/tmp'):
        assert os.getcwd() == '/tmp'
    assert os.getcwd() == original_dir



# LLM-generated content at query #32
#--------------------------

# Unit test for function work_in
def test_work_in():
    current_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    try:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == current_dir
    finally:
        shutil.rmtree(temp_dir)


# LLM-generated content at query #33
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        with work_in(temp_dir):
            assert os.getcwd() == os.path.realpath(temp_dir)
    assert os.getcwd() == original_dir


# LLM-generated content at query #34
#--------------------------

# Unit test for function work_in
def test_work_in():  # type: ignore[no-untyped-def]
    """Test the work_in context manager."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    try:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == original_dir
    finally:
        os.rmdir(temp_dir)


# LLM-generated content at query #35
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == original_dir
    finally:
        os.rmdir(temp_dir)


# LLM-generated content at query #36
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    try:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(temp_dir)


# LLM-generated content at query #37
#--------------------------

# Unit test for function simple_filter
def test_simple_filter():
    """Test the simple_filter decorator."""
    @simple_filter
    def test_filter(value):
        return value.upper()

    env = StrictEnvironment()
    ext = test_filter(env)
    assert ext.__name__ == 'test_filter'
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('test') == 'TEST'


# LLM-generated content at query #38
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
        assert os.getcwd() == original_dir
    finally:
        rmtree(test_dir)


# LLM-generated content at query #39
#--------------------------

# Unit test for function simple_filter
def test_simple_filter():
    """Test the simple_filter function."""
    @simple_filter
    def custom_filter(value):
        return value.upper()

    env = StrictEnvironment()
    extension = custom_filter(env)
    assert extension.__name__ == 'custom_filter'
    assert 'custom_filter' in env.filters
    assert env.filters['custom_filter']('test') == 'TEST'


# LLM-generated content at query #40
#--------------------------

# Unit test for function work_in
def test_work_in():
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        assert os.getcwd() == original_dir
    finally:
        os.rmdir(temp_dir)


