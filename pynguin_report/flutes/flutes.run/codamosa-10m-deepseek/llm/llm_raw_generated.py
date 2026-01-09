####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test case 1: Successful command execution
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"Hello, World!\n"
    
    # Test case 2: Command with error
    result = run_command(["ls", "nonexistent_file"], return_output=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    
    # Test case 3: Command with timeout
    result = run_command(["sleep", "2"], timeout=1, return_output=True, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None
    
    # Test case 4: Command with custom environment variable
    env = {"MY_VAR": "my_value"}
    result = run_command(["printenv", "MY_VAR"], env=env, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"my_value\n"
    
    # Test case 5: Command with custom working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.decode().strip() == tmpdir
    
    # Test case 6: Command with verbose output
    import io
    import sys
    captured_output = io.StringIO()
    sys.stdout = captured_output
    result = run_command(["echo", "Verbose output"], verbose=True)
    sys.stdout = sys.__stdout__
    assert "Verbose output" in captured_output.getvalue()
    
    print("All test cases passed!")

# Run the unit tests
if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #2
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Basic command execution
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test 2: Command with error
    result = run_command(["ls", "nonexistentfile"], ignore_errors=True)
    assert result.return_code != 0

    # Test 3: Timeout
    import time
    result = run_command(["sleep", "2"], timeout=1, ignore_errors=True)
    assert result.return_code == -32768

    # Test 4: Environment variable
    result = run_command(["printenv", "MY_VAR"], env={"MY_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output

    # Test 5: Working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test 6: Verbose mode (just ensure it doesn't crash)
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #3
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import io
    import traceback

    def capture_output(func):
        captured = io.StringIO()
        sys.stderr = captured
        try:
            func()
        finally:
            sys.stderr = sys.__stderr__
        return captured.getvalue()

    # Test CalledProcessError
    err = subprocess.CalledProcessError(1, "ls", output=b"file1\nfile2")
    wrapped = error_wrapper(err)
    output = capture_output(lambda: traceback.print_exception(type(wrapped), wrapped, None))
    assert "Captured output:" in output
    assert "file1" in output
    assert "file2" in output

    # Test TimeoutExpired
    err = subprocess.TimeoutExpired("sleep 10", 5, output=b"still running")
    wrapped = error_wrapper(err)
    output = capture_output(lambda: traceback.print_exception(type(wrapped), wrapped, None))
    assert "Captured output:" in output
    assert "still running" in output

    # Test other exception
    err = ValueError("test")
    wrapped = error_wrapper(err)
    assert wrapped is err

    print("All tests passed.")


if __name__ == "__main__":  # pragma: no cover
    test_error_wrapper()


# LLM-generated content at query #4
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Simple command that should succeed
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

    # Test 2: Command that should fail
    result = run_command(["ls", "nonexistentfile"], ignore_errors=True, return_output=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test 3: Command with timeout
    try:
        run_command(["sleep", "10"], timeout=0.1, ignore_errors=False)
    except subprocess.TimeoutExpired:
        pass  # Expected
    else:
        assert False, "Expected TimeoutExpired"

    # Test 4: Command with environment variable
    env = {"MYVAR": "myvalue"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"MYVAR=myvalue" in result.captured_output

    # Test 5: Command with cwd
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test 6: Command with shell=True
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test 7: Command with verbose=True (should print output)
    import io
    import sys
    captured_output = io.StringIO()
    sys.stdout = captured_output
    try:
        run_command(["echo", "verbose test"], verbose=True)
    finally:
        sys.stdout = sys.__stdout__
    assert "verbose test" in captured_output.getvalue()

    # Test 8: Command that returns non-zero but ignore_errors=True
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    # Test 9: Command that returns zero and return_output=False
    result = run_command(["true"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test 10: Command that returns zero and return_output=True
    result = run_command(["echo", "output"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"output" in result.captured_output

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #5
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.check_output(["ls", "nonexistent"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped_error)
        print("Test for CalledProcessError passed.")
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped_error)
        print("Test for TimeoutExpired passed.")
    
    # Test with other exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert wrapped_error is e
        print("Test for other exception passed.")



# LLM-generated content at query #6
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Simple command that should succeed
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test 2: Command that should fail
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    # Test 3: Command with timeout
    result = run_command(["sleep", "2"], timeout=1, ignore_errors=True)
    assert result.return_code == -32768  # Special return code for timeout

    # Test 4: Command with environment variable
    result = run_command(["printenv", "MY_VAR"], env={"MY_VAR": "test"}, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

    # Test 5: Command with working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test 6: Command with verbose output
    result = run_command(["echo", "Verbose test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"Verbose test" in result.captured_output

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #7
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test with CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, 'ls', output=b'error output')
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert wrapped_error.output == b'error output'
        assert 'Captured output:' in str(wrapped_error)
    
    # Test with TimeoutExpired
    try:
        raise subprocess.TimeoutExpired('ls', 10, output=b'timeout output')
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert wrapped_error.output == b'timeout output'
        assert 'Captured output:' in str(wrapped_error)
    
    # Test with other exception
    try:
        raise ValueError('test error')
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == 'test error'
    
    print("All tests passed!")



# LLM-generated content at query #8
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test with CalledProcessError
    try:
        subprocess.run(['false'], check=True)
    except subprocess.CalledProcessError as e:
        e = error_wrapper(e)
        assert 'Captured output:' in str(e)
        assert 'No output was generated.' not in str(e)
    # Test with TimeoutExpired
    try:
        subprocess.run(['sleep', '2'], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        e = error_wrapper(e)
        assert 'Captured output:' in str(e)
        assert 'No output was generated.' not in str(e)
    # Test with other exception
    try:
        raise ValueError('test')
    except ValueError as e:
        e = error_wrapper(e)
        assert 'test' in str(e)
        assert 'Captured output:' not in str(e)
    print('All tests passed.')

if __name__ == '__main__':
    test_error_wrapper()


# LLM-generated content at query #9
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Run a simple command that should succeed
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test 2: Run a command that should fail
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    # Test 3: Test timeout
    try:
        run_command(["sleep", "2"], timeout=1)
    except subprocess.TimeoutExpired:
        pass  # Expected
    else:
        assert False, "Expected TimeoutExpired"

    # Test 4: Test verbose mode (should not raise)
    run_command(["echo", "test"], verbose=True)

    # Test 5: Test with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command(["env"], env=env, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test 6: Test with cwd
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #10
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        raise subprocess.CalledProcessError(returncode=1, cmd="test", output=b"test output")
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with TimeoutExpired
    try:
        raise subprocess.TimeoutExpired(cmd="test", timeout=1, output=b"test output")
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with other exception
    try:
        raise ValueError("test")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "test"



# LLM-generated content at query #11
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: run a simple command that should succeed
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"
    print("Test 1 passed")

    # Test 2: run a command that should fail
    result = run_command(["ls", "nonexistent_file"], ignore_errors=True, return_output=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    print("Test 2 passed")

    # Test 3: run a command with a timeout that should expire
    try:
        run_command(["sleep", "2"], timeout=1)
    except subprocess.TimeoutExpired:
        print("Test 3 passed")
    else:
        print("Test 3 failed")

    # Test 4: run a command with verbose output
    result = run_command(["echo", "verbose test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"verbose test\n"
    print("Test 4 passed")

    # Test 5: run a command with custom environment variables
    result = run_command(["env"], env={"CUSTOM_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"CUSTOM_VAR=test_value" in result.captured_output
    print("Test 5 passed")

    # Test 6: run a command with a custom working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output
    print("Test 6 passed")

    # Test 7: run a command that returns output without errors
    result = run_command(["echo", "output test"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"output test\n"
    print("Test 7 passed")

    # Test 8: run a command that returns no output
    result = run_command(["true"])
    assert result.return_code == 0
    assert result.captured_output is None
    print("Test 8 passed")

    # Test 9: run a command with shell=True
    result = run_command("echo shell test", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"shell test\n"
    print("Test 9 passed")

    # Test 10: run a command with ignore_errors=True and a failing command
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    print("Test 10 passed")

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #12
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with other exception
    other_error = ValueError("test")
    wrapped_error = error_wrapper(other_error)
    assert wrapped_error is other_error



# LLM-generated content at query #13
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():


# LLM-generated content at query #14
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Run a simple command that should succeed
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"Hello, World!" in result.captured_output

    # Test 2: Run a command that should fail
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    # Test 3: Test timeout
    result = run_command(["sleep", "2"], timeout=1, ignore_errors=True)
    assert result.return_code == -32768  # Special return code for timeout

    # Test 4: Test with environment variables
    env = {"MY_VAR": "test_value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"MY_VAR=test_value" in result.captured_output

    # Test 5: Test with working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test 6: Test verbose mode (should not raise an error)
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0

    # Test 7: Test shell command
    result = run_command("echo Hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"Hello" in result.captured_output

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #15
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(returncode=1, cmd='ls')
    err.output = b'error: file not found'
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert 'Captured output:' in str(wrapped_err)
    assert 'error: file not found' in str(wrapped_err)
    
    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired(cmd='sleep 10', timeout=1)
    err.output = b'process timed out'
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert 'Captured output:' in str(wrapped_err)
    assert 'process timed out' in str(wrapped_err)
    
    # Test with other exception
    err = ValueError('test')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == 'test'
    
    print('All tests passed')

if __name__ == '__main__':
    test_error_wrapper()


# LLM-generated content at query #16
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.run(['false'], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert 'Captured output:' in str(wrapped_error)
        print("Test for CalledProcessError passed.")
    
    # Test with TimeoutExpired
    try:
        subprocess.run(['sleep', '10'], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert 'Captured output:' in str(wrapped_error)
        print("Test for TimeoutExpired passed.")
    
    # Test with other exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert wrapped_error is e
        print("Test for other exception passed.")



# LLM-generated content at query #17
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():


# LLM-generated content at query #18
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.check_output(["ls", "nonexistent"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with other exception
    other_error = ValueError("test")
    wrapped_error = error_wrapper(other_error)
    assert wrapped_error is other_error



# LLM-generated content at query #19
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Simple command that should succeed
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

    # Test 2: Command that should fail
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    # Test 3: Command with timeout
    result = run_command(["sleep", "2"], timeout=1, ignore_errors=True)
    assert result.return_code == -32768  # Special return code for timeout

    # Test 4: Command with environment variable
    result = run_command(["printenv", "MYVAR"], env={"MYVAR": "test"}, return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"test" in result.captured_output

    # Test 5: Command with working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output is not None
        assert tmpdir.encode() in result.captured_output

    # Test 6: Command with shell=True
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

    # Test 7: Command that produces a lot of output
    result = run_command(["seq", "10000"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert len(result.captured_output) > 0

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #20
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Run a simple command that should succeed
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"Hello, World!" in result.captured_output

    # Test 2: Run a command that should fail
    result = run_command(["ls", "/nonexistent"], ignore_errors=True, return_output=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test 3: Test with shell=True
    result = run_command("echo Hello, World!", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"Hello, World!" in result.captured_output

    # Test 4: Test timeout
    try:
        run_command(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired:
        pass  # Expected
    else:
        assert False, "Expected TimeoutExpired"

    # Test 5: Test environment variable
    result = run_command(["printenv", "MYVAR"], env={"MYVAR": "test"}, return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"test" in result.captured_output

    # Test 6: Test working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output is not None
        assert tmpdir.encode() in result.captured_output

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #21
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.run(['false'], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped_error)
        print("Test for CalledProcessError passed.")

    # Test with TimeoutExpired
    try:
        subprocess.run(['sleep', '10'], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped_error)
        print("Test for TimeoutExpired passed.")

    # Test with other exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert wrapped_error is e
        print("Test for other exception passed.")



# LLM-generated content at query #22
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.check_output(["false"])
    except subprocess.CalledProcessError as e:
        e = error_wrapper(e)
        assert "Captured output:" in str(e)
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        e = error_wrapper(e)
        assert "Captured output:" in str(e)
    
    # Test with other exception
    try:
        raise ValueError("test")
    except ValueError as e:
        e = error_wrapper(e)
        assert str(e) == "test"



# LLM-generated content at query #23
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.check_output(["false"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with other exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert str(wrapped_error) == "Test error"



# LLM-generated content at query #24
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        e = error_wrapper(e)
        assert "Captured output:" in str(e)
        print("Test for CalledProcessError passed.")
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        e = error_wrapper(e)
        assert "Captured output:" in str(e)
        print("Test for TimeoutExpired passed.")
    
    # Test with other exception
    try:
        raise ValueError("Test")
    except ValueError as e:
        e = error_wrapper(e)
        assert str(e) == "Test"
        print("Test for other exception passed.")



# LLM-generated content at query #25
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import io
    import sys

    # Capture stderr
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()

    # Test CalledProcessError
    try:
        subprocess.run(['false'], check=True)
    except subprocess.CalledProcessError as e:
        e = error_wrapper(e)
        print(e)
        assert "Captured output:" in str(e)

    # Test TimeoutExpired
    try:
        subprocess.run(['sleep', '2'], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        e = error_wrapper(e)
        print(e)
        assert "Captured output:" in str(e)

    # Restore stderr
    sys.stderr = old_stderr

    print("All tests passed!")


if __name__ == "__main__":  # pragma: no cover
    test_error_wrapper()


# LLM-generated content at query #26
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped)
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped)
    
    # Test with other exception
    try:
        raise ValueError("test")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert wrapped is e



# LLM-generated content at query #27
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        raise subprocess.CalledProcessError(returncode=1, cmd="test", output=b"test output")
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with TimeoutExpired
    try:
        raise subprocess.TimeoutExpired(cmd="test", timeout=10, output=b"timeout output")
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with other exception
    try:
        raise ValueError("test")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "test"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_error_wrapper()


# LLM-generated content at query #28
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Simple command that should succeed
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test 2: Command that should fail (non-zero return code)
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    # Test 3: Command with timeout that should succeed
    result = run_command(["sleep", "1"], timeout=2, return_output=True)
    assert result.return_code == 0

    # Test 4: Command with timeout that should fail
    try:
        run_command(["sleep", "3"], timeout=1)
        assert False, "Expected TimeoutExpired"
    except subprocess.TimeoutExpired:
        pass

    # Test 5: Command with custom environment variable
    env = {"MY_VAR": "test_value"}
    result = run_command(["env"], env=env, return_output=True)
    assert b"MY_VAR=test_value" in result.captured_output

    # Test 6: Command with custom working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output.strip()

    # Test 7: Command with verbose output
    import io
    import sys
    captured_output = io.StringIO()
    sys.stdout = captured_output
    run_command(["echo", "Verbose test"], verbose=True)
    sys.stdout = sys.__stdout__
    assert "Verbose test" in captured_output.getvalue()

    # Test 8: Command that returns output only when requested
    result = run_command(["echo", "Output test"], return_output=False)
    assert result.captured_output is None
    result = run_command(["echo", "Output test"], return_output=True)
    assert b"Output test" in result.captured_output

    # Test 9: Command with shell=True
    result = run_command("echo 'Shell test'", shell=True, return_output=True)
    assert b"Shell test" in result.captured_output

    # Test 10: Command that raises CalledProcessError and captures output
    try:
        run_command(["ls", "/nonexistent"])
        assert False, "Expected CalledProcessError"
    except subprocess.CalledProcessError as e:
        assert e.output is not None
        assert b"No such file or directory" in e.output

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #29
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.check_output(["ls", "nonexistent"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "No output was generated." in str(wrapped_error)
    
    print("All tests passed!")

if __name__ == "__main__":
    test_error_wrapper()


# LLM-generated content at query #30
#--------------------------

# Unit test for function run_command
def test_run_command():  
    # Test 1: Simple command that should succeed
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

    # Test 2: Command that fails
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    # Test 3: Command with timeout
    try:
        run_command(["sleep", "2"], timeout=1)
    except subprocess.TimeoutExpired:
        pass  # Expected

    # Test 4: Command with environment variable
    result = run_command(["printenv", "MYVAR"], env={"MYVAR": "test"}, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

    # Test 5: Command with working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #31
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test that error_wrapper wraps CalledProcessError
    try:
        subprocess.run(['false'], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert 'Captured output:' in str(wrapped)
    
    # Test that error_wrapper wraps TimeoutExpired
    try:
        subprocess.run(['sleep', '10'], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert 'Captured output:' in str(wrapped)
    
    # Test that error_wrapper does not wrap other exceptions
    try:
        raise ValueError('test')
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert wrapped is e



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():


# LLM-generated content at query #2
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(returncode=1, cmd='ls', output=b'File not found')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert 'Captured output:' in str(wrapped_err)
    
    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired(cmd='sleep 10', timeout=5, output=b'Process timed out')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert 'Captured output:' in str(wrapped_err)
    
    # Test with other exception
    err = ValueError('Some error')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == 'Some error'
    
    print("All tests passed!")



# LLM-generated content at query #3
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test case 1: Successful command execution
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"Hello, World!\n"
    
    # Test case 2: Command with non-zero return code
    result = run_command(["ls", "nonexistent_file.txt"], ignore_errors=True)
    assert result.return_code != 0
    
    # Test case 3: Command with timeout
    try:
        run_command(["sleep", "10"], timeout=1)
    except subprocess.TimeoutExpired:
        pass  # Expected behavior
    
    # Test case 4: Command with environment variable
    result = run_command(["printenv", "MY_VAR"], env={"MY_VAR": "test_value"}, return_output=True)
    assert result.captured_output == b"test_value\n"
    
    # Test case 5: Command with working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.captured_output.strip() == tmpdir.encode()
    
    # Test case 6: Command with verbose output
    result = run_command(["echo", "Verbose test"], verbose=True, return_output=True)
    assert result.captured_output == b"Verbose test\n"
    
    # Test case 7: Command with shell=True
    result = run_command("echo 'Shell test'", shell=True, return_output=True)
    assert result.captured_output == b"Shell test\n"
    
    # Test case 8: Command with large output
    result = run_command(["seq", "10000"], return_output=True)
    assert len(result.captured_output) > 0
    
    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #4
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import traceback

    def test_called_process_error():
        try:
            subprocess.run([sys.executable, '-c', 'import sys; sys.exit(1)'], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            e = error_wrapper(e)
            print(e)
            traceback.print_exc()

    def test_timeout_expired():
        try:
            subprocess.run([sys.executable, '-c', 'import time; time.sleep(10)'], timeout=0.1, capture_output=True)
        except subprocess.TimeoutExpired as e:
            e = error_wrapper(e)
            print(e)
            traceback.print_exc()

    test_called_process_error()
    test_timeout_expired()


if __name__ == '__main__':  # pragma: no cover
    test_error_wrapper()



# LLM-generated content at query #5
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Normal command execution
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"Hello, World!" in result.captured_output

    # Test 2: Command with error
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0

    # Test 3: Timeout
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test 4: Environment variable
    result = run_command(["printenv", "MY_VAR"], env={"MY_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test 5: Working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #6
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Simple command that should succeed
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

    # Test 2: Command that should fail
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    # Test 3: Command with timeout
    try:
        run_command(["sleep", "2"], timeout=1)
    except subprocess.TimeoutExpired:
        pass  # Expected
    else:
        assert False, "Expected TimeoutExpired"

    # Test 4: Command with environment variable
    import os
    env = os.environ.copy()
    env["TEST_VAR"] = "test_value"
    result = run_command(["printenv", "TEST_VAR"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test 5: Command with cwd
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #7
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Successful command execution
    result = run_command("echo Hello, World!", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"Hello, World!" in result.captured_output

    # Test 2: Command with error (non-zero return code)
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.return_code == 1

    # Test 3: Timeout error
    result = run_command("sleep 2", shell=True, timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test 4: Environment variable
    result = run_command("echo $MYVAR", shell=True, env={"MYVAR": "test"}, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

    # Test 5: Working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", shell=True, cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test 6: Verbose mode (should not raise exception)
    result = run_command("echo test", shell=True, verbose=True)
    assert result.return_code == 0

    # Test 7: Command as list
    result = run_command(["echo", "Hello"], return_output=True)
    assert result.return_code == 0
    assert b"Hello" in result.captured_output

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #8
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import traceback
    import io

    def test_called_process_error():
        try:
            subprocess.run([sys.executable, '-c', 'import sys; sys.exit(1)'], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            wrapped = error_wrapper(e)
            output = io.StringIO()
            traceback.print_exception(type(wrapped), wrapped, wrapped.__traceback__, file=output)
            assert "Captured output:" in output.getvalue()
            print("test_called_process_error passed")

    def test_timeout_expired():
        try:
            subprocess.run([sys.executable, '-c', 'import time; time.sleep(10)'], timeout=0.1, capture_output=True)
        except subprocess.TimeoutExpired as e:
            wrapped = error_wrapper(e)
            output = io.StringIO()
            traceback.print_exception(type(wrapped), wrapped, wrapped.__traceback__, file=output)
            assert "Captured output:" in output.getvalue()
            print("test_timeout_expired passed")

    def test_other_exception():
        try:
            raise ValueError("test")
        except ValueError as e:
            wrapped = error_wrapper(e)
            assert wrapped is e
            print("test_other_exception passed")

    test_called_process_error()
    test_timeout_expired()
    test_other_exception()



# LLM-generated content at query #9
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import io

    # Test CalledProcessError
    try:
        subprocess.run([sys.executable, "-c", "import sys; sys.exit(1)"], check=True)
    except subprocess.CalledProcessError as e:
        e.output = b"Test output"
        wrapped = error_wrapper(e)
        assert "Captured output:" in str(wrapped)
        assert "Test output" in str(wrapped)

    # Test TimeoutExpired
    try:
        subprocess.run([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        e.output = b"Timeout output"
        wrapped = error_wrapper(e)
        assert "Captured output:" in str(wrapped)
        assert "Timeout output" in str(wrapped)

    # Test other exception
    try:
        raise ValueError("Test")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert wrapped is e

    print("All tests passed.")


if __name__ == "__main__":  # pragma: no cover
    test_error_wrapper()


# LLM-generated content at query #10
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped)
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped)
    
    # Test with other exception
    other_exception = ValueError("test")
    wrapped = error_wrapper(other_exception)
    assert wrapped is other_exception



# LLM-generated content at query #11
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(returncode=1, cmd='ls')
    err.output = b'error: file not found'
    wrapped_err = error_wrapper(err)
    assert 'Captured output:' in str(wrapped_err)
    assert 'error: file not found' in str(wrapped_err)
    
    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired(cmd='sleep 10', timeout=5)
    err.output = b'timeout after 5 seconds'
    wrapped_err = error_wrapper(err)
    assert 'Captured output:' in str(wrapped_err)
    assert 'timeout after 5 seconds' in str(wrapped_err)
    
    # Test with other exception
    err = ValueError('test error')
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == 'test error'
    
    print('All tests passed')

if __name__ == '__main__':
    test_error_wrapper()


# LLM-generated content at query #12
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test case 1: Successful command execution
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"Hello, World!\n"
    
    # Test case 2: Command with non-zero return code
    result = run_command(["ls", "nonexistent_file"], ignore_errors=True)
    assert result.return_code != 0
    
    # Test case 3: Command with timeout
    import time
    start_time = time.time()
    result = run_command(["sleep", "2"], timeout=1, ignore_errors=True)
    elapsed_time = time.time() - start_time
    assert elapsed_time < 2  # Should timeout before 2 seconds
    assert result.return_code == -32768  # Special return code for timeout
    
    # Test case 4: Command with environment variable
    result = run_command(["echo", "$MY_VAR"], env={"MY_VAR": "test_value"}, return_output=True, shell=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"
    
    # Test case 5: Command with custom working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.decode().strip() == tmpdir
    
    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #13
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():


# LLM-generated content at query #14
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Simple command that should succeed
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test 2: Command that fails (non-zero return code)
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    # Test 3: Command with timeout
    import time
    start = time.time()
    result = run_command(["sleep", "2"], timeout=1, ignore_errors=True)
    elapsed = time.time() - start
    assert elapsed < 2  # Should timeout before 2 seconds
    assert result.return_code == -32768  # Special code for timeout

    # Test 4: Command with environment variable
    result = run_command(["printenv", "MY_VAR"], env={"MY_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test 5: Command with working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test 6: Command that produces a lot of output (test truncation)
    result = run_command(["yes", "A" * 100], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert b"*** (previous output truncated) ***" in result.captured_output

    # Test 7: Command with shell=True
    result = run_command("echo Hello, World!", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #15
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        e = error_wrapper(e)
        assert "Captured output:" in str(e)
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        e = error_wrapper(e)
        assert "Captured output:" in str(e)
    
    # Test with other exception
    try:
        raise ValueError("test")
    except ValueError as e:
        e = error_wrapper(e)
        assert "test" in str(e)



# LLM-generated content at query #16
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test with CalledProcessError
    try:
        subprocess.run(["ls", "nonexistent"], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped_error)
        print("Test for CalledProcessError passed.")

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, capture_output=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped_error)
        print("Test for TimeoutExpired passed.")

    # Test with other exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert wrapped_error is e
        print("Test for other exception passed.")



# LLM-generated content at query #17
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(returncode=1, cmd="test")
    err.output = b"test output"
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired(cmd="test", timeout=1)
    err.output = b"timeout output"
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with other exception
    err = Exception("test")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "test"

    print("All tests passed.")

if __name__ == "__main__":
    test_error_wrapper()


# LLM-generated content at query #18
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import io
    import traceback

    def capture_exception_str(exc):
        """Helper to capture exception string representation"""
        return str(exc)

    # Test 1: CalledProcessError with output
    print("Test 1: CalledProcessError with output")
    try:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["python", "-c", "print('error output'); exit(1)"],
            output=b"error output\n"
        )
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        result = capture_exception_str(wrapped)
        print(f"Result:\n{result}")
        assert "Captured output:" in result
        assert "error output" in result
        print("✓ Test 1 passed\n")

    # Test 2: CalledProcessError without output
    print("Test 2: CalledProcessError without output")
    try:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["python", "-c", "exit(1)"],
            output=None
        )
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        result = capture_exception_str(wrapped)
        print(f"Result:\n{result}")
        assert "No output was generated." in result
        print("✓ Test 2 passed\n")

    # Test 3: TimeoutExpired with output
    print("Test 3: TimeoutExpired with output")
    try:
        raise subprocess.TimeoutExpired(
            cmd=["python", "-c", "import time; time.sleep(10)"],
            timeout=1,
            output=b"partial output\n"
        )
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        result = capture_exception_str(wrapped)
        print(f"Result:\n{result}")
        assert "Captured output:" in result
        assert "partial output" in result
        print("✓ Test 3 passed\n")

    # Test 4: Other exception (should not be modified)
    print("Test 4: Other exception (should not be modified)")
    original_msg = "Custom error"
    try:
        raise ValueError(original_msg)
    except ValueError as e:
        wrapped = error_wrapper(e)
        result = capture_exception_str(wrapped)
        print(f"Result: {result}")
        assert result == original_msg
        print("✓ Test 4 passed\n")

    # Test 5: Unicode output handling
    print("Test 5: Unicode output handling")
    try:
        # Create output that can't be decoded as UTF-8
        invalid_utf8 = b'\xff\xfe\x00\x00'  # UTF-32LE BOM
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["test"],
            output=invalid_utf8
        )
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        result = capture_exception_str(wrapped)
        print(f"Result:\n{result}")
        assert "Failed to parse output." in result
        print("✓ Test 5 passed\n")

    print("All tests passed!")


if __name__ == "__main__":  # pragma: no cover
    test_error_wrapper()


# LLM-generated content at query #19
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import traceback

    def run_test():
        # Test CalledProcessError
        try:
            subprocess.run([sys.executable, '-c', 'import sys; sys.exit(1)'], check=True)
        except subprocess.CalledProcessError as e:
            wrapped = error_wrapper(e)
            print("CalledProcessError wrapped output:")
            print(wrapped)
            assert "Captured output:" in str(wrapped)

        # Test TimeoutExpired
        try:
            subprocess.run([sys.executable, '-c', 'import time; time.sleep(10)'], timeout=0.1, check=True)
        except subprocess.TimeoutExpired as e:
            wrapped = error_wrapper(e)
            print("\nTimeoutExpired wrapped output:")
            print(wrapped)
            assert "Captured output:" in str(wrapped)

        # Test other exception (should not be wrapped)
        try:
            raise ValueError("test")
        except ValueError as e:
            wrapped = error_wrapper(e)
            print("\nOther exception (should be unchanged):")
            print(wrapped)
            assert str(wrapped) == "test"

        print("\nAll tests passed!")

    if __name__ == '__main__':
        run_test()
    else:
        run_test()



# LLM-generated content at query #20
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        e = error_wrapper(e)
        assert "Captured output:" in str(e)
        assert "No output was generated." not in str(e)
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        e = error_wrapper(e)
        assert "Captured output:" in str(e)
        assert "No output was generated." not in str(e)
    # Test with other exception
    try:
        raise ValueError("test")
    except ValueError as e:
        e = error_wrapper(e)
        assert "test" in str(e)
        assert "Captured output:" not in str(e)
    print("All tests passed.")

if __name__ == "__main__":
    test_error_wrapper()


# LLM-generated content at query #21
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import io

    # Test with CalledProcessError
    try:
        subprocess.run([sys.executable, "-c", "import sys; sys.exit(1)"], check=True)
    except subprocess.CalledProcessError as e:
        e = error_wrapper(e)
        assert "Captured output:" in str(e)

    # Test with TimeoutExpired
    try:
        subprocess.run([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        e = error_wrapper(e)
        assert "Captured output:" in str(e)

    # Test with other exception
    try:
        raise ValueError("test")
    except ValueError as e:
        e = error_wrapper(e)
        assert str(e) == "test"

    print("All tests passed.")

if __name__ == "__main__":  # pragma: no cover
    test_error_wrapper()


# LLM-generated content at query #22
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.run(["ls", "nonexistent"], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, capture_output=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with other exception
    other_error = ValueError("Test error")
    wrapped_error = error_wrapper(other_error)
    assert wrapped_error is other_error



# LLM-generated content at query #23
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():


# LLM-generated content at query #24
#--------------------------

# Unit test for function run_command
def test_run_command():  
    # Test 1: Normal command execution
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

    # Test 2: Command with error
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0

    # Test 3: Timeout
    result = run_command(["sleep", "2"], timeout=1, ignore_errors=True)
    assert result.return_code == -32768

    # Test 4: Environment variables
    result = run_command(["printenv", "MYVAR"], env={"MYVAR": "test"}, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

    # Test 5: Working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #25
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import io

    # Test with CalledProcessError
    try:
        subprocess.run([sys.executable, "-c", "import sys; sys.exit(1)"], check=True)
    except subprocess.CalledProcessError as e:
        e.output = b"Test output"
        wrapped = error_wrapper(e)
        assert "Captured output:" in str(wrapped)
        assert "Test output" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.001)
    except subprocess.TimeoutExpired as e:
        e.output = b"Timeout output"
        wrapped = error_wrapper(e)
        assert "Captured output:" in str(wrapped)
        assert "Timeout output" in str(wrapped)

    # Test with other exception
    try:
        raise ValueError("Test")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert str(wrapped) == "Test"

    print("All tests passed!")

if __name__ == "__main__":  # pragma: no cover
    test_error_wrapper()


# LLM-generated content at query #26
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():


# LLM-generated content at query #27
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import io
    import sys

    # Capture stderr
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()

    try:
        # Test with CalledProcessError
        err = subprocess.CalledProcessError(1, 'ls', output=b'Some output')
        wrapped_err = error_wrapper(err)
        assert isinstance(wrapped_err, subprocess.CalledProcessError)
        assert wrapped_err.output == b'Some output'

        # Test with TimeoutExpired
        err = subprocess.TimeoutExpired('ls', 10, output=b'Timeout output')
        wrapped_err = error_wrapper(err)
        assert isinstance(wrapped_err, subprocess.TimeoutExpired)
        assert wrapped_err.output == b'Timeout output'

        # Test with other exception
        try:
            raise ValueError('Test')
        except ValueError as e:
            wrapped_err = error_wrapper(e)
            assert wrapped_err is e

        print("All tests passed.")
    finally:
        sys.stderr = old_stderr

if __name__ == "__main__":  # pragma: no cover
    test_error_wrapper()


# LLM-generated content at query #28
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.run(["ls", "nonexistent"], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, capture_output=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with other exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert str(wrapped_error) == "Test error"



# LLM-generated content at query #29
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test case 1: Successful command execution
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"Hello, World!\n"
    print("Test case 1 passed")

    # Test case 2: Command with error
    result = run_command(["ls", "nonexistent_file.txt"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    print("Test case 2 passed")

    # Test case 3: Command timeout
    result = run_command(["sleep", "2"], timeout=1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None
    print("Test case 3 passed")

    # Test case 4: Command with environment variables
    env = {"MY_VAR": "test_value"}
    result = run_command(["printenv", "MY_VAR"], env=env, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"
    print("Test case 4 passed")

    # Test case 5: Command with working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.decode().strip() == tmpdir
    print("Test case 5 passed")

    # Test case 6: Command with shell=True
    result = run_command("echo $HOME", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    print("Test case 6 passed")

    # Test case 7: Command with verbose output
    import io
    import sys
    captured_output = io.StringIO()
    sys.stdout = captured_output
    result = run_command(["echo", "Verbose test"], verbose=True)
    sys.stdout = sys.__stdout__
    assert "Verbose test" in captured_output.getvalue()
    print("Test case 7 passed")

    # Test case 8: Command without return_output
    result = run_command(["echo", "No output"])
    assert result.return_code == 0
    assert result.captured_output is None
    print("Test case 8 passed")

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #30
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"some output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    
    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired(cmd="ls", timeout=5, output=b"some output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    
    # Test with other exception
    err = ValueError("some error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "some error"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_error_wrapper()


