####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Simple command that should succeed
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test 2: Command that should fail
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    # Test 3: Command with timeout
    import time
    start = time.time()
    result = run_command(["sleep", "2"], timeout=1, ignore_errors=True)
    elapsed = time.time() - start
    assert elapsed < 2  # Should timeout before 2 seconds
    assert result.return_code == -32768  # Special return code for timeout

    # Test 4: Command with environment variable
    result = run_command(["printenv", "MYVAR"], env={"MYVAR": "test"}, return_output=True)
    assert b"test" in result.captured_output

    # Test 5: Command with working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test 6: Command with shell=True
    result = run_command("echo hello", shell=True, return_output=True)
    assert b"hello" in result.captured_output

    # Test 7: Command that produces a lot of output (test truncation)
    result = run_command(["python3", "-c", "print('a'*10000)"], return_output=True, ignore_errors=True)
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + 100  # Allow some overhead

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #2
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Normal command execution
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test 2: Command with error
    result = run_command(["ls", "/nonexistent"], ignore_errors=True, return_output=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test 3: Timeout
    try:
        run_command(["sleep", "10"], timeout=1)
    except subprocess.TimeoutExpired:
        pass  # Expected

    # Test 4: Environment variables
    result = run_command(["printenv", "MY_VAR"], env={"MY_VAR": "test"}, return_output=True)
    assert b"test" in result.captured_output

    # Test 5: Working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #3
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
        assert "test output" in str(wrapped_error)

    # Test with TimeoutExpired
    try:
        raise subprocess.TimeoutExpired(cmd="test", timeout=10, output=b"timeout output")
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped_error)
        assert "timeout output" in str(wrapped_error)

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


# LLM-generated content at query #4
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Basic command execution
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"Hello, World!" in result.captured_output

    # Test 2: Command with error
    result = run_command(["ls", "/nonexistent"], ignore_errors=True, return_output=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test 3: Timeout
    try:
        run_command(["sleep", "10"], timeout=0.1, ignore_errors=False)
    except subprocess.TimeoutExpired:
        pass  # Expected
    else:
        assert False, "Expected TimeoutExpired"

    # Test 4: Environment variable
    result = run_command(["printenv", "MYVAR"], env={"MYVAR": "test"}, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

    # Test 5: Working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test 6: Shell command
    result = run_command("echo Hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"Hello" in result.captured_output

    # Test 7: Verbose mode (should not raise)
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0

    # Test 8: Ignore errors
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #5
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import traceback

    def test_called_process_error():
        try:
            subprocess.check_output(['ls', 'nonexistent'])
        except subprocess.CalledProcessError as e:
            e = error_wrapper(e)
            print("Test CalledProcessError:")
            print(str(e))
            print()

    def test_timeout_expired():
        try:
            subprocess.run(['sleep', '10'], timeout=0.1)
        except subprocess.TimeoutExpired as e:
            e = error_wrapper(e)
            print("Test TimeoutExpired:")
            print(str(e))
            print()

    def test_other_exception():
        try:
            raise ValueError("Some other error")
        except ValueError as e:
            e = error_wrapper(e)
            print("Test other exception (should be unchanged):")
            print(str(e))
            print()

    test_called_process_error()
    test_timeout_expired()
    test_other_exception()

if __name__ == "__main__":  # pragma: no cover
    test_error_wrapper()


# LLM-generated content at query #6
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
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test"



# LLM-generated content at query #7
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Run a simple command that should succeed
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"
    print("Test 1 passed")

    # Test 2: Run a command that should fail
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    print("Test 2 passed")

    # Test 3: Run a command with a timeout
    result = run_command(["sleep", "2"], timeout=1, ignore_errors=True)
    assert result.return_code == -32768
    print("Test 3 passed")

    # Test 4: Run a command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command(["env"], env=env, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output
    print("Test 4 passed")

    # Test 5: Run a command with a different working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output.strip()
    print("Test 5 passed")

    # Test 6: Run a command with verbose output
    import io
    import sys
    captured_output = io.StringIO()
    sys.stdout = captured_output
    result = run_command(["echo", "verbose test"], verbose=True)
    sys.stdout = sys.__stdout__
    assert "verbose test" in captured_output.getvalue()
    print("Test 6 passed")

    # Test 7: Run a command that returns output without errors
    result = run_command(["echo", "output test"], return_output=True)
    assert result.captured_output == b"output test\n"
    print("Test 7 passed")

    # Test 8: Run a command that should be ignored on error
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    print("Test 8 passed")

    print("All tests passed!")

# Run the unit tests
if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #8
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
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
    # Test with other exceptions
    try:
        raise ValueError("test")
    except ValueError as e:
        e2 = error_wrapper(e)
        assert e is e2



# LLM-generated content at query #9
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import io
    import traceback

    def capture_exception(exception):
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            traceback.print_exception(type(exception), exception, None)
            return sys.stderr.getvalue()
        finally:
            sys.stderr = old_stderr

    # Test CalledProcessError
    err = subprocess.CalledProcessError(1, 'ls', output=b'file1\nfile2\n')
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert 'Captured output:' in str(wrapped)
    assert 'file1' in str(wrapped)
    assert 'file2' in str(wrapped)

    # Test TimeoutExpired
    err = subprocess.TimeoutExpired('sleep 10', 5, output=b'still running...\n')
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert 'Captured output:' in str(wrapped)
    assert 'still running' in str(wrapped)

    # Test other exception
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err

    # Test that output is truncated if too long
    long_output = b'a' * (MAX_OUTPUT_LENGTH + 100)
    err = subprocess.CalledProcessError(1, 'ls', output=long_output)
    wrapped = error_wrapper(err)
    assert 'truncated' in str(wrapped)

    # Test that exception can be printed
    err = subprocess.CalledProcessError(1, 'ls', output=b'test output')
    wrapped = error_wrapper(err)
    printed = capture_exception(wrapped)
    assert 'test output' in printed

    print("All tests passed!")



# LLM-generated content at query #10
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError  
    try:  
        subprocess.run(["ls", "nonexistent"], check=True, capture_output=True)  
    except subprocess.CalledProcessError as e:  
        wrapped_error = error_wrapper(e)  
        assert "Captured output:" in str(wrapped_error)  
        print("Test for CalledProcessError passed.")  
    # Test with TimeoutExpired  
    try:  
        subprocess.run(["sleep", "2"], timeout=0.1, capture_output=True)  
    except subprocess.TimeoutExpired as e:  
        wrapped_error = error_wrapper(e)  
        assert "Captured output:" in str(wrapped_error)  
        print("Test for TimeoutExpired passed.")  
    # Test with other exception  
    try:  
        raise ValueError("Test error")  
    except ValueError as e:  
        wrapped_error = error_wrapper(e)  
        assert str(wrapped_error) == "Test error"  
        print("Test for other exception passed.")  



# LLM-generated content at query #11
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.check_output(["ls", "nonexistent"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert "Captured output:" in str(wrapped)
        print("Test for CalledProcessError passed.")

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert "No output was generated." in str(wrapped)
        print("Test for TimeoutExpired passed.")

    # Test with other exception
    try:
        raise ValueError("Test")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert str(wrapped) == "Test"
        print("Test for other exception passed.")



# LLM-generated content at query #12
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import traceback

    def test_called_process_error():
        try:
            subprocess.check_output(["ls", "nonexistent"], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            e = error_wrapper(e)
            print("CalledProcessError output:")
            print(str(e))
            print()

    def test_timeout_expired():
        try:
            subprocess.run(["sleep", "10"], timeout=0.1, check=True)
        except subprocess.TimeoutExpired as e:
            e = error_wrapper(e)
            print("TimeoutExpired output:")
            print(str(e))
            print()

    def test_other_exception():
        try:
            raise ValueError("Some other error")
        except ValueError as e:
            e = error_wrapper(e)
            print("Other exception (should be unchanged):")
            print(str(e))
            print()

    test_called_process_error()
    test_timeout_expired()
    test_other_exception()

    print("All tests passed.")


if __name__ == "__main__":  # pragma: no cover
    test_error_wrapper()



# LLM-generated content at query #13
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():


# LLM-generated content at query #14
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
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"
    
    print("All tests passed!")



# LLM-generated content at query #15
#--------------------------

# Unit test for function run_command
def test_run_command():


# LLM-generated content at query #16
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test CalledProcessError
    try:
        subprocess.check_output(['ls', 'nonexistent'], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped)
    
    # Test TimeoutExpired
    try:
        subprocess.run(['sleep', '10'], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped)
    
    # Test other exception
    try:
        raise ValueError("test")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert wrapped is e



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



# LLM-generated content at query #19
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import traceback

    def test_called_process_error():
        err = subprocess.CalledProcessError(returncode=1, cmd='ls')
        err.output = b'file1\nfile2\n'
        wrapped_err = error_wrapper(err)
        assert isinstance(wrapped_err, subprocess.CalledProcessError)
        assert 'Captured output:' in str(wrapped_err)
        assert '    file1' in str(wrapped_err)
        assert '    file2' in str(wrapped_err)
        print("✓ test_called_process_error passed")

    def test_timeout_expired():
        err = subprocess.TimeoutExpired(cmd='sleep 10', timeout=1)
        err.output = b'still running...\n'
        wrapped_err = error_wrapper(err)
        assert isinstance(wrapped_err, subprocess.TimeoutExpired)
        assert 'Captured output:' in str(wrapped_err)
        assert '    still running...' in str(wrapped_err)
        print("✓ test_timeout_expired passed")

    def test_no_output():
        err = subprocess.CalledProcessError(returncode=1, cmd='ls')
        err.output = None
        wrapped_err = error_wrapper(err)
        assert 'No output was generated.' in str(wrapped_err)
        print("✓ test_no_output passed")

    def test_other_exception():
        original_err = ValueError("test")
        wrapped_err = error_wrapper(original_err)
        assert wrapped_err is original_err
        print("✓ test_other_exception passed")

    def test_unicode_decode_error():
        err = subprocess.CalledProcessError(returncode=1, cmd='ls')
        err.output = b'\xff\xfe'  # Invalid UTF-8
        wrapped_err = error_wrapper(err)
        assert 'Failed to parse output.' in str(wrapped_err)
        print("✓ test_unicode_decode_error passed")

    test_called_process_error()
    test_timeout_expired()
    test_no_output()
    test_other_exception()
    test_unicode_decode_error()



# LLM-generated content at query #20
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
    
    # Test with TimeoutExpired
    try:
        subprocess.run(['sleep', '10'], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with other exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert wrapped_error is e



# LLM-generated content at query #21
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(returncode=1, cmd='ls', output=b'No such file or directory')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No such file or directory" in str(wrapped_err)
    
    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired(cmd='sleep 10', timeout=1, output=b'Command timed out')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Command timed out" in str(wrapped_err)
    
    # Test with other exception
    err = ValueError("Some error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Some error"
    
    print("All tests passed!")



# LLM-generated content at query #22
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():


# LLM-generated content at query #23
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
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert wrapped_error is e  # Should return the same instance



# LLM-generated content at query #24
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import traceback
    import io

    def test_called_process_error():
        # Test CalledProcessError
        try:
            subprocess.run([sys.executable, "-c", "import sys; sys.exit(1)"], check=True)
        except subprocess.CalledProcessError as e:
            e = error_wrapper(e)
            output = str(e)
            assert "returned non-zero exit status" in output
            print("CalledProcessError test passed.")

    def test_timeout_expired():
        # Test TimeoutExpired
        try:
            subprocess.run([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.1, check=True)
        except subprocess.TimeoutExpired as e:
            e = error_wrapper(e)
            output = str(e)
            assert "timed out after" in output
            print("TimeoutExpired test passed.")

    def test_other_exception():
        # Test other exception (should not be wrapped)
        try:
            raise ValueError("test")
        except ValueError as e:
            e2 = error_wrapper(e)
            assert e is e2
            print("Other exception test passed.")

    test_called_process_error()
    test_timeout_expired()
    test_other_exception()
    print("All tests passed.")


if __name__ == "__main__":  # pragma: no cover
    test_error_wrapper()


# LLM-generated content at query #25
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



# LLM-generated content at query #26
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test CalledProcessError with output
    try:
        subprocess.run(["false"], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        e = error_wrapper(e)
        assert "Captured output:" in str(e)
    
    # Test TimeoutExpired with output
    try:
        subprocess.run(["sleep", "2"], timeout=0.1, capture_output=True)
    except subprocess.TimeoutExpired as e:
        e = error_wrapper(e)
        assert "Captured output:" in str(e)
    
    # Test other exception
    try:
        raise ValueError("test")
    except ValueError as e:
        e = error_wrapper(e)
        assert str(e) == "test"



# LLM-generated content at query #27
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test case 1: Successful command execution
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"Hello, World!" in result.captured_output

    # Test case 2: Command with error (non-zero return code)
    result = run_command(["ls", "nonexistent_file.txt"], ignore_errors=True, return_output=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test case 3: Timeout
    result = run_command(["sleep", "2"], timeout=1, ignore_errors=True, return_output=True)
    assert result.return_code == -32768  # Special return code for timeout
    assert result.captured_output is not None

    # Test case 4: Environment variables
    env = {"MY_VAR": "123"}
    result = run_command(["bash", "-c", "echo $MY_VAR"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"123" in result.captured_output

    # Test case 5: Working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test case 6: Shell command
    result = run_command("echo Hello from shell", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"Hello from shell" in result.captured_output

    # Test case 7: No output capture (return_output=False)
    result = run_command(["echo", "test"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test case 8: Verbose mode (should print output)
    import io
    import sys
    captured_output = io.StringIO()
    sys.stdout = captured_output
    try:
        run_command(["echo", "verbose test"], verbose=True)
    finally:
        sys.stdout = sys.__stdout__
    assert "verbose test" in captured_output.getvalue()

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #28
#--------------------------

# Unit test for function run_command
def test_run_command():


# LLM-generated content at query #29
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import io

    # Test with CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, 'test', output=b'output')
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        # Check that __str__ includes captured output
        output = io.StringIO()
        sys.stderr = output
        print(wrapped, file=sys.stderr)
        sys.stderr = sys.__stderr__
        assert 'Captured output:' in output.getvalue()

    # Test with TimeoutExpired
    try:
        raise subprocess.TimeoutExpired('test', 1, output=b'output')
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        # Check that __str__ includes captured output
        output = io.StringIO()
        sys.stderr = output
        print(wrapped, file=sys.stderr)
        sys.stderr = sys.__stderr__
        assert 'Captured output:' in output.getvalue()

    # Test with other exception
    try:
        raise ValueError('test')
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert wrapped is e

    print('All tests passed.')


if __name__ == '__main__':  # pragma: no cover
    test_error_wrapper()


# LLM-generated content at query #30
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():


# LLM-generated content at query #31
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
        subprocess.run(['sleep', '2'], timeout=0.1)
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



# LLM-generated content at query #32
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert "Captured output:" in str(wrapped)
        print("CalledProcessError test passed")
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert "Captured output:" in str(wrapped)
        print("TimeoutExpired test passed")
    
    # Test with other exception
    try:
        raise ValueError("test")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert str(wrapped) == "test"
        print("Other exception test passed")

if __name__ == "__main__":
    test_error_wrapper()


# LLM-generated content at query #33
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import io
    import traceback

    def capture_output(func):
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            func()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        return output

    # Test CalledProcessError
    err = subprocess.CalledProcessError(1, 'ls', output=b'file1\nfile2\n')
    wrapped = error_wrapper(err)
    output = capture_output(lambda: traceback.print_exception(type(wrapped), wrapped, None))
    assert 'Captured output:' in output
    assert 'file1' in output
    assert 'file2' in output

    # Test TimeoutExpired
    err = subprocess.TimeoutExpired('sleep 10', 5, output=b'still running...\n')
    wrapped = error_wrapper(err)
    output = capture_output(lambda: traceback.print_exception(type(wrapped), wrapped, None))
    assert 'Captured output:' in output
    assert 'still running...' in output

    # Test other exception
    try:
        raise ValueError("test")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert wrapped is e

    print("All tests passed.")


if __name__ == '__main__':  # pragma: no cover
    test_error_wrapper()


# LLM-generated content at query #34
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import traceback
    import io

    def test_called_process_error():
        err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"file1\nfile2\n")
        wrapped_err = error_wrapper(err)
        assert isinstance(wrapped_err, subprocess.CalledProcessError)
        assert wrapped_err.returncode == 1
        assert wrapped_err.cmd == "ls"
        assert wrapped_err.output == b"file1\nfile2\n"
        # Check __str__ method
        str_repr = str(wrapped_err)
        assert "Command 'ls' returned non-zero exit status 1." in str_repr
        assert "Captured output:" in str_repr
        assert "    file1" in str_repr
        assert "    file2" in str_repr
        print("test_called_process_error passed")

    def test_timeout_expired():
        err = subprocess.TimeoutExpired(cmd="sleep 10", timeout=5, output=b"some output")
        wrapped_err = error_wrapper(err)
        assert isinstance(wrapped_err, subprocess.TimeoutExpired)
        assert wrapped_err.cmd == "sleep 10"
        assert wrapped_err.timeout == 5
        assert wrapped_err.output == b"some output"
        # Check __str__ method
        str_repr = str(wrapped_err)
        assert "Command 'sleep 10' timed out after 5 seconds" in str_repr
        assert "Captured output:" in str_repr
        assert "    some output" in str_repr
        print("test_timeout_expired passed")

    def test_other_exception():
        err = ValueError("Some other error")
        wrapped_err = error_wrapper(err)
        assert wrapped_err is err
        print("test_other_exception passed")

    def test_no_output():
        err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
        wrapped_err = error_wrapper(err)
        str_repr = str(wrapped_err)
        assert "No output was generated." in str_repr
        print("test_no_output passed")

    def test_output_truncation():
        # Test that output is truncated if too long
        long_output = b"a" * (MAX_OUTPUT_LENGTH + 100)
        err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=long_output)
        wrapped_err = error_wrapper(err)
        str_repr = str(wrapped_err)
        assert "*** (previous output truncated) ***" in str_repr
        print("test_output_truncation passed")

    # Run tests
    test_called_process_error()
    test_timeout_expired()
    test_other_exception()
    test_no_output()
    test_output_truncation()
    print("All tests passed!")



# LLM-generated content at query #35
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.run(["ls", "nonexistent"], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert "Captured output:" in str(wrapped_error)
        print("Test for CalledProcessError passed.")

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "2"], timeout=0.1, capture_output=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert "Captured output:" in str(wrapped_error)
        print("Test for TimeoutExpired passed.")

    # Test with other exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert str(wrapped_error) == "Test error"
        print("Test for other exception passed.")



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import traceback
    import io

    def test_called_process_error():
        # Test CalledProcessError
        try:
            raise subprocess.CalledProcessError(1, 'ls', output=b'error output')
        except subprocess.CalledProcessError as e:
            wrapped = error_wrapper(e)
            assert isinstance(wrapped, subprocess.CalledProcessError)
            assert wrapped.output == b'error output'
            # Check __str__ includes output
            str_repr = str(wrapped)
            assert 'Captured output:' in str_repr
            assert 'error output' in str_repr
            print("✓ CalledProcessError test passed")

    def test_timeout_expired():
        # Test TimeoutExpired
        try:
            raise subprocess.TimeoutExpired('ls', 10, output=b'timeout output')
        except subprocess.TimeoutExpired as e:
            wrapped = error_wrapper(e)
            assert isinstance(wrapped, subprocess.TimeoutExpired)
            assert wrapped.output == b'timeout output'
            str_repr = str(wrapped)
            assert 'Captured output:' in str_repr
            assert 'timeout output' in str_repr
            print("✓ TimeoutExpired test passed")

    def test_other_exception():
        # Test other exception (should not be wrapped)
        try:
            raise ValueError("test error")
        except ValueError as e:
            wrapped = error_wrapper(e)
            assert wrapped is e  # Should return same instance
            print("✓ Other exception test passed")

    def test_no_output():
        # Test with no output
        try:
            raise subprocess.CalledProcessError(1, 'ls', output=None)
        except subprocess.CalledProcessError as e:
            wrapped = error_wrapper(e)
            str_repr = str(wrapped)
            assert 'No output was generated.' in str_repr
            print("✓ No output test passed")

    def test_unicode_decode_error():
        # Test with output that can't be decoded
        try:
            # Create bytes that can't be decoded as UTF-8
            invalid_bytes = b'\xff\xfe\x00\x00'
            raise subprocess.CalledProcessError(1, 'ls', output=invalid_bytes)
        except subprocess.CalledProcessError as e:
            wrapped = error_wrapper(e)
            str_repr = str(wrapped)
            assert 'Failed to parse output.' in str_repr
            print("✓ Unicode decode error test passed")

    # Run all tests
    test_called_process_error()
    test_timeout_expired()
    test_other_exception()
    test_no_output()
    test_unicode_decode_error()
    print("\nAll error_wrapper tests passed!")




# LLM-generated content at query #2
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():


# LLM-generated content at query #3
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Basic command execution
    result = run_command("echo Hello, World!", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test 2: Command with error
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.return_code == 1

    # Test 3: Command with timeout
    result = run_command("sleep 2", shell=True, timeout=1, ignore_errors=True)
    assert result.return_code == -32768

    # Test 4: Command with environment variables
    result = run_command("echo $MY_VAR", shell=True, env={"MY_VAR": "test"}, return_output=True)
    assert b"test" in result.captured_output

    # Test 5: Command with working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", shell=True, cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test 6: Command with verbose output
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        result = run_command("echo test", shell=True, verbose=True)
    assert "test" in f.getvalue()

    # Test 7: Command with return_output=False
    result = run_command("echo test", shell=True, return_output=False)
    assert result.captured_output is None

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #4
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
    other_error = ValueError("Test error")
    wrapped_error = error_wrapper(other_error)
    assert wrapped_error is other_error



# LLM-generated content at query #5
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Successful command execution
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"Hello, World!" in result.captured_output

    # Test 2: Command with error (non-zero return code)
    result = run_command(["ls", "/nonexistent"], ignore_errors=True, return_output=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test 3: Timeout
    import time
    result = run_command(["sleep", "2"], timeout=1, ignore_errors=True, return_output=True)
    assert result.return_code == -32768  # Special return code for timeout
    assert result.captured_output is not None

    # Test 4: Environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test 5: Working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test 6: Shell command
    result = run_command("echo Hello from shell", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"Hello from shell" in result.captured_output

    # Test 7: Verbose mode (should not raise exception)
    result = run_command(["echo", "verbose test"], verbose=True, return_output=True)
    assert result.return_code == 0

    # Test 8: No output capture
    result = run_command(["echo", "no output capture"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test 9: Exception handling without ignore_errors
    import pytest
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

    # Test 10: Exception handling with ignore_errors
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #6
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert "Captured output:" in str(wrapped)
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert "Captured output:" in str(wrapped)
    
    # Test with other exception
    try:
        raise ValueError("test")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert str(wrapped) == "test"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_error_wrapper()


# LLM-generated content at query #7
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Normal command execution
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test 2: Command with error
    result = run_command(["ls", "nonexistent_file"], ignore_errors=True)
    assert result.return_code != 0

    # Test 3: Timeout
    result = run_command(["sleep", "2"], timeout=1, ignore_errors=True)
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


# LLM-generated content at query #8
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Simple command that should succeed
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test 2: Command that should fail
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    # Test 3: Command with timeout
    try:
        run_command(["sleep", "2"], timeout=1)
    except subprocess.TimeoutExpired:
        pass  # Expected

    # Test 4: Command with environment variable
    result = run_command(["printenv", "MYVAR"], env={"MYVAR": "test"}, return_output=True)
    assert b"test" in result.captured_output

    # Test 5: Command with working directory
    import os
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert b"/tmp" in result.captured_output

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #9
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import traceback

    def test_called_process_error():
        try:
            subprocess.run([sys.executable, '-c', 'import sys; sys.exit(1)'], check=True)
        except subprocess.CalledProcessError as e:
            e = error_wrapper(e)
            print(e)
            assert "Captured output:" in str(e)

    def test_timeout_expired():
        try:
            subprocess.run([sys.executable, '-c', 'import time; time.sleep(10)'], timeout=0.1, check=True)
        except subprocess.TimeoutExpired as e:
            e = error_wrapper(e)
            print(e)
            assert "Captured output:" in str(e)

    def test_other_error():
        try:
            raise ValueError("test")
        except ValueError as e:
            e = error_wrapper(e)
            print(e)
            assert "test" in str(e)

    test_called_process_error()
    test_timeout_expired()
    test_other_error()
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
    
    print("All tests passed!")

if __name__ == "__main__":
    test_error_wrapper()


# LLM-generated content at query #11
#--------------------------

# Unit test for function run_command
def test_run_command():


# LLM-generated content at query #12
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import io

    # Test CalledProcessError
    try:
        subprocess.run([sys.executable, '-c', 'import sys; sys.exit(1)'], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        e = error_wrapper(e)
        output = str(e)
        assert "Captured output:" in output
        print("CalledProcessError test passed.")

    # Test TimeoutExpired
    try:
        subprocess.run([sys.executable, '-c', 'import time; time.sleep(10)'], timeout=0.1, capture_output=True)
    except subprocess.TimeoutExpired as e:
        e = error_wrapper(e)
        output = str(e)
        assert "Captured output:" in output
        print("TimeoutExpired test passed.")

    # Test other exception (should not be wrapped)
    try:
        raise ValueError("Test")
    except ValueError as e:
        e = error_wrapper(e)
        assert str(e) == "Test"
        print("Other exception test passed.")



# LLM-generated content at query #13
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import traceback

    def test_called_process_error():
        try:
            subprocess.check_output(["ls", "nonexistent"], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            e = error_wrapper(e)
            print("CalledProcessError output:")
            print(str(e))
            assert "Captured output:" in str(e)

    def test_timeout_expired():
        try:
            subprocess.run(["sleep", "10"], timeout=0.1, check=True)
        except subprocess.TimeoutExpired as e:
            e = error_wrapper(e)
            print("TimeoutExpired output:")
            print(str(e))
            assert "No output was generated." in str(e)

    def test_other_exception():
        try:
            raise ValueError("test")
        except ValueError as e:
            e = error_wrapper(e)
            print("Other exception output:")
            print(str(e))
            assert str(e) == "test"

    test_called_process_error()
    test_timeout_expired()
    test_other_exception()
    print("All tests passed.")

if __name__ == "__main__":  # pragma: no cover
    test_error_wrapper()


# LLM-generated content at query #14
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


# LLM-generated content at query #15
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import traceback
    import io

    def test_called_process_error():
        # Simulate a CalledProcessError with output
        try:
            raise subprocess.CalledProcessError(returncode=1, cmd='ls', output=b'file1\nfile2\n')
        except subprocess.CalledProcessError as e:
            wrapped = error_wrapper(e)
            # Check that the wrapped error has a __str__ method that includes output
            error_str = str(wrapped)
            assert 'Captured output:' in error_str
            assert '    file1' in error_str
            assert '    file2' in error_str
            print("✓ CalledProcessError wrapping works")

    def test_timeout_expired():
        # Simulate a TimeoutExpired with output
        try:
            raise subprocess.TimeoutExpired(cmd='sleep 10', timeout=1, output=b'partial output\n')
        except subprocess.TimeoutExpired as e:
            wrapped = error_wrapper(e)
            error_str = str(wrapped)
            assert 'Captured output:' in error_str
            assert '    partial output' in error_str
            print("✓ TimeoutExpired wrapping works")

    def test_other_exception():
        # Other exceptions should pass through unchanged
        try:
            raise ValueError("test error")
        except ValueError as e:
            wrapped = error_wrapper(e)
            assert wrapped is e  # Should be the same object
            print("✓ Other exceptions pass through unchanged")

    def test_no_output():
        # CalledProcessError with no output
        try:
            raise subprocess.CalledProcessError(returncode=1, cmd='ls', output=None)
        except subprocess.CalledProcessError as e:
            wrapped = error_wrapper(e)
            error_str = str(wrapped)
            assert 'No output was generated.' in error_str
            print("✓ No output case works")

    # Run all tests
    test_called_process_error()
    test_timeout_expired()
    test_other_exception()
    test_no_output()
    print("\nAll error_wrapper tests passed!")



# LLM-generated content at query #16
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import io
    import sys

    # Capture stderr
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()

    try:
        # Test CalledProcessError
        err = subprocess.CalledProcessError(1, 'ls', output=b'file1\nfile2\n')
        wrapped = error_wrapper(err)
        print(str(wrapped))
        assert 'Captured output:' in str(wrapped)
        assert 'file1' in str(wrapped)
        assert 'file2' in str(wrapped)

        # Test TimeoutExpired
        err = subprocess.TimeoutExpired('ls', 5, output=b'timeout output')
        wrapped = error_wrapper(err)
        print(str(wrapped))
        assert 'Captured output:' in str(wrapped)
        assert 'timeout output' in str(wrapped)

        # Test other exception
        try:
            raise ValueError('test')
        except ValueError as e:
            wrapped = error_wrapper(e)
            assert wrapped is e

        print("All tests passed!")
    finally:
        sys.stderr = old_stderr

if __name__ == '__main__':  # pragma: no cover
    test_error_wrapper()


# LLM-generated content at query #17
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.check_output(["ls", "nonexistent_file"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert "Captured output:" in str(wrapped_error)
        print("Test with CalledProcessError passed.")
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert "Captured output:" in str(wrapped_error)
        print("Test with TimeoutExpired passed.")
    
    # Test with other exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert str(wrapped_error) == "Test error"
        print("Test with other exception passed.")



# LLM-generated content at query #18
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import io
    import traceback

    def test_called_process_error():
        err = subprocess.CalledProcessError(1, 'ls')
        err.output = b'file1\nfile2\n'
        wrapped = error_wrapper(err)
        output = io.StringIO()
        traceback.print_exception(type(wrapped), wrapped, None, file=output)
        assert 'Captured output:' in output.getvalue()
        assert 'file1' in output.getvalue()
        print("test_called_process_error passed")

    def test_timeout_expired():
        err = subprocess.TimeoutExpired('sleep 10', 5)
        err.output = b'output line 1\noutput line 2\n'
        wrapped = error_wrapper(err)
        output = io.StringIO()
        traceback.print_exception(type(wrapped), wrapped, None, file=output)
        assert 'Captured output:' in output.getvalue()
        assert 'output line 1' in output.getvalue()
        print("test_timeout_expired passed")

    def test_no_output():
        err = subprocess.CalledProcessError(1, 'ls')
        err.output = None
        wrapped = error_wrapper(err)
        output = io.StringIO()
        traceback.print_exception(type(wrapped), wrapped, None, file=output)
        assert 'No output was generated.' in output.getvalue()
        print("test_no_output passed")

    def test_other_exception():
        err = ValueError("test error")
        wrapped = error_wrapper(err)
        assert wrapped is err
        print("test_other_exception passed")

    test_called_process_error()
    test_timeout_expired()
    test_no_output()
    test_other_exception()
    print("All tests passed!")

if __name__ == "__main__":  # pragma: no cover
    test_error_wrapper()


# LLM-generated content at query #19
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
        subprocess.run(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with other exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"
    
    print("All tests passed!")



# LLM-generated content at query #20
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test with CalledProcessError
    try:
        subprocess.run(['false'], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert 'Captured output:' in str(wrapped)
        print("Test for CalledProcessError passed.")
    
    # Test with TimeoutExpired
    try:
        subprocess.run(['sleep', '2'], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert 'Captured output:' in str(wrapped)
        print("Test for TimeoutExpired passed.")
    
    # Test with other exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert wrapped is e
        print("Test for other exception passed.")



# LLM-generated content at query #21
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test with CalledProcessError
    try:
        raise subprocess.CalledProcessError(returncode=1, cmd='ls', output=b'No such file or directory')
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped_error)
        assert "No such file or directory" in str(wrapped_error)
    
    # Test with TimeoutExpired
    try:
        raise subprocess.TimeoutExpired(cmd='sleep 10', timeout=5, output=b'Process took too long')
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped_error)
        assert "Process took too long" in str(wrapped_error)
    
    # Test with other exception (should not be wrapped)
    try:
        raise ValueError("Some other error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Some other error"
    
    print("All tests passed!")



# LLM-generated content at query #22
#--------------------------

# Unit test for function run_command
def test_run_command():


# LLM-generated content at query #23
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():


# LLM-generated content at query #24
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():


# LLM-generated content at query #25
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.run(["ls", "nonexistent"], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert "Captured output:" in str(wrapped_error)
        print("Test for CalledProcessError passed.")
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "2"], timeout=0.1, capture_output=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert "Captured output:" in str(wrapped_error)
        print("Test for TimeoutExpired passed.")
    
    # Test with other exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert wrapped_error == e
        print("Test for other exception passed.")



# LLM-generated content at query #26
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
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
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



# LLM-generated content at query #27
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import io
    import sys
    import traceback

    # Test CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, 'cmd', output=b'output')
    except subprocess.CalledProcessError as e:
        e = error_wrapper(e)
        assert e.output == b'output'
        assert 'Captured output:' in str(e)

    # Test TimeoutExpired
    try:
        raise subprocess.TimeoutExpired('cmd', 1, output=b'output')
    except subprocess.TimeoutExpired as e:
        e = error_wrapper(e)
        assert e.output == b'output'
        assert 'Captured output:' in str(e)

    # Test other exception
    try:
        raise ValueError('test')
    except ValueError as e:
        e2 = error_wrapper(e)
        assert e is e2

    print('All tests passed.')


if __name__ == '__main__':  # pragma: no cover
    test_error_wrapper()


# LLM-generated content at query #28
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Simple command execution
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"Hello, World!" in result.captured_output

    # Test 2: Command with error
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0

    # Test 3: Timeout
    try:
        run_command(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired:
        pass  # Expected

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


# LLM-generated content at query #29
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with other exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert str(wrapped_error) == "Test error"



# LLM-generated content at query #30
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(returncode=1, cmd='ls', output=b'error output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert wrapped_err.output == b'error output'
    assert 'Captured output:' in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired(cmd='sleep 10', timeout=5, output=b'timeout output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert wrapped_err.output == b'timeout output'
    assert 'Captured output:' in str(wrapped_err)

    # Test with other exception
    err = ValueError('test error')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == 'test error'

    print("All tests passed!")



# LLM-generated content at query #31
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
    
    # Test with TimeoutExpired
    try:
        subprocess.run(['sleep', '10'], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped_error)
    
    # Test with other exception
    other_error = ValueError("test")
    wrapped_error = error_wrapper(other_error)
    assert wrapped_error is other_error



# LLM-generated content at query #32
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import traceback

    def test_called_process_error():
        try:
            subprocess.check_output(['ls', 'nonexistent'], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            e = error_wrapper(e)
            print("CalledProcessError:")
            print(e)
            print()

    def test_timeout_expired():
        try:
            subprocess.run(['sleep', '10'], timeout=0.1, check=True)
        except subprocess.TimeoutExpired as e:
            e = error_wrapper(e)
            print("TimeoutExpired:")
            print(e)
            print()

    def test_other_exception():
        try:
            raise ValueError("Some other error")
        except ValueError as e:
            e = error_wrapper(e)
            print("Other exception (should be unchanged):")
            print(e)
            print()

    test_called_process_error()
    test_timeout_expired()
    test_other_exception()

if __name__ == "__main__":  # pragma: no cover
    test_error_wrapper()


# LLM-generated content at query #33
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Simple command that should succeed
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test 2: Command that should fail
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    # Test 3: Command with timeout
    try:
        run_command(["sleep", "2"], timeout=1)
    except subprocess.TimeoutExpired:
        pass  # Expected

    # Test 4: Command with environment variable
    result = run_command(["printenv", "MYVAR"], env={"MYVAR": "test"}, return_output=True)
    assert b"test" in result.captured_output

    # Test 5: Command with working directory
    import os
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert b"/tmp" in result.captured_output

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #34
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
    try:
        run_command(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired:
        pass  # Expected behavior
    else:
        assert False, "Expected TimeoutExpired exception"

    # Test 4: Command with environment variable
    result = run_command(["printenv", "MY_VAR"], env={"MY_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test 5: Command with working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test 6: Verbose mode (should not raise exceptions)
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

    # Test 7: Shell command
    result = run_command("echo Hello, World!", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #35
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Simple command that should succeed
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test 2: Command that should fail
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    # Test 3: Command with timeout
    try:
        run_command(["sleep", "2"], timeout=1)
    except subprocess.TimeoutExpired:
        pass  # Expected

    # Test 4: Command with environment variable
    result = run_command(["printenv", "MYVAR"], env={"MYVAR": "test"}, return_output=True)
    assert b"test" in result.captured_output

    # Test 5: Command with working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test 6: Command with verbose output
    # This test is a bit tricky because it involves capturing stdout
    # We'll just ensure it doesn't crash
    run_command(["echo", "test"], verbose=True)

    # Test 7: Command that returns output
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output is not None

    # Test 8: Command that doesn't return output
    result = run_command(["echo", "test"], return_output=False)
    assert result.captured_output is None

    # Test 9: Command with shell=True
    result = run_command("echo hello", shell=True, return_output=True)
    assert b"hello" in result.captured_output

    # Test 10: Command with large output (should be truncated)
    result = run_command(["python3", "-c", "print('a'*10000)"], return_output=True, ignore_errors=True)
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + 100  # Allow some overhead

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #36
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test 1: Simple command execution
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"
    print("Test 1 passed")

    # Test 2: Command with error
    result = run_command(["ls", "nonexistent_file"], ignore_errors=True, return_output=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    print("Test 2 passed")

    # Test 3: Command with timeout
    result = run_command(["sleep", "2"], timeout=1, ignore_errors=True, return_output=True)
    assert result.return_code == -32768
    print("Test 3 passed")

    # Test 4: Command with environment variable
    result = run_command(["printenv", "MY_VAR"], env={"MY_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"
    print("Test 4 passed")

    # Test 5: Command with working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.decode().strip() == tmpdir
    print("Test 5 passed")

    # Test 6: Command with shell
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"
    print("Test 6 passed")

    # Test 7: Command with verbose output
    import io
    import sys
    captured_output = io.StringIO()
    sys.stdout = captured_output
    result = run_command(["echo", "verbose test"], verbose=True, return_output=True)
    sys.stdout = sys.__stdout__
    assert "verbose test" in captured_output.getvalue()
    print("Test 7 passed")

    # Test 8: Command without returning output
    result = run_command(["echo", "no output"])
    assert result.return_code == 0
    assert result.captured_output is None
    print("Test 8 passed")

    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #37
#--------------------------

# Unit test for function run_command
def test_run_command():  
    # Test 1: Basic command execution  
    result = run_command(["echo", "Hello, World!"], return_output=True)  
    assert result.return_code == 0  
    assert b"Hello, World!" in result.captured_output  
  
    # Test 2: Command with error (non-zero return code)  
    result = run_command(["ls", "/nonexistent"], ignore_errors=True, return_output=True)  
    assert result.return_code != 0  
    assert result.captured_output is not None  
  
    # Test 3: Timeout handling  
    import time  
    start = time.time()  
    result = run_command(["sleep", "2"], timeout=1, ignore_errors=True, return_output=True)  
    elapsed = time.time() - start  
    assert elapsed < 2  # Should timeout before 2 seconds  
    assert result.return_code == -32768  # Special code for timeout  
  
    # Test 4: Environment variables  
    result = run_command(["printenv", "MY_VAR"], env={"MY_VAR": "test_value"}, return_output=True)  
    assert result.return_code == 0  
    assert b"test_value" in result.captured_output  
  
    # Test 5: Working directory  
    import tempfile  
    import os  
    with tempfile.TemporaryDirectory() as tmpdir:  
        test_file = os.path.join(tmpdir, "test.txt")  
        with open(test_file, "w") as f:  
            f.write("test")  
        result = run_command(["cat", "test.txt"], cwd=tmpdir, return_output=True)  
        assert result.return_code == 0  
        assert b"test" in result.captured_output  
  
    # Test 6: Shell command  
    result = run_command("echo Hello from shell", shell=True, return_output=True)  
    assert result.return_code == 0  
    assert b"Hello from shell" in result.captured_output  
  
    # Test 7: Verbose mode (no assertion, just ensure it doesn't crash)  
    result = run_command(["echo", "verbose test"], verbose=True, return_output=True)  
    assert result.return_code == 0  
  
    # Test 8: No output capture  
    result = run_command(["echo", "no capture"])  
    assert result.return_code == 0  
    assert result.captured_output is None  
  
    # Test 9: Exception handling (without ignore_errors)  
    import subprocess  
    try:  
        run_command(["ls", "/nonexistent"])  
        assert False, "Expected CalledProcessError"  
    except subprocess.CalledProcessError as e:  
        assert e.returncode != 0  
        assert e.output is not None  
  
    # Test 10: Exception wrapping  
    try:  
        run_command(["ls", "/nonexistent"])  
    except subprocess.CalledProcessError as e:  
        # Check that error_wrapper modified the exception  
        str_repr = str(e)  
        assert "Captured output:" in str_repr or "No output was generated." in str_repr  
  
    print("All tests passed!")  
  
if __name__ == "__main__":  
    test_run_command()


# LLM-generated content at query #38
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert "Captured output:" in str(wrapped)
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert "Captured output:" in str(wrapped)
    
    # Test with other exception
    try:
        raise ValueError("test")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert wrapped is e



# LLM-generated content at query #39
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test with CalledProcessError
    try:
        subprocess.run(["ls", "nonexistent"], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert "Captured output:" in str(wrapped_error)
        print("Test for CalledProcessError passed.")

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, capture_output=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert "Captured output:" in str(wrapped_error)
        print("Test for TimeoutExpired passed.")

    # Test with other exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert str(wrapped_error) == "Test error"
        print("Test for other exception passed.")



# LLM-generated content at query #40
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper(): 
    # Test that error_wrapper wraps subprocess.CalledProcessError
    try:
        subprocess.check_output(["ls", "nonexistent"])
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped_error)
    
    # Test that error_wrapper wraps subprocess.TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped_error)
    
    # Test that error_wrapper does not wrap other exceptions
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"



# LLM-generated content at query #41
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  # pragma: no cover
    import sys
    import traceback
    import io

    def run_and_capture(func):
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            func()
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        return output

    # Test CalledProcessError
    err = subprocess.CalledProcessError(1, 'ls', output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    output = run_and_capture(lambda: print(wrapped))
    assert 'Captured output:' in output
    assert 'file1' in output
    assert 'file2' in output

    # Test TimeoutExpired
    err = subprocess.TimeoutExpired('sleep 10', 5, output=b'still running...')
    wrapped = error_wrapper(err)
    output = run_and_capture(lambda: print(wrapped))
    assert 'Captured output:' in output
    assert 'still running...' in output

    # Test other exception
    exc = ValueError('test')
    wrapped = error_wrapper(exc)
    assert wrapped is exc

    print('All tests passed!')

if __name__ == '__main__':  # pragma: no cover
    test_error_wrapper()


# LLM-generated content at query #42
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():  
    # Test with CalledProcessError
    try:
        subprocess.run(["ls", "nonexistent"], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert "Captured output:" in str(wrapped_error)
        print("Test for CalledProcessError passed.")
    
    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, capture_output=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert "Captured output:" in str(wrapped_error)
        print("Test for TimeoutExpired passed.")
    
    # Test with other exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert str(wrapped_error) == "Test error"
        print("Test for other exception passed.")



# LLM-generated content at query #43
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():


# LLM-generated content at query #44
#--------------------------

# Unit test for function run_command
def test_run_command(): 
    # Test case 1: Successful command execution
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"Hello, World!\n"
    
    # Test case 2: Command with error
    result = run_command(["ls", "nonexistent_file"], ignore_errors=True)
    assert result.return_code != 0
    
    # Test case 3: Timeout
    result = run_command(["sleep", "5"], timeout=1, ignore_errors=True)
    assert result.return_code == -32768
    
    # Test case 4: Environment variable
    result = run_command(["printenv", "MY_VAR"], env={"MY_VAR": "test_value"}, return_output=True)
    assert result.captured_output == b"test_value\n"
    
    # Test case 5: Working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.captured_output.decode().strip() == tmpdir
    
    print("All tests passed!")

if __name__ == "__main__":
    test_run_command()


# LLM-generated content at query #45
#--------------------------

# Unit test for function error_wrapper
def test_error_wrapper():


