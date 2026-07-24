####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    Error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", 10)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 10 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #2
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "test output" in str(wrapped_error)

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_error)
    assert "timeout output" in str(wrapped_error)

    # Test with non-subprocess exception
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with no output
    error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_error)

    # Test with non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #3
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_err) or "No output was generated" in str(wrapped_err)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_err) or "No output was generated" in str(wrapped_err)

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, ValueError)
        assert str(wrapped_err) == "Test error"


# LLM-generated content at query #4
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with timeout
    result = run_command("sleep 2", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test verbose mode
    result = run_command("echo 'verbose test'", verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose test" in result.captured_output

    # Test command as list
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test error wrapper
    try:
        run_command("exit 1")
    except subprocess.CalledProcessError as e:
        assert "Captured output:" in str(e)


# LLM-generated content at query #5
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

    # Test command with ignore_errors=True
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test command with timeout and ignore_errors=True
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test verbose mode
    with patch('builtins.print') as mock_print:
        run_command("echo 'test'", verbose=True)
        mock_print.assert_called()

    # Test command with return_output=False
    result = run_command("echo 'test'", return_output=False)
    assert result.captured_output is None

    # Test command with list of arguments
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.command == ["echo", "Hello, World!"]
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output


# LLM-generated content at query #6
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test successful command execution with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

    # Test command execution with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command execution with return_output=True and verbose=True
    result = run_command(["echo", "hello"], return_output=True, verbose=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

    # Test command execution with ignore_errors=True and non-zero return code
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command execution with ignore_errors=True and timeout
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command execution with environment variables
    result = run_command(["echo", "$TEST_VAR"], env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

    # Test command execution with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

    # Test command execution with timeout and ignore_errors=False
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1, ignore_errors=False)

    # Test command execution with non-zero return code and ignore_errors=False
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent"], ignore_errors=False)

    # Test command execution with long output
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], return_output=True)
    assert result.return_code == 0
    assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #7
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with timeout
    result = run_command("sleep 2", timeout=1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command("echo $TEST_VAR", shell=True, env=env, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test command with verbose mode
    result = run_command("echo 'Verbose test'", verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"Verbose test" in result.captured_output

    # Test command with custom working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(f"echo '{long_output}'", return_output=True)
    assert result.return_code == 0
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len("*** (previous output truncated) ***\n")
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #8
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #9
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #10
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "test output" in str(wrapped_error)

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "timeout output" in str(wrapped_error)

    # Test with non-subprocess exception
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with CalledProcessError and no output
    error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "No output was generated" in str(wrapped_error)

    # Test with TimeoutExpired and no output
    error = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "No output was generated" in str(wrapped_error)

    # Test with CalledProcessError and non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Failed to parse output" in str(wrapped_error)


# LLM-generated content at query #11
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #12
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0

    # Test command with ignore_errors=True
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command(["sh", "-c", "echo $TEST_VAR"], env=env, return_output=True)
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test command that fails
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent"])

    # Test command with long output
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #13
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with TimeoutExpired and no output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #14
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    Test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Test timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    Test timeout output"

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err == err
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #15
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo hello")
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command("echo hello", return_output=True)
    assert result.captured_output == b"hello\n"

    # Test command with verbose=True
    result = run_command("echo hello", verbose=True)
    assert result.return_code == 0

    # Test command with ignore_errors=True
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test command with timeout
    result = run_command("sleep 2", timeout=1, ignore_errors=True)
    assert result.return_code == -32768

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test command with list of arguments
    result = run_command(["echo", "hello"])
    assert result.return_code == 0

    # Test command that fails
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout that raises exception
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", timeout=1)

    # Test command with very long output
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(f"echo {long_output}", ignore_errors=True, shell=True)
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #16
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_error)
        assert wrapped_error.output is not None

    # Test with subprocess.TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_error)
        assert wrapped_error.output is not None

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"

    # Test with subprocess.CalledProcessError without output
    try:
        subprocess.run(["false"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "No output was generated" in str(wrapped_error)
        assert wrapped_error.output is None


# LLM-generated content at query #17
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped) or "No output was generated." in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.01, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped) or "No output was generated." in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #18
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #19
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #20
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with subprocess.TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with subprocess.CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with subprocess.TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with subprocess.CalledProcessError with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #21
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    Error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #22
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #23
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    test output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with CalledProcessError with non-utf8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #24
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #25
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        e.output = b"Test output"
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped)
        assert "Test output" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        e.output = b"Timeout output"
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped)
        assert "Timeout output" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "Test error"

    # Test with no output
    try:
        subprocess.run(["false"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        e.output = None
        wrapped = error_wrapper(e)
        assert "No output was generated" in str(wrapped)

    # Test with non-UTF-8 output
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        e.output = b"\xff\xfe"
        wrapped = error_wrapper(e)
        assert "Failed to parse output" in str(wrapped)


# LLM-generated content at query #26
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with timeout
    result = run_command("sleep 2", shell=True, timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command("echo $TEST_VAR", shell=True, env=env, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test command with verbose mode
    result = run_command("echo 'Verbose test'", shell=True, verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"Verbose test" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", shell=True, cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test command with list args
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with ignore_errors and exception
    result = run_command("false", ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


# LLM-generated content at query #27
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output.strip() == b"hello"

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent_directory"])

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with ignore_errors=True
    result = run_command(["ls", "/nonexistent_directory"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with verbose=True (capturing log output is tricky, so we just check it doesn't raise)
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0

    # Test command with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.captured_output.strip() == tmpdir.encode()

    # Test command with return_output=False (default)
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with string argument (shell=True)
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output.strip() == b"hello"


# LLM-generated content at query #28
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #29
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #30
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "Hello, World!"

    # Test command with non-zero return code
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with timeout
    result = run_command("sleep 2", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert "test_value" in result.captured_output.decode('utf-8')

    # Test command with verbose mode
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", verbose=True)

    # Test command with return_output=False
    result = run_command("echo 'Test'", return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert tmpdir in result.captured_output.decode('utf-8')


# LLM-generated content at query #31
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_err) or "No output was generated" in str(wrapped_err)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_err) or "No output was generated" in str(wrapped_err)

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, ValueError)
        assert str(wrapped_err) == "Test error"


# LLM-generated content at query #32
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello world"], return_output=True)
    assert result.command == ["echo", "hello world"]
    assert result.return_code == 0
    assert result.captured_output == b"hello world\n"

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent_directory"])

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with ignore_errors
    result = run_command(["ls", "/nonexistent_directory"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with verbose
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent_directory"], verbose=True)

    # Test command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command(["sh", "-c", "echo $TEST_VAR"], env=env, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

    # Test command with return_output=False
    result = run_command(["echo", "hello world"])
    assert result.command == ["echo", "hello world"]
    assert result.return_code == 0
    assert result.captured_output is None


# LLM-generated content at query #33
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo hello")
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command("echo hello", return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with verbose=True
    with patch("builtins.print") as mock_print:
        result = run_command("echo hello", verbose=True)
        assert mock_print.called

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True)
    assert result.return_code == 0

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir)
        assert result.return_code == 0

    # Test command with timeout
    result = run_command("sleep 0.1", timeout=1)
    assert result.return_code == 0

    # Test command with ignore_errors=True
    result = run_command("false", ignore_errors=True)
    assert result.return_code != 0

    # Test command with error and return_output=True
    result = run_command("false", return_output=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout error and ignore_errors=True
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test command with error and ignore_errors=False
    with pytest.raises(subprocess.CalledProcessError):
        run_command("false", ignore_errors=False)

    # Test command with timeout error and ignore_errors=False
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1, ignore_errors=False)

    # Test command with long output
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(f"echo {long_output}", return_output=True, ignore_errors=True)
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #34
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #35
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", shell=True, return_output=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", shell=True)

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", shell=True, timeout=0.1)

    # Test command with ignore_errors=True
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.return_code == 1

    # Test command with timeout and ignore_errors=True
    result = run_command("sleep 2", shell=True, timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test command with verbose=True
    with patch('builtins.print') as mock_print:
        run_command("echo 'Verbose'", shell=True, verbose=True)
        mock_print.assert_called()

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", shell=True, cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command with return_output=False (default)
    result = run_command("echo 'No Output'", shell=True)
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command("echo 'With Output'", shell=True, return_output=True)
    assert b"With Output" in result.captured_output


# LLM-generated content at query #36
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=10)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert str(wrapped_error) == "Command 'test_command' timed out after 10 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with CalledProcessError and no output
    error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with CalledProcessError and non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #37
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command")
    err.output = b"Error output"
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", 1)
    err.output = b"Timeout output"
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    err.output = None
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command")
    err.output = b"\xff\xfe"
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #38
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped)

    # Test with subprocess.TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #39
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:\n    test output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:\n    timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with CalledProcessError with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #40
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    Error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with other exception types
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."

    # Test with CalledProcessError and no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."


# LLM-generated content at query #41
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #42
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0

    # Test command with ignore_errors=True
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command with error_wrapper
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(["false"])
    assert "Captured output" in str(excinfo.value)

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(["python", "-c", f"print('{long_output}')"], ignore_errors=False)
    assert b"*** (previous output truncated) ***" in excinfo.value.output


# LLM-generated content at query #43
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err == err
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #44
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello world"], return_output=True)
    assert result.command == ["echo", "hello world"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "hello world"

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent_directory"])

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with ignore_errors=True
    result = run_command(["ls", "/nonexistent_directory"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with verbose=True
    with patch('builtins.print') as mock_print:
        run_command(["echo", "test"], verbose=True)
        mock_print.assert_called()

    # Test command with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.captured_output.decode('utf-8').strip() == tmpdir

    # Test command with return_output=False
    result = run_command(["echo", "hello world"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with shell=True
    result = run_command("echo hello shell", shell=True, return_output=True)
    assert result.command == "echo hello shell"
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "hello shell"


# LLM-generated content at query #45
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with other exception types
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #46
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #47
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #48
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "error output" in str(wrapped_error)

    # Test with subprocess.TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_error)

    # Test with non-subprocess exception
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with subprocess.CalledProcessError and non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #49
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #50
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #51
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with subprocess.TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with subprocess.CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with subprocess.TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with subprocess.CalledProcessError with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #52
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #53
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

    # Test command with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with ignore_errors=True
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with ignore_errors=True and timeout
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.command == ["sleep", "10"]
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["env"]
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.command == ["pwd"]
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

    # Test command with string argument
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"


# LLM-generated content at query #54
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    test output"

    # Test with other exception types
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #55
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with subprocess.TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"


# LLM-generated content at query #56
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

    # Test command with ignore_errors=True
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with verbose=True
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent"], verbose=True)

    # Test command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command(["bash", "-c", "echo $TEST_VAR"], env=env, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

    # Test command with verbose and return_output
    result = run_command(["echo", "verbose"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"verbose\n"


# LLM-generated content at query #57
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello\n" in result.captured_output

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent"])

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test ignore_errors flag
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test verbose flag
    with patch('builtins.print') as mock_print:
        run_command(["echo", "hello"], verbose=True)
        mock_print.assert_called()

    # Test return_output flag
    result = run_command(["echo", "hello"], return_output=True)
    assert result.captured_output is not None

    # Test environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command as string
    result = run_command("echo hello", shell=True, return_output=True)
    assert b"hello\n" in result.captured_output


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped) or "No output was generated." in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped) or "No output was generated." in str(wrapped)

    # Test with other exceptions (should return unchanged)
    other_error = ValueError("test error")
    assert error_wrapper(other_error) is other_error


# LLM-generated content at query #2
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #3
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with CalledProcessError with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #4
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0

    # Test command with ignore_errors=True
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.01)

    # Test command with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command that fails
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent"])

    # Test command with long output truncation
    long_output = "x" * (MAX_OUTPUT_LENGTH + 100)
    with tempfile.NamedTemporaryFile() as f:
        f.write(long_output.encode('utf-8'))
        f.flush()
        result = run_command(["cat", f.name], ignore_errors=True)
        assert result.return_code == 0
        assert b"*** (previous output truncated) ***" in result.captured_output
        assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #5
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command execution with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

    # Test command execution with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command execution with ignore_errors=True
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command execution with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command execution with environment variables
    result = run_command(["echo", "$TEST_VAR"], env={"TEST_VAR": "test_value"})
    assert result.return_code == 0

    # Test command execution with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir)
        assert result.return_code == 0

    # Test command execution with shell=True
    result = run_command("echo hello", shell=True)
    assert result.return_code == 0

    # Test command execution with non-existent command
    with pytest.raises(FileNotFoundError):
        run_command(["non_existent_command"])


# LLM-generated content at query #6
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err).startswith("Command 'test_command' returned non-zero exit status 1.")
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err).startswith("Command 'test_command' timed out after 1 seconds.")
    assert "No output was generated." in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"


# LLM-generated content at query #7
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

    # Test ignore_errors flag
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test verbose flag (captures output)
    result = run_command("echo 'Verbose test'", verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"Verbose test" in result.captured_output

    # Test environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command as list
    result = run_command(["echo", "Hello, List!"], return_output=True)
    assert result.return_code == 0
    assert b"Hello, List!" in result.captured_output

    # Test long output truncation
    long_output = "x" * (MAX_OUTPUT_LENGTH + 1000)
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command(f"echo '{long_output}'", shell=True)
    assert b"*** (previous output truncated) ***" in exc_info.value.output


# LLM-generated content at query #8
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #9
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

    # Test command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command("echo $TEST_VAR", shell=True, env=env, return_output=True)
    assert b"test_value" in result.captured_output

    # Test command with custom working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test command with verbose mode
    with patch('builtins.print') as mock_print:
        run_command("echo 'verbose test'", verbose=True)
        mock_print.assert_called()

    # Test command with ignore_errors
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test command with return_output
    result = run_command("echo 'return output test'", return_output=True)
    assert b"return output test" in result.captured_output

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(f"echo '{long_output}'", return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #10
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #11
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with subprocess.TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #12
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output.strip() == b"hello"

    # Test command with non-zero return code
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with timeout and ignore_errors
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.captured_output.strip() == tmpdir.encode()

    # Test verbose mode (capturing log output is more involved and may require additional setup)
    result = run_command(["echo", "verbose"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output.strip() == b"verbose"

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(["echo", long_output], return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")

    # Test error_wrapper functionality
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command(["ls", "/nonexistent"])
    assert "Captured output:" in str(exc_info.value)

    # Test with string command (shell=True)
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output.strip() == b"hello"


# LLM-generated content at query #13
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #14
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_e = error_wrapper(e)
        assert isinstance(wrapped_e, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_e) or "No output was generated" in str(wrapped_e)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_e = error_wrapper(e)
        assert isinstance(wrapped_e, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_e) or "No output was generated" in str(wrapped_e)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped_e = error_wrapper(e)
        assert isinstance(wrapped_e, ValueError)
        assert str(wrapped_e) == "test error"


# LLM-generated content at query #15
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    result = run_command("exit 1", shell=True, return_output=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", shell=True, timeout=0.1)

    # Test command with ignore_errors=True
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with verbose=True
    result = run_command("echo 'Verbose test'", shell=True, verbose=True)
    assert result.return_code == 0

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", shell=True, cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(f"echo '{long_output}'", shell=True, return_output=True)
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len("*** (previous output truncated) ***\n")
    assert b"*** (previous output truncated) ***" in result.captured_output

    # Test command with UnicodeDecodeError handling in verbose mode
    result = run_command("echo -e '\\x00\\x01\\x02'", shell=True, verbose=True, return_output=True)
    assert result.return_code == 0


# LLM-generated content at query #16
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped) or "No output was generated." in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped) or "No output was generated." in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #17
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "error output" in str(wrapped_error)

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "timeout output" in str(wrapped_error)

    # Test with non-subprocess exception
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with CalledProcessError and no output
    error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "No output was generated" in str(wrapped_error)

    # Test with TimeoutExpired and no output
    error = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "No output was generated" in str(wrapped_error)


# LLM-generated content at query #18
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "test error"


# LLM-generated content at query #19
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.01, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #20
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with other exception types
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #21
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with TimeoutExpired and no output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #22
#--------------------------

```python
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
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"

    # Test with CalledProcessError without output
    try:
        subprocess.run(["false"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "No output was generated." in str(wrapped)


# LLM-generated content at query #23
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello world"], return_output=True)
    assert result.return_code == 0
    assert b"hello world" in result.captured_output

    # Test command with non-zero return code
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test verbose mode
    with pytest.mock.patch('builtins.print') as mock_print:
        run_command(["echo", "test"], verbose=True)
        assert mock_print.called

    # Test command with return_output=False
    result = run_command(["echo", "test"], return_output=False)
    assert result.captured_output is None

    # Test command with ignore_errors=True and timeout
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    with pytest.mock.patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, "test")) as mock_run:
        mock_run.return_value.stdout = long_output.encode()
        result = run_command(["test"], ignore_errors=True)
        assert b"*** (previous output truncated) ***" in result.captured_output
        assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #24
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #25
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "Error output" in str(wrapped_error)

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_error)

    # Test with non-subprocess exception
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #26
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output.strip() == b"hello"

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent"])

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test ignore_errors flag
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test verbose flag
    with patch('builtins.print') as mock_print:
        run_command(["echo", "test"], verbose=True)
        mock_print.assert_called()

    # Test return_output flag
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output.strip() == b"test"

    # Test environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test string command
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.captured_output.strip() == b"hello"


# LLM-generated content at query #27
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'test'", return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

    # Test command with non-zero return code
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test command with timeout
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test verbose mode
    result = run_command("echo 'verbose test'", verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"verbose test\n"

    # Test environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

    # Test command as list
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

    # Test error wrapper
    try:
        run_command("exit 1")
    except subprocess.CalledProcessError as e:
        assert "Captured output:" in str(e)


# LLM-generated content at query #28
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent"])

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test ignore_errors flag
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test verbose flag
    with patch('builtins.print') as mock_print:
        run_command(["echo", "hello"], verbose=True)
        mock_print.assert_called()

    # Test return_output flag
    result = run_command(["echo", "hello"], return_output=True)
    assert result.captured_output == b"hello\n"

    # Test environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command as string
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

    # Test error wrapper
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command(["ls", "/nonexistent"])
    assert "Captured output" in str(exc_info.value)


# LLM-generated content at query #29
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    original_error = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with subprocess.TimeoutExpired
    original_error = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert str(wrapped_error) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with other exception types (should return unchanged)
    original_error = ValueError("test error")
    wrapped_error = error_wrapper(original_error)
    assert wrapped_error is original_error

    # Test with unicode decode error
    original_error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #30
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #31
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "hello"

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent"])

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with ignore_errors=True
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with verbose=True
    with patch('builtins.print') as mock_print:
        result = run_command(["echo", "hello"], verbose=True)
        mock_print.assert_called()

    # Test command with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.captured_output.decode('utf-8').strip() == tmpdir

    # Test command with return_output=False
    result = run_command(["echo", "hello"], return_output=False)
    assert result.captured_output is None

    # Test command with verbose=True and return_output=True
    with patch('builtins.print') as mock_print:
        result = run_command(["echo", "hello"], verbose=True, return_output=True)
        mock_print.assert_called()
        assert result.captured_output.decode('utf-8').strip() == "hello"

    # Test command with ignore_errors=True and timeout
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


# LLM-generated content at query #32
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with TimeoutExpired and no output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #33
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=10)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 10 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #34
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)


# LLM-generated content at query #35
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #36
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_error)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_error)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "test error"


# LLM-generated content at query #37
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "Error output" in str(wrapped_error)

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_error)
    assert "Timeout output" in str(wrapped_error)

    # Test with non-subprocess exception
    error = ValueError("Test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "Test error"

    # Test with CalledProcessError without output
    error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_error)

    # Test with TimeoutExpired without output
    error = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_error)

    # Test with non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #38
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "Hello, World!"

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

    # Test ignore_errors flag
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test verbose flag
    with patch('builtins.print') as mock_print:
        run_command("echo 'Verbose test'", verbose=True)
        mock_print.assert_called()

    # Test return_output flag
    result = run_command("echo 'Return output test'", return_output=True)
    assert result.captured_output is not None
    assert result.return_code == 0

    # Test environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert "test_value" in result.captured_output.decode('utf-8')

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert tmpdir in result.captured_output.decode('utf-8')

    # Test long output truncation
    long_output = "x" * (MAX_OUTPUT_LENGTH + 1000)
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "test_cmd", output=long_output.encode())
        result = run_command("test_cmd", ignore_errors=True)
        assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #39
#--------------------------

```python
def test_run_command():
    # Test successful command execution with return_output=True
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

    # Test successful command execution with return_output=False
    result = run_command(["echo", "test"], return_output=False)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command execution with non-zero return code
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command execution with timeout
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.command == ["sleep", "10"]
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command execution with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["env"]
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test command execution with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.command == ["pwd"]
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

    # Test command execution with verbose mode
    result = run_command(["echo", "verbose_test"], verbose=True, return_output=True)
    assert result.command == ["echo", "verbose_test"]
    assert result.return_code == 0
    assert result.captured_output == b"verbose_test\n"

    # Test command execution with string command
    result = run_command("echo string_test", shell=True, return_output=True)
    assert result.command == "echo string_test"
    assert result.return_code == 0
    assert result.captured_output == b"string_test\n"


# LLM-generated content at query #40
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "error output" in str(wrapped_error)

    # Test with subprocess.TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_error)

    # Test with non-subprocess exception
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with subprocess.CalledProcessError and non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #41
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #42
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #43
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #44
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

    # Test command with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is not None

    # Test command with ignore_errors=True and non-zero return code
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.command == ["ls", "/nonexistent"]
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with ignore_errors=True and timeout
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.command == ["sleep", "10"]
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command(["sh", "-c", "echo $TEST_VAR"], env=env)
    assert result.command == ["sh", "-c", "echo $TEST_VAR"]
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir)
        assert result.command == ["pwd"]
        assert result.return_code == 0
        assert result.captured_output is None

    # Test command with shell=True
    result = run_command("echo hello", shell=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with long output
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(["echo", long_output], ignore_errors=True)
    assert result.command == ["echo", long_output]
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #45
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #46
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    Error output"

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert str(wrapped_error) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    error = ValueError("Test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "Test error"

    # Test with CalledProcessError and non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #47
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #48
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.01, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test"


# LLM-generated content at query #49
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #50
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with timeout
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test verbose mode
    result = run_command("echo 'verbose test'", verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose test" in result.captured_output

    # Test command that generates no output
    result = run_command("true", return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b""

    # Test command with very long output (truncation)
    long_output = "x" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(f"echo '{long_output}'", ignore_errors=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len("*** (previous output truncated) ***\n")


# LLM-generated content at query #51
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command(["ls", "/nonexistent_directory"])
    assert exc_info.value.returncode != 0
    assert "No such file or directory" in str(exc_info.value)

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        run_command(["sleep", "10"], timeout=0.1)
    assert exc_info.value.timeout == 0.1

    # Test command with ignore_errors=True
    result = run_command(["ls", "/nonexistent_directory"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with verbose=True
    with pytest.mock.patch('builtins.print') as mock_print:
        run_command(["echo", "verbose"], verbose=True)
        assert mock_print.called

    # Test command with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command with return_output=False
    result = run_command(["echo", "no_output"])
    assert result.captured_output is None

    # Test command with shell=True
    result = run_command("echo shell", shell=True, return_output=True)
    assert result.captured_output == b"shell\n"


# LLM-generated content at query #52
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

    # Test ignore_errors flag
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test verbose flag (capture output to verify)
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", verbose=True)

    # Test environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert b"test_value" in result.captured_output

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        run_command(f"touch test.txt", cwd=tmpdir, shell=True)
        assert os.path.exists(test_file)

    # Test command as list
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command(f"echo '{long_output}'", shell=True)
    assert b"*** (previous output truncated) ***" in exc_info.value.output


# LLM-generated content at query #53
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with return_output=False
    result = run_command("echo 'Hello, World!'", return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with non-zero return code
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with timeout
    result = run_command("sleep 2", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command("echo $TEST_VAR", shell=True, env=env, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test verbose mode
    result = run_command("echo 'Verbose test'", verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"Verbose test" in result.captured_output

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(f"echo '{long_output}'", return_output=True)
    assert result.return_code == 0
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len("*** (previous output truncated) ***\n")
    assert b"*** (previous output truncated) ***" in result.captured_output

    # Test command with Unicode output
    result = run_command("echo 'Unicode: ñ'", return_output=True)
    assert result.return_code == 0
    assert b"Unicode: \xc3\xb1" in result.captured_output


# LLM-generated content at query #54
#--------------------------

```python
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
        subprocess.run(["sleep", "10"], timeout=0.01, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped_error)

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"


# LLM-generated content at query #55
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1, output=b"test output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert str(wrapped_error) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    test output"

    # Test with non-subprocess exception
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with no output
    error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #56
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    Error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=10)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 10 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #57
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.01, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #58
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", shell=True, return_output=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "Hello, World!"

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", shell=True)

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", shell=True, timeout=0.1)

    # Test command with ignore_errors
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.return_code == 1

    # Test command with timeout and ignore_errors
    result = run_command("sleep 2", shell=True, timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test command with verbose
    result = run_command("echo 'Verbose'", shell=True, verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "Verbose"

    # Test command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command("echo $TEST_VAR", shell=True, env=env, return_output=True)
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "test_value"

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", shell=True, cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir


# LLM-generated content at query #59
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert str(wrapped_error) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #60
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"Hello, World!" in result.captured_output

    # Test command execution with error
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command("exit 1")
    assert exc_info.value.returncode == 1
    assert exc_info.value.output is not None

    # Test command execution with timeout
    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        run_command("sleep 2", timeout=0.1)
    assert exc_info.value.returncode == -32768
    assert exc_info.value.output is not None

    # Test command execution with ignore_errors
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command execution with verbose
    result = run_command("echo 'Verbose'", verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"Verbose" in result.captured_output

    # Test command execution with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"test_value" in result.captured_output

    # Test command execution with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output is not None
        assert tmpdir.encode() in result.captured_output

    # Test command execution with list of arguments
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"Hello, World!" in result.captured_output

    # Test command execution with no output
    result = run_command("echo 'No Output'", return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None


# LLM-generated content at query #61
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with CalledProcessError with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #62
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #63
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"


# LLM-generated content at query #64
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_error)
        assert wrapped_error.returncode == 1

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_error)

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"


# LLM-generated content at query #65
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #66
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with CalledProcessError with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #67
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #68
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0

    # Test command with ignore_errors=True
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "2"], timeout=0.1)

    # Test command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command(["bash", "-c", "echo $TEST_VAR"], env=env, return_output=True)
    assert b"test_value" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test command with error_wrapper
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command(["ls", "/nonexistent"])
    assert "Captured output:" in str(exc_info.value)

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], ignore_errors=True)
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #69
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"


# LLM-generated content at query #70
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with subprocess.TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with subprocess.CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with subprocess.TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with subprocess.CalledProcessError with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #71
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with subprocess.TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"


# LLM-generated content at query #72
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo hello")
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command("echo hello", return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with verbose=True
    result = run_command("echo hello", verbose=True)
    assert result.return_code == 0

    # Test command with ignore_errors=True
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", timeout=0.1)

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True)
    assert result.return_code == 0

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir)
        assert result.return_code == 0

    # Test command with long output truncation
    long_output = "echo " + "a" * 10000
    result = run_command(long_output, return_output=True)
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #73
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #74
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", timeout=0.1)

    # Test ignore_errors flag
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test verbose flag
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", verbose=True)

    # Test environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert b"test_value" in result.captured_output

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test command as list
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with no output
    result = run_command("true")
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with large output
    large_output = "x" * 10000
    result = run_command(f"echo '{large_output}'", return_output=True)
    assert result.return_code == 0
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len("*** (previous output truncated) ***\n")


# LLM-generated content at query #75
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #76
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #77
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #78
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #79
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with timeout
    result = run_command("sleep 2", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test verbose mode
    result = run_command("echo 'verbose test'", verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose test" in result.captured_output

    # Test command that generates no output
    result = run_command("true", return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b""

    # Test command with long output truncation
    long_output = "x" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(f"echo '{long_output}'", return_output=True)
    assert result.return_code == 0
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len("*** (previous output truncated) ***\n")
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #80
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #81
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #82
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "Error output" in str(wrapped_error)

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_error)
    assert "Timeout output" in str(wrapped_error)

    # Test with non-subprocess exception
    error = ValueError("Test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "Test error"

    # Test with no output
    error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(error)
    assert "No output was generated." in str(wrapped_error)

    # Test with non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #83
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "test output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_cmd", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated" in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_cmd", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated" in str(wrapped_err)


# LLM-generated content at query #84
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #85
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #86
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_cmd")
    err.output = b"Error output"
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_cmd", 10)
    err.output = b"Timeout output"
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_cmd")
    err.output = None
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_cmd")
    err.output = b"\xff\xfe"
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #87
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    err.output = b"timeout output"
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #88
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #89
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "hello"

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent"], ignore_errors=False)

    # Test command with ignore_errors=True
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1, ignore_errors=False)

    # Test command with timeout and ignore_errors=True
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with verbose=True
    result = run_command(["echo", "verbose"], verbose=True, return_output=True)
    assert result.command == ["echo", "verbose"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "verbose"

    # Test command with environment variables
    result = run_command(["echo", "$TEST_VAR"], env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.command == ["echo", "$TEST_VAR"]
    assert result.return_code == 0
    assert "test_value" in result.captured_output.decode('utf-8')

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir in result.captured_output.decode('utf-8')

    # Test command with return_output=False
    result = run_command(["echo", "no_output"], return_output=False)
    assert result.command == ["echo", "no_output"]
    assert result.return_code == 0
    assert result.captured_output is None


# LLM-generated content at query #90
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'test'", return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", timeout=0.1)

    # Test command with ignore_errors
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test command with timeout and ignore_errors
    result = run_command("sleep 2", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test command with verbose
    with pytest.mock.patch('builtins.print') as mock_print:
        run_command("echo 'verbose'", verbose=True)
        mock_print.assert_called()

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.captured_output == b"test_value\n"

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.captured_output.decode('utf-8').strip() == tmpdir

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(f"echo '{long_output}'", return_output=True)
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len("*** (previous output truncated) ***\n")
    assert result.captured_output.endswith(b"a\n")

    # Test command with Unicode decode error handling
    with pytest.mock.patch('builtins.print') as mock_print:
        run_command("echo 'binary data'", verbose=True)
        mock_print.assert_called()


# LLM-generated content at query #91
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #92
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.01, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"


# LLM-generated content at query #93
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0

    # Test command with non-zero exit code
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent"])

    # Test command with ignore_errors=True
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with timeout and ignore_errors=True
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"})
    assert result.return_code == 0

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir)
        assert result.return_code == 0

    # Test command with long output (truncation)
    long_output = "x" * (MAX_OUTPUT_LENGTH + 1000)
    with tempfile.NamedTemporaryFile() as f:
        f.write(long_output.encode())
        f.flush()
        result = run_command(["cat", f.name], ignore_errors=True)
        assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #94
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1.0, output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    test output"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe\xfd")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #95
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #96
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert result.captured_output == b"Hello, World!\n"

    # Test command execution with list of arguments
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.command == ["echo", "Hello, World!"]
    assert result.return_code == 0
    assert result.captured_output == b"Hello, World!\n"

    # Test command execution with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

    # Test command execution with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

    # Test command execution with timeout
    result = run_command("sleep 0.1", timeout=0.2)
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command execution with timeout expired
    try:
        run_command("sleep 1", timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.returncode == -32768
        assert e.output is not None

    # Test command execution with ignore_errors
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command execution with verbose
    result = run_command("echo 'Verbose test'", verbose=True)
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command execution with return_output
    result = run_command("echo 'Return output test'", return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"Return output test\n"

    # Test command execution with non-zero return code
    try:
        run_command("exit 1")
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1
        assert e.output is not None


# LLM-generated content at query #97
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #98
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #99
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #100
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #101
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #102
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped) or "No output was generated." in str(wrapped)

    # Test with subprocess.TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped) or "No output was generated." in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #103
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

    # Test command with ignore_errors=True
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with verbose=True
    result = run_command("echo 'Verbose test'", verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"Verbose test" in result.captured_output

    # Test command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command("echo $TEST_VAR", shell=True, env=env, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test command with custom working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command with very long output (truncation)
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(f"echo '{long_output}'", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")

    # Test command with Unicode output
    result = run_command("echo 'Unicode: ñ'", return_output=True)
    assert result.return_code == 0
    assert "Unicode: ñ" in result.captured_output.decode('utf-8')


# LLM-generated content at query #104
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.01, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #105
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", 10)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert str(wrapped_error) == "Command 'test_command' timed out after 10 seconds.\nNo output was generated."

    # Test with other exception types
    error = ValueError("test_error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test_error"

    # Test with output in CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"test_output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test_output"

    # Test with output in TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", 10, output=b"test_output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert str(wrapped_error) == "Command 'test_command' timed out after 10 seconds.\nCaptured output:\n    test_output"

    # Test with non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #106
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err == err
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #107
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:\n    Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:\n    Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #108
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    original_error = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "error output" in str(wrapped_error)

    # Test with TimeoutExpired
    original_error = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_error)
    assert "timeout output" in str(wrapped_error)

    # Test with non-subprocess exception
    original_error = ValueError("test error")
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with CalledProcessError without output
    original_error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_error)

    # Test with TimeoutExpired without output
    original_error = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_error)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with timeout
    result = run_command("sleep 2", shell=True, timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command("echo $TEST_VAR", shell=True, env=env, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", shell=True, cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test verbose mode
    result = run_command("echo 'Verbose test'", shell=True, verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"Verbose test" in result.captured_output

    # Test command as list
    result = run_command(["echo", "Hello, List!"], return_output=True)
    assert result.return_code == 0
    assert b"Hello, List!" in result.captured_output

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(f"echo '{long_output}'", shell=True, ignore_errors=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #2
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test successful command with output capture
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent"])

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with ignore_errors
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with verbose mode
    result = run_command(["echo", "verbose"], verbose=True)
    assert result.return_code == 0

    # Test command with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    with tempfile.NamedTemporaryFile() as f:
        f.write(long_output.encode('utf-8'))
        f.flush()
        result = run_command(["cat", f.name], return_output=True)
        assert b"*** (previous output truncated) ***" in result.captured_output
        assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #3
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "error output" in str(wrapped_error)

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=10)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_error)

    # Test with non-subprocess error
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #4
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #5
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"test output")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    test output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "test error"

    # Test with empty output
    err = subprocess.CalledProcessError(1, "test_command", output=None)
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #6
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #7
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0

    # Test command with ignore_errors=True
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with environment variables
    result = run_command(["echo", "$TEST_VAR"], env={"TEST_VAR": "test_value"}, shell=True)
    assert result.return_code == 0

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir)
        assert result.return_code == 0

    # Test command with Unicode output
    result = run_command(["echo", "こんにちは"], return_output=True)
    assert result.return_code == 0
    assert "こんにちは" in result.captured_output.decode('utf-8')

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command(["echo", long_output], ignore_errors=False)
    assert b"*** (previous output truncated) ***" in exc_info.value.output

    # Test command with OSError retry
    with pytest.raises(OSError):
        run_command(["nonexistent_command_xyz123"])

    # Test command with custom kwargs
    result = run_command(["echo", "hello"], text=True)
    assert result.return_code == 0


# LLM-generated content at query #8
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #9
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command execution with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command execution with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

    # Test command execution with ignore_errors
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test command execution with return_output
    result = run_command("echo 'Test'", return_output=True)
    assert result.return_code == 0
    assert b"Test" in result.captured_output

    # Test command execution with verbose
    result = run_command("echo 'Verbose'", verbose=True)
    assert result.return_code == 0

    # Test command execution with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test command execution with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert bytes(tmpdir, 'utf-8') in result.captured_output


# LLM-generated content at query #10
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test successful command execution with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command execution with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0

    # Test command execution with ignore_errors=True
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command execution with timeout
    result = run_command(["sleep", "1"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test command execution with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test command execution with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test command execution with error_wrapper
    try:
        run_command(["ls", "/nonexistent"])
    except subprocess.CalledProcessError as e:
        assert "Captured output:" in str(e)


# LLM-generated content at query #11
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #12
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped_err)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped_err)

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, ValueError)
        assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError without output
    try:
        subprocess.run(["false"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, subprocess.CalledProcessError)
        assert "No output was generated." in str(wrapped_err)


# LLM-generated content at query #13
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.return_code == 1

    # Test command with timeout
    result = run_command("sleep 2", shell=True, timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command("echo $TEST_VAR", shell=True, env=env, return_output=True)
    assert b"test_value" in result.captured_output

    # Test command with custom working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", shell=True, cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test command with verbose mode
    result = run_command("echo 'Verbose test'", shell=True, verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"Verbose test" in result.captured_output

    # Test command with list arguments
    result = run_command(["echo", "List args test"], return_output=True)
    assert result.return_code == 0
    assert b"List args test" in result.captured_output

    # Test command with Unicode output
    result = run_command("echo 'Unicode: ñ'", shell=True, return_output=True)
    assert b"Unicode: \xc3\xb1" in result.captured_output

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(f"echo '{long_output}'", shell=True, ignore_errors=True)
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #14
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'test'", return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", timeout=0.1)

    # Test command with ignore_errors=True
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

    # Test verbose mode
    with patch('builtins.print') as mock_print:
        run_command("echo 'verbose test'", verbose=True)
        mock_print.assert_called()

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "test")
        with patch('tempfile.TemporaryFile') as mock_file:
            mock_file.return_value.__enter__.return_value.read.return_value = long_output.encode('utf-8')
            with pytest.raises(subprocess.CalledProcessError) as exc_info:
                run_command("test")
            assert b"*** (previous output truncated) ***" in exc_info.value.output


# LLM-generated content at query #15
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #16
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    original_error = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "Error output" in str(wrapped_error)

    # Test with subprocess.TimeoutExpired
    original_error = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_error)

    # Test with non-subprocess exception
    original_error = ValueError("test error")
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with subprocess.CalledProcessError and non-UTF-8 output
    original_error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #17
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #18
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", shell=True, return_output=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "Hello, World!"

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", shell=True)

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", shell=True, timeout=0.1)

    # Test command with ignore_errors=True
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with timeout and ignore_errors=True
    result = run_command("sleep 2", shell=True, timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with verbose=True
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", shell=True, verbose=True)

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "test_value"

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", shell=True, cwd=tmpdir, return_output=True)
        assert result.captured_output.decode('utf-8').strip() == tmpdir

    # Test command with list of arguments
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.command == ["echo", "Hello, World!"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "Hello, World!"

    # Test command with long output (truncation)
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command(f"echo '{long_output}'", shell=True)
    assert b"*** (previous output truncated) ***" in exc_info.value.output


# LLM-generated content at query #19
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", shell=True)

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", shell=True, timeout=0.1)

    # Test command with ignore_errors
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.return_code == 1

    # Test command with verbose output
    with patch('builtins.print') as mock_print:
        run_command("echo 'Verbose test'", shell=True, verbose=True)
        mock_print.assert_called()

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", shell=True, cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(f"echo '{long_output}'", shell=True, return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")

    # Test command with return_output=False
    result = run_command("echo 'No output'", shell=True)
    assert result.captured_output is None


# LLM-generated content at query #20
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test command with timeout
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command with verbose mode
    result = run_command("echo 'verbose test'", verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose test" in result.captured_output

    # Test command with long output truncation
    long_output = "x" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(f"echo '{long_output}'", return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len("*** (previous output truncated) ***\n")

    # Test command that raises CalledProcessError when ignore_errors is False
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", ignore_errors=False)

    # Test command that raises TimeoutExpired when ignore_errors is False
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1, ignore_errors=False)


# LLM-generated content at query #21
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo hello", return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", timeout=0.1)

    # Test ignore_errors flag
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test return_output flag
    result = run_command("echo test", return_output=True)
    assert result.captured_output is not None

    # Test verbose flag (hard to test output, but at least ensure it doesn't crash)
    result = run_command("echo verbose", verbose=True)
    assert result.return_code == 0

    # Test environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert b"test_value" in result.captured_output

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test command as list
    result = run_command(["echo", "list"], return_output=True)
    assert b"list" in result.captured_output

    # Test long output truncation
    long_output = "x" * 10000
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(long_output)
        f.flush()
        result = run_command(f"cat {f.name}", return_output=True)
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #22
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test ignore_errors flag
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    # Test verbose flag
    with patch("builtins.print") as mock_print:
        run_command(["echo", "test"], verbose=True)
        mock_print.assert_called()

    # Test environment variables
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, "utf-8") in result.captured_output

    # Test return_output flag
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output is not None

    # Test command as string
    result = run_command("echo hello", shell=True, return_output=True)
    assert b"hello" in result.captured_output

    # Test command with long output truncation
    long_output = "a" * 10000
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(["echo", long_output], 0)
        with patch("tempfile.TemporaryFile") as mock_file:
            mock_file.return_value.__enter__.return_value.read.return_value = bytes(long_output, "utf-8")
            result = run_command(["echo", long_output], return_output=True)
            assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #23
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert str(wrapped_error) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with other exception types
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with CalledProcessError with non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #24
#--------------------------

```python
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
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped)

    # Test with other exceptions (should return unchanged)
    try:
        raise ValueError("test")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert wrapped is e
        assert isinstance(wrapped, ValueError)


# LLM-generated content at query #25
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #26
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    Error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #27
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #28
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "Hello, World!"

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command("exit 1")
    assert exc_info.value.returncode == 1
    assert exc_info.value.output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        run_command("sleep 10", timeout=0.1)
    assert exc_info.value.returncode == -32768
    assert exc_info.value.output is not None

    # Test ignore_errors flag
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test verbose flag
    with patch('builtins.print') as mock_print:
        run_command("echo 'Verbose test'", verbose=True)
        assert mock_print.called

    # Test environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "test_value"

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        run_command(f"touch test.txt", cwd=tmpdir)
        assert os.path.exists(test_file)

    # Test return_output flag
    result = run_command("echo 'Return output test'", return_output=True)
    assert result.captured_output is not None
    assert result.return_code == 0

    # Test command as list
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.command == ["echo", "Hello, World!"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "Hello, World!"

    # Test long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "test_command", output=long_output.encode())
        result = run_command("test_command", ignore_errors=True)
        assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #29
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

    # Test ignore_errors flag
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test return_output flag
    result = run_command("echo 'Test'", return_output=True)
    assert result.captured_output is not None

    # Test verbose flag (capture output)
    with patch('builtins.print') as mock_print:
        run_command("echo 'Verbose Test'", verbose=True)
        mock_print.assert_called()

    # Test environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True, shell=True)
    assert b"test_value" in result.captured_output

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command as list
    result = run_command(["echo", "Hello, List!"], return_output=True)
    assert b"Hello, List!" in result.captured_output

    # Test error wrapper
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command("exit 1")
    assert "Captured output" in str(exc_info.value)

    # Test output truncation
    long_output = "x" * (MAX_OUTPUT_LENGTH + 1000)
    with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, "test", output=long_output.encode())):
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            run_command("test")
        assert b"*** (previous output truncated) ***" in exc_info.value.output


# LLM-generated content at query #30
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0

    # Test command with ignore_errors=True and non-zero return code
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command(["sh", "-c", "echo $TEST_VAR"], env=env, return_output=True)
    assert b"test_value" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test command with shell=True
    result = run_command("echo hello", shell=True, return_output=True)
    assert b"hello" in result.captured_output

    # Test command with Unicode output
    result = run_command(["python", "-c", "print('hello 世界')"], return_output=True)
    assert b"hello" in result.captured_output

    # Test command with long output (truncation)
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["python", "-c", f"print('{long_output}')"], return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #31
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #32
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.01, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"

    # Test with output that can't be decoded
    class FakeError(subprocess.CalledProcessError):
        def __init__(self):
            self.returncode = 1
            self.cmd = "test"
            self.output = b'\xff\xfe'

    wrapped = error_wrapper(FakeError())
    assert "Failed to parse output" in str(wrapped)


# LLM-generated content at query #33
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with timeout
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test verbose mode
    result = run_command("echo 'verbose test'", verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose test" in result.captured_output

    # Test command as list
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.command == ["echo", "Hello, World!"]
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output


# LLM-generated content at query #34
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #35
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is None

    # Test with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.captured_output == b"hello\n"

    # Test with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0

    # Test with environment variables
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"})
    assert result.return_code == 0

    # Test with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir)
        assert result.return_code == 0

    # Test with timeout (should not raise)
    result = run_command(["sleep", "0.1"], timeout=1)
    assert result.return_code == 0

    # Test with ignore_errors=True and failing command
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test with ignore_errors=True and timeout
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test with non-zero return code and return_output=True
    result = run_command(["false"], return_output=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test with long output truncation
    long_output = "a" * (8192 + 100)
    result = run_command(["echo", long_output], return_output=True)
    assert len(result.captured_output) <= 8192 + len("*** (previous output truncated) ***\n")
    assert result.captured_output.endswith(b"\n")

    # Test with Unicode output
    result = run_command(["echo", "héllo"], return_output=True)
    assert result.captured_output == b"héllo\n"


# LLM-generated content at query #36
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

    # Test command with ignore_errors
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test command with verbose output
    with patch('builtins.print') as mock_print:
        run_command("echo 'Verbose'", verbose=True)
        mock_print.assert_called()

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(f"echo '{long_output}'", return_output=True)
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len("*** (previous output truncated) ***\n")
    assert b"*** (previous output truncated) ***" in result.captured_output

    # Test command with return_output=False
    result = run_command("echo 'No output'", return_output=False)
    assert result.captured_output is None

    # Test command with list of arguments
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output


# LLM-generated content at query #37
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello world"], return_output=True)
    assert result.command == ["echo", "hello world"]
    assert result.return_code == 0
    assert result.captured_output == b"hello world\n"

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent_directory"])

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with ignore_errors=True
    result = run_command(["ls", "/nonexistent_directory"], ignore_errors=True)
    assert result.return_code != 0
    assert b"cannot access" in result.captured_output

    # Test command with verbose=True
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent_directory"], verbose=True)

    # Test command with custom environment
    env = {"TEST_VAR": "test_value"}
    result = run_command(["bash", "-c", "echo $TEST_VAR"], env=env, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

    # Test command with custom working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.decode().strip() == tmpdir

    # Test command with very long output
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    with tempfile.NamedTemporaryFile() as f:
        f.write(long_output.encode())
        f.flush()
        result = run_command(["cat", f.name], return_output=True)
        assert result.return_code == 0
        assert b"*** (previous output truncated) ***" in result.captured_output
        assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")

    # Test command with Unicode output
    result = run_command(["bash", "-c", "echo 'Hello 世界'"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"Hello 世界\n"

    # Test command with shell=True
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"


# LLM-generated content at query #38
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_err) or "No output was generated" in str(wrapped_err)

    # Test with subprocess.TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_err) or "No output was generated" in str(wrapped_err)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, ValueError)
        assert str(wrapped_err) == "test error"


# LLM-generated content at query #39
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #40
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test command with timeout
    result = run_command("sleep 2", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test verbose mode
    result = run_command("echo 'verbose test'", verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose test" in result.captured_output

    # Test command that produces no output
    result = run_command("true", return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b""

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(f"echo '{long_output}'", return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #41
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #42
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", shell=True)
    assert result.return_code == 0
    assert result.captured_output is None

    # Test successful command execution with return_output=True
    result = run_command("echo 'Hello, World!'", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command execution with verbose=True
    result = run_command("echo 'Hello, World!'", shell=True, verbose=True)
    assert result.return_code == 0

    # Test command execution with non-zero return code
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command execution with timeout
    result = run_command("sleep 2", shell=True, timeout=1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command execution with environment variables
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test command execution with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", shell=True, cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command execution with list of arguments
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command execution with ignore_errors=True and non-zero return code
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command execution with ignore_errors=True and timeout
    result = run_command("sleep 2", shell=True, timeout=1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


# LLM-generated content at query #43
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output.strip() == b"test"

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent"])

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.01)

    # Test command with ignore_errors
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with verbose
    with patch('builtins.print') as mock_print:
        run_command(["echo", "test"], verbose=True)
        mock_print.assert_called()

    # Test command with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test command with return_output=False
    result = run_command(["echo", "test"], return_output=False)
    assert result.captured_output is None

    # Test command with shell=True
    result = run_command("echo test", shell=True, return_output=True)
    assert result.captured_output.strip() == b"test"


# LLM-generated content at query #44
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with CalledProcessError with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #45
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #46
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #47
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #48
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #49
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with output capture
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent"])

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with ignore_errors
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test verbose mode
    with patch('builtins.print') as mock_print:
        result = run_command(["echo", "hello"], verbose=True)
        assert mock_print.called

    # Test environment variables
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output


# LLM-generated content at query #50
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #51
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #52
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo hello")
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command("echo hello", return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with verbose=True
    result = run_command("echo hello", verbose=True)
    assert result.return_code == 0

    # Test command with ignore_errors=True
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.01)

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert b"test_value" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test command with list args
    result = run_command(["echo", "hello"], return_output=True)
    assert b"hello" in result.captured_output

    # Test command that fails
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with Unicode output
    result = run_command("echo 'Hello, 世界'", return_output=True)
    assert b"Hello, " in result.captured_output

    # Test command with long output (truncation)
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(f"echo {long_output}", return_output=True, shell=True)
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #53
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test command with timeout
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True, shell=True)
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test verbose mode
    result = run_command("echo 'verbose test'", verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose test" in result.captured_output

    # Test command as list
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test exception wrapping
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command("exit 1")
    assert "Captured output" in str(exc_info.value)

    # Test output truncation
    long_output = "x" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(f"echo '{long_output}'", ignore_errors=True, shell=True)
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #54
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command")
    err.output = b"Error output"
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", 10)
    err.output = b"Timeout output"
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    err.output = None
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command")
    err.output = b"\xff\xfe"
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #55
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_err) or "No output was generated" in str(wrapped_err)

    # Test with subprocess.TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_err) or "No output was generated" in str(wrapped_err)

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, ValueError)
        assert str(wrapped_err) == "Test error"


# LLM-generated content at query #56
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "error output" in str(wrapped_error)

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_error)
    assert "timeout output" in str(wrapped_error)

    # Test with non-subprocess exception
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with CalledProcessError without output
    error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_error)

    # Test with TimeoutExpired without output
    error = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_error)

    # Test with non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #57
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #58
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped) or "No output was generated." in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped) or "No output was generated." in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #59
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    Error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #60
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #61
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test successful command execution with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

    # Test command execution with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent"])

    # Test command execution with error and ignore_errors=True
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert b"cannot access" in result.captured_output

    # Test command execution with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command execution with timeout and ignore_errors=True
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test command execution with verbose=True
    with patch('builtins.print') as mock_print:
        result = run_command(["echo", "hello"], verbose=True)
        assert mock_print.called
        assert result.return_code == 0

    # Test command execution with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test command execution with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output


# LLM-generated content at query #62
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #63
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #64
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with subprocess.TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with subprocess.CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with subprocess.TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with subprocess.CalledProcessError with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #65
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

    # Test command with ignore_errors
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test command with verbose
    result = run_command("echo 'Verbose'", verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"Verbose" in result.captured_output

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test command with long output truncation
    long_output = "echo " + "a" * 10000
    result = run_command(long_output, return_output=True)
    assert result.return_code == 0
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #66
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #67
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_cmd' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_cmd", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_cmd' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_cmd' returned non-zero exit status 1.\nNo output was generated."

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_cmd' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #68
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #69
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #70
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with timeout
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test verbose mode
    result = run_command("echo 'verbose test'", verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose test" in result.captured_output

    # Test command that produces no output
    result = run_command("true", return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b""

    # Test command with long output (truncation)
    long_output = "x" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(f"echo '{long_output}'", ignore_errors=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #71
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command")
    error.output = b"Test output"
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "Test output" in str(wrapped_error)

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", 10)
    error.output = b"Test timeout output"
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_error)
    assert "Test timeout output" in str(wrapped_error)

    # Test with non-subprocess error
    error = ValueError("Test error")
    wrapped_error = error_wrapper(error)
    assert wrapped_error is error
    assert isinstance(wrapped_error, ValueError)
    assert "Test error" in str(wrapped_error)

    # Test with no output
    error = subprocess.CalledProcessError(1, "test_command")
    error.output = None
    wrapped_error = error_wrapper(error)
    assert "No output was generated." in str(wrapped_error)

    # Test with UnicodeDecodeError
    error = subprocess.CalledProcessError(1, "test_command")
    error.output = b"\xff\xfe"
    wrapped_error = error_wrapper(error)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #72
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #73
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #74
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #75
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #76
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with subprocess.TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with a non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"


# LLM-generated content at query #77
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    original_error = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    original_error = subprocess.TimeoutExpired("test_command", timeout=1)
    original_error.output = b"timeout output"
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert str(wrapped_error) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    original_error = ValueError("test error")
    wrapped_error = error_wrapper(original_error)
    assert wrapped_error is original_error
    assert str(wrapped_error) == "test error"

    # Test with no output
    original_error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(original_error)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    original_error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(original_error)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #78
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #79
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.captured_output == b"hello\n"

    # Test command with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0

    # Test command with ignore_errors=True and non-zero return code
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command(["echo", "$TEST_VAR"], env=env, shell=True)
    assert result.return_code == 0

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir)
        assert result.return_code == 0

    # Test command with ignore_errors=True and timeout
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with verbose=True and non-zero return code
    result = run_command(["ls", "/nonexistent"], verbose=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


# LLM-generated content at query #80
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #81
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #82
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #83
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"


# LLM-generated content at query #84
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

    # Test command with ignore_errors
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test command with verbose mode
    with patch('builtins.print') as mock_print:
        run_command("echo 'Verbose test'", verbose=True)
        mock_print.assert_called()

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(f"echo '{long_output}'", return_output=True)
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len("*** (previous output truncated) ***\n")
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #85
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)


# LLM-generated content at query #86
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "Hello, World!"

    # Test command with non-zero return code
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with timeout
    result = run_command("sleep 10", shell=True, timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command("echo $TEST_VAR", shell=True, env=env, return_output=True)
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "test_value"

    # Test command with verbose mode
    result = run_command("echo 'Verbose mode'", shell=True, verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "Verbose mode"

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", shell=True, cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir in result.captured_output.decode('utf-8').strip()

    # Test command with list of arguments
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "Hello, World!"

    # Test command with error wrapper
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command("exit 1", shell=True)
    assert exc_info.value.returncode == 1
    assert exc_info.value.output is not None

    # Test command with timeout error wrapper
    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        run_command("sleep 10", shell=True, timeout=0.1)
    assert exc_info.value.returncode is None
    assert exc_info.value.output is not None


# LLM-generated content at query #87
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(['false'], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped) or "No output was generated." in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(['sleep', '10'], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped) or "No output was generated." in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #88
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #89
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #90
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with other exception types (should remain unchanged)
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #91
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "error output" in str(wrapped_error)

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_error)
    assert "timeout output" in str(wrapped_error)

    # Test with non-subprocess error
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with no output
    error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(error)
    assert "No output was generated." in str(wrapped_error)

    # Test with non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #92
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with CalledProcessError with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #93
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with other exception types (should remain unchanged)
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #94
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with subprocess.TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with a non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"


# LLM-generated content at query #95
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    Error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #96
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "Error output" in str(wrapped_error)

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_error)
    assert "Timeout output" in str(wrapped_error)

    # Test with non-subprocess exception
    error = ValueError("Test error")
    wrapped_error = error_wrapper(error)
    assert wrapped_error is error
    assert str(wrapped_error) == "Test error"

    # Test with no output
    error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(error)
    assert "No output was generated." in str(wrapped_error)

    # Test with non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped_error = error_wrapper(error)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #97
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with subprocess.TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"


# LLM-generated content at query #98
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output:" in str(wrapped_error)

    # Test with subprocess.TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output:" in str(wrapped_error)

    # Test with other exception types (should return unchanged)
    other_error = ValueError("test error")
    wrapped_other_error = error_wrapper(other_error)
    assert wrapped_other_error is other_error


# LLM-generated content at query #99
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", timeout=0.1)

    # Test ignore_errors flag
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test return_output flag
    result = run_command("echo 'Test'", return_output=True)
    assert result.captured_output is not None

    # Test verbose flag (no assertion, just ensure it runs)
    run_command("echo 'Verbose'", verbose=True)

    # Test environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert b"test_value" in result.captured_output

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test command as list
    result = run_command(["echo", "Hello"], return_output=True)
    assert b"Hello" in result.captured_output


# LLM-generated content at query #100
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #101
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with subprocess.TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"


# LLM-generated content at query #102
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #103
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with TimeoutExpired and no output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #104
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    Error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #105
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #106
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #107
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'test'", return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

    # Test command with return_output=False
    result = run_command("echo 'test'", return_output=False)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with verbose=True
    result = run_command("echo 'test'", verbose=True, return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

    # Test command with non-zero return code
    result = run_command("exit 1", ignore_errors=True, return_output=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with timeout
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True, return_output=True)
    assert result.command == "sleep 10"
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.command == "pwd"
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

    # Test command with list of arguments
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

    # Test command with error_wrapper
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command("exit 1", return_output=True)
    assert "Captured output" in str(exc_info.value)

    # Test command with long output
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(f"echo '{long_output}'", ignore_errors=True, return_output=True)
    assert result.command == f"echo '{long_output}'"
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #108
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    Error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", 10)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 10 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #109
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)


# LLM-generated content at query #110
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #111
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #112
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert str(wrapped_error) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert wrapped_error == error
    assert str(wrapped_error) == "test error"

    # Test with no output
    error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #113
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #114
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "error output" in str(wrapped_error)

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_error)
    assert "timeout output" in str(wrapped_error)

    # Test with non-subprocess error
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with no output
    error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_error)

    # Test with non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #115
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=10)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output" in str(wrapped_err)


# LLM-generated content at query #116
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'test'", return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

    # Test command with return_output=False
    result = run_command("echo 'test'", return_output=False)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with verbose=True
    result = run_command("echo 'test'", verbose=True, return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

    # Test command with ignore_errors=True and non-zero return code
    result = run_command("exit 1", ignore_errors=True, return_output=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with ignore_errors=True and timeout
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True, return_output=True)
    assert result.command == "sleep 10"
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.command == "pwd"
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

    # Test command with list of arguments
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

    # Test command with timeout and return_output=True
    result = run_command("sleep 1", timeout=2, return_output=True)
    assert result.command == "sleep 1"
    assert result.return_code == 0
    assert result.captured_output == b""

    # Test command with timeout and return_output=False
    result = run_command("sleep 1", timeout=2, return_output=False)
    assert result.command == "sleep 1"
    assert result.return_code == 0
    assert result.captured_output is None


# LLM-generated content at query #117
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "Error output" in str(wrapped_error)

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_error)
    assert "Timeout output" in str(wrapped_error)

    # Test with non-subprocess exception
    error = ValueError("Test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "Test error"

    # Test with no output
    error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_error)

    # Test with non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #118
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_err) or "No output was generated" in str(wrapped_err)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_err) or "No output was generated" in str(wrapped_err)

    # Test with non-subprocess exception (should return unchanged)
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped_err = error_wrapper(e)
        assert isinstance(wrapped_err, ValueError)
        assert str(wrapped_err) == "test error"


# LLM-generated content at query #119
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert str(wrapped_error) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with no output
    error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #120
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #121
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #122
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


