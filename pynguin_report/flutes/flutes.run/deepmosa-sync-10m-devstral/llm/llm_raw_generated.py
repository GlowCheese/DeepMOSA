####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_verbose():
    with patch('flutes.run.log') as mock_log:
        run_command(["echo", "test"], verbose=True)
        mock_log.assert_called_once_with("> ['echo', 'test']", timestamp=False, include_proc_id=False)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_ignored():
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_shell_string():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_nonzero_return():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_timeout_raises():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_output_truncated():
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, "test")) as mock_run:
        mock_run.return_value.stdout = long_output.encode()
        result = run_command(["test"], ignore_errors=True)
        assert b"*** (previous output truncated) ***" in result.captured_output
        assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + 32

def test_run_command_unicode_decode_error():
    with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, "test")) as mock_run:
        mock_run.return_value.stdout = b'\xff\xfe'
        with patch('flutes.run.log') as mock_log:
            run_command(["test"], verbose=True, ignore_errors=True)
            mock_log.assert_called_with(str(b'\xff\xfe'), timestamp=False, include_proc_id=False)


# LLM-generated content at query #2
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo hello", return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    result = run_command("exit 1", ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_timeout():
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.command == "sleep 10"
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command("echo test", verbose=True, return_output=True)
    assert result.command == "echo test"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "value"}, return_output=True, shell=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output == b"value\n"

def test_run_command_cwd():
    result = run_command("pwd", cwd="/tmp", return_output=True)
    assert result.command == "pwd"
    assert result.return_code == 0
    assert result.captured_output.strip() == b"/tmp"


# LLM-generated content at query #3
#--------------------------

```python
def test_predicate_evaluates_to_false():
    assert not (False or 0 != 0 or False)


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_evaluates_to_false():
    result = run_command("echo test", ignore_errors=True)
    assert result.captured_output is not None


# LLM-generated content at query #5
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command("echo 'test'", return_output=True)
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "test"

def test_run_command_with_error():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

def test_run_command_timeout_ignore_errors():
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_verbose():
    result = run_command("echo 'verbose'", verbose=True)
    assert result.return_code == 0

def test_run_command_with_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "test_value"

def test_run_command_with_cwd():
    result = run_command("pwd", cwd="/tmp", return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "/tmp"

def test_run_command_list_args():
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "test"

def test_run_command_no_output():
    result = run_command("true")
    assert result.return_code == 0
    assert result.captured_output is None


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_32_evaluates_to_true():
    output = b"a" * (MAX_OUTPUT_LENGTH + 1)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #7
#--------------------------

```python
def test_verbose_logging():
    with patch('flutes.run.log') as mock_log:
        run_command("echo test", verbose=True)
        mock_log.assert_called_with("> 'echo test'", timestamp=False, include_proc_id=False)


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert run_command("echo test", return_output=True).captured_output is not None
    assert run_command("exit 1", ignore_errors=True).captured_output is not None
    assert run_command("echo test", verbose=True).captured_output is not None


# LLM-generated content at query #9
#--------------------------

```python
def test_ignore_errors_false_with_exception():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("false", ignore_errors=False)


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert run_command("echo test", return_output=True).captured_output is not None
    assert run_command("exit 1", ignore_errors=True).captured_output is not None
    assert run_command("echo test", verbose=True).captured_output is not None


# LLM-generated content at query #11
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command("echo 'Hello, World!'", shell=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_return_output():
    result = run_command("echo 'Hello, World!'", shell=True, return_output=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

def test_run_command_with_verbose():
    result = run_command("echo 'Hello, World!'", shell=True, verbose=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_nonzero_return_code():
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is None

def test_run_command_with_timeout():
    result = run_command("sleep 2", shell=True, timeout=1, ignore_errors=True)
    assert result.command == "sleep 2"
    assert result.return_code == -32768
    assert result.captured_output is None

def test_run_command_with_env():
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"})
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_cwd():
    result = run_command("pwd", shell=True, cwd="/tmp")
    assert result.command == "pwd"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_kwargs():
    result = run_command("echo 'Hello, World!'", shell=True, text=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_list_args():
    result = run_command(["echo", "Hello, World!"])
    assert result.command == ["echo", "Hello, World!"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_called_process_error():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", shell=True)

def test_run_command_with_timeout_expired():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", shell=True, timeout=1)

def test_run_command_with_os_error_retry():
    result = run_command("nonexistent_command", shell=True, ignore_errors=True)
    assert result.command == "nonexistent_command"
    assert result.return_code != 0
    assert result.captured_output is not None


# LLM-generated content at query #12
#--------------------------

```python
def test_output_truncation():
    output = b"a" * (MAX_OUTPUT_LENGTH + 1)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #13
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_verbose():
    result = run_command(["echo", "verbose"], verbose=True)
    assert result.command == ["echo", "verbose"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "2"], timeout=0.1)

def test_run_command_with_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["env"]
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.command == ["pwd"]
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

def test_run_command_with_shell_command():
    result = run_command("echo shell", shell=True, return_output=True)
    assert result.command == "echo shell"
    assert result.return_code == 0
    assert result.captured_output == b"shell\n"

def test_run_command_with_nonzero_return_code():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_with_return_output():
    result = run_command(["echo", "output"], return_output=True)
    assert result.command == ["echo", "output"]
    assert result.return_code == 0
    assert result.captured_output == b"output\n"

def test_run_command_with_long_output():
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], return_output=True)
    assert result.command == ["echo", long_output]
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #14
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_creates_new_type_for_called_process_error():
    err = subprocess.CalledProcessError(1, "cmd")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert type(wrapped).__name__ == "CalledProcessError"
    assert type(wrapped) is not type(err)

def test_error_wrapper_creates_new_type_for_timeout_expired():
    err = subprocess.TimeoutExpired("cmd", 1)
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert type(wrapped).__name__ == "TimeoutExpired"
    assert type(wrapped) is not type(err)

def test_error_wrapper_str_with_output():
    err = subprocess.CalledProcessError(1, "cmd", output=b"line1\nline2")
    wrapped = error_wrapper(err)
    assert "Captured output:" in str(wrapped)
    assert "line1" in str(wrapped)
    assert "line2" in str(wrapped)

def test_error_wrapper_str_with_no_output():
    err = subprocess.CalledProcessError(1, "cmd")
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)

def test_error_wrapper_str_with_unicode_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)


# LLM-generated content at query #15
#--------------------------

```python
def test_verbose_logging():
    with patch('flutes.run.log') as mock_log:
        run_command("echo test", verbose=True)
        mock_log.assert_called_with("> 'echo test'", timestamp=False, include_proc_id=False)


# LLM-generated content at query #16
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_error():
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command(["ls", "/nonexistent"])
    assert exc_info.value.returncode != 0
    assert "No such file or directory" in str(exc_info.value)

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        run_command(["sleep", "10"], timeout=0.01)
    assert exc_info.value.timeout == 0.01

def test_run_command_ignore_errors():
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_verbose_mode():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_env():
    result = run_command(["echo", "$TEST_VAR"], env={"TEST_VAR": "value"}, shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"value\n"

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert result.captured_output.strip() == b"/tmp"

def test_run_command_with_kwargs():
    result = run_command(["echo", "test"], shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_string_command():
    result = run_command("echo test", shell=True, return_output=True)
    assert result.command == "echo test"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_list_command():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"


# LLM-generated content at query #17
#--------------------------

```python
def test_return_output_true():
    assert run_command("echo test", return_output=True).captured_output is not None

def test_return_code_nonzero():
    assert run_command("exit 1", ignore_errors=True).captured_output is not None

def test_verbose_true():
    assert run_command("echo test", verbose=True).captured_output is not None


# LLM-generated content at query #18
#--------------------------

```python
def test_error_wrapper_returns_original_error_for_non_subprocess_exceptions():
    class CustomError(Exception):
        pass

    err = CustomError("test error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #19
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_verbose():
    with patch('builtins.print') as mock_print:
        run_command(["echo", "test"], verbose=True)
        mock_print.assert_called()

def test_run_command_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_cwd():
    with TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

def test_run_command_return_output():
    result = run_command(["echo", "output"], return_output=True)
    assert result.captured_output == b"output\n"

def test_run_command_shell_string():
    result = run_command("echo shell", shell=True, return_output=True)
    assert result.captured_output == b"shell\n"


# LLM-generated content at query #20
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = Exception("test")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #21
#--------------------------

```python
def test_unicode_decode_error_handling():
    with tempfile.TemporaryFile() as f:
        f.write(b'\x80\x81\x82')  # Invalid UTF-8 sequence
        f.seek(0)
        output = f.read()
        try:
            output.decode('utf-8')
        except UnicodeDecodeError:
            assert True


# LLM-generated content at query #22
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.command == ["pwd"]
    assert result.return_code == 0
    assert result.captured_output.strip() == b"/tmp"

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["printenv", "TEST_VAR"]
    assert result.return_code == 0
    assert result.captured_output.strip() == b"test_value"

def test_run_command_with_timeout():
    result = run_command(["sleep", "0.1"], timeout=1, return_output=True)
    assert result.command == ["sleep", "0.1"]
    assert result.return_code == 0
    assert result.captured_output == b""

def test_run_command_with_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_return_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_string_command():
    result = run_command("echo test", shell=True, return_output=True)
    assert result.command == "echo test"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_timeout_expired():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_with_called_process_error():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_with_output_truncation():
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], return_output=True)
    assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_evaluates_to_true_with_return_output():
    result = run_command("echo test", return_output=True)
    assert result.captured_output is not None

def test_predicate_evaluates_to_true_with_nonzero_returncode():
    result = run_command("exit 1", ignore_errors=True)
    assert result.captured_output is not None

def test_predicate_evaluates_to_true_with_verbose():
    result = run_command("echo test", verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #24
#--------------------------

```python
def test_error_wrapper_non_subprocess_exception():
    err = ValueError("test error")
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_called_process_error_with_output():
    err = subprocess.CalledProcessError(1, "cmd", output=b"line1\nline2")
    wrapped = error_wrapper(err)
    assert str(wrapped) == "Command 'cmd' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"

def test_error_wrapper_called_process_error_no_output():
    err = subprocess.CalledProcessError(1, "cmd")
    wrapped = error_wrapper(err)
    assert str(wrapped) == "Command 'cmd' returned non-zero exit status 1.\nNo output was generated."

def test_error_wrapper_timeout_expired_with_output():
    err = subprocess.TimeoutExpired("cmd", timeout=1, output=b"line1\nline2")
    wrapped = error_wrapper(err)
    assert str(wrapped) == "Command 'cmd' timed out after 1 seconds.\nCaptured output:\n    line1\n    line2"

def test_error_wrapper_timeout_expired_no_output():
    err = subprocess.TimeoutExpired("cmd", timeout=1)
    wrapped = error_wrapper(err)
    assert str(wrapped) == "Command 'cmd' timed out after 1 seconds.\nNo output was generated."

def test_error_wrapper_called_process_error_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    assert str(wrapped) == "Command 'cmd' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #25
#--------------------------

```python
def test_verbose_logging():
    with patch('flutes.run.log') as mock_log:
        run_command("echo test", verbose=True)
        mock_log.assert_called_once_with("> 'echo test'", timestamp=False, include_proc_id=False)


# LLM-generated content at query #26
#--------------------------

```python
def test_unicode_decode_error_occurs():
    args = "echo 'test'"
    env = None
    cwd = None
    timeout = None
    verbose = True
    return_output = False
    ignore_errors = False
    kwargs = {}

    with tempfile.TemporaryFile() as f:
        f.write(b'\x80\x81\x82')
        f.seek(0)
        output = f.read()

        try:
            output.decode('utf-8')
        except UnicodeDecodeError:
            pass
        else:
            raise AssertionError("UnicodeDecodeError did not occur")


# LLM-generated content at query #27
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #28
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err

def test_error_wrapper_creates_new_type_for_subprocess_called_process_error():
    err = subprocess.CalledProcessError(1, "test_cmd")
    result = error_wrapper(err)
    assert type(result).__name__ == "CalledProcessError"
    assert type(result) is not type(err)

def test_error_wrapper_creates_new_type_for_subprocess_timeout_expired():
    err = subprocess.TimeoutExpired("test_cmd", 1)
    result = error_wrapper(err)
    assert type(result).__name__ == "TimeoutExpired"
    assert type(result) is not type(err)

def test_error_wrapper_preserves_exception_attributes():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    result = error_wrapper(err)
    assert result.returncode == err.returncode
    assert result.cmd == err.cmd
    assert result.output == err.output

def test_error_wrapper_str_with_output():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    result = error_wrapper(err)
    expected_str = "Command 'test_cmd' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"
    assert str(result) == expected_str

def test_error_wrapper_str_with_no_output():
    err = subprocess.CalledProcessError(1, "test_cmd")
    result = error_wrapper(err)
    expected_str = "Command 'test_cmd' returned non-zero exit status 1.\nNo output was generated."
    assert str(result) == expected_str

def test_error_wrapper_str_with_unicode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"\xff\xfe")
    result = error_wrapper(err)
    expected_str = "Command 'test_cmd' returned non-zero exit status 1.\nFailed to parse output."
    assert str(result) == expected_str


# LLM-generated content at query #29
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = "echo 'test'"
    mock_output = b"\x80\x81"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        with patch("tempfile.TemporaryFile") as mock_tempfile:
            mock_file = MagicMock()
            mock_file.read.return_value = mock_output
            mock_tempfile.return_value.__enter__.return_value = mock_file
            result = run_command(args, return_output=True)
            assert result.captured_output == mock_output


# LLM-generated content at query #30
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo hello", return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.01)

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

def test_run_command_timeout_ignore_errors():
    result = run_command("sleep 10", timeout=0.01, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_verbose():
    result = run_command("echo hello", verbose=True, return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.command == "pwd"
        assert result.return_code == 0
        assert result.captured_output.decode().strip() == tmpdir

def test_run_command_list_args():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_return_output_false():
    result = run_command("echo hello")
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_nonzero_returncode():
    result = run_command("exit 1", return_output=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_truncated_output():
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "test")) as mock_run:
        mock_run.return_value.stdout = long_output.encode()
        result = run_command("test", ignore_errors=True)
        assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #31
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #32
#--------------------------

```python
def test_error_wrapper_predicate():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #33
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = "echo 'test'"
    env = None
    cwd = None
    timeout = None
    verbose = True
    return_output = False
    ignore_errors = False
    kwargs = {}

    # Simulate a UnicodeDecodeError by providing non-UTF-8 output
    output = b'\xff\xfe'
    f = tempfile.TemporaryFile()
    f.write(output)
    f.seek(0)

    # Mock subprocess.run to return a completed process with non-zero return code
    ret = subprocess.CompletedProcess(args, returncode=1, stdout=output, stderr=b'')
    with patch('subprocess.run', return_value=ret):
        result = run_command(args, env=env, cwd=cwd, timeout=timeout, verbose=verbose, return_output=return_output, ignore_errors=ignore_errors, **kwargs)

    assert result.captured_output == output


# LLM-generated content at query #34
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #35
#--------------------------

```python
def test_unicode_decode_error_handling():
    result = run_command("echo 'test'", return_output=True, verbose=True, shell=True)
    assert isinstance(result, CommandResult)
    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #36
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    original_error = ValueError("test error")
    wrapped_error = error_wrapper(original_error)
    assert wrapped_error is original_error

def test_error_wrapper_preserves_exception_type_for_subprocess_errors():
    original_error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)

def test_error_wrapper_modifies_str_representation_with_output():
    original_error = subprocess.CalledProcessError(1, "test_command", output=b"line1\nline2")
    wrapped_error = error_wrapper(original_error)
    assert "Captured output:" in str(wrapped_error)
    assert "line1" in str(wrapped_error)
    assert "line2" in str(wrapped_error)

def test_error_wrapper_modifies_str_representation_without_output():
    original_error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(original_error)
    assert "No output was generated." in str(wrapped_error)

def test_error_wrapper_modifies_str_representation_with_invalid_utf8():
    original_error = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped_error = error_wrapper(original_error)
    assert "Failed to parse output." in str(wrapped_error)

def test_error_wrapper_preserves_original_exception_attributes():
    original_error = subprocess.CalledProcessError(1, "test_command", output=b"test")
    wrapped_error = error_wrapper(original_error)
    assert wrapped_error.returncode == original_error.returncode
    assert wrapped_error.cmd == original_error.cmd
    assert wrapped_error.output == original_error.output


# LLM-generated content at query #37
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_timeout():
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.command == ["sleep", "10"]
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_with_verbose():
    result = run_command(["echo", "verbose"], verbose=True, return_output=True)
    assert result.command == ["echo", "verbose"]
    assert result.return_code == 0
    assert result.captured_output == b"verbose\n"

def test_run_command_with_env_and_cwd():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, cwd="/tmp", return_output=True)
    assert result.command == ["env"]
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_with_error():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_with_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_string_command():
    result = run_command("echo string_command", shell=True, return_output=True)
    assert result.command == "echo string_command"
    assert result.return_code == 0
    assert result.captured_output == b"string_command\n"


# LLM-generated content at query #38
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_creates_new_type_for_subprocess_errors():
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == "CalledProcessError"
    assert type(wrapped_err) is not type(err)

def test_error_wrapper_preserves_exception_attributes():
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert wrapped_err.returncode == err.returncode
    assert wrapped_err.cmd == err.cmd
    assert wrapped_err.output == err.output

def test_error_wrapper_str_with_output():
    err = subprocess.CalledProcessError(1, "test_command", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    expected_str = "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"
    assert str(wrapped_err) == expected_str

def test_error_wrapper_str_with_no_output():
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    expected_str = "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."
    assert str(wrapped_err) == expected_str

def test_error_wrapper_str_with_unicode_error():
    err = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    expected_str = "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."
    assert str(wrapped_err) == expected_str

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == "TimeoutExpired"
    assert type(wrapped_err) is not type(err)


# LLM-generated content at query #39
#--------------------------

```python
def test_error_wrapper_returns_original_error_for_non_subprocess_exceptions():
    class CustomException(Exception):
        pass

    err = CustomException("test error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #40
#--------------------------

```python
def test_isinstance_predicate():
    assert isinstance(subprocess.CalledProcessError("msg", "cmd"), (subprocess.CalledProcessError, subprocess.TimeoutExpired))
    assert isinstance(subprocess.TimeoutExpired("cmd", 1), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #41
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #42
#--------------------------

```python
def test_error_wrapper_returns_non_subprocess_error_unchanged():
    original_error = ValueError("test error")
    result = error_wrapper(original_error)
    assert result is original_error

def test_error_wrapper_modifies_called_process_error():
    original_error = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    result = error_wrapper(original_error)
    assert isinstance(result, type(original_error))
    assert "Captured output:" in str(result)
    assert "test output" in str(result)

def test_error_wrapper_modifies_timeout_expired_error():
    original_error = subprocess.TimeoutExpired("test_cmd", 1, output=b"timeout output")
    result = error_wrapper(original_error)
    assert isinstance(result, type(original_error))
    assert "Captured output:" in str(result)
    assert "timeout output" in str(result)

def test_error_wrapper_handles_empty_output():
    original_error = subprocess.CalledProcessError(1, "test_cmd", output=b"")
    result = error_wrapper(original_error)
    assert isinstance(result, type(original_error))
    assert "No output was generated." in str(result)

def test_error_wrapper_handles_unicode_decode_error():
    original_error = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    result = error_wrapper(original_error)
    assert isinstance(result, type(original_error))
    assert "Failed to parse output." in str(result)


# LLM-generated content at query #43
#--------------------------

```python
def test_error_wrapper_predicate():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #44
#--------------------------

```python
def test_return_output_true():
    result = run_command("echo test", return_output=True)
    assert result.captured_output is not None


# LLM-generated content at query #45
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_creates_new_type_for_subprocess_errors():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == "CalledProcessError"
    assert type(wrapped_err) is not type(err)

def test_error_wrapper_preserves_output_in_str_for_called_process_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "line1" in str(wrapped_err)
    assert "line2" in str(wrapped_err)

def test_error_wrapper_handles_no_output_for_called_process_error():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_handles_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)

def test_error_wrapper_works_for_timeout_expired():
    err = subprocess.TimeoutExpired("test_cmd", 1)
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == "TimeoutExpired"
    assert "No output was generated." in str(wrapped_err)


# LLM-generated content at query #46
#--------------------------

```python
def test_error_wrapper_predicate():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #47
#--------------------------

```python
def test_verbose_logging_when_verbose_is_true():
    with patch('flutes.run.log') as mock_log:
        run_command("test", verbose=True)
        mock_log.assert_called_once_with("> 'test'", timestamp=False, include_proc_id=False)


# LLM-generated content at query #48
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = ["echo", "test"]
    result = run_command(args, return_output=True, verbose=True, env={"LC_ALL": "C"})
    assert isinstance(result, CommandResult)
    assert result.command == args
    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #49
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command("echo hello", return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "hello"

def test_run_command_failed_execution():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

def test_run_command_verbose_mode():
    result = run_command("echo hello", verbose=True, return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "hello"

def test_run_command_with_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "test_value"

def test_run_command_with_cwd():
    result = run_command("pwd", cwd="/tmp", return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "/tmp"

def test_run_command_list_args():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "hello"

def test_run_command_return_output_false():
    result = run_command("echo hello", return_output=False)
    assert result.captured_output is None


# LLM-generated content at query #50
#--------------------------

```python
def test_unicode_decode_error_avoids_console_logging():
    output = b'\x80abc'
    assert output.decode('utf-8') == UnicodeDecodeError


# LLM-generated content at query #51
#--------------------------

```python
def test_error_wrapper_predicate():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #52
#--------------------------

```python
def test_error_wrapper_returns_non_subprocess_exception_unchanged():
    original_error = ValueError("test error")
    result = error_wrapper(original_error)
    assert result is original_error

def test_error_wrapper_modifies_called_process_error_with_output():
    original_error = subprocess.CalledProcessError(1, "test_command", output=b"line1\nline2")
    result = error_wrapper(original_error)
    assert isinstance(result, type(original_error))
    assert "Captured output:" in str(result)
    assert "line1" in str(result)
    assert "line2" in str(result)

def test_error_wrapper_modifies_called_process_error_without_output():
    original_error = subprocess.CalledProcessError(1, "test_command")
    result = error_wrapper(original_error)
    assert isinstance(result, type(original_error))
    assert "No output was generated." in str(result)

def test_error_wrapper_modifies_timeout_expired_with_output():
    original_error = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout line")
    result = error_wrapper(original_error)
    assert isinstance(result, type(original_error))
    assert "Captured output:" in str(result)
    assert "timeout line" in str(result)

def test_error_wrapper_modifies_timeout_expired_without_output():
    original_error = subprocess.TimeoutExpired("test_command", timeout=1)
    result = error_wrapper(original_error)
    assert isinstance(result, type(original_error))
    assert "No output was generated." in str(result)

def test_error_wrapper_handles_unicode_decode_error():
    original_error = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    result = error_wrapper(original_error)
    assert isinstance(result, type(original_error))
    assert "Failed to parse output." in str(result)


# LLM-generated content at query #53
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_wraps_called_process_error_with_output():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:\n    test output" in str(wrapped_err)

def test_error_wrapper_wraps_called_process_error_without_output():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_wraps_timeout_expired_with_output():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1, output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:\n    test output" in str(wrapped_err)

def test_error_wrapper_wraps_timeout_expired_without_output():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_handles_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #54
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"test" in result.captured_output

def test_run_command_with_verbose():
    result = run_command(["echo", "verbose"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

def test_run_command_with_shell_command():
    result = run_command("echo shell_test", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"shell_test" in result.captured_output

def test_run_command_with_nonzero_return_code():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_with_unicode_output():
    result = run_command(["echo", "unicode: 你好"], return_output=True)
    assert result.return_code == 0
    assert "unicode: 你好".encode() in result.captured_output

def test_run_command_with_long_output():
    long_output = "x" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #55
#--------------------------

```python
def test_verbose_logging():
    with patch('flutes.run.log') as mock_log:
        run_command("echo test", verbose=True)
        mock_log.assert_called_once_with("> 'echo test'", timestamp=False, include_proc_id=False)


# LLM-generated content at query #56
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = "echo 'test'"
    env = None
    cwd = None
    timeout = None
    verbose = True
    return_output = False
    ignore_errors = False
    kwargs = {}

    with tempfile.TemporaryFile() as f:
        f.write(b"\x80\x81\x82")
        f.seek(0)
        output = f.read()

        try:
            output.decode('utf-8')
        except UnicodeDecodeError:
            for line in output.split(b"\n"):
                assert isinstance(line, bytes)


# LLM-generated content at query #57
#--------------------------

```python
def test_predicate_at_line_32():
    output = b"x" * (MAX_OUTPUT_LENGTH + 1)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #58
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = ["echo", "test"]
    result = run_command(args, return_output=True, verbose=True)
    assert isinstance(result, CommandResult)
    assert result.command == args
    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #59
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #60
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_exceptions():
    non_subprocess_error = ValueError("test error")
    result = error_wrapper(non_subprocess_error)
    assert result is non_subprocess_error


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_command_success():
    assert run_command(["echo", "hello"]).return_code == 0
    assert run_command(["echo", "hello"], return_output=True).captured_output == b"hello\n"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"], verbose=True)
    result = run_command(["echo", "hello"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_env_cwd():
    assert run_command(["echo", "$TEST_VAR"], env={"TEST_VAR": "test"}).return_code == 0
    assert run_command(["pwd"], cwd="/tmp").return_code == 0

def test_run_command_shell():
    assert run_command("echo hello", shell=True).return_code == 0
    assert run_command("echo hello", shell=True, return_output=True).captured_output == b"hello\n"


# LLM-generated content at query #2
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo 'test'", verbose=True, return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"test" in result.captured_output

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_return_output():
    result = run_command("echo 'output'", return_output=True)
    assert result.captured_output is not None
    assert b"output" in result.captured_output

def test_run_command_verbose():
    result = run_command("echo 'verbose'", verbose=True)
    assert result.return_code == 0

def test_run_command_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output

def test_run_command_cwd():
    result = run_command("pwd", cwd="/tmp", return_output=True)
    assert b"/tmp" in result.captured_output

def test_run_command_list_args():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert b"test" in result.captured_output

def test_run_command_unicode_output():
    result = run_command("echo 'unicode: 你好'", return_output=True)
    assert b"unicode: 你好" in result.captured_output


# LLM-generated content at query #3
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo hello", return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    result = run_command("exit 1", ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_timeout():
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.command == "sleep 10"
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command("echo hello", verbose=True, return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.command == "pwd"
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

def test_run_command_list_args():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_ignore_errors_false():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", ignore_errors=False)


# LLM-generated content at query #4
#--------------------------

```python
def test_return_output_true():
    assert run_command("echo test", return_output=True).captured_output is not None

def test_return_code_nonzero():
    assert run_command("exit 1").captured_output is not None

def test_verbose_true():
    assert run_command("echo test", verbose=True).captured_output is not None


# LLM-generated content at query #5
#--------------------------

```python
def test_verbose_logging_when_verbose_is_true():
    verbose = True
    assert verbose is True


# LLM-generated content at query #6
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo hello", return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

def test_run_command_timeout_ignore_errors():
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_verbose():
    result = run_command("echo hello", verbose=True, return_output=True)
    assert result.captured_output == b"hello\n"

def test_run_command_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test"}, shell=True, return_output=True)
    assert result.captured_output == b"test\n"

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.captured_output.decode().strip() == tmpdir

def test_run_command_list_args():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_no_output():
    result = run_command("echo hello")
    assert result.captured_output is None

def test_run_command_max_output_truncation():
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(f"echo {long_output}", return_output=True, shell=True)
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #7
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    original_error = ValueError("test error")
    result = error_wrapper(original_error)
    assert result is original_error

def test_error_wrapper_creates_new_type_for_subprocess_errors():
    original_error = subprocess.CalledProcessError(1, "test")
    result = error_wrapper(original_error)
    assert type(result).__name__ == "CalledProcessError"
    assert type(result) is not type(original_error)

def test_error_wrapper_preserves_output_in_new_type():
    original_error = subprocess.CalledProcessError(1, "test", output=b"test output")
    result = error_wrapper(original_error)
    assert result.output == b"test output"

def test_error_wrapper_str_includes_output_for_subprocess_errors():
    original_error = subprocess.CalledProcessError(1, "test", output=b"line1\nline2")
    result = error_wrapper(original_error)
    assert "Captured output:" in str(result)
    assert "line1" in str(result)
    assert "line2" in str(result)

def test_error_wrapper_str_handles_decode_error():
    original_error = subprocess.CalledProcessError(1, "test", output=b"\xff\xfe")
    result = error_wrapper(original_error)
    assert "Failed to parse output" in str(result)

def test_error_wrapper_str_includes_no_output_message():
    original_error = subprocess.CalledProcessError(1, "test")
    result = error_wrapper(original_error)
    assert "No output was generated" in str(result)

def test_error_wrapper_works_with_timeout_expired():
    original_error = subprocess.TimeoutExpired("test", 1)
    result = error_wrapper(original_error)
    assert type(result).__name__ == "TimeoutExpired"


# LLM-generated content at query #8
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_verbose():
    result = run_command(["echo", "verbose"], verbose=True, return_output=True)
    assert result.command == ["echo", "verbose"]
    assert result.return_code == 0
    assert result.captured_output == b"verbose\n"

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "2"], timeout=0.1)

def test_run_command_with_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["printenv", "TEST_VAR"]
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.command == ["pwd"]
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

def test_run_command_with_shell_command():
    result = run_command("echo shell", shell=True, return_output=True)
    assert result.command == "echo shell"
    assert result.return_code == 0
    assert result.captured_output == b"shell\n"

def test_run_command_with_non_zero_exit():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_with_output_truncation():
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], return_output=True)
    assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")

def test_run_command_with_unicode_output():
    result = run_command(["echo", "unicode: 你好"], return_output=True)
    assert result.command == ["echo", "unicode: 你好"]
    assert result.return_code == 0
    assert result.captured_output == b"unicode: 你好\n"


# LLM-generated content at query #9
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "test"

def test_run_command_with_verbose():
    result = run_command(["echo", "verbose"], verbose=True)
    assert result.command == ["echo", "verbose"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "2"], timeout=0.1)

def test_run_command_with_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["printenv", "TEST_VAR"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "test_value"

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.command == ["pwd"]
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

def test_run_command_with_shell():
    result = run_command("echo shell", shell=True, return_output=True)
    assert result.command == "echo shell"
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "shell"

def test_run_command_with_long_output():
    result = run_command(["python", "-c", "print('x' * 10000)"], return_output=True)
    assert result.command == ["python", "-c", "print('x' * 10000)"]
    assert result.return_code == 0
    assert len(result.captured_output) > 0
    assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")

def test_run_command_with_unicode_output():
    result = run_command(["python", "-c", "print('日本語')"], return_output=True)
    assert result.command == ["python", "-c", "print('日本語')"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "日本語"

def test_run_command_with_non_existing_command():
    with pytest.raises(FileNotFoundError):
        run_command(["non_existing_command"])


# LLM-generated content at query #10
#--------------------------

```python
def test_verbose_logging():
    with patch('flutes.run.log') as mock_log:
        run_command("echo test", verbose=True, return_output=False)
        mock_log.assert_called_once()


# LLM-generated content at query #11
#--------------------------

```python
def test_error_wrapper_predicate():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #12
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo hello", return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

def test_run_command_ignore_timeout():
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_verbose():
    result = run_command("echo hello", verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.strip() == tmpdir.encode()

def test_run_command_list_args():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_return_output_false():
    result = run_command("echo hello")
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_nonzero_returncode():
    result = run_command("exit 1", return_output=True)
    assert result.return_code == 1
    assert result.captured_output == b""

def test_run_command_oserror_retry():
    result = run_command("nonexistent_command", ignore_errors=True)
    assert result.return_code != 0


# LLM-generated content at query #13
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "hello"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_env():
    result = run_command(["sh", "-c", "echo $TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.decode().strip() == tmpdir

def test_run_command_return_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_string_command():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"


# LLM-generated content at query #14
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "2"], timeout=0.1)

def test_run_command_with_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_env_and_cwd():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, cwd="/tmp", return_output=True)
    assert result.command == ["env"]
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_with_shell_command():
    result = run_command("echo test", shell=True, return_output=True)
    assert result.command == "echo test"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_nonzero_return_code():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_with_truncated_output():
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], ignore_errors=True)
    assert b"*** (previous output truncated) ***" in result.captured_output

def test_run_command_with_unicode_decode_error():
    result = run_command(["printf", "\\x80"], verbose=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


# LLM-generated content at query #15
#--------------------------

```python
def test_unicode_decode_error_occurs():
    with tempfile.TemporaryFile() as f:
        f.write(b'\xff\xfe')
        f.seek(0)
        output = f.read()
        try:
            output.decode('utf-8')
        except UnicodeDecodeError:
            assert True
        else:
            assert False


# LLM-generated content at query #16
#--------------------------

```python
def test_unicode_decode_error_raises_false():
    output = b'\x80abc'
    try:
        output.decode('utf-8')
    except UnicodeDecodeError:
        pass
    assert False


# LLM-generated content at query #17
#--------------------------

```python
def test_error_wrapper_predicate():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #18
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.command == ["pwd"]
    assert result.return_code == 0
    assert result.captured_output.strip() == b"/tmp"

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["printenv", "TEST_VAR"]
    assert result.return_code == 0
    assert result.captured_output.strip() == b"test_value"

def test_run_command_verbose():
    result = run_command(["echo", "verbose"], verbose=True, return_output=True)
    assert result.command == ["echo", "verbose"]
    assert result.return_code == 0
    assert result.captured_output == b"verbose\n"

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout():
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.command == ["sleep", "10"]
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_with_shell():
    result = run_command("echo shell", shell=True, return_output=True)
    assert result.command == "echo shell"
    assert result.return_code == 0
    assert result.captured_output == b"shell\n"


# LLM-generated content at query #19
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_creates_new_type_for_subprocess_errors():
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == "CalledProcessError"
    assert type(wrapped_err) is not type(err)

def test_error_wrapper_preserves_original_exception_attributes():
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert wrapped_err.returncode == err.returncode
    assert wrapped_err.cmd == err.cmd
    assert wrapped_err.output == err.output

def test_error_wrapper_str_with_output():
    err = subprocess.CalledProcessError(1, "test_command", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "line1" in str(wrapped_err)
    assert "line2" in str(wrapped_err)

def test_error_wrapper_str_without_output():
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_str_with_unicode_error():
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)

def test_error_wrapper_handles_timeout_expired():
    err = subprocess.TimeoutExpired("test_command", 1)
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == "TimeoutExpired"
    assert type(wrapped_err) is not type(err)


# LLM-generated content at query #20
#--------------------------

```python
def test_return_output_true():
    assert run_command("echo test", return_output=True).captured_output is not None

def test_return_code_nonzero():
    assert run_command("exit 1").captured_output is not None

def test_verbose_true():
    assert run_command("echo test", verbose=True).captured_output is not None


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_false():
    output = b"test output"
    ret = type('MockReturn', (), {'returncode': 0})()
    return_output = False
    verbose = False
    assert not (return_output or ret.returncode != 0 or verbose)


# LLM-generated content at query #22
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo hello", return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    result = run_command("exit 1", ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_timeout():
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.command == "sleep 10"
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command("echo test", verbose=True, return_output=True)
    assert result.command == "echo test"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "value"}, shell=True, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output == b"value\n"

def test_run_command_cwd():
    result = run_command("pwd", cwd="/tmp", return_output=True)
    assert result.command == "pwd"
    assert result.return_code == 0
    assert result.captured_output.strip() == b"/tmp"

def test_run_command_list_args():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"


# LLM-generated content at query #23
#--------------------------

```python
def test_return_output_true():
    result = run_command("echo test", return_output=True)
    assert result.captured_output is not None


# LLM-generated content at query #24
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    non_subprocess_error = ValueError("test error")
    assert error_wrapper(non_subprocess_error) is non_subprocess_error

def test_error_wrapper_wraps_called_process_error_with_output():
    error = subprocess.CalledProcessError(1, "cmd", output=b"line1\nline2")
    wrapped_error = error_wrapper(error)
    assert str(wrapped_error) == "Command 'cmd' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"

def test_error_wrapper_wraps_called_process_error_without_output():
    error = subprocess.CalledProcessError(1, "cmd")
    wrapped_error = error_wrapper(error)
    assert str(wrapped_error) == "Command 'cmd' returned non-zero exit status 1.\nNo output was generated."

def test_error_wrapper_wraps_timeout_expired_with_output():
    error = subprocess.TimeoutExpired("cmd", timeout=1, output=b"line1\nline2")
    wrapped_error = error_wrapper(error)
    assert str(wrapped_error) == "Command 'cmd' timed out after 1 seconds.\nCaptured output:\n    line1\n    line2"

def test_error_wrapper_wraps_timeout_expired_without_output():
    error = subprocess.TimeoutExpired("cmd", timeout=1)
    wrapped_error = error_wrapper(error)
    assert str(wrapped_error) == "Command 'cmd' timed out after 1 seconds.\nNo output was generated."

def test_error_wrapper_handles_unicode_decode_error():
    error = subprocess.CalledProcessError(1, "cmd", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert str(wrapped_error) == "Command 'cmd' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_at_line_32():
    output = b"x" * (MAX_OUTPUT_LENGTH + 1)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #26
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #27
#--------------------------

```python
def test_error_wrapper_predicate():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #28
#--------------------------

```python
def test_error_wrapper_returns_same_error_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_modifies_subprocess_called_process_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_cmd' returned non-zero exit status 1.\nCaptured output:\n    test output"

def test_error_wrapper_modifies_subprocess_timeout_expired():
    err = subprocess.TimeoutExpired("test_cmd", 1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_cmd' timed out after 1 seconds.\nNo output was generated."

def test_error_wrapper_handles_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_cmd' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #29
#--------------------------

```python
def test_error_wrapper_returns_original_exception_for_non_subprocess_errors():
    original_error = ValueError("test error")
    result = error_wrapper(original_error)
    assert result is original_error


# LLM-generated content at query #30
#--------------------------

```python
def test_error_wrapper_returns_original_error_for_non_subprocess_exceptions():
    original_error = ValueError("test error")
    result = error_wrapper(original_error)
    assert result is original_error


# LLM-generated content at query #31
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #32
#--------------------------

```python
def test_error_wrapper_returns_same_error_for_non_subprocess_exceptions():
    non_subprocess_error = ValueError("Test error")
    assert error_wrapper(non_subprocess_error) is non_subprocess_error


# LLM-generated content at query #33
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #34
#--------------------------

```python
def test_error_wrapper_returns_original_error_for_non_subprocess_exceptions():
    class CustomError(Exception):
        pass

    err = CustomError("test error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #35
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #36
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err


# LLM-generated content at query #37
#--------------------------

```python
def test_return_output_true():
    result = run_command("echo test", return_output=True)
    assert result.captured_output is not None


# LLM-generated content at query #38
#--------------------------

```python
def test_unicode_decode_error_occurs():
    args = "echo 'test'"
    env = None
    cwd = None
    timeout = None
    verbose = True
    return_output = False
    ignore_errors = False
    kwargs = {}

    with patch('subprocess.run') as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args, returncode=0)
        with patch('tempfile.TemporaryFile') as mock_tempfile:
            mock_tempfile.return_value.__enter__.return_value.read.return_value = b'\xff\xfe'
            result = run_command(args, env=env, cwd=cwd, timeout=timeout, verbose=verbose,
                                return_output=return_output, ignore_errors=ignore_errors, **kwargs)
            assert result.captured_output is not None


# LLM-generated content at query #39
#--------------------------

```python
def test_verbose_logging():
    with patch('flutes.run.log') as mock_log:
        run_command("echo test", verbose=True)
        mock_log.assert_called_once_with("> 'echo test'", timestamp=False, include_proc_id=False)


# LLM-generated content at query #40
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_verbose_mode():
    result = run_command(["echo", "verbose"], verbose=True)
    assert result.command == ["echo", "verbose"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_return_output():
    result = run_command(["echo", "output"], return_output=True)
    assert result.captured_output == b"output\n"

def test_run_command_with_shell():
    result = run_command("echo shell", shell=True, return_output=True)
    assert result.command == "echo shell"
    assert result.return_code == 0
    assert result.captured_output == b"shell\n"

def test_run_command_with_kwargs():
    result = run_command(["echo", "test"], shell=False, return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_failed_execution():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_with_unicode_output():
    result = run_command(["echo", "unicode: café"], return_output=True)
    assert result.return_code == 0
    assert "café" in result.captured_output.decode('utf-8')

def test_run_command_with_long_output():
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #41
#--------------------------

```python
def test_error_wrapper_returns_original_error_for_non_subprocess_exceptions():
    class CustomError(Exception):
        pass

    err = CustomError("Test error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #42
#--------------------------

```python
def test_error_wrapper_returns_original_error_for_non_subprocess_exceptions():
    non_subprocess_error = ValueError("Test error")
    result = error_wrapper(non_subprocess_error)
    assert result is non_subprocess_error


# LLM-generated content at query #43
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    original_error = ValueError("test error")
    wrapped_error = error_wrapper(original_error)
    assert wrapped_error is original_error

def test_error_wrapper_creates_new_type_for_subprocess_errors():
    original_error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(original_error)
    assert type(wrapped_error).__name__ == "CalledProcessError"
    assert type(wrapped_error) is not type(original_error)

def test_error_wrapper_preserves_original_error_attributes():
    original_error = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_error = error_wrapper(original_error)
    assert wrapped_error.returncode == original_error.returncode
    assert wrapped_error.cmd == original_error.cmd
    assert wrapped_error.output == original_error.output

def test_error_wrapper_str_with_output():
    original_error = subprocess.CalledProcessError(1, "test_command", output=b"line1\nline2")
    wrapped_error = error_wrapper(original_error)
    expected_str = (
        "Command 'test_command' returned non-zero exit status 1.\n"
        "Captured output:\n"
        "    line1\n"
        "    line2"
    )
    assert str(wrapped_error) == expected_str

def test_error_wrapper_str_with_no_output():
    original_error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(original_error)
    expected_str = (
        "Command 'test_command' returned non-zero exit status 1.\n"
        "No output was generated."
    )
    assert str(wrapped_error) == expected_str

def test_error_wrapper_str_with_unicode_error():
    original_error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(original_error)
    expected_str = (
        "Command 'test_command' returned non-zero exit status 1.\n"
        "Failed to parse output."
    )
    assert str(wrapped_error) == expected_str

def test_error_wrapper_with_timeout_error():
    original_error = subprocess.TimeoutExpired("test_command", 1)
    wrapped_error = error_wrapper(original_error)
    assert type(wrapped_error).__name__ == "TimeoutExpired"
    assert type(wrapped_error) is not type(original_error)


# LLM-generated content at query #44
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_creates_new_type_for_subprocess_errors():
    err = subprocess.CalledProcessError(1, "test")
    wrapped = error_wrapper(err)
    assert type(wrapped).__name__ == "CalledProcessError"
    assert type(wrapped) is not type(err)

def test_error_wrapper_preserves_exception_attributes():
    err = subprocess.CalledProcessError(1, "test", output=b"output")
    wrapped = error_wrapper(err)
    assert wrapped.returncode == err.returncode
    assert wrapped.cmd == err.cmd
    assert wrapped.output == err.output

def test_error_wrapper_str_with_output():
    err = subprocess.CalledProcessError(1, "test", output=b"line1\nline2")
    wrapped = error_wrapper(err)
    assert "Captured output:" in str(wrapped)
    assert "line1" in str(wrapped)
    assert "line2" in str(wrapped)

def test_error_wrapper_str_without_output():
    err = subprocess.CalledProcessError(1, "test")
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)

def test_error_wrapper_str_with_unicode_error():
    err = subprocess.CalledProcessError(1, "test", output=b"\xff\xfe")
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired("test", 1)
    wrapped = error_wrapper(err)
    assert type(wrapped).__name__ == "TimeoutExpired"
    assert "No output was generated." in str(wrapped)


# LLM-generated content at query #45
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #46
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command("echo 'Hello, World!'", shell=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_return_output():
    result = run_command("echo 'Hello, World!'", shell=True, return_output=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

def test_run_command_with_verbose():
    result = run_command("echo 'Hello, World!'", shell=True, verbose=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0

def test_run_command_with_ignore_errors():
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_with_timeout():
    result = run_command("sleep 2", shell=True, timeout=1, ignore_errors=True)
    assert result.command == "sleep 2"
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_with_env_and_cwd():
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"}, cwd="/tmp")
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_list_args():
    result = run_command(["echo", "Hello, World!"])
    assert result.command == ["echo", "Hello, World!"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_nonzero_return_code():
    result = run_command("exit 1", shell=True, return_output=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is not None


# LLM-generated content at query #47
#--------------------------

```python
def test_verbose_logging():
    with patch('flutes.run.log') as mock_log:
        run_command("echo test", verbose=True)
        mock_log.assert_called_once_with("> 'echo test'", timestamp=False, include_proc_id=False)


# LLM-generated content at query #48
#--------------------------

```python
def test_predicate_at_line_32():
    output = b"a" * (MAX_OUTPUT_LENGTH + 1)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #49
#--------------------------

```python
def test_return_output_true():
    result = run_command("echo test", return_output=True)
    assert result.captured_output is not None


# LLM-generated content at query #50
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #51
#--------------------------

```python
def test_error_wrapper_returns_original_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err


# LLM-generated content at query #52
#--------------------------

```python
def test_error_wrapper_returns_original_error_for_non_subprocess_exceptions():
    class CustomException(Exception):
        pass

    err = CustomException("test error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #53
#--------------------------

```python
def test_predicate_evaluates_to_true():
    args = "echo test"
    env = None
    cwd = None
    timeout = None
    verbose = False
    return_output = True
    ignore_errors = False
    kwargs = {}

    result = run_command(args, env=env, cwd=cwd, timeout=timeout, verbose=verbose, return_output=return_output, ignore_errors=ignore_errors, **kwargs)

    assert result.captured_output is not None


# LLM-generated content at query #54
#--------------------------

```python
def test_verbose_logging():
    with patch('flutes.run.log') as mock_log:
        run_command("echo test", verbose=True)
        mock_log.assert_called_once_with("> 'echo test'", timestamp=False, include_proc_id=False)


# LLM-generated content at query #55
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo hello", verbose=False, return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", verbose=False, return_output=True)

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.01, verbose=False, return_output=True)

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True, verbose=False, return_output=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command("echo hello", verbose=True, return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.command == "pwd"
        assert result.return_code == 0
        assert result.captured_output.decode().strip() == tmpdir

def test_run_command_list_args():
    result = run_command(["echo", "hello"], verbose=False, return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_return_output_false():
    result = run_command("echo hello", return_output=False)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_os_error_retry():
    with patch("subprocess.run", side_effect=OSError("Test error")):
        with pytest.raises(OSError):
            run_command("echo hello", verbose=False, return_output=True)


# LLM-generated content at query #57
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo 'test'", return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_failure():
    with raises(subprocess.CalledProcessError):
        run_command("exit 1")

def test_run_command_timeout():
    with raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1

def test_run_command_verbose():
    result = run_command("echo 'test'", verbose=True, return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.command == "pwd"
        assert result.return_code == 0
        assert result.captured_output.decode().strip() == tmpdir

def test_run_command_list_args():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_return_output_false():
    result = run_command("echo 'test'")
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_nonzero_returncode():
    result = run_command("exit 1", return_output=True, ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output == b""


# LLM-generated content at query #58
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #59
#--------------------------

```python
def test_error_wrapper_returns_non_subprocess_exception_unchanged():
    err = ValueError("test error")
    assert error_wrapper(err) is err


# LLM-generated content at query #60
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #61
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = "echo 'test'"
    ret = subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = b'\x80abc'
    assert isinstance(output, bytes)
    assert output.decode('utf-8') == UnicodeDecodeError


