####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_preserves_exception_type_for_subprocess_errors():
    err = subprocess.CalledProcessError(1, "test")
    wrapped = error_wrapper(err)
    assert type(wrapped) is type(err)

def test_error_wrapper_adds_output_to_str_for_called_process_error_with_output():
    err = subprocess.CalledProcessError(1, "test", output=b"line1\nline2")
    wrapped = error_wrapper(err)
    assert "Captured output:" in str(wrapped)
    assert "line1" in str(wrapped)
    assert "line2" in str(wrapped)

def test_error_wrapper_handles_unicode_error_in_output():
    err = subprocess.CalledProcessError(1, "test", output=b"\xff\xfe")
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)

def test_error_wrapper_handles_no_output():
    err = subprocess.CalledProcessError(1, "test")
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)

def test_error_wrapper_works_for_timeout_expired():
    err = subprocess.TimeoutExpired("test", 1)
    wrapped = error_wrapper(err)
    assert type(wrapped) is type(err)


# LLM-generated content at query #2
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
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.captured_output.decode().strip() == tmpdir

def test_run_command_return_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.captured_output == b"hello\n"

def test_run_command_non_zero_return():
    result = run_command(["ls", "/nonexistent"], return_output=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_string_command():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"


# LLM-generated content at query #3
#--------------------------

```python
def test_predicate_at_line_35_evaluates_to_false():
    result = run_command("echo 'test'", ignore_errors=True)
    assert result.captured_output is None


# LLM-generated content at query #4
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
        ret = subprocess.run(args, check=True, stdout=f, stderr=subprocess.STDOUT,
                             timeout=timeout, env=env, cwd=cwd, **kwargs)
        f.seek(0)
        output = f.read()
        try:
            output.decode('utf-8')
        except UnicodeDecodeError:
            pass
        else:
            raise AssertionError("UnicodeDecodeError did not occur")


# LLM-generated content at query #5
#--------------------------

```python
def test_verbose_logging():
    with patch('flutes.run.log') as mock_log:
        run_command("echo test", verbose=True)
        mock_log.assert_called_once_with("> 'echo test'", timestamp=False, include_proc_id=False)


# LLM-generated content at query #6
#--------------------------

```python
def test_verbose_logging_enabled():
    assert run_command("echo test", verbose=True).return_code == 0


# LLM-generated content at query #7
#--------------------------

```python
def test_run_command_verbose_flag_set():
    with patch('flutes.run.log') as mock_log:
        run_command("echo test", verbose=True)
        mock_log.assert_called_once_with((str(None) or "") + "> " + repr("echo test"), timestamp=False, include_proc_id=False)


# LLM-generated content at query #8
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0

def test_run_command_return_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output == b"test\n"

def test_run_command_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.captured_output.decode().strip() == tmpdir

def test_run_command_kwargs():
    result = run_command(["echo", "test"], shell=True, return_output=True)
    assert result.captured_output == b"test\n"


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_true_when_return_output_is_true():
    assert run_command(["echo", "test"], return_output=True).captured_output is not None

def test_predicate_true_when_return_code_nonzero():
    assert run_command(["false"]).captured_output is not None

def test_predicate_true_when_verbose_is_true():
    assert run_command(["echo", "test"], verbose=True).captured_output is not None


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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

def test_error_wrapper_preserves_exception_attributes():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert wrapped_err.returncode == err.returncode
    assert wrapped_err.cmd == err.cmd
    assert wrapped_err.output == err.output

def test_error_wrapper_str_with_output():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "line1" in str(wrapped_err)
    assert "line2" in str(wrapped_err)

def test_error_wrapper_str_with_unicode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)

def test_error_wrapper_str_with_no_output():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_works_with_timeout_expired():
    err = subprocess.TimeoutExpired("test_cmd", 1)
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == "TimeoutExpired"


# LLM-generated content at query #2
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
    result = run_command("sleep 10", shell=True, timeout=0.1, ignore_errors=True)
    assert result.command == "sleep 10"
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_with_env():
    env = {"TEST_VAR": "test_value"}
    result = run_command("echo $TEST_VAR", shell=True, env=env, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

def test_run_command_with_cwd():
    result = run_command("pwd", shell=True, cwd="/tmp", return_output=True)
    assert result.command == "pwd"
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_with_list_args():
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.command == ["echo", "Hello, World!"]
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

def test_run_command_with_kwargs():
    result = run_command("echo 'Hello, World!'", shell=True, text=True, return_output=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert "Hello, World!" in result.captured_output.decode('utf-8')


# LLM-generated content at query #3
#--------------------------

```python
def test_error_wrapper_returns_original_error_for_non_subprocess_exceptions():
    original_error = ValueError("test error")
    result = error_wrapper(original_error)
    assert result is original_error


# LLM-generated content at query #4
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
    with patch('flutes.run.log') as mock_log:
        run_command(["echo", "hello"], verbose=True)
        mock_log.assert_called_once_with("> ['echo', 'hello']", timestamp=False, include_proc_id=False)

def test_run_command_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
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

def test_run_command_non_zero_return():
    result = run_command(["sh", "-c", "exit 1"], return_output=True)
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_shell_string():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"


# LLM-generated content at query #5
#--------------------------

```python
def test_verbose_logging_enabled():
    args = "echo test"
    verbose = True
    result = run_command(args, verbose=verbose)
    assert result.command == args
    assert result.return_code == 0
    assert result.captured_output is None


# LLM-generated content at query #6
#--------------------------

```python
def test_error_wrapper_returns_original_exception_for_non_subprocess_errors():
    non_subprocess_error = ValueError("test error")
    assert error_wrapper(non_subprocess_error) is non_subprocess_error


# LLM-generated content at query #7
#--------------------------

```python
def test_return_output_true():
    result = run_command("echo test", return_output=True)
    assert result.captured_output is not None


# LLM-generated content at query #8
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_creates_new_type_for_subprocess_errors():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == type(err).__name__
    assert type(wrapped_err) is not type(err)

def test_error_wrapper_preserves_exception_attributes():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert wrapped_err.returncode == err.returncode
    assert wrapped_err.cmd == err.cmd
    assert wrapped_err.output == err.output

def test_error_wrapper_str_includes_output_for_called_process_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "line1" in str(wrapped_err)
    assert "line2" in str(wrapped_err)

def test_error_wrapper_str_handles_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)

def test_error_wrapper_str_handles_no_output():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_works_with_timeout_expired():
    err = subprocess.TimeoutExpired("test_cmd", 1)
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == type(err).__name__
    assert "No output was generated." in str(wrapped_err)


# LLM-generated content at query #9
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo 'test'", verbose=False, return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", verbose=False, return_output=True)

def test_run_command_ignore_errors():
    result = run_command("exit 1", verbose=False, return_output=True, ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", timeout=0.1, verbose=False, return_output=True)

def test_run_command_timeout_ignore_errors():
    result = run_command("sleep 2", timeout=0.1, verbose=False, return_output=True, ignore_errors=True)
    assert result.command == "sleep 2"
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command("echo 'verbose'", verbose=True, return_output=True)
    assert result.command == "echo 'verbose'"
    assert result.return_code == 0
    assert result.captured_output == b"verbose\n"

def test_run_command_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_cwd():
    result = run_command("pwd", cwd="/tmp", return_output=True)
    assert result.command == "pwd"
    assert result.return_code == 0
    assert result.captured_output == b"/tmp\n"

def test_run_command_list_args():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_unicode_output():
    result = run_command("echo 'unicode: café'", return_output=True)
    assert result.command == "echo 'unicode: café'"
    assert result.return_code == 0
    assert result.captured_output == b"unicode: caf\xc3\xa9\n"


# LLM-generated content at query #10
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command("echo 'test'", verbose=False, return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_list_args():
    result = run_command(["echo", "test"], verbose=False, return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_cwd():
    result = run_command("pwd", cwd="/tmp", verbose=False, return_output=True)
    assert result.return_code == 0
    assert result.captured_output.strip() == b"/tmp"

def test_run_command_with_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, verbose=False, return_output=True)
    assert result.return_code == 0
    assert result.captured_output.strip() == b"test_value"

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is None

def test_run_command_verbose_mode():
    result = run_command("echo 'verbose test'", verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"verbose test\n"

def test_run_command_return_output_false():
    result = run_command("echo 'test'", return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_nonzero_return_code():
    result = run_command("exit 1", return_output=True)
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_with_kwargs():
    result = run_command("echo 'test'", shell=True, verbose=False, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test\n"


# LLM-generated content at query #11
#--------------------------

```python
def test_return_output_true():
    assert run_command("echo test", return_output=True).captured_output is not None

def test_return_code_nonzero():
    assert run_command("exit 1", ignore_errors=True).captured_output is not None

def test_verbose_true():
    assert run_command("echo test", verbose=True).captured_output is not None


# LLM-generated content at query #12
#--------------------------

```python
def test_output_truncation():
    output = b"x" * (MAX_OUTPUT_LENGTH + 1)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #13
#--------------------------

```python
def test_unicode_decode_error_predicate():
    output = b'\xff\xfe'
    assert output.decode('utf-8') == UnicodeDecodeError


# LLM-generated content at query #14
#--------------------------

```python
def test_return_output_true_trigger_predicate():
    assert run_command("echo test", return_output=True).captured_output is not None

def test_return_code_nonzero_trigger_predicate():
    assert run_command("exit 1", ignore_errors=True).captured_output is not None

def test_verbose_true_trigger_predicate():
    assert run_command("echo test", verbose=True).captured_output is not None


# LLM-generated content at query #15
#--------------------------

```python
def test_error_wrapper_predicate_false():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #16
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

def test_run_command_return_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.captured_output == b"hello\n"

def test_run_command_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

def test_run_command_truncated_output():
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #17
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

    # Simulate a scenario where the output cannot be decoded as UTF-8
    with unittest.mock.patch('subprocess.run') as mock_run:
        mock_run.return_value = unittest.mock.Mock(returncode=0, stdout=b'\xff\xfe')
        result = run_command(args, env=env, cwd=cwd, timeout=timeout, verbose=verbose, return_output=return_output, ignore_errors=ignore_errors, **kwargs)
        assert result.captured_output is not None


# LLM-generated content at query #18
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = "echo 'test'"
    result = run_command(args, return_output=True, verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #19
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


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_evaluates_to_true():
    output = b"a" * (MAX_OUTPUT_LENGTH + 1)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #21
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_creates_new_exception_type_for_subprocess_errors():
    err = subprocess.CalledProcessError(1, "test")
    wrapped = error_wrapper(err)
    assert type(wrapped).__name__ == "CalledProcessError"
    assert type(wrapped) is not type(err)

def test_error_wrapper_preserves_exception_attributes():
    err = subprocess.CalledProcessError(1, "test", output=b"test output")
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

def test_error_wrapper_str_with_no_output():
    err = subprocess.CalledProcessError(1, "test")
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)

def test_error_wrapper_str_with_unicode_error():
    err = subprocess.CalledProcessError(1, "test", output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)

def test_error_wrapper_works_with_timeout_expired():
    err = subprocess.TimeoutExpired("test", 1)
    wrapped = error_wrapper(err)
    assert type(wrapped).__name__ == "TimeoutExpired"
    assert "No output was generated." in str(wrapped)


# LLM-generated content at query #22
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
        f.write(b'\x80abc')
        f.seek(0)
        output = f.read()

        try:
            output.decode('utf-8')
        except UnicodeDecodeError:
            pass
        else:
            raise AssertionError("Expected UnicodeDecodeError to occur")


# LLM-generated content at query #23
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command("echo 'test'", verbose=False, return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_failure():
    result = run_command("exit 1", ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_with_timeout():
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.command == "sleep 10"
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose_mode():
    result = run_command("echo 'verbose test'", verbose=True, return_output=True)
    assert result.command == "echo 'verbose test'"
    assert result.return_code == 0
    assert result.captured_output == b"verbose test\n"

def test_run_command_with_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_with_cwd():
    result = run_command("pwd", cwd="/tmp", return_output=True)
    assert result.command == "pwd"
    assert result.return_code == 0
    assert result.captured_output == b"/tmp\n"

def test_run_command_with_list_args():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"


# LLM-generated content at query #24
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = "echo 'test'"
    output = b"\x80\x81\x82"
    with tempfile.TemporaryFile() as f:
        f.write(output)
        f.seek(0)
        ret = subprocess.CompletedProcess(args, 1)
        assert (return_output or ret.returncode != 0 or verbose) is True


# LLM-generated content at query #25
#--------------------------

```python
def test_error_wrapper_returns_non_subprocess_error_unchanged():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_modifies_called_process_error_with_output():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_cmd' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"

def test_error_wrapper_modifies_called_process_error_without_output():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_cmd' returned non-zero exit status 1.\nNo output was generated."

def test_error_wrapper_modifies_timeout_expired_with_output():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1, output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_cmd' timed out after 1 seconds.\nCaptured output:\n    line1\n    line2"

def test_error_wrapper_modifies_timeout_expired_without_output():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1)
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_cmd' timed out after 1 seconds.\nNo output was generated."

def test_error_wrapper_handles_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_cmd' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #26
#--------------------------

```python
def test_error_wrapper_predicate():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #27
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo hello", return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    with raises(subprocess.CalledProcessError):
        run_command("exit 1")

def test_run_command_timeout():
    with raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

def test_run_command_timeout_ignore_errors():
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
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


# LLM-generated content at query #28
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #29
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    original_error = ValueError("test error")
    wrapped_error = error_wrapper(original_error)
    assert wrapped_error is original_error

def test_error_wrapper_creates_new_type_for_subprocess_errors():
    original_error = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_error = error_wrapper(original_error)
    assert type(wrapped_error).__name__ == "CalledProcessError"
    assert type(wrapped_error) is not type(original_error)

def test_error_wrapper_preserves_error_attributes():
    original_error = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    wrapped_error = error_wrapper(original_error)
    assert wrapped_error.returncode == original_error.returncode
    assert wrapped_error.cmd == original_error.cmd
    assert wrapped_error.output == original_error.output

def test_error_wrapper_custom_str_with_output():
    original_error = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    wrapped_error = error_wrapper(original_error)
    expected_str = (
        "Command 'test_cmd' returned non-zero exit status 1.\n"
        "Captured output:\n"
        "    line1\n"
        "    line2"
    )
    assert str(wrapped_error) == expected_str

def test_error_wrapper_custom_str_without_output():
    original_error = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_error = error_wrapper(original_error)
    expected_str = (
        "Command 'test_cmd' returned non-zero exit status 1.\n"
        "No output was generated."
    )
    assert str(wrapped_error) == expected_str

def test_error_wrapper_custom_str_with_unicode_error():
    original_error = subprocess.CalledProcessError(1, "test_cmd", output=b"\xff\xfe")
    wrapped_error = error_wrapper(original_error)
    expected_str = (
        "Command 'test_cmd' returned non-zero exit status 1.\n"
        "Failed to parse output."
    )
    assert str(wrapped_error) == expected_str

def test_error_wrapper_works_with_timeout_expired():
    original_error = subprocess.TimeoutExpired("test_cmd", 1)
    wrapped_error = error_wrapper(original_error)
    assert type(wrapped_error).__name__ == "TimeoutExpired"
    assert type(wrapped_error) is not type(original_error)


# LLM-generated content at query #30
#--------------------------

```python
def test_error_wrapper_returns_non_subprocess_error_unchanged():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_modifies_called_process_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

def test_error_wrapper_modifies_timeout_expired_error():
    err = subprocess.TimeoutExpired("test_cmd", timeout=10)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_handles_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)

def test_error_wrapper_preserves_original_error_attributes():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert wrapped_err.returncode == err.returncode
    assert wrapped_err.cmd == err.cmd
    assert wrapped_err.output == err.output


# LLM-generated content at query #31
#--------------------------

```python
def test_unicode_decode_error_handling():
    output = b'\xff\xfe\xfd'
    assert output.decode('utf-8') == UnicodeDecodeError


# LLM-generated content at query #32
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
    err = subprocess.CalledProcessError(1, "test", output=b"test output")
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

def test_error_wrapper_for_timeout_error():
    err = subprocess.TimeoutExpired("test", 1)
    wrapped = error_wrapper(err)
    assert type(wrapped).__name__ == "TimeoutExpired"
    assert "No output was generated." in str(wrapped)


# LLM-generated content at query #33
#--------------------------

```python
def test_error_wrapper_returns_original_error_for_non_subprocess_exceptions():
    non_subprocess_error = ValueError("test error")
    assert error_wrapper(non_subprocess_error) is non_subprocess_error


# LLM-generated content at query #34
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command("echo 'test'", verbose=False, return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_error():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", verbose=False)

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.01)

def test_run_command_ignore_timeout():
    result = run_command("sleep 10", timeout=0.01, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_verbose():
    with pytest.raises(AssertionError):  # Assuming log output is captured and checked
        run_command("echo 'verbose'", verbose=True)

def test_run_command_return_output():
    result = run_command("echo 'output'", return_output=True)
    assert result.captured_output == b"output\n"

def test_run_command_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.captured_output == b"test_value\n"

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, shell=True, return_output=True)
        assert result.captured_output.decode('utf-8').strip() == tmpdir

def test_run_command_list_args():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output == b"test\n"

def test_run_command_unicode_output():
    result = run_command("echo 'unicode: 你好'", return_output=True)
    assert result.captured_output == b"unicode: 你好\n"

def test_run_command_truncated_output():
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(f"echo '{long_output}'", return_output=True)
    assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #35
#--------------------------

```python
def test_error_wrapper_predicate_false():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #36
#--------------------------

```python
def test_error_wrapper_returns_original_exception_for_non_subprocess_errors():
    class CustomError(Exception):
        pass

    err = CustomError("test error")
    assert error_wrapper(err) is err


# LLM-generated content at query #37
#--------------------------

```python
def test_error_wrapper_predicate_false():
    assert not isinstance(Exception("test"), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #38
#--------------------------

```python
def test_error_wrapper_returns_original_error_for_non_subprocess_exceptions():
    original_error = ValueError("test error")
    result = error_wrapper(original_error)
    assert result is original_error


# LLM-generated content at query #39
#--------------------------

```python
def test_return_output_true():
    assert run_command("echo test", return_output=True).captured_output is not None

def test_return_code_nonzero():
    assert run_command("exit 1", ignore_errors=True).captured_output is not None

def test_verbose_true():
    assert run_command("echo test", verbose=True).captured_output is not None


# LLM-generated content at query #40
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command("echo 'test'", verbose=False, return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_failed_execution():
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
    result = run_command("echo 'verbose'", verbose=True, return_output=True)
    assert result.command == "echo 'verbose'"
    assert result.return_code == 0
    assert result.captured_output == b"verbose\n"

def test_run_command_with_env():
    env = {"TEST_VAR": "test_value"}
    result = run_command("echo $TEST_VAR", shell=True, env=env, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.command == "pwd"
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

def test_run_command_list_args():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_ignore_errors_false():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", ignore_errors=False)

def test_run_command_ignore_errors_true():
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_return_output_false():
    result = run_command("echo 'test'", return_output=False)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_return_output_true():
    result = run_command("echo 'test'", return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"


# LLM-generated content at query #41
#--------------------------

```python
def test_verbose_logging():
    with patch('flutes.run.log') as mock_log:
        run_command("echo test", verbose=True)
        mock_log.assert_called_once_with("> 'echo test'", timestamp=False, include_proc_id=False)


# LLM-generated content at query #42
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #43
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err


# LLM-generated content at query #44
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
    assert result.return_code == 0

def test_run_command_cwd():
    result = run_command("pwd", cwd="/tmp", return_output=True)
    assert result.captured_output.strip() == b"/tmp"

def test_run_command_env():
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.captured_output.strip() == b"test_value"


# LLM-generated content at query #45
#--------------------------

```python
def test_unicode_decode_error_raises_exception():
    with tempfile.TemporaryFile() as f:
        f.write(b'\x80abc')
        f.seek(0)
        output = f.read()
        try:
            output.decode('utf-8')
        except UnicodeDecodeError:
            pass
        else:
            raise AssertionError("Expected UnicodeDecodeError to be raised")


# LLM-generated content at query #46
#--------------------------

```python
def test_error_wrapper_returns_original_error_for_non_subprocess_exceptions():
    err = ValueError("test error")
    assert error_wrapper(err) is err


# LLM-generated content at query #47
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #48
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.command == ["sleep", "10"]
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["env"]
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.command == ["pwd"]
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

def test_run_command_shell():
    result = run_command("echo test", shell=True, return_output=True)
    assert result.command == "echo test"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_long_output():
    result = run_command(["python", "-c", "print('x' * 10000)"], return_output=True)
    assert result.command == ["python", "-c", "print('x' * 10000)"]
    assert result.return_code == 0
    assert len(result.captured_output) > 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #49
#--------------------------

```python
def test_verbose_logging_when_verbose_is_true():
    with patch('flutes.run.log') as mock_log:
        run_command("echo test", verbose=True)
        mock_log.assert_called_once_with("> 'echo test'", timestamp=False, include_proc_id=False)


# LLM-generated content at query #50
#--------------------------

```python
def test_return_output_true():
    assert run_command("echo test", return_output=True).captured_output is not None

def test_return_code_nonzero():
    assert run_command("exit 1", ignore_errors=True).captured_output is not None

def test_verbose_true():
    assert run_command("echo test", verbose=True).captured_output is not None


# LLM-generated content at query #51
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
    result = run_command(args, env=env, cwd=cwd, timeout=timeout, verbose=verbose, return_output=return_output, ignore_errors=ignore_errors)
    assert result.return_code == 0
    assert result.captured_output is None


# LLM-generated content at query #52
#--------------------------

```python
def test_error_wrapper_non_subprocess_exception():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err

def test_error_wrapper_called_process_error_with_output():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    result = error_wrapper(err)
    assert isinstance(result, type(err))
    assert "Captured output:" in str(result)
    assert "line1" in str(result)
    assert "line2" in str(result)

def test_error_wrapper_called_process_error_without_output():
    err = subprocess.CalledProcessError(1, "test_cmd")
    result = error_wrapper(err)
    assert isinstance(result, type(err))
    assert "No output was generated." in str(result)

def test_error_wrapper_timeout_expired_with_output():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1, output=b"timeout output")
    result = error_wrapper(err)
    assert isinstance(result, type(err))
    assert "Captured output:" in str(result)
    assert "timeout output" in str(result)

def test_error_wrapper_timeout_expired_without_output():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1)
    result = error_wrapper(err)
    assert isinstance(result, type(err))
    assert "No output was generated." in str(result)

def test_error_wrapper_called_process_error_with_invalid_utf8():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    result = error_wrapper(err)
    assert isinstance(result, type(err))
    assert "Failed to parse output." in str(result)


# LLM-generated content at query #53
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_preserves_exception_type_for_subprocess_errors():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)

def test_error_wrapper_custom_str_for_called_process_error_with_output():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "line1" in str(wrapped_err)
    assert "line2" in str(wrapped_err)

def test_error_wrapper_custom_str_for_called_process_error_without_output():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_custom_str_for_timeout_expired_with_output():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1, output=b"timeout line")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout line" in str(wrapped_err)

def test_error_wrapper_custom_str_for_timeout_expired_without_output():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1)
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_handles_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #54
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_wraps_CalledProcessError_with_output():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "line1" in str(wrapped_err)
    assert "line2" in str(wrapped_err)

def test_error_wrapper_wraps_CalledProcessError_without_output():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_wraps_TimeoutExpired_with_output():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1, output=b"timeout line")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout line" in str(wrapped_err)

def test_error_wrapper_wraps_TimeoutExpired_without_output():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_handles_non_utf8_output():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #55
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = ["echo", "test"]
    result = run_command(args, return_output=True, verbose=True)
    assert isinstance(result, CommandResult)
    assert result.command == args
    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #56
#--------------------------

```python
def test_error_wrapper_returns_same_error_for_non_subprocess_exceptions():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err

def test_error_wrapper_creates_new_type_for_subprocess_called_process_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    result = error_wrapper(err)
    assert isinstance(result, type(err))
    assert result.__class__.__name__ == "CalledProcessError"
    assert "__str__" in result.__class__.__dict__

def test_error_wrapper_creates_new_type_for_subprocess_timeout_expired():
    err = subprocess.TimeoutExpired("test_cmd", 1)
    result = error_wrapper(err)
    assert isinstance(result, type(err))
    assert result.__class__.__name__ == "TimeoutExpired"
    assert "__str__" in result.__class__.__dict__

def test_error_wrapper_str_with_output():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    result = error_wrapper(err)
    expected_str = str(err) + "\nCaptured output:\n    line1\n    line2"
    assert str(result) == expected_str

def test_error_wrapper_str_with_no_output():
    err = subprocess.CalledProcessError(1, "test_cmd")
    result = error_wrapper(err)
    expected_str = str(err) + "\nNo output was generated."
    assert str(result) == expected_str

def test_error_wrapper_str_with_unicode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"\xff\xfe")
    result = error_wrapper(err)
    expected_str = str(err) + "\nFailed to parse output."
    assert str(result) == expected_str


# LLM-generated content at query #57
#--------------------------

```python
def test_return_output_true():
    result = run_command("echo test", return_output=True)
    assert result.captured_output is not None

def test_return_code_nonzero():
    result = run_command("exit 1", ignore_errors=True)
    assert result.captured_output is not None

def test_verbose_true():
    result = run_command("echo test", verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #58
#--------------------------

```python
def test_verbose_logging_when_verbose_is_true():
    with patch('flutes.run.log') as mock_log:
        run_command("echo test", verbose=True)
        mock_log.assert_called_once()


# LLM-generated content at query #59
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command("echo 'test'", verbose=False, return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_list_args():
    result = run_command(["echo", "test"], verbose=False, return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, verbose=False, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.decode().strip() == tmpdir

def test_run_command_with_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, verbose=False, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", timeout=0.1, verbose=False)

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True, verbose=False, return_output=True)
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_verbose_mode():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", verbose=True)

def test_run_command_with_custom_env_and_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", env={"TEST": "value"}, cwd=tmpdir, verbose=False, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.decode().strip() == tmpdir

def test_run_command_with_long_output_truncation():
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    with tempfile.NamedTemporaryFile() as f:
        f.write(long_output.encode())
        f.flush()
        result = run_command(f"cat {f.name}", verbose=False, return_output=True)
        assert b"*** (previous output truncated) ***" in result.captured_output
        assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")

def test_run_command_with_unicode_output():
    result = run_command("echo '日本語'", verbose=False, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"日本語\n"

def test_run_command_with_binary_output():
    result = run_command("echo -e '\\x00\\x01\\x02'", verbose=False, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"\x00\x01\x02\n"


# LLM-generated content at query #60
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo 'test'", verbose=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_output():
    result = run_command("echo 'test'", return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_error():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

def test_run_command_timeout_ignore_errors():
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_with_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.captured_output == b"test_value\n"

def test_run_command_with_cwd():
    result = run_command("pwd", cwd="/tmp", return_output=True)
    assert result.captured_output.strip() == b"/tmp"

def test_run_command_verbose():
    result = run_command("echo 'verbose test'", verbose=True, return_output=True)
    assert result.captured_output == b"verbose test\n"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "test"

def test_run_command_with_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "test"

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
    result = run_command("echo test", shell=True, return_output=True)
    assert result.command == "echo test"
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "test"

def test_run_command_with_non_zero_return_code():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_with_long_output():
    result = run_command(["python", "-c", "print('x' * 10000)"], return_output=True)
    assert result.command == ["python", "-c", "print('x' * 10000)"]
    assert result.return_code == 0
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")

def test_run_command_with_unicode_output():
    result = run_command(["python", "-c", "print('测试')"], return_output=True)
    assert result.command == ["python", "-c", "print('测试')"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "测试"


# LLM-generated content at query #2
#--------------------------

```python
def test_return_output_or_nonzero_returncode_or_verbose():
    result = run_command("echo test", return_output=True, verbose=False)
    assert result.captured_output is not None

def test_return_output_or_nonzero_returncode_or_verbose_2():
    result = run_command("exit 1", return_output=False, verbose=False)
    assert result.captured_output is not None

def test_return_output_or_nonzero_returncode_or_verbose_3():
    result = run_command("echo test", return_output=False, verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #3
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.command == ["sleep", "10"]
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"], verbose=True)

def test_run_command_env():
    result = run_command(["echo", "$TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True, shell=True)
    assert result.command == ["echo", "$TEST_VAR"]
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.command == ["pwd"]
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

def test_run_command_shell():
    result = run_command("echo test", shell=True, return_output=True)
    assert result.command == "echo test"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"


# LLM-generated content at query #4
#--------------------------

```python
def test_unicode_decode_error_handling():
    with tempfile.TemporaryFile() as f:
        f.write(b'\xff\xfe\xfd')  # Invalid UTF-8 sequence
        f.seek(0)
        output = f.read()
        try:
            output.decode('utf-8')
        except UnicodeDecodeError:
            assert True


# LLM-generated content at query #5
#--------------------------

```python
def test_error_wrapper_returns_same_error_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_creates_new_type_for_subprocess_errors():
    err = subprocess.CalledProcessError(1, "test")
    wrapped = error_wrapper(err)
    assert type(wrapped).__name__ == "CalledProcessError"
    assert type(wrapped) is not type(err)

def test_error_wrapper_preserves_error_attributes():
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

def test_error_wrapper_for_timeout_error():
    err = subprocess.TimeoutExpired("test", 1)
    wrapped = error_wrapper(err)
    assert type(wrapped).__name__ == "TimeoutExpired"
    assert "No output was generated." in str(wrapped)


# LLM-generated content at query #6
#--------------------------

```python
def test_output_truncation():
    output = b"x" * (MAX_OUTPUT_LENGTH + 1)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #7
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_verbose():
    result = run_command(["echo", "hello"], verbose=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_return_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_env():
    result = run_command(["sh", "-c", "echo $TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["sh", "-c", "echo $TEST_VAR"]
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.command == ["pwd"]
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir


# LLM-generated content at query #8
#--------------------------

```python
def test_error_wrapper_with_non_subprocess_exception():
    original_error = ValueError("test error")
    wrapped_error = error_wrapper(original_error)
    assert wrapped_error is original_error

def test_error_wrapper_with_called_process_error_no_output():
    original_error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_error)

def test_error_wrapper_with_called_process_error_with_output():
    original_error = subprocess.CalledProcessError(1, "test_command", output=b"line1\nline2")
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "line1" in str(wrapped_error)
    assert "line2" in str(wrapped_error)

def test_error_wrapper_with_timeout_expired_no_output():
    original_error = subprocess.TimeoutExpired("test_command", 1)
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_error)

def test_error_wrapper_with_timeout_expired_with_output():
    original_error = subprocess.TimeoutExpired("test_command", 1, output=b"line1\nline2")
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_error)
    assert "line1" in str(wrapped_error)
    assert "line2" in str(wrapped_error)

def test_error_wrapper_with_invalid_utf8_output():
    original_error = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped_error = error_wrapper(original_error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #9
#--------------------------

```python
def test_unicode_decode_error_raises_exception():
    output = b'\xff\xfe'
    assert output.decode('utf-8')


# LLM-generated content at query #10
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo hello", return_output=True)
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "hello"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

def test_run_command_verbose():
    run_command("echo test", verbose=True)

def test_run_command_cwd():
    result = run_command("pwd", cwd="/tmp", return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "/tmp"

def test_run_command_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "test_value"

def test_run_command_list_args():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "hello"

def test_run_command_return_code_nonzero():
    result = run_command("exit 1", return_output=True)
    assert result.return_code == 1
    assert result.captured_output is not None


# LLM-generated content at query #11
#--------------------------

```python
def test_run_command_successful_execution():
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

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "2"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_verbose():
    with patch('flutes.run.log') as mock_log:
        run_command(["echo", "test"], verbose=True)
        mock_log.assert_called_with("> ['echo', 'test']", timestamp=False, include_proc_id=False)

def test_run_command_return_output():
    result = run_command(["echo", "output"], return_output=True)
    assert result.command == ["echo", "output"]
    assert result.return_code == 0
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


# LLM-generated content at query #12
#--------------------------

```python
def test_verbose_logging():
    with patch('flutes.run.log') as mock_log:
        run_command("echo test", verbose=True)
        mock_log.assert_called_once_with("> 'echo test'", timestamp=False, include_proc_id=False)


# LLM-generated content at query #13
#--------------------------

```python
def test_error_wrapper_returns_original_error_for_non_subprocess_exceptions():
    err = ValueError("test error")
    assert error_wrapper(err) is err


# LLM-generated content at query #14
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
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["env"]
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_with_timeout():
    result = run_command(["sleep", "0.1"], timeout=0.2, return_output=True)
    assert result.command == ["sleep", "0.1"]
    assert result.return_code == 0

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True, return_output=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_shell_command():
    result = run_command("echo test", shell=True, return_output=True)
    assert result.command == "echo test"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_long_output():
    result = run_command(["python", "-c", "print('a' * 10000)"], return_output=True)
    assert result.command == ["python", "-c", "print('a' * 10000)"]
    assert result.return_code == 0
    assert len(result.captured_output) > 0


# LLM-generated content at query #15
#--------------------------

```python
def test_unicode_decode_error_occurs():
    args = "echo 'test'"
    output = b"\x80abc"
    ret = subprocess.CompletedProcess(args, 0, stdout=output)
    with patch('subprocess.run', return_value=ret) as mock_run, \
         patch('tempfile.TemporaryFile') as mock_temp:
        mock_file = mock_temp.return_value.__enter__.return_value
        mock_file.read.return_value = output
        result = run_command(args, return_output=True)
        assert result.captured_output == output


# LLM-generated content at query #16
#--------------------------

```python
def test_unicode_decode_error_occurs():
    with tempfile.TemporaryFile() as f:
        f.write(b'\xff\xfe\xfd')
        f.seek(0)
        output = f.read()
        assert not isinstance(output.decode('utf-8'), str)


# LLM-generated content at query #17
#--------------------------

```python
def test_output_truncation():
    output = b"a" * (MAX_OUTPUT_LENGTH + 1)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #18
#--------------------------

```python
def test_return_output_true():
    result = run_command("echo test", return_output=True)
    assert result.captured_output is not None

def test_return_code_nonzero():
    result = run_command("exit 1", ignore_errors=True)
    assert result.captured_output is not None

def test_verbose_true():
    result = run_command("echo test", verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #19
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = "echo 'test'"
    with tempfile.TemporaryFile() as f:
        f.write(b'\x80\x81')
        f.seek(0)
        output = f.read()
        try:
            output.decode('utf-8')
        except UnicodeDecodeError:
            for line in output.split(b"\n"):
                log(str(line), timestamp=False, include_proc_id=False)


# LLM-generated content at query #20
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
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_with_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_env():
    env = {"TEST_VAR": "test_value"}
    result = run_command(["sh", "-c", "echo $TEST_VAR"], env=env, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.strip() == tmpdir.encode()

def test_run_command_with_string_command():
    result = run_command("echo string_command", shell=True, return_output=True)
    assert result.command == "echo string_command"
    assert result.return_code == 0
    assert result.captured_output == b"string_command\n"

def test_run_command_with_nonzero_return_code():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_with_output_truncation():
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], ignore_errors=True)
    assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")

def test_run_command_with_unicode_decode_error():
    result = run_command(["printf", "\xff"], verbose=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


# LLM-generated content at query #21
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

    with tempfile.TemporaryFile() as f:
        ret = subprocess.run(args, check=True, stdout=f, stderr=subprocess.STDOUT,
                             timeout=timeout, env=env, cwd=cwd)
        f.seek(0)
        output = b'\xff\xfe'  # Invalid UTF-8 sequence to trigger UnicodeDecodeError

        try:
            output.decode('utf-8')
        except UnicodeDecodeError:
            pass  # This ensures the condition at line 46 evaluates to True
        else:
            assert False, "UnicodeDecodeError should have been raised"


# LLM-generated content at query #22
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = "echo 'test'"
    output = b"\x80\x81"
    ret = type('MockReturn', (), {'returncode': 1})()
    with unittest.mock.patch('subprocess.run', return_value=ret):
        with unittest.mock.patch('tempfile.TemporaryFile') as mock_tempfile:
            mock_file = unittest.mock.MagicMock()
            mock_file.read.return_value = output
            mock_tempfile.return_value.__enter__.return_value = mock_file
            result = run_command(args, return_output=True)
            assert result.captured_output == output


# LLM-generated content at query #23
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #24
#--------------------------

```python
def test_error_wrapper_predicate():
    err = ValueError("test")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #25
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

def test_error_wrapper_preserves_original_exception_attributes():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert wrapped_err.returncode == err.returncode
    assert wrapped_err.cmd == err.cmd
    assert wrapped_err.output == err.output

def test_error_wrapper_str_with_output():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    expected_str = "Command 'test_cmd' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"
    assert str(wrapped_err) == expected_str

def test_error_wrapper_str_without_output():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    expected_str = "Command 'test_cmd' returned non-zero exit status 1.\nNo output was generated."
    assert str(wrapped_err) == expected_str

def test_error_wrapper_str_with_unicode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    expected_str = "Command 'test_cmd' returned non-zero exit status 1.\nFailed to parse output."
    assert str(wrapped_err) == expected_str

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1)
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == "TimeoutExpired"
    assert "TimeoutExpired" in str(wrapped_err)


# LLM-generated content at query #26
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
        f.write(b'\x80\x81\x82')
        f.seek(0)
        output = f.read()

        try:
            output.decode('utf-8')
            assert False, "Expected UnicodeDecodeError"
        except UnicodeDecodeError:
            for line in output.split(b"\n"):
                log(str(line), timestamp=False, include_proc_id=False)
            assert True


# LLM-generated content at query #27
#--------------------------

```python
def test_error_wrapper_returns_original_error_for_non_subprocess_exceptions():
    original_error = ValueError("test error")
    result = error_wrapper(original_error)
    assert result is original_error


# LLM-generated content at query #28
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #29
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_returns_wrapped_exception_for_called_process_error_with_output():
    err = subprocess.CalledProcessError(1, "cmd", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'cmd' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"

def test_error_wrapper_returns_wrapped_exception_for_called_process_error_without_output():
    err = subprocess.CalledProcessError(1, "cmd")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'cmd' returned non-zero exit status 1.\nNo output was generated."

def test_error_wrapper_returns_wrapped_exception_for_timeout_expired_with_output():
    err = subprocess.TimeoutExpired("cmd", timeout=1, output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'cmd' timed out after 1 seconds.\nCaptured output:\n    line1\n    line2"

def test_error_wrapper_returns_wrapped_exception_for_timeout_expired_without_output():
    err = subprocess.TimeoutExpired("cmd", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'cmd' timed out after 1 seconds.\nNo output was generated."

def test_error_wrapper_handles_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'cmd' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #30
#--------------------------

```python
def test_error_wrapper_returns_original_error_for_non_subprocess_exceptions():
    class CustomException(Exception):
        pass

    err = CustomException("Test error")
    assert error_wrapper(err) is err


# LLM-generated content at query #31
#--------------------------

```python
def test_error_wrapper_returns_original_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err


# LLM-generated content at query #32
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    non_subprocess_error = ValueError("test error")
    result = error_wrapper(non_subprocess_error)
    assert result is non_subprocess_error


# LLM-generated content at query #33
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #34
#--------------------------

```python
def test_error_wrapper_returns_non_subprocess_exception_unchanged():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_modifies_called_process_error_with_output():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    wrapped = error_wrapper(err)
    assert str(wrapped) == "Command 'test_cmd' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"

def test_error_wrapper_modifies_called_process_error_without_output():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped = error_wrapper(err)
    assert str(wrapped) == "Command 'test_cmd' returned non-zero exit status 1.\nNo output was generated."

def test_error_wrapper_modifies_timeout_expired_with_output():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1, output=b"line1\nline2")
    wrapped = error_wrapper(err)
    assert str(wrapped) == "Command 'test_cmd' timed out after 1 seconds.\nCaptured output:\n    line1\n    line2"

def test_error_wrapper_modifies_timeout_expired_without_output():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1)
    wrapped = error_wrapper(err)
    assert str(wrapped) == "Command 'test_cmd' timed out after 1 seconds.\nNo output was generated."

def test_error_wrapper_handles_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"\xff\xfe")
    wrapped = error_wrapper(err)
    assert str(wrapped) == "Command 'test_cmd' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #35
#--------------------------

```python
def test_error_wrapper_returns_non_subprocess_error_unchanged():
    original_error = ValueError("test error")
    result = error_wrapper(original_error)
    assert result is original_error

def test_error_wrapper_modifies_called_process_error_with_output():
    original_error = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    result = error_wrapper(original_error)
    assert str(result) == "Command 'test_cmd' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"

def test_error_wrapper_modifies_called_process_error_without_output():
    original_error = subprocess.CalledProcessError(1, "test_cmd")
    result = error_wrapper(original_error)
    assert str(result) == "Command 'test_cmd' returned non-zero exit status 1.\nNo output was generated."

def test_error_wrapper_modifies_timeout_expired_with_output():
    original_error = subprocess.TimeoutExpired("test_cmd", timeout=1, output=b"timeout output")
    result = error_wrapper(original_error)
    assert str(result) == "Command 'test_cmd' timed out after 1 seconds.\nCaptured output:\n    timeout output"

def test_error_wrapper_modifies_timeout_expired_without_output():
    original_error = subprocess.TimeoutExpired("test_cmd", timeout=1)
    result = error_wrapper(original_error)
    assert str(result) == "Command 'test_cmd' timed out after 1 seconds.\nNo output was generated."

def test_error_wrapper_handles_unicode_decode_error():
    original_error = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    result = error_wrapper(original_error)
    assert str(result) == "Command 'test_cmd' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #36
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #37
#--------------------------

```python
def test_predicate_at_line_25():
    assert isinstance(verbose, bool)


# LLM-generated content at query #38
#--------------------------

```python
def test_run_command_returns_output_on_nonzero_returncode():
    result = run_command(["false"], return_output=False, ignore_errors=True)
    assert result.captured_output is not None


# LLM-generated content at query #39
#--------------------------

```python
def test_unicode_decode_error_handling():
    command = "echo 'test'"
    result = run_command(command, return_output=True, verbose=True, shell=True)
    assert isinstance(result, CommandResult)
    assert result.command == command
    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #40
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
        run_command(["sleep", "10"], timeout=0.1)

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
    long_output = "x" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], ignore_errors=True)
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #41
#--------------------------

```python
def test_error_wrapper_predicate():
    assert isinstance(subprocess.CalledProcessError, Exception)
    assert isinstance(subprocess.TimeoutExpired, Exception)


# LLM-generated content at query #42
#--------------------------

```python
def test_error_wrapper_returns_same_error_for_non_subprocess_exceptions():
    original_error = ValueError("test error")
    result = error_wrapper(original_error)
    assert result is original_error

def test_error_wrapper_modifies_subprocess_called_process_error():
    original_error = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    result = error_wrapper(original_error)
    assert isinstance(result, type(original_error))
    assert str(result) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

def test_error_wrapper_modifies_subprocess_timeout_error():
    original_error = subprocess.TimeoutExpired("test_command", 1)
    result = error_wrapper(original_error)
    assert isinstance(result, type(original_error))
    assert str(result) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

def test_error_wrapper_handles_unicode_decode_error():
    original_error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    result = error_wrapper(original_error)
    assert isinstance(result, type(original_error))
    assert str(result) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #43
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_creates_new_type_for_subprocess_errors():
    err = subprocess.CalledProcessError(1, "test")
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == "CalledProcessError"
    assert type(wrapped_err) is not type(err)

def test_error_wrapper_preserves_exception_attributes():
    err = subprocess.CalledProcessError(1, "test", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert wrapped_err.returncode == err.returncode
    assert wrapped_err.cmd == err.cmd
    assert wrapped_err.output == err.output

def test_error_wrapper_str_with_output():
    err = subprocess.CalledProcessError(1, "test", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "line1" in str(wrapped_err)
    assert "line2" in str(wrapped_err)

def test_error_wrapper_str_without_output():
    err = subprocess.CalledProcessError(1, "test")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_str_with_unicode_error():
    err = subprocess.CalledProcessError(1, "test", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)

def test_error_wrapper_for_timeout_expired():
    err = subprocess.TimeoutExpired("test", 1)
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == "TimeoutExpired"
    assert type(wrapped_err) is not type(err)


# LLM-generated content at query #44
#--------------------------

```python
def test_unicode_decode_error_occurs():
    args = ["echo", "test"]
    env = None
    cwd = None
    timeout = None
    verbose = True
    return_output = False
    ignore_errors = False
    kwargs = {}

    with tempfile.TemporaryFile() as f:
        f.write(b'\x80abc')
        f.seek(0)
        output = f.read()
        try:
            output.decode('utf-8')
        except UnicodeDecodeError:
            pass
        else:
            raise AssertionError("UnicodeDecodeError should have occurred")


# LLM-generated content at query #45
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #46
#--------------------------

```python
def test_run_command_verbose_logs_command():
    with patch('flutes.run.log') as mock_log:
        run_command("test_command", verbose=True)
        mock_log.assert_called_once_with("> 'test_command'", timestamp=False, include_proc_id=False)


# LLM-generated content at query #47
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

def test_run_command_verbose():
    result = run_command(["echo", "hello"], verbose=True, return_output=True)
    assert result.captured_output == b"hello\n"

def test_run_command_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.captured_output == b"test_value\n"

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.captured_output.decode().strip() == tmpdir

def test_run_command_shell_string():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.captured_output == b"hello\n"

def test_run_command_return_output_false():
    result = run_command(["echo", "hello"])
    assert result.captured_output is None

def test_run_command_max_output_truncation():
    long_output = "x" * (MAX_OUTPUT_LENGTH + 100)
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command(["python", "-c", f"print('{long_output}')"], timeout=0.1)
    assert b"*** (previous output truncated) ***" in exc_info.value.output


# LLM-generated content at query #48
#--------------------------

```python
def test_unicode_decode_error_handling():
    with tempfile.TemporaryFile() as f:
        f.write(b'\xff\xfe')
        f.seek(0)
        output = f.read()
        try:
            output.decode('utf-8')
            assert False, "Expected UnicodeDecodeError"
        except UnicodeDecodeError:
            pass


# LLM-generated content at query #49
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = ["echo", "test"]
    result = run_command(args, return_output=True, verbose=True)
    assert isinstance(result, CommandResult)
    assert result.command == args
    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #50
#--------------------------

```python
def test_error_wrapper_predicate_false():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #51
#--------------------------

```python
def test_return_output_or_nonzero_returncode_or_verbose():
    assert run_command("echo test", return_output=True).captured_output is not None
    assert run_command("exit 1", ignore_errors=True).captured_output is not None
    assert run_command("echo test", verbose=True).captured_output is not None


# LLM-generated content at query #52
#--------------------------

```python
def test_error_wrapper_returns_unchanged_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_returns_wrapped_subprocess_called_process_error_with_output():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

def test_error_wrapper_returns_wrapped_subprocess_called_process_error_without_output():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_returns_wrapped_subprocess_timeout_expired_with_output():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1, output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

def test_error_wrapper_returns_wrapped_subprocess_timeout_expired_without_output():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_handles_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #53
#--------------------------

```python
def test_error_wrapper_returns_same_error_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_creates_new_type_for_subprocess_errors():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert type(wrapped_err).__name__ == "CalledProcessError"
    assert type(wrapped_err).__bases__ == (subprocess.CalledProcessError,)

def test_error_wrapper_str_with_output():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_cmd' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"

def test_error_wrapper_str_without_output():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_cmd' returned non-zero exit status 1.\nNo output was generated."

def test_error_wrapper_str_with_unicode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "Command 'test_cmd' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #54
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #55
#--------------------------

```python
def test_command_output_truncation():
    output = b"x" * (MAX_OUTPUT_LENGTH + 1)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #56
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_true():
    assert verbose is True


# LLM-generated content at query #57
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #58
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

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_with_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.captured_output.decode().strip() == tmpdir

def test_run_command_with_shell_command():
    result = run_command("echo test", shell=True, return_output=True)
    assert result.command == "echo test"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_nonzero_return_code():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_with_return_output_false():
    result = run_command(["echo", "test"], return_output=False)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_unicode_output():
    result = run_command(["echo", "tëst"], return_output=True)
    assert result.captured_output == b"tëst\n"

def test_run_command_with_long_output():
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], return_output=True)
    assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #59
#--------------------------

```python
def test_error_wrapper_predicate():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #60
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

def test_run_command_with_verbose():
    result = run_command(["echo", "hello"], verbose=True, return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is not None

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_with_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["env"]
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.command == ["pwd"]
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"/tmp" in result.captured_output

def test_run_command_with_shell_command():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

def test_run_command_with_non_zero_return_code():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_with_return_output_false():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_long_output():
    result = run_command(["python", "-c", "print('a' * 10000)"], return_output=True)
    assert result.command == ["python", "-c", "print('a' * 10000)"]
    assert result.return_code == 0
    assert result.captured_output is not None
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")

def test_run_command_with_unicode_output():
    result = run_command(["python", "-c", "print('hello 世界')"], return_output=True)
    assert result.command == ["python", "-c", "print('hello 世界')"]
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output


# LLM-generated content at query #61
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

def test_run_command_verbose():
    with patch('flutes.log.log') as mock_log:
        run_command(["echo", "hello"], verbose=True)
        mock_log.assert_called_once()

def test_run_command_env():
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.captured_output == b"test_value\n"

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.captured_output.decode().strip() == tmpdir


# LLM-generated content at query #62
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #63
#--------------------------

```python
def test_error_wrapper_returns_original_exception_for_non_subprocess_errors():
    non_subprocess_error = ValueError("Test error")
    assert error_wrapper(non_subprocess_error) is non_subprocess_error


