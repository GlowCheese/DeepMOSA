####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_command_success_no_output():
    result = run_command(["echo", "test"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
    assert result.command == ["echo", "test"]


def test_run_command_success_with_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output
    assert result.command == ["echo", "test"]


def test_run_command_error_with_exception():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None
        assert "Captured output:" in str(e)


def test_run_command_error_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    assert result.command == ["false"]


def test_run_command_timeout_with_exception():
    try:
        run_command(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.timeout == 0.1
        assert e.output is not None
        assert "Captured output:" in str(e)


def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None
    assert result.command == ["sleep", "10"]


def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output


def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output


def test_run_command_shell_true():
    result = run_command("echo test", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output
    assert result.command == "echo test"


def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output


def test_run_command_nonzero_return_with_output():
    result = run_command(["bash", "-c", "echo error && exit 1"], return_output=True)
    assert result.return_code == 1
    assert b"error" in result.captured_output


# LLM-generated content at query #2
#--------------------------

```python
def test_run_command_exception_captures_output():
    result = run_command(["python", "-c", "import sys; sys.exit(1)"], ignore_errors=True)
    assert result.captured_output is not None
    assert result.return_code == 1
    assert isinstance(result.captured_output, bytes)


# LLM-generated content at query #3
#--------------------------

```python
def test_run_command_return_output_true():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output is not None

def test_run_command_return_code_nonzero():
    result = run_command(["false"], ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true():
    result = run_command(["echo", "test"], verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_verbose():
    result = run_command(["echo", "test"], return_output=True, verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_nonzero():
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true_and_nonzero():
    result = run_command(["false"], verbose=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_all_conditions_true():
    result = run_command(["false"], return_output=True, verbose=True, ignore_errors=True)
    assert result.captured_output is not None


# LLM-generated content at query #4
#--------------------------

```python
def test_run_command_verbose_logs_without_timestamp_and_proc_id():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command
    from flutes.log import log

    with patch('subprocess.run') as mock_run, patch('flutes.run.log') as mock_log:
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        args = ['echo', 'test']
        result = run_command(args, verbose=True)
        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert call_args[1]['timestamp'] == False
        assert call_args[1]['include_proc_id'] == False
        assert isinstance(result.command, list)
        assert result.return_code == 0
        assert result.captured_output is None


# LLM-generated content at query #5
#--------------------------

def test_error_wrapper_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_called_process_error_no_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial' in result

def test_error_wrapper_timeout_expired_no_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_other_exception():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_output_decoding_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


# LLM-generated content at query #6
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_error_ignore():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


def test_run_command_error_raise():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None


def test_run_command_timeout_ignore():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_timeout_raise():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None


def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None


def test_run_command_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output


def test_run_command_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output


def test_run_command_shell():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


# LLM-generated content at query #7
#--------------------------

```python
def test_run_command_return_output_true():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output is not None

def test_run_command_return_code_nonzero():
    result = run_command(["false"], ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true():
    result = run_command(["echo", "test"], verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_verbose():
    result = run_command(["echo", "test"], return_output=True, verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_nonzero():
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true_and_nonzero():
    result = run_command(["false"], verbose=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_all_conditions_true():
    result = run_command(["false"], return_output=True, verbose=True, ignore_errors=True)
    assert result.captured_output is not None


# LLM-generated content at query #8
#--------------------------

```python
def test_ignore_errors_true_returns_command_result():
    result = run_command(["false"], ignore_errors=True)
    assert isinstance(result, CommandResult)
    assert result.return_code != 0
    assert result.captured_output is not None


# LLM-generated content at query #9
#--------------------------

```python
def test_run_command_verbose_logging_without_timestamp_and_proc_id():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command
    from flutes.log import log

    with patch('flutes.run.log') as mock_log:
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            run_command(["echo", "test"], verbose=True)
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[1]['timestamp'] == False
            assert call_args[1]['include_proc_id'] == False


# LLM-generated content at query #10
#--------------------------

def test_error_wrapper_wraps_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'])
    err.output = b'output line 1\noutput line 2'
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert wrapped.__class__.__name__ == 'CalledProcessError'
    assert '__str__' in wrapped.__class__.__dict__

def test_error_wrapper_wraps_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1)
    err.output = b'timeout output'
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert wrapped.__class__.__name__ == 'TimeoutExpired'
    assert '__str__' in wrapped.__class__.__dict__

def test_error_wrapper_returns_other_exceptions_unchanged():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, ValueError)

def test_error_wrapper_str_includes_captured_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'])
    err.output = b'output line 1\noutput line 2'
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'Captured output:' in str_repr
    assert '    output line 1' in str_repr
    assert '    output line 2' in str_repr

def test_error_wrapper_str_includes_no_output_message():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'])
    err.output = None
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'No output was generated.' in str_repr

def test_error_wrapper_str_handles_unicode_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'])
    err.output = b'\xff\xfe'
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'Failed to parse output.' in str_repr

def test_error_wrapper_preserves_original_attributes():
    import subprocess
    err = subprocess.CalledProcessError(returncode=5, cmd=['test', 'arg'])
    err.output = b'output'
    wrapped = error_wrapper(err)
    assert wrapped.returncode == 5
    assert wrapped.cmd == ['test', 'arg']
    assert wrapped.output == b'output'


# LLM-generated content at query #11
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output


def test_run_command_failure_raises():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None


def test_run_command_failure_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


def test_run_command_timeout_raises():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None


def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output


def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output


def test_run_command_with_cwd():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output


def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], ignore_errors=False, return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #12
#--------------------------

def test_error_wrapper_wraps_subprocess_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='ls')
    err.output = b'output line 1\noutput line 2'
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert wrapped.__class__.__name__ == 'CalledProcessError'
    assert '__str__' in wrapped.__class__.__dict__
    str_output = str(wrapped)
    assert 'Captured output:' in str_output
    assert '    output line 1' in str_output
    assert '    output line 2' in str_output

def test_error_wrapper_wraps_subprocess_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired(cmd='sleep 10', timeout=1)
    err.output = b'timeout output'
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert wrapped.__class__.__name__ == 'TimeoutExpired'
    assert '__str__' in wrapped.__class__.__dict__
    str_output = str(wrapped)
    assert 'Captured output:' in str_output
    assert '    timeout output' in str_output

def test_error_wrapper_handles_unicode_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='cmd')
    err.output = b'\xff\xfe'
    wrapped = error_wrapper(err)
    str_output = str(wrapped)
    assert 'Failed to parse output.' in str_output

def test_error_wrapper_returns_other_exceptions_unchanged():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, ValueError)

def test_error_wrapper_with_no_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='cmd')
    err.output = None
    wrapped = error_wrapper(err)
    str_output = str(wrapped)
    assert 'No output was generated.' in str_output

def test_error_wrapper_preserves_exception_attributes():
    import subprocess
    err = subprocess.CalledProcessError(returncode=5, cmd='test_cmd')
    err.output = b'some output'
    wrapped = error_wrapper(err)
    assert wrapped.returncode == 5
    assert wrapped.cmd == 'test_cmd'
    assert wrapped.output == b'some output'


# LLM-generated content at query #13
#--------------------------

```python
def test_run_command_verbose_logging_without_timestamp_and_proc_id():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command
    from flutes.log import log

    with patch('flutes.run.log') as mock_log:
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            run_command("echo test", verbose=True)
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[1]['timestamp'] == False
            assert call_args[1]['include_proc_id'] == False


# LLM-generated content at query #14
#--------------------------

def test_error_wrapper_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial' in result

def test_error_wrapper_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_other_exception():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_output_decoding_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'\xff')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


# LLM-generated content at query #15
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_failure_with_exception():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None


def test_run_command_failure_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


def test_run_command_timeout_exception():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None


def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output


def test_run_command_with_cwd():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output


def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert b"TEST_VAR=value" in result.captured_output


def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #16
#--------------------------

```python
def test_run_command_return_output_true():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output is not None

def test_run_command_return_code_nonzero():
    result = run_command(["false"], ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true():
    result = run_command(["echo", "test"], verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_verbose():
    result = run_command(["echo", "test"], return_output=True, verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_nonzero():
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true_and_nonzero():
    result = run_command(["false"], verbose=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_all_conditions_true():
    result = run_command(["false"], return_output=True, verbose=True, ignore_errors=True)
    assert result.captured_output is not None


# LLM-generated content at query #17
#--------------------------

def test_run_command_truncates_long_output_on_error():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command
    from flutes.run import MAX_OUTPUT_LENGTH
    mock_subprocess_run = MagicMock(side_effect=subprocess.CalledProcessError(1, 'cmd'))
    with patch('subprocess.run', mock_subprocess_run):
        with tempfile.TemporaryFile() as mock_file:
            long_output = b'a' * (MAX_OUTPUT_LENGTH + 100)
            mock_file.write(long_output)
            mock_file.seek(0)
            with patch('tempfile.TemporaryFile', return_value=mock_file):
                result = run_command('cmd', ignore_errors=True)
                assert result.captured_output is not None
                assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")
                assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #18
#--------------------------

def test_error_wrapper_subprocess_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_subprocess_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_subprocess_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial' in result

def test_error_wrapper_subprocess_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_non_subprocess_exception():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_output_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'\xff')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


# LLM-generated content at query #19
#--------------------------

```python
def test_run_command_verbose_with_non_utf8_output():
    import subprocess
    import tempfile
    from flutes.run import run_command
    from flutes.log import log
    import sys
    import io

    class MockLog:
        calls = []
        @staticmethod
        def log(msg, timestamp, include_proc_id):
            MockLog.calls.append((msg, timestamp, include_proc_id))

    original_log = log
    sys.modules['flutes.log'].log = MockLog.log
    MockLog.calls.clear()

    non_utf8_bytes = b'\xff\xfe\xfd\xfc'
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp:
        tmp.write(non_utf8_bytes)
        tmp.flush()
        cmd = [sys.executable, '-c', f'with open("{tmp.name}", "rb") as f: sys.stdout.buffer.write(f.read())']

    result = run_command(cmd, verbose=True, return_output=True)

    sys.modules['flutes.log'].log = original_log
    assert len(MockLog.calls) > 0
    for call in MockLog.calls:
        if isinstance(call[0], str) and repr(non_utf8_bytes) in call[0]:
            break
    else:
        assert False, "Expected log call with non-UTF-8 bytes representation"


# LLM-generated content at query #20
#--------------------------

def test_error_wrapper_wraps_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'])
    err.output = b'output line 1\noutput line 2'
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert wrapped.__class__.__name__ == 'CalledProcessError'
    assert '__str__' in wrapped.__class__.__dict__

def test_error_wrapper_wraps_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1)
    err.output = b'timeout output'
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert wrapped.__class__.__name__ == 'TimeoutExpired'
    assert '__str__' in wrapped.__class__.__dict__

def test_error_wrapper_returns_other_exceptions_unchanged():
    import subprocess
    err = ValueError('some error')
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, ValueError)

def test_error_wrapper_str_includes_captured_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'])
    err.output = b'output line 1\noutput line 2'
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'Captured output:' in str_repr
    assert '    output line 1' in str_repr
    assert '    output line 2' in str_repr

def test_error_wrapper_str_handles_unicode_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'])
    err.output = b'\xff\xfe'
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'Failed to parse output.' in str_repr

def test_error_wrapper_str_shows_no_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'])
    err.output = None
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'No output was generated.' in str_repr

def test_error_wrapper_preserves_original_attributes():
    import subprocess
    err = subprocess.CalledProcessError(returncode=5, cmd=['test', 'arg'])
    err.output = b'some output'
    wrapped = error_wrapper(err)
    assert wrapped.returncode == 5
    assert wrapped.cmd == ['test', 'arg']
    assert wrapped.output == b'some output'


# LLM-generated content at query #21
#--------------------------

def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    class CustomException(Exception):
        pass
    exc = CustomException("test")
    result = error_wrapper(exc)
    assert result is exc

def test_error_wrapper_wraps_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(1, ["cmd"], output=b"output")
    result = error_wrapper(err)
    assert isinstance(result, subprocess.CalledProcessError)
    assert result.__class__.__name__ == "CalledProcessError"
    str_rep = str(result)
    assert "Captured output:" in str_rep
    assert "    output" in str_rep

def test_error_wrapper_wraps_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(1, ["cmd"], output=None)
    result = error_wrapper(err)
    assert isinstance(result, subprocess.CalledProcessError)
    str_rep = str(result)
    assert "No output was generated." in str_rep

def test_error_wrapper_wraps_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(["cmd"], timeout=5, output=b"timeout output")
    result = error_wrapper(err)
    assert isinstance(result, subprocess.TimeoutExpired)
    assert result.__class__.__name__ == "TimeoutExpired"
    str_rep = str(result)
    assert "Captured output:" in str_rep
    assert "    timeout output" in str_rep

def test_error_wrapper_wraps_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(["cmd"], timeout=5, output=None)
    result = error_wrapper(err)
    assert isinstance(result, subprocess.TimeoutExpired)
    str_rep = str(result)
    assert "No output was generated." in str_rep

def test_error_wrapper_handles_unicode_decode_error_in_output():
    import subprocess
    err = subprocess.CalledProcessError(1, ["cmd"], output=b'\xff\xfe')
    result = error_wrapper(err)
    str_rep = str(result)
    assert "Failed to parse output." in str_rep

def test_error_wrapper_preserves_exception_attributes():
    import subprocess
    err = subprocess.CalledProcessError(returncode=42, cmd=["ls", "-la"], output=b"total 0")
    result = error_wrapper(err)
    assert result.returncode == 42
    assert result.cmd == ["ls", "-la"]
    assert result.output == b"total 0"

def test_error_wrapper_returns_original_exception_for_other_exception_types():
    class AnotherException(Exception):
        pass
    exc = AnotherException("another")
    result = error_wrapper(exc)
    assert result is exc
    assert isinstance(result, AnotherException)


# LLM-generated content at query #22
#--------------------------

def test_run_command_no_output_on_success():
    result = run_command(["echo", "test"], return_output=False)
    assert result.captured_output is None


# LLM-generated content at query #23
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

def test_run_command_failure_raises_exception():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None

def test_run_command_failure_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_raises_exception():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None

def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose_output():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #24
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_error_with_output():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_with_output():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

def test_run_command_with_cwd():
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_shell_false():
    result = run_command(["echo", "hello"], shell=False, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_output_truncation():
    long_output = "x" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], ignore_errors=True, return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #25
#--------------------------

def test_error_wrapper_returns_err_when_not_subprocess_error():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #26
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output


def test_run_command_error():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None


def test_run_command_with_cwd():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output


def test_run_command_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output


def test_run_command_shell():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


# LLM-generated content at query #27
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_error_ignore():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


def test_run_command_error_raise():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None


def test_run_command_timeout_ignore():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_timeout_raise():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None


def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None


def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output


def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output


def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_output_truncation():
    long_output = "a" * 10000
    result = run_command(["echo", long_output], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #28
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_error_no_ignore():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None


def test_run_command_error_ignore():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


def test_run_command_timeout_no_ignore():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None


def test_run_command_timeout_ignore():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output


def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output


def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output


def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_shell_false():
    result = run_command(["echo", "hello"], shell=False, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_output_truncation():
    long_output = "x" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


def test_run_command_nonzero_return_no_output():
    result = run_command(["false"], return_output=False)
    assert result.return_code != 0
    assert result.captured_output is not None


# LLM-generated content at query #29
#--------------------------

def test_error_wrapper_wraps_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(1, 'cmd')
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert wrapped.__class__.__name__ == 'CalledProcessError'
    assert hasattr(wrapped.__class__, '__str__')
    assert wrapped.__class__.__str__ != subprocess.CalledProcessError.__str__

def test_error_wrapper_wraps_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired('cmd', 10)
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert wrapped.__class__.__name__ == 'TimeoutExpired'
    assert hasattr(wrapped.__class__, '__str__')
    assert wrapped.__class__.__str__ != subprocess.TimeoutExpired.__str__

def test_error_wrapper_returns_other_exceptions_unchanged():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, ValueError)
    assert wrapped.__class__ is ValueError

def test_error_wrapper_predicate_at_line_3_true_for_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(1, 'cmd')
    result = isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
    assert result is True

def test_error_wrapper_predicate_at_line_3_true_for_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired('cmd', 10)
    result = isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
    assert result is True

def test_error_wrapper_predicate_at_line_3_false_for_other_exception():
    import subprocess
    err = KeyError('test')
    result = isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
    assert result is False


# LLM-generated content at query #30
#--------------------------

```python
def test_run_command_truncates_long_output_on_error():
    import subprocess
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command

    mock_subprocess_run = MagicMock(side_effect=subprocess.CalledProcessError(1, "test"))
    with patch("subprocess.run", mock_subprocess_run):
        with patch("tempfile.TemporaryFile") as mock_tempfile:
            mock_file = MagicMock()
            mock_tempfile.return_value.__enter__.return_value = mock_file
            mock_file.seek.return_value = None
            mock_file.read.return_value = b"x" * 10000

            result = run_command("test", ignore_errors=True)
            assert len(result.captured_output) < 10000
            assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #31
#--------------------------

def test_error_wrapper_called_process_error_with_output():
    import subprocess
    try:
        subprocess.run(['ls', 'nonexistent'], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        result = str(wrapped)
        assert "Captured output:" in result
        assert "No such file or directory" in result

def test_error_wrapper_called_process_error_no_output():
    import subprocess
    try:
        subprocess.run(['ls', 'nonexistent'], check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        result = str(wrapped)
        assert "No output was generated." in result

def test_error_wrapper_timeout_expired_with_output():
    import subprocess
    try:
        subprocess.run(['sleep', '2'], timeout=0.1, capture_output=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        result = str(wrapped)
        assert "Captured output:" in result

def test_error_wrapper_timeout_expired_no_output():
    import subprocess
    try:
        subprocess.run(['sleep', '2'], timeout=0.1, capture_output=False)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        result = str(wrapped)
        assert "No output was generated." in result

def test_error_wrapper_other_exception():
    import subprocess
    exc = ValueError("test error")
    wrapped = error_wrapper(exc)
    assert wrapped is exc
    assert str(wrapped) == "test error"

def test_error_wrapper_output_unicode_decode_error():
    import subprocess
    class MockCalledProcessError(subprocess.CalledProcessError):
        def __init__(self):
            self.returncode = 1
            self.cmd = ['cmd']
            self.output = b'\xff\xfe'
    err = MockCalledProcessError()
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert "Failed to parse output." in result

def test_error_wrapper_output_empty_bytes():
    import subprocess
    class MockCalledProcessError(subprocess.CalledProcessError):
        def __init__(self):
            self.returncode = 1
            self.cmd = ['cmd']
            self.output = b''
    err = MockCalledProcessError()
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert "No output was generated." in result

def test_error_wrapper_output_none():
    import subprocess
    class MockCalledProcessError(subprocess.CalledProcessError):
        def __init__(self):
            self.returncode = 1
            self.cmd = ['cmd']
            self.output = None
    err = MockCalledProcessError()
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert "No output was generated." in result

def test_error_wrapper_preserves_original_attributes():
    import subprocess
    try:
        subprocess.run(['ls', 'nonexistent'], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        original_returncode = e.returncode
        original_cmd = e.cmd
        original_output = e.output
        wrapped = error_wrapper(e)
        assert wrapped.returncode == original_returncode
        assert wrapped.cmd == original_cmd
        assert wrapped.output == original_output


# LLM-generated content at query #32
#--------------------------

```python
def test_run_command_output_truncation_when_exception_and_ignore_errors_false():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command

    mock_args = ["echo", "test"]
    mock_env = None
    mock_cwd = None
    mock_timeout = None
    mock_verbose = False
    mock_return_output = False
    mock_ignore_errors = False
    mock_kwargs = {}

    long_output = b"x" * 10000
    truncated_output = b"*** (previous output truncated) ***\n" + long_output[-8192:]

    with tempfile.TemporaryFile() as mock_file:
        mock_file.write(long_output)
        mock_file.seek(0)
        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, mock_args, output=long_output)):
            with patch('tempfile.TemporaryFile', return_value=mock_file):
                try:
                    run_command(mock_args, env=mock_env, cwd=mock_cwd, timeout=mock_timeout, verbose=mock_verbose, return_output=mock_return_output, ignore_errors=mock_ignore_errors, **mock_kwargs)
                except subprocess.CalledProcessError as e:
                    assert e.output == truncated_output


# LLM-generated content at query #33
#--------------------------

```python
def test_run_command_verbose_with_unicode_decode_error():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command
    from flutes.log import log

    with patch('subprocess.run') as mock_run, \
         patch('flutes.run.log') as mock_log, \
         patch('tempfile.TemporaryFile') as mock_tempfile:
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        mock_file = MagicMock()
        mock_tempfile.return_value.__enter__.return_value = mock_file
        mock_file.seek.return_value = None
        mock_file.read.return_value = b'\xff\xfe\xfd'
        mock_log.side_effect = UnicodeDecodeError('utf-8', b'\xff\xfe\xfd', 0, 1, 'invalid start byte')

        result = run_command(['echo', 'test'], verbose=True, return_output=True)

        assert mock_log.call_count > 1
        assert any(call[0][0] == str(b'\xff\xfe\xfd') for call in mock_log.call_args_list)


# LLM-generated content at query #34
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

def test_run_command_failure_raises_exception():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None

def test_run_command_failure_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_raises_exception():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None

def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose_output():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"test" in result.captured_output

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo $HOME", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None

def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #35
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output


def test_run_command_error():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None


def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output


def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output


def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_shell_false():
    result = run_command(["echo", "hello"], shell=False, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


# LLM-generated content at query #36
#--------------------------

def test_error_wrapper_wraps_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(1, 'cmd')
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert wrapped.__class__.__name__ == 'CalledProcessError'
    assert hasattr(wrapped.__class__, '__str__')
    assert wrapped.__class__.__str__ != subprocess.CalledProcessError.__str__

def test_error_wrapper_wraps_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired('cmd', 10)
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert wrapped.__class__.__name__ == 'TimeoutExpired'
    assert hasattr(wrapped.__class__, '__str__')
    assert wrapped.__class__.__str__ != subprocess.TimeoutExpired.__str__

def test_error_wrapper_returns_other_exceptions_unchanged():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, ValueError)
    assert wrapped.__class__ is ValueError


# LLM-generated content at query #37
#--------------------------

def test_error_wrapper_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped_err = error_wrapper(err)
    result = str(wrapped_err)
    assert "Command '['ls']' returned non-zero exit status 1." in result
    assert "Captured output:" in result
    assert "    file1" in result
    assert "    file2" in result

def test_error_wrapper_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped_err = error_wrapper(err)
    result = str(wrapped_err)
    assert "Command '['ls']' returned non-zero exit status 1." in result
    assert "No output was generated." in result

def test_error_wrapper_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial output')
    wrapped_err = error_wrapper(err)
    result = str(wrapped_err)
    assert "Command '['sleep', '10']' timed out after 1 seconds" in result
    assert "Captured output:" in result
    assert "    partial output" in result

def test_error_wrapper_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped_err = error_wrapper(err)
    result = str(wrapped_err)
    assert "Command '['sleep', '10']' timed out after 1 seconds" in result
    assert "No output was generated." in result

def test_error_wrapper_other_exception():
    import subprocess
    err = ValueError("Some error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    result = str(wrapped_err)
    assert result == "Some error"

def test_error_wrapper_called_process_error_with_unicode_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    result = str(wrapped_err)
    assert "Command '['ls']' returned non-zero exit status 1." in result
    assert "Failed to parse output." in result


# LLM-generated content at query #38
#--------------------------

def test_error_wrapper_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='ls', output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='ls', output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd='sleep 10', timeout=1, output=b'partial')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial' in result

def test_error_wrapper_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd='sleep 10', timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_other_exception():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_output_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='ls', output=b'\xff')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


# LLM-generated content at query #39
#--------------------------

def test_error_wrapper_returns_original_error_when_not_subprocess_error():
    import subprocess
    class CustomError(Exception):
        pass
    custom_err = CustomError("test")
    result = error_wrapper(custom_err)
    assert result is custom_err
    assert not isinstance(result, (subprocess.CalledProcessError, subprocess.TimeoutExpired))

def test_error_wrapper_returns_original_error_for_non_subprocess_exception():
    import subprocess
    err = ValueError("invalid value")
    result = error_wrapper(err)
    assert result is err
    assert isinstance(result, ValueError)

def test_error_wrapper_returns_wrapped_error_for_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=["ls"])
    result = error_wrapper(err)
    assert result is err
    assert isinstance(result, subprocess.CalledProcessError)
    assert hasattr(result.__class__, "__str__")

def test_error_wrapper_returns_wrapped_error_for_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=["sleep", "10"], timeout=1)
    result = error_wrapper(err)
    assert result is err
    assert isinstance(result, subprocess.TimeoutExpired)
    assert hasattr(result.__class__, "__str__")


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

def test_run_command_failure_raises():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None

def test_run_command_failure_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_raises():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None

def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_output_truncation():
    long_output = "a" * 10000
    result = run_command(["echo", long_output], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output

def test_run_command_nonzero_return_with_output():
    result = run_command(["sh", "-c", "echo error; exit 1"], return_output=True)
    assert result.return_code == 1
    assert b"error" in result.captured_output


# LLM-generated content at query #2
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_failure_raises():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0

def test_run_command_failure_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_raises():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired:
        pass

def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #3
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_error_raises():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None

def test_run_command_error_ignored():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_raises():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None

def test_run_command_timeout_ignored():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo $SHELL", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None

def test_run_command_output_truncated():
    long_output = "a" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #4
#--------------------------

```python
def test_run_command_truncates_long_output_on_error():
    import subprocess
    import tempfile
    from flutes.run import run_command
    from flutes.log import MAX_OUTPUT_LENGTH
    long_output = b"x" * (MAX_OUTPUT_LENGTH + 100)
    with tempfile.NamedTemporaryFile(mode='wb') as f:
        f.write(long_output)
        f.flush()
        args = ["cat", f.name]
        result = run_command(args, ignore_errors=True)
    assert result.captured_output is not None
    assert len(result.captured_output) < len(long_output)
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert result.captured_output.endswith(b"x" * MAX_OUTPUT_LENGTH)


# LLM-generated content at query #5
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_success_with_return_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_failure_raises_exception():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None


def test_run_command_failure_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


def test_run_command_timeout_raises_exception():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None


def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_verbose_output():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output


def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output


def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output


def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


# LLM-generated content at query #6
#--------------------------

```python
def test_run_command_verbose_logging_without_timestamp_and_proc_id():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output


# LLM-generated content at query #7
#--------------------------

```python
def test_run_command_no_output_on_success():
    result = run_command(["echo", "test"], return_output=False, verbose=False)
    assert result.captured_output is None


# LLM-generated content at query #8
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_failure_raises():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None

def test_run_command_failure_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_raises():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None

def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output

def test_run_command_nonzero_return_no_exception():
    result = run_command(["false"], return_output=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_unicode_output():
    result = run_command(["printf", "\\u00e9"], return_output=True)
    assert result.return_code == 0
    assert b"\xc3\xa9" in result.captured_output


# LLM-generated content at query #9
#--------------------------

def test_error_wrapper_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped_err = error_wrapper(err)
    result = str(wrapped_err)
    assert "Command '['ls']' returned non-zero exit status 1." in result
    assert "Captured output:" in result
    assert "    file1" in result
    assert "    file2" in result

def test_error_wrapper_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped_err = error_wrapper(err)
    result = str(wrapped_err)
    assert "Command '['ls']' returned non-zero exit status 1." in result
    assert "No output was generated." in result

def test_error_wrapper_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial')
    wrapped_err = error_wrapper(err)
    result = str(wrapped_err)
    assert "Command '['sleep', '10']' timed out after 1 seconds" in result
    assert "Captured output:" in result
    assert "    partial" in result

def test_error_wrapper_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped_err = error_wrapper(err)
    result = str(wrapped_err)
    assert "Command '['sleep', '10']' timed out after 1 seconds" in result
    assert "No output was generated." in result

def test_error_wrapper_other_exception():
    import subprocess
    err = ValueError("Some error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "Some error"

def test_error_wrapper_unicode_decode_error_in_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['cmd'], output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    result = str(wrapped_err)
    assert "Command '['cmd']' returned non-zero exit status 1." in result
    assert "Failed to parse output." in result


# LLM-generated content at query #10
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_error_no_ignore():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None


def test_run_command_error_ignore():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


def test_run_command_timeout_no_ignore():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None


def test_run_command_timeout_ignore():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output


def test_run_command_with_cwd():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output


def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert b"TEST_VAR=value" in result.captured_output


def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_shell_false():
    result = run_command(["echo", "hello"], shell=False, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output


def test_run_command_nonzero_return_no_output():
    result = run_command(["false"], return_output=False)
    assert result.return_code != 0
    assert result.captured_output is None


def test_run_command_nonzero_return_with_output():
    result = run_command(["false"], return_output=True)
    assert result.return_code != 0
    assert result.captured_output is not None


# LLM-generated content at query #11
#--------------------------

```python
def test_run_command_return_output_true():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output is not None

def test_run_command_return_code_nonzero():
    result = run_command(["false"], ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true():
    result = run_command(["echo", "test"], verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_verbose():
    result = run_command(["echo", "test"], return_output=True, verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_nonzero():
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true_and_nonzero():
    result = run_command(["false"], verbose=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_all_conditions_true():
    result = run_command(["false"], return_output=True, verbose=True, ignore_errors=True)
    assert result.captured_output is not None


# LLM-generated content at query #12
#--------------------------

def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    class CustomException(Exception):
        pass
    custom_exc = CustomException("test")
    result = error_wrapper(custom_exc)
    assert result is custom_exc

def test_error_wrapper_wraps_called_process_error_with_output():
    import subprocess
    exc = subprocess.CalledProcessError(1, ["cmd"], output=b"output")
    result = error_wrapper(exc)
    assert isinstance(result, subprocess.CalledProcessError)
    assert result.__class__.__name__ == "CalledProcessError"
    str_output = str(result)
    assert "Captured output:" in str_output
    assert "output" in str_output

def test_error_wrapper_wraps_called_process_error_without_output():
    import subprocess
    exc = subprocess.CalledProcessError(1, ["cmd"], output=None)
    result = error_wrapper(exc)
    assert isinstance(result, subprocess.CalledProcessError)
    str_output = str(result)
    assert "No output was generated." in str_output

def test_error_wrapper_wraps_timeout_expired_with_output():
    import subprocess
    exc = subprocess.TimeoutExpired(["cmd"], timeout=5, output=b"timeout output")
    result = error_wrapper(exc)
    assert isinstance(result, subprocess.TimeoutExpired)
    assert result.__class__.__name__ == "TimeoutExpired"
    str_output = str(result)
    assert "Captured output:" in str_output
    assert "timeout output" in str_output

def test_error_wrapper_wraps_timeout_expired_without_output():
    import subprocess
    exc = subprocess.TimeoutExpired(["cmd"], timeout=5, output=None)
    result = error_wrapper(exc)
    assert isinstance(result, subprocess.TimeoutExpired)
    str_output = str(result)
    assert "No output was generated." in str_output

def test_error_wrapper_handles_unicode_decode_error_in_output():
    import subprocess
    non_utf8_bytes = b'\xff\xfe'
    exc = subprocess.CalledProcessError(1, ["cmd"], output=non_utf8_bytes)
    result = error_wrapper(exc)
    str_output = str(result)
    assert "Failed to parse output." in str_output

def test_error_wrapper_preserves_exception_attributes():
    import subprocess
    exc = subprocess.CalledProcessError(returncode=42, cmd=["ls", "-la"], output=b"some output")
    result = error_wrapper(exc)
    assert result.returncode == 42
    assert result.cmd == ["ls", "-la"]
    assert result.output == b"some output"

def test_error_wrapper_creates_new_class_with_correct_name():
    import subprocess
    exc = subprocess.CalledProcessError(1, ["cmd"])
    result = error_wrapper(exc)
    assert result.__class__.__name__ == "CalledProcessError"
    assert result.__class__.__bases__ == (subprocess.CalledProcessError,)

def test_error_wrapper_returns_subprocess_error_instance():
    import subprocess
    exc = subprocess.CalledProcessError(1, ["cmd"])
    result = error_wrapper(exc)
    assert isinstance(result, subprocess.CalledProcessError)

def test_error_wrapper_formats_output_with_indentation():
    import subprocess
    exc = subprocess.CalledProcessError(1, ["cmd"], output=b"line1\nline2")
    result = error_wrapper(exc)
    str_output = str(result)
    assert "    line1" in str_output
    assert "    line2" in str_output


# LLM-generated content at query #13
#--------------------------

```python
def test_run_command_verbose_with_unicode_decode_error():
    args = ["echo", "-e", "\\x80\\x81"]
    result = run_command(args, verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #14
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_success_with_return_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_error_ignore():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


def test_run_command_error_raise():
    try:
        run_command(["false"])
        assert False
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None


def test_run_command_timeout_ignore():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_timeout_raise():
    try:
        run_command(["sleep", "2"], timeout=0.1)
        assert False
    except subprocess.TimeoutExpired as e:
        assert e.output is not None


def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None


def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output


def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output


def test_run_command_shell_true():
    result = run_command("echo $SHELL", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None


def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #15
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_return_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_error_ignore():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_error_raise():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None

def test_run_command_timeout_ignore():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_timeout_raise():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None

def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert b"/tmp" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #16
#--------------------------

```python
def test_run_command_return_output_true():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output is not None

def test_run_command_return_code_nonzero():
    result = run_command(["false"], ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true():
    result = run_command(["echo", "test"], verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_verbose():
    result = run_command(["echo", "test"], return_output=True, verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_nonzero():
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true_and_nonzero():
    result = run_command(["false"], verbose=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_all_conditions_true():
    result = run_command(["false"], return_output=True, verbose=True, ignore_errors=True)
    assert result.captured_output is not None


# LLM-generated content at query #17
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output


def test_run_command_failure():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None


def test_run_command_with_cwd():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output


def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output


def test_run_command_shell_true():
    result = run_command("echo $SHELL", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None


def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


def test_run_command_error_wrapper():
    try:
        run_command(["ls", "/nonexistent"])
    except subprocess.CalledProcessError as e:
        assert "Captured output:" in str(e)


# LLM-generated content at query #18
#--------------------------

def test_error_wrapper_wraps_subprocess_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'])
    err.output = b'output line 1\noutput line 2'
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, type(err))
    assert '__str__' in wrapped.__class__.__dict__
    str_output = str(wrapped)
    assert 'Captured output:' in str_output
    assert '    output line 1' in str_output
    assert '    output line 2' in str_output

def test_error_wrapper_wraps_subprocess_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1)
    err.output = b'timeout output'
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, type(err))
    assert '__str__' in wrapped.__class__.__dict__
    str_output = str(wrapped)
    assert 'Captured output:' in str_output
    assert '    timeout output' in str_output

def test_error_wrapper_preserves_other_exceptions():
    import subprocess
    err = ValueError('some error')
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, ValueError)
    assert '__str__' not in wrapped.__class__.__dict__

def test_error_wrapper_handles_unicode_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['cmd'])
    err.output = b'\xff\xfe'
    wrapped = error_wrapper(err)
    assert wrapped is err
    str_output = str(wrapped)
    assert 'Failed to parse output.' in str_output

def test_error_wrapper_handles_no_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['cmd'])
    err.output = None
    wrapped = error_wrapper(err)
    assert wrapped is err
    str_output = str(wrapped)
    assert 'No output was generated.' in str_output

def test_error_wrapper_output_empty_bytes():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['cmd'])
    err.output = b''
    wrapped = error_wrapper(err)
    assert wrapped is err
    str_output = str(wrapped)
    assert 'Captured output:' in str_output
    assert '    ' in str_output


# LLM-generated content at query #19
#--------------------------

```python
def test_run_command_verbose_logging_without_timestamp_and_proc_id():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command
    from flutes.log import log

    with patch('subprocess.run') as mock_run, patch('tempfile.TemporaryFile') as mock_tempfile, patch('flutes.run.log') as mock_log:
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_tempfile.return_value = mock_file
        mock_file.read.return_value = b"output"
        mock_file.seek = MagicMock()

        result = run_command(["echo", "test"], verbose=True)

        assert mock_log.call_count >= 1
        first_call_args = mock_log.call_args_list[0]
        assert first_call_args[1]['timestamp'] == False
        assert first_call_args[1]['include_proc_id'] == False


# LLM-generated content at query #20
#--------------------------

def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    class CustomException(Exception):
        pass
    custom_exc = CustomException("test")
    result = error_wrapper(custom_exc)
    assert result is custom_exc
    assert isinstance(result, CustomException)


# LLM-generated content at query #21
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_failure_raises():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None


def test_run_command_failure_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


def test_run_command_timeout_raises():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None


def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None


def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output


def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output


def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_output_truncation():
    long_output = "a" * 10000
    result = run_command(["echo", long_output], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #22
#--------------------------

def test_error_wrapper_returns_err_when_not_subprocess_error():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #23
#--------------------------

def test_error_wrapper_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_called_process_error_no_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial' in result

def test_error_wrapper_timeout_expired_no_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_other_exception():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_output_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'\xff')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


# LLM-generated content at query #24
#--------------------------

def test_error_wrapper_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped_err = error_wrapper(err)
    result = str(wrapped_err)
    assert "Command '['ls']' returned non-zero exit status 1." in result
    assert "Captured output:" in result
    assert "    file1" in result
    assert "    file2" in result

def test_error_wrapper_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped_err = error_wrapper(err)
    result = str(wrapped_err)
    assert "Command '['ls']' returned non-zero exit status 1." in result
    assert "No output was generated." in result

def test_error_wrapper_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial')
    wrapped_err = error_wrapper(err)
    result = str(wrapped_err)
    assert "Command '['sleep', '10']' timed out after 1 seconds" in result
    assert "Captured output:" in result
    assert "    partial" in result

def test_error_wrapper_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped_err = error_wrapper(err)
    result = str(wrapped_err)
    assert "Command '['sleep', '10']' timed out after 1 seconds" in result
    assert "No output was generated." in result

def test_error_wrapper_other_exception():
    import subprocess
    err = ValueError("Some error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    result = str(wrapped_err)
    assert result == "Some error"

def test_error_wrapper_called_process_error_unicode_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    result = str(wrapped_err)
    assert "Command '['ls']' returned non-zero exit status 1." in result
    assert "Failed to parse output." in result


# LLM-generated content at query #25
#--------------------------

def test_error_wrapper_with_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_with_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_with_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial output')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial output' in result

def test_error_wrapper_with_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_with_other_exception():
    import subprocess
    err = ValueError('Some error')
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert str(wrapped) == 'Some error'

def test_error_wrapper_output_decoding_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result

def test_error_wrapper_preserves_original_attributes():
    import subprocess
    err = subprocess.CalledProcessError(returncode=5, cmd=['test', 'arg'], output=b'output')
    wrapped = error_wrapper(err)
    assert wrapped.returncode == 5
    assert wrapped.cmd == ['test', 'arg']
    assert wrapped.output == b'output'

def test_error_wrapper_preserves_timeout_expired_attributes():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '5'], timeout=2, output=b'out')
    wrapped = error_wrapper(err)
    assert wrapped.cmd == ['sleep', '5']
    assert wrapped.timeout == 2
    assert wrapped.output == b'out'


# LLM-generated content at query #26
#--------------------------

def test_error_wrapper_wraps_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='ls')
    err.output = b'some output'
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, type(err))
    assert '__str__' in wrapped.__class__.__dict__

def test_error_wrapper_wraps_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired(cmd='sleep 10', timeout=1)
    err.output = b'timeout output'
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, type(err))
    assert '__str__' in wrapped.__class__.__dict__

def test_error_wrapper_returns_other_exceptions_unchanged():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert wrapped.__class__ is ValueError

def test_error_wrapper_str_includes_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='ls')
    err.output = b'line1\nline2'
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'Captured output:' in str_repr
    assert '    line1' in str_repr
    assert '    line2' in str_repr

def test_error_wrapper_str_includes_no_output_message():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='ls')
    err.output = None
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'No output was generated.' in str_repr

def test_error_wrapper_str_handles_unicode_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='ls')
    err.output = b'\xff\xfe'
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'Failed to parse output.' in str_repr

def test_error_wrapper_preserves_original_attributes():
    import subprocess
    err = subprocess.CalledProcessError(returncode=42, cmd='test_cmd')
    err.output = b'output'
    wrapped = error_wrapper(err)
    assert wrapped.returncode == 42
    assert wrapped.cmd == 'test_cmd'
    assert wrapped.output == b'output'


# LLM-generated content at query #27
#--------------------------

def test_error_wrapper_non_subprocess_exception():
    class CustomException(Exception):
        pass
    exc = CustomException("test")
    result = error_wrapper(exc)
    assert result is exc


# LLM-generated content at query #28
#--------------------------

def test_error_wrapper_non_subprocess_exception():
    class CustomException(Exception):
        pass
    exc = CustomException("test")
    result = error_wrapper(exc)
    assert result is exc
    assert not isinstance(result, (subprocess.CalledProcessError, subprocess.TimeoutExpired))

def test_error_wrapper_subprocess_called_process_error_without_output():
    exc = subprocess.CalledProcessError(returncode=1, cmd=["ls"])
    result = error_wrapper(exc)
    assert isinstance(result, subprocess.CalledProcessError)
    assert result.returncode == 1
    assert result.cmd == ["ls"]
    assert result.output is None
    assert str(result).endswith("\nNo output was generated.")

def test_error_wrapper_subprocess_called_process_error_with_output():
    exc = subprocess.CalledProcessError(returncode=1, cmd=["ls"], output=b"file1\nfile2")
    result = error_wrapper(exc)
    assert isinstance(result, subprocess.CalledProcessError)
    assert result.returncode == 1
    assert result.cmd == ["ls"]
    assert result.output == b"file1\nfile2"
    assert "Captured output:" in str(result)
    assert "    file1" in str(result)
    assert "    file2" in str(result)

def test_error_wrapper_subprocess_timeout_expired_without_output():
    exc = subprocess.TimeoutExpired(cmd=["sleep", "10"], timeout=5)
    result = error_wrapper(exc)
    assert isinstance(result, subprocess.TimeoutExpired)
    assert result.cmd == ["sleep", "10"]
    assert result.timeout == 5
    assert result.output is None
    assert str(result).endswith("\nNo output was generated.")

def test_error_wrapper_subprocess_timeout_expired_with_output():
    exc = subprocess.TimeoutExpired(cmd=["sleep", "10"], timeout=5, output=b"partial")
    result = error_wrapper(exc)
    assert isinstance(result, subprocess.TimeoutExpired)
    assert result.cmd == ["sleep", "10"]
    assert result.timeout == 5
    assert result.output == b"partial"
    assert "Captured output:" in str(result)
    assert "    partial" in str(result)

def test_error_wrapper_subprocess_called_process_error_with_unicode_decode_error():
    exc = subprocess.CalledProcessError(returncode=1, cmd=["ls"], output=b'\xff\xfe')
    result = error_wrapper(exc)
    assert isinstance(result, subprocess.CalledProcessError)
    assert result.returncode == 1
    assert result.cmd == ["ls"]
    assert result.output == b'\xff\xfe'
    assert "Failed to parse output." in str(result)


# LLM-generated content at query #29
#--------------------------

def test_error_wrapper_wraps_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(1, 'cmd')
    wrapped = error_wrapper(err)
    assert wrapped.__class__.__name__ == 'CalledProcessError'
    assert hasattr(wrapped.__class__, '__str__')
    assert wrapped.__class__.__str__ != subprocess.CalledProcessError.__str__

def test_error_wrapper_wraps_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired('cmd', 10)
    wrapped = error_wrapper(err)
    assert wrapped.__class__.__name__ == 'TimeoutExpired'
    assert hasattr(wrapped.__class__, '__str__')
    assert wrapped.__class__.__str__ != subprocess.TimeoutExpired.__str__

def test_error_wrapper_returns_other_exceptions_unchanged():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert wrapped.__class__ is ValueError

def test_wrapped_exception_includes_output_in_str():
    import subprocess
    err = subprocess.CalledProcessError(1, 'cmd')
    err.output = b'output line 1\noutput line 2'
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'Captured output:' in str_repr
    assert '    output line 1' in str_repr
    assert '    output line 2' in str_repr

def test_wrapped_exception_handles_unicode_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(1, 'cmd')
    err.output = b'\xff\xfe'
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'Failed to parse output.' in str_repr

def test_wrapped_exception_no_output():
    import subprocess
    err = subprocess.CalledProcessError(1, 'cmd')
    err.output = None
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'No output was generated.' in str_repr


# LLM-generated content at query #30
#--------------------------

def test_error_wrapper_wraps_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(1, 'cmd')
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert wrapped.__class__.__name__ == 'CalledProcessError'
    assert hasattr(wrapped.__class__, '__str__')
    assert wrapped.__class__.__str__ != subprocess.CalledProcessError.__str__

def test_error_wrapper_wraps_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired('cmd', 10)
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert wrapped.__class__.__name__ == 'TimeoutExpired'
    assert hasattr(wrapped.__class__, '__str__')
    assert wrapped.__class__.__str__ != subprocess.TimeoutExpired.__str__

def test_error_wrapper_returns_other_exceptions_unchanged():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, ValueError)
    assert wrapped.__class__ is ValueError

def test_error_wrapper_predicate_at_line_3_true_for_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(1, 'cmd')
    result = isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
    assert result is True

def test_error_wrapper_predicate_at_line_3_true_for_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired('cmd', 10)
    result = isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
    assert result is True

def test_error_wrapper_predicate_at_line_3_false_for_other_exception():
    import subprocess
    err = ValueError('test')
    result = isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
    assert result is False


# LLM-generated content at query #31
#--------------------------

def test_error_wrapper_returns_err_when_not_subprocess_error():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #32
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_error_with_output():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_with_output():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_shell_false():
    result = run_command(["echo", "hello"], shell=False, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], ignore_errors=True, return_output=True)
    assert result.return_code != 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #33
#--------------------------

def test_error_wrapper_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_called_process_error_no_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial' in result

def test_error_wrapper_timeout_expired_no_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_other_exception():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_output_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'\xff')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


# LLM-generated content at query #34
#--------------------------

def test_error_wrapper_called_process_error_with_output():
    import subprocess
    try:
        subprocess.run(['false'], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        result = str(wrapped)
        assert "Captured output:" in result

def test_error_wrapper_called_process_error_without_output():
    import subprocess
    try:
        subprocess.run(['false'], check=True, stdout=None, stderr=None)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        result = str(wrapped)
        assert "No output was generated." in result

def test_error_wrapper_timeout_expired_with_output():
    import subprocess
    try:
        subprocess.run(['sleep', '2'], timeout=0.1, capture_output=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        result = str(wrapped)
        assert "Captured output:" in result

def test_error_wrapper_timeout_expired_without_output():
    import subprocess
    try:
        subprocess.run(['sleep', '2'], timeout=0.1, stdout=None, stderr=None)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        result = str(wrapped)
        assert "No output was generated." in result

def test_error_wrapper_other_exception():
    import subprocess
    exc = ValueError("test")
    wrapped = error_wrapper(exc)
    assert wrapped is exc

def test_error_wrapper_output_decoding_error():
    import subprocess
    class MockCalledProcessError(subprocess.CalledProcessError):
        def __init__(self, output):
            super().__init__(1, 'cmd')
            self.output = output
    mock_output = b'\xff\xfe'
    err = MockCalledProcessError(mock_output)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert "Failed to parse output." in result


# LLM-generated content at query #35
#--------------------------

```python
def test_run_command_return_output_true():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output is not None

def test_run_command_return_code_nonzero():
    result = run_command(["false"], ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true():
    result = run_command(["echo", "test"], verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_verbose():
    result = run_command(["echo", "test"], return_output=True, verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_nonzero():
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true_and_nonzero():
    result = run_command(["false"], verbose=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_all_conditions_true():
    result = run_command(["false"], return_output=True, verbose=True, ignore_errors=True)
    assert result.captured_output is not None


# LLM-generated content at query #36
#--------------------------

```python
def test_run_command_verbose_logging_without_cwd():
    result = run_command(["echo", "test"], verbose=True, cwd=None)
    assert result.return_code == 0

def test_run_command_verbose_logging_with_cwd():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["echo", "test"], verbose=True, cwd=tmpdir)
        assert result.return_code == 0

def test_run_command_verbose_logging_with_cwd_string():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["echo", "test"], verbose=True, cwd=tmpdir)
        assert result.return_code == 0

def test_run_command_verbose_logging_with_shell_command():
    result = run_command("echo test", verbose=True, shell=True, cwd=None)
    assert result.return_code == 0

def test_run_command_verbose_logging_with_env():
    result = run_command(["echo", "test"], verbose=True, env={"TEST": "value"}, cwd=None)
    assert result.return_code == 0


# LLM-generated content at query #37
#--------------------------

def test_error_wrapper_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='ls', output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='ls', output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd='sleep 10', timeout=1, output=b'partial output')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial output' in result

def test_error_wrapper_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd='sleep 10', timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_other_exception():
    import subprocess
    err = ValueError('test error')
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_output_decoding_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='ls', output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


# LLM-generated content at query #38
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output


def test_run_command_error_ignore():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


def test_run_command_error_raise():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None


def test_run_command_timeout_ignore():
    result = run_command(["sleep", "10"], timeout=0.001, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_timeout_raise():
    try:
        run_command(["sleep", "10"], timeout=0.001)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None


def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None


def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output


def test_run_command_with_cwd():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output


def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


def test_run_command_nonzero_return_with_output():
    result = run_command(["python3", "-c", "import sys; print('error'); sys.exit(1)"], return_output=True)
    assert result.return_code == 1
    assert b"error" in result.captured_output


