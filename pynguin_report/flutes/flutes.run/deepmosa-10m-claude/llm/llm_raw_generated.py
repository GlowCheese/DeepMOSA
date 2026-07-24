####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_error_wrapper_with_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(1, 'cmd', output=b'test output')
    wrapped = error_wrapper(err)
    result_str = str(wrapped)
    assert 'Captured output:' in result_str
    assert 'test output' in result_str


def test_error_wrapper_with_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(1, 'cmd')
    wrapped = error_wrapper(err)
    result_str = str(wrapped)
    assert 'No output was generated.' in result_str


def test_error_wrapper_with_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired('cmd', 5, output=b'timeout output')
    wrapped = error_wrapper(err)
    result_str = str(wrapped)
    assert 'Captured output:' in result_str
    assert 'timeout output' in result_str


def test_error_wrapper_with_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired('cmd', 5)
    wrapped = error_wrapper(err)
    result_str = str(wrapped)
    assert 'No output was generated.' in result_str


def test_error_wrapper_with_non_subprocess_error():
    err = ValueError('test error')
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, ValueError)


def test_error_wrapper_multiline_output():
    import subprocess
    err = subprocess.CalledProcessError(1, 'cmd', output=b'line1\nline2\nline3')
    wrapped = error_wrapper(err)
    result_str = str(wrapped)
    assert 'line1' in result_str
    assert 'line2' in result_str
    assert 'line3' in result_str
    assert '    line1' in result_str


def test_error_wrapper_preserves_error_type_hierarchy():
    import subprocess
    err = subprocess.CalledProcessError(2, 'test_cmd', output=b'output')
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert wrapped.returncode == 2
    assert wrapped.cmd == 'test_cmd'


