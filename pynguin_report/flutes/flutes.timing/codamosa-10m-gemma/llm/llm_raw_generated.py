####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    assert "s)" in captured.out

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
        
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing logic and output format
    with patch("time.time") as mock_time:
        # Mock time to simulate exactly 1.234 seconds passing
        mock_time.side_effect = [100.0, 101.234]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        # Check if the duration is rounded to 2 decimal places (1.23s)
        assert "Timing test... done. (1.23s)" in captured.out

    # Test usage as a decorator
    output = []
    @work_in_progress("Decorator test")
    def decorated_func():
        output.append(True)

    decorated_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
    assert output == [True]
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and functionality
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if duration is present (regex-like check for any float)
    assert "(" in captured.out and "s)" in captured.out

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test with mocked time to ensure precision/logic
    with patch("time.time") as mock_time:
        # Set start time to 100.0 and end time to 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timed task"):
            pass
        
        captured = capsys.readouterr()
        assert "Timed task... done. (5.50s)" in captured.out

    # Test that it works as a decorator
    @work_in_progress("Decorator test")
    def decorated_func():
        return True

    decorated_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if it contains a duration pattern like (0.00s)
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing accuracy using mock
    with patch("time.time") as mock_time:
        # Mock time.time() to return 100.0 then 105.5 (5.5 seconds difference)
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test as a decorator
    @work_in_progress("Decorator test")
    def dummy_func():
        return True

    dummy_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if it contains a duration pattern like (0.00s)
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(desc=custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing accuracy using mock
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test with actual delay to ensure end='' and flush behavior works
    with patch("sys.stdout.flush"):
        with work_in_progress("Slow task"):
            time.sleep(0.1)
        
        captured = capsys.readouterr()
        assert "Slow task... done." in captured.out
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    assert "s)" in captured.out

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing accuracy using mock
    with patch("time.time") as mock_time:
        # Return 100.0 for start, 105.5 for end
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50
        assert "done. (5.50s)" in captured.out

    # Test with an exception (ensure it doesn't swallow errors)
    with pytest.raises(ValueError):
        with work_in_progress("Error test"):
            raise ValueError("Triggered error")
    
    # Note: In the current implementation, if an exception occurs, 
    # the 'done' print statement is never reached because the 
    # context manager doesn't use try/finally. 
    # This test confirms the behavior of the provided code.
```


# LLM-generated content at query #6
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage
    with work_in_progress("Testing task"):
        pass
    
    captured = capsys.readouterr()
    # Check if the output contains the description and the completion message
    assert "Testing task... " in captured.out
    assert "done." in captured.out
    assert "s)" in captured.out

    # Test with custom description
    custom_desc = "Custom process"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... " in captured.out

    # Test with timing simulation to ensure calculation logic works
    with patch('time.time') as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50
        assert "done. (5.50s)" in captured.out

    # Test as a decorator
    output_buffer = io.StringIO()
    with patch('sys.stdout', new=output_buffer):
        @work_in_progress("Decorator test")
        def decorated_func():
            return True
        
        result = decorated_func()
        output = output_buffer.getvalue()
        
        assert result is True
        assert "Decorator test... " in output
        assert "done." in output
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if time format is present (e.g., (0.00s))
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test execution time calculation using mock time
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test that the context manager works with a block of code
    counter = 0
    with work_in_progress("Increment"):
        counter += 1
        counter += 1
    
    assert counter == 2
```


# LLM-generated content at query #8
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    assert "s)" in captured.out

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing logic (mocking time.time to ensure deterministic output)
    with patch("time.time") as mock_time:
        # Set start time to 100.0 and end time to 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test execution of code block inside context manager
    executed = False
    with work_in_progress("Execution test"):
        executed = True
    
    assert executed is True
```


# LLM-generated content at query #9
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if duration-like pattern exists (e.g., (0.00s))
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        # Simulate some work
        import time
        time.sleep(0.1)
        
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out
    assert re.search(r"\(0\.[0-9]{2}s\)", captured.out)

def test_work_in_progress_timing(capsys):
    # Test that the timer actually reflects elapsed time
    delay = 0.5
    with work_in_progress("Timer test"):
        import time
        time.sleep(delay)
    
    captured = capsys.readouterr()
    # Extract the seconds value from the string like "(0.50s)"
    match = re.search(r"\(([\d.]+)s\)", captured.out)
    assert match is not None
    elapsed = float(match.group(1))
    assert elapsed >= delay - 0.1 # Allow small margin for OS scheduling
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if seconds pattern exists (e.g., (0.00s))
    assert ".00s)" in captured.out

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing calculation using mock
    with patch("time.time") as mocked_time:
        # Mock start time as 100.0 and end time as 105.5
        mocked_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timer Test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timer Test... done. (5.50s)" in captured.out

    # Test as a decorator
    @work_in_progress("Decorator Test")
    def decorated_func():
        return True

    decorated_func()
    captured = capsys.readouterr()
    assert "Decorator Test... done." in captured.out
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    assert "s)" in captured.out

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing logic using mock to ensure time.time() difference is reflected
    with patch("time.time") as mocked_time:
        # Return 100.0 on first call, 105.5 on second call
        mocked_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50
        assert "done. (5.50s)" in captured.out

    # Test that it works as a decorator
    @work_in_progress("Decorator test")
    def dummy_func():
        return True

    assert dummy_func() is True
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #12
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test using the context manager with a custom description
    description = "Testing task"
    with work_in_progress(description):
        # Simulate some work
        pass
    
    captured = capsys.readouterr()
    # Verify the output contains the description and the 'done' suffix
    assert description in captured.out
    assert "done." in captured.out
    assert "s)" in captured.out

    # Test default description
    with work_in_progress():
        pass
    
    captured_default = capsys.readouterr()
    assert "Work in progress" in captured_default.out

    # Test with simulated time to ensure the duration is captured
    with patch('time.time') as mocked_time:
        # Mock time to return 100.0 then 105.5 (5.5 seconds difference)
        mocked_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timer test"):
            pass
        
        captured_timer = capsys.readouterr()
        assert "Timer test... done. (5.50s)" in captured_timer.out

    # Test as a decorator
    @work_in_progress("Decorator test")
    def dummy_function():
        return True

    dummy_function()
    captured_decorator = capsys.readouterr()
    assert "Decorator test... done." in captured_decorator.out
```


# LLM-generated content at query #13
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Verify it contains some duration format (e.g., (0.00s))
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out.replace("done.", "done. ")) or re.search(r"\(\d+\.\d+s\)", captured.out)

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test execution time calculation logic
    with patch("time.time") as mocked_time:
        # Mock start time as 100.0 and end time as 105.5
        mocked_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test that it works as a decorator
    output_decorator = []
    @work_in_progress("Decorator test")
    def decorated_func():
        output_decorator.append("inside")

    decorated_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
    assert output_decorator == ["inside"]
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if time is present (regex for float seconds)
    import re
    assert re.search(r"\((\d+\.\d+)s\)", captured.out)

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
        
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test execution time calculation accuracy
    with patch('time.time') as mock_time:
        # Mock time: start at 100.0, end at 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test usage as a decorator
    @work_in_progress("Decorator test")
    def decorated_func():
        return True

    decorated_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #15
#--------------------------

```python
import time
from unittest.mock import patch
import io

def test_work_in_progress(capsys):
    # Test default description
    with work_in_progress():
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    assert "Work in progress... " in captured.out
    assert "done." in captured.out

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... " in captured.out
    assert "done." in captured.out

    # Test timing accuracy (mocking time to ensure deterministic results)
    with patch('time.time') as mocked_time:
        # Set start time to 100.0 and end time to 105.5
        mocked_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... " in captured.out
        assert "done. (5.50s)" in captured.out

    # Test usage as a decorator
    @work_in_progress("Decorator test")
    def decorated_func():
        time.sleep(0.01)

    decorated_func()
    captured = capsys.readouterr()
    assert "Decorator test... " in captured.out
    assert "done." in captured.out
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if it contains a time pattern like (0.00s)
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test execution time measurement accuracy using mocks
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test that it works as a decorator
    @work_in_progress("Decorator test")
    def decorated_func():
        return True

    decorated_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #17
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage
    with work_in_progress("Testing task"):
        pass
    
    captured = capsys.readouterr()
    # Check if the output contains the description and the completion message
    assert "Testing task... " in captured.out
    assert "done." in captured.out
    # Check if it contains a time pattern (e.g., (0.00s))
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

def test_work_in_progress_custom_desc():
    custom_desc = "Custom Process"
    with work_in_progress(custom_desc) as _:
        pass
    
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with work_in_progress(custom_desc):
            pass
        output = fake_out.getvalue()
        assert custom_desc in output
        assert "done." in output

def test_work_in_progress_timing_logic(capsys):
    # Use patch to control time.time() to ensure predictable output
    with patch('time.time') as mock_time:
        # First call for begin_time, second for end_time
        mock_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Timer test"):
            pass
            
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50
        assert "done. (5.50s)" in captured.out

def test_work_in_progress_exception_handling():
    # Ensure that the context manager doesn't swallow exceptions
    # Note: The current implementation doesn't have a try/finally,
    # so if an error occurs, the "done" message won't print.
    # This test verifies the default behavior.
    with pytest.raises(ValueError):
        with work_in_progress("Error test"):
            raise ValueError("Triggered error")
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    assert "s)" in captured.out

    # Test with custom description
    custom_desc = "Custom Task"
    with work_in_progress(desc=custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing logic using mock to ensure time calculation is reflected
    with patch("time.time") as mock_time:
        # Simulate 2.5 seconds passing
        mock_time.side_effect = [100.0, 102.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (2.50s)" in captured.out

    # Test that it works as a decorator
    @work_in_progress("Decorator test")
    def decorated_func():
        return True

    decorated_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #19
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if time format is present (regex-like check for seconds)
    assert "(" in captured.out and "s)" in captured.out

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test execution timing (ensure it captures a non-zero duration if we sleep)
    with patch('time.time') as mock_time:
        # Mock time to simulate 2.5 seconds passing
        mock_time.side_effect = [100.0, 102.5]
        with work_in_progress("Timer Test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timer Test... done. (2.50s)" in captured.out

    # Test as a decorator
    @work_in_progress("Decorator Test")
    def dummy_func():
        return True

    dummy_func()
    captured = capsys.readouterr()
    assert "Decorator Test... done." in captured.out
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    assert "s)" in captured.out

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test execution time calculation logic
    # We mock time.time to return predictable values
    # First call: start of context manager
    # Second call: end of context manager
    with patch("time.time") as mock_time:
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50
        assert "done. (5.50s)" in captured.out

    # Test that it works as a decorator
    @work_in_progress("Decorator test")
    def dummy_func():
        return True

    assert dummy_func() is True
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #21
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    assert "s)" in captured.out

    # Test with custom description
    custom_desc = "Custom Task"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing logic (verifying it calls time.time)
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "(5.50s)" in captured.out

    # Test as a decorator
    @work_in_progress("Decorator test")
    def dummy_func():
        pass

    dummy_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #22
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage
    with work_in_progress("Testing task"):
        pass
    
    captured = capsys.readouterr()
    assert "Testing task... done." in captured.out
    assert "s)" in captured.out

    # Test with custom description and timing simulation
    with patch("time.time") as mock_time:
        # Mock time: start at 100.0, end at 105.5 (5.5 seconds elapsed)
        mock_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Long process"):
            pass
            
        captured = capsys.readouterr()
        assert "Long process... done. (5.50s)" in captured.out

    # Test decorator usage
    @work_in_progress("Decorator test")
    def dummy_func():
        return True

    dummy_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #23
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    
    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing logic and content format
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        # Verify the duration is calculated correctly (5.50s)
        assert "Timing test... done. (5.50s)" in captured.out

    # Test execution of code block inside context manager
    execution_flag = False
    with work_in_progress("Execute block"):
        execution_flag = True
    
    assert execution_flag is True
```


# LLM-generated content at query #24
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with custom description
    desc = "Testing task"
    with work_in_progress(desc):
        pass
    
    captured = capsys.readouterr()
    # Check if the start and end messages are printed correctly
    assert captured.out.startswith(f"{desc}... ")
    assert "done." in captured.out
    assert "s)" in captured.out

    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... " in captured.out

    # Test timing logic using mock to control time
    with patch("time.time") as mock_time:
        # Set start time to 100.0 and end time to 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timer test"):
            pass
        
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50
        assert "done. (5.50s)" in captured.out

    # Test as a decorator
    @work_in_progress("Decorator test")
    def dummy_func():
        return True

    dummy_func()
    captured = capsys.readouterr()
    assert "Decorator test... " in captured.out
    assert "done." in captured.out
```


# LLM-generated content at query #25
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if duration part exists (regex-like check for pattern)
    assert "(s)" in captured.out.split('(')[-1]

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing logic and exact output format using mock
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Test Timer"):
            pass
            
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50
        assert "Test Timer... done. (5.50s)" in captured.out

def test_work_in_progress_exception(capsys):
    # Ensure the context manager handles exceptions (the print after yield won't run, 
    # but we test that the start message is printed)
    try:
        with work_in_progress("Failing task"):
            raise ValueError("Error occurred")
    except ValueError:
        pass

    captured = capsys.readouterr()
    assert "Failing task... " in captured.out
    assert "done." not in captured.out
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if duration part exists (regex-like check for float)
    assert ".2f" in captured.out or "(" in captured.out

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing accuracy using patch
    with patch('time.time') as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test execution of block content
    executed = False
    with work_in_progress("Execution test"):
        executed = True
    
    assert executed is True
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with custom description
    with work_in_progress("Testing task"):
        pass
    
    captured = capsys.readouterr()
    assert "Testing task... done." in captured.out
    assert "s)" in captured.out

    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out

    # Test timing accuracy using mock
    with patch('time.time') as mock_time:
        # Mock start time at 100.0 and end time at 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timer test"):
            pass
        
        captured = capsys.readouterr()
        assert "(5.50s)" in captured.out

    # Test error handling (ensure context manager exits even if error occurs)
    with pytest.raises(ValueError):
        with work_in_progress("Error test"):
            raise ValueError("Trigger error")
            
    captured = capsys.readouterr()
    assert "Error test... done." in captured.out
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Verify it contains a time pattern like (0.00s)
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(desc=custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing accuracy using mock
    with patch("time.time") as mocked_time:
        # Simulate 2.5 seconds passing
        mocked_time.side_effect = [100.0, 102.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (2.50s)" in captured.out

    # Test that it works as a decorator
    @work_in_progress("Decorator test")
    def dummy_func():
        return True

    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
    assert dummy_func() is True
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with custom description
    desc = "Testing task"
    with work_in_progress(desc):
        pass
    
    captured = capsys.readouterr()
    assert desc in captured.out
    assert "done." in captured.out
    assert "s)" in captured.out

    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress" in captured.out

    # Test timing logic using mock to control time
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timer test"):
            pass
        
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50
        assert "(5.50s)" in captured.out

    # Test that it works as a decorator
    @work_in_progress("Decorator test")
    def dummy_func():
        return True

    captured = capsys.readouterr()
    assert "Decorator test" in captured.out
    assert dummy_func() is True
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    description = "Test Task"
    
    with work_in_progress(description):
        pass
    
    captured = capsys.readouterr()
    # Check if the output contains the description and the completion message
    assert description in captured.out
    assert "done." in captured.out
    # Check if the duration format is present (e.g., (0.00s))
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out.replace("done.", "done. ")) or \
           re.search(r"\(\d+\.\d+s\)", captured.out)

def test_work_in_progress_timing(capsys):
    description = "Timed Task"
    sleep_duration = 0.1
    
    with patch("time.time") as mock_time:
        # Simulate time passing
        mock_time.side_effect = [100.0, 100.0 + sleep_duration]
        
        with work_in_progress(description):
            pass
            
    captured = capsys.readouterr()
    # The output should reflect the difference of 0.10s
    assert "done. (0.10s)" in captured.out

def test_work_in_progress_default_desc(capsys):
    with work_in_progress():
        pass
        
    captured = capsys.readouterr()
    assert "Work in progress... " in captured.out
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if it contains a duration pattern like (0.00s)
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test execution time measurement using mock to ensure time passes
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test that it works as a decorator
    @work_in_progress("Decorator test")
    def dummy_func():
        return True

    dummy_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import patch
import io

def test_work_in_progress(capsys):
    description = "Testing task"
    
    # Test using context manager
    with work_in_progress(description):
        pass
    
    captured = capsys.readouterr()
    # Check if the start and end messages are present
    # Using strip/split logic to handle potential time variation in the second part
    assert captured.out.startswith(f"{description}... ")
    assert "done." in captured.out
    assert "s)" in captured.out

    # Test with default parameter
    with work_in_progress():
        pass
    
    captured_default = capsys.readouterr()
    assert "Work in progress... " in captured_default.out

    # Test timing accuracy (mocking time.time)
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timer test"):
            pass
        
        captured_timing = capsys.readouterr()
        assert "Timer test... " in captured_timing.out
        assert "done. (5.50s)" in captured_timing.out
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with custom description
    with work_in_progress("Testing task"):
        pass
    
    captured = capsys.readouterr()
    assert "Testing task... done." in captured.out
    assert "s)" in captured.out

    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out

    # Test execution time calculation using mock
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timer test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timer test... done. (5.50s)" in captured.out

    # Test that it works as a context manager with a block of code
    with patch("time.time") as mock_time:
        mock_time.side_effect = [0.0, 1.0]
        with work_in_progress("Block test"):
            x = 1 + 1
            assert x == 2
        
        captured = capsys.readouterr()
        assert "Block test... done. (1.00s)" in captured.out
```


# LLM-generated content at query #8
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "... " in captured.out
    assert "done." in captured.out

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... " in captured.out
    assert "done." in captured.out

    # Test timing functionality (mocking time.time to ensure deterministic output)
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Timing test"):
            pass
            
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50
        assert "done. (5.50s)" in captured.out
```


# LLM-generated content at query #9
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check for time format pattern (e.g., (0.00s))
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        # Simulate some work
        import time
        time.sleep(0.1)
        
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out
    
    # Test timing accuracy (ensure it captures elapsed time)
    # We patch time.time to control the duration exactly
    with patch("time.time") as mocked_time:
        mocked_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timer Test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timer Test... done. (5.50s)" in captured.out

def test_work_in_progress_exception(capsys):
    # Ensure that even if an error occurs, the start message is printed
    # and the context manager handles the flow (though it doesn't catch exceptions)
    try:
        with work_in_progress("Error Test"):
            raise ValueError("Test Error")
    except ValueError:
        pass

    captured = capsys.readouterr()
    assert "Error Test... " in captured.out
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with custom description
    desc = "Testing task"
    with work_in_progress(desc):
        # Simulate some work
        pass
    
    captured = capsys.readouterr()
    assert desc in captured.out
    assert "done." in captured.out
    assert "s)" in captured.out

    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress" in captured.out

    # Test timing logic via mocking time.time
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterray().out if hasattr(capsys, 'readouterray') else capsys.readouterr().out
        # Check if the duration is correctly calculated (5.50s)
        assert "(5.50s)" in captured
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    assert "s)" in captured.out

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing accuracy using mock
    with patch("time.time") as mock_time:
        # Set start time to 100.0 and end time to 105.5
        mock_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Timer Test"):
            pass
            
        captured = capsys.readouterr()
        assert "Timer Test... done. (5.50s)" in captured.out

    # Test that it works as a context manager for code blocks
    x = 0
    with work_in_progress("Increment"):
        x += 1
    
    assert x == 1
    captured = capsys.readouterr()
    assert "Increment... done." in captured.out
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    assert "s)" in captured.out

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing logic using mock to ensure duration is captured
    with patch("time.time") as mock_time:
        # Mock time.time() to return 100.0 then 105.5 (5.5 seconds difference)
        mock_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Timer test"):
            pass
            
        captured = capsys.readouterr()
        # 5.50s should be in the output
        assert "done. (5.50s)" in captured.out

    # Test that it works as a context manager with a block of code
    with patch("time.time") as mock_time:
        mock_time.side_effect = [0.0, 1.0]
        with work_in_progress("Block test"):
            x = 1 + 1
            assert x == 2
        
        captured = capsys.readouterr()
        assert "Block test... done. (1.00s)" in captured.out
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    description = "Testing WIP"
    
    with work_in_progress(description):
        # Simulate some work
        pass
    
    captured = capsys.readouterr()
    
    # Check if description and completion message are printed
    assert description in captured.out
    assert "done." in captured.out
    # Check if the time format (s) is present
    assert "s)" in captured.out

def test_work_in_progress_default_desc(capsys):
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out

def test_work_in_progress_timing(capsys):
    # Mock time.time to return predictable values
    # First call: start time, Second call: end time
    with patch('time.time') as mocked_time:
        mocked_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Timer Test"):
            pass
            
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50
        assert "done. (5.50s)" in captured.out

def test_work_in_progress_exception_handling(capsys):
    # Note: The current implementation does not have a try/finally block,
    # so an exception will prevent the "done" message from printing.
    # This test verifies the current behavior.
    
    with pytest.raises(ValueError):
        with work_in_progress("Error test"):
            raise ValueError("Boom")
            
    captured = capsys.readouterr()
    assert "Error test... " in captured.out
    assert "done." not in captured.out
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if it contains a duration pattern like (0.00s)
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test with actual time delay to verify time calculation
    with patch("time.time") as mock_time:
        # Mock time.time() to return 100.0 then 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out
```


# LLM-generated content at query #15
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "... " in captured.out
    assert "done. (" in captured.out
    assert "s)" in captured.out

    # Test custom description
    custom_desc = "Custom Task"
    with work_in_progress(desc=custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... " in captured.out

    # Test timing calculation (mocking time to ensure deterministic output)
    with patch('time.time') as mocked_time:
        # Mock start time as 100.0 and end time as 105.5
        mocked_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50
        assert "done. (5.50s)" in captured.out
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    description = "Testing task"
    
    with work_in_progress(description):
        pass
    
    captured = capsys.readouterr()
    # Check if the description and completion message are printed
    assert f"{description}... done." in captured.out
    # Check if the time format (s) is present in the output
    assert "s)" in captured.out

def test_work_in_progress_default_desc(capsys):
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out

def test_work_in_progress_timing(capsys):
    delay = 0.1
    
    with work_in_progress("Delayed task"):
        time.sleep(delay)
        
    captured = capsys.readouterr()
    # Extract the time part from the string, e.g., (0.10s)
    # We check if the duration is at least close to the delay
    import re
    match = re.search(r"\(([\d.]+)s\)", captured.out)
    assert match is not None
    duration = float(match.group(1))
    assert duration >= delay
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    desc = "Testing task"
    
    with work_in_progress(desc):
        pass
    
    captured = capsys.readouterr()
    
    # Verify the output contains the description and the "done" status
    assert desc in captured.out
    assert "done." in captured.out
    assert "s)" in captured.out

def test_work_in_progress_timing(capsys):
    desc = "Slow task"
    sleep_duration = 0.1
    
    with work_in_progress(desc):
        time.sleep(sleep_duration)
        
    captured = capsys.readouterr()
    
    # Extract the time from the output string like "(0.10s)"
    # We check if the printed time is at least the sleep duration
    import re
    match = re.search(r"\((.*?)s\)", captured.out)
    assert match is not None
    
    elapsed_time = float(match.group(1))
    assert elapsed_time >= sleep_duration - 0.05  # Small buffer for precision

def test_work_in_progress_exception(capsys):
    desc = "Failing task"
    
    with pytest.raises(ValueError):
        with work_in_progress(desc):
            raise ValueError("Error occurred")
            
    captured = capsys.readouterr()
    
    # The context manager should still print the initial part
    assert desc in captured.out
    # Note: The original implementation does not have a try/finally 
    # to print "done" on exception, so we only verify the start message.
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if a duration was printed (regex for float)
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out) or re.search(r"\(\d+\.\d+s\)", captured.out)

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing logic using mock to ensure time_consumed is calculated correctly
    with patch('time.time') as mock_time:
        # Mock sequence: start_time, end_time
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test execution of block content
    executed = False
    with work_in_progress("Execution check"):
        executed = True
    
    assert executed is True
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if duration format exists (e.g., (0.00s))
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(desc=custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing accuracy using patch
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timer Test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timer Test... done. (5.50s)" in captured.out

    # Test functionality inside a block
    with work_in_progress("Task"):
        x = 1 + 1
        assert x == 2
    
    captured = capsys.readouterr()
    assert "Task... done." in captured.out
```


# LLM-generated content at query #20
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with custom description
    desc = "Testing custom description"
    with work_in_progress(desc):
        pass
    
    captured = capsys.readouterr()
    assert desc in captured.out
    assert "done." in captured.out
    assert "s)" in captured.out

    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress" in captured.out
    assert "done." in captured.out

    # Test timing measurement accuracy (mocking time)
    with patch('time.time') as mock_time:
        # Mock start time at 100.0 and end time at 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterroll() if hasattr(capsys, 'readouterroll') else capsys.readouterr()
        # Check if the duration is approximately 5.50s
        assert "(5.50s)" in captured.out
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with custom description
    description = "Testing task"
    with work_in_progress(description):
        pass
    
    captured = capsys.readouterr()
    assert f"{description}... done." in captured.out
    assert "(0.00s)" in captured.out

    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out

    # Test timing logic using mock
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test execution of code block within context
    execution_flag = []
    with work_in_progress("Executing block"):
        execution_flag.append(True)
    
    assert execution_flag[0] is True
    captured = capsys.readouterr()
    assert "Executing block... done." in captured.out
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    description = "Test Task"
    
    with work_in_progress(description):
        pass
    
    captured = capsys.readouterr()
    
    # Check if the start message and end message are present
    # Note: The exact time varies, so we check for the pattern
    assert f"{description}... " in captured.out
    assert "done. (" in captured.out
    assert "s)" in captured.out

def test_work_in_progress_default_desc(capsys):
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... " in captured.out

def test_work_in_progress_timing(capsys):
    delay = 0.1
    
    with work_in_progress("Delayed Task"):
        time.sleep(delay)
        
    captured = capsys.readouterr()
    
    # Extract the time from the output string, e.g., "(0.10s)"
    # We check if the printed time is at least the delay
    import re
    match = re.search(r"\((.*?)s\)", captured.out)
    assert match is not None
    
    elapsed_time = float(match.group(1))
    assert elapsed_time >= delay - 0.05  # Allow small margin for system jitter
```


# LLM-generated content at query #23
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    description = "Testing task"
    
    with work_in_progress(description):
        # Simulate work
        pass

    captured = capsys.readouterr()
    
    # Check if description and completion message exist in stdout
    assert description in captured.out
    assert "done." in captured.out
    # Check if the time format is present (e.g., (0.00s))
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

def test_work_in_progress_default_desc(capsys):
    with work_in_progress():
        pass

    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out

def test_work_in_progress_timing(capsys):
    delay = 0.1
    with work_in_progress("Delayed task"):
        time.sleep(delay)
    
    captured = capsys.readouterr()
    # The elapsed time should be at least the delay
    # We extract the number from the parenthesis using regex
    import re
    match = re.search(r"\((\d+\.\d+)s\)", captured.out)
    assert match is not None
    elapsed_time = float(match.group(1))
    assert elapsed_time >= delay
```


# LLM-generated content at query #24
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "... " in captured.out
    assert "done. (" in captured.out
    assert "s)" in captured.out

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(desc=custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... " in captured.out
    assert "done. (" in captured.out

    # Test timing accuracy (mocking time.time)
    with patch("time.time") as mock_time:
        # Set start time to 100.0 and end time to 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test that it works as a decorator
    @work_in_progress("Decorator test")
    def dummy_func():
        return True

    dummy_func()
    captured = capsys.readouterr()
    assert "Decorator test... done. (" in captured.out
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "... " in captured.out
    assert "done. (" in captured.out
    assert "s)" in captured.out

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... " in captured.out
    assert "done. (" in captured.out

    # Test timing accuracy (mocking time to ensure deterministic output)
    with patch('time.time') as mock_time:
        # Start at 100.0, end at 105.5 (5.5 seconds elapsed)
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... " in captured.out
        assert "done. (5.50s)" in captured.out

    # Test integration with a block of code
    with work_in_progress("Execution block"):
        time.sleep(0.1)
        
    captured = capsys.readouterr()
    assert "Execution block... " in captured.out
    assert "done. (" in captured.out
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_work_in_progress(capsys):
    description = "Testing WIP"
    
    with work_in_progress(description):
        # Simulate work
        pass
    
    captured = capsys.readouterr()
    
    # Check if the description is present
    assert description in captured.out
    # Check if "done." is present
    assert "done." in captured.out
    # Check if the timing format (e.g., (0.00s)) is present
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

def test_work_in_progress_default_desc(capsys):
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out

def test_work_in_progress_execution_time(capsys):
    # Mock time.time to return specific values to test calculation
    # First call: begin_time, Second call: end_time
    with patch('time.time') as mocked_time:
        mocked_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Timed task"):
            pass
            
        captured = capsys.readouterr()
        assert "Timed task... done. (5.50s)" in captured.out
```


