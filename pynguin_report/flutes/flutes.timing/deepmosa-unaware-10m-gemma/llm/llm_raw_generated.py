####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    description = "Testing task"
    
    with work_in_progress(description):
        # Simulate some work
        pass
    
    captured = capsys.readouterr()
    output = captured.out
    
    # Check if the description and completion message are present
    assert f"{description}... " in output
    assert "done." in output
    # Check for seconds format (e.g., (0.00s))
    import re
    assert re.search(r"\( \d+\.\d+s\)", output.replace("done.", "done. ")) or "s)" in output

def test_work_in_progress_default():
    captured = io.StringIO()
    with patch('sys.stdout', new=captured):
        with work_in_progress():
            pass
        output = captured.getvalue()
        assert "Work in progress... " in output
        assert "done." in output

def test_work_in_progress_execution_time(capsys):
    # Mock time.time to return controlled values
    with patch('time.time') as mocked_time:
        mocked_time.side_effect = [100.0, 105.5]  # Start at 100, end at 105.5 (5.5s diff)
        
        with work_in_progress("Timer test"):
            pass
            
        captured = capsys.readouterr()
        assert "done. (5.50s)" in captured.out
```


# LLM-generated content at query #2
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test basic functionality with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "... " in captured.out
    assert "done." in captured.out

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... " in captured.out
    assert "done." in captured.out

    # Test timing accuracy (verifying the format and that it captures duration)
    with patch('time.time') as mocked_time:
        # Mock time to return 100.0 then 102.5
        mocked_time.side_effect = [100.0, 102.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        # Check if the duration is formatted correctly as (2.50s)
        assert "(2.50s)" in captured.out

    # Test that it works as a decorator
    output = []
    @work_in_progress("Decorator test")
    def dummy_func():
        output.append(True)
    
    dummy_func()
    captured = capsys.readouterr()
    assert "Decorator test... " in captured.out
    assert "done." in captured.out
    assert output == [True]
```


# LLM-generated content at query #3
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and execution flow
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if duration is present (regex for float pattern)
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out.replace("done.", "done.")) 

    # Test custom description and simulated delay
    custom_desc = "Processing data"
    with patch('time.time', side_effect=[100.0, 105.5]):
        with work_in_progress(custom_desc):
            pass
            
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out
    assert "(5.50s)" in captured.out

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
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test using context manager
    desc = "Testing WIP"
    with work_in_progress(desc):
        pass
    
    captured = capsys.readouterr()
    assert desc in captured.out
    assert "done." in captured.out
    assert "s)" in captured.out

    # Test using decorator
    @work_in_progress("Decorator test")
    def dummy_func():
        return True

    dummy_func()
    captured = capsys.readouterr()
    assert "Decorator test" in captured.out
    assert "done." in captured.out

    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress" in captured.out

    # Test timing/execution flow with patch to ensure time is measured
    with patch('time.time') as mock_time:
        # Mock start time as 100 and end time as 105
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timed task"):
            pass
        
        captured = capsys.readouterr()
        assert "5.50s" in captured.out
```


# LLM-generated content at query #5
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with a custom description
    description = "Testing WIP"
    with work_in_progress(description):
        pass
    
    captured = capsys.readouterr()
    # Check if the start and end messages are printed correctly
    assert f"{description}... " in captured.out
    assert "done." in captured.out
    assert "s)" in captured.out

    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... " in captured.out

    # Test with actual execution time simulation to verify timing logic
    with patch("time.time") as mock_time:
        # Mock start time and end time (3 seconds difference)
        mock_time.side_effect = [100.0, 103.0]
        with work_in_progress("Timer Test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timer Test... " in captured.out
        assert "done. (3.00s)" in captured.out

    # Test as a decorator on a function
    @work_in_progress("Decorator Test")
    def dummy_func():
        return True

    dummy_func()
    captured = capsys.readouterr()
    assert "Decorator Test... " in captured.out
    assert "done." in captured.out
```


# LLM-generated content at query #6
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and execution flow
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if time duration is present in parentheses (regex-like check)
    assert "(" in captured.out and "s)" in captured.out

    # Test custom description and simulated delay
    custom_desc = "Running heavy task"
    with patch("time.time") as mocked_time:
        # Mock time to simulate a 2.5 second duration
        mocked_time.side_effect = [100.0, 102.5]
        with work_in_progress(custom_desc):
            pass
            
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out
    assert "(2.50s)" in captured.out

    # Test with a block of code that actually runs
    with patch("time.sleep", return_value=None):
        with work_in_progress("Sleeping"):
            time.sleep(0.1)
    
    captured = capsys.readouterr()
    assert "Sleeping... done." in captured.out
```


# LLM-generated content at query #7
#--------------------------

```python
import time
from io import StringIO
import pytest

def test_work_in_progress(capsys):
    description = "Test Task"
    sleep_duration = 0.1

    with work_in_progress(description):
        time.sleep(sleep_duration)

    captured = capsys.readouterr()
    output = captured.out

    # Check if description and completion message are present
    assert f"{description}... " in output
    assert "done." in output
    
    # Verify timing logic (approximate due to OS scheduling)
    # Extract the time from the string like "(0.10s)"
    import re
    match = re.search(r"\(([\d.]+)s\)", output)
    assert match is not None
    
    execution_time = float(match.group(1))
    assert execution_time >= sleep_duration - 0.05
```


# LLM-generated content at query #8
#--------------------------

```python
import time
from io import StringIO
import pytest

def test_work_in_progress(capsys):
    description = "Testing WIP"
    
    with work_in_progress(description):
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    output = captured.out
    
    assert description in output
    assert "done." in output
    assert "s)" in output
    # Check if it contains a numeric value for seconds (approx 0.1s)
    import re
    match = re.search(r'\((\d+\.\d+)s\)', output)
    assert match is not None
    assert float(match.group(1)) >= 0.1

def test_work_in_progress_default_desc(capsys):
    with work_in_progress():
        pass
        
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out

def test_work_in_progress_exception(capsys):
    with pytest.raises(ValueError):
        with work_in_progress("Error case"):
            raise ValueError("Trigger error")
            
    captured = capsys.readouterr()
    # The start of the message should still have been printed before the exception
    assert "Error case... " in captured.out
```


# LLM-generated content at query #9
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    description = "Testing WIP"
    
    # Test context manager usage
    with work_in_progress(description):
        pass
    
    captured = capsys.readouterr()
    
    # Verify the output format contains description and 'done.'
    assert description in captured.out
    assert "done." in captured.out
    # Verify the time format exists (e.g., '(0.00s)')
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out.replace("done.", "done.")) or "s)" in captured.out

def test_work_in_progress_timing(capsys):
    description = "Slow task"
    sleep_duration = 0.1
    
    with patch('time.time') as mocked_time:
        # Mock time to simulate elapsed time
        mocked_time.side_effect = [100.0, 100.5]
        
        with work_in_progress(description):
            pass
            
        captured = capsys.readouterr()
        # Check if the calculated duration is correct (0.5s)
        assert "done. (0.50s)" in captured.out

def test_work_in_progress_exception(capsys):
    description = "Failing task"
    
    try:
        with work_in_progress(description):
            raise ValueError("Test Error")
    except ValueError:
        pass
        
    captured = capsys.readouterr()
    # The context manager should still print the start message 
    # Note: In the provided implementation, 'done.' is not printed on exception 
    # because the code after 'yield' is skipped when an exception propagates.
    assert description in captured.out
    assert "..." in captured.out
```


# LLM-generated content at query #10
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with a description
    with work_in_progress("Testing task"):
        pass
    
    captured = capsys.readouterr()
    assert "Testing task... done." in captured.out
    # Check if time format is present (e.g., (0.00s))
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out

    # Test execution time measurement logic
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test with a block containing code
    with patch("time.time", side_effect=[10.0, 12.0]):
        with work_in_progress("Complex task"):
            x = 1 + 1
            assert x == 2
        
        captured = capsys.readouterr()
        assert "Complex task... done. (2.00s)" in captured.out
```


# LLM-generated content at query #11
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and basic functionality
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if duration format is present (e.g., (0.00s))
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc) as _:
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test with actual time elapsed using mock
    with patch("time.time") as mock_time:
        # Mock sequence: start_time, end_time
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


# LLM-generated content at query #12
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test basic functionality and output format
    desc = "Test Task"
    with work_in_progress(desc):
        pass
    
    captured = capsys.readouterr()
    assert desc in captured.out
    assert "done." in captured.out
    assert "s)" in captured.out

    # Test with custom delay to verify timing logic (approximate check)
    with patch("time.time") as mock_time:
        # Simulate time passing: start at 100.0, end at 105.5
        mock_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Timing test"):
            pass
            
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test default parameter
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
```


# LLM-generated content at query #13
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and execution flow
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    assert "s)" in captured.out

    # Test custom description
    custom_desc = "Testing custom description"
    with work_in_progress(custom_desc) as _:
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing logic with mocked time
    start_time = 100.0
    end_time = 105.5
    
    with patch('time.time', side_effect=[start_time, end_time]):
        with work_in_progress("Timing test"):
            pass
            
    captured = capsys.readouterr()
    # Check if the elapsed time (5.50s) is correctly formatted in the output
    assert "done. (5.50s)" in captured.out

    # Test with a block of code that performs an action
    action_performed = False
    with work_in_progress("Action"):
        action_performed = True
        
    captured = capsys.readouterr()
    assert "Action... done." in captured.out
    assert action_performed is True
```


# LLM-generated content at query #14
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test standard usage as a context manager
    with work_in_progress("Testing task"):
        pass
    
    captured = capsys.readouterr()
    assert "Testing task... " in captured.out
    assert "done." in captured.out

    # Test custom description
    custom_desc = "Custom Process"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... " in captured.out

    # Test timing logic and precision
    # Mocking time.time to return deterministic values
    # First call: start, Second call: end
    with patch("time.time", side_effect=[100.0, 105.5]):
        with work_in_progress("Timed task"):
            pass
        
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50s
        assert "done. (5.50s)" in captured.out

    # Test usage as a decorator
    @work_in_progress("Decorator test")
    def dummy_func():
        return True

    dummy_func()
    captured = capsys.readouterr()
    assert "Decorator test... " in captured.out
    assert "done." in captured.out
```


# LLM-generated content at query #15
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
    assert "done." in captured.out

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... " in captured.out
    assert "done." in captured.out

    # Test timing logic and output format using mock time
    with patch("time.time") as mock_time:
        # Setup sequence: start at 100.0, end at 105.5
        mock_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        # Check if the duration is correctly formatted (5.50s)
        assert "done. (5.50s)" in captured.out

    # Test functionality inside a block with execution time
    with patch("time.time") as mock_time:
        mock_time.side_effect = [10.0, 12.345]
        with work_in_progress("Execution"):
            # Simulate some work
            pass
        
        captured = capsys.readouterr()
        # Should round to 2 decimal places (2.35s or 2.34s depending on float precision)
        assert "done. (2.35s)" in captured.out or "done. (2.34s)" in captured.out
```


# LLM-generated content at query #16
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
    
    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing logic and format using mock to control time
    with patch('time.time') as mock_time:
        # Return 100.0 then 102.5 (duration 2.5s)
        mock_time.side_effect = [100.0, 102.5]
        with work_in_progress("Timing test"):
            pass
        captured = capsys.readouterr()
        assert "Timing test... done. (2.50s)" in captured.out

    # Test exception handling (ensure the context manager doesn't swallow errors)
    # Note: The current implementation doesn't have a try/finally, 
    # so an error will propagate and skip the 'done' print.
    with pytest.raises(ValueError):
        with work_in_progress("Error test"):
            raise ValueError("Test Error")
    captured = capsys.readouterr()
    assert "Error test... " in captured.out
    assert "done." not in captured.out
```


# LLM-generated content at query #17
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

    # Test timing logic (mocking time to ensure deterministic output)
    with patch("time.time") as mock_time:
        mock_time.side_effect = [100.0, 105.5]  # 5.5 seconds difference
        with work_in_progress("Timer Test"):
            pass
        captured = capsys.readouterr()
        assert "Timer Test... done. (5.50s)" in captured.out

    # Test with block of code execution
    with patch("time.time") as mock_time:
        mock_time.side_effect = [100.0, 100.25] # 0.25 seconds difference
        with work_in_progress("Execution"):
            time.sleep(0) # Ensure no real delay interferes with logic
        captured = capsys.readouterr()
        assert "Execution... done. (0.25s)" in captured.out
```


# LLM-generated content at query #18
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and execution flow
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if it contains a duration pattern like (0.00s)
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out) or re.search(r"\(\d+\.\d+s\)", captured.out)

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
        
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test with simulated time delay to verify timing logic
    with patch("time.time") as mock_time:
        # Mock start time at 100.0 and end time at 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test error handling (ensure print happens even if block fails)
    try:
        with work_in_progress("Error test"):
            raise ValueError("Test Error")
    except ValueError:
        pass
    
    captured = capsys.readouterr()
    assert "Error test... done." in captured.out
```


# LLM-generated content at query #19
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test standard usage with custom description
    with work_in_progress("Testing task"):
        pass
    
    captured = capsys.readouterr()
    assert "Testing task... done." in captured.out
    # Check if it contains a duration pattern (e.g., (0.00s))
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test usage with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out

    # Test timing accuracy (simulated delay)
    with patch('time.time') as mocked_time:
        # Mock time to return 100.0 then 102.5
        mocked_time.side_effect = [100.0, 102.5]
        with work_in_progress("Timed task"):
            pass
        
        captured = capsys.readouterr()
        assert "Timed task... done. (2.50s)" in captured.out

    # Test with actual sleep to ensure end='' and flush=True logic works in flow
    with work_in_progress("Sleep test"):
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    assert "Sleep test... done." in captured.out
```


# LLM-generated content at query #20
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    description = "Testing task"
    
    with work_in_progress(description):
        # Simulate some work
        pass
    
    captured = capsys.readouterr()
    
    # Verify the start and end of the message are present
    assert description + "... " in captured.out
    assert "done." in captured.out
    # Verify that it contains a time format like (X.XXs)
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out.replace("done.", "done. ")) or \
           re.search(r"\(\d+\.\d+s\)", captured.out)

def test_work_in_progress_default_desc(capsys):
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... " in captured.out

def test_work_in_progress_execution_time(capsys):
    start_time = time.time()
    duration = 0.5
    
    with patch('time.time') as mocked_time:
        # Mock two calls to time.time(): one for begin, one for end
        mocked_time.side_effect = [start_time, start_time + duration]
        
        with work_in_progress("Timer test"):
            pass
            
    captured = capsys.readouterr()
    # Check if the printed time matches our mocked duration (0.50s)
    assert "done. (0.50s)" in captured.out
```


# LLM-generated content at query #21
#--------------------------

```python
import time
from io import StringIO
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and basic execution
    with work_in_progress() as _:
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if duration is present (at least 0.1s)
    import re
    match = re.search(r"\((\d+\.\d+)s\)", captured.out)
    assert match is not None
    assert float(match.group(1)) >= 0.1

    # Test custom description
    custom_desc = "Testing custom message"
    with work_in_progress(custom_desc) as _:
        time.sleep(0.05)
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test exact timing calculation via mocking time.time
    with patch("time.time") as mock_time:
        # Mock sequence: start at 100.0, end at 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Mocked time") as _:
            pass
        
        captured = capsys.readouterr()
        assert "Mocked time... done. (5.50s)" in captured.out

    # Test with exception handling (ensure context manager doesn't swallow errors)
    with pytest.raises(ValueError):
        with work_in_progress("Error test"):
            raise ValueError("Trigger error")
    
    captured = capsys.readouterr()
    assert "Error test... " in captured.out
```


# LLM-generated content at query #22
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if it contains a duration pattern like (0.00s)
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test context manager usage with custom description and simulated delay
    with patch("time.time") as mock_time:
        # Mock time to simulate a 2.5 second execution
        mock_time.side_effect = [100.0, 102.5]
        
        with work_in_progress("Processing data"):
            pass
            
        captured = capsys.readouterr()
        assert "Processing data... done. (2.50s)" in captured.out

    # Test as a decorator
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
    # Test default description and basic execution
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if a duration pattern exists (e.g., (0.00s))
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test custom description and simulated time delay
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Processing data"):
            pass
            
        captured = capsys.readouterr()
        assert "Processing data... done. (5.50s)" in captured.out

    # Test with actual sleep to ensure time measurement is dynamic
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        start_sleep = time.time()
        with work_in_progress("Sleeping"):
            time.sleep(0.1)
        output = fake_out.getvalue()
        duration = time.time() - start_sleep
        assert "Sleeping... done." in output
        # Verify the printed duration is roughly consistent with actual sleep
        import re
        match = re.search(r"\((\d+\.\d+)s\)", output)
        if match:
            printed_duration = float(match.group(1))
            assert printed_duration >= 0.1
```


# LLM-generated content at query #24
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and execution flow
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if it contains a time pattern like (0.00s)
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test custom description and simulated delay
    with patch("time.time") as mock_time:
        # Mock start time at 100.0 and end time at 105.5
        mock_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Processing data"):
            pass
            
        captured = capsys.readouterr()
        assert "Processing data... done. (5.50s)" in captured.out

    # Test that it works as a decorator
    @work_in_progress("Decorator test")
    def dummy_func():
        pass

    dummy_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #25
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test custom description and execution flow
    description = "Testing context manager"
    with work_in_progress(description):
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    
    # Verify the output starts with the description and ends with 'done.'
    assert captured.out.startswith(f"{description}... ")
    assert "done." in captured.out
    
    # Test default description
    with work_in_progress():
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    assert "Work in progress... " in captured.out
    assert "done." in captured.out

def test_work_in_progress_timing(capsys):
    # Test if the elapsed time is approximately correct
    duration = 0.5
    with work_in_progress("Timer test"):
        time.sleep(duration)
    
    captured = capsys.readouterr()
    
    # Extract the seconds value from the output string using a simple approach
    # Expected format: "done. (0.50s)" or similar
    import re
    match = re.search(r"\(([\d.]+)\s?s\)", captured.out)
    assert match is not None
    
    elapsed_time = float(match.group(1))
    # Allow a small margin for execution overhead
    assert duration <= elapsed_time <= duration + 0.2
```


# LLM-generated content at query #26
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test using context manager with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if it contains a duration pattern (e.g., (0.00s))
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out) or re.search(r"\(\d+\.\d+s\)", captured.out)

    # Test using context manager with custom description
    custom_desc = "Processing data"
    with work_in_progress(desc=custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing accuracy with a sleep delay
    with patch("time.time") as mock_time:
        # Mock time to simulate 2.5 seconds passing
        mock_time.side_effect = [100.0, 102.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (2.50s)" in captured.out

    # Test as a decorator
    @work_in_progress("Decorator test")
    def decorated_func():
        return True

    decorated_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #27
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and basic execution
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    assert "(s)" in captured.out

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc) as _:
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing logic using mock to ensure duration is reflected correctly
    with patch("time.time") as mock_time:
        # Mock time sequence: start at 100.0, end at 105.5 (5.5s difference)
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test exception handling (ensure the context manager doesn't swallow errors)
    with pytest.raises(ValueError):
        with work_in_progress("Error test"):
            raise ValueError("Triggered error")
    
    captured = capsys.readouterr()
    # The print statement happens before yield, so the start of the message should exist
    assert "Error test... " in captured.out
```


# LLM-generated content at query #28
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
    # Check if time format exists (regex-like check for decimal)
    assert "(s)" in captured.out

    # Test custom description and simulated delay
    custom_desc = "Processing data"
    with patch("time.time") as mocked_time:
        # Mock start time at 100.0 and end time at 105.5
        mocked_time.side_effect = [100.0, 105.5]
        with work_in_progress(custom_desc):
            pass
            
        captured = capsys.readouterr()
        assert f"{custom_desc}... done." in captured.out
        assert "(5.50s)" in captured.out

    # Test context manager with actual block execution
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        with work_in_progress("Running task"):
            x = 1 + 1
            assert x == 2
        
        output = fake_out.getvalue()
        assert "Running task... done." in output
```


# LLM-generated content at query #29
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test usage as a context manager
    description = "Testing WIP"
    with work_in_progress(description):
        pass
    
    captured = capsys.readouterr()
    # Check if the output starts with description and ends with done pattern
    assert captured.out.startswith(f"{description}... ")
    assert "done." in captured.out

    # Test usage as a decorator
    @work_in_progress("Decorator test")
    def dummy_function():
        return True

    dummy_function()
    captured = capsys.readouterr()
    assert "Decorator test... " in captured.out
    assert "done." in captured.out

    # Test timing accuracy (mocking time to ensure deterministic output)
    with patch("time.time") as mock_time:
        # Start at 100.0, end at 105.5 (duration 5.5s)
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timer test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timer test... done. (5.50s)" in captured.out

    # Test default description
    with work_in_progress():
        pass
    captured = capsys.readouterr()
    assert "Work in progress... " in captured.out
```


# LLM-generated content at query #30
#--------------------------

```python
import time
from io import StringIO
import pytest

def test_work_in_progress(capsys):
    desc = "Test Task"
    
    with work_in_progress(desc):
        time.sleep(0.1)
        
    captured = capsys.readouterr()
    output = captured.out
    
    assert f"{desc}... done." in output
    # Check if the timing part exists and contains 's' (seconds)
    import re
    match = re.search(r"\((\d+\.\d+)s\)", output)
    assert match is not None
    
    duration = float(match.group(1))
    assert duration >= 0.1
```


# LLM-generated content at query #31
#--------------------------

```python
import time
from io import StringIO
from unittest.mock import patch

def test_work_in_progress():
    # Test with default description
    with patch('sys.stdout', new=StringIO()) as fake_out:
        with work_in_progress():
            time.sleep(0.1)
        output = fake_out.getvalue()
        assert "Work in progress... " in output
        assert "done." in output
        # Check if seconds are present (approx 0.1s)
        assert "(0.1" in output or "(0.0" in output or "(0.2" in output

    # Test with custom description
    custom_desc = "Processing data"
    with patch('sys.stdout', new=StringIO()) as fake_out:
        with work_in_progress(custom_desc):
            time.sleep(0.05)
        output = fake_out.getvalue()
        assert f"{custom_desc}... " in output
        assert "done." in output

    # Test timing accuracy (ensure elapsed time is captured)
    with patch('sys.stdout', new=StringIO()) as fake_out:
        start = time.time()
        with work_in_progress("Timer"):
            delay = 0.2
            time.sleep(delay)
        end = time.time()
        output = fake_out.getvalue()
        # Extract the seconds value from the string using simple parsing
        # Format: "done. (X.XXs)"
        seconds_str = output.split('(')[1].split('s)')[0]
        elapsed_val = float(seconds_str)
        
        assert delay <= elapsed_val <= (delay + 0.2)
```


# LLM-generated content at query #32
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and basic execution
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out

    # Test custom description and timing logic
    custom_desc = "Processing data"
    with patch('time.time') as mocked_time:
        # Mock start time as 100.0 and end time as 105.5
        mocked_time.side_effect = [100.0, 105.5]
        
        with work_in_progress(custom_desc):
            pass
        
        captured = capsys.readouterr()
        assert f"{custom_desc}... done." in captured.out
        assert "(5.50s)" in captured.out

    # Test execution with actual delay to ensure end='' and flush behavior works
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with work_in_progress("Fast task"):
            time.sleep(0.1)
        output = fake_out.getvalue()
        assert "Fast task... done." in output
```


# LLM-generated content at query #33
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test basic functionality and output format
    desc = "Testing task"
    with work_in_progress(desc):
        pass
    
    captured = capsys.readouterr()
    # Check if the start and end messages are printed correctly
    assert captured.out.startswith(f"{desc}... ")
    assert "done." in captured.out

def test_work_in_progress_timing(capsys):
    # Test if the time duration is reflected in the output
    desc = "Timing task"
    with work_in_progress(desc):
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    # Extract the seconds part using regex or simple parsing
    # Format expected: "... done. (X.XXs)"
    import re
    match = re.search(r"\((.*)s\)", captured.out)
    assert match is not None
    duration = float(match.group(1))
    assert duration >= 0.1

def test_work_in_progress_default_desc(capsys):
    # Test default parameter
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... " in captured.out

def test_work_in_progress_exception_handling(capsys):
    # Ensure it doesn't swallow exceptions, though the current implementation 
    # does not have a try/finally block to handle cleanup if an error occurs.
    # This test documents existing behavior (the print 'done' won't trigger).
    try:
        with work_in_progress("Error task"):
            raise ValueError("Triggered error")
    except ValueError:
        pass

    captured = capsys.readouterr()
    assert "Error task... " in captured.out
    assert "done." not in captured.out
```


# LLM-generated content at query #34
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test custom description and execution flow
    description = "Testing task"
    with work_in_progress(description):
        pass
    
    captured = capsys.readouterr()
    assert f"{description}... done." in captured.out
    assert "s)" in captured.out

    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out

    # Test timing logic using mock to ensure time calculation is correct
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


# LLM-generated content at query #35
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
    assert "done." in captured.out

    # Test with custom description
    custom_desc = "Testing custom desc"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... " in captured.out
    assert "done." in captured.out

    # Test timing calculation and output format
    # We mock time.time to return predictable values: 100.0 then 102.5
    with patch("time.time", side_effect=[100.0, 102.5]):
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        # Expected: "Timing test... done. (2.50s)"
        assert "Timing test... done. (2.50s)" in captured.out

    # Test integration with a block of code
    with patch("time.time", side_effect=[10.0, 10.1]):
        with work_in_progress("Block"):
            sum_val = sum([1, 2, 3])
        
        captured = capsys.readouterr()
        assert "Block... done." in captured.out
        assert sum_val == 6
```


# LLM-generated content at query #36
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and execution flow
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if it contains a duration pattern like (X.XXs)
    import re
    assert re.search(r"\(.\d{2}s\)", captured.out)

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(desc=custom_desc):
        pass
        
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test with simulated delay to ensure time calculation is functioning
    with patch("time.time") as mock_time:
        # Mock time to return 100.0 then 105.5 (5.5 seconds difference)
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test integration with a block of code
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        with work_in_progress("Integration"):
            time.sleep(0.01)
        output = fake_out.getvalue()
        assert "Integration... done." in output
```


# LLM-generated content at query #37
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with custom description
    with work_in_progress("Testing task"):
        pass
    
    captured = capsys.readouterr()
    assert "Testing task... " in captured.out
    assert "done." in captured.out

    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... " in captured.out
    assert "done." in captured.out

    # Test timing accuracy (mocking time to ensure predictable output)
    with patch('time.time') as mocked_time:
        # Setup sequence of times: start at 100.0, end at 105.5
        mocked_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Timed task"):
            pass
            
        captured = capsys.readouterr()
        # Check if the elapsed time is formatted correctly (5.50s)
        assert "done. (5.50s)" in captured.out

    # Test behavior with actual delay to ensure end='' and flush=True logic works
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        start_time = time.time()
        with work_in_progress("Delay test"):
            time.sleep(0.1)
        output = fake_out.getvalue()
        assert "Delay test... " in output
        assert "done." in output
```


# LLM-generated content at query #38
#--------------------------

```python
import pytest
from unittest.mock import patch
import io

def test_work_in_progress(capsys):
    # Test default description
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    assert "s)" in captured.out

    # Test custom description
    custom_desc = "Custom Task"
    with work_in_progress(desc=custom_desc) as _:
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing logic using mock to ensure precise string matching
    with patch("time.time") as mock_time:
        # Mock start time at 100.0 and end time at 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing Test"):
            pass
        
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50s
        assert "Timing Test... done. (5.50s)" in captured.out

    # Test execution of code inside the context manager
    execution_flag = False
    with work_in_progress("Executing block"):
        execution_flag = True
    
    assert execution_flag is True
```


# LLM-generated content at query #39
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and functionality
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if it contains some number followed by 's' (duration)
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out.replace("done.", "done.")) or "s)" in captured.out

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc) as _:
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing accuracy (mocking time to ensure deterministic output)
    with patch("time.time") as mock_time:
        # Start at 100.0, end at 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test execution of code block inside context manager
    executed = False
    with work_in_progress("Execution check"):
        executed = True
    
    assert executed is True
```


# LLM-generated content at query #40
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test usage as a context manager
    with work_in_progress("Testing task"):
        pass
    
    captured = capsys.readouterr()
    assert "Testing task... done." in captured.out
    assert "(s)" in captured.out

    # Test with custom description and simulated delay
    with patch('time.time') as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Slow task"):
            pass
            
        captured = capsys.readouterr()
        assert "Slow task... done. (5.50s)" in captured.out

    # Test usage as a decorator
    @work_in_progress("Decorator test")
    def decorated_func():
        return True

    decorated_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #41
#--------------------------

```python
import unittest.mock as mock
import time
import io

def test_work_in_progress(capsys):
    # Test case 1: Using as a context manager with default description
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if it contains a duration pattern like (0.00s)
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test case 2: Using with a custom description and simulated delay
    custom_desc = "Processing data"
    delay = 0.1
    
    with work_in_progress(custom_desc):
        time.sleep(delay)
        
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out
    
    # Verify time elapsed is at least the delay (allowing for small margin)
    duration_match = re.search(r"\(([\d.]+)s\)", captured.out)
    assert duration_match is not None
    duration_val = float(duration_match.group(1))
    assert duration_val >= delay

    # Test case 3: Using as a decorator
    @work_in_progress("Decorator test")
    def dummy_function():
        time.sleep(0.05)

    dummy_function()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #42
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if duration format is present (e.g., (0.00s))
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test context manager usage with custom description
    custom_desc = "Processing data"
    with work_in_progress(desc=custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing accuracy using patch to control time.time()
    with patch("time.time") as mock_time:
        # Mock first call (start) and second call (end)
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test execution of code block inside context manager
    execution_flag = False
    with work_in_progress("Execute block"):
        execution_flag = True
    
    assert execution_flag is True
    captured = capsys.readouterr()
    assert "Execute block... done." in captured.out
```


# LLM-generated content at query #43
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and functionality
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if duration format is present (e.g., (0.00s))
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test custom description and execution delay
    with patch("time.time") as mock_time:
        # Mock time to simulate 2.5 second duration
        mock_time.side_effect = [100.0, 102.5]
        with work_in_progress("Processing data"):
            pass
            
        captured = capsys.readouterr()
        assert "Processing data... done. (2.50s)" in captured.out

    # Test with a block of code containing logic
    with patch("time.time") as mock_time:
        mock_time.side_effect = [100.0, 101.123]
        with work_in_progress("Task"):
            x = 1 + 1
            assert x == 2
            
        captured = capsys.readouterr()
        # Verify rounding to 2 decimal places (1.123 -> 1.12)
        assert "done. (1.12s)" in captured.out
```


# LLM-generated content at query #44
#--------------------------

```python
import pytest
from unittest.mock import patch
import io

def test_work_in_progress():
    # Test context manager usage and printed output format
    test_description = "Testing task"
    
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with work_in_progress(test_description):
            pass
        
        output = fake_out.getvalue()
        
        # Verify the description is printed at start
        assert output.startswith(f"{test_description}... ")
        # Verify it ends with 'done.' and contains a time pattern (e.g., '(0.00s)')
        assert "done." in output
        assert "(0.00s)" in output or ".s)" in output

    # Test default description
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with work_in_progress():
            pass
        output = fake_out.getvalue()
        assert "Work in progress... done." in output

    # Test execution timing is reflected (using a sleep to ensure non-zero time)
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with work_in_progress("Sleeping"):
            time.sleep(0.1)
        output = fake_out.getvalue()
        # Check if the duration part exists and is greater than 0
        import re
        match = re.search(r'\((\d+\.\d+)s\)', output)
        assert match is not None
        duration = float(match.group(1))
        assert duration >= 0.1

    # Test error handling (ensure it prints start even if block fails)
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        try:
            with work_in_progress("Failing task"):
                raise ValueError("Boom")
        except ValueError:
            pass
        
        output = fake_out.getvalue()
        assert "Failing task... " in output
```


# LLM-generated content at query #45
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test basic functionality and output format
    with work_in_progress("Test Task"):
        pass
    
    captured = capsys.readouterr()
    assert "Test Task... done." in captured.out
    # Check if it contains a duration pattern (e.g., (0.00s))
    import re
    assert re.search(r"\((\d+\.\d+)s\)", captured.out)

def test_work_in_progress_custom_description():
    custom_desc = "Custom Process"
    with work_in_progress(custom_desc):
        pass
        
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

def test_work_in_progress_timing(capsys):
    # Mock time to ensure predictable output for duration testing
    with patch("time.time") as mock_time:
        # Set start time and end time with a 2.5 second difference
        mock_time.side_effect = [100.0, 102.5]
        
        with work_in_progress("Timing Test"):
            pass
            
        captured = capsys.readouterr()
        assert "done. (2.50s)" in captured.out

def test_work_in_progress_exception_handling(capsys):
    # Ensure the context manager doesn't suppress exceptions 
    # Note: The current implementation does NOT have a try/finally, 
    # so an exception will crash the block without printing "done."
    # This test verifies the current behavior (exception propagates).
    with pytest.raises(ValueError):
        with work_in_progress("Error Test"):
            raise ValueError("Something went wrong")
    
    captured = capsys.readouterr()
    assert "Error Test... " in captured.out
    assert "done." not in captured.out
```


# LLM-generated content at query #46
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and execution flow
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check for presence of seconds pattern (e.g., (0.00s))
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test custom description and timing simulation
    with patch("time.time") as mock_time:
        # Mock time to simulate 2.5 seconds passing
        mock_time.side_effect = [100.0, 102.5]
        
        with work_in_progress("Custom Task"):
            pass
            
        captured = capsys.readouterr()
        assert "Custom Task... done. (2.50s)" in captured.out

    # Test with a block containing code execution
    with patch("time.time") as mock_time:
        mock_time.side_effect = [100.0, 101.123]
        
        with work_in_progress("Running calculation"):
            x = 1 + 1
            assert x == 2
            
        captured = capsys.readouterr()
        assert "Running calculation... done. (1.12s)" in captured.out
```


# LLM-generated content at query #47
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    description = "Testing task"
    
    with work_in_progress(description):
        # Simulate some work
        pass
    
    captured = capsys.readouterr()
    
    # Check if the description and 'done' message are present
    assert description in captured.out
    assert "done." in captured.out
    # Verify the format includes the duration (e.g., '(0.00s)')
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out.replace("done.", "done. (")) or "s)" in captured.out

def test_work_in_progress_default_desc(capsys):
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... " in captured.out
    assert "done." in captured.out

def test_work_in_progress_execution_time(capsys):
    # Patch time.time to control the duration
    with patch('time.time') as mocked_time:
        # First call is begin_time, second call is end_time
        mocked_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Timer test"):
            pass
            
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50
        assert "done. (5.50s)" in captured.out
```


# LLM-generated content at query #48
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
    with work_in_progress(custom_desc):
        pass
        
    captured = capsys.readouterr()
    assert f"{custom_desc}... " in captured.out

    # Test timing accuracy (mocking time)
    with patch('time.time') as mock_time:
        # Set start time to 100.0 and end time to 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timer Test"):
            pass
            
        captured = capsys.readouterr()
        assert "done. (5.50s)" in captured.out

    # Test functionality within a block with execution delay
    with patch('time.sleep', return_value=None):
        with work_in_progress("Slow task"):
            time.sleep(0.1)
        captured = capsys.readouterr()
        assert "Slow task... " in captured.out
        assert "done." in captured.out
```


# LLM-generated content at query #49
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

    # Test timing accuracy (mocking time)
    with patch("time.time") as mock_time:
        # Simulate 2.5 seconds passing
        mock_time.side_effect = [100.0, 102.5]
        with work_in_progress("Timing test"):
            pass
        captured = capsys.readouterr()
        assert "done. (2.50s)" in captured.out

    # Test functionality with a block of code
    with work_in_progress("Task"):
        x = 1 + 1
        assert x == 2
    captured = capsys.readouterr()
    assert "Task... done." in captured.out
```


# LLM-generated content at query #50
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test basic functionality and output format
    description = "Testing WIP"
    with work_in_progress(description):
        pass
    
    captured = capsys.readouterr()
    # Check if the description is printed and ends with 'done.'
    assert description in captured.out
    assert "done." in captured.out
    # Verify that it contains a time pattern like (0.00s)
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out) or re.search(r"\(\d+\.\d+s\)", captured.out)

def test_work_in_progress_execution_time(capsys):
    # Test that the elapsed time is approximately correct
    description = "Sleeping"
    sleep_duration = 0.1
    with work_in_progress(description):
        time.sleep(sleep_duration)
    
    captured = capsys.readouterr()
    # Extract the numerical value from the output string (e.g., "0.12s")
    match = re.search(r"\((?P<seconds>\d+\.\d+)s\)", captured.out)
    assert match is not None
    elapsed = float(match.group("seconds"))
    # Allow a small margin for execution overhead
    assert elapsed >= sleep_duration - 0.01

def test_work_in_progress_default_desc(capsys):
    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out

def test_work_in_progress_error_handling(capsys):
    # Ensure that even if an error occurs, the context manager behaves (though it doesn't catch errors)
    # This tests if the print happens before the exception
    with pytest.raises(ValueError):
        with work_in_progress("Error test"):
            raise ValueError("Triggered error")
            
    captured = capsys.readouterr()
    assert "Error test... " in captured.out
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import time
from unittest.mock import patch
import io

def test_work_in_progress(capsys):
    # Test basic functionality and default description
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if it contains a duration pattern like (0.00s)
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test custom description and simulated delay
    delay = 0.1
    with work_in_progress("Custom Task"):
        time.sleep(delay)
    
    captured = capsys.readouterr()
    assert "Custom Task... done." in captured.out
    
    # Extract the seconds value from the output string using regex
    match = re.search(r"\((.*)s\)", captured.out)
    if match:
        duration = float(match.group(1))
        # Allow for small margin of error in timing
        assert duration >= delay - 0.05

def test_work_in_progress_exception(capsys):
    # Test that the context manager handles exceptions (yields to exception)
    # Note: The current implementation does not have a try/except/finally block,
    # so an exception will bubble up and the 'done' print won't execute.
    with pytest.raises(ValueError):
        with work_in_progress("Failing task"):
            raise ValueError("Error occurred")

    captured = capsys.readouterr()
    assert "Failing task... " in captured.out
    assert "done." not in captured.out
```


# LLM-generated content at query #2
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
    
    # Test custom description and timing simulation
    custom_desc = "Processing data"
    # We patch time.time to return predictable values for testing the duration calculation
    with patch("time.time") as mock_time:
        # Start at 100.0, end at 105.5 (diff of 5.5s)
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress(custom_desc):
            pass
        
        captured = capsys.readouterr()
        assert f"{custom_desc}... done. (5.50s)" in captured.out

def test_work_in_progress_exception(capsys):
    # Ensure the context manager doesn't swallow exceptions 
    # Note: The current implementation does not have a try/finally, 
    # so if an error occurs, the 'done' message won't print.
    # This test documents existing behavior.
    with pytest.raises(ValueError):
        with work_in_progress("Error test"):
            raise ValueError("Failure")
    
    captured = capsys.readouterr()
    assert "Error test... " in captured.out
    assert "done." not in captured.out
```


# LLM-generated content at query #3
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
    # Check if duration pattern exists (e.g., (0.00s))
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test with actual time delay to verify timing logic
    with patch("time.time") as mock_time:
        # Simulate 1.5 seconds passing
        mock_time.side_effect = [100.0, 101.5]
        with work_in_progress("Timed Task"):
            pass
        
        captured = capsys.readouterr()
        assert "Timed Task... done. (1.50s)" in captured.out

    # Test execution of code block within context
    result = []
    with work_in_progress("List append"):
        result.append(42)
    
    assert result == [42]
    captured = capsys.readouterr()
    assert "List append... done." in captured.out
```


# LLM-generated content at query #4
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with custom description
    description = "Testing task"
    with work_in_progress(description):
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    
    # Verify start and end messages are printed correctly
    assert f"{description}... " in captured.out
    assert "done." in captured.out
    # Check if duration pattern exists (e.g., "(0.10s)")
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out.replace("done.", "done. ")) or \
           re.search(r"\(\d+\.\d+s\)", captured.out)

    # Test default description
    with work_in_progress():
        time.sleep(0.05)
        
    captured_default = capsys.readouterr()
    assert "Work in progress... " in captured_default.out
    assert "done." in captured_default.out

def test_work_in_progress_timing():
    # Test that the time calculation is roughly accurate using patch to control time
    with patch('time.time') as mocked_time:
        # Simulate start at 100.0 and end at 105.5
        mocked_time.side_effect = [100.0, 105.5]
        
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            with work_in_progress("Timer test"):
                pass
            
            output = fake_out.getvalue()
            assert "done. (5.50s)" in output
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from io import StringIO
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and functionality
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if time part exists (regex-like check for pattern)
    assert "(s)" in captured.out

    # Test custom description and execution delay simulation
    custom_desc = "Processing data"
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        
        with work_in_progress(custom_desc):
            pass
            
        captured = capsys.readouterr()
        assert f"{custom_desc}... done." in captured.out
        assert "(5.50s)" in captured.out

    # Test with a block that actually performs work
    with patch("time.sleep", return_value=None):
        with work_in_progress("Sleep test"):
            time.sleep(0.1)
            
        captured = capsys.readouterr()
        assert "Sleep test... done." in captured.out
```


# LLM-generated content at query #6
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with a custom description
    description = "Testing task"
    with work_in_progress(description):
        pass
    
    captured = capsys.readouterr()
    # Check if the start and end messages are printed correctly
    assert captured.out.startswith(f"{description}... ")
    assert "done." in captured.out

def test_work_in_progress_timing(capsys):
    # Test that the elapsed time is approximately reflected in the output
    with work_in_progress("Timer test"):
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    # Extract the numeric value from the output string "done. (X.XXs)"
    import re
    match = re.search(r"\(([\d.]+)s\)", captured.out)
    assert match is not None
    elapsed_time = float(match.group(1))
    assert elapsed_time >= 0.1

def test_work_in_progress_default_desc(capsys):
    # Test usage without providing a description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert captured.out.startswith("Work in progress... ")

def test_work_in_progress_exception_handling(capsys):
    # Ensure that even if an error occurs, the print statements are handled 
    # (Note: current implementation doesn't have try/finally, so we check behavior)
    try:
        with work_in_progress("Error test"):
            raise ValueError("Test Error")
    except ValueError:
        pass

    captured = capsys.readouterr()
    assert captured.out.startswith("Error test... ")
```


# LLM-generated content at query #7
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and basic execution
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if duration is present (regex-like check for float pattern)
    assert "(" in captured.out and "s)" in captured.out

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(desc=custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing logic using mock to ensure it calculates delta correctly
    with patch("time.time") as mock_time:
        # Mock first call (start) and second call (end)
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50
        assert "done. (5.50s)" in captured.out

    # Test execution within block actually runs
    executed = False
    with work_in_progress("Running block"):
        executed = True
    
    assert executed is True
    captured = capsys.readouterr()
    assert "Running block... done." in captured.out
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

    # Test timing logic (mocking time to ensure predictable output)
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timer test"):
            pass
        captured = capsys.readouterr()
        assert "Timer test... done. (5.50s)" in captured.out

    # Test that it works within a block containing code
    with work_in_progress("Block test"):
        x = 1 + 1
        assert x == 2
    captured = capsys.readouterr()
    assert "Block test... done." in captured.out
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
    assert "... " in captured.out
    assert "done." in captured.out

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... " in captured.out
    assert "done." in captured.out

    # Test timing logic (mocking time to ensure deterministic output)
    with patch("time.time") as mock_time:
        # Set start time and end time
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timed task"):
            pass
        
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50s
        assert "done. (5.50s)" in captured.out

    # Test that it works as a decorator
    @work_in_progress("Decorator test")
    def dummy_func():
        return True

    dummy_func()
    captured = capsys.readouterr()
    assert "Decorator test... " in captured.out
    assert "done." in captured.out
```


# LLM-generated content at query #10
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test usage as a context manager
    with work_in_progress("Testing task"):
        pass
    
    captured = capsys.readouterr()
    assert "Testing task... done." in captured.out
    assert "(0.00s)" in captured.out or "(0.01s)" in captured.out

    # Test usage with a custom description and simulated delay
    with patch("time.time") as mock_time:
        # Mock start time at 100.0 and end time at 105.5
        mock_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Long task"):
            pass
            
        captured = capsys.readouterr()
        assert "Long task... done. (5.50s)" in captured.out

    # Test default description
    with work_in_progress():
        pass
        
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
```


# LLM-generated content at query #11
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and execution flow
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check that a duration-like string is present (e.g., (0.00s))
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out) or re.search(r"\(\d+\.\d+s\)", captured.out)

    # Test custom description and simulated delay
    custom_desc = "Testing custom desc"
    with patch("time.time") as mock_time:
        # Mock time to simulate 1.5 second passage
        mock_time.side_effect = [100.0, 101.5]
        with work_in_progress(custom_desc):
            pass
            
    captured = capsys.readouterr()
    assert f"{custom_desc}... " in captured.out
    assert "done. (1.50s)" in captured.out

    # Test with context manager block containing logic
    with patch("time.time") as mock_time:
        mock_time.side_effect = [200.0, 200.1]
        with work_in_progress("Task"):
            # Ensure the block executes
            result = True
        assert result is True
    
    captured = capsys.readouterr()
    assert "Task... done. (0.10s)" in captured.out
```


# LLM-generated content at query #12
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test standard usage via context manager
    with work_in_progress("Testing task"):
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    assert "Testing task... done." in captured.out
    assert "(0.1" in captured.out or "done." in captured.out

    # Test with default description
    with work_in_progress():
        pass
        
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out

    # Test using it as a decorator
    @work_in_progress("Decorator test")
    def decorated_func():
        time.sleep(0.1)
        return True

    decorated_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out

    # Test timing precision (ensure it records some duration)
    with patch('time.time') as mocked_time:
        # Mock time to return 100.0 then 102.5
        mocked_time.side_effect = [100.0, 102.5]
        with work_in_progress("Timer test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timer test... done. (2.50s)" in captured.out
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from io import StringIO
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and basic execution
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if duration format exists (regex-like check for 0.00s)
    assert "(0.00s)" in captured.out or "(0.01s)" in captured.out

    # Test custom description and simulated delay
    custom_desc = "Testing custom message"
    with patch("time.time") as mock_time:
        # Simulate 2.5 seconds passing
        mock_time.side_effect = [100.0, 102.5]
        with work_in_progress(custom_desc):
            pass
        
        captured = capsys.readouterr()
        assert f"{custom_desc}... done." in captured.out
        assert "(2.50s)" in captured.out

    # Test error handling (ensure print happens even if block fails)
    with pytest.raises(ValueError):
        with work_in_progress("Error test"):
            raise ValueError("Triggered error")
    
    captured = capsys.readouterrr() # Note: Using readouterr for the failed block
    # In context managers, if an exception occurs inside, 'done' is not printed 
    # unless handled. Based on provided code, it will raise BEFORE the 'done' print.
    # We verify only the start message was printed.
    assert "Error test... " in captured.out
```


# LLM-generated content at query #14
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and execution flow
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if duration is present (regex-like check for float)
    assert "(" in captured.out and "s)" in captured.out

    # Test custom description and timing simulation
    with patch("time.time") as mock_time:
        # Mock time to return 100.0 at start and 105.5 at end (5.5s duration)
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Custom Task"):
            pass
            
        captured = capsys.readouterr()
        assert "Custom Task... done. (5.50s)" in captured.out

    # Test with a small sleep to ensure time passes
    with work_in_progress("Sleeping task"):
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    assert "Sleeping task... done." in captured.out
```


# LLM-generated content at query #15
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test basic functionality with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "... " in captured.out
    assert "done." in captured.out

    # Test custom description and verify content
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... " in captured.out
    assert "done." in captured.out

    # Test timing logic by mocking time.time
    with patch("time.time") as mock_time:
        # Set start time to 100.0 and end time to 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timed task"):
            pass
        
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50s
        assert "done. (5.50s)" in captured.out

    # Test execution within context block
    with work_in_progress("Running logic"):
        x = 1 + 1
        assert x == 2
    
    captured = capsys.readouterr()
    assert "Running logic... done." in captured.out
```


# LLM-generated content at query #16
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and basic functionality
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if it contains a time pattern like (0.00s)
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out) or re.search(r"\(\d+\.\d+s\)", captured.out)

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(desc=custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test with simulated time delay to ensure timer logic works
    with patch("time.time") as mocked_time:
        # Mock start time as 100.0 and end time as 105.5
        mocked_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timed task"):
            pass
        
        captured = capsys.readouterr()
        assert "Timed task... done. (5.50s)" in captured.out

    # Test context manager behavior with error (ensure it still prints start, though end might not trigger)
    # Note: The current implementation does not use try/finally, 
    # so an exception will prevent the 'done' message from printing.
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        try:
            with work_in_progress("Failing task"):
                raise ValueError("Test Error")
        except ValueError:
            pass
        
        output = fake_out.getvalue()
        assert "Failing task... " in output
        assert "done." not in output
```


# LLM-generated content at query #17
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
    assert "(0.00s)" in captured.out or "(0.01s)" in captured.out

    # Test custom description and verify timing logic
    with patch("time.time") as mock_time:
        # Mock time to return 100.0 then 105.5 (5.5 second difference)
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Custom Task"):
            pass
        
        captured = capsys.readouterr()
        assert "Custom Task... done. (5.50s)" in captured.out

    # Test with a sleeping period to ensure real time-based flow works
    with work_in_progress("Sleeping task"):
        time.sleep(0.1)
    captured = capsys.readouterr()
    assert "Sleeping task... done." in captured.out
```


# LLM-generated content at query #18
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
    assert "done." in captured.out

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... " in captured.out
    assert "done." in captured.out

    # Test timing accuracy (mocking time.time)
    with patch("time.time") as mock_time:
        # Return 100.0 first, then 102.5 to simulate 2.5 seconds passing
        mock_time.side_effect = [100.0, 102.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "(2.50s)" in captured.out

    # Test integration with a block of code
    with patch("time.time") as mock_time:
        mock_time.side_effect = [100.0, 101.123]
        with work_in_progress("Executing task"):
            time.sleep(0.01) # Minimal sleep to ensure logic flow
        
        captured = capsys.readouterr()
        # Should round 1.123 down/up to 1.12 based on format .2f
        assert "Executing task... done. (1.12s)" in captured.out
```


# LLM-generated content at query #19
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if duration format is present (e.g., (0.00s))
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test context manager with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test functionality with simulated delay to verify timing logic
    with patch("time.time") as mock_time:
        # Mock time.time() to return 100.0 then 105.5 (5.5s elapsed)
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test as a decorator on a function
    @work_in_progress("Decorator test")
    def decorated_func():
        return True

    decorated_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #20
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
    # Check if time is present (regex-like check for duration format)
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out.replace("done.", "done."))

    # Test custom description and manual sleep to verify timing logic
    with work_in_progress("Custom Task"):
        time.sleep(0.1)
    captured = capsys.readouterr()
    assert "Custom Task... done." in captured.out
    assert "0.1" in captured.out or "0.0" in captured.out

    # Test precise output format with patch to control time
    with patch('time.time') as mocked_time:
        mocked_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timed Task"):
            pass
        captured = capsys.readouterr()
        assert "Timed Task... done. (5.50s)" in captured.out

def test_work_in_progress_exception(capsys):
    # Ensure that the context manager handles exceptions and still prints the start message
    # Note: In the current implementation, if an exception occurs inside the block, 
    # the 'done' part is never reached because the generator exits.
    try:
        with work_in_progress("Failing Task"):
            raise ValueError("Test Error")
    except ValueError:
        pass

    captured = capsys.readouterr()
    assert "Failing Task... " in captured.out
    assert "done." not in captured.out
```


# LLM-generated content at query #21
#--------------------------

```python
import time
from io import StringIO
import pytest

def test_work_in_progress(capsys):
    description = "Test task"
    duration = 0.1
    
    with work_in_progress(description):
        time.sleep(duration)
    
    captured = capsys.readouterr()
    output = captured.out
    
    # Check if description and completion message exist
    assert f"{description}... " in output
    assert "done." in output
    
    # Extract the time from the output string to verify it's approximately correct
    # Format: "... done. (X.XXs)"
    time_part = output.split("done. (")[1].split("s)")[0]
    execution_time = float(time_part)
    
    assert execution_time >= duration
    assert execution_time < duration + 0.5

def test_work_in_progress_default():
    captured = capsys.readouterr()
    with work_in_progress():
        pass
    
    output = capsys.readouterr().out
    assert "Work in progress... done." in output
```


# LLM-generated content at query #22
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with a custom description
    desc = "Testing task"
    with work_in_progress(desc):
        pass
    
    captured = capsys.readouterr()
    assert desc in captured.out
    assert "done." in captured.out
    assert "s)" in captured.out

    # Test default description and execution delay simulation
    with patch("time.time") as mock_time:
        # Simulate start time at 100.0 and end time at 102.5
        mock_time.side_effect = [100.0, 102.5]
        with work_in_progress():
            pass
        
        captured = capsys.readouterr()
        assert "Work in progress..." in captured.out
        assert "(2.50s)" in captured.out

    # Test with a block of code that actually sleeps
    import time as real_time
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        with work_in_progress("Sleep test"):
            real_time.sleep(0.1)
        output = fake_out.getvalue()
        assert "Sleep test... done." in output
```


# LLM-generated content at query #23
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with a custom description
    description = "Testing task"
    with work_in_progress(description):
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    
    # Check if the start message is present
    assert captured.out.startswith(f"{description}... ")
    # Check if the end message is present and contains "done."
    assert "done." in captured.out
    # Check if the duration is captured (approximate check for non-zero time)
    import re
    match = re.search(r"\((\d+\.\d+)s\)", captured.out)
    assert match is not None
    assert float(match.group(1)) >= 0.1

def test_work_in_progress_default_description(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... " in captured.out
    assert "done." in captured.out

def test_work_in_progress_exception_handling(capsys):
    # The current implementation does not have a try/finally block, 
    # so an exception will prevent the "done" message from printing.
    # This test documents the current behavior.
    try:
        with work_in_progress("Failing task"):
            raise ValueError("Test Error")
    except ValueError:
        pass

    captured = capsys.readouterr()
    assert "Failing task... " in captured.out
    assert "done." not in captured.out
```


# LLM-generated content at query #24
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test basic functionality and description printing
    description = "Testing task"
    with work_in_progress(description):
        pass
    
    captured = capsys.readouterr()
    assert description in captured.out
    assert "done." in captured.out

    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out

    # Test timing calculation accuracy using patch
    with patch("time.time") as mock_time:
        # Mock start time at 100.0 and end time at 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timer test"):
            pass
        
        captured = capsys.readouterr()
        # 105.5 - 100.0 = 5.50
        assert "done. (5.50s)" in captured.out

    # Test that it works as a decorator
    @work_in_progress("Decorator test")
    def dummy_func():
        return True

    dummy_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #25
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and execution flow
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if it contains a duration pattern like (0.00s)
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test custom description and timing simulation
    with patch("time.time") as mock_time:
        # Mock start time as 100.0 and end time as 105.5
        mock_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Processing data"):
            pass
            
        captured = capsys.readouterr()
        assert "Processing data... done. (5.50s)" in captured.out

    # Test with a block of code that actually runs
    with patch("time.sleep", return_value=None):
        with work_in_progress("Sleeping task"):
            time.sleep(0.1)
        captured = capsys.readouterr()
        assert "Sleeping task... done." in captured.out
```


# LLM-generated content at query #26
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with custom description
    with work_in_progress("Testing task"):
        pass
    
    captured = capsys.readouterr()
    assert "Testing task... done." in captured.out
    # Check if duration is present (regex for float)
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out) or re.search(r"\(\d+\.\d+s\)", captured.out)

def test_work_in_progress_default_desc(capsys):
    # Test context manager usage with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out

def test_work_in_progress_timing(capsys):
    # Test if the timer actually tracks elapsed time
    delay = 0.5
    with work_in_progress("Delay test"):
        time.sleep(delay)
    
    captured = capsys.readouterr()
    # Extract number from (X.XXs)
    match = re.search(r"\(([\d.]+)s\)", captured.out)
    assert match is not None
    elapsed = float(match.group(1))
    assert elapsed >= delay

import re
```


# LLM-generated content at query #27
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and execution flow
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if time is present (regex for float pattern)
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out) or re.search(r"\(\d+\.\d+s\)", captured.out)

    # Test custom description and simulated delay
    with patch('time.time', side_effect=[100.0, 102.5]):
        with work_in_progress("Processing"):
            pass
        
        captured = capsys.readouterr()
        assert "Processing... done. (2.50s)" in captured.out

    # Test context manager behavior with code execution
    execution_flag = []
    with work_in_progress("Running task"):
        execution_flag.append(True)
    
    assert execution_flag[0] is True
    captured = capsys.readouterr()
    assert "Running task... done." in captured.out
```


# LLM-generated content at query #28
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
    # Check if time is present (regex for float pattern)
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out.replace("done.", "done."))

    # Test with custom description
    custom_desc = "Custom Task"
    with work_in_progress(desc=custom_desc):
        pass
        
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

def test_work_in_progress_execution_time(capsys):
    # Test that time elapsed is reflected accurately
    with patch('time.time') as mocked_time:
        # Mock start time as 100.0 and end time as 105.5
        mocked_time.side_effect = [100.0, 105.5]
        
        with work_in_progress("Timing test"):
            pass
            
    captured = capsys.readouterr()
    assert "Timing test... done. (5.50s)" in captured.out

def test_work_in_progress_exception_handling():
    # Ensure the context manager doesn't suppress exceptions
    with pytest.raises(ValueError):
        with work_in_progress("Error test"):
            raise ValueError("Test Error")
```


# LLM-generated content at query #29
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
    assert "done." in captured.out

    # Test with custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... " in captured.out
    assert "done." in captured.out

    # Test timing accuracy (using mock to control time)
    with patch('time.time') as mocked_time:
        mocked_time.side_effect = [100.0, 105.5]  # Start at 100s, end at 105.5s
        with work_in_progress("Timer test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timer test... done. (5.50s)" in captured.out

    # Test that it works as a decorator
    @work_in_progress("Decorator test")
    def dummy_func():
        return True

    dummy_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #30
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with a custom description
    description = "Testing task"
    with work_in_progress(description):
        # Simulate some work
        pass
    
    captured = capsys.readouterr()
    # Verify the output contains the description and the 'done' suffix
    assert description in captured.out
    assert "done." in captured.out
    assert "s)" in captured.out

def test_work_in_progress_default_desc(capsys):
    # Test context manager usage with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress" in captured.out

def test_work_in_progress_timing(capsys):
    # Test that the timing logic is functioning by mocking time.time
    # We simulate a 2-second delay
    with patch('time.time') as mocked_time:
        mocked_time.side_effect = [100.0, 102.5]
        
        with work_in_progress("Timed task"):
            pass
            
        captured = capsys.readouterr()
        # Check if the duration (2.50s) is correctly formatted in output
        assert "done. (2.50s)" in captured.out

def test_work_in_progress_exception(capsys):
    # Verify that the context manager behaves correctly even if an error occurs
    # Note: The current implementation does not have a try/finally block, 
    # so it will raise the exception to the caller.
    with pytest.raises(ValueError):
        with work_in_progress("Error task"):
            raise ValueError("Something went wrong")
```


# LLM-generated content at query #31
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
    # Check if time format exists (e.g., (0.00s))
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

def test_work_in_progress_custom_desc():
    desc = "Custom description"
    with work_in_progress(desc):
        pass
    
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with work_in_progress(desc):
            pass
        output = fake_out.getvalue()
        assert output.startswith(f"{desc}... ")
        assert "done." in output

def test_work_in_progress_execution_time():
    # Mock time to ensure deterministic duration
    with patch('time.time') as mocked_time:
        # Simulate 2 seconds passing
        mocked_time.side_effect = [100.0, 102.5]
        
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            with work_in_progress("Timing test"):
                pass
            output = fake_out.getvalue()
            assert "done. (2.50s)" in output

def test_work_in_progress_exception_handling():
    # Ensure that even if an error occurs, we can verify behavior 
    # Note: The current implementation doesn't have a try/finally block,
    # so the 'done' message won't print if an exception is raised.
    # This test verifies the current actual behavior.
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        try:
            with work_in_progress("Error test"):
                raise ValueError("Trigger error")
        except ValueError:
            pass
        
        output = fake_out.getvalue()
        assert "Error test... " in output
        assert "done." not in output
```


# LLM-generated content at query #32
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with custom description
    description = "Testing task"
    with work_in_progress(description):
        pass
    
    captured = capsys.readouterr()
    # Check if the start and end messages are printed correctly
    # We use regex-like check for time as it varies
    assert captured.out.startswith(f"{description}... ")
    assert "done." in captured.out
    assert "s)" in captured.out

    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... " in captured.out

def test_work_in_progress_execution_time(capsys):
    # Test if the elapsed time is approximately correct
    with work_in_progress("Timer test"):
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    # Extract the duration part from "done. (X.XXs)"
    # The time should be at least 0.1s
    import re
    match = re.search(r"\((.*)s\)", captured.out)
    assert match is not None
    duration = float(match.group(1))
    assert duration >= 0.1

def test_work_in_progress_exception(capsys):
    # Ensure that if an exception occurs, the context manager doesn't swallow it
    # Note: The current implementation does NOT have a try/finally block,
    # so an exception will crash the block before printing "done."
    # This test documents the current behavior.
    with pytest.raises(ValueError):
        with work_in_progress("Error test"):
            raise ValueError("Triggered error")
    
    captured = capsys.readouterr()
    assert "Error test... " in captured.out
    assert "done." not in captured.out
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest
from unittest.mock import patch
import io

def test_work_in_progress(capsys):
    # Test context manager usage and output format
    description = "Testing task"
    
    with work_in_progress(description):
        pass
    
    captured = capsys.readouterr()
    
    # Check if description is printed correctly
    assert captured.out.startswith(f"{description}... ")
    # Check if completion message exists and contains 'done.'
    assert "done." in captured.out
    # Check if time format (e.g., (0.00s)) is present using regex-like check
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out.replace("done.", "done. ")) or \
           re.search(r"\(\d+\.\d+s\)", captured.out)

def test_work_in_progress_default_desc(capsys):
    # Test with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... " in captured.out

def test_work_in_progress_execution_time(capsys):
    # Test if the time elapsed is reflected (simulating a delay)
    with patch('time.time') as mock_time:
        # Mocking time.time() to return 100.0 then 102.5
        mock_time.side_effect = [100.0, 102.5]
        
        with work_in_progress("Timing test"):
            pass
            
        captured = capsys.readouterr()
        assert "done. (2.50s)" in captured.out

def test_work_in_progress_exception_handling(capsys):
    # Ensure the context manager doesn't swallow exceptions 
    # (Though the current implementation actually fails to print 'done.' if an exception occurs,
    # we verify that the exception still propagates)
    with pytest.raises(ValueError):
        with work_in_progress("Error test"):
            raise ValueError("Triggered error")
    
    captured = capsys.readouterr()
    # The 'done.' part should NOT be printed because the yield was interrupted
    assert "Error test... " in captured.out
    assert "done." not in captured.out
```


# LLM-generated content at query #34
#--------------------------

```python
import time
from io import StringIO
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and basic execution
    with work_in_progress() as _:
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if time is recorded (at least 0.1s approx)
    assert "(0.1" in captured.out or "(0.0" in captured.out

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc) as _:
        time.sleep(0.05)
        
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test with precise time mocking to ensure formatting is correct
    with patch("time.time") as mock_time:
        # Mock start time at 100.0 and end time at 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Mocked task"):
            pass
        
        captured = capsys.readouterr()
        assert "Mocked task... done. (5.50s)" in captured.out

def test_work_in_progress_exception(capsys):
    # Ensure the context manager handles errors if they occur within the block
    # Note: The current implementation does NOT have a try/finally, 
    # so an exception will bubble up and skip the 'done' print.
    # This test verifies existing behavior.
    try:
        with work_in_progress("Error task"):
            raise ValueError("Test Error")
    except ValueError:
        pass

    captured = capsys.readouterr()
    assert "Error task... " in captured.out
    assert "done." not in captured.out
```


# LLM-generated content at query #35
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test basic functionality and output format
    description = "Testing task"
    with work_in_progress(description):
        pass
    
    captured = capsys.readouterr()
    # Check if the start and end messages are present
    assert f"{description}... " in captured.out
    assert "done. (" in captured.out
    assert "s)" in captured.out

    # Test with default description
    with work_in_progress():
        pass
    
    captured_default = capsys.readouterr()
    assert "Work in progress... " in captured_default.out

    # Test timing calculation accuracy using mock
    # We mock time.time to return specific values: 100.0 then 105.5 (duration 5.5)
    with patch("time.time", side_effect=[100.0, 105.5]):
        with work_in_progress("Timed task"):
            pass
        
        captured_timed = capsys.readouterr()
        assert "done. (5.50s)" in captured_timed.out

    # Test that the context manager handles exceptions without swallowing them
    # but still prints the start message
    with patch("time.time", side_effect=[10.0, 11.0]):
        try:
            with work_in_progress("Error task"):
                raise ValueError("Test error")
        except ValueError as e:
            assert str(e) == "Test error"
        
        captured_error = capsys.readouterr()
        assert "Error task... " in captured_error.out
```


# LLM-generated content at query #36
#--------------------------

```python
import unittest.mock as mock
import time
import io
import pytest

def test_work_in_progress():
    # Test default description and execution flow
    with mock.patch('sys.stdout', new=io.StringIO()) as fake_out:
        with work_in_progress():
            time.sleep(0.1)
        output = fake_out.getvalue()
        assert "Work in progress... " in output
        assert "done." in output
        # Verify that a time duration is printed (e.g., "0.10s" or similar)
        assert "(0." in output or "(1." in output

    # Test custom description
    custom_desc = "Custom Task"
    with mock.patch('sys.stdout', new=io.StringIO()) as fake_out:
        with work_in_progress(desc=custom_desc):
            time.sleep(0.05)
        output = fake_out.getvalue()
        assert f"{custom_desc}... " in output
        assert "done." in output

    # Test timing accuracy (rough check)
    with mock.patch('sys.stdout', new=io.StringIO()) as fake_out:
        start = time.time()
        with work_in_progress("Timing test"):
            duration = 0.2
            time.sleep(duration)
        output = fake_out.getvalue()
        # Extract the seconds value from the string (e.g., "(0.21s)")
        import re
        match = re.search(r'\(([\d.]+)s\)', output)
        assert match is not None
        captured_duration = float(match.group(1))
        assert captured_duration >= duration - 0.05
```


# LLM-generated content at query #37
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test basic functionality and output format
    description = "Testing task"
    with work_in_progress(description):
        pass
    
    captured = capsys.readouterr()
    # Check if the start and end messages are printed
    assert description in captured.out
    assert "done." in captured.out
    # Check for time format (e.g., (0.00s))
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out) or re.search(r"\(\d+\.\d+s\)", captured.out)

def test_work_in_progress_timing(capsys):
    # Test if the elapsed time reflects actual sleep duration
    description = "Sleeping task"
    sleep_duration = 0.5
    
    with work_in_progress(description):
        time.sleep(sleep_duration)
    
    captured = capsys.readouterr()
    # Extract the number from the parenthesis using regex
    import re
    match = re.search(r"\(([\d.]+)s\)", captured.out)
    assert match is not None
    
    elapsed_time = float(match.group(1))
    # Allow for some margin of error in execution time
    assert elapsed_time >= sleep_duration - 0.1

def test_work_in_progress_default_desc(capsys):
    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... " in captured.out
```


# LLM-generated content at query #38
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
    
    # Test custom description and timing logic simulation
    custom_desc = "Processing data"
    with patch("time.time") as mock_time:
        # Simulate 2.5 seconds passing
        mock_time.side_effect = [100.0, 102.5]
        with work_in_progress(custom_desc):
            pass
            
    captured = capsys.readouterr()
    assert f"{custom_desc}... done. (2.50s)" in captured.out

def test_work_in_progress_execution():
    # Test that the code block inside the context manager actually executes
    executed = False
    with work_in_progress("Test execution"):
        executed = True
    assert executed is True
```


# LLM-generated content at query #39
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
    assert "done." in captured.out

    # Test with custom description
    custom_desc = "Custom Task"
    with work_in_progress(custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... " in captured.out
    assert "done." in captured.out

    # Test timing logic (mocking time to ensure predictable output)
    with patch("time.time") as mock_time:
        # Start at 100.0, end at 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        # Check if duration is calculated correctly (5.50s)
        assert "(5.50s)" in captured.out
```


# LLM-generated content at query #40
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with a custom description
    description = "Testing WIP"
    with work_in_progress(description):
        pass
    
    captured = capsys.readouterr()
    # Check if the start and end messages are printed correctly
    # Note: time is variable, so we check for prefix and suffix patterns
    assert captured.out.startswith(f"{description}... ")
    assert "done." in captured.out
    assert "s)" in captured.out

    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert captured.out.startswith("Work in progress... ")
    assert "done." in captured.out

    # Test execution time measurement with a sleep delay
    with patch('time.time') as mocked_time:
        # Mocking time to return specific values: start at 100.0, end at 105.5
        mocked_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timed task"):
            pass
        
        captured = capsys.readouterr()
        # Expected: "Timed task... done. (5.50s)"
        assert "Timed task... done. (5.50s)" in captured.out

def test_work_in_progress_exception(capsys):
    # Ensure that the context manager prints the start but handles exceptions
    # The current implementation does not use a try/finally block, 
    # so an exception will bubble up and prevent "done" from printing.
    with pytest.raises(ValueError):
        with work_in_progress("Failing task"):
            raise ValueError("Failure")
            
    captured = capsys.readouterr()
    assert captured.out.startswith("Failing task... ")
    assert "done." not in captured.out
```


# LLM-generated content at query #41
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with custom description
    description = "Testing task"
    with work_in_progress(description):
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    
    # Check if the start and end messages are printed correctly
    assert captured.out.startswith(f"{description}... ")
    assert "done." in captured.out
    # Verify that a duration (s) is present in the output
    import re
    assert re.search(r"\( \d+\.\d+s\)", captured.out) or re.search(r"\(\d+\.\d+s\)", captured.out)

def test_work_in_progress_default_desc(capsys):
    # Test context manager usage with default description
    with work_in_progress():
        time.sleep(0.05)
        
    captured = capsys.readouterr()
    assert captured.out.startswith("Work in progress... ")
    assert "done." in captured.out

def test_work_in_progress_execution_flow(capsys):
    # Test that the code inside the block actually executes
    executed = False
    with work_in_progress("Execution test"):
        executed = True
    
    captured = capsys.readouterr()
    assert executed is True
    assert "Execution test... " in captured.out
```


# LLM-generated content at query #42
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with custom description
    with work_in_progress("Testing WIP"):
        pass
    
    captured = capsys.readouterr()
    assert "Testing WIP... done." in captured.out
    assert "(0.00s)" in captured.out or "(0.01s)" in captured.out

    # Test context manager usage with default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out

    # Test timing logic by mocking time.time
    with patch("time.time") as mock_time:
        # Set start time to 100.0 and end time to 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterr()
        assert "Timing test... done. (5.50s)" in captured.out

    # Test that the block actually executes
    executed = False
    with work_in_progress("Execution check"):
        executed = True
    
    assert executed is True
```


# LLM-generated content at query #43
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
    # Check if duration format exists (e.g., (0.00s))
    import re
    assert re.search(r"\(0\.00s\)", captured.out)

    # Test custom description
    custom_desc = "Custom Task"
    with work_in_progress(desc=custom_desc):
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test with simulated time delay to verify timing logic
    with patch("time.time") as mock_time:
        # Mock start time at 100.0 and end time at 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timed Task"):
            pass
        
        captured = capsys.readouterr()
        assert "Timed Task... done. (5.50s)" in captured.out

    # Test with actual sleep to ensure real-world behavior
    with patch("sys.stdout") as mock_stdout:
        # We use a spy/mock approach for stdout to verify the exact sequence if needed, 
        # but capsys is cleaner for checking content.
        pass

def test_work_in_progress_integration(capsys):
    # Test that it works inside a function decorated as a decorator
    @work_in_progress("Decorated Function")
    def dummy_func():
        return True

    result = dummy_func()
    captured = capsys.readouterr()
    
    assert result is True
    assert "Decorated Function... done." in captured.out
```


# LLM-generated content at query #44
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and execution flow
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    assert "(s)" in captured.out

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc) as _:
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test timing accuracy (mocking time to ensure deterministic output)
    start_time = 100.0
    end_time = 105.5
    with patch('time.time', side_effect=[start_time, end_time]):
        with work_in_progress("Timing test") as _:
            pass
        
        captured = capsys.readouterr()
        # check if the duration is correctly formatted to 2 decimal places
        assert "Timing test... done. (5.50s)" in captured.out

    # Test that it works as a decorator
    @work_in_progress("Decorator test")
    def dummy_func():
        return True

    dummy_func()
    captured = capsys.readouterr()
    assert "Decorator test... done." in captured.out
```


# LLM-generated content at query #45
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    description = "Testing task"
    
    with work_in_progress(description):
        # Simulate some work
        pass

    captured = capsys.readouterr()
    
    # Verify the start and end messages are present
    assert f"{description}... " in captured.out
    assert "done." in captured.out
    # Check if the time format (seconds) exists in the output via regex-like check
    import re
    assert re.search(r"done\. \(\d+\.\d+s\)", captured.out)

def test_work_in_progress_default():
    captured = io.StringIO()
    with patch('sys.stdout', new=captured):
        with work_in_progress():
            pass
        output = captured.getvalue()
        assert "Work in progress... " in output
        assert "done." in output

def test_work_in_progress_execution_time_logic():
    # Mock time to ensure predictable duration for testing logic
    with patch('time.time') as mocked_time:
        mocked_time.side_effect = [100.0, 105.5]  # Start at 100, end at 105.5 (diff 5.5)
        
        captured = io.StringIO()
        with patch('sys.stdout', new=captured):
            with work_in_progress("Timer test"):
                pass
            output = captured.getvalue()
            
        assert "Timer test... " in output
        assert "done. (5.50s)" in output
```


# LLM-generated content at query #46
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and execution flow
    with work_in_progress() as _:
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Check if duration is present (regex for float)
    import re
    assert re.search(r"\(\d+\.\d{2}s\)", captured.out)

    # Test custom description
    custom_desc = "Processing data"
    with work_in_progress(custom_desc) as _:
        pass
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

    # Test with a simulated delay to ensure time is actually measured
    with patch("time.time") as mock_time:
        # Mock start time and end time (2 seconds apart)
        mock_time.side_effect = [100.0, 102.5]
        with work_in_progress("Timed task"):
            pass
        
        captured = capsys.readouterr()
        assert "Timed task... done. (2.50s)" in captured.out

    # Test with actual sleep to ensure functionality doesn't break execution
    with work_in_progress("Sleeping"):
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    assert "Sleeping... done." in captured.out
```


# LLM-generated content at query #47
#--------------------------

```python
import io
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test context manager usage with custom description
    desc = "Testing WIP"
    with work_in_progress(desc):
        pass
    
    captured = capsys.readouterr()
    assert desc + "... " in captured.out
    assert "done." in captured.out
    # Check if duration pattern exists (e.g., (0.00s))
    import re
    assert re.search(r"done\. \(\d+\.\d+s\)", captured.out)

    # Test default description
    with work_in_progress():
        pass
    
    captured = capsys.readouterr()
    assert "Work in progress... " in captured.out

    # Test execution time calculation accuracy using mock
    with patch("time.time") as mock_time:
        # Mock start at 100.0 and end at 105.5
        mock_time.side_effect = [100.0, 105.5]
        with work_in_progress("Timing test"):
            pass
        
        captured = capsys.readouterraderr().out if hasattr(capsys, 'readouterraderr') else capsys.readouterr().out
        assert "done. (5.50s)" in captured

def test_work_in_progress_exception_handling(capsys):
    # Ensure that even if an error occurs, the context manager doesn't 
    # swallow it, though we check how print behaves.
    # Note: The current implementation does not have a try/finally block,
    # so an exception will prevent the "done" message from printing.
    
    with pytest.raises(ValueError):
        with work_in_progress("Failing task"):
            raise ValueError("Error!")

    captured = capsys.readouterr()
    assert "Failing task... " in captured.out
    assert "done." not in captured.out
```


# LLM-generated content at query #48
#--------------------------

```python
import time
from io import StringIO
from unittest.mock import patch

def test_work_in_progress(capsys):
    # Test default description and basic functionality
    with work_in_progress() as _:
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    assert "Work in progress... done." in captured.out
    # Verify that it contains a duration (regex-like check for digits and 's')
    import re
    assert re.search(r"\(\d+\.\d+s\)", captured.out)

    # Test custom description
    custom_desc = "Custom Task"
    with work_in_progress(desc=custom_desc):
        time.sleep(0.1)
    
    captured = capsys.readouterr()
    assert f"{custom_desc}... done." in captured.out

def test_work_in_progress_timing_accuracy(capsys):
    # Use patch to control time and ensure predictable output
    start_time = 100.0
    end_time = 105.5
    
    with patch('time.time', side_effect=[start_time, end_time]):
        with work_in_progress("Timing test"):
            pass
            
    captured = capsys.readouterr()
    # 105.5 - 100.0 = 5.50
    assert "done. (5.50s)" in captured.out
```


