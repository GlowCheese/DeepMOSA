####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... ")
    assert output.endswith("s)\n")
    assert "done." in output
    
    # Test 2: Verify timing is measured
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    elapsed = end - start
    output = captured_output.getvalue()
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... ")
    
    # Test 4: Function decorator usage
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorator test")
    def test_function():
        time.sleep(0.1)
        return 42
    
    result = test_function()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == 42
    assert output.startswith("Decorator test... ")
    assert "done." in output
    
    # Test 5: Empty task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress(""):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("... ")


# LLM-generated content at query #2
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    elapsed = end - start
    output = captured_output.getvalue()
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done.")
    
    # Test 4: Empty task (very short duration)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Quick task... done. (0.00s)" in output
    
    # Test 5: Custom description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Custom description"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Custom description... done.")


# LLM-generated content at query #3
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    time_str = output.split("(")[1].split("s)")[0]
    time_val = float(time_str)
    assert 0.19 <= time_val <= 0.21
    
    # Test 3: Test with default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Test as decorator
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorator test")
    def test_function():
        time.sleep(0.05)
        return "result"
    
    result = test_function()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == "result"
    assert output.startswith("Decorator test... done. (0.")
    
    # Test 5: Test that output is flushed immediately
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Flush test"):
        # Immediately check if first part was printed
        sys.stdout = sys.__stdout__
        partial_output = captured_output.getvalue()
        assert partial_output == "Flush test... "
        # Restore capture for the rest
        captured_output = io.StringIO()
        sys.stdout = captured_output
    
    sys.stdout = sys.__stdout__


# LLM-generated content at query #4
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    time_str = output.split("(")[1].split("s)")[0]
    time_val = float(time_str)
    assert 0.19 <= time_val <= 0.21  # Allow small timing variance
    
    # Test 3: Test with default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Test with empty task (should still work)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress(""):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("... done. (0.")
    
    # Test 5: Test that output is flushed immediately
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Flush test"):
        # Check that description is printed before the block executes
        partial_output = captured_output.getvalue()
        assert partial_output == "Flush test... "
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    
    # Test 6: Test with very fast operation
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Fast task"):
        pass  # No operation
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "done. (0.00s)" in output or "done. (0.0" in output


# LLM-generated content at query #5
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    elapsed = end - start
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done.")
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Short task... done. (0.00s)" in output
    
    # Test 5: Decorator usage simulation
    def dummy_function():
        time.sleep(0.1)
        return "result"
    
    # Simulate decorator behavior
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress
    def decorated_function():
        time.sleep(0.1)
        return "result"
    
    # The decorator returns a context manager, not a decorated function
    # This test verifies the context manager aspect
    with work_in_progress("Decorator test"):
        result = dummy_function()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Decorator test... done. (0.")
    assert result == "result"


# LLM-generated content at query #6
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 3: Function decorator usage
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.08)
        return "result"
    
    result = test_function()
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == "result"
    assert output.startswith("Decorated function... done. (0.")
    
    # Test 4: Verify timing is reasonable
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    elapsed = end - start
    output = captured_output.getvalue()
    
    # Extract time from output
    import re
    match = re.search(r'\((\d+\.\d+)s\)', output)
    assert match is not None
    reported_time = float(match.group(1))
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 5: Multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.05)
    
    with work_in_progress("Second task"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (0.")
    assert lines[1].startswith("Second task... done. (0.")


# LLM-generated content at query #7
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    elapsed = end - start
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Empty task (very short duration)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Empty task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Empty task... done. (0.00s)" in output or output.startswith("Empty task... done. (0.0")
    
    # Test 5: Decorator usage simulation
    def dummy_function():
        time.sleep(0.05)
        return "result"
    
    # Simulate decorator behavior
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Function execution"):
        result = dummy_function()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == "result"
    assert output.startswith("Function execution... done. (0.0")


# LLM-generated content at query #8
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    time_str = output.split("(")[1].split("s)")[0]
    time_value = float(time_str)
    assert 0.19 <= time_value <= 0.21
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Empty task (very short execution)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Quick task... done. (0.")
    assert float(output.split("(")[1].split("s)")[0]) < 0.1
    
    # Test 5: Custom description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Custom description"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Custom description... done. (0.")
    
    # Test 6: Verify output format
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Format test"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "... " in output
    assert "done. (" in output
    assert "s)" in output
    assert output.count(".") == 3  # ellipsis, decimal point, and period after done


# LLM-generated content at query #9
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert output.startswith("Test task... ")
    assert output.endswith("s)\n")
    assert "done. (" in output
    
    # Test 2: Verify timing is measured
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timed task"):
        time.sleep(0.2)
    end = time.time()
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    # Extract time from output
    time_str = output.split("done. (")[1].split("s)")[0]
    measured_time = float(time_str)
    
    # Verify measured time is reasonable (within 0.05s of actual elapsed time)
    actual_time = end - start
    assert abs(measured_time - actual_time) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert output.startswith("Work in progress... ")
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert "Quick task... done. (" in output
    time_str = output.split("done. (")[1].split("s)")[0]
    measured_time = float(time_str)
    assert measured_time >= 0
    
    # Test 5: Decorator usage simulation
    def simulate_decorator():
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        @work_in_progress("Decorated function")
        def dummy_func():
            time.sleep(0.05)
            return "result"
        
        result = dummy_func()
        output = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        
        assert result == "result"
        assert "Decorated function... done. (" in output
    
    simulate_decorator()


# LLM-generated content at query #10
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Timing test... done. (0.2" in output
    
    # Test 3: Test with default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Test as decorator
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorator test")
    def dummy_function():
        time.sleep(0.1)
        return "result"
    
    result = dummy_function()
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == "result"
    assert output.startswith("Decorator test... done. (0.")
    
    # Test 5: Test flush parameter works (output appears immediately)
    import threading
    
    output_parts = []
    
    def capture_during_execution():
        captured = io.StringIO()
        sys.stdout = captured
        with work_in_progress("Flush test"):
            time.sleep(0.1)
            # Check that "Flush test... " was printed before sleep completed
            output_parts.append(captured.getvalue())
        sys.stdout = sys.__stdout__
    
    thread = threading.Thread(target=capture_during_execution)
    thread.start()
    thread.join()
    
    assert len(output_parts) == 1
    assert output_parts[0].startswith("Flush test... ")
    assert "done" not in output_parts[0]  # "done" should not appear yet


# LLM-generated content at query #11
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    elapsed = end - start
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done.")
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Quick task... done." in output
    assert "s)" in output
    
    # Test 5: Custom description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    custom_desc = "Custom task description"
    with work_in_progress(custom_desc):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith(f"{custom_desc}...")
    
    # Test 6: Verify output format
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Format test"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    
    # Check the exact format pattern
    assert "... " in output
    assert "done. (" in output
    assert "s)" in output
    assert output.count(".") >= 3  # Should have ellipsis and decimal point


# LLM-generated content at query #12
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 3: Function decorator usage
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.08)
        return "result"
    
    result = test_func()
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == "result"
    assert output.startswith("Decorated function... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 4: Verify timing is reasonable
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    elapsed = end - start
    output = captured_output.getvalue()
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 5: Multiple consecutive uses
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.05)
    
    with work_in_progress("Second task"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (0.")
    assert lines[1].startswith("Second task... done. (0.")


# LLM-generated content at query #13
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    elapsed = end - start
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Short task... done. (0.00s)" in output
    
    # Test 5: Multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.05)
    
    with work_in_progress("Second task"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (0.0")
    assert lines[1].startswith("Second task... done. (0.0")


# LLM-generated content at query #14
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    output = captured_output.getvalue()
    assert output.startswith("Test task... ")
    assert output.endswith("s)\n")
    assert "done." in output
    
    # Test 2: Verify timing is measured
    sys.stdout = sys.__stdout__
    import time
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    # The elapsed time should be approximately 0.2 seconds
    elapsed = end - start
    assert elapsed >= 0.2
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... ")
    
    # Test 4: Function decorator usage
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorator test")
    def test_function():
        time.sleep(0.05)
        return "result"
    
    result = test_function()
    assert result == "result"
    output = captured_output.getvalue()
    assert "Decorator test... done." in output
    
    # Test 5: Multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.05)
    
    with work_in_progress("Second task"):
        time.sleep(0.05)
    
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... ")
    assert lines[1].startswith("Second task... ")
    
    # Restore stdout
    sys.stdout = sys.__stdout__


# LLM-generated content at query #15
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    time_str = output.split("(")[1].split("s)")[0]
    time_measured = float(time_str)
    assert 0.19 <= time_measured <= 0.21
    
    # Test 3: Test with default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Test as decorator
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorator test")
    def dummy_function():
        time.sleep(0.05)
        return 42
    
    result = dummy_function()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == 42
    assert output.startswith("Decorator test... done. (0.")
    
    # Test 5: Test with empty task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    time_str = output.split("(")[1].split("s)")[0]
    time_measured = float(time_str)
    assert time_measured < 0.1
    
    # Test 6: Verify flush parameter works (output appears immediately)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Flush test"):
        # Check that initial message appears before execution
        partial_output = captured_output.getvalue()
        assert partial_output == "Flush test... "
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Flush test... done. (0.")


# LLM-generated content at query #16
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (")
    assert output.endswith("s)\n")
    time_value = float(output.split("(")[1].split("s)")[0])
    assert time_value >= 0.1
    
    # Test 2: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (")
    
    # Test 3: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Quick task... done. (")
    time_value = float(output.split("(")[1].split("s)")[0])
    assert time_value >= 0
    
    # Test 4: Decorator usage simulation
    def dummy_function():
        time.sleep(0.05)
        return "result"
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @contextlib.contextmanager
    def mock_work_in_progress(desc="Work in progress"):
        print(desc + "... ", end='', flush=True)
        begin_time = time.time()
        yield
        time_consumed = time.time() - begin_time
        print(f"done. ({time_consumed:.2f}s)")
    
    with mock_work_in_progress("Decorator test"):
        result = dummy_function()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Decorator test... done. (")
    assert result == "result"
    
    # Test 5: Multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.03)
    
    with work_in_progress("Second task"):
        time.sleep(0.03)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (")
    assert lines[1].startswith("Second task... done. (")


# LLM-generated content at query #17
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    # Check that time is approximately 0.2 seconds
    time_str = output.split("(")[1].split("s)")[0]
    time_val = float(time_str[:-1])
    assert 0.15 <= time_val <= 0.25
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done.")
    
    # Test 4: Empty task (very short execution)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Quick task... done. (0.00s)")
    
    # Test 5: Decorator usage simulation
    def dummy_function():
        time.sleep(0.05)
        return "result"
    
    # Simulate decorator behavior
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress
    def decorated_function():
        time.sleep(0.05)
        return "result"
    
    # The decorator returns a context manager, not a function
    # So we need to use it as context manager
    with work_in_progress("Decorator test"):
        result = dummy_function()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Decorator test... done. (0.0")
    assert result == "result"
    
    # Test 6: Multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.05)
    
    with work_in_progress("Second task"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done.")
    assert lines[1].startswith("Second task... done.")
    
    # Test 7: Custom description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Custom description here"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Custom description here... done." in output


# LLM-generated content at query #18
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Short task... done. (0.00s)" in output or "Short task... done. (0.0" in output
    
    # Test 3: Test with default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.0")
    
    # Test 4: Test as decorator
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorator test")
    def decorated_function():
        time.sleep(0.1)
        return "result"
    
    result = decorated_function()
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == "result"
    assert output.startswith("Decorator test... done. (0.")
    
    # Test 5: Test multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.05)
    
    with work_in_progress("Second task"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (0.0")
    assert lines[1].startswith("Second task... done. (0.0")


# LLM-generated content at query #19
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    import pickle
    import tempfile
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Function decorator functionality
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.05)
        return "result"
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    result = dummy_function()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == "result"
    assert output.startswith("Decorated function... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 4: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    time_str = output.split("(")[1].split("s)")[0]
    time_measured = float(time_str)
    assert 0.19 <= time_measured <= 0.21
    
    # Test 5: Multiple nested contexts
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Outer"):
        with work_in_progress("Inner"):
            time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("Outer... done. (0.")
    assert lines[1].startswith("Inner... done. (0.")
    
    # Test 6: Example from docstring - file operations
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp:
        test_data = {"key": "value", "number": 42}
        pickle.dump(test_data, tmp)
        tmp_path = tmp.name
    
    try:
        # Test loading file example
        @work_in_progress("Loading file")
        def load_file(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        loaded_data = load_file(tmp_path)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        assert loaded_data == test_data
        assert output.startswith("Loading file... done. (0.")
        
        # Test saving file example
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        with work_in_progress("Saving file"):
            with open(tmp_path, "wb") as f:
                pickle.dump(test_data, f)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        assert output.startswith("Saving file... done. (0.")
        
    finally:
        import os
        os.unlink(tmp_path)
    
    # Test 7: Verify flush parameter works (output appears immediately)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Flush test"):
        # Print something during execution to verify initial message appeared
        print("Working...", end='', flush=True)
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Flush test... Working...done. (0.")


# LLM-generated content at query #20
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Short task... done. (0.0" in output or "Short task... done. (0.05" in output
    
    # Test 3: Test with default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Test that yield allows code execution
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Yield test") as wip:
        result = 2 + 2
        assert result == 4
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Yield test... done. (0." in output
    
    # Test 5: Test with longer running task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Longer task"):
        time.sleep(0.2)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Longer task... done. (0.2" in output or "Longer task... done. (0.19" in output or "Longer task... done. (0.20" in output


# LLM-generated content at query #21
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    elapsed = end - start
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done.")
    
    # Test 4: Empty task (no sleep)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Quick task... done. (0.")
    assert float(output.split("(")[1].split("s)")[0]) < 0.1
    
    # Test 5: Custom description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    custom_desc = "Custom task description"
    with work_in_progress(custom_desc):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith(f"{custom_desc}... done.")


# LLM-generated content at query #22
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert output.startswith("Test task... ")
    assert "done. (" in output
    assert output.endswith("s)\n")
    assert float(output.split("(")[1].split("s)")[0]) >= 0.1
    
    # Test 2: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert output.startswith("Work in progress... ")
    assert "done. (" in output
    
    # Test 3: Function decorator usage
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.08)
        return "result"
    
    result = test_func()
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert result == "result"
    assert output.startswith("Decorated function... ")
    assert "done. (" in output
    assert float(output.split("(")[1].split("s)")[0]) >= 0.08
    
    # Test 4: Very short task timing
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        pass
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert output.startswith("Short task... ")
    assert "done. (" in output
    time_str = output.split("(")[1].split("s)")[0]
    time_val = float(time_str)
    assert time_val >= 0.0
    assert time_val < 0.1  # Should be very small
    
    # Test 5: Multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.03)
    
    with work_in_progress("Second task"):
        time.sleep(0.04)
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... ")
    assert lines[1].startswith("Second task... ")
    
    # Test 6: Custom description with special characters
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Task: Process data (123)"):
        time.sleep(0.02)
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert output.startswith("Task: Process data (123)... ")
    assert "done. (" in output


# LLM-generated content at query #23
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (")
    assert output.endswith("s)\n")
    assert float(output.split("(")[1].split("s)")[0]) >= 0.1
    
    # Test 2: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (")
    
    # Test 3: Function decorator usage
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorated function")
    def sample_function():
        time.sleep(0.08)
        return "result"
    
    result = sample_function()
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == "result"
    assert output.startswith("Decorated function... done. (")
    
    # Test 4: Very short execution time
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Quick task... done. (")
    assert float(output.split("(")[1].split("s)")[0]) >= 0.0
    
    # Test 5: Multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.03)
    
    with work_in_progress("Second task"):
        time.sleep(0.04)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (")
    assert lines[1].startswith("Second task... done. (")
    
    # Test 6: Verify flush parameter works (output appears immediately)
    import threading
    
    def delayed_task():
        time.sleep(0.1)
        print("Task completed", end='', flush=True)
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Async test"):
        thread = threading.Thread(target=delayed_task)
        thread.start()
        thread.join()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Async test... done. (" in output
    assert "Task completed" in output


# LLM-generated content at query #24
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start_time = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end_time = time.time()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    elapsed = end_time - start_time
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done.")
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Short task... done." in output
    
    # Test 5: Custom description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Custom description"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Custom description... done.")


# LLM-generated content at query #25
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    time_measured = float(output.split("(")[1].split("s)")[0])
    actual_time = end - start
    
    # Allow small tolerance for timing differences
    assert abs(time_measured - actual_time) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (")
    
    # Test 4: Function decorator usage
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorator test")
    def test_function():
        time.sleep(0.1)
        return "result"
    
    result = test_function()
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    
    assert result == "result"
    assert output.startswith("Decorator test... done. (")
    
    # Test 5: Empty task (very short duration)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    time_measured = float(output.split("(")[1].split("s)")[0])
    assert time_measured >= 0
    assert time_measured < 0.1  # Should be very fast


# LLM-generated content at query #26
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    time_str = output.split("(")[1].split("s)")[0]
    time_measured = float(time_str)
    assert 0.19 <= time_measured <= 0.21
    
    # Test 3: Test with default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Test as decorator
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorator test")
    def dummy_function():
        time.sleep(0.05)
        return "result"
    
    result = dummy_function()
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == "result"
    assert output.startswith("Decorator test... done. (0.")
    
    # Test 5: Test with empty task (very fast execution)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Quick task... done. (0.")
    time_str = output.split("(")[1].split("s)")[0]
    time_measured = float(time_str)
    assert time_measured < 0.01
    
    # Test 6: Test output format
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Format test"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "... " in output
    assert "done. (" in output
    assert "s)" in output
    assert output.count(".") == 3  # ellipsis + decimal point + period after done


# LLM-generated content at query #27
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    import pickle
    import tempfile
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Function decorator functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.05)
        return "result"
    
    result = test_func()
    sys.stdout = sys.__stdout__
    assert result == "result"
    output = captured_output.getvalue()
    assert output.startswith("Decorated function... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 4: Verify timing accuracy
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    time_str = output.split("(")[1].split("s)")[0]
    time_measured = float(time_str)
    assert 0.19 <= time_measured <= 0.25
    
    # Test 5: Multiple nested contexts
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Outer task"):
        time.sleep(0.05)
        with work_in_progress("Inner task"):
            time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("Outer task... done. (0.")
    assert lines[1].startswith("Inner task... done. (0.")
    
    # Test 6: Example from docstring - file operations
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp:
        tmp_path = tmp.name
        test_data = {"key": "value", "number": 42}
        pickle.dump(test_data, tmp)
    
    try:
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        @work_in_progress("Loading file")
        def load_file(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        
        loaded_data = load_file(tmp_path)
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        assert loaded_data == test_data
        assert output.startswith("Loading file... done. (0.")
        assert output.endswith("s)\n")
        
        # Test saving file example
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        with work_in_progress("Saving file"):
            with open(tmp_path, "wb") as f:
                pickle.dump(test_data, f)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        assert output.startswith("Saving file... done. (0.")
        assert output.endswith("s)\n")
    finally:
        import os
        os.unlink(tmp_path)
    
    # Test 7: Verify flush parameter works (output appears immediately)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Flush test"):
        # The "..." should appear immediately due to flush=True
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Flush test... done." in output


# LLM-generated content at query #28
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    output = captured_output.getvalue()
    assert output.startswith("Test task... ")
    assert output.endswith("s)\n")
    assert "done." in output
    
    # Test 2: Verify timing is measured
    sys.stdout = sys.__stdout__
    import time
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    # The elapsed time should be approximately 0.2 seconds
    elapsed = end - start
    assert elapsed >= 0.2
    
    # Test 3: Test with default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... ")
    
    # Test 4: Test as decorator
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorator test")
    def dummy_function():
        time.sleep(0.05)
        return 42
    
    result = dummy_function()
    assert result == 42
    
    output = captured_output.getvalue()
    assert "Decorator test... done." in output
    
    # Test 5: Test flush behavior (output appears immediately)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Immediate output"):
        # The "..." should appear immediately, not after completion
        immediate_output = captured_output.getvalue()
        assert "Immediate output... " in immediate_output
    
    # Restore stdout
    sys.stdout = sys.__stdout__


# LLM-generated content at query #29
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 3: Decorator usage simulation
    def dummy_function():
        time.sleep(0.02)
        return "result"
    
    # Simulate decorator behavior
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Function execution")
    def decorated_function():
        return dummy_function()
    
    # Actually apply the decorator and call
    result = decorated_function()
    sys.stdout = sys.__stdout__
    
    assert result == "result"
    output = captured_output.getvalue()
    assert output.startswith("Function execution... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 4: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.15)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    elapsed = end - start
    output = captured_output.getvalue()
    
    # Extract time from output
    import re
    match = re.search(r'\((\d+\.\d+)s\)', output)
    assert match is not None
    reported_time = float(match.group(1))
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 5: Multiple consecutive uses
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.03)
    
    with work_in_progress("Second task"):
        time.sleep(0.03)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (0.")
    assert lines[1].startswith("Second task... done. (0.")


# LLM-generated content at query #30
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 3: Function decorator usage
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.08)
        return 42
    
    result = test_function()
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    
    assert result == 42
    assert output.startswith("Decorated function... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 4: Multiple context managers
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.03)
    
    with work_in_progress("Second task"):
        time.sleep(0.04)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (0.")
    assert lines[1].startswith("Second task... done. (0.")
    
    # Test 5: Very short execution time
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Quick task... done. (0.")
    assert "s)\n" in output


# LLM-generated content at query #31
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    time_str = output.split("(")[1].split("s)")[0]
    time_measured = float(time_str)
    assert 0.19 <= time_measured <= 0.21
    
    # Test 3: Test with default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (")
    
    # Test 4: Test as decorator
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorator test")
    def test_function():
        time.sleep(0.05)
        return "result"
    
    result = test_function()
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == "result"
    assert output.startswith("Decorator test... done. (0.")
    
    # Test 5: Test multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.05)
    
    with work_in_progress("Second task"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (0.")
    assert lines[1].startswith("Second task... done. (0.")
    
    # Test 6: Test that output is flushed immediately
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Flush test"):
        # The "..." should appear immediately
        partial_output = captured_output.getvalue()
        assert "Flush test... " in partial_output
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "done." in output


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    output = captured_output.getvalue()
    assert output.startswith("Test task... ")
    assert output.endswith("s)\n")
    assert "done." in output
    
    # Test 2: Verify timing is measured
    sys.stdout = sys.__stdout__
    
    start_time = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end_time = time.time()
    
    # Should take at least 0.2 seconds
    assert end_time - start_time >= 0.2
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... ")
    
    # Test 4: Function decorator usage
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorator test")
    def dummy_function():
        time.sleep(0.1)
        return "result"
    
    result = dummy_function()
    assert result == "result"
    output = captured_output.getvalue()
    assert "Decorator test... done." in output
    
    # Test 5: Multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.05)
    
    with work_in_progress("Second task"):
        time.sleep(0.05)
    
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... ")
    assert lines[1].startswith("Second task... ")
    
    # Test 6: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        pass
    
    output = captured_output.getvalue()
    assert "Short task... done." in output
    # Time should be very small but formatted correctly
    assert "(0.00s)" in output or "(0.0" in output
    
    sys.stdout = sys.__stdout__


# LLM-generated content at query #2
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    output = captured_output.getvalue()
    assert output.startswith("Test task... ")
    assert output.endswith("done. (0.10s)\n") or "done. (0.1" in output
    
    sys.stdout = sys.__stdout__
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    output = captured_output.getvalue()
    assert "Timing test... done. (0.20s)" in output or "Timing test... done. (0.2" in output
    
    sys.stdout = sys.__stdout__
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... ")
    assert "done." in output
    
    sys.stdout = sys.__stdout__
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    output = captured_output.getvalue()
    assert "Quick task... done." in output
    assert "s)" in output
    
    sys.stdout = sys.__stdout__
    
    # Test 5: Decorator usage simulation
    def mock_function():
        time.sleep(0.05)
        return "result"
    
    # Simulate decorator behavior
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Mock function"):
        result = mock_function()
    
    assert result == "result"
    output = captured_output.getvalue()
    assert "Mock function... done." in output
    
    sys.stdout = sys.__stdout__


# LLM-generated content at query #3
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    elapsed = end - start
    output = captured_output.getvalue()
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Short task... done. (0.00s)" in output or "Short task... done. (0.0" in output
    
    # Test 5: Decorator usage simulation
    def dummy_function():
        time.sleep(0.1)
        return "result"
    
    # Simulate decorator behavior
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress
    def decorated_function():
        time.sleep(0.1)
        return "result"
    
    # The decorator returns a context manager, not a decorated function
    # This test shows the actual usage pattern from the docstring
    sys.stdout = sys.__stdout__


# LLM-generated content at query #4
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    time_str = output.split("(")[1].split("s)")[0]
    time_value = float(time_str)
    assert 0.19 <= time_value <= 0.25
    
    # Test 3: Test with default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (")
    
    # Test 4: Test with empty task (should still work)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress(""):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("... done. (")
    
    # Test 5: Test that output is flushed immediately
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Flush test"):
        # Check that description is printed before the block executes
        partial_output = captured_output.getvalue()
        assert "Flush test... " in partial_output
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    
    # Test 6: Test with very short execution time
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Quick task... done. (0.00s)" in output or "Quick task... done. (0.0" in output


# LLM-generated content at query #5
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    import pickle
    import tempfile
    
    # Test 1: Context manager functionality with default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.1)
    
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (")
    assert output.endswith("s)\n")
    
    # Test 2: Context manager with custom description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Custom task"):
        time.sleep(0.05)
    
    output = captured_output.getvalue()
    assert output.startswith("Custom task... done. (")
    assert output.endswith("s)\n")
    
    # Test 3: Function decorator usage
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.08)
        return "result"
    
    result = test_function()
    assert result == "result"
    
    output = captured_output.getvalue()
    assert output.startswith("Decorated function... done. (")
    assert output.endswith("s)\n")
    
    # Test 4: Verify timing is reasonable
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    output = captured_output.getvalue()
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    actual_time = end - start
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - actual_time) < 0.05
    
    # Test 5: Multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.05)
    
    with work_in_progress("Second task"):
        time.sleep(0.05)
    
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (")
    assert lines[1].startswith("Second task... done. (")
    
    # Test 6: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        pass
    
    output = captured_output.getvalue()
    assert "Short task... done. (0.00s)" in output
    
    # Restore stdout
    sys.stdout = sys.__stdout__


# LLM-generated content at query #6
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... ")
    assert output.endswith("s)\n")
    assert "done." in output
    
    # Test 2: Verify timing is measured
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timed task"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    elapsed = end - start
    output = captured_output.getvalue()
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... ")
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "done." in output
    assert "Quick task... " in output
    
    # Test 5: Exception handling - should still print done message
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        with work_in_progress("Task with exception"):
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Task with exception... " in output
    assert "done." in output


# LLM-generated content at query #7
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... ")
    assert output.endswith("done. (")
    assert "s)" in output
    
    # Test 2: Verify timing is measured
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start_time = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end_time = time.time()
    
    sys.stdout = sys.__stdout__
    elapsed = end_time - start_time
    output = captured_output.getvalue()
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... ")
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Short task... done. (0.00s)" in output or "Short task... done. (0.0" in output
    
    # Test 5: Custom description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Custom task description"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Custom task description... ")
    assert "done. (" in output


# LLM-generated content at query #8
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Short task... done. (0.0" in output or "Short task... done. (0.05" in output
    
    # Test 3: Test with default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Test as decorator
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorator test")
    def decorated_function():
        time.sleep(0.05)
        return "result"
    
    result = decorated_function()
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    
    assert result == "result"
    assert "Decorator test... done. (0." in output
    
    # Test 5: Test multiple consecutive uses
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.02)
    
    with work_in_progress("Second task"):
        time.sleep(0.03)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (0.0")
    assert lines[1].startswith("Second task... done. (0.0")
    
    # Test 6: Test that output is flushed immediately
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Flush test"):
        # The "..." should appear immediately
        immediate_output = captured_output.getvalue()
        assert immediate_output == "Flush test... "
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    
    # Test 7: Test with exception inside context (should still print done)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        with work_in_progress("Exception test"):
            time.sleep(0.01)
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Exception test... done. (0.0" in output


# LLM-generated content at query #9
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    import pickle
    import tempfile
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Function decorator usage
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.05)
        return "result"
    
    result = test_func()
    sys.stdout = sys.__stdout__
    assert result == "result"
    output = captured_output.getvalue()
    assert output.startswith("Decorated function... done. (0.")
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Verify timing accuracy
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    elapsed = end - start
    output = captured_output.getvalue()
    
    # Extract time from output
    import re
    match = re.search(r'\((\d+\.\d+)s\)', output)
    assert match is not None
    reported_time = float(match.group(1))
    
    # Allow small timing differences due to system overhead
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 5: Multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.05)
    
    with work_in_progress("Second task"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (0.")
    assert lines[1].startswith("Second task... done. (0.")
    
    # Test 6: Exception handling within context
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        with work_in_progress("Task with exception"):
            time.sleep(0.05)
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    # Should still print the completion message even with exception
    assert "done." in output
    
    # Test 7: File operation example from docstring
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp:
        tmp_path = tmp.name
        test_data = {"key": "value", "number": 42}
        pickle.dump(test_data, tmp)
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Loading file")
    def load_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    
    loaded_data = load_file(tmp_path)
    sys.stdout = sys.__stdout__
    
    assert loaded_data == test_data
    output = captured_output.getvalue()
    assert output.startswith("Loading file... done. (")
    
    import os
    os.unlink(tmp_path)


# LLM-generated content at query #10
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert output.startswith("Test task... ")
    assert output.endswith("s)\n")
    assert "done. (" in output
    
    # Test 2: Verify timing is measured
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    # Extract time from output
    time_str = output.split("done. (")[1].split("s)")[0]
    measured_time = float(time_str)
    
    # Verify measured time is reasonable (within 0.05s of actual elapsed time)
    actual_time = end - start
    assert abs(measured_time - actual_time) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert output.startswith("Work in progress... ")
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        pass
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert "Short task... done. (" in output
    assert output.endswith("s)\n")
    
    # Test 5: Multiple nested contexts
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Outer task"):
        with work_in_progress("Inner task"):
            time.sleep(0.05)
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("Outer task... ")
    assert lines[1].startswith("Inner task... ")


# LLM-generated content at query #11
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    output = captured_output.getvalue()
    assert output.startswith("Test task... ")
    assert output.endswith("s)\n")
    assert "done." in output
    
    # Test 2: Verify timing is measured
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    output = captured_output.getvalue()
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    time_value = float(time_str[:-1])  # Remove 's' and convert to float
    assert time_value >= 0.19  # Should be close to 0.2s
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... ")
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        pass
    
    output = captured_output.getvalue()
    assert "Short task... done." in output
    
    # Test 5: Decorator usage simulation
    @work_in_progress
    def dummy_function():
        time.sleep(0.05)
        return "result"
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    result = dummy_function("Function test")
    
    output = captured_output.getvalue()
    assert "Function test... done." in output
    assert result == "result"
    
    # Restore stdout
    sys.stdout = sys.__stdout__


# LLM-generated content at query #12
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is reasonable
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    time_str = output.split("(")[1].split("s)")[0]
    time_val = float(time_str)
    assert 0.19 <= time_val <= 0.25
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (")
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Quick task... done. (" in output
    
    # Test 5: Decorator usage simulation
    def mock_function():
        time.sleep(0.05)
        return "result"
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    # Simulate decorator behavior
    @work_in_progress
    def decorated_function():
        return mock_function()
    
    # Actually call the decorated function
    context_manager = decorated_function("Decorated task")
    with context_manager:
        result = mock_function()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Decorated task... done. (" in output
    assert result == "result"


# LLM-generated content at query #13
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    elapsed = end - start
    output = captured_output.getvalue()
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Short task... done. (0.00s)" in output or "Short task... done. (0.0" in output
    
    # Test 5: Decorator usage simulation
    def dummy_function():
        time.sleep(0.05)
        return "result"
    
    # Simulate decorator behavior
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress
    def decorated_function():
        time.sleep(0.05)
        return "result"
    
    # The decorator returns a context manager, not a decorated function
    # This test verifies the context manager works as a decorator
    with work_in_progress("Decorator test"):
        result = dummy_function()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Decorator test... done. (0.")
    assert result == "result"


# LLM-generated content at query #14
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    output = captured_output.getvalue()
    assert output.startswith("Test task... ")
    assert output.endswith("done. (0.10s)\n") or output.endswith("done. (0.1")
    
    sys.stdout = sys.__stdout__
    
    # Test 2: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... ")
    assert "done." in output
    
    sys.stdout = sys.__stdout__
    
    # Test 3: Function decorator usage
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.08)
        return "result"
    
    result = test_func()
    assert result == "result"
    
    output = captured_output.getvalue()
    assert output.startswith("Decorated function... ")
    assert "done." in output
    
    sys.stdout = sys.__stdout__
    
    # Test 4: Verify timing is measured
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.15)
    end = time.time()
    
    output = captured_output.getvalue()
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    measured_time = float(time_str)
    
    # Verify measured time is close to actual elapsed time
    actual_time = end - start
    assert abs(measured_time - actual_time) < 0.05
    
    sys.stdout = sys.__stdout__
    
    # Test 5: Multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.05)
    
    with work_in_progress("Second task"):
        time.sleep(0.05)
    
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... ")
    assert lines[1].startswith("Second task... ")
    
    sys.stdout = sys.__stdout__


# LLM-generated content at query #15
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert output.startswith("Test task... ")
    assert "done. (" in output
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timed task"):
        time.sleep(0.2)
    end = time.time()
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    measured_time = float(time_str)
    
    # Verify measured time is reasonable (within 0.05s of actual elapsed time)
    actual_time = end - start
    assert abs(measured_time - actual_time) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert output.startswith("Work in progress... ")
    
    # Test 4: Empty task (no yield block)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert "Quick task... done. (" in output
    assert float(output.split("(")[1].split("s)")[0]) < 0.1
    
    # Test 5: Multiple sequential uses
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.05)
    
    with work_in_progress("Second task"):
        time.sleep(0.05)
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... ")
    assert lines[1].startswith("Second task... ")
    
    # Test 6: Exception handling - timing should still be measured
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        with work_in_progress("Task with exception"):
            time.sleep(0.1)
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    # Should still print the completion message even with exception
    assert "Task with exception... done. (" in output


# LLM-generated content at query #16
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    import pickle
    import tempfile
    
    # Test 1: Context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Function decorator functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.05)
        return "result"
    
    result = dummy_function()
    sys.stdout = sys.__stdout__
    assert result == "result"
    output = captured_output.getvalue()
    assert output.startswith("Decorated function... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 4: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    time_measured = float(time_str)
    assert 0.19 <= time_measured <= 0.21
    
    # Test 5: Multiple nested contexts
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Outer"):
        with work_in_progress("Inner"):
            time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("Outer... ")
    assert lines[1].startswith("Inner... ")
    
    # Test 6: Example from docstring - file operations
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        test_data = {"key": "value", "number": 42}
        pickle.dump(test_data, f)
        temp_path = f.name
    
    try:
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        @work_in_progress("Loading file")
        def load_file(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        
        loaded_data = load_file(temp_path)
        sys.stdout = sys.__stdout__
        
        assert loaded_data == test_data
        output = captured_output.getvalue()
        assert "Loading file... done." in output
        
    finally:
        import os
        os.unlink(temp_path)


# LLM-generated content at query #17
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start_time = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end_time = time.time()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    elapsed = end_time - start_time
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Short task... done. (0.00s)" in output
    
    # Test 5: Multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.05)
    
    with work_in_progress("Second task"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (0.0")
    assert lines[1].startswith("Second task... done. (0.0")


# LLM-generated content at query #18
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    import pickle
    import tempfile
    
    # Test 1: Context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Decorator functionality
    @work_in_progress("Decorator test")
    def dummy_function():
        time.sleep(0.05)
        return "result"
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    result = dummy_function()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == "result"
    assert output.startswith("Decorator test... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 4: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    time_str = output.split("(")[1].split("s)")[0]
    time_measured = float(time_str)
    assert 0.19 <= time_measured <= 0.21
    
    # Test 5: Multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.05)
    
    with work_in_progress("Second task"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (0.")
    assert lines[1].startswith("Second task... done. (0.")
    
    # Test 6: Example from docstring - file operations
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp:
        tmp_path = tmp.name
        test_data = {"key": "value", "number": 42}
        pickle.dump(test_data, tmp)
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Loading file")
    def load_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    
    loaded_data = load_file(tmp_path)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert loaded_data == test_data
    assert output.startswith("Loading file... done. (0.")
    assert output.endswith("s)\n")
    
    import os
    os.unlink(tmp_path)


# LLM-generated content at query #19
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    elapsed = end - start
    output = captured_output.getvalue()
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (")
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Short task... done. (0.00s)" in output or "Short task... done. (0.0" in output
    
    # Test 5: Decorator usage simulation
    def dummy_function():
        time.sleep(0.05)
        return "result"
    
    # Simulate decorator behavior
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Function execution"):
        result = dummy_function()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == "result"
    assert "Function execution... done. (" in output


# LLM-generated content at query #20
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (")
    assert output.endswith("s)\n")
    assert float(output.split("(")[1].split("s)")[0]) >= 0.1
    
    # Test 2: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (")
    
    # Test 3: Function decorator usage
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorator test")
    def test_function():
        time.sleep(0.08)
        return "result"
    
    result = test_function()
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == "result"
    assert output.startswith("Decorator test... done. (")
    
    # Test 4: Very short execution time
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Quick task... done. (")
    time_value = float(output.split("(")[1].split("s)")[0])
    assert time_value >= 0.0
    
    # Test 5: Multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.03)
    
    with work_in_progress("Second task"):
        time.sleep(0.04)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (")
    assert lines[1].startswith("Second task... done. (")
    
    # Test 6: Verify flush behavior (output appears immediately)
    import threading
    
    def delayed_task():
        time.sleep(0.1)
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    thread = threading.Thread(target=delayed_task)
    
    with work_in_progress("Threaded task"):
        thread.start()
        thread.join()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    # Should have printed description immediately (before sleep)
    assert output.startswith("Threaded task... ")


# LLM-generated content at query #21
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    output = captured_output.getvalue()
    assert output.startswith("Test task... ")
    assert output.endswith("done. (0.1")
    assert "s)" in output
    
    sys.stdout = sys.__stdout__
    
    # Test 2: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... ")
    assert "done." in output
    
    sys.stdout = sys.__stdout__
    
    # Test 3: Function decorator usage
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorator test")
    def test_function():
        time.sleep(0.05)
        return "result"
    
    result = test_function()
    assert result == "result"
    
    output = captured_output.getvalue()
    assert output.startswith("Decorator test... ")
    assert "done." in output
    
    sys.stdout = sys.__stdout__
    
    # Test 4: Verify timing is measured
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    output = captured_output.getvalue()
    # Check that time is formatted with 2 decimal places
    assert "done. (0.20" in output or "done. (0.19" in output or "done. (0.21" in output
    
    sys.stdout = sys.__stdout__
    
    # Test 5: Empty block execution
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Empty task"):
        pass
    
    output = captured_output.getvalue()
    assert output.startswith("Empty task... ")
    assert "done. (0.00" in output
    
    sys.stdout = sys.__stdout__


# LLM-generated content at query #22
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    time_str = output.split("(")[1].split("s)")[0]
    time_val = float(time_str)
    assert 0.19 <= time_val <= 0.21
    
    # Test 3: Test with default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Test as decorator
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorator test")
    def dummy_function():
        time.sleep(0.05)
        return 42
    
    result = dummy_function()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == 42
    assert output.startswith("Decorator test... done. (0.")
    
    # Test 5: Test with empty task (very short execution)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Quick task... done. (0.")
    time_str = output.split("(")[1].split("s)")[0]
    time_val = float(time_str)
    assert time_val < 0.01
    
    # Test 6: Test output format
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Format test"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "... " in output
    assert "done. (" in output
    assert "s)" in output
    assert output.count(".") == 3  # ellipsis, decimal point, period after done


# LLM-generated content at query #23
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (")
    assert output.endswith("s)\n")
    time_value = float(output.split("(")[1].split("s)")[0])
    assert time_value >= 0.1
    
    # Test 2: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (")
    
    # Test 3: Function decorator usage
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorator test")
    def test_function():
        time.sleep(0.08)
        return "result"
    
    result = test_function()
    sys.stdout = sys.__stdout__
    assert result == "result"
    output = captured_output.getvalue()
    assert "Decorator test... done. (" in output
    
    # Test 4: Very short task timing
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Short task... done. (0.00s)" in output
    
    # Test 5: Multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.03)
    
    with work_in_progress("Second task"):
        time.sleep(0.04)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (")
    assert lines[1].startswith("Second task... done. (")


# LLM-generated content at query #24
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... ")
    assert output.endswith("s)\n")
    assert "done." in output
    
    # Test 2: Verify timing is measured
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timed task"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    elapsed = end - start
    output = captured_output.getvalue()
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... ")
    
    # Test 4: Function decorator usage
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)
        return "result"
    
    result = dummy_function()
    
    sys.stdout = sys.__stdout__
    assert result == "result"
    output = captured_output.getvalue()
    assert "Decorated function... done." in output
    
    # Test 5: Empty task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress(""):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("... ")


# LLM-generated content at query #25
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    elapsed = end - start
    
    # Extract time from output
    import re
    match = re.search(r'\((\d+\.\d+)s\)', output)
    assert match is not None
    reported_time = float(match.group(1))
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Short task... done. (0.00s)" in output or "Short task... done. (0.0" in output
    
    # Test 5: Decorator usage simulation
    def dummy_function():
        time.sleep(0.05)
        return "result"
    
    # Simulate decorator behavior
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Loading data"):
        result = dummy_function()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == "result"
    assert "Loading data... done. (0.0" in output


# LLM-generated content at query #26
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert output.startswith("Test task... ")
    assert output.endswith("s)\n")
    assert "done. (" in output
    
    # Test 2: Verify timing is measured
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    time_value = float(time_str)
    assert time_value >= 0.19  # Should be close to 0.2s
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert output.startswith("Work in progress... ")
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert "Quick task... done. (" in output
    
    # Test 5: Decorator usage simulation
    def dummy_function():
        time.sleep(0.05)
        return "result"
    
    # Simulate decorator behavior
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Decorator test"):
        result = dummy_function()
    
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    
    assert result == "result"
    assert "Decorator test... done. (" in output


# LLM-generated content at query #27
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    time_str = output.split("(")[1].split("s)")[0]
    time_measured = float(time_str)
    assert 0.19 <= time_measured <= 0.21
    
    # Test 3: Test with default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Test as decorator
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorator test")
    def test_function():
        time.sleep(0.05)
        return "result"
    
    result = test_function()
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == "result"
    assert output.startswith("Decorator test... done. (0.")
    
    # Test 5: Test with empty task (very short execution)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Empty task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Empty task... done. (0.")
    time_str = output.split("(")[1].split("s)")[0]
    time_measured = float(time_str)
    assert time_measured < 0.01
    
    # Test 6: Verify flush parameter works (output appears immediately)
    import threading
    
    output_parts = []
    
    def capture_in_thread():
        captured = io.StringIO()
        sys.stdout = captured
        with work_in_progress("Thread test"):
            time.sleep(0.1)
        sys.stdout = sys.__stdout__
        output_parts.append(captured.getvalue())
    
    thread = threading.Thread(target=capture_in_thread)
    thread.start()
    thread.join()
    
    output = output_parts[0]
    assert output.startswith("Thread test... done. (0.")


# LLM-generated content at query #28
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    import pickle
    import tempfile
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Function decorator usage
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.05)
        return "result"
    
    result = test_func()
    sys.stdout = sys.__stdout__
    assert result == "result"
    output = captured_output.getvalue()
    assert output.startswith("Decorated function... done. (0.")
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    
    # Test 4: Verify timing accuracy
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    actual_duration = time.time() - start
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    
    # Extract time from output
    import re
    match = re.search(r'\((\d+\.\d+)s\)', output)
    assert match is not None
    reported_duration = float(match.group(1))
    
    # Allow small timing differences due to overhead
    assert abs(actual_duration - reported_duration) < 0.05
    
    # Test 5: Multiple sequential calls
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.05)
    
    with work_in_progress("Second task"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (0.")
    assert lines[1].startswith("Second task... done. (0.")
    
    # Test 6: Exception handling within context
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        with work_in_progress("Task with exception"):
            time.sleep(0.05)
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    # Should still print completion message even with exception
    assert "done. (0." in output
    
    # Test 7: Custom description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Custom description"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Custom description... done. (0.")


# LLM-generated content at query #29
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Verify timing is measured correctly
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    elapsed = end - start
    
    # Extract time from output
    time_str = output.split("(")[1].split("s)")[0]
    reported_time = float(time_str)
    
    # Allow small tolerance for timing differences
    assert abs(reported_time - elapsed) < 0.05
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done.")
    
    # Test 4: Very short task
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Short task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Short task... done. (0.00s)" in output
    
    # Test 5: Custom description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Custom description"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Custom description... done.")


# LLM-generated content at query #30
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    import pickle
    import tempfile
    
    # Test 1: Basic context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Function decorator functionality
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.05)
        return "result"
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    result = dummy_function()
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert result == "result"
    assert output.startswith("Decorated function... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 4: Custom description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Custom task"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Custom task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 5: Very short execution time
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Quick task"):
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Quick task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 6: Verify timing is reasonable
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    actual_duration = time.time() - start
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    
    # Extract time from output
    import re
    match = re.search(r'\((\d+\.\d+)s\)', output)
    assert match is not None
    reported_duration = float(match.group(1))
    
    # Allow small timing differences due to system overhead
    assert abs(actual_duration - reported_duration) < 0.05
    
    # Test 7: Output format verification
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Format test"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    
    # Check the exact format pattern
    assert "... " in output
    assert "done. (" in output
    assert "s)" in output
    assert output.count(".") >= 3  # Should have ellipsis and decimal point
    
    # Test 8: Multiple consecutive uses
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("First task"):
        time.sleep(0.05)
    
    with work_in_progress("Second task"):
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    lines = output.strip().split('\n')
    assert len(lines) == 2
    assert lines[0].startswith("First task... done. (0.")
    assert lines[1].startswith("Second task... done. (0.")


# LLM-generated content at query #31
#--------------------------

```python
def test_work_in_progress():
    import io
    import sys
    import pickle
    import tempfile
    
    # Test 1: Context manager functionality
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Test task"):
        time.sleep(0.1)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Test task... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 2: Function decorator functionality
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.05)
        return "result"
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    result = dummy_function()
    sys.stdout = sys.__stdout__
    
    assert result == "result"
    output = captured_output.getvalue()
    assert output.startswith("Decorated function... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 3: Default description
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress():
        time.sleep(0.05)
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output.startswith("Work in progress... done. (0.")
    assert output.endswith("s)\n")
    
    # Test 4: Verify flush behavior (output appears immediately)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    with work_in_progress("Immediate output"):
        # The "..." should appear immediately due to flush=True
        pass
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Immediate output... done." in output
    
    # Test 5: Real-world example - file operations
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp:
        tmp_path = tmp.name
        test_data = {"key": "value", "number": 42}
        pickle.dump(test_data, tmp)
    
    @work_in_progress("Loading pickle file")
    def load_pickle_file(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    loaded_data = load_pickle_file(tmp_path)
    sys.stdout = sys.__stdout__
    
    assert loaded_data == test_data
    output = captured_output.getvalue()
    assert "Loading pickle file... done." in output
    
    import os
    os.unlink(tmp_path)


