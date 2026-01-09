####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test"):
        time.sleep(1)


# LLM-generated content at query #2
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Testing work_in_progress"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #3
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import os
    import tempfile

    # Test with context manager
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pkl")
        obj = {"key": "value"}
        with work_in_progress("Saving file"):
            with open(path, "wb") as f:
                pickle.dump(obj, f)
        with work_in_progress("Loading file"):
            with open(path, "rb") as f:
                loaded_obj = pickle.load(f)
        assert loaded_obj == obj

    # Test with decorator
    @work_in_progress("Loading file")
    def load_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pkl")
        obj = {"key": "value"}
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        loaded_obj = load_file(path)
        assert loaded_obj == obj

    print("All tests passed.")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #4
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress():


# LLM-generated content at query #5
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress():


# LLM-generated content at query #6
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import tempfile
    import os

    # Test as a context manager
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        test_data = {'key': 'value'}
        pickle.dump(test_data, f)
        temp_path = f.name

    try:
        with work_in_progress("Loading file"):
            with open(temp_path, 'rb') as f:
                loaded_data = pickle.load(f)
        assert loaded_data == test_data, "Data mismatch after loading"

        # Test as a decorator
        @work_in_progress("Saving file")
        def save_file(path, data):
            with open(path, 'wb') as f:
                pickle.dump(data, f)

        new_temp_path = temp_path + '_new'
        save_file(new_temp_path, test_data)
        with open(new_temp_path, 'rb') as f:
            saved_data = pickle.load(f)
        assert saved_data == test_data, "Data mismatch after saving"

        os.unlink(new_temp_path)
    finally:
        os.unlink(temp_path)

    print("All tests passed.")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #7
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress():


# LLM-generated content at query #8
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import os
    import tempfile

    # Test with context manager
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        test_data = {'key': 'value'}
        pickle.dump(test_data, f)
        temp_path = f.name

    try:
        # Test with context manager
        with work_in_progress("Loading file"):
            with open(temp_path, 'rb') as f:
                loaded_data = pickle.load(f)
        assert loaded_data == test_data, "Data mismatch after loading"

        # Test with decorator
        @work_in_progress("Saving file")
        def save_file(path, data):
            with open(path, 'wb') as f:
                pickle.dump(data, f)

        new_path = temp_path + '.new'
        save_file(new_path, test_data)
        with open(new_path, 'rb') as f:
            reloaded_data = pickle.load(f)
        assert reloaded_data == test_data, "Data mismatch after saving"

        print("All tests passed.")
    finally:
        # Cleanup
        for path in [temp_path, new_path]:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #9
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress():


# LLM-generated content at query #10
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #11
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import os
    import tempfile

    # Test with context manager
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        test_data = {'key': 'value'}
        pickle.dump(test_data, f)
        temp_path = f.name

    try:
        with work_in_progress("Loading file"):
            with open(temp_path, 'rb') as f:
                loaded_data = pickle.load(f)
        assert loaded_data == test_data, "Loaded data should match saved data"

        with work_in_progress("Saving file"):
            with open(temp_path, 'wb') as f:
                pickle.dump(test_data, f)
    finally:
        os.unlink(temp_path)

    # Test with decorator
    @work_in_progress("Custom task")
    def custom_task():
        time.sleep(0.1)  # Simulate work

    custom_task()

    print("All tests passed!")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #12
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import tempfile
    import os

    # Test with function decorator
    @work_in_progress("Loading file")
    def load_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # Test with context manager
    def save_file(obj, path):
        with work_in_progress("Saving file"):
            with open(path, "wb") as f:
                pickle.dump(obj, f)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Test saving file
        test_obj = {"key": "value"}
        save_file(test_obj, tmp_path)

        # Test loading file
        loaded_obj = load_file(tmp_path)
        assert loaded_obj == test_obj, "Loaded object does not match saved object"
    finally:
        # Clean up
        os.remove(tmp_path)

    print("All tests passed.")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #13
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress():  
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second
    # The function should print "Test task... done. (1.00s)" or similar

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #14
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Testing work_in_progress"):
        time.sleep(1)
    # The output should be "Testing work_in_progress... done. (1.00s)"

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #15
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import tempfile
    import os

    # Test with function decorator
    @work_in_progress("Loading file")
    def load_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # Test with context manager
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        test_data = {"key": "value"}
        pickle.dump(test_data, f)
        temp_path = f.name

    try:
        # Test loading file
        loaded_data = load_file(temp_path)
        assert loaded_data == test_data, "Loaded data does not match original data"

        # Test saving file
        with work_in_progress("Saving file"):
            with open(temp_path, "wb") as f:
                pickle.dump(test_data, f)
    finally:
        os.unlink(temp_path)

    print("All tests passed!")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #16
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #17
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #18
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import os
    import tempfile

    # Test with function decorator
    @work_in_progress("Loading file")
    def load_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # Test with context manager
    def save_file(obj, path):
        with work_in_progress("Saving file"):
            with open(path, "wb") as f:
                pickle.dump(obj, f)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Test saving file
        test_obj = {"key": "value"}
        save_file(test_obj, tmp_path)

        # Test loading file
        loaded_obj = load_file(tmp_path)
        assert test_obj == loaded_obj, "Loaded object does not match saved object"
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)

    print("All tests passed!")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #19
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #20
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress():


# LLM-generated content at query #21
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #22
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress():


# LLM-generated content at query #23
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #24
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)
    print("Test passed!")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #25
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #26
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second
    # The output should be: "Test task... done. (1.00s)"

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #27
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import tempfile
    import os

    # Test with a function
    @work_in_progress("Loading file")
    def load_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # Test with a context manager
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
        obj = {"test": "data"}
        pickle.dump(obj, tmp)

    try:
        loaded_obj = load_file(tmp_path)
        assert loaded_obj == obj, "Loaded object does not match original"
    finally:
        os.unlink(tmp_path)

    # Test context manager
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with work_in_progress("Saving file"):
            with open(tmp_path, "wb") as f:
                pickle.dump(obj, f)
        with open(tmp_path, "rb") as f:
            loaded_obj = pickle.load(f)
        assert loaded_obj == obj, "Loaded object does not match original"
    finally:
        os.unlink(tmp_path)

    print("All tests passed.")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #28
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress():


# LLM-generated content at query #29
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #30
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second
    # The output should be: "Test task... done. (1.00s)"


# LLM-generated content at query #31
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress():  
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #32
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Testing work_in_progress"):
        time.sleep(1)
    print("Test passed!")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #33
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import os
    import tempfile

    # Test with context manager
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pkl")
        obj = {"key": "value"}
        with work_in_progress("Saving file"):
            with open(path, "wb") as f:
                pickle.dump(obj, f)
        # Check if file exists
        assert os.path.exists(path)
        # Load and check content
        with open(path, "rb") as f:
            loaded_obj = pickle.load(f)
        assert loaded_obj == obj

    # Test with decorator
    @work_in_progress("Loading file")
    def load_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pkl")
        obj = {"key": "value"}
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        loaded_obj = load_file(path)
        assert loaded_obj == obj

    print("All tests passed!")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #34
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test"):
        time.sleep(1)
    # Should print "Test... done. (1.00s)"


# LLM-generated content at query #35
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import os
    import tempfile

    # Test with a function
    @work_in_progress("Loading file")
    def load_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # Test with a context manager
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
        obj = {"test": "data"}
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f)

    try:
        loaded_obj = load_file(tmp_path)
        assert loaded_obj == obj, "Loaded object does not match original"
    finally:
        os.unlink(tmp_path)

    # Test with a context manager directly
    with work_in_progress("Saving file"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
            with open(tmp_path, "wb") as f:
                pickle.dump(obj, f)
    try:
        with open(tmp_path, "rb") as f:
            loaded_obj = pickle.load(f)
        assert loaded_obj == obj, "Loaded object does not match original"
    finally:
        os.unlink(tmp_path)

    print("All tests passed.")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #2
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress():  
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #3
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #4
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test"):
        time.sleep(1)


# LLM-generated content at query #5
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second
    # The output should be "Test task... done. (1.00s)" (approximately)

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #6
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #7
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import os
    import tempfile

    # Test with context manager
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pkl")
        obj = {"key": "value"}
        with work_in_progress("Saving file"):
            with open(path, "wb") as f:
                pickle.dump(obj, f)
        with work_in_progress("Loading file"):
            with open(path, "rb") as f:
                loaded_obj = pickle.load(f)
        assert obj == loaded_obj

    # Test with decorator
    @work_in_progress("Loading file")
    def load_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pkl")
        obj = {"key": "value"}
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        loaded_obj = load_file(path)
        assert obj == loaded_obj

if __name__ == "__main__":
    test_work_in_progress()
    print("All tests passed.")


# LLM-generated content at query #8
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress():


# LLM-generated content at query #9
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import os
    import tempfile

    # Test with context manager
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pkl")
        obj = {"key": "value"}
        with work_in_progress("Saving file"):
            with open(path, "wb") as f:
                pickle.dump(obj, f)
        assert os.path.exists(path)

    # Test with decorator
    @work_in_progress("Loading file")
    def load_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pkl")
        obj = {"key": "value"}
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        loaded_obj = load_file(path)
        assert loaded_obj == obj

    print("All tests passed.")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #10
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second
    # The output should be: "Test task... done. (1.00s)"

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #11
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #12
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #13
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #14
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #15
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second
    print("Test passed!")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #16
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test"):
        time.sleep(1)
    print("Test passed!")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #17
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress():


# LLM-generated content at query #18
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #19
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #20
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import tempfile
    import os

    # Test with a function
    @work_in_progress("Loading file")
    def load_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # Test with a context manager
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pkl")
        obj = {"test": "object"}
        with open(path, "wb") as f:
            pickle.dump(obj, f)

        # Test loading
        loaded_obj = load_file(path)
        assert loaded_obj == obj

        # Test saving
        with work_in_progress("Saving file"):
            with open(path, "wb") as f:
                pickle.dump(obj, f)

        # Verify the file was saved
        with open(path, "rb") as f:
            saved_obj = pickle.load(f)
        assert saved_obj == obj

    print("All tests passed.")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #21
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test"):
        time.sleep(1)
    # Expected output: "Test... done. (1.00s)"


# LLM-generated content at query #22
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second
    # The output should be "Test task... done. (1.00s)" (approximately)

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #23
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #24
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test"):
        time.sleep(1)
    print("Test passed")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #25
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import tempfile
    import os

    # Test with context manager
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        test_data = {'key': 'value'}
        pickle.dump(test_data, f)
        temp_path = f.name

    try:
        with work_in_progress("Loading file"):
            with open(temp_path, 'rb') as f:
                loaded_data = pickle.load(f)
        assert loaded_data == test_data, "Data mismatch after loading"

        with work_in_progress("Saving file"):
            with open(temp_path, 'wb') as f:
                pickle.dump(test_data, f)
    finally:
        os.unlink(temp_path)

    # Test with decorator
    @work_in_progress("Custom task")
    def custom_task():
        time.sleep(0.1)  # Simulate some work

    custom_task()

    print("All tests passed!")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #26
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress():  
    with work_in_progress("Test task"):
        time.sleep(0.5)  # Simulate some work
    print("Test passed!")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #27
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #28
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #29
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import os
    import tempfile

    # Test with context manager
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pkl")
        obj = {"key": "value"}
        with work_in_progress("Saving file"):
            with open(path, "wb") as f:
                pickle.dump(obj, f)
        with work_in_progress("Loading file"):
            with open(path, "rb") as f:
                loaded_obj = pickle.load(f)
        assert obj == loaded_obj

    # Test with decorator
    @work_in_progress("Loading file")
    def load_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pkl")
        obj = {"key": "value"}
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        loaded_obj = load_file(path)
        assert obj == loaded_obj

    print("All tests passed.")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #30
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test"):
        time.sleep(1)


# LLM-generated content at query #31
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import tempfile
    import os

    # Test with function decorator
    @work_in_progress("Loading file")
    def load_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # Test with context manager
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        pickle.dump([1, 2, 3], f)
        temp_path = f.name

    try:
        # Test function decorator
        result = load_file(temp_path)
        assert result == [1, 2, 3]

        # Test context manager
        with work_in_progress("Saving file"):
            with open(temp_path, "wb") as f:
                pickle.dump([4, 5, 6], f)

        # Verify the saved data
        with open(temp_path, "rb") as f:
            saved_data = pickle.load(f)
        assert saved_data == [4, 5, 6]

        print("All tests passed.")
    finally:
        os.unlink(temp_path)

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #32
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test"):
        time.sleep(1)
    print("Test passed")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #33
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import tempfile
    import os

    # Test with function decorator
    @work_in_progress("Loading file")
    def load_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # Test with context manager
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        test_data = {'key': 'value'}
        pickle.dump(test_data, f)
        temp_path = f.name

    try:
        # Test loading
        result = load_file(temp_path)
        assert result == test_data, "Loaded data does not match original"
        
        # Test saving with context manager
        with work_in_progress("Saving file"):
            with open(temp_path, "wb") as f:
                pickle.dump(test_data, f)
    finally:
        os.unlink(temp_path)

    print("All tests passed!")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #34
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #35
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(0.1)  # simulate a task taking 0.1 seconds
    print("Test passed.")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #36
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    import pickle
    import os
    import tempfile

    # Test with context manager
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        pickle.dump([1, 2, 3], f)
        temp_path = f.name

    try:
        with work_in_progress("Loading file"):
            with open(temp_path, 'rb') as f:
                data = pickle.load(f)
        assert data == [1, 2, 3]
    finally:
        os.unlink(temp_path)

    # Test with decorator
    @work_in_progress("Saving file")
    def save_file(path, obj):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        temp_path = f.name

    try:
        save_file(temp_path, [4, 5, 6])
        with open(temp_path, 'rb') as f:
            data = pickle.load(f)
        assert data == [4, 5, 6]
    finally:
        os.unlink(temp_path)

    print("All tests passed.")

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #37
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


# LLM-generated content at query #38
#--------------------------

# Unit test for function work_in_progress
def test_work_in_progress(): 
    with work_in_progress("Test task"):
        time.sleep(1)  # Simulate a task that takes 1 second

if __name__ == "__main__":
    test_work_in_progress()


