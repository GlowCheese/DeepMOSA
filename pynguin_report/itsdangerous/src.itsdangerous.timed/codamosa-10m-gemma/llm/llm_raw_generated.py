####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

# Assuming the classes are in a module named 'signer_module'
# and imports like Signer or TimestampSigner are available.
# Since I cannot use imports per instructions, 
# this test assumes all dependencies are in the local scope.

def test_TimedSerializer_loads():
    """
    Tests the loads method of TimedSerializer covering:
    1. Successful loading (returning payload).
    2. Successful loading with return_timestamp=True.
    3. SignatureExpired exception handling.
    4. BadSignature exception handling when multiple signers are present.
    5. Integration with payload loading logic.
    """
    # Setup Mock Serializer and Signer
    # We mock the base class behavior for 'loads' which relies on iter_unsigners
    mock_payload = b"hello-world"
    mock_timestamp_val = 1600000000
    mock_dt = datetime.fromtimestamp(mock_timestamp_val, tz=timezone.utc)
    
    # Create a mock signer instance
    mock_signer = MagicMock(spec=TimestampSigner)
    # unsign returns (payload, timestamp) when return_timestamp=True is passed
    mock_signer.unsign.return_value = (mock_payload, mock_dt)

    # Create the Serializer instance
    # We need to mock load_payload which is part of the Serializer hierarchy
    serializer = TimedSerializer()
    serializer.load_payload = MagicMock(return_value=b"decoded-data")
    
    # Mock iter_unsigners to return our mock_signer
    serializer.iter_unsigners = MagicMock(return_value=[mock_signer])

    signed_input = b"some-signed-blob"

    # --- Case 1: Successful load (default) ---
    result = serializer.loads(signed_input)
    assert result == b"decoded-data"
    mock_signer.unsign.assert_called_with(signed_input, max_age=None, return_timestamp=True)

    # --- Case 2: Successful load with return_timestamp=True ---
    result_with_ts = serializer.loads(signed_input, return_timestamp=True)
    assert result_with_ts == (b"decoded-data", mock_dt)

    # --- Case 3: SignatureExpired ---
    # When unsign raises SignatureExpired, loads should re-raise it immediately
    mock_signer.unsign.side_effect = SignatureExpired("Expired", payload=b"old")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_input)

    # --- Case 4: BadSignature with multiple signers ---
    # When first signer fails with BadSignature, it should try the next one
    signer2 = MagicMock(spec=TimestampSigner)
    signer2.unsign.return_value = (b"second-payload", mock_dt)
    serializer.load_payload.return_value = b"second-decoded"
    
    # Reset side effect: first signer fails, second succeeds
    mock_signer.unsign.side_effect = BadSignature("Bad signature", payload=b"bad")
    serializer.iter_unsigners.return_value = [mock_signer, signer2]
    
    result_fallback = serializer.loads(signed_input)
    assert result_fallback == b"second-decoded"

    # --- Case 5: All signers fail with BadSignature ---
    # Should raise the last exception encountered
    mock_signer.unsign.side_effect = BadSignature("Bad 1", payload=b"1")
    signer2.unsign.side_effect = BadSignature("Bad 2", payload=b"2")
    
    with pytest.raises(BadSignature) as excinfo:
        serializer.loads(signed_input)
    assert "Bad 2" in str(excinfo.value)

    # --- Case 6: Max Age check (integration via signer) ---
    # The logic for max_age is actually inside TimestampSigner.unsign,
    # but TimedSerializer.loads passes the argument down.
    mock_signer.unsign.side_effect = None
    mock_signer.unsign.return_value = (mock_payload, mock_dt)
    serializer.loads(signed_input, max_age=60)
    mock_signer.unsign.assert_called_with(signed_input, max_age=60, return_timestamp=True)
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimestampSigner():
    # Mocking the base Signer class dependencies
    # Since we cannot import, we assume Signer is a class that takes a secret
    # and has a 'sep' attribute as used in the code.
    
    secret = b"secret-key"
    sep = b"."
    
    # Create a mock for the base Signer behavior
    class MockSigner:
        def __init__(self, secret, sep):
            self.secret = secret
            self.sep = sep
        def get_signature(self, value):
            return b"sig"
        def unsign(self, signed_value, max_age=None, return_timestamp=False):
            return b"payload"
    
    # Patching the class to act like Signer for testing the constructor/init logic
    with pytest.MonkeyPatch.context() as m:
        m.setattr("signer.Signer", MockSigner)
        
        # Test successful instantiation and attribute access
        signer = TimestampSigner(secret=secret, sep=sep)
        
        assert signer.secret == secret
        assert signer.sep == sep
        assert isinstance(signer, TimestampSigner)

    # Test that it inherits/maintains expected types for its methods
    assert callable(signer.get_timestamp)
    assert callable(signer.timestamp_to_datetime)
    assert callable(signer.sign)
    assert callable(signer.unsign)
    assert callable(signer.validate)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimedSerializer():
    """
    Tests the initialization and basic structure of TimedSerializer.
    Since the constructor doesn't take specific arguments for logic 
    in this implementation, we verify it instantiates correctly 
    and inherits the expected default signer.
    """
    # Mocking a serializer dependency if needed, but since it's a subclass,
    # we can test with a concrete implementation or a mock of its base.
    # Assuming Serializer is available in the context as per instructions.
    
    class MockSerializer(TimedSerializer):
        def load_payload(self, payload):
            return payload

    serializer = MockSerializer()
    
    # Verify that the default signer is indeed TimestampSigner
    assert serializer.default_signer is TimestampSignor
    
    # Verify it behaves like a Serializer (instantiation check)
    assert isinstance(serializer, TimedSerializer)
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimestampSigner():
    # Since the provided code inherits from Signer, 
    # we mock the base class dependency to isolate the constructor logic.
    with pytest.MonkeyPatch.context() as m:
        # We assume 'Signer' is available in the scope or imported.
        # Mocking a typical Signer behavior for initialization.
        mock_signer_class = MagicMock()
        m.setattr("Signer", mock_signer_class)
        
        # Testing instantiation with common arguments used in Signer/TimestampSigner
        secret = "secret-key"
        sep = "."
        signer = TimestampSigner(secret=secret, sep=sep)
        
        # Assertions to ensure the object is an instance of TimestampSigner
        assert isinstance(signer, TimestampSigner)
        # Check if properties from Signer (assumed via inheritance/init) are accessible
        # This verifies that the constructor correctly passed arguments up the chain.
        assert signer.sep == sep
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
import time
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

# Assuming these exist in the local environment based on the provided code context
# from your_module import TimestampSigner, BadSignature, BadTimeSignature, SignatureExpired

def test_TimestampSigner_unsign(signer_factory):
    """
    Tests the unsign method of TimestampSigner covering various scenarios:
    valid signature, expired signature, malformed timestamp, and bad signature.
    """
    signer = signer_factory()
    sep = b"."
    payload = b"hello_world"
    
    # 1. Test successful unsigning (return bytes)
    signed_val = signer.sign(payload)
    assert signer.unsign(signed_lar=signed_val) == payload

    # 2. Test successful unsigning with return_timestamp=True
    val, ts_dt = signer.unsign(signed_val, return_timestamp=True)
    assert val == payload
    assert isinstance(ts_dt, datetime)
    assert ts_dt.tzinfo == timezone.utc

    # 3. Test SignatureExpired (Too old)
    # We mock get_timestamp to simulate the passage of time
    with patch.object(TimestampSigner, 'get_timestamp') as mock_ts:
        fixed_now = 1000
        mock_ts.return_value = fixed_now
        signed_old = signer.sign(payload)
        
        # Move "now" forward by 100 seconds
        mock_ts.return_value = fixed_now + 100
        with pytest.raises(SignatureExpired) as excinfo:
            signer.unsign(signed_old, max_age=50)
        assert payload in excinfo.value.payload

    # 4. Test SignatureExpired (Future timestamp - clock drift)
    with patch.object(TimestampSigner, 'get_timestamp') as mock_ts:
        fixed_now = 1000
        mock_ts.return_value = fixed_now
        signed_future = signer.sign(payload)
        
        # Move "now" backward (simulating a signature from the future)
        mock_ts.return_value = fixed_now - 10
        with pytest.raises(SignatureExpired):
            signer.unsign(signed_future, max_age=50)

    # 5. Test BadTimeSignature (Malformed timestamp bytes)
    # We manually construct a payload: payload + sep + corrupted_base64_ts + sep + sig
    # To keep it simple, we use a known valid signature structure but break the middle part
    valid_sig = signer.sign(payload)
    parts = valid_sig.split(sep)
    # parts[0] is payload, parts[1] is timestamp, parts[2] is signature
    corrupted_ts_part = b"not-base64-encoded-properly!@#$"
    bad_ts_payload = parts[0] + sep + corrupted_ts_part + sep + parts[2]
    
    with pytest.raises(BadTimeSignature) as excinfo:
        signer.unsign(bad_ts_payload)
    assert b"Malformed timestamp" in str(excinfo.value)

    # 6. Test BadTimeSignature (Missing separator/timestamp structure)
    # A simple string that doesn't follow the payload|ts|sig format
    with pytest.raises(BadTimeSignature) as excinfo:
        signer.unsign(b"just_a_string_no_separators")
    assert b"timestamp missing" in str(excinfo.value).lower()

    # 7. Test BadSignature (Tampered payload)
    # We modify the payload part of a valid signature
    parts = valid_sig.split(sep)
    tampered_payload = b"tampered" + sep + parts[1] + sep + parts[2]
    with pytest.raises(BadSignature):
        signer.unsign(tampered_payload)

    # 8. Test BadSignature with valid timestamp (Testing the error propagation logic)
    # We want a signature that is cryptographically invalid but has a readable timestamp
    # So we use an incorrect signature component but keep the timestamp part valid
    with patch.object(TimestampSigner, 'get_signature') as mock_sig:
        mock_sig.return_value = b"wrong_signature"
        tampered_sig = signer.sign(payload) 
        # The code should catch BadSignature but extract the payload from it
        with pytest.raises(BadSignature):
            signer.unsign(tampered_sig)

@pytest.fixture
def signer_factory():
    """Fixture to provide a TimestampSigner instance."""
    from your_module import TimestampSigner # Adjust import as needed
    class MockSigner(TimestampSigner):
        def get_signature(self, value: bytes) -> bytes:
            # Simple deterministic signature for testing
            return b"sig_" + value[::-1] 
    
    return lambda: MockSigner(secret_key=b"secret")
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimedSerializer():
    """
    Tests the instantiation and basic properties of TimedSerializer.
    Since the constructor of TimedSerializer is inherited from Serializer 
    and does not take specific arguments in its definition, we verify 
    it initializes correctly with expected default attributes.
    """
    # Mocking the base Serializer dependency if necessary, 
    # but here we test the concrete class behavior.
    serializer = TimedSerializer()

    # Verify the instance is of the correct type
    assert isinstance(serializer, TimelySerializer)
    
    # Verify that it uses TimestampSigner as its default signer
    assert serializer.default_signer == TimestampSigner

    # Verify that it inherits/uses the expected class structure
    assert hasattr(serializer, 'loads')
    assert hasattr(serializer, 'dumps')
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimedSerializer():
    """
    Tests the initialization of TimedSerializer by verifying it 
    correctly inherits and sets up its default signer.
    """
    # Mocking a Serializer dependency if needed, but since we are testing 
    # the constructor/initialization behavior:
    
    # We use a dummy class or actual instance. 
    # Since TimedSerializer is a subclass of Serializer, we test that 
    # it can be instantiated and has the expected default_signer attribute.
    
    class MockSerializer(TimedSerializer):
        def __init__(self, secret_key: bytes):
            super().__init__(secret_key=secret_key)

    secret = b"secret-key"
    serializer = MockSerializer(secret_key=secret)
    
    # Verify the class is indeed a TimedSerializer
    assert isinstance(serializer, TimedSerializer)
    
    # Verify that it uses TimestampSigner as its default signer type
    assert serializer.default_signer == TimestampSigneler
    
    # Verify that the internal signer created by the constructor 
    # (inherited from Signer/Serializer) uses the correct secret.
    # In typical implementations, self.signer is initialized with the secret.
    assert hasattr(serializer, 'signer') or hasattr(serializer, '_signer')
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
import time
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

# Assuming these are available in the environment as per instructions
# from your_module import TimestampSigner, BadSignature, BadTimeSignature, SignatureExpired
# from your_module.encoding import base64_encode, int_to_bytes

def test_TimestampSigner_unsign():
    secret = b"secret-key"
    signer = TimestampSigner(secret)
    sep = b"."
    
    payload = b"hello-world"
    
    # 1. Test successful unsigning with timestamp return
    signed_value = signer.sign(payload)
    with patch.object(TimestampSignor, 'get_timestamp', return_value=1000):
        # Re-generate signature with fixed time to ensure predictable result
        fixed_signer = TimestampSigner(secret)
        signed_value = fixed_signer.sign(payload)
        
        val, ts_dt = signer.unsign(signed_value, return_timestamp=True)
        assert val == payload
        assert ts_dt == datetime.fromtimestamp(1000, tz=timezone.utc)

    # 2. Test successful unsigning without timestamp return
    val = signer.unsign(signed_value)
    assert val == payload

    # 3. Test max_age validation (Success)
    with patch.object(TimestampSigner, 'get_timestamp', return_value=1050):
        # Signature was at 1000, current is 1050, age is 50. Max age 60.
        assert signer.unsign(signed_value, max_age=60) == payload

    # 4. Test max_age validation (Failure - Expired)
    with patch.object(TimestampSigner, 'get_timestamp', return_value=1200):
        # Signature was at 1000, current is 1200, age is 200. Max age 60.
        with pytest.raises(SignatureExpired) as excinfo:
            signer.unsign(signed_value, max_age=60)
        assert "Signature age 200 > 60 seconds" in str(excinfo.value)
        assert excinfo.value.payload == payload

    # 5. Test max_age validation (Failure - Future signature/Negative age)
    with patch.object(TimestampSigner, 'get_timestamp', return_value=900):
        # Signature was at 1000, current is 900, age is -100.
        with pytest.raises(SignatureExpired) as excinfo:
            signer.unsign(signed_value, max_age=60)
        assert "Signature age -100 < 0 seconds" in str(excinfo.value)

    # 6. Test BadSignature (Invalid payload/signature)
    invalid_sig = b"wrong-signature-data"
    with pytest.raises(BadSignature):
        signer.unsign(invalid_sig)

    # 7. Test BadTimeSignature (Malformed timestamp part)
    # Construct a value that has the separator but invalid base64/int in timestamp slot
    malformed_ts = payload + sep + b"not-base64-or-not-int!!"
    # We need to bypass the signature check for this specific test case 
    # by mocking the super().unsign to return our malformed string without erroring
    with patch('signer.Signer.unsign', return_value=malformed_ts):
        with pytest.raises(BadTimeSignature) as excinfo:
            signer.unsign(malformed_ts)
        assert "Malformed timestamp" in str(excinfo.value)

    # 8. Test BadTimeSignature (Missing separator/timestamp structure)
    no_sep_data = b"just-payload-no-separator"
    with patch('signer.Signer.unsign', return_value=no_sep_data):
        with pytest.raises(BadTimeSignature) as excinfo:
            signer.unsign(no_sep_data)
        assert "timestamp missing" in str(excinfo.value)

    # 9. Test validate method
    assert signer.validate(signed_value) is True
    with patch.object(TimestampSigner, 'get_timestamp', return_value=2000):
        assert signer.validate(signed_value, max_age=10) is False
    with pytest.raises(BadSignature):
        signer.validate(b"invalid")
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

@pytest.mark.parametrize("sep", [b".", b":", b"|"])
def test_TimestampSigner_unsign(sep):
    # Setup Signer with a known secret
    secret = b"secret-key"
    signer = TimestampSigner(secret)
    signer.sep = sep

    payload = b"hello-world"
    now = 1700000000  # Fixed timestamp for testing

    with patch("time.time", return_value=float(now)):
        signed_value = signer.sign(payload)

    # Test Case 1: Basic successful unsign (no max_age)
    assert signer.unsign(signed_value) == payload

    # Test Case 2: Successful unsign with return_timestamp=True
    un_payload, un_ts = signer.unsign(signed_value, return_timestamp=True)
    assert un_payload == payload
    assert un_ts == datetime.fromtimestamp(now, tz=timezone.utc)

    # Test Case 3: Successful unsign with max_age (within limits)
    assert signer.unsign(signed_value, max_age=100) == payload

    # Test Case 4: SignatureExpired - Too old
    with pytest.raises(SignatureExpired) as excinfo:
        signer.unsign(signed_value, max_age=10)
    assert "Signature age" in str(excinfo.value)
    assert excinfo.value.payload == payload

    # Test Case 5: SignatureExpired - Future timestamp (negative age)
    future_ts = now + 100
    # Manually construct a signature with a future timestamp
    # Format: value + sep + b64(ts) + sep + signature
    ts_bytes = base64_encode(int_to_bytes(future_ts))
    future_signed = payload + sep + ts_bytes + sep + signer.get_signature(payload + sep + ts_bytes)
    
    with pytest.raises(SignatureExpired) as excinfo:
        signer.unsign(future_signed, max_age=10)
    assert "negative" in str(excinfo.value).lower() or " < 0" in str(excinfo.value)

    # Test Case 6: BadSignature - Tampered payload
    tampered_payload = b"tampered" + sep + payload + sep + ts_bytes + sep + signer.get_signature(payload + sep + ts_bytes)
    # Note: We need to be careful with construction. Let's just modify the original bits.
    tampered_value = signed_value[:-5] + b"wrong" 
    with pytest.raises(BadSignature):
        signer.unsign(tampered_value)

    # Test Case 7: BadTimeSignature - Malformed timestamp (not base64 or not int)
    malformed_ts = payload + sep + b"not-base64-!!" + sep + signer.get_signature(payload + sep + b"not-base64-!!")
    with pytest.raises(BadTimeSignature) as excinfo:
        signer.unsign(malformed_ts)
    assert "Malformed timestamp" in str(excinfo.value)

    # Test Case 8: BadTimeSignature - Missing separator (corrupted structure)
    broken_structure = payload + b"no-separator-here"
    with pytest.raises(BadTimeSignature) as excinfo:
        signer.unsign(broken_structure)
    assert "timestamp missing" in str(excinfo.value)

    # Test Case 9: BadTimeSignature - Signature error with valid timestamp structure
    # We use a bad signature but keep the timestamp part intact so it parses the date
    bad_sig_payload = payload + sep + ts_bytes + sep + b"invalid-signature-bits"
    with pytest.raises(BadTimeSignature) as excinfo:
        signer.unsign(bad_sig_payload)
    assert excinfo.value.date_signed == datetime.fromtimestamp(now, tz=timezone.utc)

def test_TimestampSigner_validate(sep):
    secret = b"secret-key"
    signer = TimestampSigner(secret)
    signer.sep = sep
    payload = b"valid"
    
    with patch("time.time", return_value=1000.0):
        signed = signer.sign(payload)

    assert signer.validate(signed) is True
    assert signer.validate(signed, max_age=10000) is True
    assert signer.validate(signed, max_age=1) is False
    assert signer.validate(b"invalid-data") is False
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

# Assuming the classes and exceptions are in the same module or accessible via imports
# Since I cannot include imports, this test assumes the environment is set up.

def test_TimedSerializer_loads():
    """
    Tests the loads method of TimedSerializer covering:
    1. Successful loading (returning payload).
    2. Successful loading with return_timestamp=True.
    3. Signature expiration (SignatureExpired).
    4. Bad signature (BadSignature).
    5. Multiple signers handling.
    """
    # Setup dependencies
    serializer = TimedSerializer()
    mock_signer = MagicMock(spec=TimestampSigner)
    
    # Mocking the payload data and timestamp
    payload_bytes = b"original_payload"
    timestamp_val = 1600000000  # Fixed timestamp
    dt_val = datetime.fromtimestamp(timestamp_val, tz=timezone.utc)
    
    # Mocking the serializer's internal method to decode payload
    serializer.load_payload = MagicMock(return_value="decoded_data")

    # Case 1: Successful loading (Standard)
    mock_signer.unsign.return_value = (payload_bytes, dt_val)
    with patch.object(TimedSerializer, 'iter_unsigners', return_value=[mock_signer]):
        result = serializer.loads(b"valid_signed_string")
        assert result == "decoded_data"

    # Case 2: Successful loading with return_timestamp=True
    with patch.object(TimedSerializer, 'iter_unsigners', return_value=[mock_signer]):
        result_payload, result_ts = serializer.loads(b"valid_signed_string", return_timestamp=True)
        assert result_payload == "decoded_data"
        assert result_ts == dt_val

    # Case 3: Signature is expired (SignatureExpired should propagate)
    mock_signer.unsign.side_effect = SignatureExpired("Expired", payload=b"expired_payload")
    with patch.object(TimedSerializer, 'iter_unsigners', return_value=[mock_signer]):
        with pytest.raises(SignatureExpired):
            serializer.loads(b"expired_string", max_age=10)

    # Case 4: Bad signature on first signer, but second signer is valid
    signer2 = MagicMock(spec=TimestampSigner)
    signer2.unsign.return_value = (b"second_payload", dt_val)
    serializer.load_payload = MagicMock(return_value="second_decoded")
    
    mock_signer.unsign.side_effect = BadSignature("Bad sig", payload=b"bad_payload")
    with patch.object(TimedSerializer, 'iter_unsigners', return_value=[mock_signer, signer2]):
        result = serializer.loads(b"multi_signer_string")
        assert result == "second_decoded"

    # Case 5: All signers fail (Should raise the last BadSignature)
    last_error = BadSignature("Final failure", payload=b"failed")
    signer2.unsign.side_effect = last_error
    with patch.object(TimedSerializer, 'iter_unsigners', return_value=[mock_signer, signer2]):
        with pytest.raises(BadSignature) as excinfo:
            serializer.loads(b"all_fail_string")
        assert "Final failure" in str(excinfo.value)

    # Case 6: Salt parameter passing
    with patch.object(TimedSerializer, 'iter_unsigners', return_value=[mock_signer]) as mock_iter:
        serializer.loads(b"salt_test", salt="my_salt")
        mock_iter.assert_called_with(salt="my_salt")
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
import time
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

# Assuming the classes and exceptions are available in the namespace 
# as per the instruction "without any additional text or information"

def test_TimestampSigner_unsign():
    secret = b"secret-key"
    sep = b"."
    signer = TimestampSigner(secret, sep=sep)
    payload = b"hello-world"
    
    # 1. Test successful unsigning (basic)
    signed_value = signer.sign(payload)
    unsign_result = signer.unsign(signedly_value := signed_value)
    assert unsign_result == payload

    # 2. Test successful unsigning with return_timestamp=True
    unsign_result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert unsign_result == payload
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # 3. Test SignatureExpired (Future/Past) - Mocking time to simulate expiration
    fixed_now = 1000000
    with patch("time.time", return_value=float(fixed_now)):
        # Create a signature at fixed_now
        signed_at_now = signer.sign(payload)
        
        # Move time forward to exceed max_age
        with patch("time.time", return_value=float(fixed_now + 100)):
            # Test expired (max_age is 50, current age is 100)
            with pytest.raises(SignatureExpired) as excinfo:
                signer.unsign(signed_at_now, max_age=50)
            assert "Signature age 100 > 50 seconds" in str(excinfo.value)
            assert excinfo.value.payload == payload

        # Test valid (max_age is 200, current age is 100)
        assert signer.unsign(signed_at_now, max_age=200) == payload

    # 4. Test BadSignature (Tampered Payload)
    tampered_value = signed_value.replace(b"hello", b"hallo")
    with pytest.raises(BadSignature):
        signer.unsign(tampered_value)

    # 5. Test BadTimeSignature (Malformed Timestamp/No Separator)
    # Manually create a string that looks like it has a separator but bad data
    bad_ts_value = b"payload.notbase64!!!"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_ts_value)

    # 6. Test BadTimeSignature (Valid signature, but corrupted timestamp part)
    # We hijack the sign process to create a validly signed but invalid TS structure
    with patch.object(TimestampSigner, 'get_signature', return_value=b"sig"):
        # Structure: payload + sep + malformed_ts + sep + sig
        # This is hard to trigger via public API without mocking internal parts
        # But we can test the logic where sep is present but bytes_to_int fails
        malformed_payload = b"payload.notbase64"
        # We use a signer that would produce a valid signature for this string
        # but the timestamp part (after last dot) cannot be decoded as int.
        with pytest.raises(BadTimeSignature) as excinfo:
            signer.unsign(b"payload." + b"notbase64_encoded_garbage")
        assert "Malformed timestamp" in str(excinfo.value)

    # 7. Test BadTimeSignature (Timestamp missing entirely)
    with pytest.raises(BadTimeSignature) as excinfo:
        signer.unsign(b"no_separator_at_all")
    assert "timestamp missing" in str(excinfo.value)

    # 8. Test validation method helper
    assert signer.validate(signed_value) is True
    assert signer.validate(tampered_value) is False
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimedSerializer():
    """
    Tests the initialization and core properties of TimedSerializer.
    Since the constructor is inherited from Serializer, we verify 
    it correctly sets up the default signer as TimestampSigner.
    """
    # Mocking a serializer (e.g., JSON) to avoid needing actual serialization logic
    mock_serializer = MagicMock(spec=TimedSerializer)
    
    # In practice, TimedSerializer is used via subclasses like JsonSerializer.
    # We test the class attribute that defines its behavior.
    assert TimedSerializer.default_signer == TimestampSigner
    
    # Verify it is indeed a subclass of Serializer (via the provided context)
    from .serializer import Serializer
    assert issubclass(TimedSerializer, Serializer)

    # Test instance identity for the default signer type
    instance = TimedSerializer() 
    # Note: In a real scenario, we'd pass a serializer/loader to the constructor.
    # Here we check if it uses TimestampSigner logic.
    assert hasattr(instance, 'iter_unsigners')
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimestampSigner():
    """
    Tests the initialization and basic properties of TimestampSigner.
    Since TimestampSigner inherits from Signer, we verify it can be 
    instantiated with a secret and maintains expected attributes.
    """
    secret = b"secret-key"
    sep = b"."
    signer = TimestampSigner(secret=secret, sep=sep)

    # Verify the instance is of the correct type
    assert isinstance(signer, TimestampSigneler)
    assert isinstance(signer, Signer)

    # Verify that key/separator attributes are correctly assigned via inheritance
    # (Assuming Signer sets these during __init__)
    assert signer.secret == secret
    assert signer.sep == sep

    # Verify the get_timestamp method returns an integer
    assert isinstance(signer.get_timestamp(), int)

def test_TimestampSigner_initialization_with_string_sep():
    """Tests that the constructor handles string separators if passed."""
    signer = TimestampSigner(secret=b"key", sep=".")
    assert signer.sep == b"."

def test_TimestampSigner_timestamp_conversion():
    """Tests that timestamp_to_datetime produces UTC aware datetimes."""
    signer = TimestampSigner(secret=b"key")
    ts = 1609459200  # 2021-01-01 00:00:00 UTC
    dt = signer.timestamp_to_datetime(ts)
    
    assert dt.year == 2021
    assert dt.month == 1
    assert dt.day == 1
    assert dt.tzinfo is not None
    assert str(dt.utcoffset()) == "0:00"
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimedSerializer():
    """
    Test the construction and basic properties of TimedSerializer.
    Since the constructor is inherited from Serializer, we verify 
    that it initializes with expected attributes and that its 
    default signer is correctly set to TimestampSigner.
    """
    # Mocking dependencies for a pure unit test of the class structure
    # if they were needed for instantiation (e.g., secret_key).
    secret_key = b"secret-key"
    
    # Initialize the serializer
    serializer = TimedSerializer(secret_key)
    
    # Verify that the default signer is indeed TimestampSigner
    assert serializer.default_signer is TimestampSignogner
    
    # Verify it behaves as a Serializer (inheriting from Signer/Serializer logic)
    # We check if the instance has access to its core components
    assert hasattr(serializer, 'loads')
    assert hasattr(serializer, 'dumps')
    
    # Test that it correctly identifies itself as an instance of TimedSerializer
    assert isinstance(serializer, TimedSerializer)
    
    # Verify that the internal signer is an instance of TimestampSigner
    # (The serializer typically creates a Signer or uses the default_signer class)
    # Note: In many implementations, serializers hold a reference to a Signer object.
    # Since we don't see the Serializer constructor logic here, 
    # we rely on the metadata provided in the snippet.
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

@pytest.mark.parametrize("payload, signed_value, expected_return", [
    ("data", b"data.timestamp.signature", "data"),
    (123, b"123.timestamp.signature", 123),
])
def test_TimedSerializer_loads(payload, signed_value, expected_return):
    # Setup Mocking for the dependencies of TimedSerializer.loads
    # We need to mock the serializer's internal mechanism and the signer behavior
    
    serializer = TimedSerializer()
    
    # Mocking the iterator of unsigners to return a single mocked TimestampSigner
    mock_signer = MagicMock(spec=TimestampSigner)
    serializer.iter_unsigners = MagicMock(return_value=[mock_signer])
    
    # Setup the behavior for signer.unsign
    # We simulate a successful unsigning returning (payload_bytes, datetime)
    payload_bytes = b"data" if isinstance(payload, str) else payload
    ts_dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
    mock_signer.unsign.return_value = (payload_bytes, ts_dt)
    
    # Mocking load_payload to return the actual object we want to test
    serializer.load_payload = MagicMock(return_value=payload)

    # Case 1: Standard loads (returns payload only)
    result = serializer.loads(signed_value)
    assert result == payload
    mock_signer.unsign.assert_called_with(signed_value, max_age=None, return_timestamp=True)

    # Case 2: loads with return_timestamp=True
    result_with_ts, result_ts = serializer.loads(signed_value, return_timestamp=True)
    assert result_with_ts == payload
    assert result_ts == ts_dt

    # Case 3: loads with max_age
    serializer.loads(signed_value, max_age=60)
    mock_signer.unsign.assert_called_with(signed_value, max_age=60, return_timestamp=True)

    # Case 4: Testing SignatureExpired exception propagation
    from .exc import SignatureExpired
    mock_signer.unsign.side_effect = SignatureExpired("expired", payload=b"data")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_value)

    # Case 5: Testing BadSignature exception (should try next signer or raise last error)
    from .exc import BadSignature
    mock_signer.unsign.side_effect = BadSignature("bad", payload=b"data")
    with pytest.raises(BadSignature):
        serializer.loads(signed_value)

def test_TimedSerializer_loads_with_salt():
    serializer = TimedSerializer()
    mock_signer = MagicMock(spec=TimestampSigner)
    serializer.iter_unsigners = MagicMock(return_value=[mock_signer])
    
    payload_bytes = b"data"
    ts_dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
    mock_signer.unsign.return_value = (payload_bytes, ts_dt)
    serializer.load_payload = MagicMock(return_value="data")

    salt = "my_salt"
    serializer.loads(b"some_signed_value", salt=salt)
    
    # Verify that the salt was passed down to the unsigners iterator
    serializer.iter_unsigners.assert_called_with(salt=salt)
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimestampSigner():
    """
    Test the construction and basic initialization of TimestampSigner.
    Since TimestampSigner inherits from Signer, we verify it can be 
    instantiated and maintains expected signer attributes.
    """
    # Mocking the secret key and separator as they would be passed to Signer/TimestampSigner
    secret_key = b"super-secret-key"
    separator = b"."

    # Initialize TimestampSigner
    signer = TimestampSigneler(secret_key, sep=separator)

    # Verify instance type
    assert isinstance(signer, TimestampSigner)
    
    # Verify inherited attributes from Signer/TimestampSigner
    assert signer.secret_key == secret_key
    assert signer.sep == separator

class TimestampSigneler(TimestampSigner):
    """A helper subclass to allow instantiation without complex dependency injection 
    if the base Signer class requires specific environmental setup."""
    def __init__(self, secret_key, sep=b"."):
        # Manually simulating what a real Signer constructor would do
        # to avoid needing the full implementation of the parent's logic
        super().__init__()
        self.secret_key = secret_key
        self.sep = sep

# Note: In a real environment, you would test the actual class 
# provided in the snippet which inherits from Signer.
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

@pytest.mark.parametrize(
    "payload, salt, return_timestamp, expected",
    [
        ("data", "salt", False, "data"),
        ("data", "salt", True, ("data", None)),  # Timestamp will be mocked
        ("data", None, False, "data"),
    ],
)
def test_TimedSerializer_loads(payload, salt, return_timestamp, expected):
    """
    Tests the loads method of TimedSerializer.
    Covers successful decryption/unsigning and handles return_timestamp logic.
    """
    # Setup mocks for the serializer components
    serializer = TimedSerializer()
    signer = MagicMock(spec=TimestampSigner)
    
    # Mock payload content (simulating base64 encoded data)
    encoded_payload = b"encoded_data"
    
    # Configure the mock signer's unsign behavior
    # We mock return_timestamp=True because loads calls it with True internally
    mock_ts = 1600000000
    mock_dt = datetime.fromtimestamp(mock_ts, tz=timezone.utc)
    signer.unsign.return_value = (encoded_payload, mock_dt)
    
    # Mock load_payload to return the original payload string
    serializer.load_payload = MagicMock(return_value=payload)
    
    # Mock iter_unsigners to yield our controlled signer
    serializer.iter_unsigners = MagicMock(return_value=[signer])

    # Execute the method under test
    # We use a patch to ensure we control the value returned by the mock logic
    result = serializer.loads(
        s=b"signed_value", 
        max_age=None, 
        return_timestamp=return_timestamp, 
        salt=salt
    )

    # Assertions
    if return_timestamp:
        # If return_timestamp is True, we expect (payload, datetime)
        assert result[0] == payload
        assert result[1] == mock_dt
    else:
        assert result == payload

    # Verify the signer was called with correct parameters
    signer.unsign.assert_called_once_with(
        b"signed_value", 
        max_age=None, 
        return_timestamp=True
    )

def test_TimedSerializer_loads_expired():
    """Tests that SignatureExpired is raised and not swallowed by the loop."""
    serializer = TimedSerializer()
    signer = MagicMock(spec=TimestampSigner)
    serializer.iter_unsigners = MagicMock(return_value=[signer])

    # Simulate an expired signature exception
    signer.unsign.side_effect = SignatureExpired("Expired", payload=b"old")

    with pytest.raises(SignatureExpired):
        serializer.loads(b"some_value")

def test_TimedSerializer_loads_bad_signature():
    """Tests that BadSignature is raised if all signers fail."""
    serializer = TimizedSerializer() # Note: assuming class exists in scope as per instructions
    signer1 = MagicMock(spec=TimestampSigner)
    signer2 = MagicMock(spec=TimestampSigner)
    
    serializer.iter_unsigners = MagicMock(return_value=[signer1, signer2])
    
    # Both signers fail with BadSignature
    err1 = BadSignature("fail 1", payload=b"p1")
    err2 = BadSignature("fail 2", payload=b"p2")
    signer1.unsign.side_effect = err1
    signer2.unsign.side_effect = err2

    with pytest.raises(BadSignature) as excinfo:
        serializer.loads(b"invalid_value")
    
    # The last exception should be the one raised
    assert "fail 2" in str(excinfo.value)
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

# Assuming the classes and exceptions are available in the namespace 
# as per the provided code context.

def test_TimestampSigner_unsign():
    secret = b"secret-key"
    sep = b"."
    signer = TimestampSigner(secret)
    
    payload = b"hello-world"
    
    # 1. Test successful unsign (returning bytes)
    signed_val = signer.sign(payload)
    assert signer.unsign(signed_val) == payload

    # 2. Test successful unsign with timestamp return
    value, ts_dt = signer.unsign(signed_val, return_timestamp=True)
    assert value == payload
    assert isinstance(ts_dt, datetime)
    assert ts_dt.tzinfo == timezone.utc

    # 3. Test SignatureExpired (too old)
    # We patch time.time to simulate the future
    with patch("time.time") as mock_time:
        # Sign at timestamp 1000
        mock_time.return_value = 1000.0
        signed_val_old = signer.sign(payload)
        
        # Check at timestamp 2000 (age is 1000, max_age is 500)
        mock_time.return_value = 2000.0
        with pytest.raises(SignatureExpired) as excinfo:
            signer.unsign(signed_val_old, max_age=500)
        assert "Signature age 1000 > 500" in str(excinfo.value)
        assert excinfo.value.payload == payload

    # 4. Test SignatureExpired (future timestamp - clock drift/malicious)
    with patch("time.time") as mock_time:
        mock_time.return_value = 1000.0
        signed_val_future = signer.sign(payload)
        
        # Check at timestamp 500 (age is -500, max_age is 500)
        mock_time.return_value = 500.0
        with pytest.raises(SignatureExpired) as excinfo:
            signer.unsign(signed_val_future, max_age=500)
        assert "Signature age -500 < 0" in str(excinfo.value)

    # 5. Test BadSignature (tampered payload)
    tampered_val = b"tampered" + signed_val[len(b"tampered"):]
    with pytest.raises(BadSignature):
        signer.unsign(tampered_val)

    # 6. Test BadTimeSignature (missing separator/timestamp structure)
    # A valid signature but not following the timestamp format (no sep in result)
    raw_sig = signer.get_signature(b"only-payload")
    malformed_structure = b"only-payload" + sep + raw_sig # This has 2 seps, which is fine, 
    # but let's try one that doesn't have the timestamp part at all
    with pytest.raises(BadTimeSignature) as excinfo:
        signer.unsign(b"no-timestamp-here")
    assert "timestamp missing" in str(excinfo.value)

    # 7. Test BadTimeSignature (malformed base64/integer timestamp)
    # Construct a payload where the last part after sep is not valid b64 for an int
    bad_ts_val = payload + sep + b"not-base64-and-not-int!!!"
    # We need to sign this so it passes the Signer.unsign check (the signature itself must be valid)
    # But we want the internal timestamp decoding to fail.
    # The simplest way is to use a real signature but corrupt the timestamp part string.
    valid_sig = signer.sign(payload)
    parts = valid_sig.split(sep)
    # parts[0] = payload, parts[1] = timestamp, parts[2] = signature
    # We replace parts[1] with junk but keep the signature part valid for the original content
    corrupted_ts_val = payload + sep + b"invalid_b64_data" + sep + signer.get_signature(payload + sep + b"invalid_b64_data")
    # Note: The logic in unsign catches Exception during bytes_to_int and raises BadTimeSignature if ts_int is None
    with pytest.raises(BadTimeSignature) as excinfo:
        signer.unsign(corrupted_ts_val)
    assert "Malformed timestamp" in str(excinfo.value)

    # 8. Test validate() method
    assert signer.validate(signed_val) is True
    assert signer.validate(tampered_val) is False
    
    with patch("time.time") as mock_time:
        mock_time.return_value = 1000.0
        old_val = signer.sign(payload)
        mock_time.return_value = 5000.0
        assert signer.validate(old_val, max_age=10) is False
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

# Assuming the classes are in the current namespace as per instructions
# If testing in a real environment, ensure TimestampSigner and exceptions are imported.

def test_TimestampSigner_unsign():
    secret = b"secret-key"
    signer = TimestampSigner(secret)
    sep = b"."
    payload = b"hello-world"
    
    # Helper to create a manual signed value: payload + sep + timestamp_b64 + sep + signature
    # We will mock the Signer.get_signature and Signer.unsign behavior via TimestampSigner's logic
    
    def create_mock_signed(p, ts_int, sig=b"valid-sig"):
        ts_bytes = base64_encode(int_to_bytes(ts_int))
        return p + sep + ts_bytes + sep + sig

    # 1. Test successful unsign (no max_age)
    current_time = 1000000
    with patch.object(TimestampSigner, 'get_timestamp', return_value=current_time):
        signed_val = create_mock_signed(payload, current_time)
        # We must mock the underlying Signer.unsign to return the stripped value (payload + sep + ts)
        with patch('__main__.Signer.unsign', return_value=payload + sep + base64_encode(int_to_bytes(current_time))):
            result = signer.unsign(signed_val)
            assert result == payload

    # 2. Test successful unsign with return_timestamp=True
    with patch('__main__.Signer.unsign', return_value=payload + sep + base64_encode(int_to_bytes(current_time))):
        result, ts_dt = signer.unsign(signed_val, return_timestamp=True)
        assert result == payload
        assert ts_dt == datetime.fromtimestamp(current_time, tz=timezone.utc)

    # 3. Test SignatureExpired (Too old)
    old_time = current_time - 100
    max_age = 50
    with patch.object(TimestampSigner, 'get_timestamp', return_value=current_time):
        signed_val_old = create_mock_signed(payload, old_time)
        with patch('__main__.Signer.unsign', returnly=payload + sep + base64_encode(int_to_bytes(old_time))):
            # Note: The actual logic uses super().unsign which we mock to pass the signature check
            with patch('__main__.Signer.unsign', return_value=payload + sep + base64_encode(int_to_bytes(old_time))):
                with pytest.raises(SignatureExpired) as excinfo:
                    signer.unsign(signed_val_old, max_age=max_age)
                assert "Signature age 100 > 50 seconds" in str(excinfo.value)

    # 4. Test SignatureExpired (Future timestamp - logic check)
    future_time = current_time + 100
    with patch.object(TimestampSigner, 'get_timestamp', return_value=current_time):
        signed_val_future = create_mock_signed(payload, future_time)
        with patch('__main__.Signer.unsign', return_value=payload + sep + base64_encode(int_to_bytes(future_time))):
            with pytest.raises(SignatureExpired) as excinfo:
                signer.unsign(signed_val_future, max_age=max_age)
            assert "Signature age -100 < 0 seconds" in str(excinfo.value)

    # 5. Test BadSignature (Signature mismatch but timestamp is valid)
    with patch('__main__.Signer.unsign', side_effect=BadSignature("bad-sig", payload=payload + sep + base64_encode(int_to_bytes(current_time)))):
        # The logic catches BadSignature, then checks the timestamp in the error's payload
        with pytest.raises(BadTimeSignature) as excinfo:
            signer.unsign(signed_val)
        assert "bad-sig" in str(excinfo.value)

    # 6. Test BadTimeSignature (Malformed timestamp)
    malformed_ts_val = payload + sep + b"not-base64-int" + sep + b"signature"
    with patch('__main__.Signer.unsign', return_value=payload + sep + b"not-base64-int"):
        with pytest.raises(BadTimeSignature) as excinfo:
            signer.unsign(malformed_ts_val)
        assert "Malformed timestamp" in str(excinfo.value)

    # 7. Test BadTimeSignature (Missing separator/timestamp structure)
    with patch('__main__.Signer.unsign', return_value=payload):
        with pytest.raises(BadTimeSignature) as excinfo:
            signer.unsign(payload + b"something-extra")
        assert "timestamp missing" in str(excinfo.value)

    # 8. Test Validate True
    with patch('__main__.Signer.unsign', return_value=payload + sep + base64_encode(int_to_bytes(current_time))):
        assert signer.validate(signed_val) is True

    # 9. Test Validate False
    with patch('__main__.Signer.unsign', side_effect=BadSignature("bad")):
        assert signer.validate(signed_val) is False
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimestampSigner():
    # Mocking the base Signer class dependencies since we cannot 
    # instantiate TimestampSigner without a valid secret key/setup
    # derived from its parent Signer class.
    
    # We create a subclass that replaces get_signature with a mock 
    # to avoid needing actual cryptographic logic for a constructor test.
    class MockSigner(TimestampSigner):
        def __init__(self, secret_key: bytes, sep: bytes = b"."):
            self.secret_key = secret_key
            self.sep = sep
        
        def get_signature(self, value: bytes) -> bytes:
            return b"mock_sig"

    # Test standard initialization
    secret = b"super-secret-key"
    signer = MockSignor(secret_key=secret, sep=b".")
    
    assert signer.secret_key == secret
    assert signer.sep == b"."

    # Test initialization with custom separator
    custom_sep = b"|"
    signer_custom = MockSigner(secret_key=secret, sep=custom_sep)
    assert signer_custom.sep == custom_sep

    # Test that it inherits the default behavior of Signer for basic attributes
    # (Assuming Signer sets up basic state in __init__)
    assert hasattr(signer, 'get_timestamp')
    assert callable(signer.get_timestamp)
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimestampSigner():
    """
    Tests the initialization and basic properties of a TimestampSigner instance.
    Since TimestampSigner inherits from Signer, we verify it can be instantiated
    and maintains standard Signer attributes like 'sep'.
    """
    # Arrange: Create a mock secret key for the signer
    secret_key = b"super-secret-key"
    sep = b"."

    # Act: Instantiate TimestampSigner
    signer = TimestampSigneler(secret_key, sep=sep)

    # Assert: Verify basic properties are correctly assigned
    assert isinstance(signer, TimestampSigner)
    assert signer.secret_key == secret_key
    assert signer.sep == sep

class TimestampSigneler(TimestampSigner):
    """Helper subclass to allow instantiation for the test 
    if the original class requires specific Signer arguments."""
    def __init__(self, secret_key, sep=b"."):
        # Mocking the base Signer initialization behavior
        self.secret_key = secret_key
        self.sep = sep
        # We mock get_signature to avoid needing the full cryptographic setup
        self.get_signature = MagicMock(return_value=b"mocked_sig")

def test_TimestampSigner_inheritance():
    """Verifies that TimestampSigner correctly inherits from Signer."""
    from .signer import Signer
    signer = TimestampSigneler(b"key")
    assert isinstance(signer, Signer)
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimestampSigner():
    # Since the provided code does not explicitly define a __init__ 
    # and inherits from Signer, we test that it can be instantiated 
    # and maintains expected behavior of a Signer subclass.
    
    secret = b"secret-key"
    sep = b"."
    signer = TimestampSigner(secret=secret, sep=sep)
    
    assert isinstance(signer, TimestampSigner)
    assert signer.secret == secret
    assert signer.sep == sep

    # Test that get_timestamp returns an integer
    ts = signer.get_timestamp()
    assert isinstance(ts, int)

    # Test timestamp_to_datetime conversion
    dt = signer.timestamp_to_datetime(ts)
    assert dt.year == datetime.fromtimestamp(ts, tz=timezone.utc).year
    assert dt.tzinfo == timezone.utc
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
import time
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

# Assuming the classes and exceptions are available in the local scope 
# as per the prompt's constraints on imports.

def test_TimestampSigner_unsign():
    secret = b"secret-key"
    signer = TimestampSigner(secret)
    payload = b"hello-world"
    sep = b"."
    
    # Setup a fixed time for deterministic testing
    fixed_now = 1700000000  # Example epoch timestamp
    
    with patch('time.time', return_value=float(fixed_name := fixed_now)):
        signed_data = signer.sign(payload)

    # 1. Test Basic Unsign (Success)
    assert signer.unsign(signed_data) == payload
    
    # 2. Test Return Timestamp
    val, ts_dt = signer.unsign(signed_data, return_timestamp=True)
    assert val == payload
    assert ts_dt == datetime.fromtimestamp(fixed_now, tz=timezone.utc)

    # 3. Test Max Age Success (Signature is fresh)
    # max_age is 100 seconds, current time is fixed at fixed_now
    assert signer.unsign(signed_data, max_age=100) == payload

    # 4. Test Signature Expired (Too old)
    with patch('time.time', return_value=float(fixed_now + 500)):
        with pytest.raises(SignatureExpired) as excinfo:
            signer.unsign(signed_data, max_age=100)
        assert payload in excinfo.value.payload
        assert "Signature age" in str(excinfo.value)

    # 5. Test Signature Expired (Future timestamp - clock drift/malicious)
    # If the signature has a timestamp from the future relative to current time
    with patch('time.time', return_value=float(fixed_now - 100)):
        with pytest.raises(SignatureExpired):
            signer.unsign(signed_data, max_age=10)

    # 6. Test Bad Signature (Tampered payload)
    tampered_payload = b"tampered" + signed_data[len(payload):]
    # We need to ensure the signature part is actually invalid
    # Since Signer.sign appends a signature, changing the content breaks it
    with pytest.raises(BadSignature):
        signer.unsign(signed_data[:-5] + b"wrong")

    # 7. Test Bad Time Signature (Malformed timestamp)
    # Manually construct a string with correct signature but garbage timestamp
    # Format: payload + sep + base64_timestamp + sep + signature
    # We'll use a validly signed structure but corrupt the timestamp part
    raw_parts = signed_data.split(sep)
    # parts[0] is payload, parts[1] is timestamp, parts[2] is sig
    corrupted_ts_data = raw_parts[0] + sep + b"not-base64-!!!" + sep + raw_parts[2]
    
    with pytest.raises(BadTimeSignature) as excinfo:
        signer.unsign(corrupted_ts_data)
    assert "Malformed timestamp" in str(excinfo.value)

    # 8. Test Missing Separator (Timestamp missing)
    with pytest.raises(BadTimeSignature) as excinfo:
        signer.unsign(b"no_separator_here")
    assert "timestamp missing" in str(excinfo.value)

    # 9. Test Bad Signature with valid timestamp structure
    # This tests the logic where sig_error is caught but we still parse the TS
    with patch.object(signer, 'get_signature', return_value=b"badsig"):
        # Create a new signature that is intentionally broken at the signature level
        broken_sig = signer.sign(payload) 
        # The actual content of broken_sig will have an invalid sig part because we mocked it
        with pytest.raises(BadTimeSignature):
            signer.unsign(broken_sig)

    # 10. Test validate method
    assert signer.validate(signed_data) is True
    with patch('time.time', return_value=float(fixed_now + 1000)):
        assert signer.validate(signed_data, max_age=10) is False
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

# Assuming the classes are in a module named 'module'
# from module import TimedSerializer, TimestampSigner, SignatureExpired, BadSignature

def test_TimedSerializer_loads():
    """
    Tests the loads method of TimedSerializer covering:
    1. Successful loading of payload.
    2. Returning timestamp when return_timestamp=True.
    3. Raising SignatureExpired when max_age is exceeded.
    4. Raising BadSignature when signature is invalid.
    5. Iterating through multiple unsigners (salts).
    """
    # Setup
    serializer = TimedSerializer()
    payload = {"key": "value"}
    # We need to mock the payload loading part, so we'll mock load_payload
    # and the underlying signer behavior.
    
    # Mocking a Signer instance that will be returned by iter_unsigners
    mock_signer = MagicMock()
    now = datetime.now(timezone.utc)
    
    # Test Case 1: Successful load (Standard)
    with patch.object(TimedSerializer, 'iter_unsigners') as mock_iter, \
         patch.object(TimedSerializer, 'load_payload') as mock_load_payload:
        
        mock_iter.return_value = [mock_signer]
        # unsign returns (bytes_payload, timestamp)
        mock_signer.unsign.return_value = (b"encoded_payload", now)
        mock_load_payload.return_value = payload
        
        result = serializer.loads(b"some_signed_data")
        
        assert result == payload
        mock_signer.unsign.assert_called_with(b"some_signed_data", max_age=None, return_timestamp=True)

    # Test Case 2: Successful load with return_timestamp=True
    with patch.object(TimedSerializer, 'iter_unsigners') as mock_iter, \
         patch.object(TimedSerializer, 'load_payload') as mock_load_payload:
        
        mock_iter.return_value = [mock_signer]
        mock_signer.unsign.return_value = (b"encoded_payload", now)
        mock_load_payload.return_value = payload
        
        result, timestamp = serializer.loads(b"some_signed_data", return_timestamp=True)
        
        assert result == payload
        assert timestamp == now

    # Test Case 3: SignatureExpired (Should raise immediately and not try next signer)
    with patch.object(TimedSerializer, 'iter_unsigners') as mock_iter:
        mock_iter.return_value = [mock_signer]
        # Simulate expiration error from the signer
        from .exc import SignatureExpired # Adjust import based on actual structure
        mock_signer.unsign.side_effect = SignatureExpired("Expired", payload=b"data")
        
        with pytest.raises(SignatureExpired):
            serializer.loads(b"some_signed_data", max_age=10)

    # Test Case 4: BadSignature (Should try the next signer in the iterator)
    with patch.object(TimedSerializer, 'iter_unsigners') as mock_iter, \
         patch.object(TimedSerializer, 'load_payload') as mock_load_payload:
        
        mock_signer_2 = MagicMock()
        mock_iter.return_value = [mock_signer, mock_signer_2]
        
        # First signer fails with BadSignature
        from .exc import BadSignature
        mock_signer.unsign.side_effect = BadSignature("Invalid", payload=b"bad")
        
        # Second signer succeeds
        mock_signer_2.unsign.return_value = (b"good_payload", now)
        mock_load_payload.return_value = {"success": True}
        
        result = serializer.loads(b"some_signed_data")
        
        assert result == {"success": True}
        assert mock_signer.unsign.called
        assert mock_signer_2.unsign.called

    # Test Case 5: All signers fail (Should raise the last BadSignature)
    with patch.object(TimedSerializer, 'iter_unsigners') as mock_iter:
        mock_signer_fail = MagicMock()
        mock_iter.return_value = [mock_signer_fail]
        from .exc import BadSignature
        error = BadSignature("Final failure", payload=b"none")
        mock_signer_fail.unsign.side_effect = error
        
        with pytest.raises(BadSignature) as excinfo:
            serializer.loads(b"some_signed_data")
        assert str(excinfo.value) == "Final failure"

    # Test Case 6: Handling salt argument
    with patch.object(TimedSerializer, 'iter_unsigners') as mock_iter, \
         patch.object(TimedSerializer, 'load_payload') as mock_load_payload:
        
        mock_iter.return_value = [mock_signer]
        mock_signer.unsign.return_value = (b"p", now)
        mock_load_payload.return_value = "p"
        
        serializer.loads(b"data", salt=b"my_salt")
        mock_iter.assert_called_with(salt=b"my_salt")
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

# Assuming the classes and exceptions are available in the namespace
# as per the provided code snippet.

def test_TimestampSigner_unsign():
    secret = b"secret-key"
    sep = b"."
    signer = TimestampSigner(secret, sep=sep)
    
    payload = b"hello-world"
    
    # 1. Test successful unsign (standard)
    signed_val = signer.sign(payload)
    assert signer.unsign(signed_val) == payload

    # 2. Test successful unsign with return_timestamp=True
    value, ts_dt = signer.unsig(signed_val, return_timestamp=True)
    assert value == payload
    assert isinstance(ts_dt, datetime)
    assert ts_dt.tzinfo == timezone.utc

    # 3. Test SignatureExpired (Too old)
    # We mock time.time() to simulate the passage of time
    fixed_now = 1000000
    with patch('time.time', return_value=fixed_now):
        signed_old = signer.sign(payload)
    
    # Move time forward past max_age
    with patch('time.time', return_value=fixed_now + 100):
        with pytest.raises(SignatureExpired) as excinfo:
            signer.unsign(signed_old, max_age=50)
        assert payload in excinfo.value.payload

    # 4. Test SignatureExpired (Future timestamp / Clock drift)
    with patch('time.time', return_value=fixed_now):
        signed_future = signer.sign(payload)
    
    # Move time backward
    with patch('time.time', return_value=fixed_now - 100):
        with pytest.raises(SignatureExpired):
            signer.unsign(signed_future, max_age=50)

    # 5. Test BadSignature (Tampered payload)
    tampered_val = signed_val.replace(b"hello", b"bad")
    with pytest.raises(BadSignature):
        signer.unsign(tampered_val)

    # 6. Test BadTimeSignature (Malformed timestamp/structure)
    # Manually create a string that has the separator but invalid base64 for timestamp
    malformed_ts = payload + sep + b"not-base64-valid-timestamp!!!"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_ts)

    # 7. Test BadTimeSignature (Missing separator entirely)
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"no-separator-here")

    # 8. Test BadSignature with valid timestamp structure
    # We create a validly formatted string but with an invalid signature part
    # payload + sep + timestamp + sep + bad_sig
    with patch('time.time', return_value=fixed_now):
        ts_part = base64_encode(int_to_bytes(fixed_now))
        bad_sig_val = payload + sep + ts_part + sep + b"invalid-signature"
        with pytest.raises(BadSignature):
            signer.unsign(bad_sig_val)

    # 9. Test validate() method
    assert signer.validate(signed_val) is True
    assert signer.validate(tampered_val) is False
    
    with patch('time.time', return_value=fixed_now + 100):
        assert signer.validate(signed_old, max_age=50) is False
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

# Assuming the classes are in a module named 'module'
# from module import TimedSerializer, TimestampSigner, BadSignature, SignatureExpired

def test_TimedSerializer_loads():
    """
    Tests the loads method of TimedSerializer covering:
    - Successful loading (returning payload)
    - Successful loading with return_timestamp=True
    - Handling of SignatureExpired (should raise error and not try next signer)
    - Handling of BadSignature (should try next signer)
    - Final failure when all signers fail
    """
    # Setup common values
    payload = b"secret_data"
    encoded_payload = b"ZW5jb2RlZF9kYXRh"  # base64 for 'encoded_data'
    timestamp_val = 1600000000
    dt_val = datetime.fromtimestamp(timestamp_val, tz=timezone.utc)
    sep = b"."

    # Create a mock Serializer/TimedSerializer behavior
    # We need to mock the serializer's load_payload and iter_unsigners
    serializer = TimedSerializer()
    serializer.load_payload = MagicMock(return_value=payload)

    # Setup Mock Signer 1 (Success)
    signer1 = MagicMock(spec=TimestampSigner)
    signer1.unsign.return_value = (encoded_payload, dt_val)
    
    # Setup Mock Signer 2 (Bad Signature - should be skipped)
    signer2 = MagicMock(spec=TimestampSigner)
    signer2.unsign.side_effect = BadSignature("Invalid", payload=b"corrupt")

    # Setup Mock Signer 3 (Expired - should stop iteration)
    signer3 = MagicMock(spec=TimestampSigner)
    signer3.unsign.side_effect = SignatureExpired("Expired", payload=encoded_payload, date_signed=dt_val)

    # Mock iter_unsigners to return our sequence of signers
    serializer.iter_unsigners = MagicMock(return_value=[signer1, signer2, signer3])

    # --- Test Case 1: Successful load (Standard) ---
    # We use a dummy signed string
    signed_str = b"payload.timestamp.signature"
    result = serializer.loads(signed_str)
    assert result == payload
    signer1.unsign.assert_called_with(signed_str, max_age=None, return_timestamp=True)

    # --- Test Case 2: Successful load with return_timestamp=True ---
    result_ts = serializer.loads(signed_str, return_timestamp=True)
    assert result_ts == (payload, dt_val)

    # --- Test Case 3: SignatureExpired (Stops iteration immediately) ---
    # We reset mocks and provide a signer that fails with expiration first
    signer1.unsign.side_effect = SignatureExpired("Expired", payload=encoded_payload, date_signed=dt_val)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_str)
    # Verify it didn't proceed to signer2
    assert signer2.unsign.call_count == 0

    # --- Test Case 4: BadSignature (Iterates to next signer, then fails) ---
    # Resetting signers for a sequence: Signer1 (Bad) -> Signer2 (Bad) -> Raises final exception
    signer1.unsign.side_effect = BadSignature("Bad 1")
    signer2.unsign.side_effect = BadSignature("Bad 2")
    # signer3 is already set to raise SignatureExpired, but we want to test the loop exhaustion
    
    with pytest.raises(BadSignature) as excinfo:
        serializer.loads(signed_str)
    assert "Bad 2" in str(excinfo.value)
    
    # --- Test Case 5: Verify max_age is passed through ---
    signer1.unsign.side_effect = None
    signer1.unsign.return_value = (encoded_payload, dt_val)
    serializer.loads(signed_str, max_age=3600)
    signer1.unsign.assert_called_with(signed_str, max_age=3600, return_timestamp=True)

    # --- Test Case 6: Verify salt is passed through ---
    serializer.loads(signed_str, salt="my_salt")
    serializer.iter_unsigners.assert_called_with(salt="my_salt")
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimedSerializer():
    """
    Tests the initialization and basic properties of TimedSerializer.
    Since the constructor is inherited from Serializer, we verify 
    the default signer assignment and class structure.
    """
    # Arrange: Create a mock serializer/signer setup if needed, 
    # but here we test the actual class instantiation.
    # We assume 'json' or similar serialization logic is available via its parent.
    
    # Act
    serializer = TimedSerializer()

    # Assert
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer == TimestampSigner
    
    # Verify that it inherits the expected behavior of a Serializer
    # (Checking if the characteristic method exists and is part of the class)
    assert hasattr(serializer, 'loads')
    assert hasattr(serializer, 'dumps')
    assert hasattr(serializer, 'iter_unsigners')

def test_TimedSerializer_inheritance():
    """
    Verify that TimedSerializer correctly identifies its default signer 
    as TimestampSigner.
    """
    serializer = TimedSerializer()
    signer = serializer.get_signer()
    
    assert isinstance(signer, TimestampSigner)
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimedSerializer():
    """
    Tests the initialization and properties of TimedSerializer.
    Since the constructor is inherited from Serializer, we verify 
    that it correctly sets up the default signer as TimestampSigner.
    """
    # Mocking dependency components to avoid complex setup for a unit test
    # focused on the class structure/constructor.
    mock_serializer = MagicMock(spec=TimedSerializer)
    
    # Test that TimedSerializer specifies TimestampSigner as its default signer
    assert TimedSerializer.default_signer is TimestampSigneler

    # Verification of inheritance and type identity
    instance = TimedSerializer()
    assert isinstance(instance, TimedSerializer)
    
    # Verify that it inherits from Serializer (via the class definition)
    from .serializer import Serializer
    assert issubclass(TimedSerializer, Serializer)

def test_TimedSerializer_properties():
    """
    Verifies properties of the TimedSerializer class itself.
    """
    # Test that the default signer attribute is indeed the TimestampSigner class
    assert TimedSerializer.default_signer == TimestampSigner

    # Verify that iter_unsigners returns an iterator (as per implementation)
    instance = TimedSerializer()
    # We mock the super().iter_unsigners behavior because it's a complex dependency
    with MagicMock() as mock_super:
        # Checking if the method exists and is part of the class
        assert hasattr(instance, 'iter_unsigners')
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimestampSigner():
    # Mocking the base Signer class dependencies since we are testing the constructor
    # and basic instantiation of TimestampSigner.
    # In a real scenario, we'd use an actual secret key.
    secret = b"secret-key"
    signer = TimestampSigner(secret)

    # Verify the instance is of correct type
    assert isinstance(signer, TimestampSigneler)
    assert isinstance(signer, Signer)

    # Verify properties inherited from Signer are accessible
    assert signer.sep == b"."

    # Test that get_timestamp returns an integer
    ts = signer.get_timestamp()
    assert isinstance(ts, int)

    # Test timestamp_to_datetime conversion logic
    dt = signer.timestamp_to_datetime(ts)
    assert dt.year > 2000  # Sanity check for valid epoch
    assert dt.tzinfo is not None  # Should be timezone aware (UTC)

    # Verify that the constructor accepts bytes and strings if handled by Signer logic
    # (Assuming Signer's __init__ handles the secret key assignment)
    signer_str = TimestampSigner("string-key")
    assert isinstance(signer_str, TimestampSigner)
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

# Assuming these are available in the testing environment as per instructions
# from .signer import TimestampSigner
# from .exc import BadSignature, BadTimeSignature, SignatureExpired
# from .encoding import base64_encode, int_to_bytes

def test_TimestampSigner_unsign():
    secret = "secret-key"
    sep = "."
    signer = TimestampSigner(secret)
    
    payload = b"test_payload"
    fixed_now = 1700000000  # A fixed timestamp

    with patch("time.time", return_value=float(fixed_now)):
        signed_val = signer.sign(payload)

    # 1. Test successful unsign (default)
    assert signer.unsign(signed_val) == payload

    # 2. Test successful unsign with return_timestamp=True
    val, ts_dt = signer.unsign(signed_val, return_timestamp=True)
    assert val == payload
    assert ts_dt == datetime.fromtimestamp(fixed_now, tz=timezone.utc)

    # 3. Test max_age within limit
    # Current time is fixed_now. Signature was created at fixed_now.
    # Max age 100 seconds should pass.
    assert signer.unsign(signed_val, max_age=100) == payload

    # 4. Test max_age expired (Signature is too old)
    with patch("time.time", return_value=float(fixed_now + 200)):
        with pytest.raises(SignatureExpired) as excinfo:
            signer.unsign(signed_val, max_age=100)
        assert payload in excinfo.value.payload
        assert "Signature age" in str(excinfo.value)

    # 5. Test max_age not allowed to be in the future (Signature is from the future)
    with patch("time.time", return_value=float(fixed_now - 100)):
        with pytest.raises(SignatureExpired) as excinfo:
            signer.unsign(signed_val, max_age=100)
        assert "age -100" in str(excinfo.value)

    # 6. Test BadSignature (Tampered payload)
    tampered_payload = b"tampered" + signed_val[len(payload):]
    with pytest.raises(BadSignature):
        signer.unsign(tampered_payload)

    # 7. Test BadTimeSignature (Malformed timestamp structure)
    # Create a value that has the separator but invalid base64/int payload in timestamp part
    malformed_ts = payload + b"." + b"not-base64-at-all!!!"
    with pytest.raises(BadTimeSignature) as excinfo:
        signer.unsign(malformed_ts)
    assert "Malformed timestamp" in str(excinfo.value)

    # 8. Test BadTimeSignature (Missing separator/timestamp entirely)
    no_ts = payload + b".something" # Separator exists but structure is wrong for rsplit logic
    # Note: The implementation uses rsplit(sep, 1). If sep not in result, it raises BadTimeSignature.
    with pytest.raises(BadTimeSignature):
        signer.unsign(payload)

    # 9. Test BadSignature but with recoverable timestamp (Checking if ts_dt is attached to error)
    # We manually construct a bad signature that contains a valid timestamp part
    ts_bytes = base64_encode(int_to_bytes(fixed_now))
    bad_sig_with_ts = payload + b"." + ts_bytes + b"." + b"wrong-signature"
    
    # Since we can't easily trigger the specific super().unsign failure without a real Signer,
    # we rely on the fact that if we pass a value that fails signature check but has 
    # valid timestamp segments, it should attempt to parse the date.
    with pytest.raises(BadTimeSignature) as excinfo:
        signer.unsign(bad_sig_with_ts)
    assert excinfo.value.date_signed == datetime.fromtimestamp(fixed_now, tz=timezone.utc)

    # 10. Test validate method
    assert signer.validate(signed_val) is True
    with patch("time.time", return_value=float(fixed_now + 500)):
        assert signer.validate(signed_val, max_age=10) is False
    assert signer.validate(b"invalid-data") is False
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimedSerializer():
    """
    Tests the construction and basic characteristics of TimedSerializer.
    Since the constructor is inherited from Serializer, we verify 
    it initializes correctly with expected default attributes.
    """
    # Mocking dependencies for a controlled test environment
    # We assume a standard setup where we can instantiate it without complex side effects
    serializer = TimedSerializer()

    # Test that the instance is of the correct type
    assert isinstance(serializer, TimendedSerializer)
    assert isinstance(serializer, Serializer)

    # Verify the default signer is indeed TimestampSigner as specified in the class attribute
    assert serializer.default_signer is TimestampSigner

    # Check if it has the characteristic methods of a Serializer/TimedSerializer
    assert hasattr(serializer, 'dumps')
    assert hasattr(serializer, 'loads')
    assert hasattr(serializer, 'loads_unsafe')

    # Verify that iter_unsigners returns an iterator of TimestampSigner
    # We use a dummy salt to trigger the iteration logic
    unsigners = list(serializer.iter_unsigners(salt=b"test_salt"))
    for signer in unsigners:
        assert isinstance(signer, TimestampSigner)

@pytest.mark.parametrize("serializer_class", [TimedSerializer])
def test_TimedSerializer_inheritance(serializer_class):
    """Verifies the class hierarchy for TimedSerializer."""
    instance = serializer_class()
    assert issubclass(type(instance), Serializer)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
import time
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

# Assuming the classes and exceptions are available in the namespace as per instructions
# from .signer import TimestampSigner
# from .exc import BadSignature, BadTimeSignature, SignatureExpired
# from .encoding import base64_encode, int_to_bytes

def test_TimestampSigner_unsign():
    secret = b"secret-key"
    sep = b"."
    signer = TimestampSigner(secret)
    
    payload = b"hello-world"
    fixed_now = 1700000000  # Fixed timestamp for testing
    
    with patch("time.time", return_value=float(fixed_now)):
        signed_value = signer.sign(payload)

    # 1. Test successful unsign (returns bytes)
    assert signer.unsign(signed_value) == payload

    # 2. Test successful unsign with return_timestamp=True
    val, ts_dt = signer.unsign(signed_value, return_timestamp=True)
    assert val == payload
    assert ts_dt == datetime.fromtimestamp(fixed_now, tz=timezone.utc)

    # 3. Test max_age validation (within limit)
    assert signer.validate(signed_value, max_age=100) is True
    assert signer.unsign(signed_value, max_age=100) == payload

    # 4. Test max_age validation (expired)
    with pytest.raises(SignatureExpired) as excinfo:
        signer.unsign(signed_value, max_age=10)
    assert "Signature age" in str(excinfo.value)
    assert excinfo.value.payload == payload

    # 5. Test max_age validation (future timestamp - edge case)
    with patch("time.time", return_value=float(fixed_now - 100)):
        with pytest.raises(SignatureExpired) as excinfo:
            signer.unsign(signed_value, max_age=10)
    assert "Signature age" in str(excinfo.value)

    # 6. Test BadSignature (tampered payload)
    tampered_payload = b"tampered" + signed_value[len(b"hello-world"):]
    with pytest.raises(BadSignature):
        signer.unsign(tampered_payload)

    # 7. Test BadTimeSignature (malformed timestamp/separator structure)
    # Create a value that has the separator but invalid base64 in timestamp part
    bad_ts_structure = payload + sep + b"not-base64-!!!"
    # We need to sign it so the signature check passes or we handle the BadSignature logic
    # In TimestampSigner.unsign, if sig_error is caught, it tries to parse timestamp anyway.
    with pytest.raises(BadTimeSignature):
        # Manually constructing a bad structure that fails the separator/timestamp split logic
        signer.unsign(payload + b"no-separator")

    # 8. Test BadTimeSignature (corrupt signature but valid timestamp part)
    # We simulate a case where the signature is invalid, but we can still parse the timestamp
    # This triggers the 'if sig_error is not None' block in unsign
    with patch.object(signer, 'get_signature', return_value=b"wrong-sig"):
        # Re-sign with bad signature manually
        ts_bytes = signer.sign(payload) # contains valid TS and signature
        # Replace the signature part with garbage
        parts = ts_bytes.split(sep)
        # parts[0] is payload, parts[1] is timestamp, parts[2] is sig (if 3 parts exist)
        # but sign() produces: value + sep + timestamp + sep + sig
        # Let's just corrupt the signature tail
        corrupt_sig = payload + sep + b"timestamp_encoded_here" + sep + b"badsignature"
        # Since we can't easily predict encoding, let's use the signer to generate a valid one 
        # and then break only the signature part.
    
    # 9. Test BadTimeSignature (Malformed timestamp bytes)
    # We use a payload that isn't actually signed by this key to trigger BadSignature
    # but contains a separator and non-integer timestamp
    bad_format = b"payload" + sep + b"invalid_base64_content"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_format)

    # 10. Test validate returns False on BadSignature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid-signature-entirely")
    assert signer.validate(b"invalid-signature-entirely") is False
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

# Assuming the classes are in a module named 'module'
# from module import TimedSerializer, TimestampSigner, BadSignature, SignatureExpired

def test_TimedSerializer_loads():
    """
    Tests the loads method of TimedSerializer covering successful loads,
    expired signatures, and invalid signatures.
    """
    # Setup a mock serializer that mimics JSON behavior for simplicity
    class MockSerializer(TimedSerializer):
        def dumps(self, value):
            return b"encoded_data"  # Simplified
        def load_payload(self, payload):
            return payload.decode('utf-8')

    serializer = MockSerializer()
    signer = TimestampSigner(secret='secret')
    
    # We need to patch iter_unsigners because TimedSerializer inherits from Serializer
    # and we want to control the Signer returned.
    with patch.object(MockSerializer, 'iter_unsigners') as mock_iter:
        
        # --- Case 1: Successful Load (No max_age) ---
        payload_data = "hello"
        signed_val = signer.sign(payload_data)
        mock_iter.return_value = [signer]
        
        result = serializer.loads(signed_val)
        assert result == payload_data

        # --- Case 2: Successful Load (With return_timestamp=True) ---
        result, ts = serializer.loads(signed_val, return_timestamp=True)
        assert result == payload_data
        assert isinstance(ts, datetime)
        assert ts.tzinfo == timezone.utc

        # --- Case 3: Successful Load (With max_age allowed) ---
        # Create a signature from 10 seconds ago
        with patch('time.time', return_value=datetime.now().timestamp() - 10):
            signed_val_old = signer.sign(payload_data)
        
        # Now check with max_age=20 (should pass)
        with patch('time.time', return_value=datetime.now().timestamp()):
            result = serializer.loads(signed_val_old, max_age=20)
            assert result == payload_data

        # --- Case 4: Signature Expired (max_age exceeded) ---
        # Create a signature from 60 seconds ago
        with patch('time.time', return_value=datetime.now().timestamp() - 60):
            signed_val_old = signer.sign(payload_data)
            
        # Check with max_age=10 (should raise SignatureExpired)
        with patch('time.time', return_value=datetime.now().timestamp()):
            with pytest.raises(SignatureExpired) as excinfo:
                serializer.loads(signed_val_old, max_age=10)
            assert payload_data in excinfo.value.payload

        # --- Case 5: Bad Signature (Invalid signature) ---
        invalid_sig = b"wrong_signature_format"
        mock_iter.return_value = [signer]
        with pytest.raises(BadSignature):
            serializer.loads(invalid_sig)

        # --- Case 6: Multiple Unsigners (Fallback mechanism) ---
        # First signer fails, second signer succeeds
        signer_bad = TimestampSigner(secret='wrong')
        signer_good = TimestampSigner(secret='secret')
        
        signed_val_good = signer_good.sign(payload_data)
        mock_iter.return_value = [signer_bad, signer_good]
        
        # Should skip the bad one and return data from the good one
        result = serializer.loads(signed_val_good)
        assert result == payload_data

        # --- Case 7: All Unsigners fail ---
        mock_iter.return_value = [signer_bad]
        with pytest.raises(BadSignature):
            serializer.loads(signed_val_good)

        # --- Case 8: Signature Expired in a chain (Should NOT try next signer) ---
        # If the first signer has an expired signature, it raises SignatureExpired.
        # The implementation says: "Do not try the next signer."
        with patch('time.time', return_value=datetime.now().timestamp() - 100):
            expired_signed_val = signer_good.sign(payload_data)
        
        mock_iter.return_value = [signer_bad, signer_good] # Assuming signer_bad is just a bad signer
        # We simulate that the first signer in the list found an expired token
        # To trigger the 'except SignatureExpired: raise' block specifically
        with patch.object(TimestampSigner, 'unsign', side_effect=[SignatureExpired("expired", payload=b""), b"valid"]):
             # This is a bit meta but tests the specific logic in loads() 
             # where it stops iteration on SignatureExpired
             with pytest.raises(SignatureExpired):
                 serializer.loads(signed_val_good, max_age=1)

```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimedSerializer():
    """
    Tests the constructor and basic initialization behavior 
    of the TimedSerializer class.
    """
    # Mocking a Serializer dependency since TimedSerializer inherits from it
    # and requires a load_payload method for its logic to function.
    mock_serializer = MagicMock(spec=TimedSerializer)
    
    # Verify that we can instantiate the class. 
    # Since TimedSerializer's constructor is inherited, 
    # we test the standard initialization.
    try:
        serializer = TimedSerializer()
        assert isinstance(serializer, TimedSerializer)
    except Exception as e:
        pytest.fail(f"TimedSerializer instantiation failed with error: {e}")

    # Verify the default signer is indeed TimestampSigner
    assert serializer.default_signer is TimestampSigner

    # Test that it maintains the expected class hierarchy/identity
    assert issubclass(TimedSerializer, Serializer)
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

@pytest.mark.parametrize("secret", [b"secret", "secret"])
@pytest.mark.parametrize("sep", [b".", "."])
def test_TimestampSigner_unsign(secret, sep):
    signer = TimestampSigner(secret=secret, sep=sep)
    payload = b"hello world"
    
    # 1. Test basic successful unsigning (no timestamp return requested)
    signed_value = signer.sign(payload)
    assert signer.unsign(signed_value) == payload

    # 2. Test successful unsigning with return_timestamp=True
    val, ts_dt = signer.unsign(signed_value, return_timestamp=True)
    assert val == payload
    assert isinstance(ts_dt, datetime)
    assert ts_dt.tzinfo == timezone.utc

    # 3. Test max_age validation (Success)
    # We don't mock time here to ensure real-time logic works for a small window
    assert signer.unsign(signed_value, max_age=10) == payload

    # 4. Test max_age validation (Failure - Expired)
    with patch.object(TimestampSigner, 'get_timestamp') as mock_now:
        # Get the timestamp embedded in the signature manually for calculation
        # The signature format is: payload + sep + ts_b64 + sep + sig
        parts = signed_value.split(want_bytes(sep))
        ts_bytes = base64_decode(parts[1])
        original_ts = bytes_to_int(ts_bytes)
        
        # Fast forward time to exceed max_age (e.g., 100 seconds later)
        mock_now.return_value = original_ts + 100
        with pytest.raises(SignatureExpired) as excinfo:
            signer.unsign(signed_value, max_age=50)
        assert "Signature age" in str(excinfo.value)

    # 5. Test max_age validation (Failure - Future timestamp/Negative age)
    with patch.object(TimestampSigner, 'get_timestamp') as mock_now:
        parts = signed_value.split(want_bytes(sep))
        ts_bytes = base64_decode(parts[1])
        original_ts = bytes_to_int(ts_bytes)
        
        # Set current time to BEFORE the signature was created
        mock_now.return_value = original_ts - 10
        with pytest.raises(SignatureExpired) as excinfo:
            signer.unsign(signed_value, max_age=50)
        assert "Signature age -10 < 0 seconds" in str(excinfo.value)

    # 6. Test BadSignature (Tampered payload)
    tampered_payload = b"tampered" + want_bytes(sep) + signed_value.split(want_bytes(sep), 1)[1]
    with pytest.raises(BadSignature):
        signer.unsign(tampered_payload)

    # 7. Test Malformed Timestamp (Corrupted timestamp part)
    # Replace the base64 timestamp with garbage
    corrupt_ts = want_bytes(sep) + b"not-base64!!!" + want_bytes(sep) + b"signature"
    corrupt_value = payload + corrupt_ts
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(corrupt_value)

    # 8. Test Missing Timestamp (No separator in the result part)
    no_ts_value = payload + want_bytes(sep) + b"just-some-data"
    # This will fail because 'unsign' expects at least two separators for payload|ts|sig
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_ts_value)

    # 9. Test Signature Error with recoverable timestamp info
    # We simulate a BadSignature error but ensure the timestamp part is readable
    # so that the exception contains the 'date_signed' attribute.
    with patch.object(Signer, 'unsign') as mock_super_unsign:
        ts_val = b"123456789" # valid int
        ts_b64 = base64_encode(int_to_bytes(123456789))
        # Construct a value that looks like payload + sep + ts_b64
        fake_signed = payload + want_bytes(sep) + ts_b64 + want_bytes(sep) + b"sig"
        
        # Force the super().unsign to raise BadSignature
        mock_super_unsign.side_effect = BadSignature("invalid sig", payload=payload)
        
        try:
            signer.unsign(fake_signed)
        except BadSignature as e:
            assert e.payload == payload
            # Check if the timestamp was successfully extracted even during error
            assert e.date_signed == datetime.fromtimestamp(123456789, tz=timezone.utc)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimedSerializer():
    """
    Tests the initialization and basic structure of TimedSerializer.
    Since the provided code doesn't define a custom __init__, 
    we verify it inherits correctly from Serializer and maintains 
    the expected class attributes.
    """
    # Mocking dependencies required for Serializer/Signer logic if necessary,
    # but since we are testing the constructor of TimedSerializer:
    
    class MockSerializer(TimedSerializer):
        def __init__(self, secret="secret"):
            super().__init__(secret=secret)

    # Test initialization with a secret
    serializer = MockSerializer(secret="test_secret")
    
    assert isinstance(serializer, TimorestSerializer if 'TimorestSerializer' in globals() else TimedSerializer)
    assert serializer.default_signer == TimestampSigner
    
    # Verify it can be instantiated without arguments (using default logic from Serializer)
    try:
        empty_serializer = TimedSerializer()
        assert isinstance(empty_serializer, TimedSerializer)
    except Exception as e:
        pytest.fail(f"TimedSerializer constructor failed: {e}")

def test_TimedSerializer_inheritance():
    """Verify the class hierarchy."""
    assert issubclass(TimedSerializer, Serializer)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

class TestTimedSerializerLoads:
    @pytest.fixture
    def serializer(self):
        # Assuming JSON serializer for testing purposes as a concrete implementation
        from .serializer import Serializer
        class MockSerializer(Serializer):
            def dumps(self, value): return b"encoded_value"
            def loads(self, s, **kwargs): pass # Overridden in target class
            def load_payload(self, payload): return payload.decode('utf-8')
        return MockSerializer()

    @pytest.fixture
    def signer(self):
        signer = MagicMock(spec=TimestampSigner)
        signer.sep = b"."
        return signer

    def test_TimedSerializer_loads_success(self, serializer, signer):
        """Test successful loading of a payload with timestamp."""
        payload_bytes = b"hello_world"
        timestamp = datetime(2023, 1, 1, tzinfo=timezone.utc)
        
        # Mock the signer returned by iter_unsigners
        with patch.object(serializer, 'iter_unsigners', return_value=[signer]):
            # Configure signer.unsign to return (payload, timestamp)
            signer.unsign.return_value = (payload_bytes, timestamp)
            
            result = serializer.loads(b"signed_data")
            
            assert result == "hello_world"
            signer.unsign.assert_called_once_with(b"signed_data", max_age=None, return_timestamp=True)

    def test_TimedSerializer_loads_with_timestamp_return(self, serializer, signer):
        """Test loading where return_timestamp=True is requested."""
        payload_bytes = b"hello_world"
        timestamp = datetime(2023, 1, 1, tzinfo=timezone.utc)
        
        with patch.object(serializer, 'iter_unsigners', return_value=[signer]):
            signer.unsign.return_value = (payload_bytes, timestamp)
            
            result, returned_ts = serializer.loads(b"signed_data", return_timestamp=True)
            
            assert result == "hello_world"
            assert returned_ts == timestamp

    def test_TimedSerializer_loads_signature_expired(self, serializer, signer):
        """Test that SignatureExpired is raised and stops iteration."""
        from .exc import SignatureExpired
        
        with patch.object(serializer, 'iter_unsigners', return_value=[signer]):
            # Simulate expired signature
            signer.unsign.side_effect = SignatureExpired("expired", payload=b"old_data")
            
            with pytest.raises(SignatureExpired):
                serializer.loads(b"signed_data")

    def test_TimedSerializer_loads_bad_signature_next_signer(self, serializer, signer):
        """Test that it tries the next signer if one fails with BadSignature."""
        from .exc import BadSignature
        
        signer2 = MagicMock(spec=TimestampSigner)
        signer2.sep = b"."
        
        with patch.object(serializer, 'iter_unsigners', return_value=[signer, signer2]):
            # First signer fails with BadSignature
            signer.unsign.side_effect = BadSignature("bad sig", payload=b"payload1")
            # Second signer succeeds
            signer2.unsign.return_value = (b"good_payload", datetime.now(timezone.utc))
            
            result = serializer.loads(b"signed_data")
            assert result == "good_payload"

    def test_TimedSerializer_loads_all_signers_fail(self, serializer, signer):
        """Test that BadSignature is raised if all signers fail."""
        from .exc import BadSignature
        
        signer2 = MagicMock(spec=TimestampSigners)
        signer2.sep = b"."
        
        with patch.object(serializer, 'iter_unsigners', return_value=[signer, signer2]):
            signer.unsign.side_effect = BadSignature("error 1", payload=b"p1")
            signer2.unsign.side_effect = BadSignature("error 2", payload=b"p2")
            
            with pytest.raises(BadSignature) as excinfo:
                serializer.loads(b"signed_data")
            assert "error 2" in str(excinfo.value)

    def test_TimedSerializer_loads_max_age_passed_to_signer(self, serializer, signer):
        """Test that max_age parameter is correctly forwarded."""
        with patch.object(serializer, 'iter_unsigners', return_value=[signer]):
            signer.unsign.return_value = (b"data", datetime.now(timezone.utc))
            
            serializer.loads(b"signed_data", max_age=60)
            
            signer.unsign.assert_called_once_with(b"signed_data", max_age=60, return_timestamp=True)
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimestampSigner():
    """
    Tests the initialization and basic properties of TimestampSigner.
    Since TimestampSigner inherits from Signer, we verify it correctly 
    inherits/uses its configuration and maintains expected attributes.
    """
    secret_key = b"secret-key"
    sep = b"."
    signer = TimestampSigner(secret_key, sep=sep)

    # Verify instance type
    assert isinstance(signer, TimestampSigner)
    
    # Verify inherited properties from Signer
    assert signer.secret_key == secret_key
    assert signer.sep == sep

    # Verify that we can call methods existing on the class 
    # (Ensures constructor didn't break basic functionality)
    ts = signer.get_timestamp()
    assert isinstance(ts, int)
    
    dt = signer.timestamp_to_datetime(ts)
    assert dt.tzinfo is not None  # Ensure it is aware as per docstring
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

# Assuming the classes and exceptions are available in the namespace as per instructions
# We will mock the dependencies required for a functional test of TimedSerializer.loads

def test_TimedSerializer_loads():
    """
    Tests the loads method of TimedSerializer covering:
    1. Successful decryption/unsigning with timestamp return.
    2. Successful decryption without timestamp return.
    3. Raising SignatureExpired when max_age is exceeded.
    4. Raising BadSignature when signature is invalid.
    5. Handling multiple unsigners (iterating through them).
    """
    
    # Setup Mocks
    mock_signer = MagicMock(spec=TimestampSigner)
    # Mocking the payload content and timestamp
    payload_bytes = b"decoded_payload"
    fixed_now = datetime(202ps, 1, 1, tzinfo=timezone.utc)
    ts_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # We need to mock the Serializer's load_payload method via a subclass or patch
    class MockTimedSerializer(TimedSerializer):
        def load_payload(self, payload: bytes) -> any:
            return payload.decode('utf-8')

    serializer = MockTimedSerializer()
    # Inject our mock signer into the serializer's logic 
    # (In a real scenario, we'd control the salt/signer factory)
    serializer.iter_unsigners = MagicMock(return_value=[mock_signer])

    # --- Test Case 1: Successful loads with return_timestamp=True ---
    mock_signer.unsign.return_value = (b"decoded_payload", ts_dt)
    result, result_ts = serializer.loads(b"signed_data", return_timestamp=True)
    assert result == "decoded_payload"
    assert result_ts == ts_dt
    mock_signer.unsign.assert_called_with(b"signed_data", max_age=None, return_timestamp=True)

    # --- Test Case 2: Successful loads with return_timestamp=False ---
    result = serializer.loads(b"signed_data", return_timestamp=False)
    assert result == "decoded_payload"

    # --- Test Case 3: SignatureExpired should raise SignatureExpired and stop iteration ---
    from .exc import SignatureExpired
    mock_signer.unsign.side_effect = SignatureExpired("Expired", payload=b"old")
    with pytest.raises(SignatureExpired):
        serializer.loads(b"expired_data", max_age=10)
    # Verify it didn't try another signer because SignatureExpired is re-raised immediately
    assert mock_signer.unsign.call_count == 2 # (Case 1, Case 2) + this call

    # --- Test Case 4: BadSignature should iterate to the next signer ---
    from .exc import BadSignature
    mock_signer_2 = MagicMock(spec=TimestampSigner)
    serializer.iter_unsigners.return_value = [mock_signer, mock_signer_2]
    
    # First signer fails with BadSignature
    mock_signer.unsign.side_effect = BadSignature("Bad signature")
    # Second signer succeeds
    mock_signer_2.unsign.return_value = (b"second_payload", ts_dt)
    
    result = serializer.loads(b"data_for_two_signers")
    assert result == "second_payload"

    # --- Test Case 5: All signers fail with BadSignature ---
    mock_signer_2.unsign.side_effect = BadSignature("Final failure")
    with pytest.raises(BadSignature) as excinfo:
        serializer.loads(b"data_for_all_failing_signers")
    assert "Final failure" in str(excinfo.value)

    # --- Test Case 6: Max Age validation (Logic inside unsign, but tested via loads) ---
    # Since 'loads' passes max_age to 'unsign', we test if the exception propagates correctly
    from .exc import SignatureExpired
    mock_signer_3 = MagicMock(spec=TimestampSigner)
    serializer.iter_unsigners.return_value = [mock_signer_3]
    mock_signer_3.unsign.side_effect = SignatureExpired("Expired", payload=b"payload")
    
    with pytest.raises(SignatureExpired):
        serializer.loads(b"data", max_age=5)

```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

class MockSerializer(TimedSerializer):
    """A minimal implementation of Serializer to test loads."""
    def dumps(self, value: any) -> bytes:
        return b"dummy_signed_value"
    
    def load_payload(self, payload: bytes) -> any:
        return payload.decode("utf-8")

@pytest.mark.parametrize("payload_str, expected_output", [
    ("hello", b"hello"),
    ("world", b"world"),
])
def test_TimedSerializer_loads(payload_str, expected_output):
    """Tests the loads method of TimedSerializer with various scenarios."""
    
    # Setup components
    serializer = MockSerializer()
    signer = MagicMock(spec=TimestampSigner)
    
    # Mocking iter_unsigners to return our mock signer
    with patch.object(TimedSerializer, 'iter_unsigning', return_value=[signer]):
        # We use a real TimestampSigner for logic testing but control the clock
        real_signer = TimestampSigner(secret="secret")
        serializer.iter_unsigners = MagicMock(return_value=[real_signer])

        # 1. Test successful loading (no timestamp requested)
        signed_val = real_signer.sign(payload_str)
        result = serializer.loads(signed_val)
        assert result == payload_str

        # 2. Test successful loading with return_timestamp=True
        result, ts_dt = serializer.loads(signed_val, return_timestamp=True)
        assert result == payload_str
        assert isinstance(ts_dt, datetime)
        assert ts_dt.tzinfo == timezone.utc

        # 3. Test successful loading with max_age (within limit)
        # We don't need to mock time if we use a fresh signature
        result = serializer.loads(signed_val, max_age=10)
        assert result == payload_str

        # 4. Test SignatureExpired exception
        with pytest.raises(SignatureExpired):
            # Create an old signature by mocking get_timestamp in the signer
            with patch.object(TimestampSigner, 'get_timestamp', return_value=int(datetime.now().timestamp()) - 100):
                old_signed_val = real_signer.sign(payload_str)
                serializer.loads(old_signed_val, max_age=50)

        # 5. Test BadSignature exception (tampered payload)
        tampered_val = signed_val.replace(b"h", b"z")
        with pytest.raises(BadSignature):
            serializer.loads(tampered_val)

        # 6. Test loading with salt (if provided to iter_unsigners)
        # This assumes the signer implementation handles salt correctly via its base class
        with patch.object(TimestampSigner, 'unsign') as mock_unsign:
            mock_unsign.return_value = (b"payload", datetime.now(timezone.utc))
            # Mocking the payload loading to return what we want
            with patch.object(MockSerializer, 'load_payload', return_value="salt_test"):
                result = serializer.loads("some_blob", salt="my_salt")
                assert result == "salt_test"
                # Verify salt was passed down through iter_unsigners if logic allows
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimedSerializer():
    """
    Tests the initialization and basic properties of TimedSerializer.
    Since the provided code shows TimedSerializer inherits from Serializer,
    we verify it correctly identifies its default signer class.
    """
    # Mocking a base Serializer setup as Serializer is not defined in the snippet 
    # but required for instantiation. We assume a standard implementation context.
    class MockSerializer:
        def __init__(self, *args, **kwargs):
            pass
        def iter_unsigners(self, salt=None):
            yield MagicMock(spec=TimestampSigner)
        def load_payload(self, payload):
            return payload

    # Patching the base class behavior for the test scope
    with pytest.MonkeyPatch.context() as m:
        m.setattr("__main__.Serializer", MockSerializer)
        
        from __main__ import TimedSerializer, TimestampSigner
        
        serializer = TimedSerializer()
        
        # Test that the default_signer is correctly set to TimestampSigner
        assert serializer.default_signer is TimestampSigner
        
        # Verify it is an instance of a class with the expected interface
        assert hasattr(serializer, 'loads')
        assert hasattr(serializer, 'loads_unsafe')
        assert hasattr(serializer, 'iter_unsigners')

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimedSerializer():
    """
    Tests the instantiation and basic properties of TimedSerializer.
    Since TimedSerializer inherits from Serializer, we verify its 
    specific type-related attributes and its default signer.
    """
    # Mocking the base Serializer dependency if necessary, 
    # but here we test the class structure directly.
    serializer = TimedSerializer()

    # Verify it is an instance of TimestampSigner's default type
    assert serializer.default_signer is TimestampSigner
    
    # Verify it is a subclass of Serializer (via inheritance)
    assert issubclass(TimedSerializer, Serializer)
    
    # Verify the class uses TimestampSigner for its logic
    assert TimedSerializer.default_signer == TimestampSigner

@pytest.mark.parametrize("salt", [None, "test_salt", b"bytes_salt"])
def test_TimedSerializer_iter_unsignners(salt):
    """
    Tests that iter_unsigners returns an iterator of TimestampSigner instances.
    """
    serializer = TimedSerializer()
    # We mock the super().iter_unsigners if we wanted to isolate, 
    # but testing the actual return type is more robust for a constructor/init test.
    unsigners = list(serializer.iter_unsigners(salt=salt))
    
    for signer in unsigners:
        assert isinstance(signer, TimestampSigner)

def test_TimedSerializer_type_integrity():
    """
    Ensures the TimedSerializer maintains its identity as a Serializer subclass.
    """
    serializer = TimedSerializer()
    # check that it inherits the expected methods from the base Serializer/Signer hierarchy
    assert hasattr(serializer, 'loads')
    assert hasattr(serializer, 'loads_unsafe')
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimedSerializer():
    """
    Tests the initialization and basic property configuration 
    of the TimedSerializer class.
    """
    # Mocking base Serializer dependencies if necessary, 
    # but since we are testing the constructor/class definition:
    
    # Test that the class can be instantiated without arguments
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    
    # Verify the default_signer is indeed TimestampSigner as defined in the class
    assert serializer.default_signer is TimestampSigner

    # Verify inheritance properties
    assert issubclass(TimedSerializer, Serializer)
    
    # Verify that it uses TimestampSigner logic via its method signature requirements
    # (Checking if iter_unsigners returns expected types/instances)
    salt = b"test_salt"
    unsigners = list(serializer.iter_unsigners(salt=salt))
    
    for unsigner in unsigners:
        assert isinstance(unsigner, TimestampSigner)
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimestampSigner():
    # Mocking Signer base class dependencies since we aren't importing it
    # In a real scenario, TimestampSigner inherits from Signer. 
    # We test the initialization and basic property availability.
    
    secret = b"secret-key"
    sep = b"."
    
    # Create an instance of TimestampSigner
    # Note: Since we cannot import Signer, this assumes the environment 
    # allows instantiation of the class provided in the snippet.
    signer = TimestampSigner(secret)
    
    # Verify fundamental attributes inherited or defined
    assert signer.secret == secret
    assert signer.sep == sep
    
    # Verify methods exist on the instance
    assert hasattr(signer, "get_timestamp")
    assert hasattr(signer, "timestamp_to_datetime")
    assert hasattr(signer, "sign")
    assert hasattr(signer, "unsign")
    assert hasattr(signer, "validate")

    # Test timestamp conversion logic functionality
    ts = 1600000000
    dt = signer.timestamp_to_datetime(ts)
    assert dt.year == 2020
    assert dt.month == 9
    assert dt.day == 13
    assert dt.tzinfo.utcoffset(dt) is None or dt.tzinfo.utcoffset(dt).total_seconds() == 0

    # Test the sign method structure (basic check of output format)
    payload = b"hello"
    signed = signer.sign(payload)
    
    # A signed value should contain parts: payload + sep + timestamp + sep + signature
    parts = signed.split(sep)
    assert len(parts) >= 3
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimedSerializer():
    # Since the provided code doesn't include a custom __init__ for TimedSerializer
    # and it inherits from Serializer, we test that it can be instantiated 
    # and maintains its expected default signer type.
    
    # We mock the base class dependencies if necessary, but assuming 
    # standard usage of the provided snippet:
    
    class MockSerializer(TimedSerializer):
        def load_payload(self, payload):
            return payload

    serializer = MockSerializer()
    
    # Test that the default signer is indeed TimestampSigner as defined in the class attribute
    assert serializer.default_signer is TimestampSigner
    
    # Test that it inherits/uses the correct signer type during iteration
    mock_signer = MagicMock(spec=TimestampSigner)
    serializer.iter_unsigners = MagicMock(return_value=[mock_signer])
    
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], TimestampSigner)
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

@pytest.mark.parametrize("payload, salt", [
    ("test_payload", "test_salt"),
    (b"test_payload", b"test_salt"),
])
def test_TimedSerializer_loads(payload, salt):
    # Setup Serializer and TimestampSigner mock environment
    # We need a concrete implementation of Serializer to test loads. 
    # Using a simple JSON-like approach or mocking the base class behavior.
    class MockSerializer(TimedSerializer):
        def load_payload(self, data: bytes) -> any:
            return data.decode('utf-s') if isinstance(data, bytes) else data
        
        def iter_unsigners(self, salt=None):
            signer = TimestampSigner(secret='secret', salt=salt or 'salt')
            return [signer]

    serializer = MockSerializer()
    signer = TimestampSigner(secret='secret', salt=salt)
    
    # Test 1: Successful loads (returns payload)
    signed_val = signer.sign(payload)
    assert serializer.loads(signed_val, salt=salt) == (payload if isinstance(payload, bytes) else payload.encode('utf-8'))

    # Test 2: Successful loads with return_timestamp=True
    result, ts = serializer.loads(signed_val, return_timestamp=True, salt=salt)
    assert result == (payload if isinstance(payload, bytes) else payload.encode('utf-8'))
    assert isinstance(ts, datetime)
    assert ts.tzinfo == timezone.utc

    # Test 3: SignatureExpired exception (max_age check)
    # Create a signature from the past
    with patch.object(TimestampSigner, 'get_timestamp', return_value=int(datetime.now().timestamp()) - 100):
        old_signed_val = signer.sign(payload)
    
    with pytest.raises(SignatureExpired) as excinfo:
        # max_age is only 50 seconds, but signature is 100 seconds old
        serializer.loads(old_signed_val, max_age=50, salt=salt)
    assert "Signature age" in str(excinfo.value)

    # Test 4: BadSignature exception (wrong secret/salt)
    wrong_signer = TimestampSigner(secret='wrong', salt=salt)
    wrong_signed_val = wrong_signer.sign(payload)
    with pytest.raises(BadSignature):
        serializer.loads(wrong_signed_val, salt=salt)

    # Test 5: Handling multiple unsigners (iter_unsigners)
    # We mock iter_unsigners to return two signers, one valid and one invalid
    signer1 = TimestampSigner(secret='secret1', salt=salt)
    signer2 = TimestamplySigner(secret='secret2', salt=salt) # Note: logic implies it tries next if BadSignature
    
    # Since we can't easily define TimelySigner here without complexity, 
    # let's mock the iteration process specifically.
    with patch.object(MockSerializer, 'iter_unsigners') as mock_iters:
        signer_fail = MagicMock(spec=TimestampSigner)
        signer_fail.unsign.side_effect = BadSignature("Fail", payload=b"corrupted")
        
        signer_success = MagicMock(spec=TimestampSigner)
        # Mocking the return of unsign for success (value, timestamp)
        ts_now = datetime.now(timezone.utc)
        signer_success.unsign.return_value = (b"payload_data", ts_now)
        
        mock_iters.return_value = [signer_fail, signer_success]
        
        # Should skip signer_fail and succeed with signer_success
        result = serializer.loads(b"some_blob", salt=salt)
        assert result == b"payload_data"

    # Test 6: Data type handling (bytes vs str)
    str_payload = "string_payload"
    signed_str = signer.sign(str_payload)
    assert serializer.loads(signed_str, salt=salt) == b"string_payload"
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimestampSigner():
    # Mocking the base Signer class dependencies
    # Since we cannot import, we assume the existence of a valid secret key setup 
    # as Signer would require it.
    secret = b"secret-key"
    
    # Initialize TimestampSigner
    signer = TimestampSigner(secret)
    
    # Test instance type
    assert isinstance(signer, TimestampSigner)
    
    # Test inheritance (assuming Signer is the base class)
    from .signer import Signer
    assert isinstance(signer, Signer)
    
    # Test default separator (usually b'.')
    assert signer.sep == b"."

    # Test that get_timestamp returns an integer
    ts = signer.get_timestamp()
    assert isinstance(ts, int)

    # Test timestamp_to_datetime functionality
    dt = signer.timestamp_to_datetime(ts)
    assert dt.tzinfo is not None
    assert dt.year > 2000
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

class TestTimestampSignerUnsign:
    @pytest.fixture
    def signer(self):
        # Mocking Signer base class behavior via a simple subclass or direct instance
        # Since we can't see the Signer implementation, we assume it works like standard Signer
        from unittest.mock import MagicMock
        signer = TimestampSigner()
        signer.sep = b"."
        return signer

    def test_TimestampSigner_unsign(self, signer):
        # Setup common values
        payload = b"hello"
        timestamp_val = 1600000000  # Fixed timestamp
        
        # Helper to create a valid signed string for the mock
        # format: payload + sep + base64(ts) + sep + signature
        from .encoding import base64_encode, int_to_bytes
        ts_encoded = base64_encode(int_to_bytes(timestamp_val))
        valid_signed_value = payload + b"." + ts_encoded + b".signature"

        # 1. Test Successful Unsign (return bytes)
        with patch.object(TimestampSigner, 'get_signature', return_value=b"signature"):
            # We need to mock the super().unsign call which is part of Signer
            # Since we can't easily mock the parent class method without knowing its structure,
            # we simulate the logic of a successful signer.
            with patch('__main__.Signer.unsign', return_value=payload + b"." + ts_encoded + b".signature"):
                result = signer.unsign(valid_signed_value)
                assert result == payload

        # 2. Test Successful Unsign (return timestamp)
        with patch('__main__.Signer.unsign', return_value=payload + b"." + ts_encoded + b".signature"):
            result, dt = signer.unsign(valid_signed_value, return_timestamp=True)
            assert result == payload
            assert dt == datetime.fromtimestamp(timestamp_val, tz=timezone.dumps(timezone.utc))

        # 3. Test Signature Expired (Too old)
        with patch('__main__.Signer.unsign', return_value=payload + b"." + ts_encoded + b".signature"):
            with patch.object(TimestampSigner, 'get_timestamp', return_value=timestamp_val + 100):
                with pytest.raises(SignatureExpired) as excinfo:
                    signer.unsign(valid_signed_value, max_age=50)
                assert payload in excinfo.value.payload

        # 4. Test Signature Expired (Future timestamp - clock drift)
        with patch('__main__.Signer.unsign', return_value=payload + b"." + ts_encoded + b".signature"):
            with patch.object(TimestampSigner, 'get_timestamp', return_value=timestamp_val - 100):
                with pytest.raises(SignatureExpired):
                    signer.unsign(valid_signed_value, max_age=50)

        # 5. Test Bad Signature (BadSignature exception raised by super)
        from .exc import BadSignature
        bad_sig_error = BadSignature("invalid signature", payload=payload)
        with patch('__main__.Signer.unsign', side_effect=bad_sig_error):
            # We need to provide a value that contains the separator so it can parse the timestamp
            # even if the signature itself is bad. 
            # The logic: result = e.payload or b"" -> result becomes payload.
            # Then it tries to split result by sep.
            with patch('__main__.base64_decode', return_value=int_to_bytes(timestamp_val)):
                # We trick the system into thinking the 'bad' payload actually contains a timestamp
                # so we can test the BadTimeSignature logic or just the re-raising of BadSignature.
                with pytest.raises(BadSignature):
                    signer.unsign(payload + b"." + ts_encoded + b".wrong")

        # 6. Test Malformed Timestamp (Not base64/int)
        malformed_ts_value = payload + b"." + b"notbase64!" + b".signature"
        with patch('__main__.Signer.unsign', return_value=payload + b"." + b"notbase64!" + b".signature"):
            with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
                signer.unsign(malformed_ts_value)

        # 7. Test Missing Timestamp (No separator in result)
        with patch('__main__.Signer.unsign', return_value=payload):
            with pytest.raises(BadTimeSignature, match="timestamp missing"):
                signer.unsign(payload)

        # 8. Test Bad Signature with valid timestamp parsing (Testing the error enrichment)
        from .exc import BadSignature
        bad_sig_error = BadSignature("invalid signature", payload=payload)
        with patch('__main__.Signer.unsign', side_effect=bad_sig_error):
            # Simulate that the bad signature's payload actually contains a timestamp segment
            # This tests if it successfully extracts date_signed from the error payload
            from .encoding import base64_encode, int_to_bytes
            ts_encoded = base64_encode(int_to_bytes(timestamp_val))
            with patch('__main__.base64_decode', return_value=int_to_bytes(timestamp_val)):
                # Note: The 'result' becomes e.payload (payload). 
                # For the code to reach the timestamp parsing, result must have a sep.
                # We mock the error payload to be the valid-looking string.
                bad_sig_error.payload = payload + b"." + ts_encoded + b".signature"
                with pytest.raises(BadSignature) as excinfo:
                    signer.unsign(b"some_input")
                assert excinfo.value.date_signed == datetime.fromtimestamp(timestamp_val, tz=timezone.utc)

    def test_validate(self, signer):
        payload = b"hello"
        ts_encoded = b"abc=" # dummy base64
        signed_value = payload + b"." + ts_encoded + b".sig"
        
        with patch('__main__.Signer.unsign', return_value=payload + b"." + ts_encoded + b".sig"):
            assert signer.validate(signed_value) is True
            
        from .exc import BadSignature
        with patch('__main__.Signer.unsign', side_effect=BadSignature("bad")):
            assert signer.validate(signed_value) is False
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_TimedSerializer():
    """
    Tests the initialization and basic properties of TimedSerializer.
    Since the constructor is inherited from Serializer, we verify 
    the class-specific attribute 'default_signer'.
    """
    # Mocking a simple serializer setup (assuming JSON as a base)
    class MockSerializer(TimedSerializer):
        def load_payload(self, payload: bytes) -> any:
            return payload.decode("utf-8")
        
        def dump_payload(self, value: any) -> bytes:
            return str(value).encode("utf-8")

    serializer = MockSerializer()

    # Verify that the default_signer is indeed TimestampSigner
    assert serializer.default_signer is TimestampSigner
    
    # Verify that it inherits functionality expected of a Serializer/TimedSerializer
    # A TimedSerializer must use TimestampSigner to handle time-based logic
    assert isinstance(serializer, TimedSerializer)
```


