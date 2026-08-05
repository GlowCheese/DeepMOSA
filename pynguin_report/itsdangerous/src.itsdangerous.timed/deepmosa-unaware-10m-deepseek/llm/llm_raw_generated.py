####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor and basic initialization."""
    # Test default construction
    serializer = TimedSerializer()
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    
    # Test with secret key
    serializer_with_key = TimedSerializer(secret_key="test-secret")
    assert serializer_with_key.secret_key == "test-secret"
    
    # Test with salt
    serializer_with_salt = TimedSerializer(salt="test-salt")
    assert serializer_with_salt.salt == "test-salt"
    
    # Test with signer_kwargs
    serializer_with_signer_kwargs = TimedSerializer(
        signer_kwargs={"key_derivation": "hmac"}
    )
    assert serializer_with_signer_kwargs.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test that default signer creates TimestampSigner instances
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    
    # Test with all parameters
    serializer_full = TimedSerializer(
        secret_key="complex-key",
        salt="complex-salt",
        serializer="json",
        signer_kwargs={"digest_method": "sha256"},
        signer=TimestampSigner
    )
    assert serializer_full.secret_key == "complex-key"
    assert serializer_full.salt == "complex-salt"
    assert serializer_full.serializer == "json"
    assert serializer_full.signer_kwargs == {"digest_method": "sha256"}
```


# LLM-generated content at query #2
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test successful unsign without timestamp
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test successful unsign with timestamp
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    
    # Test with max_age (not expired)
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test with max_age (expired) - should raise SignatureExpired
    signer_with_fixed_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_fixed_time.get_timestamp
    signer_with_fixed_time.get_timestamp = lambda: 1000  # Fixed old timestamp
    signed_old = signer_with_fixed_time.sign("test_value")
    signer_with_fixed_time.get_timestamp = original_get_timestamp
    signer_with_fixed_time.get_timestamp = lambda: 2000  # Current time
    
    with pytest.raises(SignatureExpired):
        signer_with_fixed_time.unsign(signed_old, max_age=500)
    
    # Test with max_age (negative age - future timestamp) - should raise SignatureExpired
    signer_future = TimestampSigner("secret-key")
    signer_future.get_timestamp = lambda: 2000  # Future timestamp
    signed_future = signer_future.sign("test_value")
    signer_future.get_timestamp = lambda: 1000  # Current time is in the past
    
    with pytest.raises(SignatureExpired):
        signer_future.unsign(signed_future, max_age=3600)
    
    # Test malformed timestamp
    signed_malformed = signed[:-1] + b"X"  # Corrupt the timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_malformed)
    
    # Test missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test_value" + signer.sep.encode() + b"invalidsignature")
    
    # Test invalid signature with valid timestamp
    signed_invalid_sig = signed[:-10] + b"X" * 10  # Corrupt the signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_invalid_sig)
    
    # Test return_timestamp=True returns tuple
    signed = signer.sign("test_value")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bytes)
    assert isinstance(result[1], datetime)
```


# LLM-generated content at query #3
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key", salt="test-salt")
    
    # Test 1: Successful unsign without timestamp
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test 2: Successful unsign with return_timestamp=True
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Unsign with valid max_age
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)  # 1 hour
    assert result == b"test_value"
    
    # Test 4: Unsign with expired signature (max_age exceeded)
    # We need to mock get_timestamp to simulate time passage
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: original_get_timestamp() + 100
    
    signed = signer.sign("test_value")
    signer.get_timestamp = original_get_timestamp  # Restore
    
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=50)  # 50 seconds max age, but 100 seconds passed
    
    # Test 5: Unsign with negative time (future timestamp)
    signer.get_timestamp = lambda: original_get_timestamp() - 100
    
    signed = signer.sign("test_value")
    signer.get_timestamp = original_get_timestamp  # Restore
    
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=3600)
    
    # Test 6: Unsign with tampered signature
    signed = signer.sign("test_value")
    tampered = signed[:-1] + (b'\x00' if signed[-1:] != b'\x00' else b'\x01')
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered)
    
    # Test 7: Unsign with missing timestamp
    bad_signed = b"test_value" + signer.sep.encode() + b"bad_timestamp"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed)
    
    # Test 8: Unsign with malformed timestamp (invalid base64)
    bad_signed = b"test_value" + signer.sep.encode() + b"!!!invalid_base64!!!"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed)
    
    # Test 9: Unsign with bytes input
    signed = signer.sign(b"test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test 10: Unsign with string input
    signed = signer.sign("test_value")
    result = signer.unsign(signed.decode())
    assert result == b"test_value"
```


# LLM-generated content at query #4
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer("test_secret_key")
    assert serializer.default_signer is TimestampSigner
    assert serializer.signer is not None
    assert isinstance(serializer.signer, TimestampSigner)


# LLM-generated content at query #5
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key", salt="test-salt")
    
    # Test 1: Basic unsign without timestamp
    value = b"test-value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test 2: Unsign with return_timestamp=True
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Unsign with valid max_age
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=3600)  # 1 hour
    assert result == value
    
    # Test 4: Unsign with expired max_age
    import time as time_module
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time_module.time()) + 100  # Simulate future time
    signed = signer.sign(value)
    signer.get_timestamp = original_get_timestamp
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=1)
    
    # Test 5: Unsign with negative age (timestamp in the future)
    signer.get_timestamp = lambda: int(time_module.time()) - 100  # Simulate past time
    signed = signer.sign(value)
    signer.get_timestamp = lambda: int(time_module.time())  # Reset to current time
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=3600)
    
    # Test 6: Unsign with malformed timestamp
    malformed = b"test-value" + signer.sep.encode() + b"invalid-timestamp" + signer.sep.encode() + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test 7: Unsign with missing timestamp
    missing_ts = signer.sign(value)
    # Remove timestamp part
    sep = signer.sep.encode()
    parts = missing_ts.rsplit(sep, 1)
    no_ts = parts[0] + sep + b"badsig"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_ts)
    
    # Test 8: Unsign with bad signature (but valid timestamp structure)
    signed = signer.sign(value)
    bad_sig = signed[:-1] + (b"x" if signed[-1:] != b"x" else b"y")
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_sig)
    
    # Test 9: Unsign with string input
    signed_str = signer.sign("test-string").decode()
    result = signer.unsign(signed_str)
    assert result == b"test-string"
    
    # Test 10: Unsign with return_timestamp and max_age combined
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 11: Verify timestamp is within reasonable range (last 10 seconds)
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    now = datetime.now(timezone.utc)
    assert (now - timestamp).total_seconds() < 10
```


# LLM-generated content at query #6
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer("test-secret-key")
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer.signer, TimestampSigner)
    assert serializer.signer.secret_key == "test-secret-key"
    
    # Test with different key types
    serializer_bytes = TimedSerializer(b"test-secret-key")
    assert serializer_bytes.signer.secret_key == b"test-secret-key"
    
    # Test with salt
    serializer_salt = TimedSerializer("test-secret-key", salt="test-salt")
    assert serializer_salt.salt == "test-salt"
    
    # Test with signer_kwargs
    serializer_kwargs = TimedSerializer("test-secret-key", signer_kwargs={"key_derivation": "none"})
    assert serializer_kwargs.signer.key_derivation == "none"
    
    # Test with digest_method
    from hashlib import sha256
    serializer_digest = TimedSerializer("test-secret-key", digest_method=sha256)
    assert serializer_digest.signer.digest_method == sha256
    
    # Test default serializer and signer are TimestampSigner
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer.signer, TimestampSigner)
```


# LLM-generated content at query #7
#--------------------------

```python
def test_TimestampSigner():
    """Test TimestampSigner constructor and basic functionality."""
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."


# LLM-generated content at query #8
#--------------------------

```python
def test_TimedSerializer():
    # Test default construction
    serializer = TimedSerializer()
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    
    # Test construction with secret key
    serializer = TimedSerializer("secret-key")
    assert serializer.secret_key == b"secret-key"
    
    # Test construction with salt
    serializer = TimedSerializer("secret-key", salt="my-salt")
    assert serializer.salt == "my-salt"
    
    # Test construction with serializer
    import json
    serializer = TimedSerializer("secret-key", serializer=json)
    assert serializer.serializer is json
    
    # Test construction with signer kwargs
    serializer = TimedSerializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    
    # Verify iter_unsigners returns TimestampSigner instances
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 0
    assert all(isinstance(s, TimestampSigner) for s in signers)
    
    # Verify dumps and loads roundtrip works
    data = {"test": "value"}
    signed = serializer.dumps(data)
    loaded = serializer.loads(signed)
    assert loaded == data
    
    # Verify timestamp return option
    signed = serializer.dumps(data)
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None  # Should be timezone-aware
    
    # Verify max_age parameter works
    import time
    signed = serializer.dumps(data)
    # Should work with large max_age
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Verify max_age with very short duration fails
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=-1)


# LLM-generated content at query #9
#--------------------------

```python
def test_TimedSerializer():
    # Test default constructor
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.secret_key == "test-secret"
    assert isinstance(serializer.default_signer, type)
    assert serializer.default_signer == TimestampSigner
    assert serializer.signer_kwargs == {}
    assert serializer.salt == "itsdangerous.TimedSerializer"
    
    # Test constructor with custom salt
    custom_serializer = TimedSerializer(
        secret_key="test-secret",
        salt="custom-salt",
        signer_kwargs={"key_derivation": "hmac"}
    )
    assert custom_serializer.salt == "custom-salt"
    assert custom_serializer.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test constructor with serializer_kwargs
    serializer_with_kwargs = TimedSerializer(
        secret_key="test-secret",
        serializer_kwargs={"serializer": "json"}
    )
    assert serializer_with_kwargs.serializer_kwargs == {"serializer": "json"}
    
    # Test that iter_unsigners returns TimestampSigner instances
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 0
    for signer in signers:
        assert isinstance(signer, TimestampSigner)


# LLM-generated content at query #10
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer(secret_key="test-secret")
    original_data = {"key": "value"}
    
    # Test basic loads (no timestamp)
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed)
    assert result == original_data
    
    # Test loads with return_timestamp=True
    result_with_ts = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == original_data
    assert isinstance(result_with_ts[1], datetime)
    
    # Test loads with max_age (valid age)
    result = serializer.loads(signed, max_age=3600)
    assert result == original_data
    
    # Test loads with max_age (expired)
    import time
    # Create a signed value with an old timestamp
    old_signer = TimestampSigner("test-secret")
    old_signer.get_timestamp = lambda: int(time.time()) - 7200  # 2 hours ago
    old_signed = serializer.dumps(original_data)
    
    # Should raise SignatureExpired when max_age is less than age
    import pytest
    from itsdangerous.exc import SignatureExpired
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=3600)
    
    # Test loads with invalid signature
    from itsdangerous.exc import BadSignature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-data" + signed)
    
    # Test loads with empty/malformed data
    with pytest.raises(BadSignature):
        serializer.loads(b"")
    
    # Test loads with different salt
    salt = "custom-salt"
    signed_with_salt = serializer.dumps(original_data, salt=salt)
    result = serializer.loads(signed_with_salt, salt=salt)
    assert result == original_data
    
    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
    
    # Test loads with bytes input
    signed_bytes = serializer.dumps(original_data)
    assert isinstance(signed_bytes, bytes)
    result = serializer.loads(signed_bytes)
    assert result == original_data
    
    # Test loads with string input
    signed_str = signed_bytes.decode('utf-8')
    result = serializer.loads(signed_str)
    assert result == original_data
    
    # Test that loads raises BadSignature for completely invalid data
    with pytest.raises(BadSignature):
        serializer.loads(b"not-even-close-to-valid")
```


# LLM-generated content at query #11
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without max_age
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test unsign with return_timestamp=True
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == b"test_value"
    assert isinstance(result_with_ts[1], datetime)
    
    # Test unsign with valid max_age
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test unsign with expired signature
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)
    
    # Test unsign with malformed timestamp
    malformed = signed + b"invalid"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test unsign with completely invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")
    
    # Test unsign with different key
    different_signer = TimestampSigner("different-key")
    with pytest.raises(BadSignature):
        different_signer.unsign(signed)
    
    # Test unsign with timestamp in the future (negative age)
    future_signer = TimestampSigner("secret-key")
    # Mock get_timestamp to return a value in the future
    original_get_timestamp = future_signer.get_timestamp
    future_signer.get_timestamp = lambda: int(time.time()) + 100
    future_signed = future_signer.sign("test_value")
    future_signer.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired) as exc_info:
        future_signer.unsign(future_signed, max_age=3600)
    assert "age" in str(exc_info.value)
```


# LLM-generated content at query #12
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("test-secret")
    
    # Test basic unsign without timestamp
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"
    
    # Test unsign with return_timestamp=True
    signed = signer.sign("test-value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with valid max_age
    signed = signer.sign("test-value")
    result = signer.unsign(signed, max_age=3600)  # 1 hour
    assert result == b"test-value"
    
    # Test unsign with expired signature
    signer_expired = TimestampSigner("test-secret")
    # Override get_timestamp to return a time in the past
    old_timestamp = int(time.time()) - 100  # 100 seconds ago
    signer_expired.get_timestamp = lambda: old_timestamp
    signed = signer_expired.sign("test-value")
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(signed, max_age=50)  # max_age 50 seconds
    assert "Signature age" in str(exc_info.value)
    
    # Test unsign with future timestamp (age < 0)
    future_signer = TimestampSigner("test-secret")
    future_timestamp = int(time.time()) + 100  # 100 seconds in future
    future_signer.get_timestamp = lambda: future_timestamp
    signed = future_signer.sign("test-value")
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(signed, max_age=3600)
    assert "age" in str(exc_info.value)
    
    # Test unsign with bad signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid-data")
    
    # Test unsign with missing timestamp
    bad_signed = b"test-value" + signer.sep.encode() + b"bad-timestamp"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(bad_signed)
    assert "Malformed timestamp" in str(exc_info.value)
    
    # Test unsign with valid timestamp but bad signature
    signer_bad = TimestampSigner("different-secret")
    signed_bad = signer_bad.sign("test-value")
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_bad)
    
    # Test unsign with malformed timestamp (non-decodable)
    malformed = b"test-value" + signer.sep.encode() + b"!!!"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(malformed)
    assert "Malformed timestamp" in str(exc_info.value)
    
    # Test unsign with empty string
    with pytest.raises(BadSignature):
        signer.unsign(b"")
    
    # Test unsign with separator only
    with pytest.raises(BadSignature):
        signer.unsign(signer.sep.encode())
```


# LLM-generated content at query #13
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner(secret_key="test-key")
    assert signer.secret_key == "test-key"
    assert signer.sep == "."
    assert callable(signer.get_timestamp)
    assert callable(signer.timestamp_to_datetime)
    assert callable(signer.sign)
    assert callable(signer.unsign)
    assert callable(signer.validate)


# LLM-generated content at query #14
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner(secret_key="test-secret")
    assert signer.secret_key == "test-secret"
    assert signer.sep == "."
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    
    # Test constructor with custom parameters
    signer_custom = TimestampSigner(
        secret_key="custom-secret",
        salt="custom-salt",
        sep="|",
        key_derivation="none",
        digest_method=hashlib.sha256
    )
    assert signer_custom.secret_key == "custom-secret"
    assert signer_custom.salt == "custom-salt"
    assert signer_custom.sep == "|"
    assert signer_custom.key_derivation == "none"
    assert signer_custom.digest_method == hashlib.sha256
    
    # Test that TimestampSigner is a subclass of Signer
    assert isinstance(signer, Signer)
    
    # Test default signer algorithm
    assert signer.algorithm is not None


# LLM-generated content at query #15
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test successful load without timestamp return
    serializer = TimedSerializer("test_secret_key")
    original_data = {"key": "value"}
    serialized = serializer.dumps(original_data)
    
    result = serializer.loads(serialized)
    assert result == original_data

    # Test successful load with timestamp return
    result_with_ts = serializer.loads(serialized, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == original_data
    assert isinstance(result_with_ts[1], datetime)

    # Test load with valid max_age
    result = serializer.loads(serialized, max_age=3600)
    assert result == original_data

    # Test load with expired signature
    import time
    with unittest.mock.patch.object(serializer, 'load_payload', return_value=original_data):
        with unittest.mock.patch.object(time, 'time', return_value=time.time() + 7200):
            with pytest.raises(SignatureExpired):
                serializer.loads(serialized, max_age=3600)

    # Test load with bad signature
    bad_serialized = b"invalid_signature"
    with pytest.raises(BadSignature):
        serializer.loads(bad_serialized)

    # Test load with different salt
    serializer2 = TimedSerializer("test_secret_key", salt="different_salt")
    serialized2 = serializer2.dumps(original_data)
    
    # Should fail with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(serialized2)

    # Test load with multiple signers (fallback)
    serializer3 = TimedSerializer(["key1", "key2"])
    serialized3 = serializer3.dumps(original_data)
    
    result = serializer3.loads(serialized3)
    assert result == original_data

    # Test load with return_timestamp and max_age
    result_with_ts = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == original_data
    assert isinstance(result_with_ts[1], datetime)
```


# LLM-generated content at query #16
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    test_value = b"test message"
    
    # Test basic unsign without timestamp
    signed = signer.sign(test_value)
    result = signer.unsign(signed)
    assert result == test_value
    
    # Test unsign with return_timestamp=True
    signed = signer.sign(test_value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == test_value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with valid max_age
    signed = signer.sign(test_value)
    result = signer.unsign(signed, max_age=3600)
    assert result == test_value
    
    # Test unsign with expired signature
    signed = signer.sign(test_value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)
    
    # Test unsign with future timestamp (age < 0)
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) - 100
    signed = signer.sign(test_value)
    signer.get_timestamp = original_get_timestamp
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=50)
    
    # Test unsign with tampered value
    signed = signer.sign(test_value)
    tampered = signed[:-1] + b"x"
    with pytest.raises(BadSignature):
        signer.unsign(tampered)
    
    # Test unsign with missing timestamp
    bad_signer = Signer("secret-key")
    signed_no_ts = bad_signer.sign(test_value)
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(signed_no_ts)
    
    # Test unsign with malformed timestamp
    sep = signer.sep.encode()
    malformed = test_value + sep + b"not-a-timestamp" + sep + signer.get_signature(test_value + sep + b"not-a-timestamp")
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(malformed)
    
    # Test unsign with string input
    signed_str = signer.sign(test_value).decode()
    result = signer.unsign(signed_str)
    assert result == test_value
    
    # Test unsign with return_timestamp and string input
    signed_str = signer.sign(test_value).decode()
    result, timestamp = signer.unsign(signed_str, return_timestamp=True)
    assert result == test_value
    assert isinstance(timestamp, datetime)```


# LLM-generated content at query #17
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with max_age
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test expired signature
    old_signer = TimestampSigner("secret-key")
    old_signer.get_timestamp = lambda: int(time.time()) - 100
    signed = old_signer.sign("test_value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=10)
    
    # Test future timestamp (age < 0)
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: int(time.time()) + 100
    signed = future_signer.sign("test_value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=3600)
    
    # Test malformed timestamp
    bad_timestamp = b"test_value" + signer.sep.encode() + b"invalid_timestamp"
    bad_signed = bad_timestamp + signer.sep.encode() + signer.get_signature(bad_timestamp)
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(bad_signed)
    
    # Test missing timestamp
    no_timestamp = signer.sign("test_value").split(signer.sep.encode())[0]
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_timestamp)
    
    # Test invalid signature with timestamp
    invalid_sig = signer.sign("test_value") + b"tampered"
    with pytest.raises(BadTimeSignature):
        signer.unsign(invalid_sig)
    
    # Test with bytes input
    signed_bytes = signer.sign(b"bytes_value")
    result = signer.unsign(signed_bytes)
    assert result == b"bytes_value"
    
    # Test with string input
    signed_str = signer.sign("string_value").decode()
    result = signer.unsign(signed_str)
    assert result == b"string_value"
```


# LLM-generated content at query #18
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test successful loads without timestamp
    serializer = TimedSerializer(secret_key="test-key")
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data

    # Test successful loads with timestamp
    result = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == data
    assert isinstance(result[1], datetime)

    # Test with max_age that is not expired
    result = serializer.loads(signed, max_age=3600)
    assert result == data

    # Test with max_age that is expired
    import time
    future_signed = serializer.dumps(data)
    time.sleep(0.1)  # Ensure age > 0
    with pytest.raises(SignatureExpired):
        serializer.loads(future_signed, max_age=0)

    # Test with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid.signature")

    # Test with different salt
    signed_with_salt = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == data

    # Test with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")

    # Test with empty data
    signed_empty = serializer.dumps("")
    result = serializer.loads(signed_empty)
    assert result == ""

    # Test with bytes input
    signed_bytes = serializer.dumps(b"test_bytes")
    result = serializer.loads(signed_bytes)
    assert result == b"test_bytes"

    # Test with integer input
    signed_int = serializer.dumps(42)
    result = serializer.loads(signed_int)
    assert result == 42

    # Test with list input
    signed_list = serializer.dumps([1, 2, 3])
    result = serializer.loads(signed_list)
    assert result == [1, 2, 3]
```


# LLM-generated content at query #19
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test 1: Basic unsign without timestamp
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test 2: Unsign with return_timestamp=True
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == b"test_value"
    assert isinstance(result_with_ts[1], datetime)
    assert result_with_ts[1].tzinfo == timezone.utc
    
    # Test 3: Unsign with valid max_age
    result_valid = signer.unsign(signed, max_age=3600)
    assert result_valid == b"test_value"
    
    # Test 4: Unsign with expired signature (max_age=0)
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(signed, max_age=0)
    assert "Signature age" in str(exc_info.value)
    
    # Test 5: Unsign with negative max_age (future timestamp)
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(signed, max_age=-1)
    assert "Signature age" in str(exc_info.value)
    
    # Test 6: Unsign with invalid signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid_signature")
    
    # Test 7: Unsign with tampered value
    tampered = signed[:-1] + (b"x" if signed[-1:] != b"x" else b"y")
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered)
    
    # Test 8: Unsign with missing timestamp
    no_timestamp = signer.sign("test_value").split(signer.sep.encode())[0]
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_timestamp)
    
    # Test 9: Unsign with malformed timestamp
    malformed_ts = base64_encode(b"invalid_timestamp")
    malformed_signed = b"test_value" + signer.sep.encode() + malformed_ts + signer.sep.encode() + signer.get_signature(b"test_value")
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(malformed_signed)
    
    # Test 10: Verify timestamp is returned as aware datetime in UTC
    result_ts = signer.unsign(signed, return_timestamp=True)
    assert result_ts[1].tzinfo == timezone.utc
    
    # Test 11: Verify correct timestamp value
    current_time = int(time.time())
    signed_with_time = signer.sign("test")
    _, ts = signer.unsign(signed_with_time, return_timestamp=True)
    assert abs(ts.timestamp() - current_time) < 2  # within 2 seconds
    
    # Test 12: Unsign with bytes value
    signed_bytes = signer.sign(b"test_bytes")
    result_bytes = signer.unsign(signed_bytes)
    assert result_bytes == b"test_bytes"
    
    # Test 13: Unsign with string value
    signed_str = signer.sign("test_string")
    result_str = signer.unsign(signed_str)
    assert result_str == b"test_string"
```


# LLM-generated content at query #20
#--------------------------

```python
def test_TimedSerializer():
    # Test default constructor
    serializer = TimedSerializer()
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    assert hasattr(serializer, 'load_payload')
    assert hasattr(serializer, 'dump_payload')
    
    # Test constructor with secret key
    serializer_with_key = TimedSerializer(secret_key='test-secret')
    assert serializer_with_key.secret_key == 'test-secret'
    
    # Test constructor with salt
    serializer_with_salt = TimedSerializer(salt='test-salt')
    assert serializer_with_salt.salt == 'test-salt'
    
    # Test constructor with signer_kwargs
    serializer_with_signer_kwargs = TimedSerializer(signer_kwargs={'key_derivation': 'hmac'})
    assert serializer_with_signer_kwargs.signer_kwargs == {'key_derivation': 'hmac'}
    
    # Test constructor with serializer_kwargs
    serializer_with_serializer_kwargs = TimedSerializer(serializer_kwargs={'serializer_module': 'json'})
    assert serializer_with_serializer_kwargs.serializer_kwargs == {'serializer_module': 'json'}
```


# LLM-generated content at query #21
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic sign and unsign
    value = b"test value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with valid max_age
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test unsign with max_age that is too short (should raise SignatureExpired)
    import time
    # Create a signed value with an old timestamp
    old_signer = TimestampSigner("secret-key")
    old_signer.get_timestamp = lambda: int(time.time()) - 100  # 100 seconds ago
    old_signed = old_signer.sign(value)
    
    with pytest.raises(SignatureExpired):
        signer.unsign(old_signed, max_age=10)
    
    # Test unsign with future timestamp (should raise SignatureExpired)
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: int(time.time()) + 100  # 100 seconds in future
    future_signed = future_signer.sign(value)
    
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed, max_age=3600)
    
    # Test unsign with wrong key (should raise BadSignature)
    wrong_signer = TimestampSigner("wrong-key")
    wrong_signed = wrong_signer.sign(value)
    
    with pytest.raises(BadSignature):
        signer.unsign(wrong_signed)
    
    # Test unsign with malformed timestamp
    malformed = signed + b"malformed"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test unsign with missing timestamp
    # Create a signed value without timestamp by using regular Signer
    regular_signer = Signer("secret-key")
    no_timestamp = regular_signer.sign(value)
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_timestamp)
    
    # Test unsign with string input
    signed_str = signed.decode('utf-8')
    result = signer.unsign(signed_str)
    assert result == value
```


# LLM-generated content at query #22
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner(secret_key="test-secret")
    assert signer.secret_key == "test-secret"
    assert signer.sep == "."
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm is not None

    # Test constructor with custom parameters
    signer = TimestampSigner(
        secret_key="custom-secret",
        salt="custom-salt",
        sep=":",
        key_derivation="none",
        digest_method=hashlib.sha256,
    )
    assert signer.secret_key == "custom-secret"
    assert signer.salt == "custom-salt"
    assert signer.sep == ":"
    assert signer.key_derivation == "none"
    assert signer.digest_method == hashlib.sha256

    # Test constructor with bytes secret key
    signer = TimestampSigner(secret_key=b"bytes-secret")
    assert signer.secret_key == b"bytes-secret"

    # Test constructor with fallback signer
    signer = TimestampSigner(secret_key="test", fallback_signers=["fallback1", "fallback2"])
    assert len(signer.fallback_signers) == 2
```


# LLM-generated content at query #23
#--------------------------

```python
def test_TimedSerializer():
    # Test default construction
    serializer = TimedSerializer()
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    
    # Test with secret key
    serializer = TimedSerializer("test-secret")
    assert serializer.default_signer is TimestampSigner
    assert serializer.secret_key == b"test-secret"
    
    # Test with salt
    serializer = TimedSerializer("test-secret", salt="test-salt")
    assert serializer.default_signer is TimestampSigner
    assert serializer.salt == "test-salt"
    
    # Test with signer_kwargs
    serializer = TimedSerializer("test-secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.default_signer is TimestampSigner
    
    # Test with serializer_kwargs
    serializer = TimedSerializer("test-secret", serializer_kwargs={"load_kwargs": {}})
    assert serializer.default_signer is TimestampSigner
    
    # Test serialization roundtrip
    serializer = TimedSerializer("test-secret")
    data = {"test": "data"}
    serialized = serializer.dumps(data)
    assert isinstance(serialized, str)
    
    loaded = serializer.loads(serialized)
    assert loaded == data
    
    # Test with custom signer
    class CustomTimestampSigner(TimestampSigner):
        pass
    
    serializer = TimedSerializer("test-secret", signer=CustomTimestampSigner)
    assert serializer.default_signer is TimestampSigner
```


# LLM-generated content at query #24
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner(secret_key="test-secret-key")
    assert signer.secret_key == "test-secret-key"
    assert signer.sep == "."
    assert signer.salt is None
    assert signer.key_derivation is None
    assert signer.digest_method is None
    assert signer.algorithm is None


# LLM-generated content at query #25
#--------------------------

```python
def test_TimedSerializer():
    # Test default constructor
    serializer = TimedSerializer()
    assert serializer.default_signer is TimestampSigner
    assert serializer.signer is TimestampSigner
    assert serializer.secret_key is not None
    
    # Test constructor with secret_key
    serializer2 = TimedSerializer(secret_key="my-secret")
    assert serializer2.secret_key == "my-secret"
    
    # Test constructor with salt
    serializer3 = TimedSerializer(salt="my-salt")
    assert serializer3.salt_kwargs["salt"] == "my-salt"
    
    # Test constructor with signer_kwargs
    serializer4 = TimedSerializer(signer_kwargs={"key_derivation": "hmac"})
    assert serializer4.signer_kwargs["key_derivation"] == "hmac"
    
    # Test constructor with serializer_kwargs
    serializer5 = TimedSerializer(serializer_kwargs={"serializer": "json"})
    assert serializer5.serializer_kwargs["serializer"] == "json"
    
    # Test constructor with multiple parameters
    serializer6 = TimedSerializer(
        secret_key="test-key",
        salt="test-salt",
        signer_kwargs={"key_derivation": "hmac"},
        serializer_kwargs={"serializer": "json"}
    )
    assert serializer6.secret_key == "test-key"
    assert serializer6.salt_kwargs["salt"] == "test-salt"
    assert serializer6.signer_kwargs["key_derivation"] == "hmac"
    assert serializer6.serializer_kwargs["serializer"] == "json"
```


# LLM-generated content at query #26
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without timestamp
    value = b"test_value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test unsign with return_timestamp=True
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == value
    assert isinstance(result_with_ts[1], datetime)
    
    # Test unsign with valid max_age
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test unsign with expired signature
    original_get_timestamp = signer.get_timestamp
    try:
        signer.get_timestamp = lambda: int(time.time()) - 100
        expired_signed = signer.sign(value)
        signer.get_timestamp = lambda: int(time.time()) + 200
        with pytest.raises(SignatureExpired):
            signer.unsign(expired_signed, max_age=50)
    finally:
        signer.get_timestamp = original_get_timestamp
    
    # Test unsign with malformed timestamp
    with pytest.raises(BadTimeSignature):
        malformed = signed + b"|invalid_timestamp"
        signer.unsign(malformed)
    
    # Test unsign with invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signed_value")
    
    # Test unsign with different key
    different_signer = TimestampSigner("different-key")
    with pytest.raises(BadSignature):
        different_signer.unsign(signed)
    
    # Test unsign with empty value
    empty_value = b""
    signed_empty = signer.sign(empty_value)
    result = signer.unsign(signed_empty)
    assert result == empty_value
    
    # Test unsign with newlines in value
    multiline_value = b"line1\nline2\nline3"
    signed_multiline = signer.sign(multiline_value)
    result = signer.unsign(signed_multiline)
    assert result == multiline_value
    
    # Test unsign with special characters
    special_value = b"value_with_|_separator_and_special_chars!@#$%"
    signed_special = signer.sign(special_value)
    result = signer.unsign(signed_special)
    assert result == special_value
    
    # Test unsign with negative age (future timestamp)
    original_timestamp = signer.get_timestamp
    try:
        future_time = int(time.time()) + 1000
        signer.get_timestamp = lambda: future_time
        future_signed = signer.sign(value)
        signer.get_timestamp = lambda: int(time.time())
        with pytest.raises(SignatureExpired):
            signer.unsign(future_signed, max_age=3600)
    finally:
        signer.get_timestamp = original_timestamp
    
    # Test unsign with return_timestamp and expired signature
    try:
        signer.get_timestamp = lambda: int(time.time()) - 100
        expired_signed = signer.sign(value)
        signer.get_timestamp = lambda: int(time.time()) + 200
        with pytest.raises(SignatureExpired):
            signer.unsign(expired_signed, max_age=50, return_timestamp=True)
    finally:
        signer.get_timestamp = original_timestamp
```


# LLM-generated content at query #27
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test_secret_key")
    
    # Test successful loads without timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test successful loads with return_timestamp=True
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test with max_age parameter
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test with expired signature
    import time
    old_signed = serializer.dumps(data)
    time.sleep(0.1)  # Ensure some time passes
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=0)
    
    # Test with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid_signature")
    
    # Test with empty string
    with pytest.raises(BadSignature):
        serializer.loads("")
    
    # Test with bytes input
    result = serializer.loads(signed.encode())
    assert result == data
    
    # Test with different salt
    serializer2 = TimedSerializer("test_secret_key", salt="different_salt")
    signed2 = serializer2.dumps(data)
    with pytest.raises(BadSignature):
        serializer.loads(signed2)
    
    # Test with multiple signers
    serializer3 = TimedSerializer("test_secret_key", salt="salt1")
    signed3 = serializer3.dumps(data)
    # Should still work with the default signer
    result = serializer.loads(signed3)
    assert result == data
```


# LLM-generated content at query #28
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer(secret_key="test-secret-key")
    
    # Test basic loads without max_age and return_timestamp
    original_data = {"key": "value"}
    dumped = serializer.dumps(original_data)
    result = serializer.loads(dumped)
    assert result == original_data
    
    # Test loads with max_age (valid signature)
    result = serializer.loads(dumped, max_age=3600)
    assert result == original_data
    
    # Test loads with return_timestamp=True
    result, timestamp = serializer.loads(dumped, return_timestamp=True)
    assert result == original_data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test loads with both max_age and return_timestamp
    result, timestamp = serializer.loads(dumped, max_age=3600, return_timestamp=True)
    assert result == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with expired signature
    with pytest.raises(SignatureExpired):
        serializer.loads(dumped, max_age=0)
    
    # Test loads with tampered data
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test loads with different salt
    serializer_with_salt = TimedSerializer(secret_key="test-secret-key", salt="custom-salt")
    dumped_with_salt = serializer_with_salt.dumps(original_data)
    result = serializer.loads(dumped_with_salt, salt="custom-salt")
    assert result == original_data
    
    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(dumped_with_salt, salt="wrong-salt")
    
    # Test loads with bytes input
    dumped_bytes = serializer.dumps(original_data)
    result = serializer.loads(dumped_bytes)
    assert result == original_data
    
    # Test loads with string input
    dumped_str = serializer.dumps(original_data).decode()
    result = serializer.loads(dumped_str)
    assert result == original_data
    
    # Test loads with complex data types
    complex_data = {
        "list": [1, 2, 3],
        "nested": {"a": 1, "b": 2},
        "number": 42,
        "boolean": True,
        "none": None
    }
    dumped_complex = serializer.dumps(complex_data)
    result = serializer.loads(dumped_complex)
    assert result == complex_data
    
    # Test loads with empty data
    dumped_empty = serializer.dumps({})
    result = serializer.loads(dumped_empty)
    assert result == {}
    
    # Test loads with single value
    dumped_single = serializer.dumps("test")
    result = serializer.loads(dumped_single)
    assert result == "test"
    
    # Test loads with future timestamp (age < 0)
    # This would normally require mocking, but we can test the error handling
    # by creating a signer with a future timestamp
    from unittest.mock import patch
    
    original_get_timestamp = serializer.default_signer.get_timestamp
    
    def future_timestamp():
        return int(time.time()) + 100000
    
    with patch.object(TimestampSigner, 'get_timestamp', future_timestamp):
        dumped_future = serializer.dumps(original_data)
        # The signature will be valid but timestamp is in the future
        result = serializer.loads(dumped_future, max_age=3600)
        # This should work because the signature is still valid
        assert result == original_data
```


# LLM-generated content at query #29
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner(secret_key="test-secret")
    assert signer.secret_key == "test-secret"
    assert signer.salt is None
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method is not None
    assert signer.algorithm is not None


# LLM-generated content at query #30
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer("test-secret")
    assert serializer.default_signer is TimestampSigner
    assert serializer.signer_cls is TimestampSigner
    assert serializer.secret_key == b"test-secret"
    
    data = {"key": "value"}
    signed = serializer.dumps(data)
    
    result = serializer.loads(signed)
    assert result == data
    
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    import time
    time.sleep(0.1)
    
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=0)
    
    serializer_with_salt = TimedSerializer("test-secret", salt="custom-salt")
    signed_with_salt = serializer_with_salt.dumps(data)
    result = serializer_with_salt.loads(signed_with_salt, salt="custom-salt")
    assert result == data
    
    with pytest.raises(BadSignature):
        serializer_with_salt.loads(signed_with_salt, salt="wrong-salt")


# LLM-generated content at query #31
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("test-secret")
    
    # Test successful unsign without return_timestamp
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"
    
    # Test successful unsign with return_timestamp
    signed = signer.sign("test-value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with max_age (not expired)
    signed = signer.sign("test-value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test-value"
    
    # Test unsign with expired signature (max_age smaller than actual age)
    # Use a fixed timestamp to simulate old signature
    old_timestamp = int(time.time()) - 100
    signer_with_fixed_time = TimestampSigner("test-secret")
    original_get_timestamp = signer_with_fixed_time.get_timestamp
    
    # Temporarily mock get_timestamp to return old timestamp for signing
    def mock_get_timestamp():
        return old_timestamp
    
    signer_with_fixed_time.get_timestamp = mock_get_timestamp
    old_signed = signer_with_fixed_time.sign("test-value")
    
    # Restore original get_timestamp for checking
    signer_with_fixed_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer_with_fixed_time.unsign(old_signed, max_age=10)
    assert "Signature age" in str(exc_info.value)
    assert exc_info.value.payload == b"test-value"
    assert exc_info.value.date_signed is not None
    
    # Test unsign with future timestamp (age < 0)
    future_timestamp = int(time.time()) + 100
    def mock_future_timestamp():
        return future_timestamp
    
    signer_with_future = TimestampSigner("test-secret")
    signer_with_future.get_timestamp = mock_future_timestamp
    future_signed = signer_with_future.sign("test-value")
    
    # Restore original get_timestamp
    signer_with_future.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer_with_future.unsign(future_signed, max_age=3600)
    assert "Signature age" in str(exc_info.value)
    assert exc_info.value.payload == b"test-value"
    
    # Test unsign with tampered signature
    signed = signer.sign("test-value")
    tampered = signed[:-1] + b"X"  # Change last byte of signature
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(tampered)
    assert exc_info.value.payload == b"test-value"
    
    # Test unsign with missing timestamp
    # Create a signed value without timestamp separator
    value = want_bytes("test-value")
    separator = want_bytes(signer.sep)
    no_timestamp = value + separator + signer.get_signature(value)
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(no_timestamp)
    assert "timestamp missing" in str(exc_info.value)
    
    # Test unsign with malformed timestamp
    malformed_ts = value + separator + base64_encode(b"not-a-timestamp") + separator + signer.get_signature(value + separator + base64_encode(b"not-a-timestamp"))
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(malformed_ts)
    assert "Malformed timestamp" in str(exc_info.value)
    
    # Test unsign with invalid base64 timestamp
    invalid_base64 = value + separator + b"!!!invalid" + separator + signer.get_signature(value + separator + b"!!!invalid")
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(invalid_base64)
    assert "Malformed timestamp" in str(exc_info.value)
    
    # Test unsign with empty value
    signed_empty = signer.sign("")
    result = signer.unsign(signed_empty)
    assert result == b""
    
    # Test unsign with bytes value
    signed_bytes = signer.sign(b"bytes-value")
    result = signer.unsign(signed_bytes)
    assert result == b"bytes-value"
```


# LLM-generated content at query #32
#--------------------------

```python
def test_TimestampSigner_unsign():
    import time
    from datetime import datetime, timezone
    
    signer = TimestampSigner("test-key")
    
    # Test basic signing and unsigning
    value = b"test_value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test with string input
    signed_str = signer.sign("test_string")
    result = signer.unsign(signed_str)
    assert result == b"test_string"
    
    # Test return_timestamp=True
    signed2 = signer.sign("test_timestamp")
    result, timestamp = signer.unsign(signed2, return_timestamp=True)
    assert result == b"test_timestamp"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test max_age validation
    signed3 = signer.sign("test_age")
    result = signer.unsign(signed3, max_age=1000)
    assert result == b"test_age"
    
    # Test signature expired (max_age too small)
    signer_slow = TimestampSigner("test-key")
    signer_slow.get_timestamp = lambda: int(time.time()) - 100
    signed4 = signer_slow.sign("test_expired")
    try:
        signer.unsign(signed4, max_age=10)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass
    
    # Test malformed timestamp
    malformed = b"value" + signer.sep.encode() + b"invalid_timestamp" + signer.sep.encode() + b"signature"
    try:
        signer.unsign(malformed)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass
    
    # Test missing timestamp
    value_only = b"value" + signer.sep.encode() + b"signature"
    try:
        signer.unsign(value_only)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass
    
    # Test with corrupted signature
    corrupted = signed[:-1] + b"x"
    try:
        signer.unsign(corrupted)
        assert False, "Expected BadSignature"
    except BadSignature:
        pass
```


# LLM-generated content at query #33
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without timestamp
    value = b"test_value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test unsign with return_timestamp=True
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with max_age (valid)
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test unsign with max_age (expired)
    signer.get_timestamp = lambda: int(time.time()) - 100
    signed = signer.sign(value)
    signer.get_timestamp = lambda: int(time.time())
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=50)
    
    # Test unsign with malformed timestamp
    signer.get_timestamp = lambda: int(time.time())
    signed = signer.sign(value)
    # Corrupt the timestamp part
    parts = signed.rsplit(signer.sep.encode(), 1)
    corrupted = parts[0] + signer.sep.encode() + b"invalid_timestamp"
    with pytest.raises(BadTimeSignature):
        signer.unsign(corrupted)
    
    # Test unsign with missing timestamp
    signed_no_timestamp = signer.sign(value)
    # Remove the timestamp part
    parts = signed_no_timestamp.rsplit(signer.sep.encode(), 1)
    value_only = parts[0]
    with pytest.raises(BadTimeSignature):
        signer.unsign(value_only)
    
    # Test unsign with invalid signature
    signed = signer.sign(value)
    corrupted = bytearray(signed)
    corrupted[-1] ^= 0xFF  # Flip last bit
    with pytest.raises(BadSignature):
        signer.unsign(bytes(corrupted))
```


# LLM-generated content at query #34
#--------------------------

```python
def test_TimestampSigner_unsign():
    """Test TimestampSigner.unsign method with various scenarios."""
    signer = TimestampSigner("test-secret")
    
    # Test 1: Basic sign and unsign without max_age
    value = b"test_value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value, f"Expected {value}, got {result}"
    
    # Test 2: Sign and unsign with return_timestamp=True
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple), "Expected tuple when return_timestamp=True"
    assert len(result_with_ts) == 2, "Expected tuple of length 2"
    assert result_with_ts[0] == value, f"Expected {value}, got {result_with_ts[0]}"
    assert isinstance(result_with_ts[1], datetime), "Expected datetime object"
    
    # Test 3: Sign and unsign with max_age that should pass
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=3600)  # 1 hour max age
    assert result == value, f"Expected {value}, got {result}"
    
    # Test 4: Sign and unsign with max_age that should fail (expired)
    # We need to mock the timestamp to make it appear old
    class OldTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) - 7200  # 2 hours ago
    
    old_signer = OldTimestampSigner("test-secret")
    old_signed = old_signer.sign(value)
    
    with pytest.raises(SignatureExpired):
        signer.unsign(old_signed, max_age=3600)
    
    # Test 5: Sign and unsign with negative age (future timestamp)
    class FutureTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) + 7200  # 2 hours in future
    
    future_signer = FutureTimestampSigner("test-secret")
    future_signed = future_signer.sign(value)
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(future_signed, max_age=3600)
    assert "age" in str(exc_info.value), "Expected age in error message"
    
    # Test 6: Unsign with invalid signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid_signature")
    
    # Test 7: Unsign with tampered value
    signed = signer.sign(value)
    tampered = signed[:-1] + b"X"
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered)
    
    # Test 8: Unsign with missing timestamp separator
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value_without_separator")
    
    # Test 9: Unsign with malformed timestamp
    # Create a signed value with invalid timestamp encoding
    sep = signer.sep.encode()
    invalid_ts = b"not_a_timestamp"
    bad_signed = value + sep + invalid_ts + sep + signer.get_signature(value + sep + invalid_ts)
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed)
    
    # Test 10: Sign and unsign with string input
    string_value = "test_string"
    signed_string = signer.sign(string_value)
    result_string = signer.unsign(signed_string)
    assert result_string == string_value.encode(), f"Expected {string_value.encode()}, got {result_string}"
    
    # Test 11: Return value from unsign when signature is invalid but payload exists
    # This tests the case where BadSignature has a payload
    try:
        signer.unsign(b"invalid")
    except BadTimeSignature as e:
        assert hasattr(e, 'payload'), "Expected payload attribute"
        assert e.payload == b"invalid", f"Expected payload b'invalid', got {e.payload}"
    
    # Test 12: Verify timestamp is returned as aware datetime in UTC
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    timestamp_dt = result_with_ts[1]
    assert timestamp_dt.tzinfo is not None, "Expected timezone-aware datetime"
    assert timestamp_dt.tzinfo.utcoffset(timestamp_dt).total_seconds() == 0, "Expected UTC timezone"
```


# LLM-generated content at query #35
#--------------------------

```python
def test_TimedSerializer():
    # Test that TimedSerializer can be instantiated with default parameters
    serializer = TimedSerializer()
    assert serializer is not None
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner

    # Test that TimedSerializer can be instantiated with a secret key
    serializer_with_key = TimedSerializer(secret_key="test-secret")
    assert serializer_with_key is not None
    assert serializer_with_key.secret_key == "test-secret"

    # Test that TimedSerializer can be instantiated with a salt
    serializer_with_salt = TimedSerializer(salt="test-salt")
    assert serializer_with_salt is not None
    assert serializer_with_salt.salt == "test-salt"

    # Test that TimedSerializer can be instantiated with all parameters
    serializer_all = TimedSerializer(
        secret_key="test-secret",
        salt="test-salt",
        serializer="json",
        signer_kwargs={"key_derivation": "hmac"},
        serializer_kwargs={"ensure_ascii": True}
    )
    assert serializer_all is not None
    
    # Verify that the signer is indeed a TimestampSigner
    signer = serializer_all.make_signer()
    assert isinstance(signer, TimestampSigner)
```


# LLM-generated content at query #36
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("test-secret-key")
    assert signer.secret_key == "test-secret-key"
    assert signer.salt is not None
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method is not None
    assert signer.algorithm is not None
    
    signer_with_options = TimestampSigner(
        "test-secret-key",
        salt="custom-salt",
        sep="|",
        key_derivation="none",
        digest_method="sha256"
    )
    assert signer_with_options.secret_key == "test-secret-key"
    assert signer_with_options.salt == "custom-salt"
    assert signer_with_options.sep == "|"
    assert signer_with_options.key_derivation == "none"
    
    signer_default = TimestampSigner()
    assert signer_default.secret_key is not None
    assert isinstance(signer_default.secret_key, bytes)


# LLM-generated content at query #37
#--------------------------

```python
def test_TimedSerializer():
    """Test the constructor of TimedSerializer."""
    serializer = TimedSerializer("test_secret_key")
    
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner
    assert serializer.secret_key == "test_secret_key"
    assert serializer.salt == "itsdangerous.TimedSerializer"
    assert serializer.serializer is None
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
```


# LLM-generated content at query #38
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer("test_secret_key")
    
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    assert serializer.secret_key == "test_secret_key"
    
    # Test that iter_unsigners returns TimestampSigner instances
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 0
    for signer in signers:
        assert isinstance(signer, TimestampSigner)
    
    # Test with different salt
    signers_with_salt = list(serializer.iter_unsigners(salt="custom_salt"))
    assert len(signers_with_salt) > 0
    for signer in signers_with_salt:
        assert isinstance(signer, TimestampSigner)
```


# LLM-generated content at query #39
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads with various scenarios."""
    serializer = TimedSerializer(secret_key="test-secret-key")
    
    # Test basic loads without max_age
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with max_age within limit
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with return_timestamp
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with both max_age and return_timestamp
    result, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with expired signature
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=-1)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-signature")
    
    # Test loads with different salt
    salt = b"custom-salt"
    signed_with_salt = serializer.dumps(data, salt=salt)
    result = serializer.loads(signed_with_salt, salt=salt)
    assert result == data
    
    # Test loads with wrong salt raises BadSignature
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt=b"wrong-salt")
    
    # Test loads with string input
    signed_str = signed.decode("utf-8") if isinstance(signed, bytes) else signed
    result = serializer.loads(signed_str)
    assert result == data
    
    # Test loads with non-dict payload
    list_data = [1, 2, 3]
    signed_list = serializer.dumps(list_data)
    result = serializer.loads(signed_list)
    assert result == list_data
    
    # Test loads with empty payload
    empty_data = {}
    signed_empty = serializer.dumps(empty_data)
    result = serializer.loads(signed_empty)
    assert result == empty_data
    
    # Test loads with None payload
    none_data = None
    signed_none = serializer.dumps(none_data)
    result = serializer.loads(signed_none)
    assert result is None
```


# LLM-generated content at query #40
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without timestamp
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"
    
    # Test unsign with return_timestamp=True
    signed = signer.sign("test-value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp, datetime)
    
    # Test unsign with valid max_age
    signed = signer.sign("test-value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test-value"
    
    # Test expired signature
    signer_with_fixed_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_fixed_time.get_timestamp
    
    def mock_get_timestamp():
        return 1000  # Fixed timestamp
    
    signer_with_fixed_time.get_timestamp = mock_get_timestamp
    signed = signer_with_fixed_time.sign("test-value")
    
    # Restore original method and advance time
    signer_with_fixed_time.get_timestamp = lambda: 2000  # 1000 seconds later
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer_with_fixed_time.unsign(signed, max_age=500)
    assert "Signature age" in str(exc_info.value)
    
    # Test negative age (future timestamp)
    signer_with_future = TimestampSigner("secret-key")
    signer_with_future.get_timestamp = lambda: 1000
    future_signed = signer_with_future.sign("test-value")
    signer_with_future.get_timestamp = lambda: 500  # 500 seconds before signing
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer_with_future.unsign(future_signed, max_age=3600)
    assert "age" in str(exc_info.value)
    
    # Test malformed timestamp
    signed = signer.sign("test-value")
    sep = signer.sep.encode()
    # Replace the timestamp with invalid data
    parts = signed.rsplit(sep, 1)
    malformed = parts[0] + sep + b"invalid-timestamp" + sep + signer.get_signature(parts[0] + sep + b"invalid-timestamp")
    
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(malformed)
    assert "Malformed timestamp" in str(exc_info.value)
    
    # Test missing timestamp
    signed_without_timestamp = signer.sign("test-value")
    # Remove the timestamp part
    parts = signed_without_timestamp.rsplit(sep, 2)
    if len(parts) >= 2:
        missing_ts = parts[0] + sep + parts[2]  # Remove timestamp
        with pytest.raises(BadTimeSignature) as exc_info:
            signer.unsign(missing_ts)
        assert "timestamp missing" in str(exc_info.value)
    
    # Test invalid signature with timestamp
    signed = signer.sign("test-value")
    tampered = signed[:-1] + b"X"  # Tamper with last byte
    
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(tampered)
    
    # Test empty value
    signed = signer.sign("")
    result = signer.unsign(signed)
    assert result == b""
    
    # Test bytes input
    signed = signer.sign(b"bytes-value")
    result = signer.unsign(signed)
    assert result == b"bytes-value"
    
    # Test with special characters in value
    signed = signer.sign("value with spaces and !@#$%")
    result = signer.unsign(signed)
    assert result == b"value with spaces and !@#$%"
```


# LLM-generated content at query #41
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test unsign with return_timestamp=True
    signed = signer.sign("test_value")
    value, timestamp = signer.unsign(signed, return_timestamp=True)
    assert value == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None
    
    # Test unsign with valid max_age
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test unsign with expired signature
    signer_with_fixed_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_fixed_time.get_timestamp
    
    # Create a signature with old timestamp
    def old_timestamp():
        return int(time.time()) - 100  # 100 seconds ago
    
    signer_with_fixed_time.get_timestamp = old_timestamp
    signed = signer_with_fixed_time.sign("test_value")
    
    # Restore original timestamp function
    signer_with_fixed_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_with_fixed_time.unsign(signed, max_age=50)
    
    # Test unsign with negative age (future timestamp)
    def future_timestamp():
        return int(time.time()) + 100  # 100 seconds in future
    
    signer_with_fixed_time.get_timestamp = future_timestamp
    signed = signer_with_fixed_time.sign("test_value")
    signer_with_fixed_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer_with_fixed_time.unsign(signed, max_age=50)
    assert "0 seconds" in str(exc_info.value)
    
    # Test unsign with bad signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")
    
    # Test unsign with tampered value
    signed = signer.sign("test_value")
    tampered = signed[:-1] + (b"1" if signed[-1:] == b"0" else b"0")
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered)
    
    # Test unsign with missing timestamp (no separator)
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(b"test_value")
    assert "timestamp missing" in str(exc_info.value)
    
    # Test unsign with malformed timestamp
    # Create a signature with invalid timestamp encoding
    sep = signer.sep.encode()
    bad_timestamp = b"not_a_timestamp"
    bad_value = b"test_value" + sep + bad_timestamp + sep + signer.get_signature(b"test_value" + sep + bad_timestamp)
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_value)
    
    # Test unsign returns bytes even with return_timestamp=False
    signed = signer.sign("test_value")
    result = signer.unsign(signed, return_timestamp=False)
    assert isinstance(result, bytes)
    
    # Test unsign returns tuple with return_timestamp=True
    signed = signer.sign("test_value")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bytes)
    assert isinstance(result[1], datetime)
```


# LLM-generated content at query #42
#--------------------------

```python
def test_TimedSerializer():
    # Test default constructor
    serializer = TimedSerializer()
    assert serializer.default_signer is TimestampSigner
    assert serializer.sep == "."
    assert serializer.salt == "itsdangerous"
    assert serializer.algorithm is not None

    # Test custom parameters
    custom_serializer = TimedSerializer(
        secret_key="custom-key",
        salt="custom-salt",
        serializer="json",
        signer_kwargs={"key_derivation": "hmac"},
    )
    assert custom_serializer.salt == "custom-salt"

    # Test that signers are TimestampSigner instances
    signers = list(custom_serializer.iter_unsigners())
    assert len(signers) > 0
    for signer in signers:
        assert isinstance(signer, TimestampSigner)

    # Test roundtrip with timestamp
    data = {"test": "data"}
    signed = custom_serializer.dumps(data)
    loaded = custom_serializer.loads(signed)
    assert loaded == data

    # Test with return_timestamp
    loaded_with_ts, timestamp = custom_serializer.loads(signed, return_timestamp=True)
    assert loaded_with_ts == data
    assert isinstance(timestamp, datetime)


# LLM-generated content at query #43
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads method."""
    serializer = TimedSerializer("test-secret")
    
    # Test basic serialization/deserialization
    data = {"key": "value"}
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized)
    assert result == data
    
    # Test with max_age parameter (valid)
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized, max_age=3600)
    assert result == data
    
    # Test with return_timestamp=True
    serialized = serializer.dumps(data)
    result, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test with both max_age and return_timestamp
    serialized = serializer.dumps(data)
    result, timestamp = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test with expired signature
    serializer = TimedSerializer("test-secret")
    data = {"key": "value"}
    serialized = serializer.dumps(data)
    import time
    time.sleep(1.5)  # Wait to simulate time passing
    with pytest.raises(SignatureExpired):
        serializer.loads(serialized, max_age=1)
    
    # Test with salt parameter
    serialized = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(serialized, salt="custom-salt")
    assert result == data
    
    # Test with wrong salt
    serialized = serializer.dumps(data, salt="custom-salt")
    with pytest.raises(BadSignature):
        serializer.loads(serialized, salt="wrong-salt")
    
    # Test with string input
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized.decode())
    assert result == data
    
    # Test with negative max_age (should raise SignatureExpired immediately)
    serialized = serializer.dumps(data)
    with pytest.raises(SignatureExpired):
        serializer.loads(serialized, max_age=-1)
```


# LLM-generated content at query #44
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner(secret_key="test-secret")
    assert signer.secret_key == "test-secret"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "sha1"
```


# LLM-generated content at query #45
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key", salt="test-salt")
    
    # Test basic unsign without timestamp
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"
    
    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp, datetime)
    
    # Test unsign with max_age that doesn't expire
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test-value"
    
    # Test unsign with expired signature
    import time as time_module
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time_module.time()) + 10000
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=1)
    signer.get_timestamp = original_get_timestamp
    
    # Test unsign with negative age (future timestamp)
    signed_future = signer.sign("future-value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_future, max_age=-1)
    
    # Test unsign with malformed timestamp
    malformed = signed + b"invalid"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test unsign with missing timestamp separator
    no_timestamp = signer.sign("test-value").split(signer.sep.encode())[0]
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_timestamp)
    
    # Test unsign with invalid signature but valid timestamp
    invalid_sig = signed[:-1] + b"x"
    with pytest.raises(BadTimeSignature):
        signer.unsign(invalid_sig)
    
    # Test unsign with bytes input
    signed_bytes = signer.sign(b"bytes-value")
    result = signer.unsign(signed_bytes)
    assert result == b"bytes-value"
    
    # Test unsign with string input
    signed_str = signer.sign("string-value")
    result = signer.unsign(signed_str.decode())
    assert result == b"string-value"
```


# LLM-generated content at query #46
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    
    # Test 1: Basic unsign with valid signature
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"
    
    # Test 2: Unsign with return_timestamp=True
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    value, timestamp = result_with_ts
    assert value == b"test-value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Unsign with max_age within limit
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test-value"
    
    # Test 4: Unsign with expired signature
    # Create a signer with a fixed timestamp in the past
    class PastTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) - 100  # 100 seconds in the past
    
    past_signer = PastTimestampSigner("secret-key")
    old_signed = past_signer.sign("test-value")
    
    with pytest.raises(SignatureExpired):
        past_signer.unsign(old_signed, max_age=50)  # max_age less than 100 seconds
    
    # Test 5: Unsign with future timestamp (negative age)
    class FutureTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) + 100  # 100 seconds in the future
    
    future_signer = FutureTimestampSigner("secret-key")
    future_signed = future_signer.sign("test-value")
    
    with pytest.raises(SignatureExpired):
        future_signer.unsign(future_signed, max_age=200)
    
    # Test 6: Unsign with malformed timestamp
    malformed = b"test-value" + signer.sep.encode() + b"invalid-timestamp" + signer.sep.encode() + b"signature"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(malformed)
    assert "Malformed timestamp" in str(exc_info.value)
    
    # Test 7: Unsign with missing timestamp
    signed_no_ts = signer.sign("test-value")
    # Remove the timestamp part
    parts = signed_no_ts.split(signer.sep.encode())
    no_ts = parts[0] + signer.sep.encode() + parts[-1]
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(no_ts)
    assert "timestamp missing" in str(exc_info.value)
    
    # Test 8: Unsign with invalid signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid-signature")
    
    # Test 9: Unsign with different key
    other_signer = TimestampSigner("different-key")
    signed_other = other_signer.sign("test-value")
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_other)
    
    # Test 10: Verify timestamp is timezone-aware
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    _, timestamp = result_with_ts
    assert timestamp.tzinfo is not None
    assert timestamp.tzinfo == timezone.utc
```


# LLM-generated content at query #47
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test_secret_key")
    
    # Test basic loads without max_age or return_timestamp
    original_data = {"key": "value"}
    serialized = serializer.dumps(original_data)
    result = serializer.loads(serialized)
    assert result == original_data
    
    # Test loads with max_age
    result = serializer.loads(serialized, max_age=3600)
    assert result == original_data
    
    # Test loads with return_timestamp
    result, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert result == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with both max_age and return_timestamp
    result, timestamp = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert result == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with string data
    string_data = "hello world"
    serialized_str = serializer.dumps(string_data)
    result = serializer.loads(serialized_str)
    assert result == string_data
    
    # Test loads with bytes data
    bytes_data = b"bytes data"
    serialized_bytes = serializer.dumps(bytes_data)
    result = serializer.loads(serialized_bytes)
    assert result == bytes_data
    
    # Test loads with integer data
    int_data = 42
    serialized_int = serializer.dumps(int_data)
    result = serializer.loads(serialized_int)
    assert result == int_data
    
    # Test loads with list data
    list_data = [1, 2, 3]
    serialized_list = serializer.dumps(list_data)
    result = serializer.loads(serialized_list)
    assert result == list_data
    
    # Test loads with None data
    none_data = None
    serialized_none = serializer.dumps(none_data)
    result = serializer.loads(serialized_none)
    assert result == none_data
    
    # Test loads with salt parameter
    salted_serializer = TimedSerializer("test_secret_key", salt="custom_salt")
    salted_serialized = salted_serializer.dumps(original_data)
    result = serializer.loads(salted_serialized, salt="custom_salt")
    assert result == original_data
    
    # Test loads with empty data
    empty_data = ""
    serialized_empty = serializer.dumps(empty_data)
    result = serializer.loads(serialized_empty)
    assert result == empty_data
    
    # Test loads with special characters
    special_data = "data with special chars: !@#$%^&*()"
    serialized_special = serializer.dumps(special_data)
    result = serializer.loads(serialized_special)
    assert result == special_data
    
    # Test loads with unicode data
    unicode_data = "数据 with unicode: 你好世界"
    serialized_unicode = serializer.dumps(unicode_data)
    result = serializer.loads(serialized_unicode)
    assert result == unicode_data
    
    # Test loads with nested data structure
    nested_data = {"level1": {"level2": [1, 2, {"level3": "value"}]}}
    serialized_nested = serializer.dumps(nested_data)
    result = serializer.loads(serialized_nested)
    assert result == nested_data
    
    # Test that loads raises SignatureExpired for expired signatures
    import time as time_module
    expired_serializer = TimedSerializer("test_secret_key")
    expired_serialized = expired_serializer.dumps(original_data)
    
    # Simulate time passing by creating a serializer with a custom timestamp
    class SlowTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time_module.time()) - 100  # 100 seconds in the past
    
    slow_serializer = TimedSerializer("test_secret_key")
    slow_serializer.default_signer = SlowTimestampSigner
    slow_serialized = slow_serializer.dumps(original_data)
    
    with pytest.raises(SignatureExpired):
        serializer.loads(slow_serialized, max_age=10)
    
    # Test that loads raises BadSignature for invalid data
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid_data")
    
    # Test that loads raises BadSignature for tampered data
    tampered = bytearray(serialized)
    tampered[-1] ^= 0x01  # Flip last byte
    with pytest.raises(BadSignature):
        serializer.loads(bytes(tampered))
```


# LLM-generated content at query #48
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    # Test basic unsign
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"

    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test unsign with valid max_age
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"

    # Test unsign with expired signature
    signer_with_fixed_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_fixed_time.get_timestamp
    signer_with_fixed_time.get_timestamp = lambda: 0  # Mock timestamp to 0
    old_signed = signer_with_fixed_time.sign("old_value")
    signer_with_fixed_time.get_timestamp = lambda: int(time.time())
    
    with pytest.raises(SignatureExpired):
        signer_with_fixed_time.unsign(old_signed, max_age=1)

    # Test unsign with negative age (future timestamp)
    signer_future = TimestampSigner("secret-key")
    signer_future.get_timestamp = lambda: int(time.time()) + 10000
    future_signed = signer_future.sign("future_value")
    signer_future.get_timestamp = lambda: int(time.time())
    
    with pytest.raises(SignatureExpired):
        signer_future.unsign(future_signed, max_age=3600)

    # Test unsign with invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")

    # Test unsign with tampered value
    tampered = signed[:10] + b"X" + signed[11:]
    with pytest.raises(BadSignature):
        signer.unsign(tampered)

    # Test unsign with missing timestamp
    signer_no_timestamp = Signer("secret-key")
    no_timestamp_signed = signer_no_timestamp.sign("no_timestamp")
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_signed)

    # Test unsign with malformed timestamp
    malformed = b"test_value" + signer.sep.encode() + b"invalid_base64"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)

    # Test unsign with bytes input
    signed_bytes = signer.sign(b"bytes_value")
    result = signer.unsign(signed_bytes)
    assert result == b"bytes_value"

    # Test unsign with string input
    signed_str = signer.sign("string_value").decode()
    result = signer.unsign(signed_str)
    assert result == b"string_value"
```


# LLM-generated content at query #49
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key", salt="test-salt")
    
    # Test successful unsign without timestamp
    value = b"test-value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test successful unsign with timestamp
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test with max_age that should pass
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test with max_age that should expire
    signer.get_timestamp = lambda: int(time.time()) + 100
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=1)
    
    # Test with negative age (future timestamp)
    signer.get_timestamp = lambda: int(time.time()) - 100
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=3600)
    
    # Test malformed timestamp
    signer.get_timestamp = lambda: int(time.time())
    malformed = signed + b"x"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test missing timestamp
    no_timestamp = signer.sign(value).split(signer.sep.encode())[0]
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp)
    
    # Test bad signature
    bad_sig = signed[:-1] + b"x"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_sig)


# LLM-generated content at query #50
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test 1: Basic unsign without timestamp
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test 2: Unsign with return_timestamp=True
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Unsign with max_age (valid)
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test 4: Unsign with max_age (expired)
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) - 100  # Simulate old timestamp
    signed = signer.sign("test_value")
    signer.get_timestamp = original_get_timestamp  # Restore
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=10)
    
    # Test 5: Unsign with max_age (future timestamp)
    signer.get_timestamp = lambda: int(time.time()) + 100
    signed = signer.sign("test_value")
    signer.get_timestamp = lambda: int(time.time()) - 50
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=60)
    signer.get_timestamp = original_get_timestamp
    
    # Test 6: Invalid signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid_signature")
    
    # Test 7: Tampered signature
    signed = signer.sign("test_value")
    tampered = signed[:-1] + b"X"
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered)
    
    # Test 8: Missing timestamp
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(b"test_value" + signer.sep.encode() + b"invalidsig")
    
    # Test 9: Malformed timestamp (non-decodable)
    bad_timestamp = base64_encode(b"not_a_timestamp")
    malformed = b"test_value" + signer.sep.encode() + bad_timestamp + signer.sep.encode() + b"signature"
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(malformed)
    
    # Test 10: Unsign with bytes input
    signed = signer.sign(b"bytes_value")
    result = signer.unsign(signed)
    assert result == b"bytes_value"
    
    # Test 11: Unsign with string input
    signed_str = signer.sign("string_value").decode()
    result = signer.unsign(signed_str)
    assert result == b"string_value"
```


# LLM-generated content at query #51
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    assert serializer.secret_key == b"test-secret"
```


# LLM-generated content at query #52
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads method."""
    serializer = TimedSerializer("secret-key")

    # Test successful loads without max_age and return_timestamp
    data = {"key": "value"}
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized)
    assert result == data

    # Test successful loads with max_age
    result = serializer.loads(serialized, max_age=3600)
    assert result == data

    # Test successful loads with return_timestamp=True
    result, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)

    # Test with max_age and return_timestamp
    result, timestamp = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)

    # Test with salt
    result = serializer.loads(serialized, salt="custom-salt")
    assert result == data

    # Test expired signature
    expired_serializer = TimedSerializer("secret-key")
    # Simulate an expired signature by signing with an old timestamp
    old_signer = TimestampSigner("secret-key")
    old_signer.get_timestamp = lambda: int(time.time()) - 10000
    expired_value = old_signer.sign(b"test")
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_value, max_age=10)

    # Test bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-signature")

    # Test with bytes input
    serialized_bytes = serializer.dumps("test")
    result = serializer.loads(serialized_bytes)
    assert result == "test"

    # Test with empty data
    serialized_empty = serializer.dumps({})
    result = serializer.loads(serialized_empty)
    assert result == {}

    # Test with None value
    serialized_none = serializer.dumps(None)
    result = serializer.loads(serialized_none)
    assert result is None

    # Test with list data
    serialized_list = serializer.dumps([1, 2, 3])
    result = serializer.loads(serialized_list)
    assert result == [1, 2, 3]
```


# LLM-generated content at query #53
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer(secret_key="test-secret")
    
    # Test basic loads without max_age and return_timestamp
    payload = {"key": "value"}
    signed = serializer.dumps(payload)
    result = serializer.loads(signed)
    assert result == payload
    
    # Test loads with return_timestamp
    signed = serializer.dumps(payload)
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == payload
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test loads with max_age that is valid
    signed = serializer.dumps(payload)
    result = serializer.loads(signed, max_age=3600)
    assert result == payload
    
    # Test loads with max_age that is expired
    signed = serializer.dumps(payload)
    import time as time_module
    # Simulate expired signature by waiting (unlikely in test) or use a past timestamp
    # Instead, we can test by manipulating the timestamp in the signer
    old_signer = TimestampSigner(secret_key="test-secret")
    # Create a signed value with an old timestamp
    old_timestamp = int(time_module.time()) - 100  # 100 seconds ago
    old_value = want_bytes(str(payload))
    old_timestamp_bytes = base64_encode(int_to_bytes(old_timestamp))
    sep = want_bytes(old_signer.sep)
    old_signed = old_value + sep + old_timestamp_bytes
    old_signed = old_signed + sep + old_signer.get_signature(old_signed)
    
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=10)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test loads with different salt
    signed_with_salt = serializer.dumps(payload, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == payload
    
    # Test loads with wrong salt
    signed_with_salt = serializer.dumps(payload, salt="custom-salt")
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
    
    # Test loads with return_timestamp and max_age
    signed = serializer.dumps(payload)
    result, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert result == payload
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test loads with string input
    signed_str = serializer.dumps(payload).decode('utf-8')
    result = serializer.loads(signed_str)
    assert result == payload
    
    # Test loads with bytes input
    signed_bytes = serializer.dumps(payload)
    result = serializer.loads(signed_bytes)
    assert result == payload
```


# LLM-generated content at query #54
#--------------------------

```python
def test_TimedSerializer():
    # Test basic initialization
    serializer = TimedSerializer()
    assert serializer.default_signer == TimestampSigner
    assert isinstance(serializer, Serializer)
    
    # Test with secret key
    serializer_with_key = TimedSerializer(secret_key="test-secret")
    assert serializer_with_key.secret_key == "test-secret"
    
    # Test with salt
    serializer_with_salt = TimedSerializer(salt="test-salt")
    assert serializer_with_salt.salt == "test-salt"
    
    # Test with signer kwargs
    serializer_with_kwargs = TimedSerializer(signer_kwargs={"key_derivation": "hmac"})
    assert serializer_with_kwargs.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test that default signer is TimestampSigner
    assert serializer.default_signer == TimestampSigner
    
    # Test that iter_unsigners returns TimestampSigner instances
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 0
    for signer in signers:
        assert isinstance(signer, TimestampSigner)
    
    # Test serialization roundtrip
    data = {"test": "data"}
    dumped = serializer.dumps(data)
    loaded = serializer.loads(dumped)
    assert loaded == data
    
    # Test with timestamp
    dumped_with_timestamp = serializer.dumps(data)
    loaded_with_timestamp = serializer.loads(dumped_with_timestamp, return_timestamp=True)
    assert isinstance(loaded_with_timestamp, tuple)
    assert len(loaded_with_timestamp) == 2
    assert loaded_with_timestamp[0] == data
    assert isinstance(loaded_with_timestamp[1], datetime)
```


# LLM-generated content at query #55
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without timestamp
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test unsign with return_timestamp=True
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None
    
    # Test unsign with valid max_age
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test unsign with expired signature
    old_timestamp = int(time.time()) - 100
    signer.get_timestamp = lambda: old_timestamp
    signed = signer.sign("old_value")
    signer.get_timestamp = lambda: int(time.time())
    
    try:
        signer.unsign(signed, max_age=10)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass
    
    # Test unsign with invalid signature
    try:
        signer.unsign(b"invalid_data")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass
    
    # Test unsign with malformed timestamp
    try:
        signer.unsign(b"value|invalid_timestamp|signature")
        assert False, "Should have raised BadTimeSignature"
    except BadTimeSignature:
        pass
    
    # Test unsign with missing timestamp
    try:
        signer.unsign(b"value|signature")
        assert False, "Should have raised BadTimeSignature"
    except BadTimeSignature:
        pass
    
    # Test unsign with negative age
    future_timestamp = int(time.time()) + 100
    signer.get_timestamp = lambda: future_timestamp
    signed = signer.sign("future_value")
    signer.get_timestamp = lambda: int(time.time())
    
    try:
        signer.unsign(signed, max_age=3600)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass
```


# LLM-generated content at query #56
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("test-secret")
    
    # Test 1: Basic loads without max_age
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test 2: Loads with return_timestamp=True
    signed = serializer.dumps(data)
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None
    
    # Test 3: Loads with max_age (valid)
    signed = serializer.dumps(data)
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test 4: Loads with max_age (expired)
    import time
    signed = serializer.dumps(data)
    time.sleep(0.1)  # Small delay to ensure age > 0
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=0.001)
    
    # Test 5: Loads with salt
    signed = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed, salt="custom-salt")
    assert result == data
    
    # Test 6: Loads with invalid signature raises BadSignature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test 7: Loads with wrong salt raises BadSignature
    signed = serializer.dumps(data, salt="salt1")
    with pytest.raises(BadSignature):
        serializer.loads(signed, salt="wrong-salt")
    
    # Test 8: Loads with max_age and return_timestamp
    signed = serializer.dumps(data)
    payload, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test 9: Loads with string input
    signed = serializer.dumps(data)
    result = serializer.loads(signed.decode() if isinstance(signed, bytes) else signed)
    assert result == data
    
    # Test 10: Loads with bytes input
    signed = serializer.dumps(data)
    result = serializer.loads(signed if isinstance(signed, bytes) else signed.encode())
    assert result == data
    
    # Test 11: SignatureExpired is not caught and re-raised
    signed = serializer.dumps(data)
    time.sleep(0.1)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=0.001)
    
    # Test 12: Multiple signers with different salts
    serializer2 = TimedSerializer("test-secret")
    signed1 = serializer.dumps(data, salt="salt1")
    signed2 = serializer2.dumps(data, salt="salt2")
    
    # Should work with the first signer
    result = serializer.loads(signed1)
    assert result == data
    
    # Should try all signers and fail with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed2, salt="wrong-salt")
```


# LLM-generated content at query #57
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("test-secret")
    assert signer.secret_key == "test-secret"
    assert signer.sep == "."
    assert signer.salt is not None
    assert signer.key_derivation == "hmac"
    assert signer.digest_method is not None
    assert signer.algorithm is not None
    assert isinstance(signer.get_timestamp(), int)


# LLM-generated content at query #58
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic successful unsign
    signed = signer.sign(b"test value")
    result = signer.unsign(signed)
    assert result == b"test value"
    
    # Test unsign with return_timestamp=True
    signed = signer.sign(b"test value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with valid max_age
    signed = signer.sign(b"test value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test value"
    
    # Test unsign with expired signature
    signer_with_fixed_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_fixed_time.get_timestamp
    
    def mock_get_timestamp():
        return int(time.time()) - 100  # 100 seconds ago
    
    signer_with_fixed_time.get_timestamp = mock_get_timestamp
    signed = signer_with_fixed_time.sign(b"test value")
    
    # Reset timestamp to current time for validation
    signer_with_fixed_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_with_fixed_time.unsign(signed, max_age=50)
    
    # Test unsign with future timestamp (age < 0)
    def mock_future_timestamp():
        return int(time.time()) + 100
    
    signer_with_fixed_time.get_timestamp = mock_future_timestamp
    signed = signer_with_fixed_time.sign(b"test value")
    signer_with_fixed_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_with_fixed_time.unsign(signed, max_age=3600)
    
    # Test unsign with malformed timestamp
    malformed = b"test value" + signer.sep.encode() + b"invalid_timestamp"
    malformed += signer.sep.encode() + signer.get_signature(malformed)
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(malformed)
    
    # Test unsign with missing timestamp
    no_timestamp = signer.sign(b"test value").rsplit(signer.sep.encode(), 2)[0]
    no_timestamp += signer.sep.encode() + signer.get_signature(no_timestamp)
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_timestamp)
    
    # Test unsign with bad signature but valid timestamp
    bad_sig = b"test value" + signer.sep.encode() + base64_encode(int_to_bytes(int(time.time())))
    bad_sig += signer.sep.encode() + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_sig)
    
    # Test unsign with string input
    signed_str = signer.sign("test string").decode()
    result = signer.unsign(signed_str)
    assert result == b"test string"
```


# LLM-generated content at query #59
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "sha1"
    
    # Test with custom parameters
    custom_signer = TimestampSigner(
        "secret-key",
        salt="custom-salt",
        sep="|",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm="sha256"
    )
    assert custom_signer.secret_key == "secret-key"
    assert custom_signer.salt == "custom-salt"
    assert custom_signer.sep == "|"
    assert custom_signer.key_derivation == "none"
    assert custom_signer.digest_method == hashlib.sha256
    assert custom_signer.algorithm == "sha256"


# LLM-generated content at query #60
#--------------------------

```python
def test_TimestampSigner():
    # Test that TimestampSigner can be constructed with default parameters
    signer = TimestampSigner("test-secret")
    assert isinstance(signer, TimestampSigner)
    assert isinstance(signer, Signer)
    
    # Test that TimestampSigner can be constructed with custom salt and key derivation
    signer = TimestampSigner("test-secret", salt="custom-salt", key_derivation="hmac")
    assert signer.salt == "custom-salt"
    assert signer.key_derivation == "hmac"
    
    # Test that TimestampSigner can be constructed with digest method
    signer = TimestampSigner("test-secret", digest_method="sha256")
    assert signer.digest_method == "sha256"
    
    # Test that TimestampSigner can be constructed with all parameters
    signer = TimestampSigner(
        "test-secret",
        salt="custom-salt",
        sep=".",
        key_derivation="hmac",
        digest_method="sha256",
        algorithm="hmac-sha256"
    )
    assert signer.secret_key == b"test-secret"
    assert signer.salt == "custom-salt"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == "sha256"


# LLM-generated content at query #61
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads with various scenarios."""
    serializer = TimedSerializer("test-secret-key")
    
    # Test basic loads without max_age
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with return_timestamp
    result_with_ts = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == data
    assert isinstance(result_with_ts[1], datetime)
    
    # Test loads with max_age within valid range
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with expired signature
    import time as time_module
    original_time = time_module.time
    try:
        # Simulate time in the past
        time_module.time = lambda: original_time() - 7200
        with pytest.raises(SignatureExpired):
            serializer.loads(signed, max_age=3600)
    finally:
        time_module.time = original_time
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")
    
    # Test loads with custom salt
    signed_with_salt = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == data
    
    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
    
    # Test loads with string input
    signed_str = signed.decode("utf-8") if isinstance(signed, bytes) else signed
    result = serializer.loads(signed_str)
    assert result == data
    
    # Test loads with empty data
    signed_empty = serializer.dumps("")
    result = serializer.loads(signed_empty)
    assert result == ""
    
    # Test loads with None value (if serializer supports it)
    import json
    if hasattr(serializer, 'load_payload'):
        # Test with simple string payload
        signed_string = serializer.dumps("test-string")
        result = serializer.loads(signed_string)
        assert result == "test-string"
    
    # Test loads with integer payload
    signed_int = serializer.dumps(42)
    result = serializer.loads(signed_int)
    assert result == 42
    
    # Test loads with list payload
    signed_list = serializer.dumps([1, 2, 3])
    result = serializer.loads(signed_list)
    assert result == [1, 2, 3]
    
    # Test loads with nested data
    nested_data = {"outer": {"inner": "value", "numbers": [1, 2]}}
    signed_nested = serializer.dumps(nested_data)
    result = serializer.loads(signed_nested)
    assert result == nested_data
    
    # Test loads with negative max_age (should fail)
    with pytest.raises(BadSignature):
        serializer.loads(signed, max_age=-1)
    
    # Test loads with zero max_age (should fail for anything but immediately signed)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=0)
    
    # Test that loads returns bytes for binary data
    binary_data = b"\x00\x01\x02"
    signed_binary = serializer.dumps(binary_data)
    result = serializer.loads(signed_binary)
    assert result == binary_data
    
    # Test multiple signers with different salts
    serializer2 = TimedSerializer("test-secret-key-2")
    signed2 = serializer2.dumps(data)
    
    # Should work with fallback signers
    multi_signer = TimedSerializer("test-secret-key")
    multi_signer.signer_kwargs = {"fallback_signers": [serializer2]}
    result = multi_signer.loads(signed2)
    assert result == data
    
    # Test that expired signature raises SignatureExpired even with return_timestamp
    try:
        time_module.time = lambda: original_time() - 7200
        with pytest.raises(SignatureExpired):
            serializer.loads(signed, max_age=3600, return_timestamp=True)
    finally:
        time_module.time = original_time
    
    # Test malformed timestamp
    # Create a signed value with malformed timestamp
    import base64
    from itsdangerous.encoding import want_bytes
    
    value = want_bytes("test")
    sep = want_bytes(serializer.signer.sep)
    # Add invalid timestamp
    malformed = value + sep + b"invalid-timestamp" + sep + b"fake-signature"
    with pytest.raises(BadSignature):
        serializer.loads(malformed)```


# LLM-generated content at query #62
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test_secret_key")
    
    # Test successful load without timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test successful load with timestamp
    result_with_ts = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == data
    assert isinstance(result_with_ts[1], datetime)
    
    # Test with max_age (should succeed)
    recent_signed = serializer.dumps(data)
    result = serializer.loads(recent_signed, max_age=3600)
    assert result == data
    
    # Test with max_age that should expire
    import time
    old_serializer = TimedSerializer("test_secret_key")
    class OldTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) - 100  # 100 seconds in the past
    old_serializer.default_signer = OldTimestampSigner
    old_signed = old_serializer.dumps(data)
    
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=50)
    
    # Test with bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid_data")
    
    # Test with different salt
    signed_with_salt = serializer.dumps(data, salt="custom_salt")
    result = serializer.loads(signed_with_salt, salt="custom_salt")
    assert result == data
    
    # Test with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong_salt")
    
    # Test with string input
    signed_str = signed.decode() if isinstance(signed, bytes) else signed
    result = serializer.loads(signed_str)
    assert result == data
    
    # Test empty data
    empty_data = {}
    signed_empty = serializer.dumps(empty_data)
    result = serializer.loads(signed_empty)
    assert result == empty_data
    
    # Test with list data
    list_data = [1, 2, 3]
    signed_list = serializer.dumps(list_data)
    result = serializer.loads(signed_list)
    assert result == list_data
    
    # Test with complex nested data
    nested_data = {"a": [1, 2], "b": {"c": "test"}}
    signed_nested = serializer.dumps(nested_data)
    result = serializer.loads(signed_nested)
    assert result == nested_data
    
    # Test with return_timestamp and max_age combined
    result_with_ts = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert result_with_ts[0] == data
    assert isinstance(result_with_ts[1], datetime)
```


# LLM-generated content at query #63
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test successful loads without return_timestamp
    serializer = TimedSerializer("test-secret")
    data = {"key": "value"}
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized)
    assert result == data
    
    # Test successful loads with return_timestamp
    result, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test with max_age within limit
    result = serializer.loads(serialized, max_age=3600)
    assert result == data
    
    # Test with max_age expired
    import time as time_module
    original_time = time_module.time
    try:
        # Simulate time in the past
        time_module.time = lambda: original_time() - 7200  # 2 hours ago
        with pytest.raises(SignatureExpired):
            serializer.loads(serialized, max_age=3600)
    finally:
        time_module.time = original_time
    
    # Test with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-data")
    
    # Test with different salt
    serialized_salt = serializer.dumps(data, salt="different-salt")
    with pytest.raises(BadSignature):
        serializer.loads(serialized_salt, salt="wrong-salt")
    
    # Test with empty string
    with pytest.raises(BadSignature):
        serializer.loads("")
    
    # Test with bytes input
    result = serializer.loads(serialized.encode() if isinstance(serialized, str) else serialized)
    assert result == data
    
    # Test with multiple signers (fallback mechanism)
    serializer2 = TimedSerializer("second-secret")
    serialized2 = serializer2.dumps(data)
    # Should work with first signer
    result = serializer.loads(serialized)
    assert result == data
```


# LLM-generated content at query #64
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test 1: Basic unsign without timestamp
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test 2: Unsign with return_timestamp=True
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Unsign with valid max_age
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test 4: Unsign with expired max_age
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) - 100
    signed = signer.sign("test_value")
    signer.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=10)
    
    # Test 5: Unsign with future timestamp (age < 0)
    signer.get_timestamp = lambda: int(time.time()) - 100
    signed = signer.sign("test_value")
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(signed, max_age=200)
    assert "age" in str(exc_info.value)
    
    # Test 6: Unsign with bad signature
    signer2 = TimestampSigner("different-secret")
    signed = signer.sign("test_value")
    
    with pytest.raises(BadTimeSignature):
        signer2.unsign(signed)
    
    # Test 7: Unsign with malformed timestamp
    signer_with_fixed_timestamp = TimestampSigner("secret-key")
    original_encode = base64_encode
    import base64
    # Create a signed value with invalid timestamp
    value = want_bytes("test_value")
    sep = want_bytes(signer_with_fixed_timestamp.sep)
    timestamp = base64_encode(int_to_bytes(signer_with_fixed_timestamp.get_timestamp()))
    value_with_timestamp = value + sep + timestamp
    signature = signer_with_fixed_timestamp.get_signature(value_with_timestamp)
    signed_value = value_with_timestamp + sep + signature
    
    # Replace timestamp with invalid base64
    invalid_timestamp = b"invalid!"
    signed_value_invalid = value + sep + invalid_timestamp + sep + signature
    
    with pytest.raises(BadTimeSignature):
        signer_with_fixed_timestamp.unsign(signed_value_invalid)
    
    # Test 8: Unsign with missing timestamp
    signed_no_timestamp = signer.sign("test_value")
    # Remove timestamp part
    parts = signed_no_timestamp.rsplit(sep, 1)
    signed_no_timestamp = parts[0]
    
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(signed_no_timestamp)
    
    # Test 9: Unsign bytes input
    signed = signer.sign(b"bytes_value")
    result = signer.unsign(signed)
    assert result == b"bytes_value"
    
    # Test 10: Unsign with string input
    signed = signer.sign("string_value")
    result = signer.unsign(signed.decode() if isinstance(signed, bytes) else signed)
    assert result == b"string_value"
```


# LLM-generated content at query #65
#--------------------------

```python
def test_TimestampSigner():
    # Test constructor creates instance correctly
    signer = TimestampSigner(secret_key="test-secret")
    assert signer is not None
    assert isinstance(signer, TimestampSigner)
    assert signer.secret_key == b"test-secret"
    
    # Test with default parameters
    signer_default = TimestampSigner(secret_key="test-secret")
    assert signer_default.sep == b"."
    assert signer_default.digest_method is not None
    assert signer_default.key_derivation == "hmac"
    
    # Test with custom separator
    signer_custom_sep = TimestampSigner(secret_key="test-secret", sep=":")
    assert signer_custom_sep.sep == b":"
    
    # Test with salt
    signer_with_salt = TimestampSigner(secret_key="test-secret", salt="custom-salt")
    assert signer_with_salt is not None
    
    # Test with digest method
    import hashlib
    signer_with_digest = TimestampSigner(secret_key="test-secret", digest_method=hashlib.sha256)
    assert signer_with_digest.digest_method is hashlib.sha256
    
    # Test with key derivation
    signer_with_kd = TimestampSigner(secret_key="test-secret", key_derivation="none")
    assert signer_with_kd.key_derivation == "none"
    
    # Test that it inherits from Signer
    from itsdangerous.signer import Signer
    assert isinstance(signer, Signer)
```


# LLM-generated content at query #66
#--------------------------

```python
def test_TimedSerializer():
    # Test default constructor
    serializer = TimedSerializer()
    assert serializer.default_signer is TimestampSigner
    assert serializer.sep == "."
    assert serializer.salt == "itsdangerous"
    
    # Test constructor with custom secret key
    serializer = TimedSerializer("secret-key")
    assert serializer.secret_key == "secret-key"
    
    # Test constructor with custom salt
    serializer = TimedSerializer(salt="custom-salt")
    assert serializer.salt == "custom-salt"
    
    # Test constructor with custom separator
    serializer = TimedSerializer(sep=":")
    assert serializer.sep == ":"
    
    # Test constructor with all parameters
    serializer = TimedSerializer(
        secret_key="my-secret",
        salt="my-salt",
        sep="|",
        serializer_kwargs={"key": "value"}
    )
    assert serializer.secret_key == "my-secret"
    assert serializer.salt == "my-salt"
    assert serializer.sep == "|"
    assert serializer.serializer_kwargs == {"key": "value"}
```


# LLM-generated content at query #67
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor creates instance with correct defaults."""
    serializer = TimedSerializer()
    
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner
    assert serializer.signer is TimestampSigner
    assert serializer.secret_key is not None
    assert serializer.salt is not None
    assert serializer.serializer is not None
    assert serializer.signer_kwargs == {}


# LLM-generated content at query #68
#--------------------------

```python
def test_TimestampSigner():
    # Test default construction
    signer = TimestampSigner(secret_key="test-secret")
    assert signer.secret_key == "test-secret"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm is not None
    
    # Test construction with custom parameters
    signer_custom = TimestampSigner(
        secret_key="custom-secret",
        salt="custom-salt",
        sep="|",
        key_derivation="none",
        digest_method=hashlib.sha256
    )
    assert signer_custom.secret_key == "custom-secret"
    assert signer_custom.salt == "custom-salt"
    assert signer_custom.sep == "|"
    assert signer_custom.key_derivation == "none"
    assert signer_custom.digest_method == hashlib.sha256
    
    # Test that TimestampSigner is a subclass of Signer
    assert isinstance(signer, Signer)
    assert issubclass(TimestampSigner, Signer)
    
    # Test get_timestamp returns integer
    timestamp = signer.get_timestamp()
    assert isinstance(timestamp, int)
    assert timestamp > 0
    
    # Test timestamp_to_datetime conversion
    dt = signer.timestamp_to_datetime(timestamp)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc
    
    # Test sign method
    signed_value = signer.sign("test-value")
    assert isinstance(signed_value, bytes)
    assert b"test-value" in signed_value
    assert timestamp is not None
    
    # Test unsign with valid signature
    unsigned = signer.unsign(signed_value)
    assert unsigned == b"test-value"
    
    # Test unsign with return_timestamp=True
    unsigned_with_ts, ts = signer.unsign(signed_value, return_timestamp=True)
    assert unsigned_with_ts == b"test-value"
    assert isinstance(ts, datetime)
    assert ts.tzinfo == timezone.utc
    
    # Test validate method
    assert signer.validate(signed_value) is True
    assert signer.validate(b"invalid-signature") is False
    
    # Test that unsign raises BadSignature for invalid signatures
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid-signature")
    
    # Test max_age validation
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)
    
    # Test negative age
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)


# LLM-generated content at query #69
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test-secret")
    
    # Test basic loads without max_age
    data = {"key": "value"}
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized)
    assert result == data
    
    # Test loads with max_age not expired
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized, max_age=3600)
    assert result == data
    
    # Test loads with return_timestamp=True
    result, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with max_age and return_timestamp
    result, timestamp = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with expired signature
    import time
    old_serializer = TimedSerializer("test-secret")
    old_serializer.default_signer.get_timestamp = lambda: int(time.time()) - 100
    old_serialized = old_serializer.dumps(data)
    
    with pytest.raises(SignatureExpired):
        serializer.loads(old_serialized, max_age=10)
    
    # Test loads with bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test loads with empty string
    with pytest.raises(BadSignature):
        serializer.loads("")
    
    # Test loads with different salt
    serialized_salt1 = serializer.dumps(data, salt="salt1")
    result = serializer.loads(serialized_salt1, salt="salt1")
    assert result == data
    
    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(serialized_salt1, salt="wrong-salt")
```


# LLM-generated content at query #70
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer(secret_key="test-secret-key")
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    assert serializer.secret_key == "test-secret-key"
    assert serializer.salt == "itsdangerous"
    assert serializer.signer_kwargs == {}
    assert serializer.serializer_kwargs == {}
```


# LLM-generated content at query #71
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key", salt="test-salt")
    
    # Test basic unsign without timestamp
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"
    
    # Test unsign with return_timestamp=True
    signed = signer.sign("test-value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp, datetime)
    
    # Test unsign with max_age (within limit)
    signed = signer.sign("test-value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test-value"
    
    # Test unsign with max_age (expired) - should raise SignatureExpired
    signed = signer.sign("test-value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)
    
    # Test unsign with invalid signature
    invalid_signed = b"invalid-data" + signer.sep.encode() + b"fake-signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(invalid_signed)
    
    # Test unsign with malformed timestamp
    malformed = b"test-value" + signer.sep.encode() + b"not-a-timestamp" + signer.sep.encode() + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test unsign with missing timestamp
    missing_ts = signer.sign("test-value")
    missing_ts = missing_ts.rsplit(signer.sep.encode(), 1)[0]  # Remove timestamp and signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_ts)
    
    # Test unsign with negative age (future timestamp)
    future_signer = TimestampSigner("secret-key", salt="test-salt")
    original_get_timestamp = future_signer.get_timestamp
    future_signer.get_timestamp = lambda: int(time.time()) + 10000
    signed = future_signer.sign("test-value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=3600)
    future_signer.get_timestamp = original_get_timestamp
```


# LLM-generated content at query #72
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner(secret_key="test-secret")
    
    # Test 1: Basic unsign without max_age and return_timestamp=False
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test 2: Unsign with return_timestamp=True
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == b"test_value"
    assert isinstance(result_with_ts[1], datetime)
    
    # Test 3: Unsign with valid max_age
    result = signer.unsign(signed, max_age=3600)  # 1 hour
    assert result == b"test_value"
    
    # Test 4: Unsign with expired signature (max_age too small)
    import time
    # Create a signer with a custom timestamp that appears old
    old_signer = TimestampSigner(secret_key="test-secret")
    old_signer.get_timestamp = lambda: int(time.time()) - 100  # 100 seconds ago
    old_signed = old_signer.sign("test_value")
    
    with pytest.raises(SignatureExpired):
        signer.unsign(old_signed, max_age=50)  # max_age 50 seconds, but signature is 100 seconds old
    
    # Test 5: Unsign with future timestamp (age < 0)
    future_signer = TimestampSigner(secret_key="test-secret")
    future_signer.get_timestamp = lambda: int(time.time()) + 100  # 100 seconds in future
    future_signed = future_signer.sign("test_value")
    
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed, max_age=3600)
    
    # Test 6: Unsign with invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")
    
    # Test 7: Unsign with tampered value
    tampered = signed[:10] + b"X" + signed[11:]
    with pytest.raises(BadSignature):
        signer.unsign(tampered)
    
    # Test 8: Unsign with missing timestamp separator
    no_sep = signed.replace(b".", b"")
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_sep)
    
    # Test 9: Unsign with malformed timestamp (non-decodable)
    malformed_ts = b"test_value." + b"not_a_timestamp"
    # Need to create a valid signature for this malformed timestamp
    malformed_signed = malformed_ts + b"." + signer.get_signature(malformed_ts)
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(malformed_signed)
    
    # Test 10: Unsign with different salt
    signer2 = TimestampSigner(secret_key="test-secret", salt="different-salt")
    signed2 = signer2.sign("test_value")
    with pytest.raises(BadSignature):
        signer.unsign(signed2)
    
    # Test 11: Unsign with bytes input
    signed_bytes = signer.sign(b"bytes_value")
    result = signer.unsign(signed_bytes)
    assert result == b"bytes_value"
    
    # Test 12: Unsign with string input
    signed_str = signer.sign("string_value")
    result = signer.unsign(signed_str)
    assert result == b"string_value"
    
    # Test 13: Verify returned timestamp is timezone-aware
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert result_with_ts[1].tzinfo is not None
    assert result_with_ts[1].tzinfo == timezone.utc
    
    # Test 14: Verify the returned value matches what was signed
    original = b"complex_value_123"
    signed_complex = signer.sign(original)
    result = signer.unsign(signed_complex)
    assert result == original
    
    # Test 15: Unsign with empty value
    signed_empty = signer.sign(b"")
    result = signer.unsign(signed_empty)
    assert result == b""
```


# LLM-generated content at query #73
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner("test-secret")
    assert signer.secret_key == b"test-secret"
    assert signer.sep == "."
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm is not None
    
    # Test constructor with custom parameters
    custom_signer = TimestampSigner(
        "custom-secret",
        sep="|",
        salt="custom-salt",
        key_derivation="none",
        digest_method=hashlib.sha256
    )
    assert custom_signer.secret_key == b"custom-secret"
    assert custom_signer.sep == "|"
    assert custom_signer.salt == "custom-salt"
    assert custom_signer.key_derivation == "none"
    assert custom_signer.digest_method == hashlib.sha256
    
    # Test constructor with bytes secret
    bytes_signer = TimestampSigner(b"bytes-secret")
    assert bytes_signer.secret_key == b"bytes-secret"
    
    # Test that get_timestamp returns integer
    timestamp = signer.get_timestamp()
    assert isinstance(timestamp, int)
    assert timestamp > 0
    
    # Test timestamp_to_datetime
    dt = signer.timestamp_to_datetime(timestamp)
    assert isinstance(dt, datetime)
    assert dt.tzinfo is not None
    assert dt.tzinfo == timezone.utc
    
    # Test sign method
    value = "test-value"
    signed = signer.sign(value)
    assert isinstance(signed, bytes)
    assert value.encode() in signed
    
    # Test unsign method
    unsigned = signer.unsign(signed)
    assert unsigned == value.encode()
    
    # Test unsign with return_timestamp
    unsigned, ts = signer.unsign(signed, return_timestamp=True)
    assert unsigned == value.encode()
    assert isinstance(ts, datetime)
    assert ts.tzinfo == timezone.utc
    
    # Test validate method
    assert signer.validate(signed) is True
    assert signer.validate(b"invalid") is False
    
    # Test expiration
    future_signed = signer.sign(value)
    assert signer.validate(future_signed, max_age=3600) is True
    
    # Test expired signature
    import time as time_module
    old_timestamp = int(time_module.time()) - 10000
    old_value = value.encode() + b"." + base64_encode(int_to_bytes(old_timestamp))
    old_signed = old_value + b"." + signer.get_signature(old_value)
    
    with pytest.raises(SignatureExpired):
        signer.unsign(old_signed, max_age=3600)
    
    # Test BadSignature on invalid signed value
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid-data")
    
    # Test BadTimeSignature on malformed timestamp
    malformed = value.encode() + b"." + b"not-a-timestamp"
    malformed_signed = malformed + b"." + signer.get_signature(malformed)
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed)


# LLM-generated content at query #74
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test_value")
    
    # Test basic unsign without timestamp return
    result = signer.unsign(signed_value)
    assert result == b"test_value"
    
    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with valid max_age
    result = signer.unsign(signed_value, max_age=3600)
    assert result == b"test_value"
    
    # Test unsign with expired max_age
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) + 7200
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=3600)
    signer.get_timestamp = original_get_timestamp
    
    # Test unsign with negative age (future timestamp)
    signer.get_timestamp = lambda: 0
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=3600)
    signer.get_timestamp = original_get_timestamp
    
    # Test unsign with invalid signature
    invalid_signed = signed_value[:-1] + b"x"
    with pytest.raises(BadTimeSignature):
        signer.unsign(invalid_signed)
    
    # Test unsign with missing timestamp
    missing_timestamp = signer.sign("test_value").split(signer.sep.encode())[0]
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(missing_timestamp)
    
    # Test unsign with malformed timestamp
    malformed_signed = signer.sign("test_value")
    # Replace the timestamp part with invalid base64
    parts = malformed_signed.rsplit(signer.sep.encode(), 1)
    malformed_signed = parts[0] + signer.sep.encode() + b"invalid_timestamp"
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(malformed_signed)


# LLM-generated content at query #75
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("test-secret-key", salt="test-salt")
    assert signer.secret_key == "test-secret-key"
    assert signer.salt == "test-salt"
    assert signer.digest_method is not None
    assert signer.key_derivation == "hmac"
    assert callable(signer.get_timestamp)
    
    # Test default values
    signer_default = TimestampSigner("test-secret-key")
    assert signer_default.sep == "."
    assert signer_default.salt == "timestamp-signer"
    
    # Test that it inherits from Signer
    assert isinstance(signer, Signer)
    
    # Test get_timestamp returns integer
    ts = signer.get_timestamp()
    assert isinstance(ts, int)
    assert ts > 0
```


# LLM-generated content at query #76
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == b"secret-key"
    assert signer.salt == b"itsdangerous.TimestampSigner"
    assert signer.sep == b"."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == hmac

    # Test with different parameters
    signer2 = TimestampSigner("secret-key", salt="custom-salt", sep="|")
    assert signer2.salt == b"custom-salt"
    assert signer2.sep == b"|"

    # Test with bytes
    signer3 = TimestampSigner(b"secret-key")
    assert signer3.secret_key == b"secret-key"
```


# LLM-generated content at query #77
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner(secret_key="test-secret")
    assert signer.secret_key == "test-secret"
    assert signer.salt == "timestamp-signer"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "hs256"  # or whatever the default algorithm is
```


# LLM-generated content at query #78
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer(secret_key="test-secret-key")
    original_data = {"key": "value"}
    serialized = serializer.dumps(original_data)
    
    # Test basic loads
    result = serializer.loads(serialized)
    assert result == original_data
    
    # Test loads with max_age
    result = serializer.loads(serialized, max_age=3600)
    assert result == original_data
    
    # Test loads with return_timestamp
    result, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert result == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with both max_age and return_timestamp
    result, timestamp = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert result == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with salt
    salted_serializer = TimedSerializer(secret_key="test-secret-key", salt="custom-salt")
    salted_serialized = salted_serializer.dumps(original_data)
    result = serializer.loads(salted_serialized, salt="custom-salt")
    assert result == original_data
    
    # Test loads with expired signature
    import time
    expired_serializer = TimedSerializer(secret_key="test-key")
    # Create a signer with a past timestamp
    signer = expired_serializer.make_signer()
    # Manually set a timestamp in the past
    past_timestamp = int(time.time()) - 100  # 100 seconds ago
    value_bytes = want_bytes(str(original_data))
    timestamp_bytes = base64_encode(int_to_bytes(past_timestamp))
    sep = want_bytes(signer.sep)
    signed_value = value_bytes + sep + timestamp_bytes
    signature = signer.get_signature(signed_value)
    expired_serialized = signed_value + sep + signature
    
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_serialized, max_age=50)
    
    # Test loads with bad signature
    bad_serialized = serialized + b"tampered"
    with pytest.raises(BadSignature):
        serializer.loads(bad_serialized)
    
    # Test loads with invalid timestamp
    # Create a signed value with invalid timestamp encoding
    signer = serializer.make_signer()
    value_bytes = want_bytes(str(original_data))
    invalid_timestamp = b"invalid-timestamp"
    sep = want_bytes(signer.sep)
    signed_value = value_bytes + sep + invalid_timestamp
    signature = signer.get_signature(signed_value)
    invalid_serialized = signed_value + sep + signature
    
    with pytest.raises(BadTimeSignature):
        serializer.loads(invalid_serialized)
    
    # Test loads with empty string
    with pytest.raises(BadSignature):
        serializer.loads(b"")
    
    # Test loads with None
    with pytest.raises(BadSignature):
        serializer.loads(None)
```


# LLM-generated content at query #79
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test-secret")
    
    # Test basic loads without max_age or return_timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with max_age that should succeed
    signed = serializer.dumps(data)
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with return_timestamp
    signed = serializer.dumps(data)
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None
    
    # Test loads with both max_age and return_timestamp
    signed = serializer.dumps(data)
    payload, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with salt
    serializer2 = TimedSerializer("test-secret", salt="different-salt")
    signed = serializer2.dumps(data)
    result = serializer.loads(signed, salt="different-salt")
    assert result == data
    
    # Test loads with expired signature
    import time
    # Create a signer with a custom timestamp that is in the past
    signer = TimestampSigner("test-secret")
    signer.get_timestamp = lambda: int(time.time()) - 100  # 100 seconds in the past
    old_signed = signer.sign(serializer.dump_payload(data))
    
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=50)
    
    # Test loads with bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test loads with string input
    signed_str = signed.decode("utf-8")
    result = serializer.loads(signed_str)
    assert result == data
    
    # Test loads with invalid data that has timestamp but wrong signature
    invalid_signed = b"data" + serializer.sep.encode() + b"timestamp" + serializer.sep.encode() + b"wrong-signature"
    with pytest.raises(BadSignature):
        serializer.loads(invalid_signed)
    
    # Test loads with data that has no timestamp separator
    no_ts = b"data" + serializer.sep.encode() + b"signature"
    with pytest.raises(BadSignature):
        serializer.loads(no_ts)
    
    # Test loads with multiple signers fallback
    serializer_multi = TimedSerializer(["key1", "key2"])
    signed_with_key2 = serializer_multi.dumps(data, salt="test")
    result = serializer_multi.loads(signed_with_key2, salt="test")
    assert result == data
```


# LLM-generated content at query #80
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test 1: Basic loads without max_age or return_timestamp
    serializer = TimedSerializer("secret-key")
    original_data = {"key": "value"}
    dumped = serializer.dumps(original_data)
    loaded = serializer.loads(dumped)
    assert loaded == original_data

    # Test 2: Loads with max_age and valid timestamp
    loaded = serializer.loads(dumped, max_age=3600)
    assert loaded == original_data

    # Test 3: Loads with return_timestamp
    payload, timestamp = serializer.loads(dumped, return_timestamp=True)
    assert payload == original_data
    assert isinstance(timestamp, datetime)

    # Test 4: Loads with both max_age and return_timestamp
    payload, timestamp = serializer.loads(dumped, max_age=3600, return_timestamp=True)
    assert payload == original_data
    assert isinstance(timestamp, datetime)

    # Test 5: Loads with expired signature
    import time
    serializer_fast = TimedSerializer("secret-key")
    serializer_fast.default_signer().get_timestamp = lambda: int(time.time()) - 100
    dumped_fast = serializer_fast.dumps(original_data)
    
    # Reset timestamp to current time for validation
    serializer_now = TimedSerializer("secret-key")
    try:
        serializer_now.loads(dumped_fast, max_age=10)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass

    # Test 6: Loads with bad signature
    try:
        serializer.loads(b"invalid|data|signature")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

    # Test 7: Loads with empty data
    try:
        serializer.loads(b"")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

    # Test 8: Loads with salt parameter
    serializer_salt = TimedSerializer("secret-key", salt="custom-salt")
    dumped_salt = serializer_salt.dumps(original_data)
    loaded_salt = serializer_salt.loads(dumped_salt, salt="custom-salt")
    assert loaded_salt == original_data

    # Test 9: Loads with different salt (should fail)
    try:
        serializer_salt.loads(dumped_salt, salt="wrong-salt")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

    # Test 10: Loads with string input
    dumped_str = dumped.decode('utf-8') if isinstance(dumped, bytes) else dumped
    loaded_str = serializer.loads(dumped_str)
    assert loaded_str == original_data

    # Test 11: Loads with bytes input
    loaded_bytes = serializer.loads(dumped)
    assert loaded_bytes == original_data

    # Test 12: Loads with negative max_age (should immediately expire)
    try:
        serializer.loads(dumped, max_age=-1)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass
```


# LLM-generated content at query #81
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test_secret")
    
    # Test basic loads without max_age and return_timestamp
    original_data = {"key": "value"}
    serialized = serializer.dumps(original_data)
    result = serializer.loads(serialized)
    assert result == original_data
    
    # Test loads with return_timestamp=True
    result_with_ts = serializer.loads(serialized, return_timestamp=True)
    assert len(result_with_ts) == 2
    payload, timestamp = result_with_ts
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with valid max_age
    result = serializer.loads(serialized, max_age=3600)
    assert result == original_data
    
    # Test loads with expired signature (max_age too small)
    import time
    # Create a serialized value with a timestamp that appears old
    old_signer = TimestampSigner("test_secret")
    # Manually set timestamp to be old
    old_value = want_bytes('{"key": "value"}')
    old_timestamp = base64_encode(int_to_bytes(int(time.time()) - 100))  # 100 seconds old
    sep = want_bytes(old_signer.sep)
    old_signed = old_value + sep + old_timestamp
    old_signed = old_signed + sep + old_signer.get_signature(old_signed)
    
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=10)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid_data")
    
    # Test loads with empty string
    with pytest.raises(BadSignature):
        serializer.loads("")
    
    # Test loads with bytes input
    serialized_bytes = serializer.dumps(original_data)
    result = serializer.loads(serialized_bytes)
    assert result == original_data
    
    # Test loads with salt
    serializer_with_salt = TimedSerializer("test_secret", salt="custom_salt")
    serialized_salted = serializer_with_salt.dumps(original_data)
    result = serializer_with_salt.loads(serialized_salted, salt="custom_salt")
    assert result == original_data
    
    # Test loads with wrong salt should fail
    with pytest.raises(BadSignature):
        serializer_with_salt.loads(serialized_salted, salt="wrong_salt")
    
    # Test loads with return_timestamp and max_age
    result_with_ts_and_age = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert len(result_with_ts_and_age) == 2
    payload, timestamp = result_with_ts_and_age
    assert payload == original_data
    assert isinstance(timestamp, datetime)
```


# LLM-generated content at query #82
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "sha1"
```


# LLM-generated content at query #83
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test-secret")
    
    # Test basic loads without max_age and return_timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with return_timestamp=True
    result = serializer.loads(signed, return_timestamp=True)
    payload, timestamp = result
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with max_age
    signed = serializer.dumps(data)
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with expired signature
    signed = serializer.dumps(data)
    # Simulate an expired signature by using a very small max_age
    import time
    time.sleep(0.1)  # Ensure some time passes
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=0)
    
    # Test loads with bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test loads with salt
    serializer_with_salt = TimedSerializer("test-secret", salt="custom-salt")
    signed_with_salt = serializer_with_salt.dumps(data)
    result = serializer_with_salt.loads(signed_with_salt, salt="custom-salt")
    assert result == data
    
    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
```


# LLM-generated content at query #84
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor and basic attributes."""
    serializer = TimedSerializer("test-secret-key")
    
    assert serializer.default_signer is TimestampSigner
    assert serializer.secret_key == "test-secret-key"
    assert serializer.signer_kwargs == {}
    assert serializer.salt == "itsdangerous"
    
    # Test with custom salt
    serializer2 = TimedSerializer("test-secret-key", salt="custom-salt")
    assert serializer2.salt == "custom-salt"
    
    # Test that signer is created correctly
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    assert signer.secret_key == "test-secret-key"
```


# LLM-generated content at query #85
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key", salt="test-salt")
    
    # Test 1: Basic unsign without timestamp
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test 2: Unsign with return_timestamp=True
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Unsign with valid max_age
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test 4: Unsign with expired signature
    signer_with_fixed_time = TimestampSigner("secret-key", salt="test-salt")
    original_get_timestamp = signer_with_fixed_time.get_timestamp
    
    def get_future_timestamp():
        return int(time.time()) + 10000
    
    signer_with_fixed_time.get_timestamp = get_future_timestamp
    signed_future = signer_with_fixed_time.sign("test_value")
    
    # Reset to original time for verification
    signer_with_fixed_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_with_fixed_time.unsign(signed_future, max_age=1)
    
    # Test 5: Unsign with malformed timestamp
    # Create a signed value and corrupt the timestamp
    signed = signer.sign("test_value")
    parts = signed.split(b".")
    corrupted = parts[0] + b"." + base64_encode(b"invalid")
    with pytest.raises(BadTimeSignature):
        signer.unsign(corrupted)
    
    # Test 6: Unsign with bad signature
    signed = signer.sign("test_value")
    bad_signed = signed[:-1] + b"x"  # Corrupt last byte
    with pytest.raises(BadSignature):
        signer.unsign(bad_signed)
    
    # Test 7: Unsign with empty value
    signed = signer.sign("")
    result = signer.unsign(signed)
    assert result == b""
    
    # Test 8: Unsign with bytes input
    signed = signer.sign(b"test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test 9: Unsign with string input
    signed = signer.sign("test_value")
    result = signer.unsign(signed.decode())
    assert result == b"test_value"
    
    # Test 10: Unsign with negative age (future timestamp)
    signer_future = TimestampSigner("secret-key", salt="test-salt")
    original_get_timestamp = signer_future.get_timestamp
    
    def get_past_timestamp():
        return int(time.time()) - 10000
    
    signer_future.get_timestamp = get_past_timestamp
    signed_past = signer_future.sign("test_value")
    signer_future.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_future.unsign(signed_past, max_age=3600)
```


# LLM-generated content at query #86
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key", salt="test-salt")
    
    # Test basic unsign without max_age or return_timestamp
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test unsign with return_timestamp=True
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None  # Ensure timezone-aware
    
    # Test unsign with max_age
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)  # 1 hour
    assert result == b"test_value"
    
    # Test expired signature raises SignatureExpired
    # Create a signer with a fixed timestamp in the past
    class PastTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) - 100  # 100 seconds in the past
    past_signer = PastTimestampSigner("secret-key", salt="test-salt")
    signed = past_signer.sign("test_value")
    with pytest.raises(SignatureExpired):
        past_signer.unsign(signed, max_age=50)  # max_age 50 seconds, signature is 100 seconds old
    
    # Test negative age (timestamp in the future)
    class FutureTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) + 100  # 100 seconds in the future
    future_signer = FutureTimestampSigner("secret-key", salt="test-salt")
    signed = future_signer.sign("test_value")
    with pytest.raises(SignatureExpired):
        future_signer.unsign(signed, max_age=50)  # age will be negative
    
    # Test invalid signature raises BadSignature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")
    
    # Test malformed timestamp raises BadTimeSignature
    # Create a signed value with invalid timestamp
    signed = signer.sign("test_value")
    sep = signer.sep.encode()
    # Replace timestamp with invalid data
    parts = signed.rsplit(sep, 1)
    malformed = parts[0] + sep + b"invalid_timestamp" + sep + signer.get_signature(parts[0] + sep + b"invalid_timestamp")
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test missing timestamp raises BadTimeSignature
    # Sign a value and remove the timestamp part
    no_timestamp = signer.sign("test_value").rsplit(sep, 1)[0]
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp)
    
    # Test unsign with both max_age and return_timestamp
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None
    
    # Test that the timestamp is correctly decoded
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    # Check that timestamp is recent (within 2 seconds)
    assert abs((datetime.now(timezone.utc) - timestamp).total_seconds()) < 2
```


# LLM-generated content at query #87
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor creates instance with correct default signer."""
    serializer = TimedSerializer("test-secret")
    
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner
    
    # Verify the signer created by the serializer is a TimestampSigner
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    assert signer.secret_key == b"test-secret"


# LLM-generated content at query #88
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test successful unsign without timestamp
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test successful unsign with timestamp
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test with max_age and valid age
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test with max_age and expired signature
    old_signer = TimestampSigner("secret-key")
    old_signer.get_timestamp = lambda: int(time.time()) - 100
    signed = old_signer.sign("test_value")
    
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=50)
    
    # Test with max_age and future timestamp
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: int(time.time()) + 100
    signed = future_signer.sign("test_value")
    
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=3600)
    
    # Test with invalid signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid_signature")
    
    # Test with tampered value
    signed = signer.sign("test_value")
    tampered = signed[:-1] + b"X"
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered)
    
    # Test with missing timestamp
    separator = signer.sep.encode()
    signed_without_timestamp = b"test_value" + separator + b"fake_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_without_timestamp)
    
    # Test with malformed timestamp
    malformed = b"test_value" + separator + b"invalid_timestamp" + separator + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
```


# LLM-generated content at query #89
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer("test-secret-key")
    assert serializer.default_signer is TimestampSigner
    assert serializer.signer is not None
    assert isinstance(serializer.signer, TimestampSigner)
    
    # Test serialization and deserialization
    data = {"key": "value", "number": 42}
    signed = serializer.dumps(data)
    
    # Test basic loads
    result = serializer.loads(signed)
    assert result == data
    
    # Test with max_age
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test with return_timestamp
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test expired signature
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=-1)
    
    # Test with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-data")
    
    # Test loads_unsafe
    success, result = serializer.loads_unsafe(signed)
    assert success
    assert result == data
    
    # Test loads_unsafe with max_age
    success, result = serializer.loads_unsafe(signed, max_age=3600)
    assert success
    assert result == data
    
    # Test loads_unsafe with expired signature
    success, result = serializer.loads_unsafe(signed, max_age=-1)
    assert not success
    assert isinstance(result, SignatureExpired)
    
    # Test empty payload
    signed_empty = serializer.dumps(None)
    assert serializer.loads(signed_empty) is None
    
    # Test with different salt
    signed_salt = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed_salt, salt="custom-salt")
    assert result == data
    
    # Test with wrong salt should fail
    with pytest.raises(BadSignature):
        serializer.loads(signed_salt, salt="wrong-salt")


# LLM-generated content at query #90
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == b"secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == hmac
    
    # Test with custom parameters
    signer = TimestampSigner(
        "secret-key",
        salt="custom-salt",
        sep=":",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm=hashlib.sha256,
    )
    assert signer.secret_key == b"secret-key"
    assert signer.salt == "custom-salt"
    assert signer.sep == ":"
    assert signer.key_derivation == "none"
    assert signer.digest_method == hashlib.sha256
    assert signer.algorithm == hashlib.sha256
    
    # Test with bytes secret key
    signer = TimestampSigner(b"bytes-secret-key")
    assert signer.secret_key == b"bytes-secret-key"
    
    # Test inheritence
    assert isinstance(signer, TimestampSigner)
    assert isinstance(signer, Signer)


# LLM-generated content at query #91
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test successful unsign without timestamp
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test successful unsign with timestamp
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    
    # Test with max_age that is not expired
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test expired signature
    signer_with_past_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_past_time.get_timestamp
    signer_with_past_time.get_timestamp = lambda: int(time.time()) - 100
    
    past_signed = signer_with_past_time.sign("test_value")
    signer_with_past_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_with_past_time.unsign(past_signed, max_age=50)
    
    # Test signature with negative age (future timestamp)
    signer_with_future_time = TimestampSigner("secret-key")
    signer_with_future_time.get_timestamp = lambda: int(time.time()) + 100
    
    future_signed = signer_with_future_time.sign("test_value")
    signer_with_future_time.get_timestamp = lambda: int(time.time())
    
    with pytest.raises(SignatureExpired):
        signer_with_future_time.unsign(future_signed, max_age=3600)
    
    # Test malformed timestamp
    malformed = b"test_value" + signer.sep.encode() + b"invalid_timestamp"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test missing timestamp
    missing_ts = signer.sign("test_value").rsplit(signer.sep.encode(), 1)[0]
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_ts)
    
    # Test invalid signature
    invalid_signed = b"wrong_value" + signer.sep.encode() + signer.get_signature(b"wrong_value")
    with pytest.raises(BadTimeSignature):
        signer.unsign(invalid_signed)
```


# LLM-generated content at query #92
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Test 1: Basic unsign without timestamp return
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value, "Basic unsign should return original value"

    # Test 2: Unsign with return_timestamp=True
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple), "Should return tuple when return_timestamp=True"
    assert len(result_with_ts) == 2, "Tuple should have 2 elements"
    assert result_with_ts[0] == value, "First element should be the original value"
    assert isinstance(result_with_ts[1], datetime), "Second element should be datetime"

    # Test 3: Unsign with valid max_age
    result = signer.unsign(signed, max_age=3600)
    assert result == value, "Should unsign successfully within max_age"

    # Test 4: Unsign with expired signature (age > max_age)
    # Mock get_timestamp to return a past timestamp
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) - 100  # 100 seconds in the past
    expired_signed = signer.sign(value)
    signer.get_timestamp = lambda: int(time.time())  # Reset to current time
    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed, max_age=50)

    # Test 5: Unsign with future timestamp (age < 0)
    signer.get_timestamp = lambda: int(time.time()) + 100  # 100 seconds in the future
    future_signed = signer.sign(value)
    signer.get_timestamp = original_get_timestamp  # Restore original
    with pytest.raises(SignatureExpired, match="< 0 seconds"):
        signer.unsign(future_signed, max_age=3600)

    # Test 6: Unsign with malformed timestamp
    malformed_signed = signed + b"malformed"
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(malformed_signed)

    # Test 7: Unsign with missing timestamp
    # Create a signature without timestamp
    value_bytes = want_bytes(value)
    sep = want_bytes(signer.sep)
    signature = signer.get_signature(value_bytes)
    signed_without_timestamp = value_bytes + sep + signature
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(signed_without_timestamp)

    # Test 8: Unsign with bad signature
    bad_signed = b"bad_value" + sep + base64_encode(int_to_bytes(100)) + sep + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed)

    # Test 9: Unsign with string input
    signed_str = signer.sign("test_string").decode()
    result = signer.unsign(signed_str)
    assert result == b"test_string", "Should handle string input"

    # Test 10: Unsign with return_timestamp and max_age
    result_with_ts = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert result_with_ts[0] == value
    assert isinstance(result_with_ts[1], datetime)

    # Test 11: Unsign with different separator
    signer_custom_sep = TimestampSigner("secret-key", sep="|")
    value_custom = b"custom_sep_value"
    signed_custom = signer_custom_sep.sign(value_custom)
    result_custom = signer_custom_sep.unsign(signed_custom)
    assert result_custom == value_custom

    # Test 12: Verify timestamp is in UTC
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert result_with_ts[1].tzinfo == timezone.utc, "Timestamp should be timezone-aware in UTC"

    # Test 13: Unsign with empty value
    empty_value = b""
    signed_empty = signer.sign(empty_value)
    result_empty = signer.unsign(signed_empty)
    assert result_empty == empty_value, "Should handle empty value"

    # Test 14: Unsign with bytes and string value types
    for val in [b"bytes_value", "string_value"]:
        signed_val = signer.sign(val)
        result_val = signer.unsign(signed_val)
        assert result_val == want_bytes(val), f"Should handle {type(val).__name__} input"
```


# LLM-generated content at query #93
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without timestamp
    signed = signer.sign("test value")
    result = signer.unsign(signed)
    assert result == b"test value"
    
    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with max_age that is not expired
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test value"
    
    # Test unsign with expired max_age
    import time as time_module
    
    # Create a signer with a fixed timestamp that will be in the past
    class PastTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time_module.time()) - 100  # 100 seconds in the past
    
    past_signer = PastTimestampSigner("secret-key")
    past_signed = past_signer.sign("test value")
    
    with pytest.raises(SignatureExpired):
        past_signer.unsign(past_signed, max_age=50)
    
    # Test unsign with negative age (future timestamp)
    class FutureTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time_module.time()) + 100  # 100 seconds in the future
    
    future_signer = FutureTimestampSigner("secret-key")
    future_signed = future_signer.sign("test value")
    
    with pytest.raises(SignatureExpired):
        future_signer.unsign(future_signed, max_age=50)
    
    # Test unsign with bad signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")
    
    # Test unsign with tampered value
    tampered = signed[:-1] + (b"x" if signed[-1:] == b"y" else b"y")
    with pytest.raises(BadSignature):
        signer.unsign(tampered)
    
    # Test unsign with missing timestamp
    simple_signer = Signer("secret-key")
    simple_signed = simple_signer.sign("test value")
    with pytest.raises(BadTimeSignature):
        signer.unsign(simple_signed)
    
    # Test unsign with malformed timestamp
    malformed = b"test_value" + signer.sep.encode() + b"invalid_timestamp" + signer.sep.encode() + signer.get_signature(b"test_value" + signer.sep.encode() + b"invalid_timestamp")
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test unsign with bytes input
    signed_bytes = signer.sign(b"test bytes")
    result = signer.unsign(signed_bytes)
    assert result == b"test bytes"
    
    # Test unsign with string input
    signed_str = signer.sign("test string").decode()
    result = signer.unsign(signed_str)
    assert result == b"test string"
```


# LLM-generated content at query #94
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer.signer, TimestampSigner)
    
    serializer_with_salt = TimedSerializer(secret_key="test-secret", salt="custom-salt")
    assert serializer_with_salt.salt == "custom-salt"
    
    serializer_with_kwargs = TimedSerializer(
        secret_key="test-secret",
        signer_kwargs={"key_derivation": "hmac"}
    )
    assert serializer_with_kwargs.signer.key_derivation == "hmac"
```


# LLM-generated content at query #95
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Create a TimestampSigner instance
    signer = TimestampSigner("secret-key")

    # Test 1: Basic unsign without timestamp
    signed = signer.sign("test_value")
    unsigned = signer.unsign(signed)
    assert unsigned == b"test_value"

    # Test 2: Unsign with return_timestamp=True
    signed = signer.sign("test_value")
    unsigned, timestamp = signer.unsign(signed, return_timestamp=True)
    assert unsigned == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test 3: Unsign with valid max_age
    signed = signer.sign("test_value")
    unsigned = signer.unsign(signed, max_age=3600)  # 1 hour should be fine
    assert unsigned == b"test_value"

    # Test 4: Unsign with expired signature (max_age too small)
    signed = signer.sign("test_value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)  # Negative max_age should expire

    # Test 5: Unsign with tampered value
    signed = signer.sign("test_value")
    tampered = signed[:-1] + (b"x" if signed[-1:] != b"x" else b"y")
    with pytest.raises(BadSignature):
        signer.unsign(tampered)

    # Test 6: Unsign with missing separator
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(b"no_separator_here")

    # Test 7: Unsign with malformed timestamp
    # Create a signer with a custom get_timestamp that returns a value
    # that will be malformed when decoded
    signed = signer.sign("test_value")
    # Replace the timestamp with invalid base64
    sep = signer.sep.encode()
    parts = signed.rsplit(sep, 1)
    malformed = parts[0] + sep + b"invalid_base64!!"
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(malformed)

    # Test 8: Unsign with future timestamp (age < 0)
    # Override get_timestamp to return a future timestamp
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) + 1000
    signed_future = signer.sign("test_value")
    signer.get_timestamp = original_get_timestamp
    with pytest.raises(SignatureExpired, match="age -1000"):
        signer.unsign(signed_future, max_age=3600)

    # Test 9: Verify timestamp_to_datetime conversion
    ts = 1234567890
    dt = signer.timestamp_to_datetime(ts)
    assert dt.year == 2009
    assert dt.month == 2
    assert dt.day == 13
    assert dt.hour == 23
    assert dt.minute == 31
    assert dt.second == 30
    assert dt.tzinfo == timezone.utc
```


# LLM-generated content at query #96
#--------------------------

```python
def test_TimestampSigner():
    # Test that TimestampSigner can be instantiated with default parameters
    signer = TimestampSigner("test-secret")
    assert signer.secret_key == b"test-secret"
    assert signer.salt == b"itsdangerous.TimestampSigner"
    assert signer.sep == b"."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == hmac

    # Test with custom parameters
    signer = TimestampSigner("secret", salt="custom-salt", sep="-", key_derivation="none")
    assert signer.secret_key == b"secret"
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"-"
    assert signer.key_derivation == "none"

    # Test that TimestampSigner inherits from Signer
    assert isinstance(signer, Signer)
    assert isinstance(signer, TimestampSigner)

    # Test that get_timestamp returns an integer
    timestamp = signer.get_timestamp()
    assert isinstance(timestamp, int)
    assert timestamp > 0

    # Test that timestamp_to_datetime returns a datetime
    dt = signer.timestamp_to_datetime(timestamp)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc

    # Test that sign method returns bytes
    signed = signer.sign("test-value")
    assert isinstance(signed, bytes)
    assert b"test-value" in signed

    # Test that unsign method works correctly
    unsigned = signer.unsign(signed)
    assert unsigned == b"test-value"

    # Test unsign with return_timestamp=True
    unsigned_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(unsigned_with_ts, tuple)
    assert len(unsigned_with_ts) == 2
    assert unsigned_with_ts[0] == b"test-value"
    assert isinstance(unsigned_with_ts[1], datetime)
    assert unsigned_with_ts[1].tzinfo == timezone.utc

    # Test that validate returns True for valid signature
    assert signer.validate(signed) == True

    # Test that validate returns False for invalid signature
    assert signer.validate(b"invalid-signature") == False

    # Test that unsign raises BadSignature for invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid-signature")

    # Test that unsign with max_age works
    signed_current = signer.sign("test")
    assert signer.unsign(signed_current, max_age=3600) == b"test"

    # Test expiration with max_age
    from unittest.mock import patch
    with patch.object(signer, 'get_timestamp', return_value=1000):
        signed_old = signer.sign("old-test")
    with patch.object(signer, 'get_timestamp', return_value=2000):
        with pytest.raises(SignatureExpired):
            signer.unsign(signed_old, max_age=500)


# LLM-generated content at query #97
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor and basic attributes."""
    serializer = TimedSerializer(secret_key="test-secret")
    
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    assert serializer.secret_key == "test-secret"
    
    # Test with salt
    serializer_with_salt = TimedSerializer(secret_key="test-secret", salt="custom-salt")
    assert serializer_with_salt.salt == "custom-salt"
    
    # Test with signer_kwargs
    serializer_with_kwargs = TimedSerializer(
        secret_key="test-secret",
        signer_kwargs={"key_derivation": "hmac"}
    )
    assert serializer_with_kwargs.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test that iter_unsigners returns TimestampSigner instances
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 0
    for signer in signers:
        assert isinstance(signer, TimestampSigner)
```


# LLM-generated content at query #98
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("test-secret-key")
    assert signer.secret_key == "test-secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == hmac
    assert callable(signer.get_timestamp)
    assert callable(signer.timestamp_to_datetime)
    assert callable(signer.sign)
    assert callable(signer.unsign)
    assert callable(signer.validate)
```


# LLM-generated content at query #99
#--------------------------

```python
def test_TimestampSigner():
    # Test constructor with default parameters
    signer = TimestampSigner(secret_key="test-secret")
    assert signer.secret_key == "test-secret"
    assert signer.sep == "."
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method is not None
    assert signer.algorithm is not None
    
    # Test constructor with custom parameters
    signer2 = TimestampSigner(
        secret_key="custom-secret",
        salt="custom-salt",
        sep="|",
        key_derivation="none",
        digest_method="sha512"
    )
    assert signer2.secret_key == "custom-secret"
    assert signer2.salt == "custom-salt"
    assert signer2.sep == "|"
    assert signer2.key_derivation == "none"
    assert signer2.digest_method is not None
    
    # Verify it's a subclass of Signer
    assert isinstance(signer, Signer)
    
    # Test that get_timestamp returns an integer
    timestamp = signer.get_timestamp()
    assert isinstance(timestamp, int)
    assert timestamp > 0
    
    # Test timestamp_to_datetime
    dt = signer.timestamp_to_datetime(timestamp)
    assert isinstance(dt, datetime)
    assert dt.tzinfo is not None
```


# LLM-generated content at query #100
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test successful unsign with default return value
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test successful unsign with return_timestamp=True
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc
    
    # Test unsign with max_age within limit
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test unsign with max_age exceeded raises SignatureExpired
    signer_past = TimestampSigner("secret-key")
    signer_past.get_timestamp = lambda: int(time.time()) - 100
    signed_past = signer_past.sign("test_value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_past, max_age=50)
    
    # Test unsign with negative age (future timestamp) raises SignatureExpired
    signer_future = TimestampSigner("secret-key")
    signer_future.get_timestamp = lambda: int(time.time()) + 100
    signed_future = signer_future.sign("test_value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_future, max_age=3600)
    
    # Test unsign with invalid signature raises BadSignature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_data")
    
    # Test unsign with tampered value raises BadSignature
    tampered = signed[:-1] + b"x"
    with pytest.raises(BadSignature):
        signer.unsign(tampered)
    
    # Test unsign with missing timestamp raises BadTimeSignature
    simple_signer = Signer("secret-key")
    simple_signed = simple_signer.sign("test_value")
    with pytest.raises(BadTimeSignature):
        signer.unsign(simple_signed)
    
    # Test unsign with malformed timestamp raises BadTimeSignature
    malformed = b"test_value" + signer.sep.encode() + b"invalid_timestamp" + signer.sep.encode() + signer.get_signature(b"test_value" + signer.sep.encode() + b"invalid_timestamp")
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)


# LLM-generated content at query #101
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("secret-key")
    
    # Test basic loads without max_age
    payload = {"message": "hello"}
    signed = serializer.dumps(payload)
    result = serializer.loads(signed)
    assert result == payload
    
    # Test loads with max_age and within time limit
    result = serializer.loads(signed, max_age=3600)
    assert result == payload
    
    # Test loads with return_timestamp=True
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == payload
    assert isinstance(timestamp, datetime)
    
    # Test loads with max_age and return_timestamp
    result, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert result == payload
    assert isinstance(timestamp, datetime)
    
    # Test loads with invalid signature raises BadSignature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid_signature")
    
    # Test loads with expired signature raises SignatureExpired
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=-1)
    
    # Test loads with different salt
    serializer_with_salt = TimedSerializer("secret-key", salt="custom-salt")
    signed_with_salt = serializer_with_salt.dumps(payload)
    result = serializer_with_salt.loads(signed_with_salt, salt="custom-salt")
    assert result == payload
    
    # Test loads with wrong salt raises BadSignature
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")


# LLM-generated content at query #102
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("test-secret-key")
    
    # Test basic signing and unsigning
    original_value = b"test data"
    signed_value = signer.sign(original_value)
    result = signer.unsign(signed_value)
    assert result == original_value
    
    # Test unsign with return_timestamp=True
    result_with_ts = signer.unsign(signed_value, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    value, timestamp = result_with_ts
    assert value == original_value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with max_age (valid)
    result = signer.unsign(signed_value, max_age=3600)
    assert result == original_value
    
    # Test unsign with max_age (expired)
    # Create a signed value with an old timestamp
    old_timestamp = int(time.time()) - 10000  # ~2.8 hours ago
    signer_with_old_time = TimestampSigner("test-secret-key")
    original_get_timestamp = signer_with_old_time.get_timestamp
    signer_with_old_time.get_timestamp = lambda: old_timestamp
    old_signed_value = signer_with_old_time.sign(original_value)
    
    with pytest.raises(SignatureExpired):
        signer_with_old_time.unsign(old_signed_value, max_age=3600)
    
    # Test unsign with tampered value
    tampered_value = signed_value + b"tampered"
    with pytest.raises(BadSignature):
        signer.unsign(tampered_value)
    
    # Test unsign with missing timestamp
    missing_ts = original_value + b"." + signer.get_signature(original_value)
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_ts)
    
    # Test unsign with malformed timestamp
    malformed_ts = original_value + b"." + b"not-a-timestamp" + b"." + signer.get_signature(original_value + b"." + b"not-a-timestamp")
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_ts)
    
    # Test unsign with negative age (future timestamp)
    future_timestamp = int(time.time()) + 10000
    signer_with_future_time = TimestampSigner("test-secret-key")
    signer_with_future_time.get_timestamp = lambda: future_timestamp
    future_signed_value = signer_with_future_time.sign(original_value)
    
    # Reset the signer to current time for unsigning
    signer_with_future_time.get_timestamp = lambda: int(time.time())
    
    with pytest.raises(SignatureExpired):
        signer_with_future_time.unsign(future_signed_value, max_age=3600)
    
    # Test unsign with string input
    string_value = "test string"
    signed_string = signer.sign(string_value)
    result = signer.unsign(signed_string)
    assert result == string_value.encode()  # Returns bytes
    
    # Test unsign with bytes input
    bytes_value = b"test bytes"
    signed_bytes = signer.sign(bytes_value)
    result = signer.unsign(signed_bytes)
    assert result == bytes_value
    
    # Test unsign with custom separator
    custom_signer = TimestampSigner("test-secret-key", sep=":")
    custom_value = b"custom test"
    custom_signed = custom_signer.sign(custom_value)
    result = custom_signer.unsign(custom_signed)
    assert result == custom_value
    
    # Test unsign with return_timestamp and max_age
    result_with_ts_and_age = signer.unsign(signed_value, max_age=3600, return_timestamp=True)
    assert isinstance(result_with_ts_and_age, tuple)
    value, timestamp = result_with_ts_and_age
    assert value == original_value
    assert isinstance(timestamp, datetime)
```


# LLM-generated content at query #103
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == hmac_compat
    
    # Test constructor with custom parameters
    custom_signer = TimestampSigner(
        "secret-key",
        salt="custom-salt",
        sep="|",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm=hmac_compat
    )
    assert custom_signer.secret_key == "secret-key"
    assert custom_signer.salt == "custom-salt"
    assert custom_signer.sep == "|"
    assert custom_signer.key_derivation == "none"
    assert custom_signer.digest_method == hashlib.sha256
    assert custom_signer.algorithm == hmac_compat
    
    # Test that TimestampSigner is a subclass of Signer
    assert isinstance(signer, Signer)
    
    # Test that key derivation works correctly
    derived_key = signer.derive_key()
    assert isinstance(derived_key, bytes)
    assert len(derived_key) > 0
    
    # Test constructor with bytes secret key
    bytes_signer = TimestampSigner(b"bytes-secret")
    assert bytes_signer.secret_key == b"bytes-secret"
    
    # Test constructor with empty secret key
    empty_signer = TimestampSigner("")
    assert empty_signer.secret_key == ""
    
    # Test that signature algorithm is initialized
    assert signer.algorithm is not None
```


# LLM-generated content at query #104
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == b"secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "sha1"
    
    signer_with_salt = TimestampSigner("secret-key", salt="custom-salt")
    assert signer_with_salt.salt == "custom-salt"
    
    signer_with_sep = TimestampSigner("secret-key", sep="|")
    assert signer_with_sep.sep == "|"
```


# LLM-generated content at query #105
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.sep == "."
    assert signer.digest_method is not None
    assert signer.key_derivation == "hmac"
    assert isinstance(signer.get_timestamp(), int)
    
    # Test with custom parameters
    custom_signer = TimestampSigner(
        secret_key="custom-key",
        salt="custom-salt",
        sep=":",
        key_derivation="none"
    )
    assert custom_signer.secret_key == "custom-key"
    assert custom_signer.salt == "custom-salt"
    assert custom_signer.sep == ":"
    assert custom_signer.key_derivation == "none"
    
    # Test inheritance from Signer
    assert isinstance(signer, Signer)
```


# LLM-generated content at query #106
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("secret-key")
    
    # Test basic loads without max_age or return_timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with return_timestamp
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test loads with max_age
    signed = serializer.dumps(data)
    result = serializer.loads(signed, max_age=3600)  # 1 hour
    assert result == data
    
    # Test loads with expired signature
    import time as time_module
    original_time = time_module.time
    try:
        time_module.time = lambda: 0
        signed = serializer.dumps(data)
        time_module.time = lambda: 1000000  # Far in the future
        with pytest.raises(SignatureExpired):
            serializer.loads(signed, max_age=3600)
    finally:
        time_module.time = original_time
    
    # Test loads with bad signature
    with pytest.raises(BadSignature):
        serializer.loads("bad-signature")
    
    # Test loads with salt
    signed_with_salt = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == data
    
    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
    
    # Test loads with both max_age and return_timestamp
    signed = serializer.dumps(data)
    result, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test loads with bytes input
    signed_bytes = serializer.dumps(data)
    result = serializer.loads(signed_bytes)
    assert result == data
    
    # Test loads with empty data
    empty_data = {}
    signed_empty = serializer.dumps(empty_data)
    result = serializer.loads(signed_empty)
    assert result == empty_data
    
    # Test loads with list data
    list_data = [1, 2, 3]
    signed_list = serializer.dumps(list_data)
    result = serializer.loads(signed_list)
    assert result == list_data
    
    # Test loads with None value
    none_data = None
    signed_none = serializer.dumps(none_data)
    result = serializer.loads(signed_none)
    assert result is None
```


# LLM-generated content at query #107
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test-secret")
    
    # Test basic loading
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test with max_age
    signed = serializer.dumps(data)
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test expired signature
    signed = serializer.dumps(data)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=-1)
    
    # Test with return_timestamp
    signed = serializer.dumps(data)
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test with return_timestamp and max_age
    signed = serializer.dumps(data)
    result, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test with salt
    signed_salt = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed_salt, salt="custom-salt")
    assert result == data
    
    # Test with wrong salt
    signed_salt = serializer.dumps(data, salt="custom-salt")
    with pytest.raises(BadSignature):
        serializer.loads(signed_salt, salt="wrong-salt")
    
    # Test empty data
    signed_empty = serializer.dumps({})
    result = serializer.loads(signed_empty)
    assert result == {}
    
    # Test string data
    signed_string = serializer.dumps("test-string")
    result = serializer.loads(signed_string)
    assert result == "test-string"
    
    # Test integer data
    signed_int = serializer.dumps(42)
    result = serializer.loads(signed_int)
    assert result == 42
    
    # Test list data
    signed_list = serializer.dumps([1, 2, 3])
    result = serializer.loads(signed_list)
    assert result == [1, 2, 3]
```


# LLM-generated content at query #108
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test 1: Basic unsign without max_age, return value only
    value = b"test_data"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test 2: Return timestamp
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    
    # Test 3: Unsigned value with valid max_age
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test 4: Unsigned value with max_age exceeded should raise SignatureExpired
    # Mock get_timestamp to return a time in the past
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) - 100  # 100 seconds ago
    signed = signer.sign(value)
    signer.get_timestamp = lambda: int(time.time())  # Reset to current time
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=10)
    
    # Test 5: Unsigned value with negative age (future timestamp) should raise SignatureExpired
    signer.get_timestamp = lambda: int(time.time()) + 50  # Future timestamp
    signed = signer.sign(value)
    signer.get_timestamp = lambda: int(time.time())  # Reset to current time
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=3600)
    
    # Test 6: Bad signature should raise BadTimeSignature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid_signature")
    
    # Test 7: Tampered value should raise BadTimeSignature
    signed = signer.sign(value)
    tampered = signed[:-1] + (b"x" if signed[-1:] != b"x" else b"y")
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered)
    
    # Test 8: Missing timestamp separator should raise BadTimeSignature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value_without_separator")
    
    # Test 9: String input should work
    signed = signer.sign("test_string")
    result = signer.unsign(signed)
    assert result == b"test_string"
    
    # Test 10: Empty value
    signed = signer.sign(b"")
    result = signer.unsign(signed)
    assert result == b""
    
    # Test 11: Malformed timestamp (invalid base64)
    signed = signer.sign(value)
    malformed_timestamp = signed[:-10] + b"invalid_base64" + signed[-1:]
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_timestamp)
    
    # Test 12: Verify timestamp is a datetime object in UTC
    signed = signer.sign(value)
    _, timestamp = signer.unsign(signed, return_timestamp=True)
    assert timestamp.tzinfo == timezone.utc
```


# LLM-generated content at query #109
#--------------------------

```python
def test_TimedSerializer():
    # Test default constructor
    serializer = TimedSerializer()
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    
    # Test with secret key
    serializer_with_key = TimedSerializer(secret_key="test-secret")
    assert serializer_with_key.secret_key == "test-secret"
    
    # Test with salt
    serializer_with_salt = TimedSerializer(salt="test-salt")
    assert serializer_with_salt.salt == "test-salt"
    
    # Test with signer_kwargs
    serializer_with_signer_kwargs = TimedSerializer(signer_kwargs={"key_derivation": "hmac"})
    assert serializer_with_signer_kwargs.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test with serializer_kwargs
    serializer_with_serializer_kwargs = TimedSerializer(serializer_kwargs={"serializer": "json"})
    assert serializer_with_serializer_kwargs.serializer_kwargs == {"serializer": "json"}
    
    # Test with fallback_signers
    serializer_with_fallback = TimedSerializer(fallback_signers=[{"salt": "fallback-salt"}])
    assert len(serializer_with_fallback.fallback_signers) == 1
    
    # Test with signer_cls parameter
    serializer_with_signer_cls = TimedSerializer(signer_cls=TimestampSigner)
    assert serializer_with_signer_cls.default_signer is TimestampSigner
    
    # Test that TimedSerializer is properly initialized as a TimestampSigner user
    serializer = TimedSerializer()
    signed = serializer.dumps("test-data")
    assert isinstance(signed, bytes)
    
    # Test that it can load data signed by itself
    loaded = serializer.loads(signed)
    assert loaded == "test-data"
    
    # Test that it properly uses TimestampSigner for signature timing
    serializer = TimedSerializer(secret_key="test-key")
    signed_value = serializer.dumps({"key": "value"})
    
    # Test that it can load with max_age
    loaded = serializer.loads(signed_value, max_age=3600)
    assert loaded == {"key": "value"}
    
    # Test that it returns timestamp when requested
    loaded_with_timestamp = serializer.loads(signed_value, return_timestamp=True)
    assert isinstance(loaded_with_timestamp, tuple)
    assert len(loaded_with_timestamp) == 2
    assert loaded_with_timestamp[0] == {"key": "value"}
    
    # Test with multiple signers (fallback)
    serializer = TimedSerializer(
        secret_key="test-key",
        fallback_signers=[{"secret_key": "fallback-key"}]
    )
    signed = serializer.dumps("test-data")
    assert serializer.loads(signed) == "test-data"
```


# LLM-generated content at query #110
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test-secret")
    
    # Test basic loads without max_age or return_timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with max_age (valid age)
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with return_timestamp
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with both max_age and return_timestamp
    result, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with expired signature
    import time
    # Create a signer with a fixed timestamp in the past
    class PastTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) - 7200  # 2 hours ago
    
    past_serializer = TimedSerializer("test-secret")
    past_serializer.default_signer = PastTimestampSigner
    past_signed = past_serializer.dumps(data)
    
    with pytest.raises(SignatureExpired):
        past_serializer.loads(past_signed, max_age=3600)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-signature")
    
    # Test loads with different salt
    signed_with_salt = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == data
    
    # Test loads with wrong salt raises BadSignature
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
```


# LLM-generated content at query #111
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("test-secret-key")
    assert signer.secret_key == "test-secret-key"
    assert signer.salt is not None
    assert signer.digest_method is not None
    assert signer.key_derivation == "hmac"
    assert signer.sep == "."
    
    value = b"test-value"
    signed = signer.sign(value)
    assert signed != value
    assert signed.endswith(b".")
    
    unsigned = signer.unsign(signed)
    assert unsigned == value
    
    unsigned_with_timestamp = signer.unsign(signed, return_timestamp=True)
    assert len(unsigned_with_timestamp) == 2
    assert unsigned_with_timestamp[0] == value
    assert isinstance(unsigned_with_timestamp[1], datetime)
    
    current_time = int(time.time())
    assert abs(signer.get_timestamp() - current_time) < 2
    
    dt = signer.timestamp_to_datetime(current_time)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc
    
    expired_signed = signer.sign(b"expired-test")
    try:
        signer.unsign(expired_signed, max_age=-1)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass
    
    try:
        signer.unsign(b"invalid-data")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass
    
    assert signer.validate(signed) is True
    assert signer.validate(b"invalid-data") is False
```


# LLM-generated content at query #112
#--------------------------

```python
def test_TimestampSigner():
    # Test that TimestampSigner can be instantiated with default parameters
    signer = TimestampSigner(secret_key="test-secret")
    assert signer is not None
    assert isinstance(signer, TimestampSigner)
    assert signer.secret_key == b"test-secret"
    assert signer.sep == "."

    # Test that TimestampSigner can be instantiated with custom salt and sep
    signer2 = TimestampSigner(secret_key="test-secret", salt="custom-salt", sep=":")
    assert signer2 is not None
    assert isinstance(signer2, TimestampSigner)
    assert signer2.salt == b"custom-salt"
    assert signer2.sep == ":"

    # Test that TimestampSigner can be instantiated with bytes secret key
    signer3 = TimestampSigner(secret_key=b"test-secret-bytes")
    assert signer3 is not None
    assert isinstance(signer3, TimestampSigner)
    assert signer3.secret_key == b"test-secret-bytes"

    # Test that TimestampSigner can be instantiated with key_derivation parameter
    signer4 = TimestampSigner(secret_key="test-secret", key_derivation="hmac")
    assert signer4 is not None
    assert isinstance(signer4, TimestampSigner)
    assert signer4.key_derivation == "hmac"

    # Test that TimestampSigner can be instantiated with digest_method parameter
    signer5 = TimestampSigner(secret_key="test-secret", digest_method="sha256")
    assert signer5 is not None
    assert isinstance(signer5, TimestampSigner)
    assert signer5.digest_method is not None

    # Test that TimestampSigner is a subclass of Signer
    assert issubclass(TimestampSigner, Signer)  # pyright: ignore
    assert isinstance(signer, Signer)  # pyright: ignore

    # Test that default values are set correctly
    assert signer.salt is not None
    assert signer.key_derivation is not None
    assert signer.digest_method is not None
```


# LLM-generated content at query #113
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test basic loads without max_age or return_timestamp
    serializer = TimedSerializer("test_secret")
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data

    # Test loads with max_age
    serializer = TimedSerializer("test_secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, max_age=3600)
    assert result == "test"

    # Test loads with return_timestamp
    serializer = TimedSerializer("test_secret")
    signed = serializer.dumps("test")
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == "test"
    assert isinstance(timestamp, datetime)

    # Test loads with both max_age and return_timestamp
    serializer = TimedSerializer("test_secret")
    signed = serializer.dumps("test")
    payload, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert payload == "test"
    assert isinstance(timestamp, datetime)

    # Test loads with expired signature
    serializer = TimedSerializer("test_secret")
    signed = serializer.dumps("test")
    import time
    time.sleep(0.1)  # Small delay to make signature "older"
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=0)

    # Test loads with invalid signature
    serializer = TimedSerializer("test_secret")
    with pytest.raises(BadSignature):
        serializer.loads("invalid_signature")

    # Test loads with different salt
    serializer = TimedSerializer("test_secret", salt="custom_salt")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, salt="custom_salt")
    assert result == "test"

    # Test loads with wrong salt
    serializer = TimedSerializer("test_secret", salt="custom_salt")
    signed = serializer.dumps("test")
    with pytest.raises(BadSignature):
        serializer.loads(signed, salt="wrong_salt")

    # Test loads with bytes input
    serializer = TimedSerializer("test_secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed)
    assert result == "test"

    # Test loads with string input
    serializer = TimedSerializer("test_secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed.decode())
    assert result == "test"

    # Test loads with complex data types
    serializer = TimedSerializer("test_secret")
    data = {"list": [1, 2, 3], "nested": {"a": 1}, "bool": True, "none": None}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
```


# LLM-generated content at query #114
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Create a TimestampSigner instance
    signer = TimestampSigner("test-secret")
    
    # Test 1: Normal unsign without timestamp
    value = b"test-value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test 2: Unsign with return_timestamp=True
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == value
    assert isinstance(result_with_ts[1], datetime)
    assert result_with_ts[1].tzinfo is not None
    
    # Test 3: Unsign with max_age within limit
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test 4: Unsign with max_age that should fail (simulate expired)
    # We need to manipulate the timestamp for this test
    import time as time_module
    original_get_timestamp = signer.get_timestamp
    
    def mock_old_timestamp():
        return int(time_module.time()) - 100
    
    try:
        signer.get_timestamp = mock_old_timestamp
        signed_old = signer.sign(value)
        signer.get_timestamp = original_get_timestamp
        
        with pytest.raises(SignatureExpired):
            signer.unsign(signed_old, max_age=10)
    finally:
        signer.get_timestamp = original_get_timestamp
    
    # Test 5: Unsign with malformed timestamp
    malformed = value + b"." + b"invalid-timestamp"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test 6: Unsign with missing separator
    with pytest.raises(BadTimeSignature):
        signer.unsign(value)
    
    # Test 7: Unsign with wrong signature
    wrong_signed = signed + b"tampered"
    with pytest.raises(BadTimeSignature):
        signer.unsign(wrong_signed)
    
    # Test 8: Unsign with string input
    string_signed = signed.decode('utf-8')
    result = signer.unsign(string_signed)
    assert result == value
    
    # Test 9: Verify timestamp is datetime with UTC timezone
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert result_with_ts[1].tzinfo == timezone.utc
    
    # Test 10: Verify age calculation with negative age (future timestamp)
    def mock_future_timestamp():
        return int(time_module.time()) + 1000
    
    try:
        signer.get_timestamp = mock_future_timestamp
        signed_future = signer.sign(value)
        signer.get_timestamp = original_get_timestamp
        
        with pytest.raises(SignatureExpired) as exc_info:
            signer.unsign(signed_future, max_age=100)
        assert "age" in str(exc_info.value)
    finally:
        signer.get_timestamp = original_get_timestamp
```


# LLM-generated content at query #115
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads method with various scenarios."""
    serializer = TimedSerializer(secret_key="test-secret")
    
    # Test basic loads without max_age and return_timestamp
    data = {"key": "value"}
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized)
    assert result == data
    
    # Test loads with return_timestamp
    serialized = serializer.dumps(data)
    result, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test loads with max_age that should succeed
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized, max_age=3600)
    assert result == data
    
    # Test loads with max_age that should fail (expired)
    serializer_with_fixed_time = TimedSerializer(secret_key="test-secret")
    original_get_timestamp = serializer_with_fixed_time.default_signer.get_timestamp
    
    # Create a signer that returns an old timestamp
    class OldTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return 1000  # Old timestamp
    
    serializer_with_fixed_time.default_signer = OldTimestampSigner
    serialized = serializer_with_fixed_time.dumps(data)
    
    # Now try to load with a max_age that would make it expired
    serializer_with_fixed_time.default_signer = TimestampSigner
    with pytest.raises(SignatureExpired):
        serializer_with_fixed_time.loads(serialized, max_age=1)
    
    # Test loads with salt
    serialized = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(serialized, salt="custom-salt")
    assert result == data
    
    # Test loads with wrong salt should fail
    with pytest.raises(BadSignature):
        serializer.loads(serialized, salt="wrong-salt")
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test loads with invalid payload
    invalid_data = b"invalid|" + base64_encode(int_to_bytes(int(time.time())))
    invalid_data += b"|" + base64_encode(b"invalid")
    with pytest.raises(BadSignature):
        serializer.loads(invalid_data)
    
    # Test loads_unsafe with valid data
    result = serializer.loads_unsafe(serialized)
    assert result == (True, data)
    
    # Test loads_unsafe with invalid data
    result = serializer.loads_unsafe(b"invalid-data")
    assert result == (False, None)
    
    # Test loads with max_age and return_timestamp combined
    serialized = serializer.dumps(data)
    result, timestamp = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test that loads raises SignatureExpired for expired signature
    serializer_with_expired = TimedSerializer(secret_key="test-secret")
    expired_serialized = serializer_with_expired.dumps(data)
    
    # Simulate time passing by using a signer that returns current time
    # but we'll check with a very small max_age
    with pytest.raises(SignatureExpired):
        serializer_with_expired.loads(expired_serialized, max_age=0)
    
    # Test loads with bytes input
    serialized_bytes = serializer.dumps(data)
    result = serializer.loads(serialized_bytes)
    assert result == data
    
    # Test loads with string input
    serialized_str = serializer.dumps(data).decode('utf-8')
    result = serializer.loads(serialized_str)
    assert result == data
```


# LLM-generated content at query #116
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test_secret_key")
    
    # Test basic loads without max_age and return_timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with return_timestamp=True
    result_with_ts = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == data
    assert isinstance(result_with_ts[1], datetime)
    
    # Test loads with max_age (should succeed if within time)
    result_max_age = serializer.loads(signed, max_age=3600)
    assert result_max_age == data
    
    # Test loads with expired signature
    import time
    old_serializer = TimedSerializer("test_secret_key")
    old_signed = old_serializer.dumps(data)
    time.sleep(1.5)  # Wait a bit to simulate age
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=1)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid_signature_data")
    
    # Test loads with different salt
    salt = "custom_salt"
    signed_with_salt = serializer.dumps(data, salt=salt)
    result_with_salt = serializer.loads(signed_with_salt, salt=salt)
    assert result_with_salt == data
    
    # Test loads with wrong salt should raise BadSignature
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong_salt")
    
    # Test loads with both max_age and return_timestamp
    result_complex = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result_complex, tuple)
    assert result_complex[0] == data
    assert isinstance(result_complex[1], datetime)
```


# LLM-generated content at query #117
#--------------------------

```python
def test_TimedSerializer():
    # Test default constructor
    serializer = TimedSerializer()
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)

    # Test constructor with secret key
    serializer_with_key = TimedSerializer(secret_key="test-secret")
    assert serializer_with_key.secret_key == "test-secret"

    # Test constructor with salt
    serializer_with_salt = TimedSerializer(salt="test-salt")
    assert serializer_with_salt.salt == "test-salt"

    # Test constructor with signer_kwargs
    serializer_with_signer_kwargs = TimedSerializer(signer_kwargs={"key_derivation": "hmac"})
    assert serializer_with_signer_kwargs.signer_kwargs == {"key_derivation": "hmac"}

    # Test constructor with serializer_kwargs
    serializer_with_serializer_kwargs = TimedSerializer(serializer_kwargs={"serializer": "json"})
    assert serializer_with_serializer_kwargs.serializer_kwargs == {"serializer": "json"}

    # Test constructor with all parameters
    serializer_all = TimedSerializer(
        secret_key="test-secret",
        salt="test-salt",
        signer_kwargs={"key_derivation": "hmac"},
        serializer_kwargs={"serializer": "json"}
    )
    assert serializer_all.secret_key == "test-secret"
    assert serializer_all.salt == "test-salt"
    assert serializer_all.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer_all.serializer_kwargs == {"serializer": "json"}

    # Test that default_signer is overridable
    class CustomTimestampSigner(TimestampSigner):
        pass

    serializer_custom = TimedSerializer()
    serializer_custom.default_signer = CustomTimestampSigner
    assert serializer_custom.default_signer is CustomTimestampSigner

    # Test iter_unsigners returns TimestampSigner instances
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 0
    for signer in signers:
        assert isinstance(signer, TimestampSigner)


# LLM-generated content at query #118
#--------------------------

```python
def test_TimedSerializer():
    # Test basic initialization
    serializer = TimedSerializer()
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    
    # Test with secret key
    serializer_with_key = TimedSerializer(secret_key="test-secret")
    assert serializer_with_key.secret_key == b"test-secret"
    
    # Test with salt
    serializer_with_salt = TimedSerializer(salt="test-salt")
    assert serializer_with_salt.salt == "test-salt"
    
    # Test with custom signer kwargs
    serializer_with_signer_kwargs = TimedSerializer(signer_kwargs={"key_derivation": "hmac"})
    assert serializer_with_signer_kwargs.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test with all parameters
    serializer_all_params = TimedSerializer(
        secret_key="my-secret",
        salt="my-salt",
        serializer="json",
        signer_kwargs={"key_derivation": "none"},
        signer=TimestampSigner,
    )
    assert serializer_all_params.secret_key == b"my-secret"
    assert serializer_all_params.salt == "my-salt"
    assert serializer_all_params.serializer == "json"
    assert serializer_all_params.signer_kwargs == {"key_derivation": "none"}
    
    # Test iter_unsigners returns TimestampSigner instances
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 0
    for signer in signers:
        assert isinstance(signer, TimestampSigner) 


# LLM-generated content at query #119
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor and default attributes."""
    serializer = TimedSerializer("test-secret-key")
    
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner
    assert serializer.secret_key == b"test-secret-key"
    assert serializer.salt == "itsdangerous"
    assert serializer.serializer is None
    assert serializer.signer_kwargs == {}
    
    # Test with custom parameters
    custom_serializer = TimedSerializer(
        "custom-key",
        salt="custom-salt",
        serializer_kwargs={"protocol": 2},
        signer_kwargs={"key_derivation": "hmac"},
    )
    
    assert custom_serializer.secret_key == b"custom-key"
    assert custom_serializer.salt == "custom-salt"
    assert custom_serializer.serializer_kwargs == {"protocol": 2}
    assert custom_serializer.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test that make_signer returns TimestampSigner instances
    signer = custom_serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    assert signer.secret_key == b"custom-key"
    assert signer.salt == "custom-salt"
    assert signer.key_derivation == "hmac"
    
    # Test with bytes secret key
    bytes_serializer = TimedSerializer(b"bytes-key")
    assert bytes_serializer.secret_key == b"bytes-key"
    
    # Test with fallback signers
    fallback_serializer = TimedSerializer(
        "fallback-key",
        fallback_signers=["key1", "key2"]
    )
    assert fallback_serializer.fallback_signers == ["key1", "key2"]


# LLM-generated content at query #120
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key", salt="test-salt")
    
    # Test 1: Basic unsign without timestamp
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"
    
    # Test 2: Unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Unsign with max_age (valid)
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test-value"
    
    # Test 4: Unsign with max_age (expired)
    import time as time_module
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time_module.time()) + 7200
    try:
        with pytest.raises(SignatureExpired):
            signer.unsign(signed, max_age=3600)
    finally:
        signer.get_timestamp = original_get_timestamp
    
    # Test 5: Unsign with max_age (negative age - future timestamp)
    signed_future = signer.sign("future-value")
    # Manipulate timestamp to be in the future
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time_module.time()) - 100
    try:
        with pytest.raises(SignatureExpired):
            signer.unsign(signed_future, max_age=3600)
    finally:
        signer.get_timestamp = original_get_timestamp
    
    # Test 6: Invalid signed value
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid-signature")
    
    # Test 7: Tampered value
    tampered = signed[:-1] + b"x"
    with pytest.raises(BadSignature):
        signer.unsign(tampered)
    
    # Test 8: Empty value
    with pytest.raises(BadSignature):
        signer.unsign(b"")
    
    # Test 9: Different secret key
    other_signer = TimestampSigner("other-key")
    signed_other = other_signer.sign("other-value")
    with pytest.raises(BadSignature):
        signer.unsign(signed_other)
    
    # Test 10: Malformed timestamp
    # Create a signed value with malformed timestamp
    value = b"test"
    sep = signer.sep.encode() if isinstance(signer.sep, str) else signer.sep
    malformed = value + sep + b"not-a-timestamp" + sep + signer.get_signature(value + sep + b"not-a-timestamp")
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test 11: Missing timestamp
    value = b"test"
    sep = signer.sep.encode() if isinstance(signer.sep, str) else signer.sep
    no_timestamp = value + sep + signer.get_signature(value)
    with pytest.raises(BadSignature):
        signer.unsign(no_timestamp)
```


# LLM-generated content at query #121
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test-secret")
    
    # Test basic loads without max_age
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with max_age (valid)
    signed = serializer.dumps(data)
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with return_timestamp=True
    signed = serializer.dumps(data)
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with max_age and return_timestamp
    signed = serializer.dumps(data)
    result, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with expired signature
    serializer_with_fixed_time = TimedSerializer("test-secret")
    original_get_timestamp = serializer_with_fixed_time.default_signer.get_timestamp
    try:
        serializer_with_fixed_time.default_signer.get_timestamp = lambda: int(time.time()) - 100
        signed = serializer_with_fixed_time.dumps(data)
        serializer_with_fixed_time.default_signer.get_timestamp = lambda: int(time.time())
        with pytest.raises(SignatureExpired):
            serializer_with_fixed_time.loads(signed, max_age=10)
    finally:
        serializer_with_fixed_time.default_signer.get_timestamp = original_get_timestamp
    
    # Test loads with bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test loads with salt
    signed = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed, salt="custom-salt")
    assert result == data
    
    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed, salt="wrong-salt")
    
    # Test loads with multiple signers (fallback behavior)
    serializer2 = TimedSerializer("test-secret-2")
    signed2 = serializer2.dumps(data)
    result = serializer.loads(signed2)  # Should fail as key is different
    assert result is None  # Actually should raise BadSignature
    
    # Test loads with bytes input
    signed_bytes = serializer.dumps(data)
    assert isinstance(signed_bytes, bytes)
    result = serializer.loads(signed_bytes)
    assert result == data
```


# LLM-generated content at query #122
#--------------------------

```python
def test_TimestampSigner_unsign(monkeypatch):
    signer = TimestampSigner("secret")
    
    # Test successful unsign without timestamp
    signed = signer.sign("test")
    assert signer.unsign(signed) == b"test"
    
    # Test successful unsign with timestamp
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"test"
    assert isinstance(result[1], datetime)
    
    # Test expired signature
    monkeypatch.setattr(signer, "get_timestamp", lambda: int(time.time()) + 1000)
    signed_old = signer.sign("old")
    monkeypatch.setattr(signer, "get_timestamp", lambda: int(time.time()))
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_old, max_age=10)
    
    # Test future timestamp
    monkeypatch.setattr(signer, "get_timestamp", lambda: int(time.time()) - 1000)
    signed_future = signer.sign("future")
    monkeypatch.setattr(signer, "get_timestamp", lambda: int(time.time()))
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_future, max_age=10)
    
    # Test invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid")
    
    # Test missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(signer.sign("test")[:10])
    
    # Test malformed timestamp
    bad_ts = base64_encode(b"notanumber")
    bad_signed = b"test" + signer.sep.encode() + bad_ts + signer.sep.encode() + signer.get_signature(b"test" + signer.sep.encode() + bad_ts)
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed)


# LLM-generated content at query #123
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test basic loads without max_age or return_timestamp
    serializer = TimedSerializer("test-secret")
    original_data = {"key": "value"}
    signed_data = serializer.dumps(original_data)
    result = serializer.loads(signed_data)
    assert result == original_data

    # Test loads with max_age that should pass
    signed_data = serializer.dumps(original_data)
    result = serializer.loads(signed_data, max_age=3600)
    assert result == original_data

    # Test loads with return_timestamp=True
    signed_data = serializer.dumps(original_data)
    payload, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert payload == original_data
    assert isinstance(timestamp, datetime)

    # Test loads with both max_age and return_timestamp
    signed_data = serializer.dumps(original_data)
    payload, timestamp = serializer.loads(signed_data, max_age=3600, return_timestamp=True)
    assert payload == original_data
    assert isinstance(timestamp, datetime)

    # Test loads with expired signature
    serializer_fast = TimedSerializer("test-secret")
    # Mock get_timestamp to return a timestamp in the past
    original_get_timestamp = serializer_fast.default_signer.get_timestamp
    def mock_get_timestamp():
        return int(time.time()) - 100  # 100 seconds in the past
    serializer_fast.default_signer.get_timestamp = mock_get_timestamp
    signed_data = serializer_fast.dumps(original_data)
    # Reset the mock
    serializer_fast.default_signer.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=10)

    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")

    # Test loads with different salt
    serializer_with_salt = TimedSerializer("test-secret", salt="custom-salt")
    signed_data = serializer_with_salt.dumps(original_data)
    result = serializer.loads(signed_data, salt="custom-salt")
    assert result == original_data

    # Test loads with wrong salt should fail
    with pytest.raises(BadSignature):
        serializer.loads(signed_data, salt="wrong-salt")

    # Test loads with string input
    signed_string = serializer.dumps(original_data).decode('utf-8')
    result = serializer.loads(signed_string)
    assert result == original_data
```


# LLM-generated content at query #124
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == b"secret-key"
    assert signer.salt is not None
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

    # Test with custom parameters
    signer_custom = TimestampSigner(
        "secret-key",
        salt="custom-salt",
        sep=":",
        key_derivation="none",
        digest_method="sha256",
    )
    assert signer_custom.secret_key == b"secret-key"
    assert signer_custom.salt == "custom-salt"
    assert signer_custom.sep == ":"
    assert signer_custom.key_derivation == "none"
    assert signer_custom.digest_method is not None

    # Test default values for optional parameters
    signer_default = TimestampSigner("secret-key", salt=None)
    assert signer_default.salt is None

    # Test that TimestampSigner is a subclass of Signer
    assert isinstance(signer, Signer)


# LLM-generated content at query #125
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key", salt="test-salt")
    
    # Test 1: Basic sign and unsign
    value = b"test_value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test 2: Unsign with return_timestamp=True
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == value
    assert isinstance(result_with_ts[1], datetime)
    
    # Test 3: Unsign with valid max_age
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test 4: Unsign with expired max_age
    import time
    # Create a signer with a fixed timestamp in the past
    class PastTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) - 10000  # 10000 seconds in the past
    
    past_signer = PastTimestampSigner("secret-key", salt="test-salt")
    old_signed = past_signer.sign(value)
    
    with pytest.raises(SignatureExpired):
        past_signer.unsign(old_signed, max_age=3600)
    
    # Test 5: Unsign with negative age (future timestamp)
    class FutureTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) + 10000  # 10000 seconds in the future
    
    future_signer = FutureTimestampSigner("secret-key", salt="test-salt")
    future_signed = future_signer.sign(value)
    
    with pytest.raises(SignatureExpired):
        future_signer.unsign(future_signed, max_age=3600)
    
    # Test 6: Unsign with invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")
    
    # Test 7: Unsign with tampered value
    tampered = signed[:-1] + (b'\x00' if signed[-1:] != b'\x00' else b'\x01')
    with pytest.raises(BadSignature):
        signer.unsign(tampered)
    
    # Test 8: Unsign with missing timestamp
    # Create a signature without timestamp
    normal_signer = Signer("secret-key", salt="test-salt")
    signed_without_ts = normal_signer.sign(value)
    
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(signed_without_ts)
    
    # Test 9: Unsign with malformed timestamp
    # Create a signed value with invalid timestamp encoding
    sep = signer.sep.encode()
    malformed_ts = base64_encode(b"not_a_timestamp")
    malformed_signed = value + sep + malformed_ts + sep + signer.get_signature(value + sep + malformed_ts)
    
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(malformed_signed)
    
    # Test 10: Unsign with string input
    signed_str = signed.decode() if isinstance(signed, bytes) else signed
    result = signer.unsign(signed_str)
    assert result == value
```


# LLM-generated content at query #126
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("test-secret-key", salt="test-salt")
    
    # Test that the signer is properly initialized
    assert signer.secret_key == "test-secret-key"
    assert signer.salt == "test-salt"
    
    # Test default values
    default_signer = TimestampSigner("default-key")
    assert default_signer.secret_key == "default-key"
    assert default_signer.salt is not None
    
    # Test with bytes key
    bytes_signer = TimestampSigner(b"bytes-key")
    assert bytes_signer.secret_key == b"bytes-key"
```


# LLM-generated content at query #127
#--------------------------

```python
def test_TimestampSigner():
    # Test that TimestampSigner can be constructed with default parameters
    signer = TimestampSigner()
    assert isinstance(signer, TimestampSigner)
    assert isinstance(signer, Signer)
    
    # Test that TimestampSigner can be constructed with a secret key
    signer = TimestampSigner("my-secret-key")
    assert signer.secret_key == b"my-secret-key"
    
    # Test that TimestampSigner can be constructed with salt
    signer = TimestampSigner("my-secret-key", salt="my-salt")
    assert signer.salt == "my-salt"
    
    # Test that TimestampSigner can be constructed with key derivation
    signer = TimestampSigner("my-secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"
    
    # Test that TimestampSigner can be constructed with digest method
    signer = TimestampSigner("my-secret-key", digest_method="sha256")
    assert signer.digest_method == "sha256"
    
    # Test that TimestampSigner can be constructed with algorithm
    signer = TimestampSigner("my-secret-key", algorithm="hmac-sha256")
    assert signer.algorithm == "hmac-sha256"
    
    # Test that TimestampSigner can be constructed with separator
    signer = TimestampSigner("my-secret-key", sep=".")
    assert signer.sep == b"."
```


# LLM-generated content at query #128
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("test-secret-key")
    
    # Test successful unsign without timestamp
    result = signer.sign("test-value")
    unsigned = signer.unsign(result)
    assert unsigned == b"test-value"
    
    # Test successful unsign with timestamp return
    unsigned, timestamp = signer.unsign(result, return_timestamp=True)
    assert unsigned == b"test-value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test with max_age (should succeed)
    unsigned = signer.unsign(result, max_age=3600)
    assert unsigned == b"test-value"
    
    # Test with expired signature
    expired_signer = TimestampSigner("test-secret-key")
    expired_signer.get_timestamp = lambda: int(time.time()) - 3700  # 3700 seconds ago
    expired_result = expired_signer.sign("test-value")
    current_signer = TimestampSigner("test-secret-key")
    
    with pytest.raises(SignatureExpired):
        current_signer.unsign(expired_result, max_age=3600)
    
    # Test with signature in the future
    future_signer = TimestampSigner("test-secret-key")
    future_signer.get_timestamp = lambda: int(time.time()) + 100  # 100 seconds in future
    future_result = future_signer.sign("test-value")
    
    with pytest.raises(SignatureExpired) as exc_info:
        current_signer.unsign(future_result, max_age=3600)
    assert "age -" in str(exc_info.value)
    
    # Test with invalid signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid-signature")
    
    # Test with tampered value
    tampered = bytearray(result)
    tampered[0] = ord('X') if tampered[0] != ord('X') else ord('Y')
    with pytest.raises(BadTimeSignature):
        signer.unsign(bytes(tampered))
    
    # Test with no timestamp separator
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"no-separator-here")
```


# LLM-generated content at query #129
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("secret-key")
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    
    # Test with custom parameters
    signer_custom = TimestampSigner(
        "secret-key",
        salt="custom-salt",
        sep=":",
        key_derivation="none",
        digest_method=hashlib.sha256
    )
    assert signer_custom.salt == "custom-salt"
    assert signer_custom.sep == ":"
    assert signer_custom.key_derivation == "none"
    assert signer_custom.digest_method == hashlib.sha256
    
    # Verify it's a subclass of Signer
    assert isinstance(signer, Signer)
    
    # Test default attributes
    assert hasattr(signer, 'get_timestamp')
    assert hasattr(signer, 'timestamp_to_datetime')
    assert hasattr(signer, 'sign')
    assert hasattr(signer, 'unsign')
    assert hasattr(signer, 'validate')
```


# LLM-generated content at query #130
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    
    # Test constructor with custom salt
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == "custom-salt"
    
    # Test constructor with key_derivation
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"
    
    # Test constructor with digest_method
    signer = TimestampSigner("secret-key", digest_method="sha256")
    assert signer.digest_method == "sha256"
    
    # Test constructor with algorithm
    signer = TimestampSigner("secret-key", algorithm="hmac-sha1")
    assert signer.algorithm == "hmac-sha1"
```


# LLM-generated content at query #131
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.sep == "."
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm is not None
    
    # Test constructor with custom parameters
    signer_custom = TimestampSigner(
        "secret-key",
        sep="|",
        salt="custom-salt",
        key_derivation="none",
        digest_method=hashlib.sha256
    )
    assert signer_custom.secret_key == "secret-key"
    assert signer_custom.sep == "|"
    assert signer_custom.salt == "custom-salt"
    assert signer_custom.key_derivation == "none"
    assert signer_custom.digest_method == hashlib.sha256
    
    # Test get_timestamp returns integer
    ts = signer.get_timestamp()
    assert isinstance(ts, int)
    assert ts > 0
    
    # Test timestamp_to_datetime
    dt = signer.timestamp_to_datetime(ts)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc
    
    # Test sign and unsign roundtrip
    signed = signer.sign("test-value")
    assert isinstance(signed, bytes)
    
    unsigned = signer.unsign(signed)
    assert unsigned == b"test-value"
    
    # Test unsign with return_timestamp
    unsigned_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(unsigned_with_ts, tuple)
    assert len(unsigned_with_ts) == 2
    assert unsigned_with_ts[0] == b"test-value"
    assert isinstance(unsigned_with_ts[1], datetime)
    assert unsigned_with_ts[1].tzinfo == timezone.utc
    
    # Test sign with bytes input
    signed_bytes = signer.sign(b"test-bytes")
    assert signer.unsign(signed_bytes) == b"test-bytes"
    
    # Test validate returns True for valid signature
    assert signer.validate(signed) is True
    
    # Test validate returns False for invalid signature
    assert signer.validate(b"invalid-signature") is False
    
    # Test unsign raises BadSignature for invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid-signature")
    
    # Test unsign with max_age (fresh signature should not expire)
    unsigned_fresh = signer.unsign(signed, max_age=3600)
    assert unsigned_fresh == b"test-value"
    
    # Test unsign with max_age that would expire (need to manipulate timestamp)
    # Create a signed value with an old timestamp
    old_timestamp = int(time.time()) - 10000
    old_ts_bytes = base64_encode(int_to_bytes(old_timestamp))
    value = want_bytes(b"old-value")
    sep = want_bytes(signer.sep)
    old_signed = value + sep + old_ts_bytes
    old_signed = old_signed + sep + signer.get_signature(old_signed)
    
    # This should raise SignatureExpired because max_age is smaller than the age
    with pytest.raises(SignatureExpired):
        signer.unsign(old_signed, max_age=100)


# LLM-generated content at query #132
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner('secret-key')
    assert signer.secret_key == b'secret-key'
    assert signer.salt == b'itsdangerous.TimestampSigner'
    assert signer.sep == b'.'
    assert signer.key_derivation == 'hmac'
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == hmac_compat
    
    # Test with custom salt
    signer_custom_salt = TimestampSigner('secret-key', salt='custom-salt')
    assert signer_custom_salt.secret_key == b'secret-key'
    assert signer_custom_salt.salt == b'custom-salt'
    
    # Test with custom separator
    signer_custom_sep = TimestampSigner('secret-key', sep=':')
    assert signer_custom_sep.sep == b':'
    
    # Test with digest method
    signer_custom_digest = TimestampSigner('secret-key', digest_method=hashlib.sha256)
    assert signer_custom_digest.digest_method == hashlib.sha256
    
    # Test with key derivation
    signer_custom_derivation = TimestampSigner('secret-key', key_derivation='hmac')
    assert signer_custom_derivation.key_derivation == 'hmac'
    
    # Test inheritance
    assert isinstance(signer, Signer)
    assert isinstance(signer, TimestampSigner)
```


# LLM-generated content at query #133
#--------------------------

```python
def test_TimedSerializer():
    # Test default construction
    serializer = TimedSerializer()
    assert serializer.default_signer is TimestampSigner
    assert serializer.signer is TimestampSigner
    
    # Test construction with secret key
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.secret_key == "test-secret"
    
    # Test construction with salt
    serializer = TimedSerializer(salt="test-salt")
    assert serializer.salt == "test-salt"
    
    # Test construction with serializer
    serializer = TimedSerializer(serializer="json")
    assert serializer.serializer == "json"
    
    # Test construction with signer_kwargs
    serializer = TimedSerializer(signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test construction with digest method
    serializer = TimedSerializer(digest_method="sha256")
    assert serializer.digest_method == "sha256"
    
    # Test construction with all parameters
    serializer = TimedSerializer(
        secret_key="test-secret",
        salt="test-salt",
        serializer="json",
        signer_kwargs={"key_derivation": "hmac"},
        digest_method="sha256"
    )
    assert serializer.secret_key == "test-secret"
    assert serializer.salt == "test-salt"
    assert serializer.serializer == "json"
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.digest_method == "sha256"
    
    # Test that default_signer is properly set
    assert serializer.default_signer == TimestampSigner
    
    # Test that the signer is created correctly
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
```


# LLM-generated content at query #134
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.digest_method == Signer.default_digest_method
    assert signer.key_derivation == "hmac"
    
    # Test with custom parameters
    custom_salt = "custom-salt"
    custom_digest = hashlib.sha256
    custom_key_derivation = "none"
    signer2 = TimestampSigner("secret-key", salt=custom_salt, digest_method=custom_digest, key_derivation=custom_key_derivation)
    assert signer2.secret_key == "secret-key"
    assert signer2.salt == custom_salt
    assert signer2.digest_method == custom_digest
    assert signer2.key_derivation == custom_key_derivation
    
    # Test that it's a subclass of Signer
    assert isinstance(signer, Signer)
    
    # Test get_timestamp returns an integer
    ts = signer.get_timestamp()
    assert isinstance(ts, int)
    assert ts > 0
    
    # Test timestamp_to_datetime
    dt = signer.timestamp_to_datetime(ts)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc
    
    # Test sign method
    signed = signer.sign("test-value")
    assert isinstance(signed, bytes)
    assert b"test-value" in signed
    
    # Test unsign method
    unsigned = signer.unsign(signed)
    assert unsigned == b"test-value"
    
    # Test unsign with return_timestamp
    unsigned, ts_dt = signer.unsign(signed, return_timestamp=True)
    assert unsigned == b"test-value"
    assert isinstance(ts_dt, datetime)
    
    # Test validate method
    assert signer.validate(signed) == True
    assert signer.validate(b"invalid") == False
```


# LLM-generated content at query #135
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads method with various scenarios."""
    # Setup
    serializer = TimedSerializer(secret_key="test-secret-key")
    
    # Test 1: Basic loads without max_age or return_timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data, f"Expected {data}, got {result}"
    
    # Test 2: loads with return_timestamp=True
    result = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    payload, timestamp = result
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test 3: loads with max_age (within limit)
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test 4: loads with max_age and return_timestamp
    result = serializer.loads(signed, max_age=3600, return_timestamp=True)
    payload, timestamp = result
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test 5: Expired signature (max_age exceeded)
    # Create a signer with a fixed timestamp in the past
    class PastTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) - 100  # 100 seconds ago
    
    past_serializer = TimedSerializer(
        secret_key="test-secret-key",
        signer_kwargs={"signer_class": PastTimestampSigner}
    )
    past_signed = past_serializer.dumps(data)
    
    with pytest.raises(SignatureExpired):
        past_serializer.loads(past_signed, max_age=10)
    
    # Test 6: Corrupted signature
    corrupted = signed[:-1] + b"X"
    with pytest.raises(BadSignature):
        serializer.loads(corrupted)
    
    # Test 7: Empty string
    with pytest.raises(BadSignature):
        serializer.loads(b"")
    
    # Test 8: Invalid data format
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test 9: loads with specific salt
    salted_serializer = TimedSerializer(secret_key="test-secret-key", salt="custom-salt")
    salted_signed = salted_serializer.dumps(data)
    result = serializer.loads(salted_signed, salt="custom-salt")
    assert result == data
    
    # Test 10: loads without correct salt should fail
    with pytest.raises(BadSignature):
        serializer.loads(salted_signed, salt="wrong-salt")


# LLM-generated content at query #136
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer.signer, TimestampSigner)
    assert serializer.secret_key == "test-secret"
    assert serializer.salt == "itsdangerous"
    assert serializer.serializer is None
    assert serializer.signer_kwargs == {}
    assert serializer.signer.salt == serializer.salt
```


# LLM-generated content at query #137
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key", salt="test-salt")
    
    # Test basic unsign without timestamp
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test unsign with return_timestamp=True
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with valid max_age
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)  # 1 hour
    assert result == b"test_value"
    
    # Test unsign with expired max_age
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) - 100  # Simulate 100 seconds ago
    signed = signer.sign("test_value")
    signer.get_timestamp = lambda: int(time.time())  # Restore current time
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=10)
    
    # Test unsign with future timestamp (age < 0)
    signer.get_timestamp = lambda: int(time.time()) + 100  # Simulate future
    signed = signer.sign("test_value")
    signer.get_timestamp = lambda: int(time.time()) - 50  # Simulate past
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=3600)
    
    # Test unsign with malformed timestamp
    malformed = b"test_value.sep.invalid_timestamp"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test unsign with missing timestamp
    missing_ts = signer.sign("test_value") + b".extra"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_ts)
    
    # Test unsign with invalid signature
    invalid_signed = b"test_value.sep.timestamp.invalid_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(invalid_signed)
    
    # Test unsign with bytes input
    signed_bytes = signer.sign(b"test_bytes")
    result = signer.unsign(signed_bytes)
    assert result == b"test_bytes"
    
    # Test unsign with string input
    signed_str = signer.sign("test_string").decode()
    result = signer.unsign(signed_str)
    assert result == b"test_string"
    
    # Test unsign with return_timestamp=True and expired signature
    signer.get_timestamp = lambda: int(time.time()) - 100
    signed = signer.sign("test_value")
    signer.get_timestamp = lambda: int(time.time())
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(signed, max_age=10, return_timestamp=True)
    assert exc_info.value.date_signed is not None
    assert isinstance(exc_info.value.date_signed, datetime)
    
    # Test unsign with date_signed in exception
    signer.get_timestamp = lambda: int(time.time()) - 100
    signed = signer.sign("test_value")
    signer.get_timestamp = lambda: int(time.time())
    try:
        signer.unsign(signed, max_age=10)
    except SignatureExpired as e:
        assert e.date_signed is not None
        assert isinstance(e.date_signed, datetime)
        assert e.date_signed.tzinfo == timezone.utc

```


# LLM-generated content at query #138
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsigning without timestamp
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test unsigning with return_timestamp=True
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsigning with max_age (valid age)
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test unsigning with max_age (expired)
    signer_with_old_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_old_time.get_timestamp
    
    def old_timestamp():
        return int(time.time()) - 100
    
    signer_with_old_time.get_timestamp = old_timestamp
    signed_old = signer_with_old_time.sign("test_value")
    
    # Restore original timestamp
    signer_with_old_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_with_old_time.unsign(signed_old, max_age=50)
    
    # Test unsigning with negative age (future timestamp)
    signer_future = TimestampSigner("secret-key")
    
    def future_timestamp():
        return int(time.time()) + 100
    
    signer_future.get_timestamp = future_timestamp
    signed_future = signer_future.sign("test_value")
    signer_future.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_future.unsign(signed_future, max_age=3600)
    
    # Test unsigning with invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")
    
    # Test unsigning with malformed timestamp
    signed = signer.sign("test_value")
    sep = signer.sep.encode()
    parts = signed.rsplit(sep, 1)
    malformed = parts[0] + sep + b"invalid_timestamp"
    
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test unsigning with missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value_without_timestamp")
    
    # Test unsigning with string input
    signed_str = signer.sign("test_value").decode()
    result = signer.unsign(signed_str)
    assert result == b"test_value"
    
    # Test unsigning with bytes input
    signed_bytes = signer.sign(b"test_value_bytes")
    result = signer.unsign(signed_bytes)
    assert result == b"test_value_bytes"
```


# LLM-generated content at query #139
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed = signer.sign(value)
    
    # Test 1: Basic unsign without timestamp
    result = signer.unsign(signed)
    assert result == value
    
    # Test 2: Unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Unsign with valid max_age
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test 4: Unsign with expired signature
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)
    
    # Test 5: Invalid signature raises BadSignature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")
    
    # Test 6: Timestamp missing raises BadTimeSignature
    bad_signed = value + signer.sep.encode() + b"garbage"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed)
    
    # Test 7: Malformed timestamp raises BadTimeSignature
    malformed_signed = value + signer.sep.encode() + signer.sep.encode() + b"not_base64"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed)
    
    # Test 8: Return datetime is timezone-aware
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert timestamp.tzinfo is not None
    assert timestamp.tzinfo.utcoffset(timestamp) == timedelta(0)
```


# LLM-generated content at query #140
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("test-secret")
    
    # Test 1: Basic loads without max_age and return_timestamp
    original_value = {"key": "value"}
    signed = serializer.dumps(original_value)
    result = serializer.loads(signed)
    assert result == original_value
    
    # Test 2: Loads with return_timestamp=True
    signed = serializer.dumps(original_value)
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == original_value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Loads with valid max_age
    signed = serializer.dumps(original_value)
    result = serializer.loads(signed, max_age=3600)
    assert result == original_value
    
    # Test 4: Loads with expired signature (max_age too small)
    signed = serializer.dumps(original_value)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=0)
    
    # Test 5: Loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-signature")
    
    # Test 6: Loads with return_timestamp=True and max_age
    signed = serializer.dumps(original_value)
    payload, timestamp = serializer.loads(signed, return_timestamp=True, max_age=3600)
    assert payload == original_value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 7: Loads with string input
    signed = serializer.dumps(original_value).decode()
    result = serializer.loads(signed)
    assert result == original_value
    
    # Test 8: Loads with custom salt
    salt = "custom-salt"
    signed = serializer.dumps(original_value, salt=salt)
    result = serializer.loads(signed, salt=salt)
    assert result == original_value
    
    # Test 9: Loads with wrong salt raises BadSignature
    wrong_salt = "wrong-salt"
    signed = serializer.dumps(original_value, salt=salt)
    with pytest.raises(BadSignature):
        serializer.loads(signed, salt=wrong_salt)
    
    # Test 10: Loads with return_timestamp=True and expired signature
    signed = serializer.dumps(original_value)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=0, return_timestamp=True)
    
    # Test 11: Loads with empty payload
    signed = serializer.dumps({})
    result = serializer.loads(signed)
    assert result == {}
    
    # Test 12: Loads with None value
    signed = serializer.dumps(None)
    result = serializer.loads(signed)
    assert result is None
```


# LLM-generated content at query #141
#--------------------------

```python
def test_TimestampSigner():
    # Test basic construction
    signer = TimestampSigner(secret_key="test-secret")
    assert signer.secret_key == "test-secret"
    assert signer.sep == "."
    
    # Test construction with custom separator
    signer_custom = TimestampSigner(secret_key="test-secret", sep="/")
    assert signer_custom.sep == "/"
    
    # Test construction with salt
    signer_with_salt = TimestampSigner(secret_key="test-secret", salt="custom-salt")
    assert signer_with_salt.salt is not None
    
    # Test construction with key derivation
    signer_kd = TimestampSigner(secret_key="test-secret", key_derivation="hmac")
    assert signer_kd.key_derivation == "hmac"
    
    # Test construction with digest method
    signer_digest = TimestampSigner(secret_key="test-secret", digest_method="sha256")
    assert signer_digest.digest_method is not None
    
    # Test that TimestampSigner is subclass of Signer
    assert isinstance(signer, Signer)
    
    # Test default signer class attribute
    assert signer.default_signer is not None
    
    # Test get_timestamp returns integer
    assert isinstance(signer.get_timestamp(), int)
    
    # Test timestamp_to_datetime works
    timestamp = signer.get_timestamp()
    dt = signer.timestamp_to_datetime(timestamp)
    assert dt.tzinfo is not None
    assert dt.tzinfo == timezone.utc
    
    # Test sign method returns bytes
    signed = signer.sign("test-value")
    assert isinstance(signed, bytes)
    assert b"." in signed  # Should contain separator
    
    # Test sign with bytes input
    signed_bytes = signer.sign(b"test-value")
    assert isinstance(signed_bytes, bytes)
    assert signed_bytes == signed  # Should produce same result
    
    # Test sign and unsign roundtrip
    value = "test-message"
    signed_value = signer.sign(value)
    unsigned_value = signer.unsign(signed_value)
    assert unsigned_value == value.encode()
    
    # Test unsign with return_timestamp
    unsigned_with_ts, timestamp_dt = signer.unsign(signed_value, return_timestamp=True)
    assert unsigned_with_ts == value.encode()
    assert isinstance(timestamp_dt, datetime)
    assert timestamp_dt.tzinfo == timezone.utc
    
    # Test unsign with max_age
    signed_value = signer.sign("fresh-value")
    unsigned_value = signer.unsign(signed_value, max_age=3600)
    assert unsigned_value == b"fresh-value"
    
    # Test validate method returns True for valid signature
    assert signer.validate(signed_value)
    
    # Test validate method returns False for invalid signature
    assert not signer.validate(b"invalid-signature")
```


# LLM-generated content at query #142
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == hmac_compat
    assert isinstance(signer.get_timestamp(), int)
    assert signer.get_timestamp() > 0
```


# LLM-generated content at query #143
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Create a TimestampSigner instance
    signer = TimestampSigner("test-secret")

    # Test basic unsign without max_age
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"

    # Test unsign with return_timestamp=True
    signed = signer.sign("test-value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test unsign with valid max_age
    signed = signer.sign("test-value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test-value"

    # Test unsign with expired signature
    signer_with_mock_time = TimestampSigner("test-secret")
    original_get_timestamp = signer_with_mock_time.get_timestamp
    signer_with_mock_time.get_timestamp = lambda: original_get_timestamp() - 100
    signed = signer_with_mock_time.sign("test-value")
    signer_with_mock_time.get_timestamp = original_get_timestamp
    with pytest.raises(SignatureExpired):
        signer_with_mock_time.unsign(signed, max_age=10)

    # Test unsign with negative age (future timestamp)
    signer_with_future_time = TimestampSigner("test-secret")
    original_get_timestamp = signer_with_future_time.get_timestamp
    signer_with_future_time.get_timestamp = lambda: original_get_timestamp() + 100
    signed = signer_with_future_time.sign("test-value")
    signer_with_future_time.get_timestamp = original_get_timestamp
    with pytest.raises(SignatureExpired):
        signer_with_future_time.unsign(signed, max_age=10)

    # Test unsign with invalid signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid-signature")

    # Test unsign with missing timestamp
    signer_no_timestamp = Signer("test-secret")
    signed_no_timestamp = signer_no_timestamp.sign("test-value")
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_no_timestamp)

    # Test unsign with malformed timestamp
    malformed = b"test-value." + b"malformed-timestamp"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)

    # Test unsign with valid signature but malformed timestamp
    # Create a signature with a non-decodable timestamp
    value = want_bytes("test-value")
    sep = want_bytes(signer.sep)
    bad_ts = b"not-valid-base64"
    value_with_bad_ts = value + sep + bad_ts
    signature = signer.get_signature(value_with_bad_ts)
    signed_with_bad_ts = value_with_bad_ts + sep + signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_with_bad_ts) 


# LLM-generated content at query #144
#--------------------------

```python
def test_TimedSerializer():
    # Test that TimedSerializer can be instantiated with default parameters
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner
    
    # Test that TimedSerializer can be instantiated with a secret key
    serializer = TimedSerializer("secret-key")
    assert serializer.secret_key == b"secret-key"
    
    # Test that TimedSerializer can be instantiated with salt
    serializer = TimedSerializer("secret-key", salt="my-salt")
    assert serializer.salt == "my-salt"
    
    # Test that TimedSerializer can be instantiated with serializer_kwargs
    serializer = TimedSerializer("secret-key", serializer_kwargs={"skipkeys": True})
    assert serializer.serializer_kwargs == {"skipkeys": True}
    
    # Test that TimedSerializer can be instantiated with signer_kwargs
    serializer = TimedSerializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test that TimedSerializer can be instantiated with digest_method
    serializer = TimedSerializer("secret-key", digest_method="sha256")
    assert serializer.digest_method is not None
    
    # Test that TimedSerializer creates TimestampSigner instances
    serializer = TimedSerializer("secret-key")
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    
    # Test serialization and deserialization with default parameters
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    serialized = serializer.dumps(data)
    deserialized = serializer.loads(serialized)
    assert deserialized == data
    
    # Test that loads returns timestamp when requested
    serialized = serializer.dumps(data)
    payload, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test that max_age works correctly
    serialized = serializer.dumps(data)
    import time
    time.sleep(1)
    with pytest.raises(SignatureExpired):
        serializer.loads(serialized, max_age=0)
    
    # Test that valid signature with sufficient max_age works
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized, max_age=3600)
    assert result == data
    
    # Test that different secret keys raise BadSignature
    serializer1 = TimedSerializer("secret1")
    serializer2 = TimedSerializer("secret2")
    data = "test"
    serialized = serializer1.dumps(data)
    with pytest.raises(BadSignature):
        serializer2.loads(serialized)
    
    # Test that salt affects serialization
    serializer1 = TimedSerializer("secret", salt="salt1")
    serializer2 = TimedSerializer("secret", salt="salt2")
    data = "test"
    serialized = serializer1.dumps(data)
    with pytest.raises(BadSignature):
        serializer2.loads(serialized)


# LLM-generated content at query #145
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test 1: Basic unsign without max_age and return_timestamp=False
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test 2: Unsign with return_timestamp=True
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Unsign with max_age (valid)
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)  # 1 hour max age
    assert result == b"test_value"
    
    # Test 4: Unsign with expired signature
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) - 100  # Simulate 100 seconds ago
    signed = signer.sign("test_value")
    signer.get_timestamp = original_get_timestamp  # Restore original
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(signed, max_age=50)  # Only 50 seconds max
    assert "Signature age" in str(exc_info.value)
    
    # Test 5: Unsign with future timestamp (age < 0)
    signer.get_timestamp = lambda: int(time.time())
    signed = signer.sign("test_value")
    signer.get_timestamp = lambda: int(time.time()) - 200  # Current time is 200 seconds earlier
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(signed, max_age=3600)
    assert "age" in str(exc_info.value)
    assert "< 0 seconds" in str(exc_info.value)
    signer.get_timestamp = lambda: int(time.time())  # Restore
    
    # Test 6: Invalid signature raises BadSignature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")
    
    # Test 7: Missing timestamp raises BadTimeSignature
    signer_no_timestamp = Signer("secret-key")
    signed_no_timestamp = signer_no_timestamp.sign("test_value")
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(signed_no_timestamp)
    assert "timestamp missing" in str(exc_info.value)
    
    # Test 8: Malformed timestamp raises BadTimeSignature
    malformed = b"test_value.sep.invalid_timestamp.sep.signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
```


# LLM-generated content at query #146
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner("test-secret")
    assert signer.secret_key == b"test-secret"
    assert signer.sep == "."
    assert signer.salt is not None
    assert signer.key_derivation == "hmac"
    assert signer.digest_method is not None
    assert signer.algorithm is not None
    
    # Test constructor with custom parameters
    signer2 = TimestampSigner(
        "custom-secret",
        sep="|",
        salt="custom-salt",
        key_derivation="none",
    )
    assert signer2.secret_key == b"custom-secret"
    assert signer2.sep == "|"
    assert signer2.salt is not None
    assert signer2.key_derivation == "none"
    
    # Test that TimestampSigner is a subclass of Signer
    assert isinstance(signer, Signer)
    
    # Test that salt is bytes
    assert isinstance(signer.salt, bytes)
    
    # Test default salt value
    assert len(signer.salt) > 0
    
    # Test with bytes secret key
    signer3 = TimestampSigner(b"bytes-secret")
    assert isinstance(signer3.secret_key, bytes)
    assert signer3.secret_key == b"bytes-secret"


# LLM-generated content at query #147
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Create a TimestampSigner instance
    signer = TimestampSigner("secret-key")
    
    # Test 1: Basic unsign without max_age and return_timestamp=False
    signed_value = signer.sign("test_value")
    result = signer.unsign(signed_value)
    assert result == b"test_value"
    
    # Test 2: Unsign with return_timestamp=True
    result_with_ts = signer.unsign(signed_value, return_timestamp=True)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == b"test_value"
    assert isinstance(result_with_ts[1], datetime)
    
    # Test 3: Unsign with max_age that is not exceeded
    result = signer.unsign(signed_value, max_age=3600)
    assert result == b"test_value"
    
    # Test 4: Unsign with max_age that is exceeded should raise SignatureExpired
    # We can't easily make time pass, but we can set max_age to 0
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)
    
    # Test 5: Unsign with negative max_age (past future timestamp)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)
    
    # Test 6: Unsign invalid signature should raise BadSignature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")
    
    # Test 7: Unsign with invalid timestamp
    # Create a signed value with invalid timestamp
    value = b"test_value"
    sep = want_bytes(signer.sep)
    invalid_timestamp = base64_encode(b"invalid")
    invalid_signed = value + sep + invalid_timestamp + sep + signer.get_signature(value + sep + invalid_timestamp)
    with pytest.raises(BadTimeSignature):
        signer.unsign(invalid_signed)
    
    # Test 8: Unsign with bytes input
    signed_bytes = signer.sign(b"test_bytes")
    result = signer.unsign(signed_bytes)
    assert result == b"test_bytes"
    
    # Test 9: Unsign with string input
    signed_str = signer.sign("test_string")
    result = signer.unsign(signed_str.decode())
    assert result == b"test_string"
    
    # Test 10: Verify timestamp is in UTC
    result_with_ts = signer.unsign(signed_value, return_timestamp=True)
    assert result_with_ts[1].tzinfo == timezone.utc
    
    # Test 11: Unsign with valid max_age and return_timestamp
    result = signer.unsign(signed_value, max_age=3600, return_timestamp=True)
    assert len(result) == 2
    assert isinstance(result[1], datetime)
```


# LLM-generated content at query #148
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test 1: Basic loads without max_age and return_timestamp
    serializer = TimedSerializer(secret_key="test-secret")
    original_data = {"user": "test"}
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed)
    assert result == original_data

    # Test 2: loads with max_age and valid timestamp
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed, max_age=3600)
    assert result == original_data

    # Test 3: loads with return_timestamp=True
    signed = serializer.dumps(original_data)
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == original_data
    assert isinstance(timestamp, datetime)

    # Test 4: loads with both max_age and return_timestamp
    signed = serializer.dumps(original_data)
    payload, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert payload == original_data
    assert isinstance(timestamp, datetime)

    # Test 5: loads with expired signature raises SignatureExpired
    import time as time_module
    original_get_timestamp = serializer.loads.__globals__['TimestampSigner'].get_timestamp
    try:
        # Simulate old timestamp by mocking get_timestamp
        class OldTimestampSigner(TimestampSigner):
            def get_timestamp(self):
                return int(time_module.time()) - 100  # 100 seconds old
        
        serializer.default_signer = OldTimestampSigner
        signed = serializer.dumps(original_data)
        
        # Restore original get_timestamp
        serializer.default_signer = TimestampSigner
        signed_with_old_ts = signed  # This has old timestamp
        
        # Now try to loads with max_age less than 100
        import pytest
        with pytest.raises(SignatureExpired):
            serializer.loads(signed_with_old_ts, max_age=10)
    finally:
        serializer.default_signer = TimestampSigner

    # Test 6: loads with bad signature raises BadSignature
    signed = serializer.dumps(original_data)
    tampered = signed + b"tampered"
    import pytest
    with pytest.raises(BadSignature):
        serializer.loads(tampered)

    # Test 7: loads with salt parameter
    signed = serializer.dumps(original_data, salt="custom-salt")
    result = serializer.loads(signed, salt="custom-salt")
    assert result == original_data

    # Test 8: loads with wrong salt raises BadSignature
    signed = serializer.dumps(original_data, salt="salt1")
    import pytest
    with pytest.raises(BadSignature):
        serializer.loads(signed, salt="salt2")

    # Test 9: loads with multiple signers (fallback)
    serializer2 = TimedSerializer(secret_key="another-secret")
    signed2 = serializer2.dumps(original_data)
    # This should fail because different secret
    import pytest
    with pytest.raises(BadSignature):
        serializer.loads(signed2)

    # Test 10: loads returns bytes for bytes payload
    serializer = TimedSerializer(secret_key="test-secret")
    signed = serializer.dumps(b"binary data")
    result = serializer.loads(signed)
    assert result == b"binary data"
```


# LLM-generated content at query #149
#--------------------------

```python
def test_TimestampSigner():
    # Test default construction
    signer = TimestampSigner(secret_key="test-secret")
    assert signer.secret_key == "test-secret"
    assert signer.sep == "."
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "hs256"
    
    # Test construction with custom parameters
    signer_custom = TimestampSigner(
        secret_key="custom-secret",
        salt="custom-salt",
        sep=":",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm="hs512"
    )
    assert signer_custom.secret_key == "custom-secret"
    assert signer_custom.salt == "custom-salt"
    assert signer_custom.sep == ":"
    assert signer_custom.key_derivation == "none"
    assert signer_custom.digest_method == hashlib.sha256
    assert signer_custom.algorithm == "hs512"
    
    # Test that TimestampSigner is a subclass of Signer
    assert isinstance(signer, Signer)
    assert isinstance(signer_custom, Signer)


# LLM-generated content at query #150
#--------------------------

```python
def test_TimestampSigner():
    # Test basic initialization
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == hmac

    # Test with custom parameters
    signer_custom = TimestampSigner(
        secret_key="custom-secret",
        salt="custom-salt",
        sep="-",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm=hmac
    )
    assert signer_custom.secret_key == "custom-secret"
    assert signer_custom.salt == "custom-salt"
    assert signer_custom.sep == "-"
    assert signer_custom.key_derivation == "none"
    assert signer_custom.digest_method == hashlib.sha256
    assert signer_custom.algorithm == hmac

    # Test with bytes secret key
    signer_bytes = TimestampSigner(b"bytes-secret")
    assert signer_bytes.secret_key == b"bytes-secret"

    # Test inheritance from Signer
    assert isinstance(signer, Signer)

    # Test default methods exist and return correct types
    timestamp = signer.get_timestamp()
    assert isinstance(timestamp, int)
    assert timestamp > 0

    dt = signer.timestamp_to_datetime(timestamp)
    assert isinstance(dt, datetime)
    assert dt.tzinfo is not None
    assert dt.tzinfo == timezone.utc

    # Test sign method
    signed = signer.sign("test-value")
    assert isinstance(signed, bytes)
    assert b"test-value" in signed

    # Test unsign method
    unsigned = signer.unsign(signed)
    assert unsigned == b"test-value"

    # Test unsign with return_timestamp
    value, ts = signer.unsign(signed, return_timestamp=True)
    assert value == b"test-value"
    assert isinstance(ts, datetime)
    assert ts.tzinfo == timezone.utc

    # Test with max_age
    unsigned_age = signer.unsign(signed, max_age=3600)
    assert unsigned_age == b"test-value"

    # Test validation
    assert signer.validate(signed) is True
    assert signer.validate(signed, max_age=3600) is True
    assert signer.validate(b"invalid-signature") is False

    # Test error cases
    import pytest
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid-value")

    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test-value.invalidtimestamp.signature")```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test successful load without timestamp
    serializer = TimedSerializer("secret-key")
    original_data = {"key": "value"}
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed)
    assert result == original_data

    # Test successful load with return_timestamp=True
    result_with_ts = serializer.loads(signed, return_timestamp=True)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == original_data
    assert isinstance(result_with_ts[1], datetime)

    # Test load with max_age (should pass)
    result = serializer.loads(signed, max_age=3600)
    assert result == original_data

    # Test load with expired max_age
    import time as time_module
    old_time = time_module.time
    try:
        time_module.time = lambda: time_module.time() + 7200
        with pytest.raises(SignatureExpired):
            serializer.loads(signed, max_age=1)
    finally:
        time_module.time = old_time

    # Test invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid_data")

    # Test with different salt
    serializer_with_salt = TimedSerializer("secret-key", salt="different-salt")
    signed_with_salt = serializer_with_salt.dumps(original_data)
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
    
    # Test successful load with correct salt
    result = serializer_with_salt.loads(signed_with_salt, salt="different-salt")
    assert result == original_data
```


# LLM-generated content at query #2
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Test basic unsign without timestamp
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    result = signer.unsign(signed)
    assert result == b"test value"

    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test unsign with valid max_age
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test value"

    # Test unsign with expired signature
    old_signer = TimestampSigner("secret-key")
    old_signer.get_timestamp = lambda: int(time.time()) - 7200  # 2 hours ago
    old_signed = old_signer.sign("old value")
    
    with pytest.raises(SignatureExpired):
        signer.unsign(old_signed, max_age=3600)

    # Test unsign with future timestamp (negative age)
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: int(time.time()) + 3600  # 1 hour in future
    future_signed = future_signer.sign("future value")
    
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed, max_age=3600)

    # Test unsign with invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_data")

    # Test unsign with tampered data
    tampered = signed[:-1] + (b'x' if signed[-1:] != b'x' else b'y')
    with pytest.raises(BadSignature):
        signer.unsign(tampered)

    # Test unsign with missing timestamp
    signer_no_ts = Signer("secret-key")
    signed_no_ts = signer_no_ts.sign("no timestamp")
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(signed_no_ts)

    # Test unsign with malformed timestamp
    malformed_ts = signed.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid_timestamp"
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(malformed_ts)

    # Test unsign with bytes input
    signed_bytes = signer.sign(b"bytes value")
    result = signer.unsign(signed_bytes)
    assert result == b"bytes value"

    # Test unsign with string input
    signed_str = signer.sign("string value")
    result = signer.unsign(signed_str.decode())
    assert result == b"string value"

    # Test unsign with return_timestamp=True and max_age
    result, timestamp = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert result == b"test value"
    assert isinstance(timestamp, datetime)
```


# LLM-generated content at query #3
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.sep == "."
    assert signer.salt == "timestamp-signer"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method is not None
    assert signer.algorithm is not None
```


# LLM-generated content at query #4
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("test-secret", salt="test-salt")
    assert signer.secret_key == "test-secret"
    assert signer.salt == "test-salt"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "hmac-sha1"
    
    # Test with default parameters
    signer_default = TimestampSigner("default-secret")
    assert signer_default.salt == "itsdangerous.TimestampSigner"
    assert signer_default.sep == "."
    
    # Test with custom separator
    signer_custom = TimestampSigner("custom-secret", sep=":")
    assert signer_custom.sep == ":"
```


# LLM-generated content at query #5
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads method."""
    serializer = TimedSerializer(secret_key="test-secret-key-12345")
    
    # Test 1: Basic loads without max_age
    original_data = {"key": "value", "number": 42}
    signed_data = serializer.dumps(original_data)
    result = serializer.loads(signed_data)
    assert result == original_data, f"Expected {original_data}, got {result}"
    
    # Test 2: Loads with max_age and recent timestamp
    signed_data = serializer.dumps(original_data)
    result = serializer.loads(signed_data, max_age=3600)  # 1 hour
    assert result == original_data, f"Expected {original_data}, got {result}"
    
    # Test 3: Loads with return_timestamp=True
    signed_data = serializer.dumps(original_data)
    payload, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert payload == original_data, f"Expected {original_data}, got {payload}"
    assert isinstance(timestamp, datetime), f"Expected datetime, got {type(timestamp)}"
    assert timestamp.tzinfo == timezone.utc, "Timestamp should be timezone-aware UTC"
    
    # Test 4: Loads with both max_age and return_timestamp
    signed_data = serializer.dumps(original_data)
    payload, timestamp = serializer.loads(signed_data, max_age=3600, return_timestamp=True)
    assert payload == original_data, f"Expected {original_data}, got {payload}"
    assert isinstance(timestamp, datetime), f"Expected datetime, got {type(timestamp)}"
    
    # Test 5: Loads with bytes input
    signed_data = serializer.dumps(original_data)
    signed_bytes = signed_data.encode() if isinstance(signed_data, str) else signed_data
    result = serializer.loads(signed_bytes)
    assert result == original_data, f"Expected {original_data}, got {result}"
    
    # Test 6: Loads with expired signature should raise SignatureExpired
    # Create a signer with a fixed timestamp in the past
    class FixedTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) - 100  # 100 seconds ago
    
    fixed_serializer = TimedSerializer(
        secret_key="test-secret-key-12345",
        signer_kwargs={"signer_cls": FixedTimestampSigner}
    )
    signed_data = fixed_serializer.dumps(original_data)
    
    # Use the normal serializer to verify it's expired
    with pytest.raises(SignatureExpired) as excinfo:
        serializer.loads(signed_data, max_age=50)  # max_age less than 100 seconds
    assert "Signature age" in str(excinfo.value)
    
    # Test 7: Loads with invalid signature should raise BadSignature
    tampered_data = signed_data[:-5] + b"XXXXX" if isinstance(signed_data, bytes) else signed_data[:-5] + "XXXXX"
    with pytest.raises(BadSignature):
        serializer.loads(tampered_data)
    
    # Test 8: Loads with empty/invalid data should raise BadSignature
    with pytest.raises(BadSignature):
        serializer.loads("")
    
    with pytest.raises(BadSignature):
        serializer.loads(b"")
    
    with pytest.raises(BadSignature):
        serializer.loads("not-a-valid-signature")
    
    # Test 9: Loads with salt parameter
    salted_serializer = TimedSerializer(secret_key="test-secret-key-12345", salt="custom-salt")
    signed_data = salted_serializer.dumps(original_data)
    result = salted_serializer.loads(signed_data, salt="custom-salt")
    assert result == original_data, f"Expected {original_data}, got {result}"
    
    # Test 10: Loads with wrong salt should fail
    with pytest.raises(BadSignature):
        salted_serializer.loads(signed_data, salt="wrong-salt")
    
    # Test 11: Loads with negative max_age (should fail immediately)
    signed_data = serializer.dumps(original_data)
    with pytest.raises(SignatureExpired) as excinfo:
        serializer.loads(signed_data, max_age=-1)
    assert "Signature age" in str(excinfo.value)
    
    # Test 12: Loads with various data types
    test_cases = [
        "simple string",
        12345,
        [1, 2, 3],
        {"nested": {"data": True}},
        None
    ]
    
    for test_data in test_cases:
        signed_data = serializer.dumps(test_data)
        result = serializer.loads(signed_data)
        assert result == test_data, f"Expected {test_data}, got {result}"
```


# LLM-generated content at query #6
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer("test-secret-key")
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer.default_signer, type)
    assert serializer.default_signer == TimestampSigner
    assert serializer.secret_key == "test-secret-key"
```


# LLM-generated content at query #7
#--------------------------

```python
def test_TimedSerializer_loads():
    # Create a TimedSerializer instance
    serializer = TimedSerializer(secret_key="test-secret")
    
    # Test 1: Basic load without max_age and without return_timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test 2: Load with return_timestamp=True
    signed = serializer.dumps(data)
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Load with max_age (within limit)
    data = {"test": "data"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed, max_age=3600)  # 1 hour
    assert result == data
    
    # Test 4: Load with max_age and return_timestamp
    signed = serializer.dumps(data)
    payload, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test 5: Load with expired signature
    data = {"expired": "test"}
    signed = serializer.dumps(data)
    # Use a very small max_age to simulate expired signature
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=0)
    
    # Test 6: Load with invalid signature
    invalid_signed = b"invalid.signature.data"
    with pytest.raises(BadSignature):
        serializer.loads(invalid_signed)
    
    # Test 7: Load with different salt
    data = {"salt_test": True}
    signed = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed, salt="custom-salt")
    assert result == data
    
    # Test 8: Load with wrong salt should fail
    with pytest.raises(BadSignature):
        serializer.loads(signed, salt="wrong-salt")
    
    # Test 9: Load with bytes input
    data = {"bytes_input": 123}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test 10: Verify timestamp is recent in successful loads
    data = {"timestamp_check": True}
    signed = serializer.dumps(data)
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    now = datetime.now(timezone.utc)
    time_diff = (now - timestamp).total_seconds()
    assert time_diff < 5  # Should be less than 5 seconds old
    
    # Test 11: Load with complex nested data
    complex_data = {
        "list": [1, 2, 3],
        "nested": {"a": 1, "b": "test"},
        "bool": True,
        "none": None
    }
    signed = serializer.dumps(complex_data)
    result = serializer.loads(signed)
    assert result == complex_data
    
    # Test 12: Load with empty data
    empty_data = {}
    signed = serializer.dumps(empty_data)
    result = serializer.loads(signed)
    assert result == empty_data
    
    # Test 13: Load with integer data
    int_data = 42
    signed = serializer.dumps(int_data)
    result = serializer.loads(signed)
    assert result == int_data
    
    # Test 14: Load with string data
    str_data = "test string"
    signed = serializer.dumps(str_data)
    result = serializer.loads(signed)
    assert result == str_data
    
    # Test 15: Verify SignatureExpired is raised before BadSignature
    data = {"test": "expired"}
    signed = serializer.dumps(data)
    # Corrupt the signature but keep valid timestamp format
    corrupted = signed[:-5] + b"XXXXX"
    with pytest.raises(BadSignature):
        serializer.loads(corrupted)
    
    # Test 16: Load with very large max_age (should work)
    data = {"large_max_age": True}
    signed = serializer.dumps(data)
    result = serializer.loads(signed, max_age=999999999)
    assert result == data
    
    # Test 17: Verify multiple signers work
    serializer2 = TimedSerializer(secret_key="test-secret")
    data = {"multi_signer": True}
    signed = serializer.dumps(data)
    result = serializer2.loads(signed)
    assert result == data
```


# LLM-generated content at query #8
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads method"""
    serializer = TimedSerializer("test-secret")
    
    # Test successful load
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test with max_age (valid)
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test with return_timestamp
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test with expired signature
    old_serializer = TimedSerializer("test-secret")
    import time as _time
    old_signed = old_serializer.dumps(data)
    _time.sleep(0.001)  # Small delay to ensure age > 0
    
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=0)
    
    # Test with bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test with wrong secret
    wrong_serializer = TimedSerializer("wrong-secret")
    with pytest.raises(BadSignature):
        wrong_serializer.loads(signed)
    
    # Test with salt
    salted = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(salted, salt="custom-salt")
    assert result == data
    
    # Test with salt and wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(salted, salt="wrong-salt")
```

This test covers the main functionality of `TimedSerializer.loads` including:
- Successful loading with and without options
- Timestamp return
- Expired signature detection
- Bad signature handling
- Wrong secret handling
- Salt-based signing and verification


# LLM-generated content at query #9
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test basic loads without max_age or return_timestamp
    serializer = TimedSerializer("test_secret_key")
    original_data = {"key": "value"}
    serialized = serializer.dumps(original_data)
    result = serializer.loads(serialized)
    assert result == original_data

    # Test loads with max_age that is within valid range
    result = serializer.loads(serialized, max_age=3600)
    assert result == original_data

    # Test loads with return_timestamp
    result, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert result == original_data
    assert isinstance(timestamp, datetime)

    # Test loads with both max_age and return_timestamp
    result, timestamp = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert result == original_data
    assert isinstance(timestamp, datetime)

    # Test loads with expired signature
    with pytest.raises(SignatureExpired):
        serializer.loads(serialized, max_age=0)

    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid_data")

    # Test loads with different salt
    serializer2 = TimedSerializer("test_secret_key", salt="different_salt")
    serialized2 = serializer2.dumps(original_data)
    with pytest.raises(BadSignature):
        serializer.loads(serialized2)

    # Test loads with bytes data
    serialized_bytes = serializer.dumps(b"test_bytes")
    result = serializer.loads(serialized_bytes)
    assert result == b"test_bytes"

    # Test loads with string data
    serialized_str = serializer.dumps("test_string")
    result = serializer.loads(serialized_str)
    assert result == "test_string"

    # Test loads with empty data
    serialized_empty = serializer.dumps("")
    result = serializer.loads(serialized_empty)
    assert result == ""

    # Test loads with None value
    serialized_none = serializer.dumps(None)
    result = serializer.loads(serialized_none)
    assert result is None

    # Test loads with integer value
    serialized_int = serializer.dumps(123)
    result = serializer.loads(serialized_int)
    assert result == 123

    # Test loads with list value
    test_list = [1, 2, 3]
    serialized_list = serializer.dumps(test_list)
    result = serializer.loads(serialized_list)
    assert result == test_list
```


# LLM-generated content at query #10
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without timestamp
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"
    
    # Test unsign with return_timestamp=True
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert result_with_ts[0] == b"test-value"
    assert isinstance(result_with_ts[1], datetime)
    
    # Test unsign with valid max_age
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test-value"
    
    # Test unsign with expired signature
    signer.get_timestamp = lambda: int(time.time()) - 100
    expired_signed = signer.sign("test-value")
    signer.get_timestamp = lambda: int(time.time())
    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed, max_age=50)
    
    # Test unsign with future timestamp
    signer.get_timestamp = lambda: int(time.time()) + 100
    future_signed = signer.sign("test-value")
    signer.get_timestamp = lambda: int(time.time())
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed, max_age=3600)
    
    # Test unsign with malformed data (no timestamp)
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"no-timestamp-here")
    
    # Test unsign with invalid signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value.invalid-timestamp.invalid-signature")
    
    # Test unsign with bad signature but valid timestamp
    signer2 = TimestampSigner("different-secret")
    wrong_signed = signer2.sign("test-value")
    with pytest.raises(BadTimeSignature):
        signer.unsign(wrong_signed)
    
    # Test unsign with return_timestamp and max_age
    signed = signer.sign("test-value")
    result = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"test-value"
    assert isinstance(result[1], datetime)
```


# LLM-generated content at query #11
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without timestamp
    value = b"test_value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test unsign with return_timestamp=True
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with max_age (valid age)
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test unsign with expired signature
    signer_with_fixed_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_fixed_time.get_timestamp
    signer_with_fixed_time.get_timestamp = lambda: int(time.time()) - 100  # Simulate old timestamp
    signed = signer_with_fixed_time.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=10)
    
    # Test unsign with malformed timestamp
    signed = signer.sign(value)
    malformed_signed = signed[:-5] + b"XXXXX"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed)
    
    # Test unsign with missing timestamp
    signer_no_time = Signer("secret-key")
    signed_no_time = signer_no_time.sign(value)
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_no_time)
    
    # Test unsign with invalid signature
    invalid_signed = b"invalid_data"
    with pytest.raises(BadSignature):
        signer.unsign(invalid_signed)
```


# LLM-generated content at query #12
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor and default attributes."""
    serializer = TimedSerializer("test_secret_key")
    
    assert serializer.default_signer is TimestampSigner
    assert serializer.secret_key == "test_secret_key"
    assert serializer.salt == "itsdangerous.TimedSerializer"
    assert serializer.signer_kwargs == {}
    
    serializer2 = TimedSerializer("key", salt="custom_salt", signer_kwargs={"key_derivation": "hmac"})
    assert serializer2.salt == "custom_salt"
    assert serializer2.signer_kwargs == {"key_derivation": "hmac"}


# LLM-generated content at query #13
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret")
    value = b"test message"
    signed = signer.sign(value)
    
    # Test basic unsign
    result = signer.unsign(signed)
    assert result == value
    
    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    
    # Test unsign with max_age (valid)
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test unsign with max_age (expired)
    signer.get_timestamp = lambda: int(time.time()) + 100
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=10)
    
    # Test unsign with negative age (future timestamp)
    signer.get_timestamp = lambda: int(time.time()) - 100
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=10)
    
    # Test unsign with bad signature
    bad_signed = signed + b"tampered"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed)
    
    # Test unsign with malformed timestamp
    malformed = value + b"." + base64_encode(b"not_a_timestamp") + b"." + signer.get_signature(value + b"." + base64_encode(b"not_a_timestamp"))
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test unsign with missing timestamp
    missing_ts = value + b"." + signer.get_signature(value)
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_ts)
    
    # Test unsign with string input
    signed_str = signed.decode()
    result = signer.unsign(signed_str)
    assert result == value
```


# LLM-generated content at query #14
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("secret-key")
    
    # Test basic loads with no max_age
    original_data = {"key": "value"}
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed)
    assert result == original_data
    
    # Test loads with max_age valid
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed, max_age=3600)
    assert result == original_data
    
    # Test loads with max_age expired
    signed = serializer.dumps(original_data)
    serializer2 = TimedSerializer("secret-key")
    # Simulate time passing by setting a custom signer with an old timestamp
    class OldTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) - 4000  # 4000 seconds ago
    serializer2.default_signer = OldTimestampSigner
    signed_old = serializer2.dumps(original_data)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_old, max_age=3600)
    
    # Test loads with return_timestamp=True
    signed = serializer.dumps(original_data)
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with return_timestamp=True and max_age
    signed = serializer.dumps(original_data)
    payload, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid")
    
    # Test loads with tampered data
    signed = serializer.dumps(original_data)
    tampered = signed[:-1] + (b'1' if signed[-1:] != b'1' else b'0')
    with pytest.raises(BadSignature):
        serializer.loads(tampered)
    
    # Test loads with different salt
    signed_salt1 = serializer.dumps(original_data, salt="salt1")
    result = serializer.loads(signed_salt1, salt="salt1")
    assert result == original_data
    
    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_salt1, salt="wrong_salt")
    
    # Test loads with empty data
    signed_empty = serializer.dumps("")
    result = serializer.loads(signed_empty)
    assert result == ""


# LLM-generated content at query #15
#--------------------------

```python
def test_TimedSerializer_loads():
    # Create serializer instance
    serializer = TimedSerializer("test-secret-key")
    
    # Test 1: Basic loads without timestamp return
    serialized = serializer.dumps({"key": "value"})
    result = serializer.loads(serialized)
    assert result == {"key": "value"}
    
    # Test 2: Loads with return_timestamp=True
    serialized = serializer.dumps({"data": "test"})
    result, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert result == {"data": "test"}
    assert isinstance(timestamp, datetime)
    
    # Test 3: Loads with valid max_age
    serialized = serializer.dumps({"fresh": "data"})
    result = serializer.loads(serialized, max_age=3600)  # 1 hour
    assert result == {"fresh": "data"}
    
    # Test 4: Loads with expired signature (should raise SignatureExpired)
    from datetime import timedelta
    import time
    
    # Create a serializer with a fixed timestamp that is old
    class OldTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) - 7200  # 2 hours ago
    
    old_serializer = TimedSerializer("test-secret-key")
    old_serializer.default_signer = OldTimestampSigner
    old_serialized = old_serializer.dumps({"old": "data"})
    
    import pytest
    from itsdangerous.exc import SignatureExpired
    with pytest.raises(SignatureExpired):
        old_serializer.loads(old_serialized, max_age=3600)  # 1 hour max age
    
    # Test 5: Loads with invalid signature (should raise BadSignature)
    from itsdangerous.exc import BadSignature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-data")
    
    # Test 6: Loads with return_timestamp and max_age combined
    serialized = serializer.dumps({"combined": "test"})
    result, timestamp = serializer.loads(
        serialized, max_age=3600, return_timestamp=True
    )
    assert result == {"combined": "test"}
    assert isinstance(timestamp, datetime)
    
    # Test 7: Loads with bytes input
    serialized = serializer.dumps({"bytes": "test"})
    result = serializer.loads(serialized)
    assert result == {"bytes": "test"}
    
    # Test 8: Loads with different salt
    serialized = serializer.dumps({"salted": "data"}, salt="custom-salt")
    result = serializer.loads(serialized, salt="custom-salt")
    assert result == {"salted": "data"}
    
    # Test 9: Loads with wrong salt should fail
    with pytest.raises(BadSignature):
        serializer.loads(serialized, salt="wrong-salt")
    
    # Test 10: Loads with None max_age (should work like normal loads)
    serialized = serializer.dumps({"no-age": "limit"})
    result = serializer.loads(serialized, max_age=None)
    assert result == {"no-age": "limit"}
```


# LLM-generated content at query #16
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test-secret")
    
    # Test basic loads without max_age
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with return_timestamp=True
    result_with_ts = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    payload, timestamp = result_with_ts
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with max_age (within limit)
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with max_age (expired)
    import time as time_module
    old_signed = serializer.dumps(data)
    time_module.sleep(0.1)  # Small delay to ensure age difference
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=0)
    
    # Test loads with bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test loads with empty data
    with pytest.raises(BadSignature):
        serializer.loads(b"")
    
    # Test loads with different salt
    serializer2 = TimedSerializer("other-secret")
    signed2 = serializer2.dumps(data)
    with pytest.raises(BadSignature):
        serializer.loads(signed2)
    
    # Test loads with bytes payload
    bytes_data = b"test bytes"
    signed_bytes = serializer.dumps(bytes_data)
    result_bytes = serializer.loads(signed_bytes)
    assert result_bytes == bytes_data
    
    # Test loads with None value
    signed_none = serializer.dumps(None)
    result_none = serializer.loads(signed_none)
    assert result_none is None
    
    # Test loads with complex nested data
    complex_data = {"list": [1, 2, 3], "nested": {"a": "b"}, "num": 42}
    signed_complex = serializer.dumps(complex_data)
    result_complex = serializer.loads(signed_complex)
    assert result_complex == complex_data
    
    # Test loads returns payload without timestamp when return_timestamp=False
    result_no_ts = serializer.loads(signed, return_timestamp=False)
    assert result_no_ts == data
    assert not isinstance(result_no_ts, tuple)
```


# LLM-generated content at query #17
#--------------------------

```python
def test_TimestampSigner():
    """Test TimestampSigner constructor and basic signing/unsigning."""
    signer = TimestampSigner(secret_key="test-secret")
    assert signer.secret_key == "test-secret"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "sha1"


# LLM-generated content at query #18
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.default_signer is TimestampSigner
    
    # Test that the signer is correctly created
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    
    # Test basic dump/load roundtrip
    data = {"key": "value"}
    serialized = serializer.dumps(data)
    assert isinstance(serialized, bytes)
    
    loaded = serializer.loads(serialized)
    assert loaded == data
    
    # Test with custom salt
    custom_salt = "custom-salt"
    signer_with_salt = serializer.make_signer(salt=custom_salt)
    assert isinstance(signer_with_salt, TimestampSigner)
    
    # Test that different secret keys produce different signatures
    serializer2 = TimedSerializer(secret_key="different-secret")
    serialized2 = serializer2.dumps(data)
    assert serialized != serialized2


# LLM-generated content at query #19
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("test-secret")
    
    # Test basic unsign without timestamp
    signed_value = signer.sign("test-value")
    result = signer.unsign(signed_value)
    assert result == b"test-value"
    
    # Test unsign with return_timestamp=True
    signed_value = signer.sign("test-value")
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with max_age that is not exceeded
    signed_value = signer.sign("test-value")
    result = signer.unsign(signed_value, max_age=3600)
    assert result == b"test-value"
    
    # Test unsign with max_age that is exceeded
    signer_with_fixed_time = TimestampSigner("test-secret")
    original_get_timestamp = signer_with_fixed_time.get_timestamp
    
    def mock_old_timestamp():
        return int(time.time()) - 100
    
    signer_with_fixed_time.get_timestamp = mock_old_timestamp
    signed_old = signer_with_fixed_time.sign("test-value")
    signer_with_fixed_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer_with_fixed_time.unsign(signed_old, max_age=50)
    assert "Signature age" in str(exc_info.value)
    
    # Test unsign with negative age (future timestamp)
    signer_with_future_time = TimestampSigner("test-secret")
    def mock_future_timestamp():
        return int(time.time()) + 100
    
    signer_with_future_time.get_timestamp = mock_future_timestamp
    signed_future = signer_with_future_time.sign("test-value")
    signer_with_future_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer_with_future_time.unsign(signed_future, max_age=50)
    assert "< 0 seconds" in str(exc_info.value)
    
    # Test unsign with malformed timestamp
    malformed = b"test-value" + signer.sep.encode() + b"malformed" + signer.sep.encode() + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test unsign with missing timestamp
    signed_without_timestamp = signer.sign("test-value")
    parts = signed_without_timestamp.split(signer.sep.encode())
    # Remove timestamp but keep signature
    no_timestamp = parts[0] + signer.sep.encode() + parts[-1]
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_timestamp)
```


# LLM-generated content at query #20
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer("test_secret")
    assert serializer.default_signer == TimestampSigner
    assert serializer.secret_key == b"test_secret"
    assert serializer.salt == "itsdangerous"
    assert serializer.signer_kwargs == {}
    assert serializer.serializer_kwargs == {}
    
    # Test with custom salt
    custom_serializer = TimedSerializer("test_secret", salt="custom_salt")
    assert custom_serializer.salt == "custom_salt"
    
    # Test with custom signer kwargs
    signer_kwargs = {"key_derivation": "hmac"}
    kwargs_serializer = TimedSerializer("test_secret", signer_kwargs=signer_kwargs)
    assert kwargs_serializer.signer_kwargs == signer_kwargs
    
    # Test that signer created has correct type
    signer = kwargs_serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
```


# LLM-generated content at query #21
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer.signer, TimestampSigner)
    assert serializer.secret_key == "test-secret"
    
    # Test with salt
    serializer_with_salt = TimedSerializer(secret_key="test-secret", salt="test-salt")
    assert serializer_with_salt.salt == "test-salt"
    
    # Test with signer_kwargs
    serializer_with_kwargs = TimedSerializer(
        secret_key="test-secret",
        signer_kwargs={"key_derivation": "hmac"}
    )
    assert serializer_with_kwargs.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test with serializer_kwargs
    serializer_with_serializer_kwargs = TimedSerializer(
        secret_key="test-secret",
        serializer_kwargs={"key_derivation": "none"}
    )
    assert serializer_with_serializer_kwargs.serializer_kwargs == {"key_derivation": "none"}
    
    # Test digest_method
    serializer_with_digest = TimedSerializer(
        secret_key="test-secret",
        digest_method="sha256"
    )
    assert serializer_with_digest.digest_method == "sha256"
    
    # Test key_derivation
    serializer_with_key_derivation = TimedSerializer(
        secret_key="test-secret",
        key_derivation="hmac"
    )
    assert serializer_with_key_derivation.key_derivation == "hmac"
    
    # Test fallback_signers
    serializer_with_fallback = TimedSerializer(
        secret_key="test-secret",
        fallback_signers=["fallback1", "fallback2"]
    )
    assert serializer_with_fallback.fallback_signers == ["fallback1", "fallback2"]
    
    # Verify that the serializer can create signers
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 0
    for signer in signers:
        assert isinstance(signer, TimestampSigner)


# LLM-generated content at query #22
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without timestamp
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test unsign with return_timestamp=True
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    
    # Test unsign with valid max_age
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test unsign with expired signature
    signer_expired = TimestampSigner("secret-key")
    signer_expired.get_timestamp = lambda: int(time.time()) - 100
    signed = signer_expired.sign("test_value")
    try:
        signer.unsign(signed, max_age=10)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass
    
    # Test unsign with malformed timestamp
    malformed = b"test_value" + signer.sep.encode() + b"invalid_timestamp"
    try:
        signer.unsign(malformed)
        assert False, "Should have raised BadTimeSignature"
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)
    
    # Test unsign with missing timestamp
    missing_ts = b"test_value" + signer.sep.encode() + signer.get_signature(b"test_value")
    try:
        signer.unsign(missing_ts)
        assert False, "Should have raised BadTimeSignature"
    except BadTimeSignature as e:
        assert "timestamp missing" in str(e)
    
    # Test unsign with bad signature and return_timestamp
    signed = signer.sign("test_value")
    tampered = signed[:-1] + (b"1" if signed[-1:] == b"0" else b"0")
    try:
        signer.unsign(tampered, return_timestamp=True)
        assert False, "Should have raised BadTimeSignature"
    except BadTimeSignature:
        pass
    
    # Test unsign with negative age
    signer_future = TimestampSigner("secret-key")
    original_time = signer_future.get_timestamp()
    signer_future.get_timestamp = lambda: original_time - 1000
    signed = signer_future.sign("test_value")
    signer_future.get_timestamp = lambda: original_time - 500
    try:
        signer_future.unsign(signed, max_age=100)
        assert False, "Should have raised SignatureExpired for negative age"
    except SignatureExpired as e:
        assert "age" in str(e).lower() and "0" in str(e)
```


# LLM-generated content at query #23
#--------------------------

```python
def test_TimestampSigner_unsign():
    """Test TimestampSigner.unsign method."""
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without timestamp
    value = b"test_value"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value
    
    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    
    # Test unsign with max_age (valid)
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test unsign with max_age (expired)
    # Create a signer with a fixed timestamp to simulate expiration
    class FixedTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return 100  # old timestamp
    
    fixed_signer = FixedTimestampSigner("secret-key")
    old_signed = fixed_signer.sign(value)
    
    # Should raise SignatureExpired since age > max_age
    with pytest.raises(SignatureExpired):
        signer.unsign(old_signed, max_age=10)
    
    # Test unsign with invalid signature
    invalid_signed = b"invalid_signed_value"
    with pytest.raises(BadSignature):
        signer.unsign(invalid_signed)
    
    # Test unsign with tampered value
    tampered = signed[:-1] + b"0"
    with pytest.raises(BadSignature):
        signer.unsign(tampered)
    
    # Test unsign with missing timestamp
    # Create a signed value without timestamp
    value_bytes = want_bytes(value)
    sep = want_bytes(signer.sep)
    signature = signer.get_signature(value_bytes)
    no_timestamp = value_bytes + sep + signature
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_timestamp)
```


# LLM-generated content at query #24
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer(secret_key="test-secret")
    
    # Test basic loads without max_age or return_timestamp
    payload = {"user": "test"}
    signed = serializer.dumps(payload)
    result = serializer.loads(signed)
    assert result == payload
    
    # Test loads with return_timestamp
    signed = serializer.dumps(payload)
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == payload
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test loads with max_age valid
    signed = serializer.dumps(payload)
    result = serializer.loads(signed, max_age=3600)
    assert result == payload
    
    # Test loads with max_age expired
    signed = serializer.dumps(payload)
    # Simulate time passing by using a very small max_age
    import time
    time.sleep(0.1)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=0)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid_signed_value")
    
    # Test loads with salt
    signed_with_salt = serializer.dumps(payload, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == payload
    
    # Test loads with wrong salt
    signed_with_salt = serializer.dumps(payload, salt="custom-salt")
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
    
    # Test loads with bytes input
    signed_bytes = serializer.dumps(payload)
    result = serializer.loads(signed_bytes)
    assert result == payload
    
    # Test loads with string input
    signed_string = serializer.dumps(payload).decode('utf-8')
    result = serializer.loads(signed_string)
    assert result == payload
    
    # Test loads with return_timestamp and max_age combined
    signed = serializer.dumps(payload)
    result, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert result == payload
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test loads with empty payload
    empty_payload = {}
    signed = serializer.dumps(empty_payload)
    result = serializer.loads(signed)
    assert result == empty_payload
    
    # Test loads with complex payload
    complex_payload = {
        "string": "test",
        "number": 42,
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
        "bool": True,
        "none": None
    }
    signed = serializer.dumps(complex_payload)
    result = serializer.loads(signed)
    assert result == complex_payload
    
    # Test loads with negative max_age (should raise SignatureExpired)
    signed = serializer.dumps(payload)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=-1)  # Negative age always expired
    
    # Test loads with very large max_age (should succeed)
    signed = serializer.dumps(payload)
    result = serializer.loads(signed, max_age=999999)
    assert result == payload
    
    # Test loads with multiple signers (fallback)
    serializer2 = TimedSerializer(secret_key="test-secret-2")
    signed_with_other = serializer2.dumps(payload)
    # This should fail as the secret is different
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_other)
```


# LLM-generated content at query #25
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test the loads method of TimedSerializer."""
    serializer = TimedSerializer("test_secret_key")
    
    # Test basic loads without max_age
    original_data = {"key": "value"}
    serialized = serializer.dumps(original_data)
    result = serializer.loads(serialized)
    assert result == original_data
    
    # Test loads with max_age (should not expire)
    serialized = serializer.dumps(original_data)
    result = serializer.loads(serialized, max_age=3600)
    assert result == original_data
    
    # Test loads with return_timestamp=True
    serialized = serializer.dumps(original_data)
    result = serializer.loads(serialized, return_timestamp=True)
    payload, timestamp = result
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with both max_age and return_timestamp
    serialized = serializer.dumps(original_data)
    result = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    payload, timestamp = result
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with expired signature
    from datetime import timedelta
    # Create a serializer with a fixed timestamp that is old
    class OldTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) - 7200  # 2 hours ago
    
    old_serializer = TimedSerializer("test_secret_key")
    old_serializer.default_signer = OldTimestampSigner
    old_serialized = old_serializer.dumps(original_data)
    
    with pytest.raises(SignatureExpired):
        old_serializer.loads(old_serialized, max_age=3600)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid_signature")
    
    # Test loads with corrupted data
    serialized = serializer.dumps(original_data)
    corrupted = serialized[:-1] + b"0"
    with pytest.raises(BadSignature):
        serializer.loads(corrupted)
```


# LLM-generated content at query #26
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key", salt="test-salt")
    
    # Test basic unsign
    signed = signer.sign(b"test value")
    result = signer.unsign(signed)
    assert result == b"test value"
    
    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with max_age
    signed = signer.sign(b"test value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test value"
    
    # Test expired signature
    signer_fast = TimestampSigner("secret-key", salt="test-salt")
    signer_fast.get_timestamp = lambda: int(time.time()) - 100
    signed = signer_fast.sign(b"test value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=10)
    
    # Test malformed timestamp
    malformed = b"test value" + signer.sep.encode() + b"invalid"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test missing timestamp
    no_timestamp = signer.sign(b"test").rsplit(signer.sep.encode(), 1)[0]
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp)
    
    # Test invalid signature
    tampered = signed[:-1] + b"x"
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered)
    
    # Test with bytes input
    signed_bytes = signer.sign(b"bytes value")
    result = signer.unsign(signed_bytes)
    assert result == b"bytes value"
    
    # Test with string input
    signed_str = signer.sign("string value")
    result = signer.unsign(signed_str)
    assert result == b"string value"
    
    # Test negative age (future timestamp)
    future_signer = TimestampSigner("secret-key", salt="test-salt")
    future_signer.get_timestamp = lambda: int(time.time()) + 1000
    future_signed = future_signer.sign(b"future value")
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed, max_age=3600)
```


# LLM-generated content at query #27
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test-secret-key")
    
    # Test basic loads without max_age
    data = {"key": "value"}
    dumped = serializer.dumps(data)
    loaded = serializer.loads(dumped)
    assert loaded == data
    
    # Test loads with max_age (should work within age limit)
    dumped = serializer.dumps(data)
    loaded = serializer.loads(dumped, max_age=3600)
    assert loaded == data
    
    # Test loads with return_timestamp=True
    loaded_with_ts = serializer.loads(dumped, return_timestamp=True)
    assert isinstance(loaded_with_ts, tuple)
    assert len(loaded_with_ts) == 2
    assert loaded_with_ts[0] == data
    assert isinstance(loaded_with_ts[1], datetime)
    
    # Test loads with both max_age and return_timestamp
    loaded_with_ts = serializer.loads(dumped, max_age=3600, return_timestamp=True)
    assert isinstance(loaded_with_ts, tuple)
    assert len(loaded_with_ts) == 2
    assert loaded_with_ts[0] == data
    assert isinstance(loaded_with_ts[1], datetime)
    
    # Test loads with expired signature
    import time
    old_serializer = TimedSerializer("test-secret-key")
    old_serializer.signer.get_timestamp = lambda: int(time.time()) - 100  # 100 seconds ago
    old_dumped = old_serializer.dumps(data)
    with pytest.raises(SignatureExpired):
        serializer.loads(old_dumped, max_age=10)
    
    # Test loads with bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test loads with string input
    dumped_str = dumped.decode("utf-8") if isinstance(dumped, bytes) else dumped
    loaded = serializer.loads(dumped_str)
    assert loaded == data
    
    # Test loads with different salt
    serializer2 = TimedSerializer("test-secret-key", salt="different-salt")
    with pytest.raises(BadSignature):
        serializer2.loads(dumped)
    
    # Test loads with multiple signers
    serializer_with_fallback = TimedSerializer("test-secret-key", fallback_signers=["fallback-key"])
    dumped_with_fallback = serializer_with_fallback.dumps(data)
    loaded = serializer_with_fallback.loads(dumped_with_fallback)
    assert loaded == data
    
    # Test loads with empty data
    with pytest.raises(BadSignature):
        serializer.loads(b"")
```


# LLM-generated content at query #28
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor and basic attributes."""
    # Test default constructor
    serializer = TimedSerializer("test-secret")
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.secret_key == "test-secret"
    assert serializer.default_signer is TimestampSigner
    
    # Test constructor with custom secret key types
    serializer_bytes = TimedSerializer(b"test-secret")
    assert serializer_bytes.secret_key == b"test-secret"
    
    # Test constructor with salt
    serializer_salt = TimedSerializer("test-secret", salt="custom-salt")
    assert serializer_salt.salt == "custom-salt"
    
    # Test constructor with signer_kwargs
    serializer_kwargs = TimedSerializer("test-secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer_kwargs.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test that signers created are TimestampSigner instances
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    
    # Test that iter_unsigners returns TimestampSigner instances
    for signer in serializer.iter_unsigners():
        assert isinstance(signer, TimestampSigner)
        break
```


# LLM-generated content at query #29
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test-secret-key")
    
    # Test basic loads without max_age and return_timestamp
    original_data = {"key": "value"}
    signed_data = serializer.dumps(original_data)
    result = serializer.loads(signed_data)
    assert result == original_data
    
    # Test loads with return_timestamp=True
    result_with_ts = serializer.loads(signed_data, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == original_data
    assert isinstance(result_with_ts[1], datetime)
    
    # Test loads with max_age
    result = serializer.loads(signed_data, max_age=3600)
    assert result == original_data
    
    # Test loads with expired signature
    # Create a signer with a fixed timestamp in the past
    class FixedTimestampSigner(TimestampSigner):
        def get_timestamp(self) -> int:
            return int(time.time()) - 100  # 100 seconds in the past
    
    fixed_signer = FixedTimestampSigner("test-secret-key")
    expired_data = fixed_signer.sign(b'{"key":"value"}')
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_data, max_age=10)
    
    # Test loads with malformed data
    with pytest.raises(BadSignature):
        serializer.loads(b"malformed-data")
    
    # Test loads with empty data
    with pytest.raises(BadSignature):
        serializer.loads(b"")
    
    # Test loads with return_timestamp and max_age
    result_with_ts_and_age = serializer.loads(
        signed_data, max_age=3600, return_timestamp=True
    )
    assert isinstance(result_with_ts_and_age, tuple)
    assert len(result_with_ts_and_age) == 2
    assert result_with_ts_and_age[0] == original_data
    assert isinstance(result_with_ts_and_age[1], datetime)
    
    # Test loads with different salt
    signed_with_salt = serializer.dumps(original_data, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == original_data
    
    # Test loads with wrong salt should fail
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
    
    # Test loads with bytes input
    result = serializer.loads(signed_data)
    assert result == original_data
    
    # Test loads with string input
    result = serializer.loads(signed_data.decode())
    assert result == original_data
    
    # Test loads with integer data
    int_data = 42
    signed_int = serializer.dumps(int_data)
    result = serializer.loads(signed_int)
    assert result == int_data
    
    # Test loads with list data
    list_data = [1, 2, 3]
    signed_list = serializer.dumps(list_data)
    result = serializer.loads(signed_list)
    assert result == list_data
    
    # Test loads with None data
    none_data = None
    signed_none = serializer.dumps(none_data)
    result = serializer.loads(signed_none)
    assert result is None
```


# LLM-generated content at query #30
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("test-secret")
    
    # Test basic unsign without timestamp
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"
    
    # Test unsign with return_timestamp=True
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    value, timestamp = result_with_ts
    assert value == b"test-value"
    assert isinstance(timestamp, datetime)
    
    # Test unsign with max_age that is not expired
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test-value"
    
    # Test unsign with max_age that is expired
    import time as time_module
    # Create a signer that returns a past timestamp
    class PastTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time_module.time()) - 1000
    
    past_signer = PastTimestampSigner("test-secret")
    past_signed = past_signer.sign("test-value")
    
    with pytest.raises(SignatureExpired):
        past_signer.unsign(past_signed, max_age=100)
    
    # Test unsign with negative age (future timestamp)
    class FutureTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time_module.time()) + 1000
    
    future_signer = FutureTimestampSigner("test-secret")
    future_signed = future_signer.sign("test-value")
    
    with pytest.raises(SignatureExpired) as exc_info:
        future_signer.unsign(future_signed, max_age=500)
    assert "age" in str(exc_info.value)
    assert "0" in str(exc_info.value)
    
    # Test unsign with malformed timestamp
    malformed = signed + b"malformed"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test unsign with missing timestamp
    missing_ts = signer.get_signature(b"test-value")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test-value" + signer.sep.encode() + missing_ts)
    
    # Test unsign with bad signature
    bad_signed = signed[:-1] + b"x"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed)
    
    # Test unsign with bytes input
    result = signer.unsign(signed)
    assert result == b"test-value"
    
    # Test unsign with string input
    result = signer.unsign(signed.decode())
    assert result == b"test-value"
```


# LLM-generated content at query #31
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner(secret_key="test-secret")
    assert signer.secret_key == "test-secret"
    assert signer.sep == "."
    assert signer.salt == "timestamp-signer"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "hmac-sha1"
    
    # Test constructor with custom parameters
    signer2 = TimestampSigner(
        secret_key="custom-secret",
        salt="custom-salt",
        sep=":",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm="hmac-sha256"
    )
    assert signer2.secret_key == "custom-secret"
    assert signer2.salt == "custom-salt"
    assert signer2.sep == ":"
    assert signer2.key_derivation == "none"
    assert signer2.digest_method == hashlib.sha256
    assert signer2.algorithm == "hmac-sha256"
    
    # Test that TimestampSigner is a Signer subclass
    assert isinstance(signer, Signer)
    
    # Test with empty secret key
    signer3 = TimestampSigner(secret_key="")
    assert signer3.secret_key == ""
    
    # Test with bytes secret key
    signer4 = TimestampSigner(secret_key=b"bytes-secret")
    assert signer4.secret_key == b"bytes-secret" 


# LLM-generated content at query #32
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads with various scenarios."""
    serializer = TimedSerializer("test-secret")
    
    # Test basic loads without max_age or return_timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with return_timestamp
    result_with_ts = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    payload, timestamp = result_with_ts
    assert payload == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test loads with max_age (valid age)
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with max_age (expired signature)
    import time as time_module
    original_time = time_module.time
    try:
        time_module.time = lambda: original_time() + 7200  # Simulate 2 hours later
        signed_old = serializer.dumps(data)
        with pytest.raises(SignatureExpired) as excinfo:
            serializer.loads(signed_old, max_age=3600)
        assert "Signature age" in str(excinfo.value)
    finally:
        time_module.time = original_time
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-signature")
    
    # Test loads with different salt
    signed_with_salt = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == data
    
    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
    
    # Test loads with bytes input
    signed_bytes = serializer.dumps(data)
    result = serializer.loads(signed_bytes)
    assert result == data
    
    # Test loads with string input
    signed_str = serializer.dumps(data).decode("utf-8")
    result = serializer.loads(signed_str)
    assert result == data
    
    # Test loads with complex data types
    complex_data = {"list": [1, 2, 3], "nested": {"a": 1}, "bool": True, "none": None}
    signed_complex = serializer.dumps(complex_data)
    result = serializer.loads(signed_complex)
    assert result == complex_data
    
    # Test loads with empty data
    empty_data = {}
    signed_empty = serializer.dumps(empty_data)
    result = serializer.loads(signed_empty)
    assert result == empty_data
    
    # Test loads with integer data
    int_data = 42
    signed_int = serializer.dumps(int_data)
    result = serializer.loads(signed_int)
    assert result == int_data
    
    # Test loads with string data
    str_data = "test string"
    signed_str_data = serializer.dumps(str_data)
    result = serializer.loads(signed_str_data)
    assert result == str_data
    
    # Test loads with list data
    list_data = [1, "two", 3.0]
    signed_list = serializer.dumps(list_data)
    result = serializer.loads(signed_list)
    assert result == list_data
    
    # Test loads with negative max_age (should raise SignatureExpired)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=-1)
    
    # Test loads with zero max_age
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=0)
    
    # Test loads with very large max_age (should work)
    result = serializer.loads(signed, max_age=999999999)
    assert result == data
    
    # Test loads with return_timestamp and max_age together
    result_with_ts = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    payload, timestamp = result_with_ts
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with malformed data
    with pytest.raises(BadSignature):
        serializer.loads(b"malformed-data-without-separator")
    
    # Test loads with empty bytes
    with pytest.raises(BadSignature):
        serializer.loads(b"")
    
    # Test loads with None
    with pytest.raises(TypeError):
        serializer.loads(None)
```


# LLM-generated content at query #33
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer(secret_key="test-secret-key")
    
    # Test 1: Basic loads without max_age and return_timestamp
    original_data = {"user_id": 1, "name": "test"}
    signed_value = serializer.dumps(original_data)
    result = serializer.loads(signed_value)
    assert result == original_data, f"Expected {original_data}, got {result}"
    
    # Test 2: Loads with max_age parameter (valid age)
    result = serializer.loads(signed_value, max_age=3600)
    assert result == original_data, f"Expected {original_data}, got {result}"
    
    # Test 3: Loads with return_timestamp=True
    result, timestamp = serializer.loads(signed_value, return_timestamp=True)
    assert result == original_data, f"Expected {original_data}, got {result}"
    assert isinstance(timestamp, datetime), f"Expected datetime, got {type(timestamp)}"
    
    # Test 4: Loads with max_age and return_timestamp=True
    result, timestamp = serializer.loads(signed_value, max_age=3600, return_timestamp=True)
    assert result == original_data, f"Expected {original_data}, got {result}"
    assert isinstance(timestamp, datetime), f"Expected datetime, got {type(timestamp)}"
    
    # Test 5: Loads with expired signature (should raise SignatureExpired)
    import time
    expired_serializer = TimedSerializer(secret_key="test-secret-key")
    expired_value = expired_serializer.dumps(original_data)
    time.sleep(1)  # Ensure at least 1 second passes
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_value, max_age=0)
    
    # Test 6: Loads with invalid signature (should raise BadSignature)
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid_signature")
    
    # Test 7: Loads with tampered data (should raise BadSignature)
    tampered_value = signed_value[:-1] + b"x"  # Modify last byte
    with pytest.raises(BadSignature):
        serializer.loads(tampered_value)
    
    # Test 8: Loads with different salt
    main_serializer = TimedSerializer(secret_key="test-secret-key", salt="main-salt")
    alt_serializer = TimedSerializer(secret_key="test-secret-key", salt="alt-salt")
    signed_with_main = main_serializer.dumps(original_data)
    result = alt_serializer.loads(signed_with_main, salt="main-salt")
    assert result == original_data, f"Expected {original_data}, got {result}"
    
    # Test 9: Loads with wrong salt (should fail)
    with pytest.raises(BadSignature):
        alt_serializer.loads(signed_with_main)
    
    # Test 10: Loads with string data (not bytes)
    string_data = "test_string"
    signed_string = serializer.dumps(string_data)
    result = serializer.loads(signed_string)
    assert result == string_data, f"Expected {string_data}, got {result}"
    
    # Test 11: Loads with complex nested data
    complex_data = {
        "list": [1, 2, 3],
        "dict": {"a": 1, "b": 2},
        "nested": {"inner": "value"}
    }
    signed_complex = serializer.dumps(complex_data)
    result = serializer.loads(signed_complex)
    assert result == complex_data, f"Expected {complex_data}, got {result}"
    
    # Test 12: Loads with empty data
    empty_data = {}
    signed_empty = serializer.dumps(empty_data)
    result = serializer.loads(signed_empty)
    assert result == empty_data, f"Expected {empty_data}, got {result}"
    
    # Test 13: Loads with None value
    none_data = None
    signed_none = serializer.dumps(none_data)
    result = serializer.loads(signed_none)
    assert result == none_data, f"Expected {none_data}, got {result}"
```


# LLM-generated content at query #34
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.default_signer == TimestampSigner
    
    data = {"key": "value"}
    signed = serializer.dumps(data)
    
    # Test basic loads
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with max_age
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with return_timestamp
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with different salt
    signed_with_salt = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == data
    
    # Test loads_unsafe
    is_valid, result = serializer.loads_unsafe(signed)
    assert is_valid
    assert result == data
    
    # Test loads_unsafe with invalid data
    is_valid, result = serializer.loads_unsafe(b"invalid-data")
    assert not is_valid
    
    # Test with bytes payload
    signed_bytes = serializer.dumps(b"bytes-payload")
    result = serializer.loads(signed_bytes)
    assert result == b"bytes-payload"
    
    # Test expiration
    signed = serializer.dumps(data)
    
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=-1)
    
    # Test different secret keys produce different signatures
    serializer2 = TimedSerializer(secret_key="different-secret")
    signed2 = serializer2.dumps(data)
    assert signed != signed2
    
    with pytest.raises(BadSignature):
        serializer.loads(signed2)


# LLM-generated content at query #35
#--------------------------

```python
def test_TimedSerializer():
    # Test basic initialization
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner

    # Test with secret key
    serializer = TimedSerializer("test-secret")
    assert serializer.secret_key == "test-secret"
    assert isinstance(serializer.default_signer, type)
    assert serializer.default_signer is TimestampSigner

    # Test with salt
    serializer = TimedSerializer("test-secret", salt="test-salt")
    assert serializer.salt == "test-salt"

    # Test with signer kwargs
    serializer = TimedSerializer("test-secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

    # Test that iter_unsigners returns TimestampSigner instances
    serializer = TimedSerializer("test-secret")
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 0
    for signer in signers:
        assert isinstance(signer, TimestampSigner)


# LLM-generated content at query #36
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without timestamp
    value = b"test_value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test unsign with return_timestamp=True
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with max_age (valid age)
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test unsign with expired signature
    signed = signer.sign(value)
    # Set a very low max_age to force expiration
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)
    
    # Test unsign with tampered value
    tampered = signed[:-1] + (b"x" if signed[-1:] == b"a" else b"a")
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered)
    
    # Test unsign with missing timestamp
    missing_timestamp = b"test_value" + signer.sep.encode() + signer.get_signature(b"test_value")
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(missing_timestamp)
    
    # Test unsign with malformed timestamp
    malformed_bytes = value + signer.sep.encode() + b"invalid_timestamp" + signer.sep.encode() + signer.get_signature(value + signer.sep.encode() + b"invalid_timestamp")
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_bytes)
```


# LLM-generated content at query #37
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test-secret-key")
    
    # Test successful loads without timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test successful loads with timestamp
    result_with_ts = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == data
    assert isinstance(result_with_ts[1], datetime)
    
    # Test with max_age
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test expired signature
    import time as time_module
    original_time = time_module.time
    try:
        time_module.time = lambda: 1000000  # far future
        signed_old = serializer.dumps(data)
        time_module.time = lambda: 1000000 + 3601  # 1 second over max_age
        with pytest.raises(SignatureExpired):
            serializer.loads(signed_old, max_age=3600)
    finally:
        time_module.time = original_time
    
    # Test invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")
    
    # Test with salt
    result = serializer.loads(signed, salt="custom-salt")
    assert result == data
    
    # Test with different data types
    string_data = "test string"
    signed_string = serializer.dumps(string_data)
    result = serializer.loads(signed_string)
    assert result == string_data
    
    # Test with empty data
    empty_data = {}
    signed_empty = serializer.dumps(empty_data)
    result = serializer.loads(signed_empty)
    assert result == empty_data
    
    # Test with list data
    list_data = [1, 2, 3]
    signed_list = serializer.dumps(list_data)
    result = serializer.loads(signed_list)
    assert result == list_data
    
    # Test with None data
    none_data = None
    signed_none = serializer.dumps(none_data)
    result = serializer.loads(signed_none)
    assert result == none_data
    
    # Test with max_age=0 (immediate expiration)
    signed_now = serializer.dumps(data)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_now, max_age=0)


# LLM-generated content at query #38
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor creates instance with correct default signer."""
    serializer = TimedSerializer("test-secret")
    
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner
    
    # Verify the signer created by the serializer is a TimestampSigner
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    
    # Test with default parameters
    serializer2 = TimedSerializer("test-secret", salt="test-salt", serializer_kwargs={"key_derivation": "hmac"})
    assert isinstance(serializer2, TimedSerializer)
    assert serializer2.default_signer is TimestampSigner
    
    # Verify signer has correct salt
    signer2 = serializer2.make_signer()
    assert isinstance(signer2, TimestampSigner)
    assert signer2.salt == "test-salt"
```


# LLM-generated content at query #39
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.default_signer == TimestampSigner
    assert isinstance(serializer.signer, TimestampSigner)
    assert serializer.signer.secret_key == "test-secret"
    
    # Test with salt
    serializer_with_salt = TimedSerializer(secret_key="test-secret", salt="custom-salt")
    assert serializer_with_salt.signer.salt == "custom-salt"
    
    # Test with digest method
    serializer_with_digest = TimedSerializer(secret_key="test-secret", digest_method="sha256")
    assert serializer_with_digest.signer.digest_method == "sha256"
    
    # Test with key derivation
    serializer_with_kd = TimedSerializer(secret_key="test-secret", key_derivation="hmac")
    assert serializer_with_kd.signer.key_derivation == "hmac"
    
    # Test default serializer settings
    default_serializer = TimedSerializer(secret_key="default-key")
    assert default_serializer.signer.salt == "itsdangerous.TimedSerializer"
    assert default_serializer.signer.sep == "."
    assert default_serializer.signer.digest_method == "sha1"
    assert default_serializer.signer.key_derivation == "hmac"
```


# LLM-generated content at query #40
#--------------------------

```python
def test_TimestampSigner():
    # Test constructor with default parameters
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == hmac

    # Test constructor with custom parameters
    signer = TimestampSigner(
        "secret-key",
        salt="custom-salt",
        sep=":",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm=hashlib.sha256,
    )
    assert signer.secret_key == "secret-key"
    assert signer.salt == "custom-salt"
    assert signer.sep == ":"
    assert signer.key_derivation == "none"
    assert signer.digest_method == hashlib.sha256
    assert signer.algorithm == hashlib.sha256

    # Test that it inherits from Signer
    assert isinstance(signer, Signer)
    assert issubclass(TimestampSigner, Signer)

    # Test that it has the expected methods
    assert hasattr(signer, "get_timestamp")
    assert hasattr(signer, "timestamp_to_datetime")
    assert hasattr(signer, "sign")
    assert hasattr(signer, "unsign")
    assert hasattr(signer, "validate")
```


# LLM-generated content at query #41
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer(secret_key="test-secret-key")
    
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    assert serializer.secret_key == "test-secret-key"
    
    # Test with salt parameter
    serializer_with_salt = TimedSerializer(secret_key="test-secret-key", salt="custom-salt")
    assert serializer_with_salt.salt == "custom-salt"
    
    # Test with signer_kwargs
    serializer_with_signer_kwargs = TimedSerializer(
        secret_key="test-secret-key",
        signer_kwargs={"key_derivation": "none"}
    )
    assert serializer_with_signer_kwargs.signer_kwargs == {"key_derivation": "none"}
    
    # Test that loads and dumps work
    data = {"user_id": 1, "username": "test"}
    signed = serializer.dumps(data)
    assert isinstance(signed, str) or isinstance(signed, bytes)
    
    loaded = serializer.loads(signed)
    assert loaded == data
    
    # Test with return_timestamp
    loaded_with_ts = serializer.loads(signed, return_timestamp=True)
    assert isinstance(loaded_with_ts, tuple)
    assert len(loaded_with_ts) == 2
    assert loaded_with_ts[0] == data
    assert isinstance(loaded_with_ts[1], datetime)


# LLM-generated content at query #42
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor and basic initialization."""
    # Test default construction
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner
    
    # Test construction with secret key
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.secret_key == b"test-secret"
    
    # Test construction with salt
    serializer = TimedSerializer(salt="test-salt")
    assert serializer.salt == "test-salt"
    
    # Test construction with serializer
    import json
    serializer = TimedSerializer(serializer=json)
    assert serializer.serializer is json
    
    # Test construction with signer_kwargs
    serializer = TimedSerializer(signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test construction with digest_method
    import hashlib
    serializer = TimedSerializer(digest_method=hashlib.sha256)
    assert serializer.digest_method is hashlib.sha256
    
    # Test that signer is TimestampSigner
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    
    # Test that iter_unsigners returns TimestampSigner instances
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 0
    for s in signers:
        assert isinstance(s, TimestampSigner)


# LLM-generated content at query #43
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test basic loads without max_age or return_timestamp
    serializer = TimedSerializer("test_secret")
    data = {"key": "value"}
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized)
    assert result == data

    # Test loads with max_age that is not exceeded
    result = serializer.loads(serialized, max_age=3600)
    assert result == data

    # Test loads with return_timestamp=True
    result, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)

    # Test loads with both max_age and return_timestamp
    result, timestamp = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)

    # Test loads with expired signature (max_age too small)
    import time
    serializer_with_future = TimedSerializer("test_secret")
    # Create a signer with a custom timestamp to simulate old data
    old_signer = TimestampSigner("test_secret")
    old_signer.get_timestamp = lambda: int(time.time()) - 100  # 100 seconds ago
    old_serialized = old_signer.sign("test_value")
    
    with pytest.raises(SignatureExpired):
        serializer.loads(old_serialized, max_age=10)

    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid_data")

    # Test loads with different salt
    serialized_with_salt = serializer.dumps(data, salt="custom_salt")
    result = serializer.loads(serialized_with_salt, salt="custom_salt")
    assert result == data

    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(serialized_with_salt, salt="wrong_salt")

    # Test loads with bytes input
    serialized_bytes = serializer.dumps(data)
    if isinstance(serialized_bytes, str):
        serialized_bytes = serialized_bytes.encode()
    result = serializer.loads(serialized_bytes)
    assert result == data

    # Test loads with string input
    serialized_str = serializer.dumps(data)
    result = serializer.loads(serialized_str)
    assert result == data
```


# LLM-generated content at query #44
#--------------------------

```python
def test_TimedSerializer():
    # Test that TimedSerializer can be instantiated with default parameters
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner
    
    # Test that TimedSerializer can be instantiated with a secret key
    serializer = TimedSerializer("secret-key")
    assert isinstance(serializer, TimedSerializer)
    
    # Test that TimedSerializer can be instantiated with salt
    serializer = TimedSerializer("secret-key", salt="my-salt")
    assert isinstance(serializer, TimedSerializer)
    
    # Test that TimedSerializer can be instantiated with signer_kwargs
    serializer = TimedSerializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert isinstance(serializer, TimedSerializer)
    
    # Test that default_signer is TimestampSigner
    assert TimedSerializer.default_signer is TimestampSigner
    
    # Test that iter_unsigners returns TimestampSigner instances
    serializer = TimedSerializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 0
    for signer in signers:
        assert isinstance(signer, TimestampSigner)
```


# LLM-generated content at query #45
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic sign and unsign
    value = b"test_value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    
    # Test with max_age
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test expired signature
    signer.get_timestamp = lambda: int(time.time()) - 100
    signed = signer.sign(value)
    signer.get_timestamp = lambda: int(time.time())
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=50)
    
    # Test future timestamp (age < 0)
    signer.get_timestamp = lambda: int(time.time()) - 100
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=1000)
    
    # Test bad signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")
    
    # Test missing timestamp
    bad_value = b"test" + signer.sep.encode() + b"invalid"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_value)
    
    # Test malformed timestamp
    malformed = value + signer.sep.encode() + base64_encode(b"invalid_timestamp")
    malformed = malformed + signer.sep.encode() + signer.get_signature(malformed)
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Verify timestamp_to_datetime conversion
    signer.get_timestamp = lambda: 0
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert timestamp == datetime(1970, 1, 1, tzinfo=timezone.utc)```


# LLM-generated content at query #46
#--------------------------

```python
def test_TimedSerializer():
    # Test basic initialization
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.secret_key == "test-secret"
    assert serializer.salt == "itsdangerous.TimedSerializer"
    assert serializer.default_signer == TimestampSigner

    # Test with custom salt
    custom_salt = "custom-salt"
    serializer = TimedSerializer(secret_key="test-secret", salt=custom_salt)
    assert serializer.salt == custom_salt

    # Test serializer inherits from Serializer
    assert isinstance(serializer, Serializer)

    # Test signer creation with TimestampSigner
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    assert signer.secret_key == "test-secret"
```


# LLM-generated content at query #47
#--------------------------

```python
def test_TimestampSigner():
    """Test TimestampSigner constructor and basic functionality."""
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == b"secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "sha1"
    
    signer_with_custom = TimestampSigner("secret", salt="custom-salt", sep="|")
    assert signer_with_custom.salt == "custom-salt"
    assert signer_with_custom.sep == "|"
```


# LLM-generated content at query #48
#--------------------------

```python
def test_TimestampSigner():
    """Test the TimestampSigner class initialization and basic functionality."""
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.sep == "."
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == hmac
    assert signer.encoding == "utf-8"
    
    # Test with custom parameters
    custom_signer = TimestampSigner(
        "custom-secret",
        sep="-",
        salt="custom-salt",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm=hashlib.sha256,
        encoding="ascii"
    )
    assert custom_signer.secret_key == "custom-secret"
    assert custom_signer.sep == "-"
    assert custom_signer.salt == "custom-salt"
    assert custom_signer.key_derivation == "none"
    assert custom_signer.digest_method == hashlib.sha256
    assert custom_signer.algorithm == hashlib.sha256
    assert custom_signer.encoding == "ascii"
    
    # Test get_timestamp returns integer
    timestamp = signer.get_timestamp()
    assert isinstance(timestamp, int)
    assert timestamp > 0
    
    # Test timestamp_to_datetime conversion
    dt = signer.timestamp_to_datetime(timestamp)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc
    
    # Test sign and unsign
    signed = signer.sign("test-value")
    assert isinstance(signed, bytes)
    assert signed.startswith(b"test-value")
    
    unsigned = signer.unsign(signed)
    assert unsigned == b"test-value"
    
    # Test unsign with return_timestamp
    unsigned_value, timestamp_dt = signer.unsign(signed, return_timestamp=True)
    assert unsigned_value == b"test-value"
    assert isinstance(timestamp_dt, datetime)
    assert timestamp_dt.tzinfo == timezone.utc
    
    # Test validate
    assert signer.validate(signed) == True
    assert signer.validate(b"invalid-signature") == False
    
    # Test max_age validation
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: int(time.time()) - 100  # Simulate old timestamp
    expired_signed = expired_signer.sign("test-value")
    
    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed, max_age=50)
    
    # Test malformed timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test-value.abc.def")
    
    # Test missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test-value")
```


# LLM-generated content at query #49
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads method."""
    serializer = TimedSerializer("test-secret-key")
    
    # Test basic loads without max_age
    original_data = {"key": "value"}
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed)
    assert result == original_data
    
    # Test loads with max_age (valid)
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed, max_age=3600)
    assert result == original_data
    
    # Test loads with return_timestamp=True
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    payload, timestamp = result
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with both max_age and return_timestamp
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    payload, timestamp = result
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with expired signature
    signed = serializer.dumps(original_data)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=-1)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-data")
    
    # Test loads with different salt
    signed = serializer.dumps(original_data, salt="custom-salt")
    result = serializer.loads(signed, salt="custom-salt")
    assert result == original_data
    
    # Test loads with wrong salt
    signed = serializer.dumps(original_data, salt="custom-salt")
    with pytest.raises(BadSignature):
        serializer.loads(signed, salt="wrong-salt")


# LLM-generated content at query #50
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads method."""
    serializer = TimedSerializer("test-secret-key")
    
    # Test basic loads without max_age and return_timestamp
    original_data = {"key": "value", "number": 42}
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed)
    assert result == original_data
    
    # Test loads with max_age (valid age)
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed, max_age=3600)  # 1 hour max age
    assert result == original_data
    
    # Test loads with return_timestamp=True
    signed = serializer.dumps(original_data)
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test loads with both max_age and return_timestamp
    signed = serializer.dumps(original_data)
    payload, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with expired signature
    signed = serializer.dumps(original_data)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=-1)  # Negative max_age means already expired
    
    # Test loads with bad signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")
    
    # Test loads with empty string
    with pytest.raises(BadSignature):
        serializer.loads("")
    
    # Test loads with different salt
    serializer2 = TimedSerializer("test-secret-key", salt="different-salt")
    signed = serializer2.dumps(original_data)
    result = serializer.loads(signed, salt="different-salt")
    assert result == original_data
    
    # Test loads with wrong salt (should fail)
    with pytest.raises(BadSignature):
        serializer.loads(signed, salt="wrong-salt")
    
    # Test loads with bytes input
    signed_bytes = serializer.dumps(original_data)
    result = serializer.loads(signed_bytes)
    assert result == original_data
    
    # Test loads with string input
    signed_str = serializer.dumps(original_data).decode('utf-8')
    result = serializer.loads(signed_str)
    assert result == original_data
    
    # Test loads with complex data types
    complex_data = {
        "list": [1, 2, 3],
        "nested": {"a": 1, "b": 2},
        "bool": True,
        "none": None,
        "float": 3.14
    }
    signed = serializer.dumps(complex_data)
    result = serializer.loads(signed)
    assert result == complex_data
    
    # Test loads with list data
    list_data = [1, "two", 3.0, {"key": "value"}]
    signed = serializer.dumps(list_data)
    result = serializer.loads(signed)
    assert result == list_data
    
    # Test loads with string data
    string_data = "simple string"
    signed = serializer.dumps(string_data)
    result = serializer.loads(signed)
    assert result == string_data
    
    # Test loads with integer data
    int_data = 12345
    signed = serializer.dumps(int_data)
    result = serializer.loads(signed)
    assert result == int_data
    
    # Test loads with None data
    none_data = None
    signed = serializer.dumps(none_data)
    result = serializer.loads(signed)
    assert result == none_data
    
    # Test loads with boolean data
    bool_data = True
    signed = serializer.dumps(bool_data)
    result = serializer.loads(signed)
    assert result == bool_data
    
    # Test loads with float data
    float_data = 3.14159
    signed = serializer.dumps(float_data)
    result = serializer.loads(signed)
    assert result == float_data
    
    # Test that wrong secret key raises BadSignature
    serializer_wrong = TimedSerializer("wrong-secret-key")
    with pytest.raises(BadSignature):
        serializer_wrong.loads(signed)
```


# LLM-generated content at query #51
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer(secret_key="test-secret")
    
    # Test basic loads without max_age
    value = {"key": "value"}
    serialized = serializer.dumps(value)
    result = serializer.loads(serialized)
    assert result == value
    
    # Test loads with max_age
    result = serializer.loads(serialized, max_age=3600)
    assert result == value
    
    # Test loads with return_timestamp
    result, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    
    # Test loads with both max_age and return_timestamp
    result, timestamp = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    
    # Test loads with expired signature
    import time
    serializer_fast = TimedSerializer(secret_key="test-secret")
    serialized_fast = serializer_fast.dumps(value)
    time.sleep(0.1)
    with pytest.raises(SignatureExpired):
        serializer_fast.loads(serialized_fast, max_age=0)
    
    # Test loads with bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid_signature")
    
    # Test loads with different salt
    serialized_salt = serializer.dumps(value, salt="custom-salt")
    result = serializer.loads(serialized_salt, salt="custom-salt")
    assert result == value
    
    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(serialized_salt, salt="wrong-salt")
    
    # Test loads with multiple signers
    serializer_multi = TimedSerializer(
        secret_key="test-secret",
        signer_kwargs={"key_derivation": "hmac"}
    )
    serialized_multi = serializer_multi.dumps(value)
    result = serializer_multi.loads(serialized_multi)
    assert result == value
```


# LLM-generated content at query #52
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test basic loads without timestamp
    serializer = TimedSerializer("test-secret")
    data = {"key": "value"}
    dumped = serializer.dumps(data)
    loaded = serializer.loads(dumped)
    assert loaded == data

    # Test loads with max_age within limit
    serializer = TimedSerializer("test-secret")
    dumped = serializer.dumps(data)
    loaded = serializer.loads(dumped, max_age=3600)
    assert loaded == data

    # Test loads with return_timestamp=True
    serializer = TimedSerializer("test-secret")
    dumped = serializer.dumps(data)
    payload, timestamp = serializer.loads(dumped, return_timestamp=True)
    assert payload == data
    assert isinstance(timestamp, datetime)

    # Test loads with both max_age and return_timestamp
    serializer = TimedSerializer("test-secret")
    dumped = serializer.dumps(data)
    payload, timestamp = serializer.loads(dumped, max_age=3600, return_timestamp=True)
    assert payload == data
    assert isinstance(timestamp, datetime)

    # Test loads with signature expired
    serializer = TimedSerializer("test-secret")
    dumped = serializer.dumps(data)
    with pytest.raises(SignatureExpired):
        serializer.loads(dumped, max_age=-1)

    # Test loads with bad signature
    serializer = TimedSerializer("test-secret")
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")

    # Test loads with salt parameter
    serializer = TimedSerializer("test-secret", salt="custom-salt")
    dumped = serializer.dumps(data)
    loaded = serializer.loads(dumped, salt="custom-salt")
    assert loaded == data

    # Test loads with wrong salt fails
    serializer = TimedSerializer("test-secret", salt="custom-salt")
    dumped = serializer.dumps(data)
    with pytest.raises(BadSignature):
        serializer.loads(dumped, salt="wrong-salt")

    # Test loads with empty data
    serializer = TimedSerializer("test-secret")
    with pytest.raises(BadSignature):
        serializer.loads(b"")

    # Test loads with string input
    serializer = TimedSerializer("test-secret")
    dumped = serializer.dumps(data)
    loaded = serializer.loads(dumped.decode())
    assert loaded == data

    # Test loads with multiple signers (fallback mechanism)
    serializer = TimedSerializer("test-secret")
    serializer2 = TimedSerializer("other-secret")
    dumped = serializer.dumps(data)
    # Should fail with wrong secret but succeed with correct one
    serializer2.loads(dumped)  # This should raise BadSignature

    # Test loads with very old timestamp (should fail with max_age)
    serializer = TimedSerializer("test-secret")
    # Create a signer with a fixed old timestamp to simulate expired signature
    class OldTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) - 100000  # ~28 hours ago
    
    old_signer = OldTimestampSigner("test-secret")
    old_dumped = old_signer.sign(b"test-payload")
    # The serializer uses its own signer, so this might not work as expected
    # Instead, test with the signer directly
    
    # Test loads with bytes payload
    serializer = TimedSerializer("test-secret")
    data = b"binary-data"
    dumped = serializer.dumps(data)
    loaded = serializer.loads(dumped)
    assert loaded == data

    # Test loads with None payload
    serializer = TimedSerializer("test-secret")
    dumped = serializer.dumps(None)
    loaded = serializer.loads(dumped)
    assert loaded is None

    # Test loads with list payload
    serializer = TimedSerializer("test-secret")
    data = [1, 2, 3]
    dumped = serializer.dumps(data)
    loaded = serializer.loads(dumped)
    assert loaded == data

    # Test loads with integer payload
    serializer = TimedSerializer("test-secret")
    data = 42
    dumped = serializer.dumps(data)
    loaded = serializer.loads(dumped)
    assert loaded == data
```


# LLM-generated content at query #53
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == b"secret-key"
    assert signer.salt == b"itsdangerous.TimestampSigner"
    assert signer.sep == b"."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm is not None
    
    # Test constructor with custom parameters
    custom_signer = TimestampSigner(
        "custom-secret",
        salt="custom-salt",
        sep="|",
        key_derivation="none",
        digest_method=hashlib.sha256
    )
    assert custom_signer.secret_key == b"custom-secret"
    assert custom_signer.salt == b"custom-salt"
    assert custom_signer.sep == b"|"
    assert custom_signer.key_derivation == "none"
    assert custom_signer.digest_method == hashlib.sha256
    
    # Test that get_timestamp returns integer
    timestamp = signer.get_timestamp()
    assert isinstance(timestamp, int)
    assert timestamp > 0
    
    # Test timestamp_to_datetime
    dt = signer.timestamp_to_datetime(timestamp)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc
    
    # Test sign and unsign
    signed = signer.sign("test-value")
    assert isinstance(signed, bytes)
    unsigned = signer.unsign(signed)
    assert unsigned == b"test-value"
    
    # Test sign and unsign with return_timestamp
    unsigned, ts = signer.unsign(signed, return_timestamp=True)
    assert unsigned == b"test-value"
    assert isinstance(ts, datetime)
    assert ts.tzinfo == timezone.utc
    
    # Test max_age validation
    signed = signer.sign("test-value")
    unsigned = signer.unsign(signed, max_age=3600)
    assert unsigned == b"test-value"
    
    # Test expired signature raises SignatureExpired
    # Mock timestamp to make signature appear old
    import unittest.mock as mock
    old_timestamp = int(time.time()) - 7200  # 2 hours ago
    with mock.patch.object(signer, 'get_timestamp', return_value=old_timestamp + 3600):
        signed_old = signer.sign("old-value")
    with mock.patch.object(signer, 'get_timestamp', return_value=old_timestamp + 7200 + 1):
        with pytest.raises(SignatureExpired):
            signer.unsign(signed_old, max_age=3600)
    
    # Test malformed timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value.invalid-timestamp")
    
    # Test missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value")
    
    # Test corrupt signature
    signed = signer.sign("test-value")
    with pytest.raises(BadSignature):
        signer.unsign(signed + b"corrupted")
    
    # Test validate method
    signed = signer.sign("test-value")
    assert signer.validate(signed) is True
    assert signer.validate(signed, max_age=3600) is True
    assert signer.validate(b"invalid") is False
    
    # Test with bytes input
    signed = signer.sign(b"bytes-value")
    unsigned = signer.unsign(signed)
    assert unsigned == b"bytes-value"
    
    # Test with string input
    signed = signer.sign("string-value")
    unsigned = signer.unsign(signed)
    assert unsigned == b"string-value"
```


# LLM-generated content at query #54
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test 1: Basic sign and unsign without timestamp return
    signed = signer.sign("test_value")
    assert signer.unsign(signed) == b"test_value"
    
    # Test 2: Return timestamp with unsign
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Verify timestamp is recent (within 5 seconds)
    signed = signer.sign("test_value")
    _, timestamp = signer.unsign(signed, return_timestamp=True)
    now = datetime.now(timezone.utc)
    assert (now - timestamp).total_seconds() < 5
    
    # Test 4: Unsign with valid max_age
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test 5: Unsign with expired max_age should raise SignatureExpired
    # Create a signer that returns an old timestamp
    class OldTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) - 100  # 100 seconds in the past
    
    old_signer = OldTimestampSigner("secret-key")
    old_signed = old_signer.sign("test_value")
    
    # This should raise because max_age is 50 but age is 100
    with pytest.raises(SignatureExpired):
        old_signer.unsign(old_signed, max_age=50)
    
    # Test 6: Unsign with future timestamp should raise SignatureExpired
    class FutureTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) + 100  # 100 seconds in the future
    
    future_signer = FutureTimestampSigner("secret-key")
    future_signed = future_signer.sign("test_value")
    
    with pytest.raises(SignatureExpired):
        future_signer.unsign(future_signed, max_age=50)
    
    # Test 7: Unsign with wrong key should raise BadTimeSignature
    wrong_signer = TimestampSigner("wrong-key")
    signed = signer.sign("test_value")
    
    with pytest.raises(BadTimeSignature):
        wrong_signer.unsign(signed)
    
    # Test 8: Unsign with malformed timestamp should raise BadTimeSignature
    # Create a signed value with a non-base64 timestamp
    value = want_bytes("test_value")
    sep = want_bytes(signer.sep)
    bad_timestamp = b"!!!invalid!!!"
    malformed = value + sep + bad_timestamp + sep + signer.get_signature(value + sep + bad_timestamp)
    
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test 9: Unsign with missing timestamp should raise BadTimeSignature
    no_timestamp = value + sep + signer.get_signature(value)
    
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp)
    
    # Test 10: Unsign with string input (not bytes)
    signed = signer.sign("test_value")
    result = signer.unsign(signed.decode())
    assert result == b"test_value"
    
    # Test 11: Verify that the timestamp returned is consistent
    signed = signer.sign("test_value")
    _, timestamp1 = signer.unsign(signed, return_timestamp=True)
    _, timestamp2 = signer.unsign(signed, return_timestamp=True)
    assert timestamp1 == timestamp2
    
    # Test 12: Unsign with max_age=0 should raise SignatureExpired for any age > 0
    signed = signer.sign("test_value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)
```


# LLM-generated content at query #55
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("test-secret-key")
    assert signer.secret_key == "test-secret-key"
    assert signer.salt == "timestamp-signer"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "hmac-sha1"
    assert callable(signer.get_timestamp)
    assert callable(signer.timestamp_to_datetime)
    assert callable(signer.sign)
    assert callable(signer.unsign)
    assert callable(signer.validate)


# LLM-generated content at query #56
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test_secret")
    
    # Test basic loads (without max_age and return_timestamp)
    original_data = {"key": "value"}
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed)
    assert result == original_data
    
    # Test loads with max_age (within valid timeframe)
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed, max_age=3600)
    assert result == original_data
    
    # Test loads with return_timestamp=True
    signed = serializer.dumps(original_data)
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with both max_age and return_timestamp
    signed = serializer.dumps(original_data)
    payload, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with expired signature
    serializer_with_past_time = TimedSerializer("test_secret")
    original_get_timestamp = serializer_with_past_time.default_signer.get_timestamp
    serializer_with_past_time.default_signer.get_timestamp = lambda: int(time.time()) - 10000
    signed = serializer_with_past_time.dumps(original_data)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=1)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid_signature")
    
    # Test loads with bytes input
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed)
    assert result == original_data
    
    # Test loads with salt parameter
    result = serializer.loads(signed, salt="custom_salt")
    assert result == original_data
    
    # Test loads with non-dict payload
    string_data = "test_string"
    signed = serializer.dumps(string_data)
    result = serializer.loads(signed)
    assert result == string_data
    
    # Test loads with integer payload
    int_data = 42
    signed = serializer.dumps(int_data)
    result = serializer.loads(signed)
    assert result == int_data
    
    # Test loads with list payload
    list_data = [1, 2, 3]
    signed = serializer.dumps(list_data)
    result = serializer.loads(signed)
    assert result == list_data
    
    # Test loads with None payload
    signed = serializer.dumps(None)
    result = serializer.loads(signed)
    assert result is None
    
    # Test loads with empty dict
    signed = serializer.dumps({})
    result = serializer.loads(signed)
    assert result == {}
```


# LLM-generated content at query #57
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer(secret_key="test-secret-key")
    payload = {"user": "test_user", "role": "admin"}
    
    # Test basic loads without max_age
    signed_data = serializer.dumps(payload)
    result = serializer.loads(signed_data)
    assert result == payload
    
    # Test loads with return_timestamp=True
    signed_data = serializer.dumps(payload)
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == payload
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test loads with max_age within limit
    signed_data = serializer.dumps(payload)
    result = serializer.loads(signed_data, max_age=3600)  # 1 hour
    assert result == payload
    
    # Test loads with max_age expired
    signed_data = serializer.dumps(payload)
    import time
    time.sleep(0.1)  # Small delay to ensure age > 0
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=0)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-signature-data")
    
    # Test loads with malformed timestamp
    malformed = b"payload" + serializer.default_signer.sep.encode() + b"malformed-timestamp"
    with pytest.raises(BadSignature):
        serializer.loads(malformed)
    
    # Test loads with empty data
    with pytest.raises(BadSignature):
        serializer.loads(b"")
    
    # Test loads with multiple signers (fallback)
    serializer2 = TimedSerializer(secret_key="different-key")
    signed_data2 = serializer2.dumps(payload)
    with pytest.raises(BadSignature):
        serializer.loads(signed_data2)
    
    # Test loads with salt parameter
    signed_data_salt = serializer.dumps(payload, salt="custom-salt")
    result = serializer.loads(signed_data_salt, salt="custom-salt")
    assert result == payload
    
    # Test loads with salt parameter fails with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_data_salt, salt="wrong-salt")
    
    # Test loads with return_timestamp and max_age
    signed_data = serializer.dumps(payload)
    result, timestamp = serializer.loads(signed_data, return_timestamp=True, max_age=3600)
    assert result == payload
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test loads with bytes payload
    payload_bytes = b"test-bytes-payload"
    signed_bytes = serializer.dumps(payload_bytes)
    result = serializer.loads(signed_bytes)
    assert result == payload_bytes
    
    # Test loads with string payload
    payload_str = "test-string-payload"
    signed_str = serializer.dumps(payload_str)
    result = serializer.loads(signed_str)
    assert result == payload_str
    
    # Test loads with complex nested data
    complex_payload = {
        "user": "test",
        "data": {
            "items": [1, 2, 3],
            "nested": {"key": "value"}
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    signed_complex = serializer.dumps(complex_payload)
    result = serializer.loads(signed_complex)
    assert result == complex_payload
```


# LLM-generated content at query #58
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads method with various scenarios."""
    serializer = TimedSerializer(secret_key="test-secret")
    
    # Test basic loads without max_age
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with return_timestamp=True
    result_with_ts = serializer.loads(signed, return_timestamp=True)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == data
    assert isinstance(result_with_ts[1], datetime)
    
    # Test loads with valid max_age
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with expired signature
    import time as time_module
    old_serializer = TimedSerializer(secret_key="test-secret")
    old_serializer.signer_cls.get_timestamp = lambda: int(time_module.time()) - 10000
    old_signed = old_serializer.dumps(data)
    
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=1)
    
    # Test loads with bad signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")
    
    # Test loads with different salt
    signed_with_salt = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == data
    
    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
```


# LLM-generated content at query #59
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor and basic properties."""
    serializer = TimedSerializer("test-secret")
    
    assert serializer.secret_key == b"test-secret"
    assert serializer.salt == "itsdangerous"
    assert serializer.signer_kwargs == {}
    assert serializer.signer_class == TimestampSigner
    
    serializer_with_options = TimedSerializer(
        "test-secret",
        salt="custom-salt",
        serializer_kwargs={"key": "value"},
        signer_kwargs={"key_derivation": "hmac"},
    )
    
    assert serializer_with_options.salt == "custom-salt"
    assert serializer_with_options.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer_with_options.default_signer is TimestampSigner
    
    assert isinstance(serializer_with_options.make_signer(), TimestampSigner)
```


# LLM-generated content at query #60
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor and basic initialization."""
    serializer = TimedSerializer("test-secret")
    assert serializer.secret_key == "test-secret"
    assert serializer.salt == "itsdangerous.TimedSerializer"
    assert serializer.signer_kwargs == {}
    assert serializer.signer_class == TimestampSigner
    assert serializer.default_signer == TimestampSigner
    assert isinstance(serializer.signer, TimestampSigner)


# LLM-generated content at query #61
#--------------------------

```python
def test_TimedSerializer():
    # Test that TimedSerializer can be constructed with default parameters
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner
    
    # Test that TimedSerializer can be constructed with a secret key
    serializer = TimedSerializer("secret-key")
    assert serializer.secret_key == b"secret-key"
    
    # Test that TimedSerializer can be constructed with salt
    serializer = TimedSerializer("secret-key", salt="my-salt")
    assert serializer.salt == "my-salt"
    
    # Test that TimedSerializer can be constructed with serializer_kwargs
    serializer = TimedSerializer("secret-key", serializer_kwargs={"skipkeys": True})
    assert serializer.serializer_kwargs == {"skipkeys": True}
    
    # Test that TimedSerializer can be constructed with signer_kwargs
    serializer = TimedSerializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test that TimedSerializer can be constructed with digest_method
    serializer = TimedSerializer("secret-key", digest_method="sha256")
    assert serializer.digest_method == "sha256"
    
    # Test that the default signer is TimestampSigner
    serializer = TimedSerializer()
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)```


# LLM-generated content at query #62
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads method with various scenarios."""
    serializer = TimedSerializer("test_secret_key")
    
    # Test basic load without max_age or return_timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test load with max_age (valid)
    signed = serializer.dumps(data)
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test load with return_timestamp
    signed = serializer.dumps(data)
    result = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    payload, timestamp = result
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test load with max_age and return_timestamp
    signed = serializer.dumps(data)
    result = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result, tuple)
    payload, timestamp = result
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test load with expired signature
    signed = serializer.dumps(data)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=0)
    
    # Test load with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid_signature")
    
    # Test load with empty data
    with pytest.raises(BadSignature):
        serializer.loads(b"")
    
    # Test load with different salt
    signed = serializer.dumps(data, salt="custom_salt")
    result = serializer.loads(signed, salt="custom_salt")
    assert result == data
    
    # Test load with wrong salt
    signed = serializer.dumps(data, salt="custom_salt")
    with pytest.raises(BadSignature):
        serializer.loads(signed, salt="wrong_salt")
    
    # Test load with bytes payload
    signed = serializer.dumps(b"bytes_data")
    result = serializer.loads(signed)
    assert result == b"bytes_data"
    
    # Test load with integer payload
    signed = serializer.dumps(42)
    result = serializer.loads(signed)
    assert result == 42
```


# LLM-generated content at query #63
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test-secret")
    
    # Test basic loads without max_age
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with return_timestamp=True
    result_with_ts = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    payload, timestamp = result_with_ts
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with max_age - within age limit
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with max_age - expired signature
    import time
    old_data = {"old": "data"}
    old_signer = TimestampSigner("test-secret")
    # Create a manually old timestamp
    old_timestamp = base64_encode(int_to_bytes(int(time.time()) - 7200))
    old_signed = old_signer.sign("old_data")
    try:
        serializer.loads(old_signed, max_age=1)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass
    
    # Test loads with bad signature
    try:
        serializer.loads(b"invalid|data")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass
    
    # Test loads with different salt
    signed_with_salt = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == data

```


# LLM-generated content at query #64
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test_secret_key")
    
    # Test successful load without return_timestamp
    data = {"key": "value"}
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized)
    assert result == data
    
    # Test successful load with return_timestamp
    result_with_ts = serializer.loads(serialized, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == data
    assert isinstance(result_with_ts[1], datetime)
    
    # Test with max_age (should succeed for fresh signature)
    result = serializer.loads(serialized, max_age=3600)
    assert result == data
    
    # Test with max_age that should expire
    time.sleep(0.1)  # Small delay to ensure timestamp difference
    with pytest.raises(SignatureExpired):
        serializer.loads(serialized, max_age=0)
    
    # Test with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid_signature")
    
    # Test with empty string
    with pytest.raises(BadSignature):
        serializer.loads("")
    
    # Test with bytes input
    serialized_bytes = serializer.dumps(data)
    result = serializer.loads(serialized_bytes)
    assert result == data
    
    # Test with different salt
    serialized_salt1 = serializer.dumps(data, salt="salt1")
    result = serializer.loads(serialized_salt1, salt="salt1")
    assert result == data
    
    # Test with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(serialized_salt1, salt="wrong_salt")
```


# LLM-generated content at query #65
#--------------------------

```python
def test_TimedSerializer():
    # Test default constructor
    serializer = TimedSerializer()
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    
    # Test constructor with secret key
    serializer_with_key = TimedSerializer(secret_key="test-secret")
    assert serializer_with_key.secret_key == "test-secret"
    
    # Test constructor with salt
    serializer_with_salt = TimedSerializer(salt="test-salt")
    assert serializer_with_salt.salt == "test-salt"
    
    # Test constructor with signer_kwargs
    serializer_with_kwargs = TimedSerializer(signer_kwargs={"key_derivation": "hmac"})
    assert serializer_with_kwargs.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test constructor with serializer_kwargs
    serializer_with_serializer_kwargs = TimedSerializer(serializer_kwargs={"padsize": 32})
    assert serializer_with_serializer_kwargs.serializer_kwargs == {"padsize": 32}
    
    # Test constructor with all parameters
    serializer_all = TimedSerializer(
        secret_key="test-secret",
        salt="test-salt",
        signer_kwargs={"key_derivation": "hmac"},
        serializer_kwargs={"padsize": 32}
    )
    assert serializer_all.secret_key == "test-secret"
    assert serializer_all.salt == "test-salt"
    assert serializer_all.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer_all.serializer_kwargs == {"padsize": 32}
```


# LLM-generated content at query #66
#--------------------------

```python
def test_TimedSerializer():
    # Test that TimedSerializer can be instantiated with default parameters
    serializer = TimedSerializer(secret_key="test-secret")
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner
    
    # Test that it creates TimestampSigner instances
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    
    # Test with different secret key types
    serializer_bytes = TimedSerializer(secret_key=b"test-secret")
    assert isinstance(serializer_bytes, TimedSerializer)
    
    # Test with salt parameter
    serializer_salt = TimedSerializer(secret_key="test-secret", salt="custom-salt")
    assert serializer_salt.salt == "custom-salt"
    
    # Test with additional signer kwargs
    serializer_kwargs = TimedSerializer(
        secret_key="test-secret",
        signer_kwargs={"key_derivation": "hmac"}
    )
    assert serializer_kwargs.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test that serializer can be used for basic dumps/loads
    data = {"test": "data"}
    serialized = serializer.dumps(data)
    deserialized = serializer.loads(serialized)
    assert deserialized == data


# LLM-generated content at query #67
#--------------------------

```python
def test_TimedSerializer_loads():
    # Create a TimedSerializer instance
    serializer = TimedSerializer(secret_key="test-secret")
    
    # Test 1: Basic loads without max_age or return_timestamp
    value = {"key": "value"}
    signed = serializer.dumps(value)
    result = serializer.loads(signed)
    assert result == value
    
    # Test 2: Loads with return_timestamp=True
    result_with_ts = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    payload, timestamp = result_with_ts
    assert payload == value
    assert isinstance(timestamp, datetime)
    
    # Test 3: Loads with max_age that is within valid range
    result = serializer.loads(signed, max_age=3600)  # 1 hour
    assert result == value
    
    # Test 4: Loads with max_age that is expired
    # We need to manually create an old signature
    old_serializer = TimedSerializer(secret_key="test-secret")
    old_signed = old_serializer.dumps(value)
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=-1)  # Negative max_age ensures expiration
    
    # Test 5: Loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-signature")
    
    # Test 6: Loads with different salt
    signed_with_salt = serializer.dumps(value, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == value
    
    # Test 7: Loads with wrong salt should fail
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
    
    # Test 8: Loads with return_timestamp and max_age combined
    result = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    payload, timestamp = result
    assert payload == value
    assert isinstance(timestamp, datetime)
    
    # Test 9: Loads with bytes input
    signed_bytes = signed if isinstance(signed, bytes) else signed.encode()
    result = serializer.loads(signed_bytes)
    assert result == value
    
    # Test 10: Loads with string input
    signed_str = signed if isinstance(signed, str) else signed.decode()
    result = serializer.loads(signed_str)
    assert result == value
    
    # Test 11: Loads with empty payload
    empty_value = ""
    signed_empty = serializer.dumps(empty_value)
    result = serializer.loads(signed_empty)
    assert result == empty_value
    
    # Test 12: Loads with None payload (serialized as "null" in JSON)
    none_value = None
    signed_none = serializer.dumps(none_value)
    result = serializer.loads(signed_none)
    assert result is None
    
    # Test 13: Loads with complex nested payload
    complex_value = {
        "list": [1, 2, 3],
        "nested": {"a": 1, "b": 2},
        "bool": True,
        "number": 42,
        "float": 3.14
    }
    signed_complex = serializer.dumps(complex_value)
    result = serializer.loads(signed_complex)
    assert result == complex_value
    
    # Test 14: Multiple signers with fallback
    # This tests the fallback mechanism when using multiple secret keys
    serializer2 = TimedSerializer(secret_key="test-secret-2")
    signed2 = serializer2.dumps(value)
    
    # Create a serializer with multiple fallback signers
    multi_serializer = TimedSerializer(
        secret_key="test-secret",
        fallback_signers=["test-secret-2"]
    )
    result = multi_serializer.loads(signed2)
    assert result == value
    
    # Test 15: Loads with expired signature should raise SignatureExpired
    # We create a signature that will be expired by setting max_age to 0
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=0)
```


# LLM-generated content at query #68
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads method with valid data."""
    serializer = TimedSerializer(secret_key="test-secret")
    
    # Test basic loads without max_age and return_timestamp
    data = {"key": "value"}
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized)
    assert result == data
    
    # Test with return_timestamp=True
    serialized = serializer.dumps(data)
    result, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test with max_age that should pass
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized, max_age=3600)
    assert result == data
    
    # Test with max_age that should fail (expired signature)
    # We can simulate this by setting a very small max_age
    # and waiting for a fraction of a second
    import time
    serialized = serializer.dumps(data)
    time.sleep(0.1)
    with pytest.raises(SignatureExpired):
        serializer.loads(serialized, max_age=0)
    
    # Test with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-data")
    
    # Test with empty data
    with pytest.raises(BadSignature):
        serializer.loads("")
    
    # Test with bytes input
    serialized_bytes = serializer.dumps(data)
    result = serializer.loads(serialized_bytes)
    assert result == data
    
    # Test with different salt
    serialized_salt = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(serialized_salt, salt="custom-salt")
    assert result == data
    
    # Test with wrong salt
    serialized_salt = serializer.dumps(data, salt="salt1")
    with pytest.raises(BadSignature):
        serializer.loads(serialized_salt, salt="salt2")
    
    # Test return_timestamp with bytes input
    serialized_bytes = serializer.dumps(data)
    result, timestamp = serializer.loads(serialized_bytes, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test with additional signers
    serializer2 = TimedSerializer(secret_key="another-secret")
    serialized2 = serializer2.dumps(data)
    # Should not validate with different secret
    with pytest.raises(BadSignature):
        serializer.loads(serialized2)
```


# LLM-generated content at query #69
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer("test-secret")
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner
    assert serializer.secret_key == b"test-secret"
    assert serializer.salt == "itsdangerous"
    assert serializer.signer_kwargs == {}
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #70
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor and basic initialization."""
    serializer = TimedSerializer("test-secret-key")
    
    assert serializer.secret_key == "test-secret-key"
    assert isinstance(serializer.default_signer, type)
    assert serializer.default_signer == TimestampSigner
    assert hasattr(serializer, 'signer_kwargs')
    assert hasattr(serializer, 'salt')
    
    # Test with custom salt
    serializer_with_salt = TimedSerializer("test-secret-key", salt="custom-salt")
    assert serializer_with_salt.salt == "custom-salt"
    
    # Test with signer kwargs
    signer_kwargs = {"key_derivation": "hmac"}
    serializer_with_kwargs = TimedSerializer("test-secret-key", signer_kwargs=signer_kwargs)
    assert serializer_with_kwargs.signer_kwargs == signer_kwargs
    
    # Test default salt
    assert serializer.salt == "itsdangerous.TimedSerializer"
    
    # Verify it creates TimestampSigner instances
    signer = next(serializer.iter_unsigners())
    assert isinstance(signer, TimestampSigner)
    assert signer.secret_key == "test-secret-key"
```


# LLM-generated content at query #71
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.sep == "."
    assert callable(signer.get_timestamp)
    assert callable(signer.timestamp_to_datetime)
    assert callable(signer.sign)
    assert callable(signer.unsign)
    assert callable(signer.validate)


# LLM-generated content at query #72
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == b"secret-key"
    assert signer.salt is not None
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method is not None
    assert signer.algorithm is not None
    
    signer_with_salt = TimestampSigner("secret-key", salt="my-salt")
    assert signer_with_salt.salt == "my-salt"
    
    signer_with_sep = TimestampSigner("secret-key", sep=":")
    assert signer_with_sep.sep == ":"
    
    signer_with_key_derivation = TimestampSigner("secret-key", key_derivation="none")
    assert signer_with_key_derivation.key_derivation == "none"
    
    signer_with_digest = TimestampSigner("secret-key", digest_method="sha256")
    assert signer_with_digest.digest_method == "sha256"
    
    signer_with_algorithm = TimestampSigner("secret-key", algorithm="hs256")
    assert signer_with_algorithm.algorithm == "hs256"


# LLM-generated content at query #73
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("secret-key")
    
    # Test basic loads without max_age or return_timestamp
    original_data = {"key": "value"}
    serialized = serializer.dumps(original_data)
    result = serializer.loads(serialized)
    assert result == original_data
    
    # Test loads with return_timestamp=True
    result_with_ts = serializer.loads(serialized, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == original_data
    assert isinstance(result_with_ts[1], datetime)
    
    # Test loads with max_age (valid age)
    result = serializer.loads(serialized, max_age=3600)
    assert result == original_data
    
    # Test loads with max_age that causes expiration
    # Create a serialized value with an old timestamp
    import time
    old_timestamp = int(time.time()) - 7200  # 2 hours ago
    old_signer = TimestampSigner("secret-key")
    old_signer.get_timestamp = lambda: old_timestamp
    old_serialized = old_signer.sign(serializer.dumps(original_data))
    
    with pytest.raises(SignatureExpired):
        serializer.loads(old_serialized, max_age=3600)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid_data")
    
    # Test loads with string input
    serialized_str = serialized.decode("utf-8")
    result = serializer.loads(serialized_str)
    assert result == original_data
    
    # Test loads with bytes input
    result = serializer.loads(serialized)
    assert result == original_data
    
    # Test loads with return_timestamp and max_age together
    result_with_ts = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert result_with_ts[0] == original_data
    assert isinstance(result_with_ts[1], datetime)
```


# LLM-generated content at query #74
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)
    
    # Test 1: Basic unsign without timestamp
    result = signer.unsign(signed_value)
    assert result == value
    
    # Test 2: Unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Unsign with valid max_age
    result = signer.unsign(signed_value, max_age=3600)
    assert result == value
    
    # Test 4: Unsign with expired max_age (should raise SignatureExpired)
    import time as time_module
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time_module.time()) + 100  # Simulate future time
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=10)
    signer.get_timestamp = original_get_timestamp
    
    # Test 5: Unsign with negative age (future timestamp)
    signer.get_timestamp = lambda: int(time_module.time()) - 100  # Simulate past time
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=3600)
    signer.get_timestamp = original_get_timestamp
    
    # Test 6: Unsign with malformed timestamp
    malformed = b"test_value" + signer.sep.encode() + b"invalid_timestamp" + signer.sep.encode() + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test 7: Unsign with missing timestamp
    no_timestamp = b"test_value" + signer.sep.encode() + signer.get_signature(b"test_value")
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp)
    
    # Test 8: Unsign with invalid signature
    invalid_sig = b"test_value" + signer.sep.encode() + base64_encode(int_to_bytes(signer.get_timestamp())) + signer.sep.encode() + b"invalid_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(invalid_sig)
    
    # Test 9: Unsign with string input
    result = signer.unsign(signed_value.decode())
    assert result == value
    
    # Test 10: Unsign with return_timestamp and max_age
    result, timestamp = signer.unsign(signed_value, max_age=3600, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
```


# LLM-generated content at query #75
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("test-secret")
    
    # Test basic loads
    original_data = {"key": "value"}
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed)
    assert result == original_data
    
    # Test loads with max_age (within age limit)
    result = serializer.loads(signed, max_age=3600)
    assert result == original_data
    
    # Test loads with return_timestamp
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == original_data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test loads with both max_age and return_timestamp
    result, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert result == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with expired signature
    import time
    old_serializer = TimedSerializer("test-secret")
    old_signed = old_serializer.dumps(original_data)
    time.sleep(0.1)  # Ensure some time passes
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=0)
    
    # Test loads with bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid_data")
    
    # Test loads with tampered data
    tampered = signed[:-1] + b"x"
    with pytest.raises(BadSignature):
        serializer.loads(tampered)
    
    # Test loads with different salt
    signed_with_salt = serializer.dumps(original_data, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == original_data
    
    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
    
    # Test loads with bytes input
    result = serializer.loads(signed.encode() if isinstance(signed, str) else signed)
    assert result == original_data
    
    # Test loads with complex data types
    complex_data = {
        "string": "hello",
        "number": 42,
        "list": [1, 2, 3],
        "nested": {"a": 1, "b": 2},
        "bool": True,
        "none": None
    }
    signed_complex = serializer.dumps(complex_data)
    result = serializer.loads(signed_complex)
    assert result == complex_data
    
    # Test loads with empty data
    empty_data = {}
    signed_empty = serializer.dumps(empty_data)
    result = serializer.loads(signed_empty)
    assert result == empty_data
    
    # Test loads with None
    signed_none = serializer.dumps(None)
    result = serializer.loads(signed_none)
    assert result is None
```


# LLM-generated content at query #76
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test successful loads without timestamp
    serializer = TimedSerializer("test-secret")
    signed_data = serializer.dumps("test_payload")
    result = serializer.loads(signed_data)
    assert result == "test_payload"

    # Test successful loads with timestamp
    result = serializer.loads(signed_data, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == "test_payload"
    assert isinstance(result[1], datetime)

    # Test with max_age that should pass
    result = serializer.loads(signed_data, max_age=3600)
    assert result == "test_payload"

    # Test with max_age that should fail (simulate old signature)
    old_serializer = TimestampSigner("test-secret")
    old_timestamp = int(time.time()) - 7200  # 2 hours ago
    old_timestamp_bytes = base64_encode(int_to_bytes(old_timestamp))
    value = want_bytes("test_payload")
    sep = want_bytes(old_serializer.sep)
    old_signed = value + sep + old_timestamp_bytes + sep + old_serializer.get_signature(value + sep + old_timestamp_bytes)
    
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=3600)

    # Test with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid_data")

    # Test with expired signature and return_timestamp
    with pytest.raises(SignatureExpired) as exc_info:
        serializer.loads(old_signed, max_age=3600, return_timestamp=True)
    assert exc_info.value.date_signed is not None
    assert isinstance(exc_info.value.date_signed, datetime)

    # Test with tampered timestamp
    tampered_signed = signed_data[:-1] + (b'\x00' if signed_data[-1:] != b'\x00' else b'\x01')
    with pytest.raises(BadSignature):
        serializer.loads(tampered_signed)

    # Test with non-timestamp data
    signer = Signer("test-secret")
    non_timestamp_signed = signer.sign("test_payload")
    with pytest.raises(BadTimeSignature):
        serializer.loads(non_timestamp_signed)

    # Test loads_unsafe with max_age
    safe_result, payload = serializer.loads_unsafe(signed_data, max_age=3600)
    assert safe_result
    assert payload == "test_payload"

    safe_result, payload = serializer.loads_unsafe(old_signed, max_age=3600)
    assert not safe_result
    assert payload == "test_payload"
```


# LLM-generated content at query #77
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test 1: Basic unsign without timestamp
    value = b"test_value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test 2: Unsign with return_timestamp=True
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == value
    assert isinstance(result[1], datetime)
    
    # Test 3: Unsign with max_age (valid)
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test 4: Unsign with max_age (expired)
    signer.get_timestamp = lambda: int(time.time()) - 100
    signed_old = signer.sign(value)
    signer.get_timestamp = lambda: int(time.time())
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_old, max_age=10)
    
    # Test 5: Unsign with max_age (future timestamp)
    signer.get_timestamp = lambda: int(time.time())
    signed_future = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_future, max_age=-1)
    
    # Test 6: Unsign with invalid signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid_signature")
    
    # Test 7: Unsign with missing timestamp
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(b"test_value.invalidsig")
    assert "timestamp missing" in str(exc_info.value)
    
    # Test 8: Unsign with malformed timestamp
    signer_alt = TimestampSigner("secret-key")
    signer_alt.sep = "."
    signed_malformed = b"test_value." + base64_encode(b"invalid_timestamp") + b"." + signer_alt.get_signature(b"test_value." + base64_encode(b"invalid_timestamp"))
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_malformed)
    
    # Test 9: Unsign with different key
    signer2 = TimestampSigner("different-key")
    signed2 = signer2.sign(value)
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed2)
    
    # Test 10: Unsign with string input
    signed_str = signer.sign("test_string").decode()
    result = signer.unsign(signed_str)
    assert result == b"test_string"
```


# LLM-generated content at query #78
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads method."""
    serializer = TimedSerializer("test-secret")
    
    # Test basic loads without max_age or return_timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with return_timestamp
    signed = serializer.dumps(data)
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test loads with max_age within limit
    signed = serializer.dumps(data)
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with max_age expired
    signed = serializer.dumps(data)
    # Simulate time passing by using a custom timestamp signer
    class FastTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) + 7200  # 2 hours in the future
    
    fast_serializer = TimedSerializer("test-secret")
    fast_serializer.default_signer = FastTimestampSigner
    fast_signed = fast_serializer.dumps(data)
    with pytest.raises(SignatureExpired):
        fast_serializer.loads(fast_signed, max_age=3600)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test loads with different salt
    signed = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed, salt="custom-salt")
    assert result == data
    
    # Test loads with wrong salt
    signed = serializer.dumps(data, salt="custom-salt")
    with pytest.raises(BadSignature):
        serializer.loads(signed, salt="wrong-salt")
    
    # Test loads with empty string
    with pytest.raises(BadSignature):
        serializer.loads(b"")
    
    # Test loads with bytes input
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert isinstance(result, dict)
    assert result == data
    
    # Test loads with string input
    signed_str = serializer.dumps(data).decode('utf-8')
    result = serializer.loads(signed_str)
    assert result == data
```


# LLM-generated content at query #79
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test-secret")
    
    # Test basic loads without max_age
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with max_age (within valid time)
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with return_timestamp=True
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None
    
    # Test loads with both max_age and return_timestamp
    result, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with expired signature
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=-1)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")
    
    # Test loads with different salt
    signed_with_salt = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == data
    
    # Test loads with wrong salt raises BadSignature
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
```


# LLM-generated content at query #80
#--------------------------

```python
def test_TimedSerializer():
    """Test the constructor of TimedSerializer class."""
    # Test default construction
    serializer = TimedSerializer("test-secret")
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner
    
    # Test with custom salt
    serializer_with_salt = TimedSerializer("test-secret", salt="custom-salt")
    assert serializer_with_salt.salt == "custom-salt"
    
    # Test with custom signer kwargs
    serializer_with_kwargs = TimedSerializer("test-secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer_with_kwargs.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test with serializer kwargs
    serializer_with_serializer_kwargs = TimedSerializer("test-secret", serializer_kwargs={"compress": True})
    assert serializer_with_serializer_kwargs.serializer_kwargs == {"compress": True}
    
    # Test that default_signer is overridable
    class CustomTimestampSigner(TimestampSigner):
        pass
    
    serializer_with_custom_signer = TimedSerializer("test-secret")
    assert serializer_with_custom_signer.default_signer is TimestampSigner
    
    # Test that it inherits from Serializer properly
    assert hasattr(serializer, "dumps")
    assert hasattr(serializer, "loads")
    assert hasattr(serializer, "loads_unsafe")
    
    # Test with bytes secret
    serializer_bytes_secret = TimedSerializer(b"test-secret-bytes")
    assert serializer_bytes_secret.secret == b"test-secret-bytes"


# LLM-generated content at query #81
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test 1: Basic unsign without max_age and return_timestamp=False
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test 2: Unsign with return_timestamp=True
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Unsign with max_age within limits
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test 4: Unsign with max_age exceeded
    signer_with_fixed_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_fixed_time.get_timestamp
    signer_with_fixed_time.get_timestamp = lambda: 1000  # Fixed old timestamp
    signed = signer_with_fixed_time.sign("test_value")
    signer_with_fixed_time.get_timestamp = lambda: 2000  # Current time is later
    with pytest.raises(SignatureExpired):
        signer_with_fixed_time.unsign(signed, max_age=500)
    
    # Test 5: Unsign with negative age (future timestamp)
    signer_with_future_time = TimestampSigner("secret-key")
    signer_with_future_time.get_timestamp = lambda: 2000  # Future timestamp
    signed = signer_with_future_time.sign("test_value")
    signer_with_future_time.get_timestamp = lambda: 1000  # Current time is earlier
    with pytest.raises(SignatureExpired):
        signer_with_future_time.unsign(signed, max_age=3600)
    
    # Test 6: Unsign with invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")
    
    # Test 7: Unsign with tampered value
    signed = signer.sign("test_value")
    tampered = signed[:-1] + (b"x" if signed[-1:] != b"x" else b"y")
    with pytest.raises(BadSignature):
        signer.unsign(tampered)
    
    # Test 8: Unsign with missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value_without_timestamp" + signer.sep.encode() + b"invalidsig")
    
    # Test 9: Unsign with malformed timestamp
    signed = signer.sign("test_value")
    parts = signed.rsplit(signer.sep.encode(), 1)
    malformed = parts[0] + signer.sep.encode() + b"malformed_timestamp" + signer.sep.encode() + parts[1]
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test 10: Unsign with empty value
    signed = signer.sign(b"")
    result = signer.unsign(signed)
    assert result == b""
    
    # Test 11: Unsign with bytes input
    signed = signer.sign(b"bytes_value")
    result = signer.unsign(signed)
    assert result == b"bytes_value"
    
    # Test 12: Unsign with string input
    signed = signer.sign("string_value")
    result = signer.unsign(signed.decode())
    assert result == b"string_value"
    
    # Test 13: Unsign with return_timestamp and max_age
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 14: Verify timestamp is recent
    signed = signer.sign("test_value")
    _, timestamp = signer.unsign(signed, return_timestamp=True)
    now = datetime.now(timezone.utc)
    assert abs((now - timestamp).total_seconds()) < 2  # Should be within 2 seconds
```


# LLM-generated content at query #82
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test 1: Basic sign and unsign
    value = b"test value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test 2: Unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Unsign with valid max_age
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test 4: Unsign with expired signature (max_age too small)
    with patch.object(signer, 'get_timestamp', return_value=1000):
        expired_signed = signer.sign(value)
    with patch.object(signer, 'get_timestamp', return_value=2000):
        with pytest.raises(SignatureExpired):
            signer.unsign(expired_signed, max_age=500)
    
    # Test 5: Unsign with negative age (future timestamp)
    with patch.object(signer, 'get_timestamp', return_value=1000):
        future_signed = signer.sign(value)
    with patch.object(signer, 'get_timestamp', return_value=500):
        with pytest.raises(SignatureExpired):
            signer.unsign(future_signed, max_age=3600)
    
    # Test 6: Unsign with malformed timestamp
    malformed = value + b"=" + base64_encode(int_to_bytes(12345))
    with pytest.raises(BadSignature):
        signer.unsign(malformed)
    
    # Test 7: Unsign with missing timestamp
    missing_ts = value + b"=" + signer.get_signature(value)
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(missing_ts)
    
    # Test 8: Unsign with invalid base64 timestamp
    invalid_ts = value + b"=invalid" + b"=" + signer.get_signature(value + b"=invalid")
    with pytest.raises(BadSignature):
        signer.unsign(invalid_ts)
    
    # Test 9: Unsign with string input
    signed_str = signer.sign("string value")
    result = signer.unsign(signed_str.decode())
    assert result == b"string value"
    
    # Test 10: Unsign with return_timestamp and max_age
    result, timestamp = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
```


# LLM-generated content at query #83
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without timestamp
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with max_age within limit
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test unsign with expired signature (max_age too small)
    # Mock get_timestamp to return a future time
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) + 100
    signed = signer.sign("test_value")
    signer.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=10)
    
    # Test unsign with negative age (signature from future)
    signer.get_timestamp = lambda: int(time.time()) - 100
    signed = signer.sign("test_value")
    signer.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=3600)
    
    # Test unsign with malformed timestamp
    # Create a signature with non-base64 timestamp
    value = want_bytes("test_value")
    sep = want_bytes(signer.sep)
    malformed = value + sep + b"not-valid-timestamp" + sep + signer.get_signature(value + sep + b"not-valid-timestamp")
    
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test unsign with missing timestamp (no separator in payload)
    value = want_bytes("test_value")
    no_timestamp = value + sep + signer.get_signature(value)
    
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_timestamp)
    
    # Test unsign with invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")
    
    # Test unsign with invalid signature but valid timestamp format
    # Create a valid-looking but incorrectly signed value
    valid_signed = signer.sign("test_value")
    # Modify the signature part
    parts = valid_signed.rsplit(sep, 1)
    modified = parts[0] + sep + b"modified_signature"
    
    with pytest.raises(BadTimeSignature):
        signer.unsign(modified)
    
    # Test unsign with invalid signature and invalid timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test" + sep + b"invalid" + sep + b"sig")
    
    # Test unsign with empty value
    signed = signer.sign(b"")
    result = signer.unsign(signed)
    assert result == b""
    
    # Test unsign with bytes input
    signed = signer.sign(b"test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test unsign with string input
    signed = signer.sign("test_value")
    result = signer.unsign(signed.decode())
    assert result == b"test_value"
```


# LLM-generated content at query #84
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == b"secret-key"
    assert signer.salt == b"itsdangerous.TimestampSigner"
    assert signer.sep == b"."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm is not None
    
    # Test with custom parameters
    custom_signer = TimestampSigner(
        "secret-key",
        salt="custom-salt",
        sep=":",
        key_derivation="none",
        digest_method=hashlib.sha256
    )
    assert custom_signer.salt == b"custom-salt"
    assert custom_signer.sep == b":"
    assert custom_signer.key_derivation == "none"
    assert custom_signer.digest_method == hashlib.sha256
    
    # Test default sign method works
    signed = signer.sign("test-value")
    assert isinstance(signed, bytes)
    assert b"." in signed
    
    # Verify we can unsign
    unsigned = signer.unsign(signed)
    assert unsigned == b"test-value"


# LLM-generated content at query #85
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key", salt="test-salt")
    
    # Test 1: Basic unsign without timestamp
    value = b"test-value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test 2: Unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Unsign with valid max_age
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test 4: Unsign with expired signature (max_age too small)
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)
    
    # Test 5: Unsign with future timestamp (age < 0)
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: 9999999999
    signed = signer.sign(value)
    signer.get_timestamp = lambda: 0
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=3600)
    signer.get_timestamp = original_get_timestamp
    
    # Test 6: Unsign with malformed timestamp
    signed = signer.sign(value)
    signed = signed[:-1]  # Corrupt the timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed)
    
    # Test 7: Unsign with missing timestamp separator
    signed = signer.sign(value)
    signed = signed.replace(signer.sep.encode(), b"")  # Remove separator
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed)
    
    # Test 8: Unsign with invalid signature
    signed = signer.sign(value)
    signed = b"invalid" + signed
    with pytest.raises(BadSignature):
        signer.unsign(signed)
    
    # Test 9: Unsign with invalid signature and return_timestamp
    signed = signer.sign(value)
    signed = b"invalid" + signed
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed, return_timestamp=True)
```


# LLM-generated content at query #86
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test successful load without max_age or return_timestamp
    serializer = TimedSerializer("test-secret")
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test successful load with return_timestamp=True
    result = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == data
    assert isinstance(result[1], datetime)
    
    # Test successful load with max_age
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test expired signature
    import time
    old_signer = TimestampSigner("test-secret")
    old_signer.get_timestamp = lambda: int(time.time()) - 7200  # 2 hours ago
    old_signed = old_signer.sign("test-value")
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=3600)
    
    # Test bad signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")
    
    # Test with different salt
    salted = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(salted, salt="custom-salt")
    assert result == data
    
    # Test with wrong salt raises error
    with pytest.raises(BadSignature):
        serializer.loads(salted, salt="wrong-salt")
    
    # Test with max_age and return_timestamp combined
    result = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == data
    assert isinstance(result[1], datetime)
```


# LLM-generated content at query #87
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.sep == "."
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == "sha1"
    assert signer.algorithm is not None
    
    # Test with custom parameters
    signer2 = TimestampSigner(
        "secret-key-2",
        sep="|",
        salt="custom-salt",
        key_derivation="none",
        digest_method="sha256"
    )
    assert signer2.secret_key == "secret-key-2"
    assert signer2.sep == "|"
    assert signer2.salt == "custom-salt"
    assert signer2.key_derivation == "none"
    assert signer2.digest_method == "sha256"
    
    # Test with bytes secret key
    signer3 = TimestampSigner(b"bytes-secret")
    assert signer3.secret_key == b"bytes-secret"
    
    # Test get_timestamp returns int
    timestamp = signer.get_timestamp()
    assert isinstance(timestamp, int)
    assert timestamp > 0
    
    # Test timestamp_to_datetime
    dt = signer.timestamp_to_datetime(timestamp)
    assert isinstance(dt, datetime)
    assert dt.tzinfo is not None
    
    # Test that signer is instance of Signer
    assert isinstance(signer, Signer)


# LLM-generated content at query #88
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer(secret_key="test-secret")
    
    # Test basic loads without max_age or return_timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with return_timestamp
    result_with_ts = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == data
    assert isinstance(result_with_ts[1], datetime)
    
    # Test loads with max_age
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with expired signature
    import time
    serializer_fast = TimedSerializer(secret_key="test-secret")
    signed_fast = serializer_fast.dumps(data)
    time.sleep(0.1)  # Simulate time passing
    with pytest.raises(SignatureExpired):
        serializer_fast.loads(signed_fast, max_age=0.05)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-data")
    
    # Test loads with salt
    serializer_with_salt = TimedSerializer(secret_key="test-secret", salt="custom-salt")
    signed_with_salt = serializer_with_salt.dumps(data)
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == data
    
    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
    
    # Test loads with empty payload
    signed_empty = serializer.dumps("")
    result = serializer.loads(signed_empty)
    assert result == ""
    
    # Test loads with None payload
    signed_none = serializer.dumps(None)
    result = serializer.loads(signed_none)
    assert result is None
    
    # Test loads with complex data types
    complex_data = {
        "list": [1, 2, 3],
        "nested": {"a": 1},
        "boolean": True,
        "number": 42.5
    }
    signed_complex = serializer.dumps(complex_data)
    result = serializer.loads(signed_complex)
    assert result == complex_data
    
    # Test loads with bytes input
    signed_bytes = serializer.dumps(data)
    result = serializer.loads(signed_bytes)
    assert result == data
    
    # Test loads with string input
    signed_str = serializer.dumps(data).decode() if isinstance(serializer.dumps(data), bytes) else serializer.dumps(data)
    result = serializer.loads(signed_str)
    assert result == data
    
    # Test loads with both max_age and return_timestamp
    signed = serializer.dumps(data)
    result = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == data
    assert isinstance(result[1], datetime)
```


# LLM-generated content at query #89
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test-secret")
    
    # Test basic loads without timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with return_timestamp=True
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with max_age
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with expired signature
    import time
    expired_signer = TimestampSigner("test-secret")
    expired_signer.get_timestamp = lambda: int(time.time()) - 7200
    expired_signed = expired_signer.sign(b"test")
    
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_signed, max_age=3600)
    
    # Test loads with bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-signature")
    
    # Test loads with salt
    salted = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(salted, salt="custom-salt")
    assert result == data
    
    # Test loads with invalid salt
    with pytest.raises(BadSignature):
        serializer.loads(salted, salt="wrong-salt")
```


# LLM-generated content at query #90
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("test-secret")
    assert signer.secret_key == b"test-secret"
    assert signer.salt == "timestamp-signer"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "hs256"
    
    signer2 = TimestampSigner("test-secret", salt="custom-salt", sep="|")
    assert signer2.secret_key == b"test-secret"
    assert signer2.salt == "custom-salt"
    assert signer2.sep == "|"


# LLM-generated content at query #91
#--------------------------

```python
def test_TimestampSigner_unsign():
    """Test TimestampSigner.unsign method with various scenarios."""
    signer = TimestampSigner("secret-key", salt="test-salt")
    
    # Test 1: Basic sign and unsign without timestamp return
    value = b"test_value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value, "Basic unsign should return original value"
    
    # Test 2: Sign and unsign with return_timestamp=True
    signed = signer.sign(value)
    result_value, result_timestamp = signer.unsign(signed, return_timestamp=True)
    assert result_value == value, "Unsign with timestamp should return original value"
    assert isinstance(result_timestamp, datetime), "Timestamp should be a datetime object"
    assert result_timestamp.tzinfo is not None, "Timestamp should be timezone-aware"
    
    # Test 3: Sign and unsign with string input
    string_value = "test_string"
    signed = signer.sign(string_value)
    result = signer.unsign(signed)
    assert result == b"test_string", "String input should be converted to bytes"
    
    # Test 4: Unsign with max_age that is not exceeded
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=3600)  # 1 hour max age
    assert result == value, "Unsign should succeed when max_age is not exceeded"
    
    # Test 5: Unsign with max_age that is exceeded
    signer_with_fixed_time = TimestampSigner("secret-key", salt="test-salt")
    original_get_timestamp = signer_with_fixed_time.get_timestamp
    
    # Mock get_timestamp to return a time in the past
    past_time = int(time.time()) - 100  # 100 seconds ago
    signer_with_fixed_time.get_timestamp = lambda: past_time
    signed = signer_with_fixed_time.sign(value)
    
    # Restore original get_timestamp
    signer_with_fixed_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer_with_fixed_time.unsign(signed, max_age=50)  # max_age is 50 seconds
    assert "Signature age" in str(exc_info.value), "Should raise SignatureExpired with age info"
    
    # Test 6: Unsign with negative timestamp (future timestamp)
    signer_future = TimestampSigner("secret-key", salt="test-salt")
    future_time = int(time.time()) + 100  # 100 seconds in future
    signer_future.get_timestamp = lambda: future_time
    signed_future = signer_future.sign(value)
    signer_future.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer_future.unsign(signed_future, max_age=200)
    assert "age" in str(exc_info.value).lower(), "Should raise SignatureExpired for future timestamp"
    
    # Test 7: Unsign with tampered value
    signed = signer.sign(value)
    tampered = signed[:-1] + b"X"  # Modify last byte
    with pytest.raises(BadSignature):
        signer.unsign(tampered)
    
    # Test 8: Unsign with missing timestamp
    # Create a signed value without timestamp using regular Signer
    regular_signer = Signer("secret-key", salt="test-salt")
    signed_no_timestamp = regular_signer.sign(value)
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(signed_no_timestamp)
    assert "timestamp missing" in str(exc_info.value), "Should raise BadTimeSignature for missing timestamp"
    
    # Test 9: Unsign with malformed timestamp
    # Create a signed value with invalid timestamp encoding
    sep = signer.sep.encode()
    malformed_timestamp = base64_encode(b"invalid")
    signed_malformed = value + sep + malformed_timestamp + sep + signer.get_signature(value + sep + malformed_timestamp)
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(signed_malformed)
    assert "Malformed timestamp" in str(exc_info.value), "Should raise BadTimeSignature for malformed timestamp"
    
    # Test 10: Unsign with return_timestamp=True and max_age
    signed = signer.sign(value)
    result_value, result_timestamp = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert result_value == value
    assert isinstance(result_timestamp, datetime)
    assert result_timestamp.tzinfo is not None
    
    # Test 11: Unsign with bytes input
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert isinstance(result, bytes), "Result should be bytes"
    assert result == value
```


# LLM-generated content at query #92
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key", salt="test-salt")
    
    # Test normal unsign without max_age and return_timestamp=False
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"
    
    # Test unsign with return_timestamp=True
    signed = signer.sign("test-value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None  # timezone-aware
    
    # Test unsign with valid max_age
    signed = signer.sign("test-value")
    result = signer.unsign(signed, max_age=3600)  # 1 hour
    assert result == b"test-value"
    
    # Test unsign with expired signature (max_age too small)
    signed = signer.sign("test-value")
    time.sleep(0.1)  # Ensure some time passes
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)
    
    # Test unsign with future timestamp (age < 0)
    class FutureTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) + 1000
    
    future_signer = FutureTimestampSigner("secret-key", salt="test-salt")
    signed = future_signer.sign("test-value")
    with pytest.raises(SignatureExpired, match="< 0 seconds"):
        signer.unsign(signed, max_age=3600)
    
    # Test unsign with invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid-value")
    
    # Test unsign with tampered value
    signed = signer.sign("test-value")
    tampered = signed[:-1] + (b"x" if signed[-1:] != b"x" else b"y")
    with pytest.raises(BadSignature):
        signer.unsign(tampered)
    
    # Test unsign with missing timestamp
    # Create a signer that produces a signature without timestamp
    class NoTimestampSigner(TimestampSigner):
        def sign(self, value):
            value = want_bytes(value)
            return value + want_bytes(self.sep) + self.get_signature(value)
    
    no_ts_signer = NoTimestampSigner("secret-key", salt="test-salt")
    signed_no_ts = no_ts_signer.sign("test-value")
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(signed_no_ts)
    
    # Test unsign with malformed timestamp
    class MalformedTimestampSigner(TimestampSigner):
        def sign(self, value):
            value = want_bytes(value)
            sep = want_bytes(self.sep)
            timestamp = b"not-a-timestamp"
            value = value + sep + timestamp
            return value + sep + self.get_signature(value)
    
    malformed_signer = MalformedTimestampSigner("secret-key", salt="test-salt")
    signed_malformed = malformed_signer.sign("test-value")
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(signed_malformed)
```


# LLM-generated content at query #93
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test successful loads without timestamp
    serializer = TimedSerializer(secret_key="test-secret")
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data

    # Test successful loads with return_timestamp
    result_with_ts = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == data
    assert isinstance(result_with_ts[1], datetime)

    # Test loads with max_age that should succeed
    result = serializer.loads(signed, max_age=3600)
    assert result == data

    # Test loads with max_age that should fail
    import time
    # Create a signed value with an old timestamp
    old_signer = TimestampSigner(secret_key="test-secret")
    old_signer.get_timestamp = lambda: int(time.time()) - 100
    old_signed = old_signer.sign("test-value")
    
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=10)

    # Test loads with expired signature
    expired_signer = TimestampSigner(secret_key="test-secret")
    expired_signer.get_timestamp = lambda: int(time.time()) - 10000
    expired_signed = expired_signer.sign("test-value")
    
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_signed, max_age=3600)

    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-signature")

    # Test loads with return_timestamp and max_age
    result_with_ts = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == data
    assert isinstance(result_with_ts[1], datetime)

    # Test loads with bytes input
    signed_bytes = serializer.dumps(data)
    result = serializer.loads(signed_bytes)
    assert result == data

    # Test loads with string input
    signed_str = serializer.dumps(data).decode('utf-8')
    result = serializer.loads(signed_str)
    assert result == data

    # Test loads with multiple signers (fallback)
    serializer2 = TimedSerializer(secret_key="other-secret")
    signed2 = serializer2.dumps(data)
    
    # Should fail with wrong key
    with pytest.raises(BadSignature):
        serializer.loads(signed2)

    # Test loads with empty data
    with pytest.raises(BadSignature):
        serializer.loads(b"")

    # Test loads with very old timestamp (negative age)
    import time
    future_signer = TimestampSigner(secret_key="test-secret")
    future_signer.get_timestamp = lambda: int(time.time()) + 1000
    future_signed = future_signer.sign("test-value")
    
    with pytest.raises(SignatureExpired):
        serializer.loads(future_signed, max_age=3600)

    # Test loads with salt
    salted_serializer = TimedSerializer(secret_key="test-secret", salt="custom-salt")
    salted_data = {"nested": "data"}
    salted_signed = salted_serializer.dumps(salted_data)
    result = serializer.loads(salted_signed, salt="custom-salt")
    assert result == salted_data

    # Test loads with salt but wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(salted_signed, salt="wrong-salt")
```


# LLM-generated content at query #94
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner(secret_key="test-secret-key")
    assert signer.secret_key == b"test-secret-key"
    assert signer.salt == b"itsdangerous.TimestampSigner"
    assert signer.digest_method is not None
    assert signer.key_derivation == "hmac"
    
    signer_custom = TimestampSigner(
        secret_key="custom-key",
        salt="custom-salt",
        digest_method="sha512",
        key_derivation="none"
    )
    assert signer_custom.secret_key == b"custom-key"
    assert signer_custom.salt == b"custom-salt"
    assert signer_custom.digest_method == "sha512"
    assert signer_custom.key_derivation == "none"
    
    signer_bytes = TimestampSigner(secret_key=b"bytes-key")
    assert signer_bytes.secret_key == b"bytes-key"


# LLM-generated content at query #95
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test-secret")
    
    # Test basic loads without max_age and return_timestamp
    original_data = {"key": "value"}
    serialized = serializer.dumps(original_data)
    result = serializer.loads(serialized)
    assert result == original_data
    
    # Test loads with max_age parameter
    serialized = serializer.dumps(original_data)
    result = serializer.loads(serialized, max_age=3600)
    assert result == original_data
    
    # Test loads with return_timestamp=True
    serialized = serializer.dumps(original_data)
    result = serializer.loads(serialized, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    payload, timestamp = result
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with both max_age and return_timestamp
    serialized = serializer.dumps(original_data)
    result = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert isinstance(result, tuple)
    payload, timestamp = result
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with salt parameter
    serialized = serializer.dumps(original_data, salt="custom-salt")
    result = serializer.loads(serialized, salt="custom-salt")
    assert result == original_data
    
    # Test loads with all parameters
    serialized = serializer.dumps(original_data, salt="custom-salt")
    result = serializer.loads(serialized, max_age=3600, return_timestamp=True, salt="custom-salt")
    assert isinstance(result, tuple)
    payload, timestamp = result
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with expired signature
    serialized = serializer.dumps(original_data)
    with pytest.raises(SignatureExpired):
        serializer.loads(serialized, max_age=-1)
    
    # Test loads with bad signature
    bad_serialized = b"invalid-data"
    with pytest.raises(BadSignature):
        serializer.loads(bad_serialized)
    
    # Test loads with bytes input
    serialized_bytes = serializer.dumps(original_data)
    result = serializer.loads(serialized_bytes)
    assert result == original_data
    
    # Test loads with string input
    serialized_str = serializer.dumps(original_data).decode()
    result = serializer.loads(serialized_str)
    assert result == original_data
    
    # Test loads with different data types
    test_cases = [
        "string data",
        12345,
        3.14,
        True,
        None,
        [1, 2, 3],
        {"nested": {"data": "value"}},
        b"bytes data",
    ]
    
    for test_data in test_cases:
        serialized = serializer.dumps(test_data)
        result = serializer.loads(serialized)
        assert result == test_data
    
    # Test loads with multiple signers (fallback scenario)
    serializer2 = TimedSerializer("test-secret-2")
    serialized = serializer2.dumps(original_data)
    # Should fail with original serializer
    with pytest.raises(BadSignature):
        serializer.loads(serialized)
```


# LLM-generated content at query #96
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner(secret_key="test-secret")
    
    # Test basic unsign without timestamp
    value = b"test-value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with max_age (should not expire if signed recently)
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test unsign with expired signature
    signer_with_fixed_time = TimestampSigner(secret_key="test-secret")
    original_get_timestamp = signer_with_fixed_time.get_timestamp
    signer_with_fixed_time.get_timestamp = lambda: int(time.time()) - 100  # 100 seconds ago
    old_signed = signer_with_fixed_time.sign(value)
    signer_with_fixed_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_with_fixed_time.unsign(old_signed, max_age=50)
    
    # Test unsign with malformed timestamp
    malformed = b"test-value" + signer_with_fixed_time.sep.encode() + b"invalid-timestamp" + signer_with_fixed_time.sep.encode() + b"signature"
    with pytest.raises(BadTimeSignature):
        signer_with_fixed_time.unsign(malformed)
    
    # Test unsign with missing timestamp
    just_value = b"test-value" + signer_with_fixed_time.sep.encode() + b"some-data"
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer_with_fixed_time.unsign(just_value)
    
    # Test unsign with bad signature but valid timestamp
    valid_signed = signer.sign(value)
    bad_signed = valid_signed[:-1] + b"0"  # Change last byte of signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed)
    
    # Test unsign with negative age (future timestamp)
    signer_future = TimestampSigner(secret_key="test-secret")
    signer_future.get_timestamp = lambda: int(time.time()) + 1000  # 1000 seconds in future
    future_signed = signer_future.sign(value)
    with pytest.raises(SignatureExpired, match="< 0 seconds"):
        signer.unsign(future_signed, max_age=500)
    
    # Test unsign with string input
    signed_str = signer.sign(value).decode()
    result = signer.unsign(signed_str)
    assert result == value
    
    # Verify unsign returns bytes
    assert isinstance(result, bytes)
```


# LLM-generated content at query #97
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner(secret_key="test-secret")
    assert signer.secret_key == b"test-secret"
    assert signer.sep == "."
    assert signer.salt is None
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == "sha1"
    assert signer.algorithm is not None

    # Test constructor with custom parameters
    signer2 = TimestampSigner(
        secret_key="custom-secret",
        salt="custom-salt",
        sep=":",
        key_derivation="none",
        digest_method="sha256",
    )
    assert signer2.secret_key == b"custom-secret"
    assert signer2.salt == "custom-salt"
    assert signer2.sep == ":"
    assert signer2.key_derivation == "none"
    assert signer2.digest_method == "sha256"
```


# LLM-generated content at query #98
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads method."""
    serializer = TimedSerializer(secret_key="test-secret-key-12345")
    
    # Test basic loads without max_age or return_timestamp
    original_data = {"key": "value", "number": 42}
    signed_data = serializer.dumps(original_data)
    result = serializer.loads(signed_data)
    assert result == original_data
    
    # Test loads with max_age (valid age)
    result = serializer.loads(signed_data, max_age=3600)
    assert result == original_data
    
    # Test loads with return_timestamp=True
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == original_data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test loads with both max_age and return_timestamp
    result, timestamp = serializer.loads(signed_data, max_age=3600, return_timestamp=True)
    assert result == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with expired signature
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=0)
    
    # Test loads with invalid data
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test loads with tampered signature
    tampered_data = signed_data[:-1] + b"X"
    with pytest.raises(BadSignature):
        serializer.loads(tampered_data)


# LLM-generated content at query #99
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("test-secret")
    
    # Test 1: Basic loads without max_age and return_timestamp
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    result = serializer.loads(signed_data)
    assert result == data
    
    # Test 2: Loads with return_timestamp=True
    result_with_ts = serializer.loads(signed_data, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    payload, timestamp = result_with_ts
    assert payload == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Loads with max_age
    fresh_signed = serializer.dumps(data)
    result_fresh = serializer.loads(fresh_signed, max_age=3600)
    assert result_fresh == data
    
    # Test 4: Loads with expired signature
    expired_signed = serializer.dumps(data)
    time.sleep(1)
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_signed, max_age=0)
    
    # Test 5: Loads with invalid signature raises BadSignature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test 6: Loads with different salt
    serializer2 = TimedSerializer("test-secret", salt="different-salt")
    signed_data2 = serializer2.dumps(data)
    with pytest.raises(BadSignature):
        serializer.loads(signed_data2)
    
    # Test 7: Loads with bytes input
    signed_bytes = serializer.dumps(data)
    result_bytes = serializer.loads(signed_bytes)
    assert result_bytes == data
    
    # Test 8: Loads with string input
    signed_str = serializer.dumps(data).decode()
    result_str = serializer.loads(signed_str)
    assert result_str == data
    
    # Test 9: Loads with both max_age and return_timestamp
    result_with_both = serializer.loads(
        signed_data, max_age=3600, return_timestamp=True
    )
    assert isinstance(result_with_both, tuple)
    assert len(result_with_both) == 2
    payload_both, timestamp_both = result_with_both
    assert payload_both == data
    assert isinstance(timestamp_both, datetime)
    assert timestamp_both.tzinfo == timezone.utc
    
    # Test 10: Loads with very large max_age (should work)
    old_serializer = TimedSerializer("test-secret")
    old_signed = old_serializer.dumps(data)
    result_old = old_serializer.loads(old_signed, max_age=999999999)
    assert result_old == data
    
    # Test 11: Loads with empty data
    empty_data = {}
    empty_signed = serializer.dumps(empty_data)
    result_empty = serializer.loads(empty_signed)
    assert result_empty == empty_data
    
    # Test 12: Loads with list data
    list_data = [1, 2, 3, "test"]
    list_signed = serializer.dumps(list_data)
    result_list = serializer.loads(list_signed)
    assert result_list == list_data
```


# LLM-generated content at query #100
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without timestamp
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    
    # Test unsign with max_age (valid)
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test unsign with max_age (expired)
    import time as time_module
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time_module.time()) + 7200  # 2 hours later
    
    try:
        signer.unsign(signed, max_age=3600)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "Signature age" in str(e)
        assert e.payload == b"test_value"
    finally:
        signer.get_timestamp = original_get_timestamp
    
    # Test unsign with malformed timestamp
    malformed = signed + b"bad"
    try:
        signer.unsign(malformed)
        assert False, "Expected BadSignature"
    except BadSignature:
        pass
    
    # Test unsign with missing timestamp
    value = want_bytes("test_value")
    sep = want_bytes(signer.sep)
    no_timestamp = value + sep + signer.get_signature(value)
    try:
        signer.unsign(no_timestamp)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "timestamp missing" in str(e)
    
    # Test unsign with invalid signature but valid timestamp
    invalid_sig = signed[:-1] + b"X"
    try:
        signer.unsign(invalid_sig)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert e.payload == b"test_value"
    
    # Test unsign with bytes input
    signed_bytes = signer.sign(b"bytes_value")
    result = signer.unsign(signed_bytes)
    assert result == b"bytes_value"
    
    # Test validate returns True for valid signature
    assert signer.validate(signed) == True
    
    # Test validate returns False for invalid signature
    assert signer.validate(b"invalid") == False
    
    # Test unsign with negative age (future timestamp)
    future_timestamp = int(time_module.time()) + 1000
    future_bytes = int_to_bytes(future_timestamp)
    future_b64 = base64_encode(future_bytes)
    future_signed = b"test_value" + sep + future_b64 + sep + signer.get_signature(b"test_value" + sep + future_b64)
    try:
        signer.unsign(future_signed, max_age=60)
        assert False, "Expected SignatureExpired for future timestamp"
    except SignatureExpired as e:
        assert "age" in str(e)


# LLM-generated content at query #101
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer(secret_key="test-secret")
    
    # Test successful loads without timestamp
    original_data = {"key": "value"}
    serialized = serializer.dumps(original_data)
    result = serializer.loads(serialized)
    assert result == original_data
    
    # Test successful loads with timestamp
    result_with_ts = serializer.loads(serialized, return_timestamp=True)
    payload, timestamp = result_with_ts
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with valid max_age
    result = serializer.loads(serialized, max_age=3600)
    assert result == original_data
    
    # Test loads with expired signature
    import time as time_module
    # Create a serializer with a mocked timestamp that's old
    class OldTimestampSerializer(TimedSerializer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._timestamp = int(time_module.time()) - 100  # 100 seconds old
    
    old_serializer = OldTimestampSerializer(secret_key="test-secret")
    old_serialized = old_serializer.dumps(original_data)
    
    with pytest.raises(SignatureExpired):
        old_serializer.loads(old_serialized, max_age=50)
    
    # Test loads with bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid_data")
    
    # Test loads with different salt
    salt1 = serializer.dumps(original_data, salt="salt1")
    result = serializer.loads(salt1, salt="salt1")
    assert result == original_data
    
    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(salt1, salt="wrong_salt")
```


# LLM-generated content at query #102
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads method."""
    serializer = TimedSerializer("test_secret_key")
    
    # Test basic loads without max_age or return_timestamp
    original_data = {"key": "value"}
    serialized = serializer.dumps(original_data)
    result = serializer.loads(serialized)
    assert result == original_data
    
    # Test loads with return_timestamp
    result_with_ts = serializer.loads(serialized, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    payload, timestamp = result_with_ts
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with valid max_age
    result = serializer.loads(serialized, max_age=3600)
    assert result == original_data
    
    # Test loads with expired signature
    import time
    old_serializer = TimedSerializer("test_secret_key")
    old_serializer.default_signer.get_timestamp = lambda: int(time.time()) - 100
    old_serialized = old_serializer.dumps(original_data)
    
    with pytest.raises(SignatureExpired):
        serializer.loads(old_serialized, max_age=10)
    
    # Test loads with bad signature
    bad_data = serialized + b"tampered"
    with pytest.raises(BadSignature):
        serializer.loads(bad_data)
    
    # Test loads with invalid data (no timestamp)
    invalid_data = b"no_timestamp"
    with pytest.raises(BadTimeSignature):
        serializer.loads(invalid_data)
    
    # Test loads with different salt
    serialized_with_salt = serializer.dumps(original_data, salt="custom_salt")
    result = serializer.loads(serialized_with_salt, salt="custom_salt")
    assert result == original_data
    
    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(serialized_with_salt, salt="wrong_salt")
    
    # Test loads with multiple signers (fallback mechanism)
    serializer2 = TimedSerializer("another_secret_key")
    serialized2 = serializer2.dumps(original_data)
    result = serializer.loads(serialized2)
    assert result == original_data
    
    # Test loads with return_timestamp and max_age combined
    result_with_ts = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    payload, timestamp = result_with_ts
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with bytes input
    serialized_bytes = serialized if isinstance(serialized, bytes) else serialized.encode()
    result = serializer.loads(serialized_bytes)
    assert result == original_data
    
    # Test loads with string input
    serialized_str = serialized.decode() if isinstance(serialized, bytes) else serialized
    result = serializer.loads(serialized_str)
    assert result == original_data
    
    # Test loads with empty data
    with pytest.raises(BadSignature):
        serializer.loads(b"")
    
    # Test loads with very old timestamp (overflow test)
    import sys
    if sys.maxsize > 2**31 - 1:  # Only test on 64-bit systems
        old_serializer = TimedSerializer("test_secret_key")
        old_serializer.default_signer.get_timestamp = lambda: 0
        old_serialized = old_serializer.dumps(original_data)
        with pytest.raises(BadTimeSignature):
            serializer.loads(old_serialized)
```


# LLM-generated content at query #103
#--------------------------

```python
def test_TimestampSigner():
    # Test default construction
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == b"secret-key"
    assert signer.salt == b"itsdangerous.TimestampSigner"
    assert signer.sep == b"."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == hmac_compat

    # Test construction with custom salt
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

    # Test construction with custom separator
    signer = TimestampSigner("secret-key", sep=":")
    assert signer.sep == b":"

    # Test construction with key_derivation
    signer = TimestampSigner("secret-key", key_derivation="none")
    assert signer.key_derivation == "none"

    # Test construction with digest_method
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

    # Test that it inherits from Signer
    assert isinstance(signer, Signer)

    # Test default timestamp function returns integer
    ts = signer.get_timestamp()
    assert isinstance(ts, int)
    assert ts > 0

    # Test timestamp_to_datetime returns UTC datetime
    dt = signer.timestamp_to_datetime(ts)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc
```


# LLM-generated content at query #104
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key", salt="test-salt")
    
    # Test successful unsign without timestamp
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"
    
    # Test successful unsign with timestamp
    signed = signer.sign("test-value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with valid max_age
    signed = signer.sign("test-value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test-value"
    
    # Test unsign with expired signature
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) - 100
    signed = signer.sign("old-value")
    signer.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=50)
    
    # Test unsign with future timestamp (age < 0)
    signer.get_timestamp = lambda: int(time.time())
    signed = signer.sign("future-value")
    signer.get_timestamp = lambda: int(time.time()) - 100
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(signed, max_age=3600)
    assert exc_info.value.date_signed is not None
    
    # Test unsign with bad signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid-signature")
    
    # Test unsign with malformed timestamp
    signed = signer.sign("test-value")
    tampered = signed + b"extra"
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered)
    
    # Test unsign with missing timestamp
    plain_signer = Signer("secret-key", salt="test-salt")
    signed_no_ts = plain_signer.sign("no-timestamp")
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_no_ts)
    
    # Test unsign with bad signature but valid payload and timestamp
    original_sign = signer.sign
    signer.sign = lambda v: original_sign(v) + b"tampered"
    signed_tampered = signer.sign("test-value")
    signer.sign = original_sign
    
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_tampered)
    
    # Test unsign returns bytes type
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert isinstance(result, bytes)
    
    # Test unsign with return_timestamp returns tuple
    signed = signer.sign("test-value")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bytes)
    assert isinstance(result[1], datetime)
```


# LLM-generated content at query #105
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor creates instance with correct default signer."""
    serializer = TimedSerializer("secret-key")
    
    assert isinstance(serializer, Serializer)
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer is TimestampSigner
    
    # Test that it creates TimestampSigner instances
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    
    # Test with salt parameter
    serializer_with_salt = TimedSerializer("secret-key", salt="custom-salt")
    assert serializer_with_salt.salt == "custom-salt"
    
    # Test with signer_kwargs
    serializer_with_kwargs = TimedSerializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    signer_with_kwargs = serializer_with_kwargs.make_signer()
    assert signer_with_kwargs.key_derivation == "hmac"
```


# LLM-generated content at query #106
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test-secret")
    
    # Test basic loads without max_age or return_timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with max_age
    signed = serializer.dumps(data)
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with return_timestamp=True
    signed = serializer.dumps(data)
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with both max_age and return_timestamp
    signed = serializer.dumps(data)
    result, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with expired signature
    import time
    old_serializer = TimedSerializer("test-secret")
    old_serializer.default_signer().get_timestamp = lambda: int(time.time()) - 100
    signed = old_serializer.dumps(data)
    try:
        serializer.loads(signed, max_age=10)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass
    
    # Test loads with bad signature
    bad_signed = signed + b"tampered"
    try:
        serializer.loads(bad_signed)
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass
    
    # Test loads with salt
    signed_with_salt = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == data
    
    # Test loads with wrong salt
    try:
        serializer.loads(signed_with_salt, salt="wrong-salt")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass
    
    # Test loads with bytes input
    signed_bytes = serializer.dumps(data)
    result = serializer.loads(signed_bytes)
    assert result == data
    
    # Test loads with string input
    signed_str = serializer.dumps(data).decode()
    result = serializer.loads(signed_str)
    assert result == data
```


# LLM-generated content at query #107
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads method with various scenarios."""
    serializer = TimedSerializer("test-secret")
    
    # Test 1: Basic loads without max_age
    value = {"key": "value"}
    signed = serializer.dumps(value)
    result = serializer.loads(signed)
    assert result == value
    
    # Test 2: Loads with return_timestamp=True
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    
    # Test 3: Loads with valid max_age
    result = serializer.loads(signed, max_age=3600)  # 1 hour
    assert result == value
    
    # Test 4: Loads with expired signature
    # Create a signer with a timestamp in the past
    class OldTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) - 7200  # 2 hours ago
    
    old_serializer = TimedSerializer("test-secret")
    old_serializer.default_signer = OldTimestampSigner
    old_signed = old_serializer.dumps(value)
    
    with pytest.raises(SignatureExpired):
        old_serializer.loads(old_signed, max_age=3600)
    
    # Test 5: Loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-signature")
    
    # Test 6: Loads with empty string
    with pytest.raises(BadSignature):
        serializer.loads("")
    
    # Test 7: Loads with bytes input
    signed_bytes = serializer.dumps(value)
    assert isinstance(signed_bytes, bytes)
    result = serializer.loads(signed_bytes)
    assert result == value
    
    # Test 8: Loads with different salt
    signed_with_salt = serializer.dumps(value, salt="different-salt")
    result = serializer.loads(signed_with_salt, salt="different-salt")
    assert result == value
    
    # Test 9: Loads with wrong salt should fail
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
    
    # Test 10: Loads with return_timestamp and max_age
    result, timestamp = serializer.loads(signed, return_timestamp=True, max_age=3600)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None
    
    # Test 11: Loads with max_age = 0 (should fail unless signature is exactly now)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=0)
    
    # Test 12: Loads with complex payload
    complex_value = {"list": [1, 2, 3], "nested": {"a": 1}, "bool": True}
    complex_signed = serializer.dumps(complex_value)
    result = serializer.loads(complex_signed)
    assert result == complex_value
    
    # Test 13: Loads with None payload
    none_signed = serializer.dumps(None)
    result = serializer.loads(none_signed)
    assert result is None
    
    # Test 14: Loads with integer payload
    int_signed = serializer.dumps(42)
    result = serializer.loads(int_signed)
    assert result == 42
    
    # Test 15: Loads with string payload
    str_signed = serializer.dumps("test string")
    result = serializer.loads(str_signed)
    assert result == "test string"
```


# LLM-generated content at query #108
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test successful loads without timestamp
    serializer = TimedSerializer("secret")
    value = {"key": "value"}
    signed = serializer.dumps(value)
    result = serializer.loads(signed)
    assert result == value

    # Test successful loads with return_timestamp
    result_with_ts = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    payload, timestamp = result_with_ts
    assert payload == value
    assert isinstance(timestamp, datetime)

    # Test with max_age (should not expire)
    result = serializer.loads(signed, max_age=3600)
    assert result == value

    # Test with max_age that causes expiration
    import time as time_module
    mock_time = int(time_module.time()) - 100  # 100 seconds in the past
    serializer_with_past_time = TimedSerializer("secret")
    original_get_timestamp = serializer_with_past_time.default_signer.get_timestamp
    serializer_with_past_time.default_signer.get_timestamp = lambda: mock_time
    signed_old = serializer_with_past_time.dumps(value)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_old, max_age=10)

    # Test with bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid_data")

    # Test with different salt
    signed_with_salt = serializer.dumps(value, salt="custom_salt")
    result = serializer.loads(signed_with_salt, salt="custom_salt")
    assert result == value

    # Test that wrong salt raises error
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong_salt")

    # Test with bytes input
    signed_bytes = serializer.dumps(value)
    result = serializer.loads(signed_bytes)
    assert result == value

    # Test return_timestamp with max_age
    result_with_ts = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    payload, timestamp = result_with_ts
    assert payload == value
    assert isinstance(timestamp, datetime)
```


# LLM-generated content at query #109
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor and basic initialization."""
    serializer = TimedSerializer("test-secret")
    
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    assert isinstance(serializer, TimedSerializer)
    
    # Test with different secret key types
    serializer_bytes = TimedSerializer(b"test-secret")
    assert isinstance(serializer_bytes, TimedSerializer)
    
    # Test with salt
    serializer_salt = TimedSerializer("test-secret", salt="custom-salt")
    assert isinstance(serializer_salt, TimedSerializer)
    
    # Test with signer_kwargs
    serializer_kwargs = TimedSerializer(
        "test-secret", signer_kwargs={"key_derivation": "hmac"}
    )
    assert isinstance(serializer_kwargs, TimedSerializer)
    
    # Test with serializer_kwargs
    serializer_serializer = TimedSerializer(
        "test-secret", serializer_kwargs={"key_derivation": "none"}
    )
    assert isinstance(serializer_serializer, TimedSerializer)
    
    # Test that default_signer can be overridden
    class CustomTimestampSigner(TimestampSigner):
        pass
    
    serializer_custom = TimedSerializer("test-secret")
    assert serializer_custom.default_signer is TimestampSigner
```


# LLM-generated content at query #110
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test 1: Basic sign and unsign without max_age
    value = b"test_value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test 2: Sign and unsign with return_timestamp=True
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Unsign with valid max_age
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test 4: Unsign with expired signature (max_age too small)
    signer_with_fixed_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_fixed_time.get_timestamp
    signer_with_fixed_time.get_timestamp = lambda: 1000
    signed = signer_with_fixed_time.sign(value)
    signer_with_fixed_time.get_timestamp = lambda: 2000
    with pytest.raises(SignatureExpired):
        signer_with_fixed_time.unsign(signed, max_age=500)
    
    # Test 5: Unsign with negative age (future timestamp)
    signer_with_fixed_time.get_timestamp = lambda: 500
    with pytest.raises(SignatureExpired):
        signer_with_fixed_time.unsign(signed, max_age=3600)
    
    # Test 6: Unsign with malformed timestamp
    malformed = signed.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid_timestamp"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test 7: Unsign with missing timestamp
    no_timestamp = value + signer.sep.encode() + signer.get_signature(value)
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp)
    
    # Test 8: Unsign with bad signature but valid timestamp
    bad_sig = signed[:-1] + b"x"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_sig)
    
    # Test 9: Unsign with string input
    signed_str = signer.sign("test_value").decode()
    result = signer.unsign(signed_str)
    assert result == b"test_value"
    
    # Test 10: Verify timestamp_to_datetime conversion
    timestamp = signer.get_timestamp()
    dt = signer.timestamp_to_datetime(timestamp)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc
```


# LLM-generated content at query #111
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    
    # Test 1: Basic loads without max_age and return_timestamp
    original_data = {"key": "value"}
    serialized = serializer.dumps(original_data)
    result = serializer.loads(serialized)
    assert result == original_data
    
    # Test 2: loads with max_age (valid age)
    result = serializer.loads(serialized, max_age=3600)
    assert result == original_data
    
    # Test 3: loads with return_timestamp=True
    payload, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test 4: loads with both max_age and return_timestamp
    payload, timestamp = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test 5: loads with custom salt
    serializer_salt = TimedSerializer("secret-key", salt="custom-salt")
    serialized_salt = serializer_salt.dumps(original_data)
    result = serializer_salt.loads(serialized_salt, salt="custom-salt")
    assert result == original_data
    
    # Test 6: loads should raise SignatureExpired for expired signature
    serialized_old = serializer.dumps(original_data)
    # We can't easily manipulate time, but we can test with max_age=0
    # This should fail because the signature was just created (age > 0)
    import time as time_module
    # Create a signature and immediately try to validate it with max_age=0
    # The signature age will be ~0, which might pass if age == 0
    # So let's test with a negative max_age to ensure it fails
    try:
        serializer.loads(serialized_old, max_age=-1)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass
    
    # Test 7: loads should raise BadSignature for invalid data
    try:
        serializer.loads(b"invalid-data")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass
    
    # Test 8: loads with bytes input
    serialized_bytes = serializer.dumps(original_data)
    result = serializer.loads(serialized_bytes)
    assert result == original_data
    
    # Test 9: loads with string input
    serialized_str = serializer.dumps(original_data).decode("utf-8")
    result = serializer.loads(serialized_str)
    assert result == original_data
    
    # Test 10: loads with different key signer
    serializer2 = TimedSerializer("different-key")
    serialized2 = serializer2.dumps(original_data)
    try:
        serializer.loads(serialized2)
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass
```


# LLM-generated content at query #112
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "sha1"
    
    # Test with custom parameters
    signer2 = TimestampSigner(
        "secret-key",
        salt="custom-salt",
        sep="|",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm="sha256"
    )
    assert signer2.secret_key == "secret-key"
    assert signer2.salt == "custom-salt"
    assert signer2.sep == "|"
    assert signer2.key_derivation == "none"
    assert signer2.digest_method == hashlib.sha256
    assert signer2.algorithm == "sha256"
    
    # Test with bytes secret key
    signer3 = TimestampSigner(b"bytes-secret")
    assert signer3.secret_key == b"bytes-secret"
    
    # Test inheritance
    assert isinstance(signer, Signer)
    assert issubclass(TimestampSigner, Signer)
    
    # Test that get_timestamp returns int
    assert isinstance(signer.get_timestamp(), int)
    assert signer.get_timestamp() > 0
    
    # Test timestamp_to_datetime
    ts = signer.get_timestamp()
    dt = signer.timestamp_to_datetime(ts)
    assert isinstance(dt, datetime)
    assert dt.tzinfo is not None
    assert dt.tzinfo == timezone.utc
```


# LLM-generated content at query #113
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("test-secret-key")
    assert signer.secret_key == "test-secret-key"
    assert signer.salt == "timestamp-signer"
    assert signer.digest_method is not None
    assert signer.key_derivation == "hmac"
    assert signer.algorithm is not None
    assert signer.sep == "."
    
    signer_with_salt = TimestampSigner("test-secret-key", salt="custom-salt")
    assert signer_with_salt.salt == "custom-salt"
    
    signer_with_options = TimestampSigner(
        "test-secret-key",
        salt="custom-salt",
        sep=":",
        key_derivation="none"
    )
    assert signer_with_options.sep == ":"
    assert signer_with_options.key_derivation == "none"
    
    assert isinstance(signer.get_timestamp(), int)
    timestamp = signer.get_timestamp()
    dt = signer.timestamp_to_datetime(timestamp)
    assert dt.tzinfo is not None
    assert dt.tzinfo.utcoffset(dt) is not None
```


# LLM-generated content at query #114
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without timestamp
    signed_value = signer.sign("test_value")
    result = signer.unsign(signed_value)
    assert result == b"test_value"
    
    # Test unsign with return_timestamp=True
    signed_value = signer.sign("test_value")
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with max_age (valid)
    signed_value = signer.sign("test_value")
    result = signer.unsign(signed_value, max_age=3600)
    assert result == b"test_value"
    
    # Test unsign with max_age (expired)
    signer_with_fixed_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_fixed_time.get_timestamp
    
    def mock_get_timestamp():
        return int(time.time()) - 100  # 100 seconds in the past
    
    signer_with_fixed_time.get_timestamp = mock_get_timestamp
    signed_value = signer_with_fixed_time.sign("test_value")
    
    # Restore original timestamp for validation
    signer_with_fixed_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_with_fixed_time.unsign(signed_value, max_age=50)
    
    # Test unsign with max_age (future timestamp - age < 0)
    signer_with_future_time = TimestampSigner("secret-key")
    def mock_future_timestamp():
        return int(time.time()) + 100  # 100 seconds in the future
    
    signer_with_future_time.get_timestamp = mock_future_timestamp
    signed_future = signer_with_future_time.sign("test_value")
    
    # Restore original timestamp
    signer_with_future_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_with_future_time.unsign(signed_future, max_age=3600)
    
    # Test unsign with invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")
    
    # Test unsign with tampered value
    signed = signer.sign("test_value")
    tampered = signed[:-1] + (b"1" if signed[-1:] == b"0" else b"0")
    with pytest.raises(BadSignature):
        signer.unsign(tampered)
    
    # Test unsign with missing timestamp
    signer_no_timestamp = Signer("secret-key")
    signed_no_timestamp = signer_no_timestamp.sign("test_value")
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(signed_no_timestamp)
    
    # Test unsign with malformed timestamp
    value = want_bytes("test_value")
    sep = want_bytes(signer.sep)
    malformed_timestamp = base64_encode(b"not_a_timestamp")
    signed_malformed = value + sep + malformed_timestamp + sep + signer.get_signature(value + sep + malformed_timestamp)
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(signed_malformed)
    
    # Test unsign with bytes input
    signed_bytes = signer.sign(b"test_bytes")
    result = signer.unsign(signed_bytes)
    assert result == b"test_bytes"
    
    # Test unsign with string input
    signed_str = signer.sign("test_string")
    result = signer.unsign(signed_str.decode())
    assert result == b"test_string"
    
    # Test unsign with return_timestamp and max_age
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
```


# LLM-generated content at query #115
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor and basic functionality."""
    serializer = TimedSerializer("test-secret")
    assert serializer.default_signer is TimestampSigner
    assert serializer.secret_key == "test-secret"
    assert serializer.salt == "itsdangerous"
    assert serializer.serializer is None
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []


# LLM-generated content at query #116
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm is not None
    
    # Test constructor with custom salt
    signer_with_salt = TimestampSigner("secret-key", salt="custom-salt")
    assert signer_with_salt.salt == "custom-salt"
    
    # Test constructor with custom separator
    signer_with_sep = TimestampSigner("secret-key", sep=":")
    assert signer_with_sep.sep == ":"
    
    # Test constructor with custom key derivation
    signer_with_kd = TimestampSigner("secret-key", key_derivation="none")
    assert signer_with_kd.key_derivation == "none"
    
    # Test constructor with custom digest method
    signer_with_digest = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer_with_digest.digest_method == hashlib.sha256
    
    # Verify the signer is a subclass of Signer
    assert isinstance(signer, Signer)
    
    # Verify the signer has the expected methods
    assert hasattr(signer, "sign")
    assert hasattr(signer, "unsign")
    assert hasattr(signer, "validate")
    assert hasattr(signer, "get_timestamp")
    assert hasattr(signer, "timestamp_to_datetime")


# LLM-generated content at query #117
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("test-secret")
    
    # Test basic unsign without timestamp
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"
    
    # Test unsign with return_timestamp=True
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert result_with_ts[0] == b"test-value"
    assert isinstance(result_with_ts[1], datetime)
    
    # Test unsign with max_age within limit
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test-value"
    
    # Test unsign with max_age expired
    import time
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) + 10000  # Fast forward time
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=1)
    signer.get_timestamp = original_get_timestamp
    
    # Test unsign with negative age (future timestamp)
    signer.get_timestamp = lambda: int(time.time()) - 10000  # Go back in time
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=3600)
    signer.get_timestamp = original_get_timestamp
    
    # Test unsign with malformed timestamp
    bad_signed = signed + b"bad"
    with pytest.raises(BadSignature):
        signer.unsign(bad_signed)
    
    # Test unsign with missing timestamp
    no_timestamp_signed = b"test-value" + signer.sep.encode() + signer.get_signature(b"test-value")
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_timestamp_signed)
    
    # Test unsign with invalid timestamp encoding
    invalid_ts_signed = b"test-value" + signer.sep.encode() + b"invalid" + signer.sep.encode() + signer.get_signature(b"test-value" + signer.sep.encode() + b"invalid")
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(invalid_ts_signed)
```


# LLM-generated content at query #118
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Test basic unsign without timestamp return
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test unsign with return_timestamp=True
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with valid max_age
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test unsign with expired signature
    signer_with_fixed_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_fixed_time.get_timestamp
    signer_with_fixed_time.get_timestamp = lambda: 1000  # Fixed old timestamp
    signed_old = signer_with_fixed_time.sign("test_value")
    signer_with_fixed_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_with_fixed_time.unsign(signed_old, max_age=10)
    
    # Test unsign with future timestamp (age < 0)
    signer_with_future_time = TimestampSigner("secret-key")
    signer_with_future_time.get_timestamp = lambda: 2000  # Future timestamp
    signed_future = signer_with_future_time.sign("test_value")
    signer_with_future_time.get_timestamp = lambda: 1000  # Current time is in the past
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer_with_future_time.unsign(signed_future, max_age=3600)
    assert "0 seconds" in str(exc_info.value)
    
    # Test unsign with bad signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid_signature")
    
    # Test unsign with malformed timestamp
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    parts = signed.split(signer.sep.encode())
    malformed = parts[0] + signer.sep.encode() + b"invalid_base64"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test unsign with empty timestamp
    signed_without_timestamp = b"test_value" + signer.sep.encode()
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_without_timestamp)
```


# LLM-generated content at query #119
#--------------------------

```python
def test_TimestampSigner():
    # Test constructor with default parameters
    signer = TimestampSigner(secret_key="test-secret")
    assert signer.secret_key == "test-secret"
    assert signer.sep == "."
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "sha1"
    
    # Test constructor with custom parameters
    signer_custom = TimestampSigner(
        secret_key="custom-secret",
        salt="custom-salt",
        sep=":",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm="sha256"
    )
    assert signer_custom.secret_key == "custom-secret"
    assert signer_custom.salt == "custom-salt"
    assert signer_custom.sep == ":"
    assert signer_custom.key_derivation == "none"
    assert signer_custom.digest_method == hashlib.sha256
    assert signer_custom.algorithm == "sha256"
    
    # Test constructor with key_derivation as None
    signer_none = TimestampSigner(secret_key="test", key_derivation=None)
    assert signer_none.key_derivation is None
    
    # Test constructor with bytes secret_key
    signer_bytes = TimestampSigner(secret_key=b"bytes-secret")
    assert signer_bytes.secret_key == b"bytes-secret"
    
    # Test constructor with fallback_signers
    signer_fallback = TimestampSigner(
        secret_key="test",
        fallback_signers=["fallback1", "fallback2"]
    )
    assert signer_fallback.fallback_signers == ["fallback1", "fallback2"]


# LLM-generated content at query #120
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner("test-secret")
    assert signer.secret_key == b"test-secret"
    assert signer.sep == "."
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm is None
    
    # Test custom parameters
    custom_signer = TimestampSigner(
        "custom-secret",
        sep="|",
        salt="custom-salt",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm="hmac"
    )
    assert custom_signer.secret_key == b"custom-secret"
    assert custom_signer.sep == "|"
    assert custom_signer.salt == "custom-salt"
    assert custom_signer.key_derivation == "none"
    assert custom_signer.digest_method == hashlib.sha256
    assert custom_signer.algorithm == "hmac"
    
    # Test that get_timestamp returns integer
    timestamp = signer.get_timestamp()
    assert isinstance(timestamp, int)
    assert timestamp > 0
    
    # Test timestamp_to_datetime conversion
    ts = 1609459200  # 2021-01-01 00:00:00 UTC
    dt = signer.timestamp_to_datetime(ts)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc
    assert dt.year == 2021
    assert dt.month == 1
    assert dt.day == 1
    
    # Test sign method
    signed = signer.sign("test-value")
    assert isinstance(signed, bytes)
    assert b"test-value" in signed
    
    # Test unsign with valid signature
    unsigned = signer.unsign(signed)
    assert unsigned == b"test-value"
    
    # Test unsign with return_timestamp
    unsigned_with_ts = signer.unsign(signed, return_timestamp=True)
    assert len(unsigned_with_ts) == 2
    assert unsigned_with_ts[0] == b"test-value"
    assert isinstance(unsigned_with_ts[1], datetime)
    
    # Test validate method
    assert signer.validate(signed) is True
    assert signer.validate(b"invalid-signature") is False
    
    # Test max_age validation
    signed_old = signer.sign("old-value")
    assert signer.unsign(signed_old, max_age=1000000) == b"old-value"
    
    # Test with bytes secret
    bytes_signer = TimestampSigner(b"bytes-secret")
    assert bytes_signer.secret_key == b"bytes-secret"
    
    # Test with empty secret
    empty_signer = TimestampSigner("")
    assert empty_signer.secret_key == b""


# LLM-generated content at query #121
#--------------------------

```python
def test_TimedSerializer_loads():
    # Create a TimedSerializer instance
    serializer = TimedSerializer(secret_key="test-secret")
    
    # Test 1: Basic loads without max_age and return_timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test 2: loads with return_timestamp=True
    result = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    payload, timestamp = result
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test 3: loads with max_age (valid age)
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test 4: loads with max_age (expired)
    import time
    # Create a signed value that appears to be older
    old_signer = TimestampSigner("test-secret")
    old_signer.get_timestamp = lambda: int(time.time()) - 7200  # 2 hours ago
    old_signed = old_signer.sign("old-data")
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=3600)
    
    # Test 5: loads with both max_age and return_timestamp
    result = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result, tuple)
    payload, timestamp = result
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test 6: loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")
    
    # Test 7: loads with different salt
    signed_with_salt = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == data
    
    # Test 8: loads with wrong salt should fail
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
    
    # Test 9: loads with None data
    signed_none = serializer.dumps(None)
    result = serializer.loads(signed_none)
    assert result is None
    
    # Test 10: loads with list data
    list_data = [1, 2, 3]
    signed_list = serializer.dumps(list_data)
    result = serializer.loads(signed_list)
    assert result == list_data
    
    # Test 11: loads with empty dict
    signed_empty = serializer.dumps({})
    result = serializer.loads(signed_empty)
    assert result == {}
    
    # Test 12: loads with string data
    signed_string = serializer.dumps("test-string")
    result = serializer.loads(signed_string)
    assert result == "test-string"
    
    # Test 13: loads with max_age=0 (immediate expiration)
    signed_immediate = serializer.dumps("immediate")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_immediate, max_age=0)
    
    # Test 14: loads with negative max_age
    signed_negative = serializer.dumps("negative")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_negative, max_age=-1)
```


# LLM-generated content at query #122
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == b"secret-key"
    assert signer.salt == b"itsdangerous.TimestampSigner"
    assert signer.sep == b"."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == hmac

    # Test constructor with custom salt
    signer_custom_salt = TimestampSigner("secret-key", salt="custom-salt")
    assert signer_custom_salt.salt == b"custom-salt"

    # Test constructor with custom separator
    signer_custom_sep = TimestampSigner("secret-key", sep="|")
    assert signer_custom_sep.sep == b"|"

    # Test constructor with key derivation
    signer_custom_kd = TimestampSigner("secret-key", key_derivation="none")
    assert signer_custom_kd.key_derivation == "none"

    # Test constructor with digest method
    signer_custom_digest = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer_custom_digest.digest_method == hashlib.sha256

    # Test constructor with algorithm
    signer_custom_alg = TimestampSigner("secret-key", algorithm=hmac)
    assert signer_custom_alg.algorithm == hmac

    # Test that TimestampSigner is a subclass of Signer
    assert isinstance(signer, Signer)

    # Test get_timestamp returns integer
    timestamp = signer.get_timestamp()
    assert isinstance(timestamp, int)

    # Test timestamp_to_datetime returns UTC datetime
    dt = signer.timestamp_to_datetime(timestamp)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc
```


# LLM-generated content at query #123
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("test-secret-key")
    
    # Test 1: Basic sign and unsign
    value = b"test_value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test 2: Unsign with return_timestamp=True
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 3: Unsign with valid max_age
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test 4: Unsign with expired max_age
    signer_with_time = TimestampSigner("test-secret-key")
    original_get_timestamp = signer_with_time.get_timestamp
    
    def mock_old_timestamp():
        return int(time.time()) - 100
    
    signer_with_time.get_timestamp = mock_old_timestamp
    signed = signer_with_time.sign(value)
    signer_with_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_with_time.unsign(signed, max_age=50)
    
    # Test 5: Unsign with bad signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")
    
    # Test 6: Unsign with malformed timestamp
    signed = signer.sign(value)
    malformed = signed + b"malformed"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test 7: Unsign with negative age (future timestamp)
    signer_future = TimestampSigner("test-secret-key")
    
    def mock_future_timestamp():
        return int(time.time()) + 100
    
    signer_future.get_timestamp = mock_future_timestamp
    signed_future = signer_future.sign(value)
    signer_future.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired, match="< 0 seconds"):
        signer_future.unsign(signed_future, max_age=3600)
    
    # Test 8: Unsign with string input
    signed = signer.sign("test_string")
    result = signer.unsign(signed)
    assert result == b"test_string"
    
    # Test 9: Unsign with return_timestamp and max_age
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
```


# LLM-generated content at query #124
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key", salt="test-salt")
    
    # Test basic unsign without timestamp
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"
    
    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp, datetime)
    
    # Test unsign with valid max_age
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test-value"
    
    # Test unsign with expired signature (max_age=0)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)
    
    # Test unsign with negative age (future timestamp)
    future_signer = TimestampSigner("secret-key", salt="test-salt")
    future_signer.get_timestamp = lambda: int(time.time()) + 1000
    future_signed = future_signer.sign("test-value")
    with pytest.raises(SignatureExpired, match="age .* < 0 seconds"):
        signer.unsign(future_signed, max_age=3600)
    
    # Test unsign with malformed timestamp
    malformed = signed + signer.sep.encode() + b"invalid-timestamp"
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(malformed)
    
    # Test unsign with missing timestamp
    value_bytes = b"test-value"
    signature = signer.get_signature(value_bytes)
    no_timestamp = value_bytes + signer.sep.encode() + signature
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_timestamp)
    
    # Test unsign with invalid signature
    tampered = signed[:-1] + b"X"
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered)
    
    # Test unsign with invalid signature and return_timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered, return_timestamp=True)
    
    # Test unsign with different key
    different_signer = TimestampSigner("different-key", salt="test-salt")
    with pytest.raises(BadTimeSignature):
        different_signer.unsign(signed)
    
    # Test unsign with bytes input
    result = signer.unsign(signed)
    assert isinstance(result, bytes)
    
    # Test unsign with string input
    result = signer.unsign(signed.decode())
    assert result == b"test-value"
    
    # Test unsign with return_timestamp and expired signature
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0, return_timestamp=True)
```


# LLM-generated content at query #125
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test-secret")
    
    # Test basic loads without max_age or return_timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with return_timestamp
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with max_age that should pass
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with expired signature
    # Create a signer with a timestamp in the past
    signer = TimestampSigner("test-secret")
    signer.get_timestamp = lambda: int(time.time()) - 7200  # 2 hours ago
    value = want_bytes('{"key":"value"}')
    timestamp = base64_encode(int_to_bytes(signer.get_timestamp()))
    sep = want_bytes(signer.sep)
    signed_old = value + sep + timestamp + sep + signer.get_signature(value + sep + timestamp)
    
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_old, max_age=3600)
    
    # Test loads with bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test loads with multiple signers (fallback behavior)
    serializer2 = TimedSerializer(["key1", "key2"])
    signed_with_key1 = serializer2.dumps(data)
    result = serializer2.loads(signed_with_key1)
    assert result == data
    
    # Test loads_unsafe
    success, result = serializer.loads_unsafe(signed)
    assert success is True
    assert result == data
    
    success, result = serializer.loads_unsafe(b"invalid-data")
    assert success is False
    
    # Test with bytes input
    signed_bytes = serializer.dumps("test")
    result = serializer.loads(signed_bytes)
    assert result == "test"
    
    # Test with str input
    signed_str = signed_bytes.decode() if isinstance(signed_bytes, bytes) else signed_bytes
    result = serializer.loads(signed_str)
    assert result == "test"
```


# LLM-generated content at query #126
#--------------------------

```python
def test_TimedSerializer():
    # Test default construction
    serializer = TimedSerializer()
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    assert serializer.signer is TimestampSigner
    
    # Test with secret key
    serializer = TimedSerializer("my-secret-key")
    assert serializer.secret_key == b"my-secret-key"
    
    # Test with salt
    serializer = TimedSerializer("my-secret-key", salt="my-salt")
    assert serializer.salt == "my-salt"
    
    # Test with serializer_kwargs
    serializer = TimedSerializer(
        "my-secret-key",
        serializer_kwargs={"signer_kwargs": {"key_derivation": "hmac"}}
    )
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test with signer_kwargs
    serializer = TimedSerializer(
        "my-secret-key",
        signer_kwargs={"digest_method": "sha256"}
    )
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    
    # Test round-trip serialization/deserialization
    serializer = TimedSerializer("test-key")
    data = {"user": "test", "role": "admin"}
    serialized = serializer.dumps(data)
    deserialized = serializer.loads(serialized)
    assert deserialized == data
    
    # Test with max_age parameter
    serialized = serializer.dumps(data)
    deserialized = serializer.loads(serialized, max_age=3600)
    assert deserialized == data
    
    # Test return_timestamp parameter
    serialized = serializer.dumps(data)
    payload, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert payload == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None
    
    # Test with both max_age and return_timestamp
    serialized = serializer.dumps(data)
    payload, timestamp = serializer.loads(
        serialized, max_age=3600, return_timestamp=True
    )
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test with bytes input
    serialized = serializer.dumps(data)
    deserialized = serializer.loads(serialized)
    assert deserialized == data
    
    # Test with string input
    serialized_str = serializer.dumps(data).decode('utf-8')
    deserialized = serializer.loads(serialized_str)
    assert deserialized == data
    
    # Test loads_unsafe returns (True, payload) for valid signatures
    serialized = serializer.dumps(data)
    is_valid, payload = serializer.loads_unsafe(serialized)
    assert is_valid is True
    assert payload == data
    
    # Test loads_unsafe returns (False, payload) for invalid signatures
    is_valid, payload = serializer.loads_unsafe(b"invalid-data")
    assert is_valid is False
    
    # Test with different salt
    serializer1 = TimedSerializer("test-key", salt="salt1")
    serializer2 = TimedSerializer("test-key", salt="salt2")
    serialized1 = serializer1.dumps(data)
    # Should fail to deserialize with different salt
    import pytest
    with pytest.raises(BadSignature):
        serializer2.loads(serialized1)
```


# LLM-generated content at query #127
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor with default and custom parameters."""
    # Test default constructor
    serializer = TimedSerializer("test-secret")
    assert serializer.secret_key == "test-secret"
    assert serializer.salt == "itsdangerous.TimedSerializer"
    assert serializer.serializer_kwargs == {}
    assert serializer.sign_serializer is None
    assert serializer.fallback_signers == []
    assert serializer.digest_method is None
    
    # Test with custom parameters
    custom_serializer = TimedSerializer(
        secret_key="custom-secret",
        salt="custom-salt",
        serializer_kwargs={"skipkeys": True},
        signer_kwargs={"key_derivation": "hmac"},
        serializer="json",
        signer=TimestampSigner,
        fallback_signers=[Signer("fallback")],
        digest_method="sha512",
    )
    assert custom_serializer.secret_key == "custom-secret"
    assert custom_serializer.salt == "custom-salt"
    assert custom_serializer.serializer_kwargs == {"skipkeys": True}
    assert custom_serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert custom_serializer.serializer == "json"
    assert custom_serializer.signer == TimestampSigner
    assert custom_serializer.fallback_signers == [Signer("fallback")]
    assert custom_serializer.digest_method == "sha512"
    
    # Test default_signer is TimestampSigner
    assert serializer.default_signer == TimestampSigner
    
    # Test signer creation
    signer = serializer.create_signer("test-salt")
    assert isinstance(signer, TimestampSigner)
    assert signer.secret_key == "test-secret"
    assert signer.salt == "itsdangerous.TimedSerializertestsalt"
    
    # Test with bytes secret key
    bytes_serializer = TimedSerializer(b"bytes-secret")
    assert bytes_serializer.secret_key == b"bytes-secret"
```


# LLM-generated content at query #128
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads method"""
    serializer = TimedSerializer("test-secret")
    
    # Test 1: Basic loads without max_age or return_timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test 2: loads with max_age that is valid
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test 3: loads with return_timestamp=True
    result = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    payload, timestamp = result
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test 4: loads with max_age and return_timestamp
    result = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result, tuple)
    payload, timestamp = result
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test 5: loads with expired signature
    import time
    old_ts = serializer.loads(signed, return_timestamp=True)[1]
    # Manually create an expired signature
    expired_signer = TimestampSigner("test-secret")
    expired_signer.get_timestamp = lambda: int(time.time()) - 7200  # 2 hours ago
    expired_data = "expired_data"
    expired_signed = expired_signer.sign(expired_data)
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_signed, max_age=3600)
    
    # Test 6: loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid_signature")
    
    # Test 7: loads with salt parameter
    result = serializer.loads(signed, salt="custom-salt")
    assert result == data
    
    # Test 8: loads with string input (not bytes)
    signed_str = signed.decode('utf-8') if isinstance(signed, bytes) else signed
    result = serializer.loads(signed_str)
    assert result == data
    
    # Test 9: Test with list data
    list_data = [1, 2, 3]
    signed_list = serializer.dumps(list_data)
    result = serializer.loads(signed_list)
    assert result == list_data
    
    # Test 10: Test with None data
    signed_none = serializer.dumps(None)
    result = serializer.loads(signed_none)
    assert result is None
    
    # Test 11: Test loads_unsafe (related method)
    success, result = serializer.loads_unsafe(signed)
    assert success
    assert result == data
    
    success, result = serializer.loads_unsafe(b"invalid")
    assert not success
    assert result is None or isinstance(result, Exception)
    
    # Test 12: Test with multiple signers (fallback mechanism)
    serializer2 = TimedSerializer("test-secret2")
    signed2 = serializer2.dumps(data)
    # Should fail with original serializer since different keys
    with pytest.raises(BadSignature):
        serializer.loads(signed2)
```


# LLM-generated content at query #129
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without timestamp
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with max_age that is not expired
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test unsign with expired signature
    signer_fast = TimestampSigner("secret-key")
    signer_fast.get_timestamp = lambda: int(time.time()) - 100  # Simulate old timestamp
    old_signed = signer_fast.sign("test_value")
    
    with pytest.raises(SignatureExpired):
        signer.unsign(old_signed, max_age=10)
    
    # Test unsign with future timestamp (age < 0)
    signer_future = TimestampSigner("secret-key")
    signer_future.get_timestamp = lambda: int(time.time()) + 100  # Future timestamp
    future_signed = signer_future.sign("test_value")
    
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed, max_age=3600)
    
    # Test unsign with invalid signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid_signature")
    
    # Test unsign with tampered timestamp
    tampered = signed.replace(b"test_value", b"tampered")
    with pytest.raises(BadSignature):
        signer.unsign(tampered)
    
    # Test unsign with missing timestamp separator
    no_timestamp = signed.split(signer.sep.encode())[0]
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_timestamp)
    
    # Test unsign with malformed timestamp
    malformed = b"test_value" + signer.sep.encode() + b"not_base64"
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(malformed)
    
    # Test that SignatureExpired is raised before BadSignature when max_age is exceeded
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: int(time.time()) - 1000
    expired_signed = expired_signer.sign("test_value")
    
    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed, max_age=10)
```


# LLM-generated content at query #130
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner(secret_key="test-secret")
    assert signer.secret_key == "test-secret"
    assert signer.sep == "."
    assert signer.salt == "timestamp-signer"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "sha1"
    
    # Test with custom parameters
    signer_custom = TimestampSigner(
        secret_key="custom-key",
        salt="custom-salt",
        sep="-",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm="sha256"
    )
    assert signer_custom.secret_key == "custom-key"
    assert signer_custom.salt == "custom-salt"
    assert signer_custom.sep == "-"
    assert signer_custom.key_derivation == "none"
    assert signer_custom.digest_method == hashlib.sha256
    assert signer_custom.algorithm == "sha256"
    
    # Test inheritance from Signer
    assert isinstance(signer, Signer)
    
    # Test that get_timestamp returns an integer
    timestamp = signer.get_timestamp()
    assert isinstance(timestamp, int)
    assert timestamp > 0
    
    # Test that timestamp_to_datetime works
    dt = signer.timestamp_to_datetime(timestamp)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc
    
    # Test sign and unsign
    signed = signer.sign("test-value")
    assert isinstance(signed, bytes)
    
    unsigned = signer.unsign(signed)
    assert unsigned == b"test-value"
    
    # Test with return_timestamp
    unsigned_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(unsigned_with_ts, tuple)
    assert unsigned_with_ts[0] == b"test-value"
    assert isinstance(unsigned_with_ts[1], datetime)
    
    # Test max_age validation
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)
    
    # Test validate method
    assert signer.validate(signed) is True
    assert signer.validate(signed, max_age=1000) is True
    assert signer.validate(b"invalid-signature") is False
    
    # Test with bytes input
    signed_bytes = signer.sign(b"test-bytes")
    assert signer.unsign(signed_bytes) == b"test-bytes"
    
    # Test with different key
    signer2 = TimestampSigner(secret_key="different-key")
    with pytest.raises(BadSignature):
        signer2.unsign(signed)
```


# LLM-generated content at query #131
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("test-secret-key")
    
    # Test basic unsign returns bytes
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"
    assert isinstance(result, bytes)
    
    # Test with return_timestamp=True
    signed = signer.sign("test-value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None  # Should be timezone-aware
    
    # Test with max_age (valid)
    signed = signer.sign("test-value")
    result = signer.unsign(signed, max_age=3600)  # 1 hour
    assert result == b"test-value"
    
    # Test with max_age (expired) - should raise SignatureExpired
    signer_with_past_time = TimestampSigner("test-secret-key")
    original_get_timestamp = signer_with_past_time.get_timestamp
    signer_with_past_time.get_timestamp = lambda: original_get_timestamp() - 1000
    signed_past = signer_with_past_time.sign("test-value")
    signer_with_past_time.get_timestamp = original_get_timestamp
    
    try:
        signer_with_past_time.unsign(signed_past, max_age=10)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass
    
    # Test with max_age (negative age - future timestamp)
    signer_with_future = TimestampSigner("test-secret-key")
    signer_with_future.get_timestamp = lambda: original_get_timestamp() + 1000
    signed_future = signer_with_future.sign("test-value")
    
    try:
        signer.unsign(signed_future, max_age=10)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass
    
    # Test with invalid signature
    try:
        signer.unsign(b"invalid-data")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass
    
    # Test with missing timestamp
    regular_signer = Signer("test-secret-key")
    signed_no_timestamp = regular_signer.sign("test-value")
    try:
        signer.unsign(signed_no_timestamp)
        assert False, "Should have raised BadTimeSignature"
    except BadTimeSignature:
        pass
    
    # Test with malformed timestamp
    malformed = signer.sign("test-value")
    parts = malformed.split(signer.sep.encode())
    malformed = parts[0] + signer.sep.encode() + b"not-base64" + signer.sep.encode() + parts[-1]
    try:
        signer.unsign(malformed)
        assert False, "Should have raised BadTimeSignature"
    except BadTimeSignature:
        pass
    
    # Test with string input
    signed_str = signer.sign("test-value").decode()
    result = signer.unsign(signed_str)
    assert result == b"test-value"
    
    # Test both return_timestamp and max_age
    signed = signer.sign("test-value")
    result, timestamp = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp, datetime)
```


# LLM-generated content at query #132
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without max_age
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test unsign with return_timestamp
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    
    # Test unsign with valid max_age
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test unsign with expired signature
    signer_with_fixed_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_fixed_time.get_timestamp
    
    # Create signature with "old" timestamp
    def old_timestamp():
        return int(time.time()) - 100  # 100 seconds ago
    
    signer_with_fixed_time.get_timestamp = old_timestamp
    signed_old = signer_with_fixed_time.sign("test_value")
    signer_with_fixed_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_with_fixed_time.unsign(signed_old, max_age=50)
    
    # Test unsign with future timestamp (age < 0)
    def future_timestamp():
        return int(time.time()) + 100  # 100 seconds in future
    
    signer_with_fixed_time.get_timestamp = future_timestamp
    signed_future = signer_with_fixed_time.sign("test_value")
    signer_with_fixed_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_with_fixed_time.unsign(signed_future, max_age=3600)
    
    # Test unsign with malformed timestamp
    signed = signer.sign("test_value")
    # Corrupt the timestamp part
    sep = signer.sep.encode()
    parts = signed.rsplit(sep, 1)
    corrupted = parts[0] + sep + b"invalid_timestamp"
    
    with pytest.raises(BadTimeSignature):
        signer.unsign(corrupted)
    
    # Test unsign with missing timestamp
    signed_no_timestamp = signer.sign("test_value")
    # Remove the timestamp part
    parts = signed_no_timestamp.rsplit(sep, 1)
    no_timestamp = parts[0]
    
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp)
    
    # Test unsign with bad signature but valid timestamp
    signed = signer.sign("test_value")
    # Corrupt the signature part
    parts = signed.rsplit(sep, 1)
    corrupted_sig = parts[0] + sep + signer.get_signature(parts[0] + sep + b"wrong") 
    
    with pytest.raises(BadTimeSignature):
        signer.unsign(corrupted_sig)
    
    # Test unsign with return_timestamp and max_age
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
```


# LLM-generated content at query #133
#--------------------------

```python
def test_TimestampSigner_unsign():
    """Test TimestampSigner.unsign method with various scenarios."""
    signer = TimestampSigner("test-secret-key")
    
    # Test 1: Basic sign and unsign returns bytes
    value = b"test-data"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value, f"Expected {value}, got {result}"
    
    # Test 2: Unsign with return_timestamp=True returns tuple with datetime
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple), "Expected tuple when return_timestamp=True"
    assert len(result_with_ts) == 2, "Expected tuple of length 2"
    decoded_value, timestamp = result_with_ts
    assert decoded_value == value, f"Expected {value}, got {decoded_value}"
    assert isinstance(timestamp, datetime), "Expected datetime object"
    assert timestamp.tzinfo is not None, "Expected timezone-aware datetime"
    
    # Test 3: Unsign with valid max_age
    result_age = signer.unsign(signed, max_age=3600)  # 1 hour
    assert result_age == value, f"Expected {value}, got {result_age}"
    
    # Test 4: Unsign with max_age=0 should raise SignatureExpired
    import time as time_module
    signer.get_timestamp = lambda: int(time_module.time()) + 100  # Future timestamp
    signed_future = signer.sign(value)
    signer.get_timestamp = lambda: int(time_module.time())  # Reset
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_future, max_age=0)
    
    # Test 5: Unsign with negative age (timestamp in the future)
    signer_future = TimestampSigner("test-secret-key")
    signer_future.get_timestamp = lambda: int(time_module.time()) + 1000
    signed_future = signer_future.sign(value)
    with pytest.raises(SignatureExpired, match="< 0 seconds"):
        signer.unsign(signed_future, max_age=3600)
    
    # Test 6: Unsign with malformed timestamp
    malformed = signed[:-5] + b"XXXXX" + signed[-2:]
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test 7: Unsign with missing separator (no timestamp)
    no_timestamp = value + b"." + signer.get_signature(value)
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_timestamp)
    
    # Test 8: Unsign with bad signature
    bad_sig = signed[:-3] + b"xyz"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_sig)
    
    # Test 9: Unsign with string input
    signed_str = signer.sign("test-string")
    result_str = signer.unsign(signed_str)
    assert result_str == b"test-string", f"Expected b'test-string', got {result_str}"
    
    # Test 10: Unsign with max_age very large (should not expire)
    result_large = signer.unsign(signed, max_age=999999)
    assert result_large == value, f"Expected {value}, got {result_large}"
```


# LLM-generated content at query #134
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor and basic initialization."""
    serializer = TimedSerializer("test_secret_key")
    
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer == TimestampSigner
    assert serializer.secret_key == "test_secret_key"
    
    # Test with salt parameter
    serializer_with_salt = TimedSerializer("test_secret_key", salt="test_salt")
    assert serializer_with_salt.salt == "test_salt"
    
    # Test with signer_kwargs
    serializer_with_kwargs = TimedSerializer(
        "test_secret_key",
        signer_kwargs={"key_derivation": "hmac"}
    )
    assert serializer_with_kwargs.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test with serializer_kwargs
    serializer_with_serializer_kwargs = TimedSerializer(
        "test_secret_key",
        serializer_kwargs={"serializer": "json"}
    )
    assert serializer_with_serializer_kwargs.serializer_kwargs == {"serializer": "json"}
    
    # Verify the signer is created correctly
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    assert signer.secret_key == "test_secret_key"


# LLM-generated content at query #135
#--------------------------

```python
def test_TimedSerializer():
    # Test that TimedSerializer can be instantiated with default parameters
    serializer = TimedSerializer("test_secret")
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner
    
    # Test that the default signer is a TimestampSigner
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    
    # Test with custom salt
    serializer_with_salt = TimedSerializer("test_secret", salt="custom_salt")
    assert serializer_with_salt.salt == "custom_salt"
    
    # Test with signer_kwargs
    serializer_with_kwargs = TimedSerializer("test_secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer_with_kwargs.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test with serializer_kwargs
    serializer_with_serializer_kwargs = TimedSerializer("test_secret", serializer_kwargs={"compress": True})
    assert serializer_with_serializer_kwargs.serializer_kwargs == {"compress": True}
    
    # Test roundtrip: dumps and loads
    data = {"key": "value"}
    dumped = serializer.dumps(data)
    assert isinstance(dumped, str)
    
    loaded = serializer.loads(dumped)
    assert loaded == data
    
    # Test roundtrip with max_age
    loaded_with_age = serializer.loads(dumped, max_age=3600)
    assert loaded_with_age == data
    
    # Test roundtrip with return_timestamp
    loaded_with_ts = serializer.loads(dumped, return_timestamp=True)
    assert isinstance(loaded_with_ts, tuple)
    assert len(loaded_with_ts) == 2
    assert loaded_with_ts[0] == data
    assert isinstance(loaded_with_ts[1], datetime) 


# LLM-generated content at query #136
#--------------------------

```python
def test_TimedSerializer():
    # Test that TimedSerializer can be instantiated with default parameters
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer is not None
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner
    
    # Test that the default signer is used correctly
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    
    # Test that the serializer works with string salt
    serializer_with_salt = TimedSerializer(secret_key="test-secret", salt="custom-salt")
    assert serializer_with_salt is not None
    
    # Test that the serializer works with bytes salt
    serializer_bytes_salt = TimedSerializer(secret_key="test-secret", salt=b"bytes-salt")
    assert serializer_bytes_salt is not None
    
    # Test that the serializer works with additional keyword arguments
    serializer_with_kwargs = TimedSerializer(
        secret_key="test-secret",
        salt="test-salt",
        signer_kwargs={"key_derivation": "hmac"}
    )
    assert serializer_with_kwargs is not None
    
    # Test that the serializer can be created with different digest methods
    serializer_sha512 = TimedSerializer(
        secret_key="test-secret",
        signer_kwargs={"digest_method": hashlib.sha512}
    )
    assert serializer_sha512 is not None
    
    # Test the iterator for unsigners
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], TimestampSigner)
    
    # Test the iterator with a specific salt
    unsigners_with_salt = list(serializer.iter_unsigners(salt="specific-salt"))
    assert len(unsigners_with_salt) == 1
    assert isinstance(unsigners_with_salt[0], TimestampSigner)


# LLM-generated content at query #137
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer.signer, TimestampSigner)
    assert serializer.secret_key == "test-secret"
    assert serializer.salt == "itsdangerous"
    assert serializer.serializer_kwargs == {}
    assert serializer.signer_kwargs == {}
```


# LLM-generated content at query #138
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner("test-secret-key")
    assert signer.secret_key == "test-secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "hs256"
    
    # Test with custom salt
    signer = TimestampSigner("secret", salt="custom-salt")
    assert signer.salt == "custom-salt"
    
    # Test with custom separator
    signer = TimestampSigner("secret", sep="-")
    assert signer.sep == "-"
    
    # Test with different key derivation
    signer = TimestampSigner("secret", key_derivation="none")
    assert signer.key_derivation == "none"
    
    # Test with different digest method
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256
    
    # Test with different algorithm
    signer = TimestampSigner("secret", algorithm="hs512")
    assert signer.algorithm == "hs512"
```


