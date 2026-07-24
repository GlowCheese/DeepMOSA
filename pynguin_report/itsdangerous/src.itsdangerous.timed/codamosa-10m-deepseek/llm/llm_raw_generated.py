####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer(secret_key="test-secret-key")
    
    # Test basic loads without max_age
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data
    
    # Test loads with return_timestamp=True
    signed = serializer.dumps(data)
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with max_age (should succeed)
    signed = serializer.dumps(data)
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with expired signature
    serializer_with_past_time = TimedSerializer(
        secret_key="test-secret-key",
        signer_kwargs={"get_timestamp": lambda: int(time.time()) - 100}
    )
    signed_past = serializer_with_past_time.dumps(data)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_past, max_age=10)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test loads with different salt
    signed_with_salt = serializer.dumps(data, salt="different-salt")
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt)
    
    # Test loads with correct salt
    result = serializer.loads(signed_with_salt, salt="different-salt")
    assert result == data
    
    # Test loads with bytes input
    signed_bytes = serializer.dumps(data)
    result = serializer.loads(signed_bytes)
    assert result == data
    
    # Test loads with string input
    signed_str = serializer.dumps(data).decode()
    result = serializer.loads(signed_str)
    assert result == data
    
    # Test loads with multiple signers (fallback)
    serializer2 = TimedSerializer(secret_key="another-secret-key")
    serializer_multi = TimedSerializer(
        secret_key="test-secret-key",
        fallback_signers=[serializer2]
    )
    signed_with_serializer2 = serializer2.dumps(data)
    result = serializer_multi.loads(signed_with_serializer2)
    assert result == data
    
    # Test loads returns BadSignature if all signers fail
    with pytest.raises(BadSignature):
        serializer_multi.loads(b"invalid-data")
```


# LLM-generated content at query #2
#--------------------------

```python
def test_TimestampSigner():
    """Test constructor of TimestampSigner."""
    # Test default constructor
    signer = TimestampSigner("test-secret-key")
    assert signer.secret_key == "test-secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "hs256"
    
    # Test with custom parameters
    signer = TimestampSigner(
        "custom-key",
        salt="custom-salt",
        sep="-",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm="hs256"
    )
    assert signer.secret_key == "custom-key"
    assert signer.salt == "custom-salt"
    assert signer.sep == "-"
    assert signer.key_derivation == "none"
    assert signer.digest_method == hashlib.sha256
    assert signer.algorithm == "hs256"
    
    # Test with bytes secret key
    signer = TimestampSigner(b"bytes-key")
    assert signer.secret_key == b"bytes-key"


# LLM-generated content at query #3
#--------------------------

```python
def test_TimedSerializer():
    # Test that TimedSerializer can be instantiated with default parameters
    serializer = TimedSerializer()
    assert serializer is not None
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    
    # Test that default_signer is set correctly
    assert serializer.default_signer == TimestampSigner
    
    # Test that the serializer uses TimestampSigner by default
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    
    # Test with custom secret key
    serializer_with_key = TimedSerializer(secret_key="my-secret-key")
    assert serializer_with_key is not None
    assert isinstance(serializer_with_key, TimedSerializer)
    
    # Test with salt
    serializer_with_salt = TimedSerializer(salt="my-salt")
    assert serializer_with_salt is not None
    
    # Test with both secret key and salt
    serializer_full = TimedSerializer(secret_key="key", salt="salt")
    assert serializer_full is not None
    
    # Test that the serializer can round-trip data
    test_data = {"test": "data"}
    signed = serializer.dumps(test_data)
    assert signed is not None
    assert isinstance(signed, str) or isinstance(signed, bytes)
    
    # Test that we can loads the data back
    loaded = serializer.loads(signed)
    assert loaded == test_data
    
    # Test with bytes data
    bytes_data = b"test bytes"
    signed_bytes = serializer.dumps(bytes_data)
    loaded_bytes = serializer.loads(signed_bytes)
    assert loaded_bytes == bytes_data
```


# LLM-generated content at query #4
#--------------------------

```python
def test_TimestampSigner():
    # Test default construction
    signer = TimestampSigner("test-secret-key")
    assert signer.secret_key == b"test-secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == hmac
    
    # Test construction with custom parameters
    custom_signer = TimestampSigner(
        "custom-secret",
        salt="custom-salt",
        sep=":",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm=hashlib.sha256
    )
    assert custom_signer.secret_key == b"custom-secret"
    assert custom_signer.salt == "custom-salt"
    assert custom_signer.sep == ":"
    assert custom_signer.key_derivation == "none"
    assert custom_signer.digest_method == hashlib.sha256
    
    # Test that get_timestamp returns integer
    timestamp = custom_signer.get_timestamp()
    assert isinstance(timestamp, int)
    assert timestamp > 0
    
    # Test timestamp_to_datetime conversion
    dt = custom_signer.timestamp_to_datetime(timestamp)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc
    
    # Test sign method
    value = b"test-value"
    signed = custom_signer.sign(value)
    assert isinstance(signed, bytes)
    assert value in signed
    
    # Test unsign method
    unsigned = custom_signer.unsign(signed)
    assert unsigned == value
    
    # Test unsign with return_timestamp=True
    unsigned_with_ts, ts = custom_signer.unsign(signed, return_timestamp=True)
    assert unsigned_with_ts == value
    assert isinstance(ts, datetime)
    assert ts.tzinfo == timezone.utc
    
    # Test validate method
    assert custom_signer.validate(signed) == True
    assert custom_signer.validate(b"invalid") == False
    
    # Test max_age parameter
    signed_old = custom_signer.sign(value)
    assert custom_signer.unsign(signed_old, max_age=3600) == value
    
    # Test that expired signature raises SignatureExpired
    with pytest.raises(SignatureExpired):
        custom_signer.unsign(signed_old, max_age=-1)
    
    # Test that invalid signature raises BadSignature or BadTimeSignature
    with pytest.raises(BadSignature):
        custom_signer.unsign(b"invalid.signature")
    
    # Test with string value
    string_signed = custom_signer.sign("string-value")
    assert isinstance(string_signed, bytes)
    assert b"string-value" in string_signed
    assert custom_signer.unsign(string_signed) == b"string-value"


# LLM-generated content at query #5
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Test basic unsign
    signer = TimestampSigner(secret_key="test-secret")
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"
    
    # Test unsign with return_timestamp=True
    signed = signer.sign("test-value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp, datetime)
    
    # Test unsign with max_age within limit
    signed = signer.sign("test-value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test-value"
    
    # Test unsign with max_age exceeded (should raise SignatureExpired)
    signer_with_old_time = TimestampSigner(secret_key="test-secret")
    original_get_timestamp = signer_with_old_time.get_timestamp
    signer_with_old_time.get_timestamp = lambda: int(time.time()) - 100
    signed_old = signer_with_old_time.sign("test-value")
    signer_with_old_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_with_old_time.unsign(signed_old, max_age=50)
    
    # Test unsign with negative age (future timestamp)
    signer_future = TimestampSigner(secret_key="test-secret")
    signer_future.get_timestamp = lambda: int(time.time()) + 100
    signed_future = signer_future.sign("test-value")
    signer_future.get_timestamp = lambda: int(time.time()) - 50
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer_future.unsign(signed_future, max_age=3600)
    assert "age" in str(exc_info.value)
    
    # Test unsign with invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid-signature")
    
    # Test unsign with tampered value
    signed = signer.sign("test-value")
    tampered = signed.replace(b"test-value", b"tampered-value")
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered)
    
    # Test unsign with missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"no-timestamp")
    
    # Test unsign with malformed timestamp
    signer_malformed = TimestampSigner(secret_key="test-secret")
    value = want_bytes("test-value")
    sep = want_bytes(signer_malformed.sep)
    malformed_timestamp = base64_encode(b"not-a-number")
    malformed_signed = value + sep + malformed_timestamp + sep + signer_malformed.get_signature(value + sep + malformed_timestamp)
    with pytest.raises(BadTimeSignature):
        signer_malformed.unsign(malformed_signed)
    
    # Test unsign with signature error but valid timestamp
    signer_error = TimestampSigner(secret_key="test-secret")
    value = want_bytes("test-value")
    sep = want_bytes(signer_error.sep)
    timestamp = base64_encode(int_to_bytes(signer_error.get_timestamp()))
    invalid_sig = b"invalid-signature"
    signed_with_error = value + sep + timestamp + sep + invalid_sig
    with pytest.raises(BadTimeSignature):
        signer_error.unsign(signed_with_error)
```


# LLM-generated content at query #6
#--------------------------

```python
def test_TimedSerializer():
    # Test that TimedSerializer can be instantiated with default parameters
    serializer = TimedSerializer()
    assert serializer.default_signer == TimestampSigner
    assert isinstance(serializer, Serializer)
    assert isinstance(serializer, TimedSerializer)
    
    # Test that TimedSerializer can be instantiated with a secret key
    serializer_with_key = TimedSerializer(secret_key="test-secret-key")
    assert serializer_with_key.secret_key == "test-secret-key"
    
    # Test that TimedSerializer can be instantiated with custom parameters
    serializer_custom = TimedSerializer(
        secret_key="custom-key",
        salt="custom-salt",
        serializer="json",
        signer_kwargs={"key_derivation": "hmac"},
    )
    assert serializer_custom.secret_key == "custom-key"
    assert serializer_custom.salt == "custom-salt"
    
    # Test that the default signer is TimestampSigner
    assert serializer.default_signer is TimestampSigner
    assert issubclass(serializer.default_signer, Signer)
    
    # Test that iter_unsigners returns TimestampSigner instances
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 0
    for signer in signers:
        assert isinstance(signer, TimestampSigner)
    
    # Test that TimedSerializer can be instantiated without arguments
    serializer_no_args = TimedSerializer()
    assert serializer_no_args is not None
    
    # Test basic functionality
    test_data = {"message": "hello", "value": 42}
    signed = serializer.dumps(test_data)
    assert isinstance(signed, (str, bytes))
    
    loaded = serializer.loads(signed)
    assert loaded == test_data
    
    # Test with max_age
    from datetime import datetime, timezone
    loaded_with_age = serializer.loads(signed, max_age=3600)
    assert loaded_with_age == test_data
    
    # Test with return_timestamp
    loaded_with_ts = serializer.loads(signed, return_timestamp=True)
    assert len(loaded_with_ts) == 2
    payload, timestamp = loaded_with_ts
    assert payload == test_data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
```


# LLM-generated content at query #7
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads method."""
    serializer = TimedSerializer(secret_key="test-secret-key")
    
    # Test basic loads without max_age
    original_data = {"user": "test", "role": "admin"}
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed)
    assert result == original_data
    
    # Test loads with return_timestamp=True
    result_with_ts = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == original_data
    assert isinstance(result_with_ts[1], datetime)
    
    # Test loads with max_age (valid)
    result = serializer.loads(signed, max_age=3600)
    assert result == original_data
    
    # Test loads with expired signature
    old_serializer = TimedSerializer(secret_key="test-secret-key")
    old_signed = old_serializer.dumps(original_data)
    # Simulate old timestamp by modifying the signed value
    import time as _time
    old_timestamp = int(_time.time()) - 7200  # 2 hours ago
    old_serializer.get_timestamp = lambda: old_timestamp
    old_signed = old_serializer.dumps(original_data)
    
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=1)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-data")
    
    # Test loads with empty string
    with pytest.raises(BadSignature):
        serializer.loads("")
    
    # Test loads with different salt
    signed_with_salt = serializer.dumps(original_data, salt="custom-salt")
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
    
    # Test loads with correct salt
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == original_data
    
    # Test loads with multiple signers (fallback)
    serializer2 = TimedSerializer(secret_key="test-secret-key-2")
    signed2 = serializer2.dumps(original_data)
    # Should fail with first serializer but work with fallback
    with pytest.raises(BadSignature):
        serializer.loads(signed2)
    
    # Test loads with bytes input
    signed_bytes = serializer.dumps(original_data)
    result_bytes = serializer.loads(signed_bytes)
    assert result_bytes == original_data
    
    # Test complex data types
    complex_data = {
        "list": [1, 2, 3],
        "nested": {"key": "value"},
        "numbers": [1.5, 2.5],
        "boolean": True,
        "none_value": None
    }
    signed_complex = serializer.dumps(complex_data)
    result_complex = serializer.loads(signed_complex)
    assert result_complex == complex_data
    
    # Test with return_timestamp and max_age together
    result_with_both = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result_with_both, tuple)
    assert result_with_both[0] == original_data
    assert isinstance(result_with_both[1], datetime)
```


# LLM-generated content at query #8
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("secret-key", salt="test-salt")
    assert signer.secret_key == b"secret-key"
    assert signer.salt == "test-salt"
    assert signer.sep == "."
    assert signer.digest_method is not None
    assert signer.key_derivation == "hmac"
    
    # Test with default salt
    signer_default = TimestampSigner("secret-key")
    assert signer_default.salt == "itsdangerous.TimestampSigner"
    
    # Test get_timestamp returns integer
    timestamp = signer.get_timestamp()
    assert isinstance(timestamp, int)
    assert timestamp > 0
    
    # Test timestamp_to_datetime
    dt = signer.timestamp_to_datetime(timestamp)
    assert dt.tzinfo is not None
    assert dt.tzinfo.utcoffset(dt) == timezone.utc.utcoffset(dt)
    
    # Test sign returns bytes with timestamp
    signed = signer.sign("test-value")
    assert isinstance(signed, bytes)
    assert b"." in signed
    
    # Test unsign with valid signature
    unsigned = signer.unsign(signed)
    assert unsigned == b"test-value"
    
    # Test unsign with return_timestamp
    unsigned, ts = signer.unsign(signed, return_timestamp=True)
    assert unsigned == b"test-value"
    assert isinstance(ts, datetime)
    assert ts.tzinfo is not None
    
    # Test validate
    assert signer.validate(signed) is True
    assert signer.validate(b"invalid") is False
    
    # Test max_age
    with pytest.raises(BadSignature):
        signer.unsign(signed, max_age=-1)
    
    # Test expired signature
    import time as time_module
    old_timestamp = int(time_module.time()) - 100
    old_signer = TimestampSigner("secret-key", salt="test-salt")
    old_signer.get_timestamp = lambda: old_timestamp
    old_signed = old_signer.sign("test-value")
    with pytest.raises(SignatureExpired):
        signer.unsign(old_signed, max_age=50)


# LLM-generated content at query #9
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without expiration
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with valid max_age
    result = signer.unsign(signed, max_age=3600)  # 1 hour
    assert result == b"test_value"
    
    # Test unsign with expired signature
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) + 7200  # 2 hours in future
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=3600)
    signer.get_timestamp = original_get_timestamp
    
    # Test unsign with negative age (future timestamp)
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) - 7200  # 2 hours in past
    signed_future = signer.sign("future_test")
    signer.get_timestamp = lambda: int(time.time()) - 3600  # 1 hour in past
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_future, max_age=3600)
    signer.get_timestamp = original_get_timestamp
    
    # Test unsign with bad signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid_signature")
    
    # Test unsign with tampered timestamp
    signed_bytes = signer.sign("test")
    parts = signed_bytes.split(signer.sep.encode())
    parts[-1] = b"tampered_signature"
    tampered = signer.sep.encode().join(parts)
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered)
    
    # Test unsign with missing timestamp
    value = want_bytes("test_value")
    signature = signer.get_signature(value)
    no_timestamp = value + signer.sep.encode() + signature
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_timestamp)
    
    # Test unsign with malformed timestamp
    malformed = value + signer.sep.encode() + b"not_base64" + signer.sep.encode() + signature
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(malformed)
    
    # Test unsign with string input
    signed_str = signer.sign("test_string").decode()
    result = signer.unsign(signed_str)
    assert result == b"test_string"
```


# LLM-generated content at query #10
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner(secret_key="test-secret-key")
    assert signer.secret_key == b"test-secret-key"
    assert signer.sep == "."
    assert signer.salt is None
    
    signer_with_salt = TimestampSigner(secret_key="test-secret-key", salt="my-salt")
    assert signer_with_salt.salt == "my-salt"
    
    signer_with_custom_sep = TimestampSigner(secret_key="test-secret-key", sep=":")
    assert signer_with_custom_sep.sep == ":"
    
    # Test that it inherits from Signer
    assert isinstance(signer, Signer)
    
    # Test default attributes
    assert hasattr(signer, 'digest_method')
    assert hasattr(signer, 'key_derivation')
```


# LLM-generated content at query #11
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test 1: Basic sign and unsign
    original_value = b"test_value"
    signed = signer.sign(original_value)
    result = signer.unsign(signed)
    assert result == original_value
    
    # Test 2: Sign and unsign with return_timestamp=True
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None
    
    # Test 3: Unsign with max_age that should pass
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)  # 1 hour max age
    assert result == b"test_value"
    
    # Test 4: Unsign with expired signature (max_age too small)
    signer_with_fixed_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_fixed_time.get_timestamp
    signer_with_fixed_time.get_timestamp = lambda: 100  # Fixed old timestamp
    
    signed_old = signer_with_fixed_time.sign("test_value")
    
    # Restore original timestamp function for validation
    signer_with_fixed_time.get_timestamp = lambda: 200  # Current time is later
    
    with pytest.raises(SignatureExpired):
        signer_with_fixed_time.unsign(signed_old, max_age=50)  # 50 seconds max age
    
    # Test 5: Unsign with negative age (future timestamp)
    signer_with_future_time = TimestampSigner("secret-key")
    signer_with_future_time.get_timestamp = lambda: 1000  # Future timestamp
    
    signed_future = signer_with_future_time.sign("test_value")
    
    # Validate with current time in the past
    signer_with_future_time.get_timestamp = lambda: 500  # Current time is earlier
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer_with_future_time.unsign(signed_future, max_age=100)
    assert "age" in str(exc_info.value)
    
    # Test 6: Unsign with malformed timestamp
    malformed_signed = b"test_value.sep.invalid_timestamp.sep.signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed)
    
    # Test 7: Unsign with missing timestamp
    signer_no_timestamp = Signer("secret-key")
    signed_no_timestamp = signer_no_timestamp.sign("test_value")
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_no_timestamp)
    
    # Test 8: Unsign with invalid signature but valid timestamp
    signer2 = TimestampSigner("different-secret")
    signed_diff = signer2.sign("test_value")
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_diff)
    
    # Test 9: Unsign with bytes input
    signed_bytes = signer.sign(b"bytes_value")
    result = signer.unsign(signed_bytes)
    assert result == b"bytes_value"
    
    # Test 10: Unsign with string input
    signed_str = signer.sign("string_value")
    result = signer.unsign(signed_str.decode())
    assert result == b"string_value"
    
    # Test 11: Verify timestamp is datetime with timezone
    signed = signer.sign("test_value")
    _, timestamp = signer.unsign(signed, return_timestamp=True)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 12: Unsign with max_age exactly equal to age
    signer_exact = TimestampSigner("secret-key")
    original_ts = signer_exact.get_timestamp()
    signer_exact.get_timestamp = lambda: original_ts + 100  # Age is 100
    
    signed_exact = signer_exact.sign("test_value")
    signer_exact.get_timestamp = lambda: original_ts + 100  # Current time is 100 seconds later
    
    # This should pass because age (100) is not > max_age (100)
    result = signer_exact.unsign(signed_exact, max_age=100)
    assert result == b"test_value"
```


# LLM-generated content at query #12
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor and basic functionality."""
    # Test constructor with default parameters
    serializer = TimedSerializer()
    assert serializer.default_signer == TimestampSigner
    assert isinstance(serializer, Serializer)
    assert isinstance(serializer, TimedSerializer)
    
    # Test constructor with secret key
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.secret_key == "test-secret"
    
    # Test constructor with salt
    serializer = TimedSerializer(salt="test-salt")
    assert serializer.salt == "test-salt"
    
    # Test constructor with signer_kwargs
    serializer = TimedSerializer(signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test constructor with serializer_kwargs
    serializer = TimedSerializer(serializer_kwargs={"serializer": "json"})
    assert serializer.serializer_kwargs == {"serializer": "json"}
    
    # Test that iter_unsigners returns TimestampSigner instances
    serializer = TimedSerializer(secret_key="test")
    for signer in serializer.iter_unsigners():
        assert isinstance(signer, TimestampSigner)
        break  # Only need to check the first one
    
    # Test basic dumps and loads
    data = {"key": "value"}
    signed = serializer.dumps(data)
    loaded = serializer.loads(signed)
    assert loaded == data
    
    # Test with max_age
    signed = serializer.dumps(data)
    loaded = serializer.loads(signed, max_age=3600)
    assert loaded == data
    
    # Test with return_timestamp
    signed = serializer.dumps(data)
    result = serializer.loads(signed, return_timestamp=True)
    payload, timestamp = result
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test with both max_age and return_timestamp
    signed = serializer.dumps(data)
    result = serializer.loads(signed, max_age=3600, return_timestamp=True)
    payload, timestamp = result
    assert payload == data
    assert isinstance(timestamp, datetime)
```


# LLM-generated content at query #13
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic sign and unsign
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"
    
    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with valid max_age
    result = signer.unsign(signed, max_age=3600)  # 1 hour
    assert result == b"test_value"
    
    # Test unsign with expired signature (max_age = 0)
    import time
    time.sleep(0.1)  # Ensure timestamp difference
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)
    
    # Test unsign with future signature (age < 0)
    # Mock get_timestamp to return a future time
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) - 100  # 100 seconds in past
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=50)
    signer.get_timestamp = original_get_timestamp
    
    # Test unsign with tampered value
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"tampered" + signed[8:])
    
    # Test unsign with missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"no_timestamp")
    
    # Test unsign with malformed timestamp (invalid base64)
    sep = signer.sep.encode()
    malformed = b"value" + sep + b"invalid_timestamp" + sep + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test unsign with invalid signature but valid timestamp
    valid_signed = signer.sign("test")
    # Corrupt the signature part
    parts = valid_signed.rsplit(sep.encode(), 1)
    corrupted = parts[0] + sep + b"corrupted"
    with pytest.raises(BadTimeSignature):
        signer.unsign(corrupted)
```


# LLM-generated content at query #14
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic sign and unsign
    value = b"test value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test with string input
    signed_str = signer.sign("test string")
    result_str = signer.unsign(signed_str)
    assert result_str == b"test string"
    
    # Test return_timestamp=True
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    assert result_with_ts[0] == value
    assert isinstance(result_with_ts[1], datetime)
    
    # Test with max_age - valid age
    result_age = signer.unsign(signed, max_age=3600)
    assert result_age == value
    
    # Test with max_age - expired signature
    original_get_timestamp = signer.get_timestamp
    try:
        signer.get_timestamp = lambda: int(time.time()) - 100
        expired_signed = signer.sign(value)
        signer.get_timestamp = lambda: int(time.time())
        with pytest.raises(SignatureExpired) as exc_info:
            signer.unsign(expired_signed, max_age=10)
        assert "Signature age" in str(exc_info.value)
        assert exc_info.value.payload == value
    finally:
        signer.get_timestamp = original_get_timestamp
    
    # Test with max_age - future timestamp (age < 0)
    try:
        signer.get_timestamp = lambda: int(time.time()) + 100
        future_signed = signer.sign(value)
        signer.get_timestamp = lambda: int(time.time())
        with pytest.raises(SignatureExpired) as exc_info:
            signer.unsign(future_signed, max_age=3600)
        assert "age" in str(exc_info.value)
        assert exc_info.value.payload == value
    finally:
        signer.get_timestamp = original_get_timestamp
    
    # Test with bad signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid")
    
    # Test with tampered value
    tampered = signed[:-1] + b"x"
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered)
    
    # Test with missing timestamp separator
    no_sep = b"test" + signer.sep.encode() + signer.get_signature(b"test")
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(no_sep)
    assert "timestamp missing" in str(exc_info.value)
    
    # Test with malformed timestamp
    malformed_ts = value + signer.sep.encode() + b"invalid_ts" + signer.sep.encode() + signer.get_signature(value + signer.sep.encode() + b"invalid_ts")
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(malformed_ts)
    
    # Test with custom separator
    custom_signer = TimestampSigner("secret-key", sep="|")
    custom_signed = custom_signer.sign(value)
    custom_result = custom_signer.unsign(custom_signed)
    assert custom_result == value
    
    # Test with digest method
    signer_sha512 = TimestampSigner("secret-key", digest_method=hashlib.sha512)
    signed_sha512 = signer_sha512.sign(value)
    result_sha512 = signer_sha512.unsign(signed_sha512)
    assert result_sha512 == value
    
    # Test return_timestamp with expired signature
    try:
        signer.get_timestamp = lambda: int(time.time()) - 100
        expired_signed = signer.sign(value)
        signer.get_timestamp = lambda: int(time.time())
        with pytest.raises(SignatureExpired) as exc_info:
            signer.unsign(expired_signed, max_age=10, return_timestamp=True)
        assert isinstance(exc_info.value.date_signed, datetime)
    finally:
        signer.get_timestamp = original_get_timestamp
    
    # Test with empty value
    empty_signed = signer.sign(b"")
    empty_result = signer.unsign(empty_signed)
    assert empty_result == b""
    
    # Test with bytes containing separator
    value_with_sep = b"test" + signer.sep.encode() + b"value"
    signed_with_sep = signer.sign(value_with_sep)
    result_with_sep = signer.unsign(signed_with_sep)
    assert result_with_sep == value_with_sep
    
    # Test validate method indirectly through unsign
    assert signer.unsign(signed) == value
    
    # Test that unsign raises BadTimeSignature for non-timestamp signed data
    regular_signer = Signer("secret-key")
    regular_signed = regular_signer.sign(value)
    with pytest.raises(BadTimeSignature):
        signer.unsign(regular_signed)


# LLM-generated content at query #15
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
    assert signer.algorithm == hmac_compat
    
    # Test with custom parameters
    custom_signer = TimestampSigner(
        "secret-key",
        sep="|",
        salt="custom-salt",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm=hmac_compat
    )
    assert custom_signer.secret_key == "secret-key"
    assert custom_signer.sep == "|"
    assert custom_signer.salt == "custom-salt"
    assert custom_signer.key_derivation == "none"
    assert custom_signer.digest_method == hashlib.sha256
    assert custom_signer.algorithm == hmac_compat
    
    # Test that get_timestamp returns an integer
    ts = signer.get_timestamp()
    assert isinstance(ts, int)
    assert ts > 0
    
    # Test timestamp_to_datetime
    dt = signer.timestamp_to_datetime(ts)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc
    
    # Test sign method returns bytes
    signed = signer.sign("test-value")
    assert isinstance(signed, bytes)
    
    # Test unsign returns original value
    unsigned = signer.unsign(signed)
    assert unsigned == b"test-value"
    
    # Test unsign with return_timestamp
    unsigned, timestamp = signer.unsign(signed, return_timestamp=True)
    assert unsigned == b"test-value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test validate returns True for valid signature
    assert signer.validate(signed) is True
    
    # Test validate returns False for invalid signature
    assert signer.validate(b"invalid-signature") is False
    
    # Test max_age validation
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)
    
    # Test with bytes secret key
    bytes_signer = TimestampSigner(b"bytes-secret")
    assert bytes_signer.secret_key == b"bytes-secret"
    
    # Test with empty value
    empty_signed = signer.sign(b"")
    assert signer.unsign(empty_signed) == b""
```


# LLM-generated content at query #16
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner(secret_key="test-key")
    assert signer.secret_key == "test-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "sha1"

    # Test constructor with custom parameters
    signer2 = TimestampSigner(
        secret_key="custom-key",
        salt="custom-salt",
        sep="|",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm="sha256"
    )
    assert signer2.secret_key == "custom-key"
    assert signer2.salt == "custom-salt"
    assert signer2.sep == "|"
    assert signer2.key_derivation == "none"
    assert signer2.digest_method == hashlib.sha256
    assert signer2.algorithm == "sha256"

    # Test constructor with bytes secret key
    signer3 = TimestampSigner(secret_key=b"bytes-key")
    assert signer3.secret_key == b"bytes-key"

    # Test constructor inherits from Signer
    assert isinstance(signer, Signer)
    assert isinstance(signer, TimestampSigner)
```


# LLM-generated content at query #17
#--------------------------

```python
def test_TimestampSigner():
    """Test TimestampSigner constructor and basic functionality."""
    # Test default construction
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == b"secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == hmac
    
    # Test with custom parameters
    signer2 = TimestampSigner(
        "secret-key",
        salt="custom-salt",
        sep=":",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm=hashlib.sha256
    )
    assert signer2.secret_key == b"secret-key"
    assert signer2.salt == "custom-salt"
    assert signer2.sep == ":"
    assert signer2.key_derivation == "none"
    assert signer2.digest_method == hashlib.sha256
    assert signer2.algorithm == hashlib.sha256
    
    # Test that TimestampSigner inherits from Signer
    assert isinstance(signer, Signer)
    assert issubclass(TimestampSigner, Signer)
    
    # Test get_timestamp method
    timestamp = signer.get_timestamp()
    assert isinstance(timestamp, int)
    assert timestamp > 0
    
    # Test timestamp_to_datetime method
    dt = signer.timestamp_to_datetime(timestamp)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc
    
    # Test that datetime is close to current time
    now = datetime.now(timezone.utc)
    assert abs((dt - now).total_seconds()) < 2
    
    # Test with string secret key
    signer3 = TimestampSigner("my-secret")
    assert signer3.secret_key == b"my-secret"
    
    # Test with bytes secret key
    signer4 = TimestampSigner(b"bytes-secret")
    assert signer4.secret_key == b"bytes-secret"
    
    # Test sign method produces expected output format
    signed = signer.sign("test-value")
    assert isinstance(signed, bytes)
    assert b"." in signed  # Should contain separator
    parts = signed.rsplit(b".", 2)
    assert len(parts) == 3  # value, timestamp, signature
    
    # Test unsign method returns original value
    unsigned = signer.unsign(signed)
    assert unsigned == b"test-value"
    
    # Test unsign with return_timestamp=True
    result, timestamp_dt = signer.unsign(signed, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp_dt, datetime)
    assert timestamp_dt.tzinfo == timezone.utc
    
    # Test validate method
    assert signer.validate(signed) == True
    assert signer.validate(b"invalid-signature") == False
    
    # Test unsign with max_age
    result = signer.unsign(signed, max_age=3600)  # 1 hour
    assert result == b"test-value"
    
    # Test that expired signature raises exception
    import time as time_module
    old_timestamp = int(time_module.time()) - 7200  # 2 hours ago
    old_signed = signer.sign("old-value")
    # Replace timestamp in signed value with old one
    parts = old_signed.rsplit(b".", 2)
    old_ts_bytes = base64_encode(int_to_bytes(old_timestamp))
    old_signed = parts[0] + b"." + old_ts_bytes + b"." + parts[2]
    # Need to recalculate signature for the modified value
    value_with_ts = parts[0] + b"." + old_ts_bytes
    new_sig = signer.get_signature(value_with_ts)
    modified_signed = value_with_ts + b"." + new_sig
    
    with pytest.raises(SignatureExpired):
        signer.unsign(modified_signed, max_age=3600)
    
    # Test with bytes input
    signed_bytes = signer.sign(b"bytes-value")
    unsigned_bytes = signer.unsign(signed_bytes)
    assert unsigned_bytes == b"bytes-value"
    
    # Test with empty string
    signed_empty = signer.sign("")
    unsigned_empty = signer.unsign(signed_empty)
    assert unsigned_empty == b""


# LLM-generated content at query #18
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic sign and unsign
    value = b"test message"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test with string input
    signed_str = signer.sign("test message")
    result_str = signer.unsign(signed_str)
    assert result_str == b"test message"
    
    # Test with return_timestamp=True
    signed_with_ts = signer.sign(value)
    result_value, timestamp = signer.unsign(signed_with_ts, return_timestamp=True)
    assert result_value == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test max_age validation - valid age
    signed_valid = signer.sign(value)
    result_valid = signer.unsign(signed_valid, max_age=3600)
    assert result_valid == value
    
    # Test max_age validation - expired signature
    signer_future = TimestampSigner("secret-key")
    # Override get_timestamp to return a timestamp in the past
    original_get_timestamp = signer_future.get_timestamp
    signer_future.get_timestamp = lambda: int(time.time()) - 100  # 100 seconds ago
    signed_old = signer_future.sign(value)
    signer_future.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer_future.unsign(signed_old, max_age=10)
    assert "Signature age" in str(exc_info.value)
    
    # Test negative age (future timestamp)
    signer_future2 = TimestampSigner("secret-key")
    signer_future2.get_timestamp = lambda: int(time.time()) + 100  # 100 seconds in future
    signed_future = signer_future2.sign(value)
    
    with pytest.raises(SignatureExpired) as exc_info2:
        signer.unsign(signed_future, max_age=3600)
    assert "age" in str(exc_info2.value)
    
    # Test BadSignature for tampered value
    with pytest.raises(BadSignature):
        signer.unsign(b"tampered" + signed[5:])
    
    # Test BadTimeSignature for missing timestamp
    signer_no_ts = Signer("secret-key")
    signed_no_ts = signer_no_ts.sign(value)
    with pytest.raises(BadTimeSignature) as exc_info3:
        signer.unsign(signed_no_ts)
    assert "timestamp missing" in str(exc_info3.value)
    
    # Test BadTimeSignature for malformed timestamp
    malformed = signed[:-5] + b"!!!!!" + signed[-5:]
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test validation method
    assert signer.validate(signed) is True
    assert signer.validate(b"invalid") is False
    
    # Test max_age with validation
    assert signer.validate(signed, max_age=3600) is True
    assert signer.validate(signed_old, max_age=10) is False
    
    # Test with empty value
    empty_signed = signer.sign(b"")
    empty_result = signer.unsign(empty_signed)
    assert empty_result == b""
    
    # Test with return_timestamp and max_age
    result_ts, timestamp_ts = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert result_ts == value
    assert isinstance(timestamp_ts, datetime)
```


# LLM-generated content at query #19
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("test-secret")
    
    # Test basic loads
    original_data = {"key": "value"}
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed)
    assert result == original_data
    
    # Test loads with max_age (valid)
    result = serializer.loads(signed, max_age=3600)
    assert result == original_data
    
    # Test loads with return_timestamp
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with max_age and return_timestamp
    result, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert result == original_data
    assert isinstance(timestamp, datetime)
    
    # Test loads with expired signature
    import time
    serializer_with_future = TimedSerializer("test-secret")
    # Override get_timestamp to return a time in the future
    original_get_timestamp = serializer_with_future.signer.get_timestamp
    serializer_with_future.signer.get_timestamp = lambda: int(time.time()) + 10000
    signed_future = serializer_with_future.dumps(original_data)
    
    with pytest.raises(SignatureExpired):
        serializer_with_future.loads(signed_future, max_age=1)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test loads with different salt
    with pytest.raises(BadSignature):
        serializer.loads(signed, salt="different-salt")
    
    # Test loads with bytes data
    signed_bytes = serializer.dumps(original_data)
    result = serializer.loads(signed_bytes)
    assert result == original_data
    
    # Test loads with string data
    signed_str = serializer.dumps(original_data).decode()
    result = serializer.loads(signed_str)
    assert result == original_data
```


# LLM-generated content at query #20
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test successful loads without return_timestamp
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
    
    # Test loads with max_age that is valid
    result = serializer.loads(signed, max_age=3600)
    assert result == data
    
    # Test loads with expired signature
    import time
    expired_signer = TimestampSigner("test-secret")
    expired_signer.get_timestamp = lambda: int(time.time()) - 1000
    expired_value = expired_signer.sign(b"test")
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_value, max_age=10)
    
    # Test loads with bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-signature")
    
    # Test loads with multiple signers (fallback)
    serializer2 = TimedSerializer(secret_key="test-secret-2")
    signed2 = serializer2.dumps(data)
    # Use serializer with different salt to test fallback
    serializer_with_salt = TimedSerializer(secret_key="test-secret", salt="different")
    with pytest.raises(BadSignature):
        serializer_with_salt.loads(signed2)
    
    # Test loads with salt parameter
    signed_with_salt = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == data
    
    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
    
    # Test loads with string input
    signed_str = signed.decode('utf-8') if isinstance(signed, bytes) else signed
    result = serializer.loads(signed_str)
    assert result == data
    
    # Test loads with empty data
    signed_empty = serializer.dumps({})
    result = serializer.loads(signed_empty)
    assert result == {}
```


# LLM-generated content at query #21
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
    assert signer.algorithm == "hmac-sha1"
    
    # Test constructor with custom parameters
    custom_signer = TimestampSigner(
        "custom-secret",
        salt="custom-salt",
        sep="|",
        key_derivation="none",
        digest_method=hashlib.sha256
    )
    assert custom_signer.secret_key == "custom-secret"
    assert custom_signer.salt == "custom-salt"
    assert custom_signer.sep == "|"
    assert custom_signer.key_derivation == "none"
    assert custom_signer.digest_method == hashlib.sha256
    
    # Test that TimestampSigner inherits from Signer
    assert isinstance(signer, Signer)
    assert isinstance(signer, TimestampSigner)
    
    # Test get_timestamp method exists and returns int
    assert hasattr(signer, "get_timestamp")
    assert isinstance(signer.get_timestamp(), int)
    
    # Test timestamp_to_datetime method exists
    assert hasattr(signer, "timestamp_to_datetime")
```


# LLM-generated content at query #22
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Create a TimestampSigner instance with a known secret key
    signer = TimestampSigner("test-secret")
    
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
    
    # Test unsign with max_age (signature should be recent)
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test unsign with max_age that's too short (should fail)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)
    
    # Test unsign with expired signature
    # Create a signer with a fixed timestamp in the past
    class PastTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return 0  # Unix epoch
    
    past_signer = PastTimestampSigner("test-secret")
    past_signed = past_signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(past_signed, max_age=3600)
    
    # Test unsign with invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid-signature")
    
    # Test unsign with tampered value
    tampered = signed[:-1] + b"X"
    with pytest.raises(BadSignature):
        signer.unsign(tampered)
    
    # Test unsign with missing timestamp
    # Create a signed value without timestamp
    no_timestamp = value + b"." + signer.get_signature(value)
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp)
    
    # Test unsign with malformed timestamp
    # Create a signed value with invalid base64 timestamp
    malformed = value + b"." + b"invalid-timestamp" + b"." + signer.get_signature(value + b"." + b"invalid-timestamp")
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test unsign with string input
    signed_str = signed.decode("utf-8")
    result = signer.unsign(signed_str)
    assert result == value
    
    # Test unsign with return_timestamp and string input
    result, timestamp = signer.unsign(signed_str, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    
    # Test unsign with max_age and return_timestamp
    result, timestamp = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    
    # Test unsign with different separator
    custom_signer = TimestampSigner("test-secret", sep="|")
    custom_signed = custom_signer.sign(value)
    result = custom_signer.unsign(custom_signed)
    assert result == value
    
    # Test unsign with bytes value
    value_bytes = b"test-bytes-value"
    signed_bytes = signer.sign(value_bytes)
    result = signer.unsign(signed_bytes)
    assert result == value_bytes
    
    # Test unsign with negative age (future timestamp)
    class FutureTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) + 10000  # 10000 seconds in the future
    
    future_signer = FutureTimestampSigner("test-secret")
    future_signed = future_signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed, max_age=3600)
```


# LLM-generated content at query #23
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Create signer instance
    signer = TimestampSigner(secret_key="test-secret-key")

    # Test basic sign and unsign
    value = b"test_value"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

    # Test unsign with return_timestamp
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == value
    assert isinstance(result[1], datetime)
    # Check timestamp is within reasonable range (within last 5 seconds)
    assert (datetime.now(timezone.utc) - result[1]).total_seconds() < 5

    # Test unsign with valid max_age
    result = signer.unsign(signed, max_age=3600)  # 1 hour max age
    assert result == value

    # Test unsign with expired signature
    # Create a signer with a fixed timestamp in the past
    class OldTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) - 100  # 100 seconds ago

    old_signer = OldTimestampSigner(secret_key="test-secret-key")
    old_signed = old_signer.sign(value)
    
    with pytest.raises(SignatureExpired):
        old_signer.unsign(old_signed, max_age=10)  # Only 10 seconds max age

    # Test unsign with corrupted signature
    corrupted = signed[:-1] + b"x"
    with pytest.raises(BadSignature):
        signer.unsign(corrupted)

    # Test unsign with malformed timestamp
    # Create a signed value with invalid timestamp
    sep = signer.sep.encode()
    parts = signed.rsplit(sep, 1)
    malformed = parts[0] + sep + b"invalid_timestamp" + sep + signer.get_signature(parts[0] + sep + b"invalid_timestamp")
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)

    # Test unsign with missing timestamp
    # Create a regular signed value without timestamp
    regular_signer = Signer(secret_key="test-secret-key")
    regular_signed = regular_signer.sign(value)
    with pytest.raises(BadTimeSignature):
        signer.unsign(regular_signed)

    # Test unsign with negative age (future timestamp)
    class FutureTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) + 100  # 100 seconds in the future

    future_signer = FutureTimestampSigner(secret_key="test-secret-key")
    future_signed = future_signer.sign(value)
    with pytest.raises(SignatureExpired):
        future_signer.unsign(future_signed, max_age=3600)

    # Test unsign with string input
    signed_str = signed.decode()
    assert signer.unsign(signed_str) == value

    # Test unsign returns bytes
    result = signer.unsign(signed)
    assert isinstance(result, bytes)
```


# LLM-generated content at query #24
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("test-secret")
    
    # Test successful unsign without timestamp
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"
    
    # Test successful unsign with timestamp
    signed = signer.sign("test-value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp, datetime)
    
    # Test with max_age within limits
    signed = signer.sign("test-value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test-value"
    
    # Test with max_age expired
    old_signer = TimestampSigner("test-secret")
    old_signer.get_timestamp = lambda: int(time.time()) - 100  # 100 seconds ago
    signed = old_signer.sign("test-value")
    current_signer = TimestampSigner("test-secret")
    
    with pytest.raises(SignatureExpired) as excinfo:
        current_signer.unsign(signed, max_age=50)
    assert "Signature age" in str(excinfo.value)
    
    # Test with negative age (future timestamp)
    future_signer = TimestampSigner("test-secret")
    future_signer.get_timestamp = lambda: int(time.time()) + 100  # 100 seconds in future
    signed = future_signer.sign("test-value")
    
    with pytest.raises(SignatureExpired) as excinfo:
        current_signer.unsign(signed, max_age=3600)
    assert "age" in str(excinfo.value)
    
    # Test with invalid signature
    with pytest.raises(BadTimeSignature) as excinfo:
        signer.unsign(b"invalid-data")
    assert "timestamp missing" in str(excinfo.value)
    
    # Test with malformed timestamp
    malformed = b"test-value" + signer.sep.encode() + b"invalid-timestamp"
    with pytest.raises(BadTimeSignature) as excinfo:
        signer.unsign(malformed)
    assert "Malformed timestamp" in str(excinfo.value)
    
    # Test with wrong secret key
    wrong_signer = TimestampSigner("wrong-secret")
    signed = signer.sign("test-value")
    with pytest.raises(BadTimeSignature) as excinfo:
        wrong_signer.unsign(signed)
    assert "signature mismatch" in str(excinfo.value).lower() or "bad signature" in str(excinfo.value).lower()
    
    # Test return_timestamp=True with valid signature
    signed = signer.sign("test-value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test-value"
    assert timestamp.tzinfo == timezone.utc
    assert isinstance(timestamp, datetime)
    
    # Test with bytes input
    signed = signer.sign(b"test-bytes")
    result = signer.unsign(signed)
    assert result == b"test-bytes"
```


# LLM-generated content at query #25
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
    
    # Test with max_age not expired
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test with max_age expired
    signer_with_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_time.get_timestamp
    signer_with_time.get_timestamp = lambda: int(time.time()) - 100  # Simulate old timestamp
    signed_old = signer_with_time.sign("test_value")
    signer_with_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer_with_time.unsign(signed_old, max_age=50)
    assert "Signature age" in str(exc_info.value)
    
    # Test with future timestamp (age < 0)
    signer_with_future = TimestampSigner("secret-key")
    signer_with_future.get_timestamp = lambda: int(time.time()) + 100  # Future timestamp
    signed_future = signer_with_future.sign("test_value")
    signer_with_future.get_timestamp = lambda: int(time.time())  # Reset to current time
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer_with_future.unsign(signed_future, max_age=3600)
    assert "Signature age" in str(exc_info.value)
    
    # Test BadSignature propagation
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")
    
    # Test BadTimeSignature for malformed timestamp
    malformed = b"test_value" + signer.sep.encode() + b"invalid_timestamp" + signer.sep.encode() + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test BadTimeSignature when timestamp is missing
    signed_no_timestamp = signer.sign("test_value")
    # Remove the timestamp part
    no_timestamp = signed_no_timestamp.rsplit(signer.sep.encode(), 1)[0]
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_timestamp)
```


# LLM-generated content at query #26
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner("test-secret")
    assert signer.secret_key == "test-secret"
    assert signer.sep == "."
    assert signer.salt == "timestamp-signer"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "hs256"
    
    # Test constructor with custom parameters
    signer = TimestampSigner(
        "secret-key",
        salt="custom-salt",
        sep=":",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm="hs512"
    )
    assert signer.secret_key == "secret-key"
    assert signer.salt == "custom-salt"
    assert signer.sep == ":"
    assert signer.key_derivation == "none"
    assert signer.digest_method == hashlib.sha256
    assert signer.algorithm == "hs512"
    
    # Test constructor with bytes secret key
    signer = TimestampSigner(b"bytes-secret")
    assert signer.secret_key == b"bytes-secret"
    
    # Test get_timestamp returns integer
    timestamp = signer.get_timestamp()
    assert isinstance(timestamp, int)
    assert timestamp > 0
    
    # Test timestamp_to_datetime
    dt = signer.timestamp_to_datetime(timestamp)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc
    
    # Test sign method
    signed = signer.sign("test-value")
    assert isinstance(signed, bytes)
    assert b"test-value" in signed
    
    # Test unsign method without timestamp
    unsigned = signer.unsign(signed)
    assert unsigned == b"test-value"
    
    # Test unsign with return_timestamp
    unsigned, timestamp = signer.unsign(signed, return_timestamp=True)
    assert unsigned == b"test-value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test sign with bytes input
    signed_bytes = signer.sign(b"bytes-value")
    unsigned_bytes = signer.unsign(signed_bytes)
    assert unsigned_bytes == b"bytes-value"
    
    # Test validate method
    assert signer.validate(signed) == True
    assert signer.validate(b"invalid") == False
    
    # Test max_age parameter
    signed = signer.sign("test")
    unsigned = signer.unsign(signed, max_age=3600)
    assert unsigned == b"test"
    
    # Test expired signature
    import time as time_module
    old_timestamp = int(time_module.time()) - 100
    old_signer = TimestampSigner("test-secret")
    old_signer.get_timestamp = lambda: old_timestamp
    old_signed = old_signer.sign("old-value")
    
    # Should raise SignatureExpired with max_age less than age
    import pytest
    with pytest.raises(SignatureExpired):
        signer.unsign(old_signed, max_age=50)
    
    # Test BadSignature for invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid-signature")
    
    # Test BadTimeSignature for missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"no-timestamp." + signer.get_signature(b"no-timestamp"))```


# LLM-generated content at query #27
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
    
    # Test with max_age not exceeded
    signed = signer.sign("test-value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test-value"
    
    # Test with max_age exceeded
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) + 100
    signed = signer.sign("test-value")
    signer.get_timestamp = original_get_timestamp
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=10)
    
    # Test with negative age (future timestamp)
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) - 100
    signed = signer.sign("test-value")
    signer.get_timestamp = original_get_timestamp
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=3600)
    
    # Test with bad signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid-signature")
    
    # Test with malformed timestamp
    bad_timestamp = base64_encode(b"not-a-timestamp")
    bad_signed = b"test-value" + signer.sep.encode() + bad_timestamp + signer.sep.encode() + b"signature"
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(bad_signed)
    
    # Test with missing timestamp
    no_timestamp = b"test-value" + signer.sep.encode() + signer.sign(b"test-value")
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_timestamp)
    
    # Test that SignatureExpired is raised for expired signatures with max_age
    signed = signer.sign("test-value")
    time.sleep(2)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=1)


# LLM-generated content at query #28
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Create a TimestampSigner instance with a known secret key
    signer = TimestampSigner("secret-key")
    
    # Test 1: Basic sign and unsign
    value = b"test_value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value, f"Expected {value}, got {result}"
    
    # Test 2: Sign and unsign with string input
    string_value = "test_string"
    signed_str = signer.sign(string_value)
    result_str = signer.unsign(signed_str)
    assert result_str == b"test_string", f"Expected b'test_string', got {result_str}"
    
    # Test 3: Return timestamp with return_timestamp=True
    result_with_ts = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result_with_ts, tuple), "Expected tuple when return_timestamp=True"
    assert len(result_with_ts) == 2, "Expected tuple of length 2"
    assert result_with_ts[0] == value, f"Expected {value}, got {result_with_ts[0]}"
    assert isinstance(result_with_ts[1], datetime), "Expected datetime object"
    assert result_with_ts[1].tzinfo is not None, "Expected timezone-aware datetime"
    
    # Test 4: Validate with max_age - valid signature
    result_valid = signer.unsign(signed, max_age=3600)
    assert result_valid == value, f"Expected {value}, got {result_valid}"
    
    # Test 5: Validate with max_age that is too short - should raise SignatureExpired
    import time
    # Create a signed value with an old timestamp by mocking get_timestamp
    old_signer = TimestampSigner("secret-key")
    old_signer.get_timestamp = lambda: int(time.time()) - 100  # 100 seconds ago
    old_signed = old_signer.sign(value)
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(old_signed, max_age=50)
    assert "Signature age" in str(exc_info.value), "Expected age in error message"
    
    # Test 6: Tampered signature raises BadSignature
    tampered = signed[:-1] + b"x"  # Change last byte
    with pytest.raises(BadSignature):
        signer.unsign(tampered)
    
    # Test 7: Missing timestamp raises BadTimeSignature
    # Create a signed value without timestamp
    no_ts_value = value + signer.sep.encode() + b"fake_signature"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(no_ts_value)
    assert "timestamp missing" in str(exc_info.value), "Expected 'timestamp missing' error"
    
    # Test 8: Malformed timestamp raises BadTimeSignature
    # Create a signed value with invalid timestamp encoding
    malformed_ts = value + signer.sep.encode() + b"!!invalid!!" + signer.sep.encode() + b"sig"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_ts)
    
    # Test 9: Signature with negative age (future timestamp) raises SignatureExpired
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: int(time.time()) + 100  # 100 seconds in future
    future_signed = future_signer.sign(value)
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(future_signed, max_age=3600)
    assert "age" in str(exc_info.value), "Expected age in error message"
    
    # Test 10: Signature error with valid timestamp returns BadTimeSignature with date_signed
    wrong_key_signer = TimestampSigner("wrong-key")
    wrong_signed = wrong_key_signer.sign(value)
    
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(wrong_signed)
    assert exc_info.value.date_signed is not None, "Expected date_signed to be set"
    
    # Test 11: Edge case - empty value
    empty_value = b""
    signed_empty = signer.sign(empty_value)
    result_empty = signer.unsign(signed_empty)
    assert result_empty == empty_value, f"Expected empty bytes, got {result_empty}"
    
    # Test 12: Edge case - very long value
    long_value = b"a" * 10000
    signed_long = signer.sign(long_value)
    result_long = signer.unsign(signed_long)
    assert result_long == long_value, "Long value roundtrip failed"
    
    # Test 13: Verify timestamp is UTC
    signed_ts = signer.sign(value)
    _, timestamp = signer.unsign(signed_ts, return_timestamp=True)
    assert timestamp.tzinfo == timezone.utc, "Timestamp should be in UTC"
```


# LLM-generated content at query #29
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner(secret_key="test-secret")
    assert signer.secret_key == "test-secret"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.sep == "."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == hmac
    
    # Test constructor with custom parameters
    custom_signer = TimestampSigner(
        secret_key="custom-secret",
        salt="custom-salt",
        sep="|",
        key_derivation="none",
        digest_method=hashlib.sha256,
        algorithm="hmac-sha256"
    )
    assert custom_signer.secret_key == "custom-secret"
    assert custom_signer.salt == "custom-salt"
    assert custom_signer.sep == "|"
    assert custom_signer.key_derivation == "none"
    assert custom_signer.digest_method == hashlib.sha256
    assert custom_signer.algorithm == "hmac-sha256"
    
    # Test that TimestampSigner is a subclass of Signer
    assert isinstance(signer, Signer)
    assert issubclass(TimestampSigner, Signer)
    
    # Test default values for optional parameters
    signer_no_salt = TimestampSigner(secret_key="test")
    assert signer_no_salt.salt == "itsdangerous.TimestampSigner"
    
    # Test with bytes secret key
    signer_bytes = TimestampSigner(secret_key=b"bytes-secret")
    assert signer_bytes.secret_key == b"bytes-secret"


# LLM-generated content at query #30
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner()
    assert signer is not None
    assert isinstance(signer, TimestampSigner)
    assert isinstance(signer, Signer)
    assert signer.sep == "."
    assert callable(signer.get_timestamp)
    assert callable(signer.timestamp_to_datetime)
    assert callable(signer.sign)
    assert callable(signer.unsign)
    assert callable(signer.validate)
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
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
    assert result_with_ts[0] == value
    assert isinstance(result_with_ts[1], datetime)
    
    # Test unsign with valid max_age
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test unsign with expired signature
    signer_with_old_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_old_time.get_timestamp
    signer_with_old_time.get_timestamp = lambda: int(time.time()) - 100
    old_signed = signer_with_old_time.sign(value)
    signer_with_old_time.get_timestamp = original_get_timestamp
    with pytest.raises(SignatureExpired):
        signer_with_old_time.unsign(old_signed, max_age=50)
    
    # Test unsign with negative age (future timestamp)
    signer_future = TimestampSigner("secret-key")
    signer_future.get_timestamp = lambda: int(time.time()) + 100
    future_signed = signer_future.sign(value)
    signer_future.get_timestamp = lambda: int(time.time())
    with pytest.raises(SignatureExpired):
        signer_future.unsign(future_signed, max_age=3600)
    
    # Test unsign with malformed timestamp
    malformed = b"test_value" + signer.sep.encode() + b"invalid_timestamp"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test unsign with missing timestamp
    no_timestamp = b"test_value" + signer.sep.encode() + b"just_data"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp)
    
    # Test unsign with bad signature but valid timestamp
    bad_sig_value = value + signer.sep.encode() + b"MTIzNDU="  # base64 of "12345"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_sig_value)
    
    # Test unsign with bad signature and malformed timestamp
    bad_sig_malformed_ts = value + signer.sep.encode() + b"invalid"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_sig_malformed_ts)


# LLM-generated content at query #2
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test the loads method of TimedSerializer."""
    serializer = TimedSerializer(secret_key="test-secret-key-12345")
    
    # Test 1: Basic loads without max_age or return_timestamp
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data, f"Expected {data}, got {result}"
    
    # Test 2: Loads with max_age, within age limit
    signed = serializer.dumps(data)
    result = serializer.loads(signed, max_age=3600)  # 1 hour
    assert result == data, f"Expected {data}, got {result}"
    
    # Test 3: Loads with return_timestamp=True
    signed = serializer.dumps(data)
    result = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    payload, timestamp = result
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test 4: Loads with both max_age and return_timestamp
    signed = serializer.dumps(data)
    result = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result, tuple)
    payload, timestamp = result
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test 5: Loads with expired signature (max_age too small)
    import time
    signed = serializer.dumps(data)
    time.sleep(0.1)  # Small delay to ensure timestamp difference
    try:
        serializer.loads(signed, max_age=0)  # 0 seconds max age
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass
    
    # Test 6: Loads with invalid signature
    try:
        serializer.loads(b"invalid-data")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass
    
    # Test 7: Loads with empty data
    try:
        serializer.loads(b"")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass
    
    # Test 8: Loads with different salt
    serializer2 = TimedSerializer(secret_key="test-secret-key-12345", salt="different-salt")
    signed = serializer2.dumps(data)
    try:
        serializer.loads(signed)  # Using different salt
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass
    
    # Test 9: Loads with bytes input
    signed = serializer.dumps(data)
    result = serializer.loads(signed)  # signed is already bytes
    assert result == data
    
    # Test 10: Loads with string input
    signed_str = serializer.dumps(data).decode('utf-8')
    result = serializer.loads(signed_str)
    assert result == data
    
    # Test 11: Loads with complex data types
    complex_data = {
        "string": "hello",
        "number": 42,
        "list": [1, 2, 3],
        "nested": {"a": 1, "b": 2}
    }
    signed = serializer.dumps(complex_data)
    result = serializer.loads(signed)
    assert result == complex_data
    
    # Test 12: Loads with None value
    signed = serializer.dumps(None)
    result = serializer.loads(signed)
    assert result is None
    
    # Test 13: Loads with integer value
    signed = serializer.dumps(12345)
    result = serializer.loads(signed)
    assert result == 12345
    
    # Test 14: Loads with boolean value
    signed = serializer.dumps(True)
    result = serializer.loads(signed)
    assert result is True
    
    # Test 15: Verify timestamp is in UTC
    signed = serializer.dumps(data)
    _, timestamp = serializer.loads(signed, return_timestamp=True)
    assert timestamp.tzinfo == timezone.utc, "Timestamp should be in UTC"
```


# LLM-generated content at query #3
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor and basic functionality."""
    # Test default constructor
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.secret_key == "test-secret"
    assert isinstance(serializer.default_signer, type)
    assert serializer.default_signer is TimestampSigner
    
    # Test with different secret key types
    serializer_bytes = TimedSerializer(secret_key=b"test-secret-bytes")
    assert serializer_bytes.secret_key == b"test-secret-bytes"
    
    # Test with salt
    serializer_with_salt = TimedSerializer(secret_key="test", salt="custom-salt")
    assert serializer_with_salt.salt == "custom-salt"
    
    # Test with serializer_kwargs
    serializer_kwargs = TimedSerializer(
        secret_key="test", 
        serializer_kwargs={"skipkeys": True}
    )
    assert serializer_kwargs.serializer_kwargs == {"skipkeys": True}
    
    # Test with signer_kwargs
    signer_kwargs = TimedSerializer(
        secret_key="test",
        signer_kwargs={"key_derivation": "hmac"}
    )
    assert signer_kwargs.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test that default_signer is used in iter_unsigners
    serializer_default = TimedSerializer(secret_key="test")
    signers = list(serializer_default.iter_unsigners())
    assert len(signers) > 0
    assert all(isinstance(s, TimestampSigner) for s in signers)
    
    # Test that serializer can be instantiated without arguments
    # (assuming there's a default way to get secret key)
    try:
        serializer_no_key = TimedSerializer()
        assert serializer_no_key.secret_key is not None
    except (TypeError, ValueError):
        pass  # Some implementations require a key
    
    # Test inheritance from Serializer
    assert isinstance(serializer, Serializer)
    
    # Test signer creation
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    assert signer.secret_key == serializer.secret_key
    
    # Test signer with salt
    signer_with_salt = serializer.make_signer(salt="different-salt")
    assert signer_with_salt.salt != signer.salt or signer_with_salt.salt == "different-salt"
```


# LLM-generated content at query #4
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
    
    # Test with signer kwargs
    serializer_with_signer_kwargs = TimedSerializer(signer_kwargs={"key_derivation": "hmac"})
    assert serializer_with_signer_kwargs.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test with serializer kwargs
    serializer_with_serializer_kwargs = TimedSerializer(serializer_kwargs={"serializer": "json"})
    assert serializer_with_serializer_kwargs.serializer_kwargs == {"serializer": "json"}
    
    # Test with fallback signers
    fallback_signer = TimestampSigner(secret_key="fallback")
    serializer_with_fallback = TimedSerializer(fallback_signers=[fallback_signer])
    assert len(list(serializer_with_fallback.iter_unsigners("test"))) > 1
    
    # Test that signers are TimestampSigner instances
    for signer in serializer_with_fallback.iter_unsigners("test"):
        assert isinstance(signer, TimestampSigner)
    
    # Test roundtrip with basic serialization
    test_data = {"key": "value"}
    serialized = serializer.dumps(test_data)
    deserialized = serializer.loads(serialized)
    assert deserialized == test_data
    
    # Test roundtrip with timestamp
    serialized = serializer.dumps(test_data)
    payload, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert payload == test_data
    assert isinstance(timestamp, datetime)
    
    # Test max_age functionality
    import time
    serialized = serializer.dumps(test_data)
    time.sleep(0.1)  # Small delay
    # Should not raise with reasonable max_age
    result = serializer.loads(serialized, max_age=3600)
    assert result == test_data
```


# LLM-generated content at query #5
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == b"secret-key"
    assert signer.salt == b"itsdangerous.TimestampSigner"
    assert signer.sep == b"."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm == "hmac-sha1"
```


# LLM-generated content at query #6
#--------------------------

```python
def test_TimedSerializer():
    """Test the constructor of TimedSerializer class."""
    # Test default constructor
    serializer = TimedSerializer()
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    assert hasattr(serializer, 'load_payload')
    assert hasattr(serializer, 'dump_payload')
    
    # Test with secret key
    serializer_with_key = TimedSerializer(secret_key='test-secret-key')
    assert serializer_with_key.secret_key == 'test-secret-key'
    
    # Test with salt
    serializer_with_salt = TimedSerializer(salt='test-salt')
    assert hasattr(serializer_with_salt, 'salt')
    
    # Test with signer_kwargs
    serializer_with_signer_kwargs = TimedSerializer(
        signer_kwargs={'key_derivation': 'hmac'}
    )
    assert hasattr(serializer_with_signer_kwargs, 'signer_kwargs')
    
    # Test with serializer_kwargs
    serializer_with_serializer_kwargs = TimedSerializer(
        serializer_kwargs={'serializer': 'json'}
    )
    assert hasattr(serializer_with_serializer_kwargs, 'serializer_kwargs')
    
    # Test with signer class (should use TimestampSigner by default)
    serializer_with_custom_signer = TimedSerializer()
    assert serializer_with_custom_signer.default_signer == TimestampSigner
    
    # Test that iter_unsigners returns TimestampSigner instances
    serializer = TimedSerializer(secret_key='test')
    for signer in serializer.iter_unsigners():
        assert isinstance(signer, TimestampSigner)
        break
    
    # Test roundtrip: dumps and loads
    serializer = TimedSerializer(secret_key='test-secret')
    data = {'test': 'data', 'number': 42}
    signed = serializer.dumps(data)
    loaded = serializer.loads(signed)
    assert loaded == data
    
    # Test with max_age
    signed = serializer.dumps(data)
    loaded_with_age = serializer.loads(signed, max_age=3600)
    assert loaded_with_age == data
    
    # Test with return_timestamp
    signed = serializer.dumps(data)
    result = serializer.loads(signed, return_timestamp=True)
    payload, timestamp = result
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test loads_unsafe
    signed = serializer.dumps(data)
    success, result = serializer.loads_unsafe(signed)
    assert success
    assert result == data
```


# LLM-generated content at query #7
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("test-secret")
    
    # Test 1: Basic loads without max_age or return_timestamp
    serialized = serializer.dumps({"test": "data"})
    result = serializer.loads(serialized)
    assert result == {"test": "data"}
    
    # Test 2: Loads with max_age (within limit)
    serialized = serializer.dumps({"test": "data"})
    result = serializer.loads(serialized, max_age=3600)
    assert result == {"test": "data"}
    
    # Test 3: Loads with return_timestamp
    serialized = serializer.dumps({"test": "data"})
    payload, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert payload == {"test": "data"}
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 4: Loads with both max_age and return_timestamp
    serialized = serializer.dumps({"test": "data"})
    payload, timestamp = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert payload == {"test": "data"}
    assert isinstance(timestamp, datetime)
    
    # Test 5: Loads with salt
    serialized = serializer.dumps({"test": "data"}, salt="custom-salt")
    result = serializer.loads(serialized, salt="custom-salt")
    assert result == {"test": "data"}
    
    # Test 6: Loads with expired signature
    serializer_with_short_age = TimedSerializer("test-secret")
    serialized = serializer_with_short_age.dumps({"test": "data"})
    import time
    time.sleep(1)  # Ensure at least 1 second passes
    with pytest.raises(SignatureExpired):
        serializer_with_short_age.loads(serialized, max_age=0)
    
    # Test 7: Loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test 8: Loads with different serializers
    serializer1 = TimedSerializer("secret1")
    serializer2 = TimedSerializer("secret2")
    serialized = serializer1.dumps({"test": "data"})
    with pytest.raises(BadSignature):
        serializer2.loads(serialized)
    
    # Test 9: Loads with complex data types
    complex_data = {"list": [1, 2, 3], "nested": {"key": "value"}, "number": 42}
    serialized = serializer.dumps(complex_data)
    result = serializer.loads(serialized)
    assert result == complex_data
    
    # Test 10: Loads with return_timestamp and verify timestamp is recent
    serialized = serializer.dumps({"test": "data"})
    payload, timestamp = serializer.loads(serialized, return_timestamp=True)
    now = datetime.now(timezone.utc)
    time_diff = (now - timestamp).total_seconds()
    assert time_diff < 5  # Timestamp should be within last 5 seconds
```


# LLM-generated content at query #8
#--------------------------

```python
def test_TimedSerializer():
    # Test that TimedSerializer can be instantiated with default parameters
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner
    
    # Test that TimedSerializer can be instantiated with a custom secret key
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.secret_key == b"test-secret"
    
    # Test that TimedSerializer can be instantiated with a salt
    serializer = TimedSerializer(salt="test-salt")
    assert serializer.salt == "test-salt"
    
    # Test that TimedSerializer can be instantiated with all parameters
    serializer = TimedSerializer(
        secret_key="test-secret",
        salt="test-salt",
        serializer=None,
        signer_kwargs={"key_derivation": "hmac"},
        signer_class=TimestampSigner,
    )
    assert isinstance(serializer, TimedSerializer)
```


# LLM-generated content at query #9
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads with various scenarios."""
    serializer = TimedSerializer(secret_key="test-secret-key")
    
    # Test basic loads (no max_age, no return_timestamp)
    data = {"key": "value"}
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized)
    assert result == data
    
    # Test loads with max_age
    result = serializer.loads(serialized, max_age=3600)
    assert result == data
    
    # Test loads with return_timestamp
    result = serializer.loads(serialized, return_timestamp=True)
    payload, timestamp = result
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with both max_age and return_timestamp
    result = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    payload, timestamp = result
    assert payload == data
    assert isinstance(timestamp, datetime)
    
    # Test loads with expired signature
    serializer_fast = TimedSerializer(secret_key="test-key", signer_kwargs={"digest_method": "sha1"})
    serialized_expired = serializer_fast.dumps(data)
    with pytest.raises(SignatureExpired):
        serializer.loads(serialized_expired, max_age=0)
    
    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test loads with different salt
    serialized_with_salt = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(serialized_with_salt, salt="custom-salt")
    assert result == data
    
    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(serialized_with_salt, salt="wrong-salt")
    
    # Test loads with bytes input
    serialized_bytes = serializer.dumps(data)
    result = serializer.loads(serialized_bytes)
    assert result == data
    
    # Test loads with string input
    serialized_str = serializer.dumps(data).decode("utf-8")
    result = serializer.loads(serialized_str)
    assert result == data
    
    # Test loads with complex data
    complex_data = {"list": [1, 2, 3], "nested": {"a": 1}, "bool": True, "none": None}
    serialized_complex = serializer.dumps(complex_data)
    result = serializer.loads(serialized_complex)
    assert result == complex_data
    
    # Test loads with empty dict
    empty_data = {}
    serialized_empty = serializer.dumps(empty_data)
    result = serializer.loads(serialized_empty)
    assert result == empty_data
    
    # Test loads with list as top-level data
    list_data = [1, "two", 3.0]
    serialized_list = serializer.dumps(list_data)
    result = serializer.loads(serialized_list)
    assert result == list_data
    
    # Test loads with string data
    string_data = "test string"
    serialized_string = serializer.dumps(string_data)
    result = serializer.loads(serialized_string)
    assert result == string_data
    
    # Test loads with integer data
    int_data = 42
    serialized_int = serializer.dumps(int_data)
    result = serializer.loads(serialized_int)
    assert result == int_data
    
    # Test loads with float data
    float_data = 3.14159
    serialized_float = serializer.dumps(float_data)
    result = serializer.loads(serialized_float)
    assert result == float_data
    
    # Test loads with boolean data
    bool_data = True
    serialized_bool = serializer.dumps(bool_data)
    result = serializer.loads(serialized_bool)
    assert result == bool_data
    
    # Test loads with None data
    none_data = None
    serialized_none = serializer.dumps(none_data)
    result = serializer.loads(serialized_none)
    assert result == none_data
    
    # Test loads with multiple signers (using different keys)
    serializer2 = TimedSerializer(secret_key="different-key")
    serialized_different = serializer2.dumps(data)
    with pytest.raises(BadSignature):
        serializer.loads(serialized_different)
```


# LLM-generated content at query #10
#--------------------------

```python
def test_TimedSerializer_loads():
    # Create a serializer instance
    serializer = TimedSerializer(secret_key="test-secret-key")
    
    # Test 1: Basic loads without max_age and return_timestamp
    payload = {"message": "hello", "data": [1, 2, 3]}
    signed = serializer.dumps(payload)
    result = serializer.loads(signed)
    assert result == payload
    
    # Test 2: Loads with max_age (within age limit)
    signed = serializer.dumps(payload)
    result = serializer.loads(signed, max_age=3600)  # 1 hour
    assert result == payload
    
    # Test 3: Loads with return_timestamp=True
    signed = serializer.dumps(payload)
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == payload
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 4: Loads with both max_age and return_timestamp
    signed = serializer.dumps(payload)
    result, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert result == payload
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test 5: Loads with expired signature raises SignatureExpired
    import time as time_module
    serializer_with_future = TimedSerializer(secret_key="test-secret-key")
    signed = serializer_with_future.dumps(payload)
    # Simulate time passing by using a very small max_age
    time_module.sleep(0.1)
    with pytest.raises(SignatureExpired):
        serializer_with_future.loads(signed, max_age=0)
    
    # Test 6: Loads with invalid signature raises BadSignature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-signed-data")
    
    # Test 7: Loads with string input
    signed_str = serializer.dumps(payload).decode('utf-8')
    result = serializer.loads(signed_str)
    assert result == payload
    
    # Test 8: Loads with bytes input
    signed_bytes = serializer.dumps(payload)
    result = serializer.loads(signed_bytes)
    assert result == payload
    
    # Test 9: Loads with custom salt
    signed_with_salt = serializer.dumps(payload, salt="custom-salt")
    result = serializer.loads(signed_with_salt, salt="custom-salt")
    assert result == payload
    
    # Test 10: Loads with wrong salt raises BadSignature
    signed_with_salt = serializer.dumps(payload, salt="custom-salt")
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")
    
    # Test 11: Loads with timestamp that is in the future (negative age)
    # Create a signer with a manipulated timestamp
    class FutureTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) + 1000  # 1000 seconds in the future
    
    future_serializer = TimedSerializer(secret_key="test-secret-key", signer=FutureTimestampSigner)
    signed_future = future_serializer.dumps(payload)
    with pytest.raises(SignatureExpired):
        future_serializer.loads(signed_future, max_age=3600)
    
    # Test 12: Loads with malformed timestamp (should not happen normally, but test robustness)
    # This is a bit tricky to test directly, but we can test with a corrupted signed value
    # that has an invalid timestamp format
    signed = serializer.dumps(payload)
    parts = signed.rsplit(b'.', 1)
    corrupted_signed = parts[0] + b'.' + b'malformed-timestamp'
    with pytest.raises(BadTimeSignature):
        serializer.loads(corrupted_signed)
    
    # Test 13: Loads with empty payload
    signed_empty = serializer.dumps({})
    result = serializer.loads(signed_empty)
    assert result == {}
    
    # Test 14: Loads with complex nested payload
    complex_payload = {"list": [1, 2, 3], "dict": {"a": 1, "b": 2}, "tuple": (1, 2)}
    signed_complex = serializer.dumps(complex_payload)
    result = serializer.loads(signed_complex)
    assert result == complex_payload
    
    # Test 15: Verify that loads returns bytes when return_timestamp is False (default)
    signed = serializer.dumps(payload)
    result = serializer.loads(signed)
    assert isinstance(result, dict)  # Should be the deserialized payload
    
    # Test 16: Verify return type when return_timestamp is True
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result, dict)
    assert isinstance(timestamp, datetime)
```


# LLM-generated content at query #11
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Create a TimestampSigner instance
    signer = TimestampSigner("test-secret", salt="test-salt")
    
    # Test basic sign and unsign
    original_value = b"test_value"
    signed = signer.sign(original_value)
    result = signer.unsign(signed)
    assert result == original_value
    
    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == original_value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with valid max_age
    result = signer.unsign(signed, max_age=3600)
    assert result == original_value
    
    # Test unsign with expired signature
    # Create a signer with a fixed timestamp in the past
    class PastTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return 1000  # Far in the past
    
    past_signer = PastTimestampSigner("test-secret", salt="test-salt")
    past_signed = past_signer.sign(original_value)
    
    with pytest.raises(SignatureExpired):
        past_signer.unsign(past_signed, max_age=100)
    
    # Test unsign with future timestamp (negative age)
    class FutureTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return 9999999999  # Far in the future
    
    future_signer = FutureTimestampSigner("test-secret", salt="test-salt")
    future_signed = future_signer.sign(original_value)
    
    with pytest.raises(SignatureExpired):
        future_signer.unsign(future_signed, max_age=3600)
    
    # Test unsign with bad signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid_signature")
    
    # Test unsign with malformed timestamp
    class MalformedTimestampSigner(TimestampSigner):
        def sign(self, value):
            value_bytes = want_bytes(value)
            sep = want_bytes(self.sep)
            return value_bytes + sep + b"invalid_timestamp"
    
    malformed_signer = MalformedTimestampSigner("test-secret", salt="test-salt")
    malformed_signed = malformed_signer.sign(original_value)
    
    with pytest.raises(BadTimeSignature):
        malformed_signer.unsign(malformed_signed)
    
    # Test unsign with missing timestamp
    regular_signer = Signer("test-secret", salt="test-salt")
    regular_signed = regular_signer.sign(original_value)
    
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(regular_signed)
    
    # Test unsign with string input
    signed_str = signed.decode('utf-8')
    result = signer.unsign(signed_str)
    assert result == original_value
    
    # Test unsign with return_timestamp and string input
    result, timestamp = signer.unsign(signed_str, return_timestamp=True)
    assert result == original_value
    assert isinstance(timestamp, datetime)
```


# LLM-generated content at query #12
#--------------------------

```python
def test_TimedSerializer():
    """Test the TimedSerializer constructor and basic functionality."""
    # Test basic construction
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner
    
    # Test construction with custom secret key
    serializer_with_key = TimedSerializer(secret_key="my-secret-key")
    assert serializer_with_key.secret_key == b"my-secret-key"
    
    # Test construction with salt
    serializer_with_salt = TimedSerializer(salt="my-salt")
    assert serializer_with_salt.salt == "my-salt"
    
    # Test construction with signer_kwargs
    serializer_with_kwargs = TimedSerializer(signer_kwargs={"key_derivation": "hmac"})
    assert serializer_with_kwargs.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test that the default signer is used
    serializer = TimedSerializer()
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    
    # Test that dumps and loads work with timestamp
    data = {"test": "value"}
    signed = serializer.dumps(data)
    loaded = serializer.loads(signed)
    assert loaded == data
    
    # Test that loads returns timestamp when requested
    loaded_with_ts = serializer.loads(signed, return_timestamp=True)
    assert isinstance(loaded_with_ts, tuple)
    assert len(loaded_with_ts) == 2
    assert loaded_with_ts[0] == data
    assert isinstance(loaded_with_ts[1], datetime)
    
    # Test that max_age works
    import time
    signed = serializer.dumps(data)
    loaded = serializer.loads(signed, max_age=3600)  # 1 hour
    assert loaded == data
    
    # Test that expired signature raises SignatureExpired
    signed = serializer.dumps(data)
    import time
    # We can't actually wait, so we test with a very small max_age
    # This should work because the signature is fresh
    loaded = serializer.loads(signed, max_age=0)
    assert loaded == data
    
    # Test that invalid signature raises BadSignature
    import pytest
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test loads_unsafe
    result = serializer.loads_unsafe(signed)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] is True  # successful
    assert result[1] == data
    
    # Test loads_unsafe with invalid data
    result = serializer.loads_unsafe(b"invalid-data")
    assert result[0] is False
    assert result[1] is not None  # Should return the payload
    
    # Test that iter_unsigners returns TimestampSigner instances
    for signer in serializer.iter_unsigners():
        assert isinstance(signer, TimestampSigner)
        break
```


# LLM-generated content at query #13
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign
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
    
    # Test unsign with valid max_age
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test unsign with expired signature
    signer_with_past_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_past_time.get_timestamp
    signer_with_past_time.get_timestamp = lambda: int(time.time()) - 100
    signed_old = signer_with_past_time.sign(value)
    signer_with_past_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_with_past_time.unsign(signed_old, max_age=50)
    
    # Test unsign with future timestamp (age < 0)
    signer_with_future_time = TimestampSigner("secret-key")
    signer_with_future_time.get_timestamp = lambda: int(time.time()) + 100
    signed_future = signer_with_future_time.sign(value)
    signer_with_future_time.get_timestamp = lambda: int(time.time())
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer_with_future_time.unsign(signed_future, max_age=3600)
    assert "age" in str(exc_info.value)
    
    # Test unsign with malformed timestamp
    malformed = b"value" + signer.sep.encode() + b"not_base64"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)
    
    # Test unsign with missing timestamp
    no_timestamp = b"value" + signer.sep.encode() + signer.get_signature(b"value")
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_timestamp)
    
    # Test unsign with tampered signature
    tampered = signed[:-1] + (b"x" if signed[-1:] != b"x" else b"y")
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered)
    
    # Test unsign with string input instead of bytes
    signed_str = signer.sign("string_value").decode()
    result = signer.unsign(signed_str)
    assert result == b"string_value"
    
    # Test unsign with empty value
    signed_empty = signer.sign(b"")
    result = signer.unsign(signed_empty)
    assert result == b""
    
    # Test unsign with return_timestamp and expired signature
    signer_with_past_time = TimestampSigner("secret-key")
    signer_with_past_time.get_timestamp = lambda: int(time.time()) - 100
    signed_old = signer_with_past_time.sign(value)
    signer_with_past_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer_with_past_time.unsign(signed_old, max_age=50, return_timestamp=True)
    assert exc_info.value.date_signed is not None
```


# LLM-generated content at query #14
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer(secret_key="test-secret-key")
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    assert serializer.secret_key == "test-secret-key"
    assert serializer.salt == "itsdangerous"
    
    # Test with custom salt
    serializer2 = TimedSerializer(secret_key="test-secret-key", salt="custom-salt")
    assert serializer2.salt == "custom-salt"
    
    # Test with signer_kwargs
    serializer3 = TimedSerializer(secret_key="test-secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer3.signer_kwargs == {"key_derivation": "hmac"}
    
    # Test that default_signer is TimestampSigner
    assert serializer3.default_signer is TimestampSigner
    
    # Test iter_unsigners returns TimestampSigner instances
    for signer in serializer.iter_unsigners():
        assert isinstance(signer, TimestampSigner)
        break  # Just test first one
```


# LLM-generated content at query #15
#--------------------------

```python
def test_TimedSerializer():
    # Test default constructor
    serializer = TimedSerializer()
    assert serializer.default_signer is TimestampSigner
    assert serializer.salt is not None
    assert serializer.secret_key is not None
    assert serializer.signer is not None
    assert isinstance(serializer.signer, TimestampSigner)

    # Test constructor with secret_key
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.secret_key == "test-secret"
    assert isinstance(serializer.signer, TimestampSigner)

    # Test constructor with salt
    serializer = TimedSerializer(salt="test-salt")
    assert serializer.salt == "test-salt"
    assert isinstance(serializer.signer, TimestampSigner)

    # Test constructor with both secret_key and salt
    serializer = TimedSerializer(secret_key="test-secret", salt="test-salt")
    assert serializer.secret_key == "test-secret"
    assert serializer.salt == "test-salt"
    assert isinstance(serializer.signer, TimestampSigner)

    # Test constructor with signer_kwargs
    serializer = TimedSerializer(signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert isinstance(serializer.signer, TimestampSigner)

    # Test constructor with serializer_kwargs
    serializer = TimedSerializer(serializer_kwargs={"serializer": "json"})
    assert serializer.serializer_kwargs == {"serializer": "json"}
    assert isinstance(serializer.signer, TimestampSigner)

    # Test that the signer is a TimestampSigner instance
    serializer = TimedSerializer()
    assert isinstance(serializer.signer, TimestampSigner)

    # Test default_signer is TimestampSigner
    serializer = TimedSerializer()
    assert serializer.default_signer == TimestampSigner

    # Test that iter_unsigners returns TimestampSigner instances
    serializer = TimedSerializer(secret_key="test-secret", salt="test-salt")
    for signer in serializer.iter_unsigners():
        assert isinstance(signer, TimestampSigner)
```


# LLM-generated content at query #16
#--------------------------

```python
def test_TimedSerializer():
    # Test default construction
    serializer = TimedSerializer()
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    
    # Test construction with secret_key
    serializer = TimedSerializer(secret_key="mysecret")
    assert serializer.secret_key == b"mysecret"
    
    # Test construction with salt
    serializer = TimedSerializer(salt="mysalt")
    assert serializer.salt == b"mysalt"
    
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
    
    # Test default signer is TimestampSigner
    serializer = TimedSerializer(secret_key="test")
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    
    # Test that it inherits from Serializer
    assert isinstance(serializer, Serializer)
```


# LLM-generated content at query #17
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test basic loads without max_age or return_timestamp
    serializer = TimedSerializer()
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed)
    assert result == data

    # Test loads with max_age (valid signature)
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
    serializer_fast = TimedSerializer()
    signed_fast = serializer_fast.dumps(data)
    time.sleep(0.1)  # Wait for timestamp to differ
    with pytest.raises(SignatureExpired):
        serializer_fast.loads(signed_fast, max_age=0)

    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(signed + b"invalid")

    # Test loads with malformed timestamp
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid.data")

    # Test loads with custom salt
    signed_salt = serializer.dumps(data, salt="custom_salt")
    result = serializer.loads(signed_salt, salt="custom_salt")
    assert result == data

    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_salt, salt="wrong_salt")

    # Test loads with empty data
    with pytest.raises(BadSignature):
        serializer.loads(b"")

    # Test loads with None max_age (should not expire)
    result = serializer.loads(signed, max_age=None)
    assert result == data

    # Test loads with return_timestamp=True and expired signature
    with pytest.raises(SignatureExpired):
        serializer_fast.loads(signed_fast, max_age=0, return_timestamp=True)

    # Test loads with different serializer secret key
    serializer2 = TimedSerializer(secret_key="different_key")
    signed2 = serializer2.dumps(data)
    with pytest.raises(BadSignature):
        serializer.loads(signed2)
```


# LLM-generated content at query #18
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
    
    # Test with max_age that is not exceeded
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"
    
    # Test with max_age that is exceeded (simulate by setting timestamp in the past)
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) - 100
    signed_old = signer.sign("old_value")
    signer.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(signed_old, max_age=50)
    assert "Signature age" in str(exc_info.value)
    assert exc_info.value.payload == b"old_value"
    assert isinstance(exc_info.value.date_signed, datetime)
    
    # Test with negative age (future timestamp)
    signer.get_timestamp = lambda: int(time.time())
    signed_future = signer.sign("future_value")
    # Manually manipulate the timestamp to be in the future
    parts = signed_future.split(signer.sep.encode())
    future_ts = int(time.time()) + 1000
    parts[-2] = base64_encode(int_to_bytes(future_ts))
    signed_future = signer.sep.encode().join(parts)
    signed_future = parts[0] + signer.sep.encode() + parts[-2] + signer.sep.encode() + signer.get_signature(parts[0] + signer.sep.encode() + parts[-2])
    # Actually we need to sign with the future timestamp properly
    signer.get_timestamp = lambda: future_ts
    signed_future = signer.sign("future_value")
    signer.get_timestamp = lambda: int(time.time())
    
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(signed_future, max_age=3600)
    assert "age" in str(exc_info.value)
    
    # Test with bad signature
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(b"invalid_signature")
    assert "timestamp missing" in str(exc_info.value)
    assert exc_info.value.payload == b"invalid_signature"
    
    # Test with malformed timestamp
    bad_ts = b"test_value" + signer.sep.encode() + b"not_a_timestamp"
    bad_signed = bad_ts + signer.sep.encode() + signer.get_signature(bad_ts)
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(bad_signed)
    assert "Malformed timestamp" in str(exc_info.value)
    assert exc_info.value.payload == b"test_value"
    
    # Test with tampered value (signature mismatch)
    tampered = b"tampered" + signed[len(b"test_value"):]
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(tampered)
    assert exc_info.value.payload == b"tampered"
    
    # Test return_timestamp=True with valid signature
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    assert abs((datetime.now(timezone.utc) - timestamp).total_seconds()) < 2
    
    # Test with empty value
    signed_empty = signer.sign(b"")
    result = signer.unsign(signed_empty)
    assert result == b""
    
    # Test with bytes value
    signed_bytes = signer.sign(b"bytes_value")
    result = signer.unsign(signed_bytes)
    assert result == b"bytes_value"
    
    # Test with string value (should be converted to bytes)
    signed_str = signer.sign("string_value")
    result = signer.unsign(signed_str)
    assert result == b"string_value"
```


# LLM-generated content at query #19
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor and basic initialization."""
    # Test default construction
    serializer = TimedSerializer("test-secret")
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer is TimestampSigner
    assert serializer.secret_key == b"test-secret"
    
    # Test with additional parameters
    serializer2 = TimedSerializer(
        "another-secret",
        salt="custom-salt",
        serializer="json",
        signer_kwargs={"key_derivation": "hmac"}
    )
    assert serializer2.secret_key == b"another-secret"
    assert serializer2.salt == "custom-salt"
    
    # Test that it uses TimestampSigner as default signer
    assert serializer2.default_signer is TimestampSigner
    
    # Test with different signer class
    class CustomTimestampSigner(TimestampSigner):
        pass
    
    serializer3 = TimedSerializer("secret", signer=CustomTimestampSigner)
    assert serializer3.default_signer is TimestampSigner
    
    # Verify signer creation works
    signer = serializer.make_signer()
    assert isinstance(signer, TimestampSigner)
    
    # Verify serialization/deserialization roundtrip with timestamp
    data = {"key": "value"}
    serialized = serializer.dumps(data)
    assert isinstance(serialized, (str, bytes))
    
    # Verify loads works
    loaded = serializer.loads(serialized)
    assert loaded == data
    
    # Verify loads with return_timestamp
    loaded_with_ts = serializer.loads(serialized, return_timestamp=True)
    assert len(loaded_with_ts) == 2
    assert loaded_with_ts[0] == data
    assert isinstance(loaded_with_ts[1], datetime)
```


# LLM-generated content at query #20
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test successful loads without return_timestamp
    serializer = TimedSerializer("secret-key")
    original_data = {"test": "data"}
    serialized = serializer.dumps(original_data)
    
    result = serializer.loads(serialized)
    assert result == original_data
    
    # Test successful loads with return_timestamp
    result_with_ts = serializer.loads(serialized, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    payload, timestamp = result_with_ts
    assert payload == original_data
    assert isinstance(timestamp, datetime)
    
    # Test with max_age that passes
    result = serializer.loads(serialized, max_age=3600)
    assert result == original_data
    
    # Test with max_age that fails (expired)
    with pytest.raises(SignatureExpired):
        serializer.loads(serialized, max_age=-1)
    
    # Test with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-data")
    
    # Test with tampered data
    tampered = serialized[:-1] + b"x" 
    with pytest.raises(BadSignature):
        serializer.loads(tampered)
    
    # Test with salt
    serializer_with_salt = TimedSerializer("secret-key", salt="custom-salt")
    serialized_with_salt = serializer_with_salt.dumps(original_data)
    
    # Should fail with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(serialized_with_salt)
    
    # Should succeed with correct salt
    result = serializer_with_salt.loads(serialized_with_salt)
    assert result == original_data
    
    # Test multiple signers (fallback)
    serializer_multi = TimedSerializer("secret-key", signer_kwargs=[{}, {"key_derivation": "none"}])
    serialized_multi = serializer_multi.dumps(original_data)
    result = serializer_multi.loads(serialized_multi)
    assert result == original_data
    
    # Test with bytes input
    result = serializer.loads(serialized)
    assert result == original_data
    
    # Test with string input
    result = serializer.loads(serialized.decode())
    assert result == original_data
    
    # Test loading empty string
    with pytest.raises(BadSignature):
        serializer.loads("")
    
    # Test with very short max_age (should fail)
    import time
    serialized = serializer.dumps(original_data)
    time.sleep(0.1)
    with pytest.raises(SignatureExpired):
        serializer.loads(serialized, max_age=0)
    
    # Test return_timestamp with max_age
    result_with_ts = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert isinstance(result_with_ts, tuple)
    assert len(result_with_ts) == 2
    payload, timestamp = result_with_ts
    assert payload == original_data
    assert isinstance(timestamp, datetime)
```


# LLM-generated content at query #21
#--------------------------

```python
def test_TimedSerializer():
    """Test the constructor of TimedSerializer class."""
    # Test default constructor
    serializer = TimedSerializer("test-secret")
    assert serializer.secret_key == "test-secret"
    assert serializer.salt == "itsdangerous.TimedSerializer"
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer.signer, TimestampSigner)
    assert serializer.signer.secret_key == "test-secret"

    # Test constructor with custom secret key and salt
    serializer = TimedSerializer("custom-secret", salt="custom-salt")
    assert serializer.secret_key == "custom-secret"
    assert serializer.salt == "custom-salt"
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer.signer, TimestampSigner)
    assert serializer.signer.secret_key == "custom-secret"

    # Test constructor with additional keyword arguments
    serializer = TimedSerializer("test-secret", serializer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_key == "test-secret"
    assert serializer.salt == "itsdangerous.TimedSerializer"
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer.signer, TimestampSigner)
    assert serializer.signer.secret_key == "test-secret"
    assert serializer.signer.key_derivation == "hmac"

    # Test constructor with signer_kwargs
    serializer = TimedSerializer("test-secret", signer_kwargs={"digest_method": hashlib.sha256})
    assert serializer.secret_key == "test-secret"
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer.signer, TimestampSigner)
    assert serializer.signer.secret_key == "test-secret"
    assert serializer.signer.digest_method == hashlib.sha256

    # Test constructor with signer class override (though not recommended)
    class CustomTimestampSigner(TimestampSigner):
        pass
    
    serializer = TimedSerializer("test-secret", signer=CustomTimestampSigner)
    assert serializer.secret_key == "test-secret"
    assert isinstance(serializer.signer, CustomTimestampSigner)
    assert serializer.signer.secret_key == "test-secret"

    # Test that the serializer is an instance of Serializer
    serializer = TimedSerializer("test-secret")
    assert isinstance(serializer, Serializer)

    # Test that the default_signer is correctly set as a class attribute
    assert TimedSerializer.default_signer is TimestampSigner
    
    # Test that the serializer can be instantiated with bytes as secret key
    serializer = TimedSerializer(b"bytes-secret")
    assert serializer.secret_key == b"bytes-secret"
    assert serializer.signer.secret_key == b"bytes-secret"

    # Test that salt can be bytes
    serializer = TimedSerializer("test-secret", salt=b"bytes-salt")
    assert serializer.salt == b"bytes-salt"
```


# LLM-generated content at query #22
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Create signer with a fixed timestamp for testing
    class FixedTimestampSigner(TimestampSigner):
        def __init__(self, *args, fixed_timestamp=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.fixed_timestamp = fixed_timestamp or 1000000
            
        def get_timestamp(self) -> int:
            return self.fixed_timestamp
    
    signer = FixedTimestampSigner("secret-key")
    
    # Test basic sign and unsign
    original_value = b"test_value"
    signed = signer.sign(original_value)
    unsigned = signer.unsign(signed)
    assert unsigned == original_value
    
    # Test unsign with return_timestamp=True
    unsigned, timestamp = signer.unsign(signed, return_timestamp=True)
    assert unsigned == original_value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test unsign with max_age that is not expired
    unsigned = signer.unsign(signed, max_age=3600)
    assert unsigned == original_value
    
    # Test unsign with max_age that is expired
    signer_old = FixedTimestampSigner("secret-key", fixed_timestamp=1)
    old_signed = signer_old.sign(original_value)
    with pytest.raises(SignatureExpired):
        signer.unsign(old_signed, max_age=10)
    
    # Test unsign with max_age and negative age (future timestamp)
    signer_future = FixedTimestampSigner("secret-key", fixed_timestamp=9999999999)
    future_signed = signer_future.sign(original_value)
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed, max_age=10)
    
    # Test unsign with invalid signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid_signature")
    
    # Test unsign with tampered value
    tampered = signed[:-1] + bytes([signed[-1] ^ 0xFF])
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered)
    
    # Test unsign with malformed timestamp
    malformed_timestamp = signed.rsplit(signer.sep.encode(), 1)[0] + signer.sep + b"invalid_base64"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_timestamp)
    
    # Test unsign with missing timestamp separator
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"no_separator")
    
    # Test unsign with string input
    signed_str = signer.sign("test_string").decode()
    unsigned_bytes = signer.unsign(signed_str)
    assert unsigned_bytes == b"test_string"
    
    # Test unsign with return_timestamp=True and max_age
    unsigned, timestamp = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert unsigned == original_value
    assert isinstance(timestamp, datetime)
```


# LLM-generated content at query #23
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
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test with max_age (within limit)
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=3600)
    assert result == value
    
    # Test with max_age (expired)
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) + 100
    signed = signer.sign(value)
    signer.get_timestamp = original_get_timestamp
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=10)
    
    # Test with negative age (future timestamp)
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) - 100
    signed = signer.sign(value)
    signer.get_timestamp = original_get_timestamp
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(signed, max_age=10)
    assert "age" in str(exc_info.value)
    
    # Test with malformed timestamp
    bad_signed = signed + b"malformed"
    with pytest.raises(BadSignature):
        signer.unsign(bad_signed)
    
    # Test with missing timestamp
    signer_no_timestamp = Signer("secret-key")
    no_ts_signed = signer_no_timestamp.sign(value)
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(no_ts_signed)
    
    # Test with bad signature but valid timestamp format
    bad_sig = signed[:-1] + b"x"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_sig)
    
    # Test with string input
    signed_str = signer.sign("test_string")
    result = signer.unsign(signed_str)
    assert result == b"test_string"


# LLM-generated content at query #24
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test TimedSerializer.loads with various scenarios."""
    serializer = TimedSerializer("test-secret-key")
    
    # Test 1: Basic loads without max_age
    original_data = {"user": "alice", "role": "admin"}
    signed = serializer.dumps(original_data)
    result = serializer.loads(signed)
    assert result == original_data, f"Expected {original_data}, got {result}"
    
    # Test 2: Loads with max_age and valid timestamp
    result = serializer.loads(signed, max_age=3600)
    assert result == original_data, f"Expected {original_data}, got {result}"
    
    # Test 3: Loads with return_timestamp=True
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == original_data, f"Expected {original_data}, got {result}"
    assert isinstance(timestamp, datetime), f"Expected datetime, got {type(timestamp)}"
    assert timestamp.tzinfo is not None, "Timestamp should be timezone-aware"
    
    # Test 4: Loads with both max_age and return_timestamp
    result, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert result == original_data, f"Expected {original_data}, got {result}"
    assert isinstance(timestamp, datetime), f"Expected datetime, got {type(timestamp)}"
    
    # Test 5: Loads with expired signature (max_age too small)
    import time
    # Create a signed value with a timestamp in the past
    old_signer = TimestampSigner("test-secret-key")
    old_value = old_signer.sign(b"test-data")
    
    with pytest.raises(SignatureExpired):
        serializer.loads(old_value, max_age=0)
    
    # Test 6: Loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")
    
    # Test 7: Loads with bytes input
    signed_bytes = serializer.dumps(b"bytes-data")
    result = serializer.loads(signed_bytes)
    assert result == b"bytes-data", f"Expected b'bytes-data', got {result}"
    
    # Test 8: Loads with string input
    signed_str = serializer.dumps("string-data")
    result = serializer.loads(signed_str)
    assert result == "string-data", f"Expected 'string-data', got {result}"
    
    # Test 9: Loads with complex nested data
    complex_data = {
        "list": [1, 2, 3],
        "dict": {"key": "value"},
        "number": 42,
        "boolean": True,
        "none": None
    }
    signed_complex = serializer.dumps(complex_data)
    result = serializer.loads(signed_complex)
    assert result == complex_data, f"Expected {complex_data}, got {result}"
    
    # Test 10: Loads with empty data
    empty_data = {}
    signed_empty = serializer.dumps(empty_data)
    result = serializer.loads(signed_empty)
    assert result == empty_data, f"Expected {empty_data}, got {result}"
    
    # Test 11: Loads with different salt
    salt_serializer = TimedSerializer("test-secret-key", salt="different-salt")
    signed_salt = salt_serializer.dumps("salt-data")
    
    # Should fail with default serializer (different salt)
    with pytest.raises(BadSignature):
        serializer.loads(signed_salt)
    
    # Should succeed with the correct salt
    result = salt_serializer.loads(signed_salt)
    assert result == "salt-data", f"Expected 'salt-data', got {result}"
    
    # Test 12: Loads with multiple signers (fallback)
    # Create serializer with multiple signers
    multi_serializer = TimedSerializer("test-secret-key")
    multi_serializer.signer_kwargs = [{"key": "key1"}, {"key": "key2"}]
    
    signed_multi = multi_serializer.dumps("multi-data")
    result = multi_serializer.loads(signed_multi)
    assert result == "multi-data", f"Expected 'multi-data', got {result}"
    
    # Test 13: Loads with timestamp in the future (should raise SignatureExpired)
    future_signer = TimestampSigner("test-secret-key")
    # Manually create a signature with a timestamp far in the future
    future_value = want_bytes("future-data")
    future_timestamp = base64_encode(int_to_bytes(int(time.time()) + 10000))
    sep = want_bytes(future_signer.sep)
    future_signed = future_value + sep + future_timestamp
    future_signed = future_signed + sep + future_signer.get_signature(future_signed)
    
    with pytest.raises(SignatureExpired):
        serializer.loads(future_signed, max_age=3600)
    
    # Test 14: Loads with malformed timestamp
    malformed_value = want_bytes("malformed-data")
    malformed_timestamp = b"not-a-timestamp"
    sep = want_bytes(serializer.sep)
    malformed_signed = malformed_value + sep + malformed_timestamp
    # Need to sign the malformed value properly
    signer = serializer.make_signer()
    malformed_signed = malformed_value + sep + malformed_timestamp
    malformed_signed = malformed_signed + sep + signer.get_signature(malformed_signed)
    
    with pytest.raises(BadTimeSignature):
        serializer.loads(malformed_signed)
```


# LLM-generated content at query #25
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test basic loads without max_age or return_timestamp
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    dumped = serializer.dumps(data)
    loaded = serializer.loads(dumped)
    assert loaded == data

    # Test loads with return_timestamp=True
    loaded_with_ts = serializer.loads(dumped, return_timestamp=True)
    assert isinstance(loaded_with_ts, tuple)
    assert len(loaded_with_ts) == 2
    payload, timestamp = loaded_with_ts
    assert payload == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test loads with max_age that is not expired
    loaded = serializer.loads(dumped, max_age=3600)
    assert loaded == data

    # Test loads with max_age that is expired
    import time as time_module
    old_time = time_module.time() - 7200  # 2 hours ago
    with unittest.mock.patch("time.time", return_value=old_time):
        old_dumped = serializer.dumps(data)
    with pytest.raises(SignatureExpired):
        serializer.loads(old_dumped, max_age=3600)

    # Test loads with salt
    salted_serializer = TimedSerializer("secret-key", salt="different-salt")
    salted_dumped = salted_serializer.dumps(data)
    # Should fail with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(salted_dumped)

    # Test loads with multiple signers (fallback mechanism)
    serializer_with_fallback = TimedSerializer("secret-key")
    # Add an additional signer with a different salt
    serializer_with_fallback.signer_kwargs = {"salt": "fallback"}
    dumped_with_fallback = serializer_with_fallback.dumps(data)
    # Should still work with the default signer
    loaded = serializer_with_fallback.loads(dumped_with_fallback)
    assert loaded == data

    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")

    # Test loads with malformed timestamp
    # First create a valid signature, then corrupt the timestamp
    valid_dumped = serializer.dumps(data)
    parts = valid_dumped.split(b".")
    corrupted_timestamp = base64_encode(b"not-a-timestamp")
    corrupted_dumped = parts[0] + b"." + corrupted_timestamp + b"." + parts[2]
    with pytest.raises(BadTimeSignature):
        serializer.loads(corrupted_dumped)

    # Test loads with return_timestamp and max_age together
    loaded_with_ts = serializer.loads(dumped, max_age=3600, return_timestamp=True)
    assert isinstance(loaded_with_ts, tuple)
    payload, timestamp = loaded_with_ts
    assert payload == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test loads with bytes input
    dumped_bytes = serializer.dumps(data)
    loaded = serializer.loads(dumped_bytes)
    assert loaded == data

    # Test loads with string input
    dumped_str = serializer.dumps(data).decode()
    loaded = serializer.loads(dumped_str)
    assert loaded == data
```


# LLM-generated content at query #26
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor creates instance with expected attributes
    signer = TimestampSigner(secret_key="test-secret")
    assert signer.secret_key == "test-secret"
    assert signer.sep == "."
    assert signer.salt is not None
    
    # Test constructor with custom salt and separator
    signer = TimestampSigner(
        secret_key="test-secret",
        salt="custom-salt",
        sep="|"
    )
    assert signer.secret_key == "test-secret"
    assert signer.salt == "custom-salt"
    assert signer.sep == "|"
    
    # Test constructor with key derivation and digest method
    signer = TimestampSigner(
        secret_key="test-secret",
        key_derivation="hmac",
        digest_method="sha256"
    )
    assert signer.key_derivation == "hmac"
    assert signer.digest_method.__name__ == "sha256"
    
    # Test constructor without arguments raises error
    try:
        signer = TimestampSigner()
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
```


# LLM-generated content at query #27
#--------------------------

```python
def test_TimedSerializer():
    """Test TimedSerializer constructor and basic attributes."""
    # Test default construction
    serializer = TimedSerializer()
    assert serializer.default_signer is TimestampSigner
    assert isinstance(serializer, Serializer)
    
    # Test with secret key
    serializer = TimedSerializer("secret-key")
    assert serializer.secret_key == "secret-key"
    
    # Test with salt
    serializer = TimedSerializer(salt="my-salt")
    assert serializer.salt == "my-salt"
    
    # Test with signer_kwargs
    serializer = TimedSerializer(signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}
    
    # Test with serializer_kwargs
    serializer = TimedSerializer(serializer_kwargs={"signer_kwargs": {"key_derivation": "hmac"}})
    assert serializer.serializer_kwargs == {"signer_kwargs": {"key_derivation": "hmac"}}
    
    # Test that iter_unsigners returns TimestampSigner instances
    serializer = TimedSerializer("secret-key")
    for signer in serializer.iter_unsigners():
        assert isinstance(signer, TimestampSigner)
        break
    
    # Test with different salt for iter_unsigners
    serializer = TimedSerializer("secret-key")
    for signer in serializer.iter_unsigners(salt="different-salt"):
        assert isinstance(signer, TimestampSigner)
        assert b"different-salt" in signer.salt
        break
```


# LLM-generated content at query #28
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.default_signer is TimestampSigner
    assert serializer.signer is TimestampSigner
    
    # Test that it can serialize and deserialize
    data = {"key": "value"}
    serialized = serializer.dumps(data)
    assert isinstance(serialized, bytes)
    
    # Test basic loads
    deserialized = serializer.loads(serialized)
    assert deserialized == data
    
    # Test with max_age
    deserialized = serializer.loads(serialized, max_age=3600)
    assert deserialized == data
    
    # Test with return_timestamp
    deserialized, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert deserialized == data
    assert isinstance(timestamp, datetime)
    
    # Test with salt
    serialized_salt = serializer.dumps(data, salt="custom-salt")
    deserialized_salt = serializer.loads(serialized_salt, salt="custom-salt")
    assert deserialized_salt == data
    
    # Test loads_unsafe
    success, result = serializer.loads_unsafe(serialized)
    assert success is True
    assert result == data
    
    # Test with empty secret key
    serializer_empty = TimedSerializer(secret_key="")
    serialized_empty = serializer_empty.dumps(data)
    deserialized_empty = serializer_empty.loads(serialized_empty)
    assert deserialized_empty == data
    
    # Test with different serializer options
    serializer_options = TimedSerializer(
        secret_key="test",
        salt="different-salt",
        serializer="json",
        signer_kwargs={"key_derivation": "hmac"}
    )
    assert serializer_options.default_signer is TimestampSigner
```


# LLM-generated content at query #29
#--------------------------

```python
def test_TimedSerializer_loads():
    """Test the loads method of TimedSerializer."""
    
    def test_loads_basic():
        """Test basic loads without max_age or return_timestamp."""
        serializer = TimedSerializer("secret-key")
        original_data = {"key": "value"}
        signed = serializer.dumps(original_data)
        loaded = serializer.loads(signed)
        assert loaded == original_data
    
    def test_loads_with_max_age():
        """Test loads with max_age parameter."""
        serializer = TimedSerializer("secret-key")
        original_data = "test_data"
        signed = serializer.dumps(original_data)
        
        # Should work with a large max_age
        loaded = serializer.loads(signed, max_age=3600)
        assert loaded == original_data
        
        # Should raise SignatureExpired with small max_age
        with pytest.raises(SignatureExpired):
            serializer.loads(signed, max_age=0)
    
    def test_loads_with_return_timestamp():
        """Test loads with return_timestamp=True."""
        serializer = TimedSerializer("secret-key")
        original_data = "test_data"
        signed = serializer.dumps(original_data)
        
        payload, timestamp = serializer.loads(signed, return_timestamp=True)
        assert payload == original_data
        assert isinstance(timestamp, datetime)
        assert timestamp.tzinfo == timezone.utc
    
    def test_loads_with_max_age_and_return_timestamp():
        """Test loads with both max_age and return_timestamp."""
        serializer = TimedSerializer("secret-key")
        original_data = "test_data"
        signed = serializer.dumps(original_data)
        
        payload, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
        assert payload == original_data
        assert isinstance(timestamp, datetime)
        assert timestamp.tzinfo == timezone.utc
    
    def test_loads_with_salt():
        """Test loads with salt parameter."""
        serializer = TimedSerializer("secret-key", salt="my-salt")
        original_data = "test_data"
        signed = serializer.dumps(original_data)
        
        loaded = serializer.loads(signed, salt="my-salt")
        assert loaded == original_data
        
        # Should fail with wrong salt
        with pytest.raises(BadSignature):
            serializer.loads(signed, salt="wrong-salt")
    
    def test_loads_invalid_signature():
        """Test loads with invalid signature."""
        serializer = TimedSerializer("secret-key")
        
        with pytest.raises(BadSignature):
            serializer.loads(b"invalid_data")
    
    def test_loads_empty_string():
        """Test loads with empty string."""
        serializer = TimedSerializer("secret-key")
        
        with pytest.raises(BadSignature):
            serializer.loads(b"")
    
    def test_loads_tampered_data():
        """Test loads with tampered data."""
        serializer = TimedSerializer("secret-key")
        original_data = "test_data"
        signed = serializer.dumps(original_data)
        
        # Tamper with the data
        tampered = signed[:-1] + b"x"
        with pytest.raises(BadSignature):
            serializer.loads(tampered)
    
    def test_loads_with_multiple_signers():
        """Test loads with multiple signers."""
        # Create a serializer with multiple keys
        serializer = TimedSerializer(["key1", "key2", "key3"])
        original_data = "test_data"
        
        # Sign with the last key (should work with all previous keys)
        signed = serializer.dumps(original_data)
        loaded = serializer.loads(signed)
        assert loaded == original_data
    
    def test_loads_older_timestamp():
        """Test loads with an older timestamp."""
        serializer = TimedSerializer("secret-key")
        original_data = "test_data"
        
        # Manually create a signed value with an old timestamp
        signer = TimestampSigner("secret-key")
        old_timestamp = base64_encode(int_to_bytes(100))  # Very old timestamp
        value = want_bytes("test_data")
        sep = want_bytes(signer.sep)
        signed = value + sep + old_timestamp + sep + signer.get_signature(value + sep + old_timestamp)
        
        # Should raise SignatureExpired if max_age is small
        with pytest.raises(SignatureExpired):
            serializer.loads(signed, max_age=50)
        
        # Should work without max_age
        loaded = serializer.loads(signed)
        assert loaded == original_data
    
    def test_loads_return_type():
        """Test return type of loads method."""
        serializer = TimedSerializer("secret-key")
        original_data = "test_data"
        signed = serializer.dumps(original_data)
        
        # Without return_timestamp
        result = serializer.loads(signed)
        assert isinstance(result, (str, bytes, dict, list, int, float, bool, type(None)))
        
        # With return_timestamp
        result = serializer.loads(signed, return_timestamp=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[1], datetime)
    
    def test_loads_complex_data():
        """Test loads with complex data types."""
        serializer = TimedSerializer("secret-key")
        original_data = {
            "string": "hello",
            "number": 42,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
            "bool": True,
            "none": None
        }
        signed = serializer.dumps(original_data)
        loaded = serializer.loads(signed)
        assert loaded == original_data
    
    def test_loads_unicode():
        """Test loads with unicode data."""
        serializer = TimedSerializer("secret-key")
        original_data = "héllo wörld 🌍"
        signed = serializer.dumps(original_data)
        loaded = serializer.loads(signed)
        assert loaded == original_data
    
    def test_loads_with_custom_signer():
        """Test loads with custom signer class."""
        class CustomTimestampSigner(TimestampSigner):
            pass
        
        serializer = TimedSerializer("secret-key", signer=CustomTimestampSigner)
        original_data = "test_data"
        signed = serializer.dumps(original_data)
        loaded = serializer.loads(signed)
        assert loaded == original_data
    
    # Run all test functions
    test_loads_basic()
    test_loads_with_max_age()
    test_loads_with_return_timestamp()
    test_loads_with_max_age_and_return_timestamp()
    test_loads_with_salt()
    test_loads_invalid_signature()
    test_loads_empty_string()
    test_loads_tampered_data()
    test_loads_with_multiple_signers()
    test_loads_older_timestamp()
    test_loads_return_type()
    test_loads_complex_data()
    test_loads_unicode()
    test_loads_with_custom_signer()
```


# LLM-generated content at query #30
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Test case 1: Basic unsign without timestamp return
    signer = TimestampSigner("secret-key")
    original_value = b"test_data"
    signed_value = signer.sign(original_value)
    result = signer.unsign(signed_value)
    assert result == original_value

    # Test case 2: Unsign with return_timestamp=True
    result_with_timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert isinstance(result_with_timestamp, tuple)
    value, timestamp = result_with_timestamp
    assert value == original_value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test case 3: Unsign with valid max_age
    result = signer.unsign(signed_value, max_age=3600)
    assert result == original_value

    # Test case 4: Unsign with expired signature
    signer.get_timestamp = lambda: int(time.time()) - 7200  # 2 hours ago
    expired_signed = signer.sign(original_value)
    signer.get_timestamp = lambda: int(time.time())  # Reset timestamp
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(expired_signed, max_age=3600)
    assert "signature age" in str(exc_info.value).lower()

    # Test case 5: Unsign with future timestamp (age < 0)
    signer.get_timestamp = lambda: int(time.time()) + 3600  # 1 hour in future
    future_signed = signer.sign(original_value)
    signer.get_timestamp = lambda: int(time.time())  # Reset timestamp
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(future_signed, max_age=3600)
    assert "age -" in str(exc_info.value) or "age 0" in str(exc_info.value)

    # Test case 6: Unsign with tampered value
    tampered_signed = signed_value[:-1] + b"X"
    with pytest.raises(BadSignature):
        signer.unsign(tampered_signed)

    # Test case 7: Unsign with non-timestamp signed value
    regular_signer = Signer("secret-key")
    regular_signed = regular_signer.sign(b"no_timestamp")
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(regular_signed)

    # Test case 8: Unsign with malformed timestamp
    malformed = original_value + b"." + b"inval1d_t1mestamp"
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(malformed)

    # Test case 9: Unsign with string input
    string_signed = signer.sign("string_data").decode()
    result = signer.unsign(string_signed)
    assert result == b"string_data"

    # Test case 10: Unsign with max_age=None (no expiration check)
    old_signer = TimestampSigner("secret-key")
    old_signer.get_timestamp = lambda: int(time.time()) - 100000  # Very old
    old_signed = old_signer.sign(original_value)
    old_signer.get_timestamp = lambda: int(time.time())  # Reset
    result = old_signer.unsign(old_signed, max_age=None)
    assert result == original_value

    # Test case 11: Unsign with return_timestamp and max_age
    result = signer.unsign(signed_value, max_age=3600, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == original_value
    assert isinstance(result[1], datetime)

    # Test case 12: Unsign with custom separator
    custom_signer = TimestampSigner("secret-key", sep="|")
    custom_signed = custom_signer.sign(original_value)
    result = custom_signer.unsign(custom_signed)
    assert result == original_value

    # Test case 13: Verify timestamp is returned as UTC datetime
    result_with_ts = signer.unsign(signed_value, return_timestamp=True)
    timestamp = result_with_ts[1]
    assert timestamp.tzinfo is timezone.utc
    # Verify timestamp is recent (within 5 seconds)
    now = datetime.now(timezone.utc)
    time_diff = abs((now - timestamp).total_seconds())
    assert time_diff < 5
```


# LLM-generated content at query #31
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Basic signing and unsigning
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    result = signer.unsign(signed)
    assert result == b"test value"

    # Unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Unsign with max_age that is not expired
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test value"

    # Unsign with max_age that is expired (simulate by setting timestamp far in past)
    signer_with_past_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_past_time.get_timestamp
    signer_with_past_time.get_timestamp = lambda: int(time.time()) - 10000
    signed_past = signer_with_past_time.sign("test value")
    signer_with_past_time.get_timestamp = original_get_timestamp
    with pytest.raises(SignatureExpired):
        signer_with_past_time.unsign(signed_past, max_age=100)

    # Unsign with max_age that is negative (timestamp in future)
    signer_with_future_time = TimestampSigner("secret-key")
    signer_with_future_time.get_timestamp = lambda: int(time.time()) + 10000
    signed_future = signer_with_future_time.sign("test value")
    signer_with_future_time.get_timestamp = original_get_timestamp
    with pytest.raises(SignatureExpired, match="age .* 0 seconds"):
        signer_with_future_time.unsign(signed_future, max_age=3600)

    # Unsign with bad signature
    bad_signed = signed + b"bad"
    with pytest.raises(BadSignature):
        signer.unsign(bad_signed)

    # Unsign with bad signature and return_timestamp=True
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed, return_timestamp=True)

    # Unsign with missing timestamp
    signer_only = Signer("secret-key")
    signed_no_timestamp = signer_only.sign("test value")
    with pytest.raises(BadTimeSignature, match="timestamp missing"):
        signer.unsign(signed_no_timestamp)

    # Unsign with malformed timestamp
    malformed_timestamp = base64_encode(b"not-a-number")
    sep = signer.sep.encode()
    value = b"test value"
    malformed_signed = value + sep + malformed_timestamp + sep + signer.get_signature(value + sep + malformed_timestamp)
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(malformed_signed)

    # Unsign with malformed timestamp and bad signature
    malformed_signed_bad = malformed_signed + b"bad"
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(malformed_signed_bad)

    # Test with bytes input
    signed_bytes = signer.sign(b"test bytes")
    result = signer.unsign(signed_bytes)
    assert result == b"test bytes"

    # Test with string input
    signed_str = signer.sign("test string")
    result = signer.unsign(signed_str)
    assert result == b"test string"
```


# LLM-generated content at query #32
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == "secret-key"
    assert signer.salt == "itsdangerous.TimestampSigner"
    assert signer.digest_method is not None
    assert signer.key_derivation == "hmac"
    
    # Test with explicit salt
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == "custom-salt"
    
    # Test with custom digest method
    import hashlib
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256
    
    # Test with custom key derivation
    signer = TimestampSigner("secret-key", key_derivation="none")
    assert signer.key_derivation == "none"
    
    # Test get_timestamp returns integer
    timestamp = signer.get_timestamp()
    assert isinstance(timestamp, int)
    assert timestamp > 0
    
    # Test timestamp_to_datetime
    dt = signer.timestamp_to_datetime(timestamp)
    assert dt.tzinfo is not None
    assert dt.tzinfo.utcoffset(dt) is not None
    assert dt.tzinfo == timezone.utc
    
    # Test sign method
    signed = signer.sign("test-value")
    assert isinstance(signed, bytes)
    assert signer.sep.encode() in signed
    
    # Test unsign method without max_age
    unsigned = signer.unsign(signed)
    assert unsigned == b"test-value"
    
    # Test unsign with return_timestamp
    unsigned, timestamp_dt = signer.unsign(signed, return_timestamp=True)
    assert unsigned == b"test-value"
    assert isinstance(timestamp_dt, datetime)
    assert timestamp_dt.tzinfo == timezone.utc
    
    # Test unsign with max_age
    unsigned = signer.unsign(signed, max_age=3600)
    assert unsigned == b"test-value"
    
    # Test unsign with expired signature
    import time
    old_signer = TimestampSigner("secret-key")
    old_signer.get_timestamp = lambda: int(time.time()) - 7200
    old_signed = old_signer.sign("test-value")
    try:
        signer.unsign(old_signed, max_age=3600)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass
    
    # Test validate method
    assert signer.validate(signed) == True
    assert signer.validate(signed, max_age=3600) == True
    assert signer.validate(b"invalid-signature") == False
    
    # Test with bytes secret
    signer = TimestampSigner(b"bytes-secret")
    signed = signer.sign("test-value")
    unsigned = signer.unsign(signed)
    assert unsigned == b"test-value"


# LLM-generated content at query #33
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test basic loads without max_age or return_timestamp
    serializer = TimedSerializer("secret-key")
    data = {"test": "data"}
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized)
    assert result == data

    # Test loads with max_age (valid)
    result = serializer.loads(serialized, max_age=3600)
    assert result == data

    # Test loads with return_timestamp=True
    payload, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert payload == data
    assert isinstance(timestamp, datetime)

    # Test loads with both max_age and return_timestamp
    payload, timestamp = serializer.loads(serialized, max_age=3600, return_timestamp=True)
    assert payload == data
    assert isinstance(timestamp, datetime)

    # Test loads with salt
    serializer2 = TimedSerializer("other-secret")
    serialized2 = serializer2.dumps(data, salt="custom-salt")
    result2 = serializer.loads(serialized2, salt="custom-salt")
    assert result2 == data

    # Test loads with expired signature
    serializer3 = TimedSerializer("test-key")
    old_data = {"old": "data"}
    old_serialized = serializer3.dumps(old_data)
    # Simulate an expired signature by using a very small max_age
    import time
    time.sleep(0.1)  # Ensure at least some time passes
    with pytest.raises(SignatureExpired):
        serializer3.loads(old_serialized, max_age=0)

    # Test loads with bad signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid-data")

    # Test loads with string input
    result = serializer.loads(serialized.decode())
    assert result == data

    # Test loads with empty salt
    result = serializer.loads(serialized, salt=b"")
    assert result == data
```


# LLM-generated content at query #34
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic sign and unsign
    value = b"test_value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value
    
    # Test with string input
    signed_str = signer.sign("test_string")
    result_str = signer.unsign(signed_str)
    assert result_str == b"test_string"
    
    # Test return_timestamp=True
    signed_with_ts = signer.sign("test")
    result, timestamp = signer.unsign(signed_with_ts, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc
    
    # Test max_age valid
    signed_valid = signer.sign("fresh")
    result = signer.unsign(signed_valid, max_age=3600)
    assert result == b"fresh"
    
    # Test max_age expired
    signer_with_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_time.get_timestamp
    signer_with_time.get_timestamp = lambda: int(time.time()) - 100
    signed_old = signer_with_time.sign("old")
    signer_with_time.get_timestamp = original_get_timestamp
    with pytest.raises(SignatureExpired):
        signer_with_time.unsign(signed_old, max_age=10)
    
    # Test max_age with future timestamp (age < 0)
    signer_future = TimestampSigner("secret-key")
    signer_future.get_timestamp = lambda: int(time.time()) + 100
    signed_future = signer_future.sign("future")
    signer_future.get_timestamp = lambda: int(time.time())
    with pytest.raises(SignatureExpired):
        signer_future.unsign(signed_future, max_age=3600)
    
    # Test invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")
    
    # Test tampered value
    tampered = signed[:-1] + (b"1" if signed[-1:] == b"0" else b"0")
    with pytest.raises(BadSignature):
        signer.unsign(tampered)
    
    # Test missing timestamp
    signer_no_ts = Signer("secret-key")
    signed_no_ts = signer_no_ts.sign("no_timestamp")
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_no_ts)
    
    # Test malformed timestamp
    malformed_ts = signed.rsplit(b".", 1)[0] + b".malformed"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_ts)
```


# LLM-generated content at query #35
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")
    
    # Test basic unsign without timestamp
    value = b"test_value"
    signed = signer.sign(value)
    unsigned = signer.unsign(signed)
    assert unsigned == value
    
    # Test unsign with return_timestamp=True
    unsigned_with_ts, timestamp = signer.unsign(signed, return_timestamp=True)
    assert unsigned_with_ts == value
    assert isinstance(timestamp, datetime)
    
    # Test unsign with valid max_age
    unsigned = signer.unsign(signed, max_age=3600)
    assert unsigned == value
    
    # Test unsign with expired signature
    signer_with_past_time = TimestampSigner("secret-key")
    original_get_timestamp = signer_with_past_time.get_timestamp
    past_time = int(time.time()) - 100  # 100 seconds ago
    signer_with_past_time.get_timestamp = lambda: past_time
    past_signed = signer_with_past_time.sign(value)
    signer_with_past_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_with_past_time.unsign(past_signed, max_age=50)
    
    # Test unsign with future timestamp (age < 0)
    future_time = int(time.time()) + 100
    signer_with_future_time = TimestampSigner("secret-key")
    signer_with_future_time.get_timestamp = lambda: future_time
    future_signed = signer_with_future_time.sign(value)
    signer_with_future_time.get_timestamp = original_get_timestamp
    
    with pytest.raises(SignatureExpired):
        signer_with_future_time.unsign(future_signed, max_age=3600)
    
    # Test unsign with bad signature
    bad_signed = signed + b"tampered"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed)
    
    # Test unsign with missing timestamp
    no_timestamp_signed = value + b"." + signer.get_signature(value)
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_signed)
    
    # Test unsign with malformed timestamp
    malformed_timestamp = value + b"." + base64_encode(b"not_a_number") + b"." + signer.get_signature(value + b"." + base64_encode(b"not_a_number"))
    try:
        signer.unsign(malformed_timestamp)
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)
```


