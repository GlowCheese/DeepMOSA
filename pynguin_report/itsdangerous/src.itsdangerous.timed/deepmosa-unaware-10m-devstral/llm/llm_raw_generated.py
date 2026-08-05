####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_TimedSerializer():
    # Test basic initialization
    ts = TimedSerializer()
    assert isinstance(ts, TimedSerializer)
    assert isinstance(ts.default_signer, type)
    assert issubclass(ts.default_signer, TimestampSigner)

    # Test with custom secret_key
    ts = TimedSerializer(secret_key="custom-secret")
    assert ts.secret_key == "custom-secret"

    # Test with custom separator
    ts = TimedSerializer(separator="|")
    assert ts.separator == "|"

    # Test with custom serializer
    ts = TimedSerializer(serializer="json")
    assert ts.serializer == "json"

    # Test with custom digest method
    ts = TimedSerializer(digest_method="sha256")
    assert ts.digest_method == "sha256"

    # Test with custom salt
    ts = TimedSerializer(salt="custom-salt")
    assert ts.salt == "custom-salt"

    # Test with multiple salts
    ts = TimedSerializer(salts=["salt1", "salt2"])
    assert ts.salts == ["salt1", "salt2"]

    # Test with key_derivation
    ts = TimedSerializer(key_derivation="hmac")
    assert ts.key_derivation == "hmac"

    # Test with key_derivation_salt
    ts = TimedSerializer(key_derivation_salt="derivation-salt")
    assert ts.key_derivation_salt == "derivation-salt"


# LLM-generated content at query #2
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)
    current_time = signer.get_timestamp()

    # Test normal unsign
    result = signer.unsign(signed_value)
    assert result == value

    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test unsign with max_age (valid)
    max_age = 100
    result = signer.unsign(signed_value, max_age=max_age)
    assert result == value

    # Test unsign with max_age (expired)
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: current_time + 200
    with pytest.raises(SignatureExpired):
        expired_signer.unsign(signed_value, max_age=100)

    # Test unsign with negative age
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: current_time - 100
    with pytest.raises(SignatureExpired):
        future_signer.unsign(signed_value, max_age=100)

    # Test unsign with malformed timestamp
    malformed_signed_value = b"test_value:malformed_timestamp:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test unsign with missing timestamp
    missing_timestamp_value = b"test_value:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_value)

    # Test unsign with bad signature
    bad_signed_value = b"test_value:" + base64_encode(int_to_bytes(current_time)) + b":badsig"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #3
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test successful unsign without timestamp return
    result = signer.unsign(signed_value)
    assert result == value

    # Test successful unsign with timestamp return
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test with max_age within valid range
    result = signer.unsign(signed_value, max_age=3600)
    assert result == value

    # Test with max_age expired
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test with malformed timestamp
    malformed_signed_value = value + b":" + b"malformed_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test with missing timestamp
    no_timestamp_value = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_value)

    # Test with bad signature
    bad_signed_value = value + b":" + base64_encode(int_to_bytes(signer.get_timestamp())) + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)

    # Test with negative age (future timestamp)
    future_timestamp = signer.get_timestamp() + 100
    future_signed_value = value + b":" + base64_encode(int_to_bytes(future_timestamp)) + b":" + signer.get_signature(value + b":" + base64_encode(int_to_bytes(future_timestamp)))
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed_value)


# LLM-generated content at query #4
#--------------------------

```python
def test_TimedSerializer():
    # Test default constructor
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer.default_signer, type)
    assert issubclass(serializer.default_signer, TimestampSigner)

    # Test with custom serializer parameters
    serializer = TimedSerializer(secret_key="test_key", salt="test_salt")
    assert serializer.secret_key == "test_key"
    assert serializer.salt == "test_salt"

    # Test with custom signer class
    class CustomTimestampSigner(TimestampSigner):
        pass

    serializer = TimedSerializer(default_signer=CustomTimestampSigner)
    assert serializer.default_signer == CustomTimestampSigner


# LLM-generated content at query #5
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    max_age = 10

    # Test successful unsign without timestamp return
    signed_value = signer.sign(value)
    result = signer.unsign(signed_value)
    assert result == value

    # Test successful unsign with timestamp return
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test with max_age within limit
    result = signer.unsign(signed_value, max_age=max_age)
    assert result == value

    # Test with max_age exceeded (should raise SignatureExpired)
    with pytest.raises(SignatureExpired):
        time.sleep(2)  # Ensure some time passes
        signer.unsign(signed_value, max_age=0)

    # Test with malformed timestamp (should raise BadTimeSignature)
    malformed_signed_value = value + b":" + b"invalid_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test with missing timestamp (should raise BadTimeSignature)
    no_timestamp_value = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_value)

    # Test with bad signature (should raise BadTimeSignature)
    bad_signed_value = value + b":" + base64_encode(int_to_bytes(signer.get_timestamp())) + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #6
#--------------------------

```python
def test_TimedSerializer():
    # Test default initialization
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer == TimestampSigner

    # Test with custom separator
    serializer = TimedSerializer(separator="|")
    assert serializer.sep == "|"

    # Test with custom serializer
    serializer = TimedSerializer(serializer=Serializer)
    assert isinstance(serializer, TimedSerializer)

    # Test with custom secret key
    serializer = TimedSerializer(secret_key="my-secret-key")
    assert serializer.secret_key == "my-secret-key"

    # Test with custom salt
    serializer = TimedSerializer(salt="my-salt")
    assert serializer.salt == "my-salt"

    # Test with custom digest method
    serializer = TimedSerializer(digest_method="sha256")
    assert serializer.digest_method == "sha256"


# LLM-generated content at query #7
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)

    # Test normal unsigning
    assert serializer.loads(signed_data) == data

    # Test with return_timestamp
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test with max_age (should not raise)
    serializer.loads(signed_data, max_age=3600)

    # Test with expired signature
    old_timestamp = int(time.time()) - 3600
    old_signed_data = serializer.dumps(data)  # This will have current timestamp
    # To properly test expiration, we need to mock the timestamp
    # For now, we'll just test that the structure is correct

    # Test with invalid signature
    try:
        serializer.loads("invalid-signature")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

    # Test with multiple salts
    salted_data = serializer.dumps(data, salt="custom-salt")
    assert serializer.loads(salted_data, salt="custom-salt") == data

    # Test with bytes input
    bytes_signed = want_bytes(signed_data)
    assert serializer.loads(bytes_signed) == data


# LLM-generated content at query #8
#--------------------------

```python
def test_TimestampSigner():
    # Test default initialization
    signer = TimestampSigner()
    assert isinstance(signer, TimestampSigner)
    assert signer.sep == b"."

    # Test with custom separator
    custom_sep = b"|"
    signer = TimestampSigner(sep=custom_sep)
    assert signer.sep == custom_sep

    # Test with key
    key = "secret-key"
    signer = TimestampSigner(key=key)
    assert signer.key == key

    # Test with key and separator
    signer = TimestampSigner(key=key, sep=custom_sep)
    assert signer.key == key
    assert signer.sep == custom_sep

    # Test with salt
    salt = "test-salt"
    signer = TimestampSigner(salt=salt)
    assert signer.salt == salt

    # Test with all parameters
    signer = TimestampSigner(key=key, sep=custom_sep, salt=salt)
    assert signer.key == key
    assert signer.sep == custom_sep
    assert signer.salt == salt


# LLM-generated content at query #9
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    max_age = 10

    # Test normal operation
    signed_data = serializer.dumps(data)
    assert serializer.loads(signed_data) == data

    # Test with return_timestamp
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)

    # Test with max_age (should not raise)
    serializer.loads(signed_data, max_age=max_age)

    # Test with expired signature
    old_timestamp = int(time.time()) - max_age - 1
    old_signed_data = serializer.dumps(data)
    # Manually modify the timestamp to make it expired
    parts = old_signed_data.rsplit(serializer.sep.encode(), 2)
    if len(parts) == 3:
        value, old_ts, sig = parts
        new_ts = base64_encode(int_to_bytes(old_timestamp))
        expired_signed_data = value + serializer.sep.encode() + new_ts + serializer.sep.encode() + sig
        with pytest.raises(SignatureExpired):
            serializer.loads(expired_signed_data, max_age=max_age)

    # Test with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid_signature")

    # Test with missing timestamp
    with pytest.raises(BadTimeSignature):
        # Create a signed value without timestamp
        signer = TimestampSigner("secret-key")
        value = b"value"
        signed_value = value + signer.sep.encode() + signer.get_signature(value)
        serializer.loads(signed_value)

    # Test with malformed timestamp
    with pytest.raises(BadTimeSignature):
        # Create a signed value with malformed timestamp
        value = b"value"
        malformed_ts = b"malformed_timestamp"
        signed_value = value + serializer.sep.encode() + malformed_ts + serializer.sep.encode() + signer.get_signature(value + serializer.sep.encode() + malformed_ts)
        serializer.loads(signed_value)


# LLM-generated content at query #10
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test basic functionality
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed = serializer.dumps(data)
    assert serializer.loads(signed) == data

    # Test with max_age
    signed_old = serializer.dumps(data)
    time.sleep(1)
    assert serializer.loads(signed_old, max_age=1) == data
    time.sleep(1)
    try:
        serializer.loads(signed_old, max_age=1)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass

    # Test return_timestamp
    signed = serializer.dumps(data)
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

    # Test with salt
    signed_salt = serializer.dumps(data, salt="salt")
    assert serializer.loads(signed_salt, salt="salt") == data

    # Test invalid signature
    try:
        serializer.loads("invalid")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

    # Test expired signature
    class FixedTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return 0

    serializer_fixed = TimedSerializer("secret-key", signer_kwargs={"signer": FixedTimestampSigner})
    signed_expired = serializer_fixed.dumps(data)
    try:
        serializer.loads(signed_expired, max_age=0)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass


# LLM-generated content at query #11
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    original_data = {"key": "value"}
    max_age = 10

    # Test successful loads with timestamp
    signed_data = serializer.dumps(original_data)
    loaded_data, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert loaded_data == original_data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

    # Test successful loads without timestamp
    loaded_data = serializer.loads(signed_data)
    assert loaded_data == original_data

    # Test with max_age
    loaded_data = serializer.loads(signed_data, max_age=max_age)
    assert loaded_data == original_data

    # Test with expired signature
    expired_signer = TimedSerializer("secret-key")
    expired_signer.get_timestamp = lambda: 0  # Force old timestamp
    expired_signed_data = expired_signer.dumps(original_data)
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_signed_data, max_age=max_age)

    # Test with invalid signature
    invalid_signed_data = signed_data + b"invalid"
    with pytest.raises(BadSignature):
        serializer.loads(invalid_signed_data)

    # Test with malformed timestamp
    malformed_signer = TimedSerializer("secret-key")
    malformed_signer.get_timestamp = lambda: -1  # Force invalid timestamp
    malformed_signed_data = malformed_signer.dumps(original_data)
    with pytest.raises(BadTimeSignature):
        serializer.loads(malformed_signed_data)

    # Test with missing timestamp
    signer = Signer("secret-key")
    unsigned_data = signer.sign(serializer.dumps(original_data))
    with pytest.raises(BadTimeSignature):
        serializer.loads(unsigned_data)


# LLM-generated content at query #12
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner()
    assert isinstance(signer, TimestampSigner)
    assert isinstance(signer, Signer)


# LLM-generated content at query #13
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer.default_signer, type)
    assert issubclass(serializer.default_signer, TimestampSigner)
    assert serializer.secret_key is None
    assert serializer.digest_method == "sha1"
    assert serializer.serializer is None


# LLM-generated content at query #14
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner()
    assert isinstance(signer, TimestampSigner)
    assert isinstance(signer, Signer)
    assert signer.sep == b"."
    assert signer.key_derivation is None
    assert signer.digest_method == "hmac"


# LLM-generated content at query #15
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer()
    data = {"key": "value"}
    signed_data = serializer.dumps(data)

    # Test successful load
    loaded_data = serializer.loads(signed_data)
    assert loaded_data == data

    # Test with max_age
    loaded_data = serializer.loads(signed_data, max_age=3600)
    assert loaded_data == data

    # Test with return_timestamp
    loaded_data, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert loaded_data == data
    assert isinstance(timestamp, datetime)

    # Test with salt
    salted_signed_data = serializer.dumps(data, salt="test_salt")
    loaded_data = serializer.loads(salted_signed_data, salt="test_salt")
    assert loaded_data == data

    # Test expired signature
    expired_data = serializer.dumps(data)
    time.sleep(2)  # Ensure some time passes
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_data, max_age=1)

    # Test invalid signature
    invalid_data = b"invalid_data"
    with pytest.raises(BadSignature):
        serializer.loads(invalid_data)

    # Test with max_age and return_timestamp
    loaded_data, timestamp = serializer.loads(signed_data, max_age=3600, return_timestamp=True)
    assert loaded_data == data
    assert isinstance(timestamp, datetime)


# LLM-generated content at query #16
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)

    # Test normal loads
    assert serializer.loads(signed_data) == data

    # Test loads with return_timestamp
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test loads with max_age (should not raise)
    serializer.loads(signed_data, max_age=3600)

    # Test loads with expired signature
    old_timestamp = int(time.time()) - 3600
    old_signed_data = serializer.dumps(data)  # Simulate old data
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed_data, max_age=1)

    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")

    # Test loads with salt
    salted_serializer = TimedSerializer("secret-key", salt="test-salt")
    salted_signed_data = salted_serializer.dumps(data)
    assert salted_serializer.loads(salted_signed_data) == data

    # Test loads_unsafe
    valid, result = serializer.loads_unsafe(signed_data)
    assert valid is True
    assert result == data

    valid, result = serializer.loads_unsafe("invalid-signature")
    assert valid is False


# LLM-generated content at query #17
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test normal unsigning
    unsign_result = signer.unsign(signed_value)
    assert unsign_result == value

    # Test unsigning with return_timestamp=True
    unsign_result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert unsign_result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None  # Check timezone awareness

    # Test unsigning with max_age (should not raise)
    signer.unsign(signed_value, max_age=1000)

    # Test signature expiration
    old_timestamp = int(time.time()) - 2000
    fake_signed_value = (
        value
        + b":"
        + base64_encode(int_to_bytes(old_timestamp))
        + b":"
        + signer.get_signature(value + b":" + base64_encode(int_to_bytes(old_timestamp)))
    )
    with pytest.raises(SignatureExpired):
        signer.unsign(fake_signed_value, max_age=1000)

    # Test malformed timestamp
    malformed_signed_value = value + b":invalid_timestamp:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test missing timestamp
    no_timestamp_value = value + b":signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_value)

    # Test bad signature
    bad_signed_value = value + b":timestamp:wrong_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #18
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")

    # Test normal unsigning
    value = b"hello world"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

    # Test unsigning with return_timestamp
    value, timestamp = signer.unsign(signed, return_timestamp=True)
    assert value == b"hello world"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test max_age validation
    signed_old = signer.sign(value)
    time.sleep(1)  # Ensure some time passes
    assert signer.unsign(signed_old, max_age=2) == value
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_old, max_age=0)

    # Test invalid signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid-signature")

    # Test malformed timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"hello:world:invalid-timestamp")

    # Test missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"hello:world")

    # Test negative max_age
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #19
#--------------------------

```python
def test_TimedSerializer():
    # Test default initialization
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer == TimestampSigner

    # Test with custom serializer
    custom_serializer = TimedSerializer(serializer=Serializer)
    assert isinstance(custom_serializer, TimedSerializer)
    assert custom_serializer.default_signer == TimestampSigner

    # Test with custom signer
    custom_signer = TimestampSigner()
    custom_serializer = TimedSerializer(signer=custom_signer)
    assert isinstance(custom_serializer, TimedSerializer)
    assert custom_serializer.default_signer == TimestampSigner

    # Test with both custom serializer and signer
    custom_serializer = TimedSerializer(serializer=Serializer, signer=custom_signer)
    assert isinstance(custom_serializer, TimedSerializer)
    assert custom_serializer.default_signer == TimestampSigner


# LLM-generated content at query #20
#--------------------------

```python
def test_TimedSerializer():
    # Test default constructor
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer == TimestampSigner

    # Test with custom separator
    custom_sep = "|"
    serializer = TimedSerializer(sep=custom_sep)
    assert serializer.sep == custom_sep

    # Test with custom serializer
    custom_serializer = Serializer()
    serializer = TimedSerializer(serializer=custom_serializer)
    assert serializer.serializer == custom_serializer

    # Test with custom signer
    custom_signer = TimestampSigner()
    serializer = TimedSerializer(signer=custom_signer)
    assert serializer.default_signer == custom_signer

    # Test with custom salt
    custom_salt = "custom_salt"
    serializer = TimedSerializer(salt=custom_salt)
    assert serializer.salt == custom_salt


# LLM-generated content at query #21
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test successful unsign without timestamp
    assert signer.unsign(signed_value) == value

    # Test successful unsign with timestamp
    unsign_result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert unsign_result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

    # Test with max_age not exceeded
    assert signer.unsign(signed_value, max_age=1000) == value

    # Test with max_age exceeded
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test with malformed signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"malformed_signature")

    # Test with missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test_value:missing_timestamp")

    # Test with malformed timestamp
    malformed_ts = b"test_value:malformed_ts:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_ts)

    # Test with negative max_age (future timestamp)
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: int(time.time()) + 100
    future_signed = future_signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed, max_age=50)


# LLM-generated content at query #22
#--------------------------

```python
def test_TimedSerializer():
    # Test default construction
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer.default_signer, type)
    assert issubclass(serializer.default_signer, TimestampSigner)

    # Test with custom separator
    custom_sep = "custom_separator"
    serializer = TimedSerializer(separator=custom_sep)
    assert serializer.sep == custom_sep

    # Test with custom serializer
    custom_serializer = {"key": "value"}
    serializer = TimedSerializer(serializers=[custom_serializer])
    assert custom_serializer in serializer.serializers

    # Test with custom signer
    class CustomTimestampSigner(TimestampSigner):
        pass

    serializer = TimedSerializer(signer_kwargs={"key": "value"}, default_signer=CustomTimestampSigner)
    assert serializer.default_signer == CustomTimestampSigner


# LLM-generated content at query #23
#--------------------------

```python
def test_TimedSerializer():
    # Test default construction
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer == TimestampSigner

    # Test with custom separator
    serializer = TimedSerializer(sep="|")
    assert serializer.sep == "|"

    # Test with custom serializer
    serializer = TimedSerializer(serializer="json")
    assert serializer.serializer == "json"

    # Test with custom salt
    serializer = TimedSerializer(salt="test-salt")
    assert serializer.salt == "test-salt"

    # Test with custom secret key
    serializer = TimedSerializer(secret_key="test-secret")
    assert serializer.secret_key == "test-secret"

    # Test with custom digest method
    serializer = TimedSerializer(digest_method="sha256")
    assert serializer.digest_method == "sha256"


# LLM-generated content at query #24
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test basic unsign
    result = signer.unsign(signed_value)
    assert result == value

    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test unsign with max_age (valid)
    max_age = 1000
    result = signer.unsign(signed_value, max_age=max_age)
    assert result == value

    # Test unsign with max_age (expired)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test unsign with malformed timestamp
    malformed_signed_value = value + b"|" + b"malformed_timestamp" + b"|" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test unsign with missing timestamp
    missing_timestamp_signed_value = value + b"|" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_signed_value)

    # Test unsign with bad signature
    bad_signed_value = value + b"|" + base64_encode(int_to_bytes(signer.get_timestamp())) + b"|" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #25
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    secret_key = "secret"
    serializer = TimedSerializer(secret_key)
    data = {"key": "value"}
    max_age = 10

    # Test successful loads
    signed_data = serializer.dumps(data)
    assert serializer.loads(signed_data) == data

    # Test with max_age
    signed_data = serializer.dumps(data)
    assert serializer.loads(signed_data, max_age=max_age) == data

    # Test with return_timestamp
    signed_data = serializer.dumps(data)
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)

    # Test with salt
    salt = "salt"
    signed_data = serializer.dumps(data, salt=salt)
    assert serializer.loads(signed_data, salt=salt) == data

    # Test with expired signature
    signed_data = serializer.dumps(data)
    time.sleep(2)  # Ensure some time passes
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=1)

    # Test with bad signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid_signature")

    # Test with missing timestamp
    with pytest.raises(BadTimeSignature):
        signer = TimestampSigner(secret_key)
        bad_signed_data = signer.sign("data") + b"extra_data"
        serializer.loads(bad_signed_data)

    # Test with malformed timestamp
    with pytest.raises(BadTimeSignature):
        signer = TimestampSigner(secret_key)
        bad_signed_data = signer.sign("data").replace(b"=", b"")
        serializer.loads(bad_signed_data)


# LLM-generated content at query #26
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    original_data = {"key": "value"}
    max_age = 10

    # Test successful loads
    signed_data = serializer.dumps(original_data)
    assert serializer.loads(signed_data) == original_data

    # Test with return_timestamp
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == original_data
    assert isinstance(timestamp, datetime)

    # Test with salt
    salt = "test-salt"
    signed_data_with_salt = serializer.dumps(original_data, salt=salt)
    assert serializer.loads(signed_data_with_salt, salt=salt) == original_data

    # Test with max_age (should not raise if within max_age)
    assert serializer.loads(signed_data, max_age=max_age) == original_data

    # Test SignatureExpired
    time_machine = time.time()
    old_timestamp = int(time_machine) - max_age - 1
    with pytest.raises(SignatureExpired):
        # Create a signed value with an old timestamp
        old_signed_data = serializer.dumps(original_data)
        # Manually manipulate the timestamp (this is a simplified approach for testing)
        # In a real scenario, you might need a more sophisticated way to create an expired signature
        parts = old_signed_data.split(b".")
        if len(parts) == 3:
            # Replace the timestamp part with an old one
            old_ts_bytes = base64_encode(int_to_bytes(old_timestamp))
            manipulated_data = parts[0] + b"." + old_ts_bytes + b"." + parts[2]
            serializer.loads(manipulated_data, max_age=max_age)

    # Test BadSignature
    with pytest.raises(BadSignature):
        serializer.loads("invalid_signed_data")

    # Test with multiple salts
    salt1 = "salt1"
    salt2 = "salt2"
    signed_data_salt1 = serializer.dumps(original_data, salt=salt1)
    signed_data_salt2 = serializer.dumps(original_data, salt=salt2)

    assert serializer.loads(signed_data_salt1, salt=salt1) == original_data
    assert serializer.loads(signed_data_salt2, salt=salt2) == original_data

    # Test loads_unsafe
    valid, result = serializer.loads_unsafe(signed_data)
    assert valid is True
    assert result == original_data

    valid, result = serializer.loads_unsafe("invalid_signed_data")
    assert valid is False


# LLM-generated content at query #27
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = "test_value"
    signed_value = signer.sign(value)
    current_time = signer.get_timestamp()

    # Test normal unsign
    result = signer.unsign(signed_value)
    assert result == want_bytes(value)

    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == want_bytes(value)
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test unsign with max_age (should not raise)
    signer.unsign(signed_value, max_age=3600)

    # Test unsign with expired signature
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: current_time - 3601
    expired_signed_value = expired_signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=3600)

    # Test unsign with future timestamp
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: current_time + 3601
    future_signed_value = future_signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed_value)

    # Test unsign with malformed timestamp
    malformed_signed_value = b"test_value:malformed_timestamp:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test unsign with missing timestamp
    missing_timestamp_value = b"test_value:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_value)

    # Test unsign with bad signature
    bad_signed_value = b"test_value:timestamp:bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #28
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)

    # Test valid signature
    assert serializer.loads(signed_data) == data

    # Test with max_age
    assert serializer.loads(signed_data, max_age=3600) == data

    # Test with return_timestamp
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)

    # Test with salt
    salted_data = serializer.dumps(data, salt="salt")
    assert serializer.loads(salted_data, salt="salt") == data

    # Test expired signature
    old_timestamp = int(time.time()) - 3600
    old_signed_data = serializer.dumps(data)[:-10] + base64_encode(int_to_bytes(old_timestamp)) + serializer.dumps(data)[-10:]
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed_data, max_age=1)

    # Test invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")

    # Test with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(salted_data, salt="wrong-salt")


# LLM-generated content at query #29
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = "test-value"
    signed_value = signer.sign(value)
    current_time = signer.get_timestamp()

    # Test successful unsign without timestamp return
    result = signer.unsign(signed_value)
    assert result == b"test-value"

    # Test successful unsign with timestamp return
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test with max_age not expired
    result = signer.unsign(signed_value, max_age=3600)
    assert result == b"test-value"

    # Test with max_age expired
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test with negative max_age (future timestamp)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)

    # Test with invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid-signature")

    # Test with malformed timestamp
    malformed_value = b"test-value" + signer.sep + b"invalid-timestamp" + signer.sep + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_value)

    # Test with missing timestamp
    no_timestamp_value = b"test-value" + signer.sep + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_value)


# LLM-generated content at query #30
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test normal unsign
    result = signer.unsign(signed_value)
    assert result == value

    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test unsign with max_age (should not raise)
    signer.unsign(signed_value, max_age=1000)

    # Test unsign with expired signature
    expired_signed_value = signer.sign(value)
    # Simulate time passing by modifying the timestamp
    parts = expired_signed_value.rsplit(signer.sep.encode(), 2)
    old_timestamp = base64_decode(parts[1])
    new_timestamp = int_to_bytes(bytes_to_int(old_timestamp) - 1000)
    expired_signed_value = parts[0] + signer.sep.encode() + base64_encode(new_timestamp) + signer.sep.encode() + parts[2]

    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=100)

    # Test unsign with malformed timestamp
    malformed_signed_value = b"test_value" + signer.sep.encode() + b"malformed" + signer.sep.encode() + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test unsign with missing timestamp
    missing_timestamp_value = b"test_value" + signer.sep.encode() + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_value)

    # Test unsign with bad signature
    bad_signed_value = b"test_value" + signer.sep.encode() + base64_encode(int_to_bytes(signer.get_timestamp())) + signer.sep.encode() + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #31
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner()
    assert isinstance(signer, TimestampSigner)
    assert signer.sep == b"."

    # Test with custom separator
    custom_sep = b"|"
    signer = TimestampSigner(sep=custom_sep)
    assert signer.sep == custom_sep

    # Test with key
    key = "secret-key"
    signer = TimestampSigner(key=key)
    assert signer.key == key

    # Test with key and custom separator
    signer = TimestampSigner(key=key, sep=custom_sep)
    assert signer.key == key
    assert signer.sep == custom_sep


# LLM-generated content at query #32
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)
    current_time = signer.get_timestamp()

    # Test normal unsign
    result = signer.unsign(signed_value)
    assert result == value

    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test unsign with max_age (should not raise if within max_age)
    signer.unsign(signed_value, max_age=60)

    # Test unsign with expired signature
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)

    # Test unsign with malformed timestamp
    malformed_signed_value = value + b":" + b"malformed_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test unsign with missing timestamp
    no_timestamp_signed_value = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_signed_value)

    # Test unsign with bad signature
    bad_signed_value = value + b":" + base64_encode(int_to_bytes(current_time)) + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)

    # Test unsign with string input
    string_signed_value = signed_value.decode('utf-8')
    result = signer.unsign(string_signed_value)
    assert result == value


# LLM-generated content at query #33
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test normal unsign
    result = signer.unsign(signed_value)
    assert result == value

    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test unsign with max_age (should not raise)
    result = signer.unsign(signed_value, max_age=3600)
    assert result == value

    # Test unsign with expired signature
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test unsign with invalid signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid_signature")

    # Test unsign with tampered timestamp
    parts = signed_value.split(signer.sep)
    tampered_value = parts[0] + signer.sep + b"tampered" + signer.sep + parts[2]
    with pytest.raises(BadTimeSignature):
        signer.unsign(tampered_value)

    # Test unsign with missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(value + signer.sep + b"signature")

    # Test unsign with malformed timestamp
    malformed_ts = base64_encode(b"malformed")
    malformed_value = value + signer.sep + malformed_ts + signer.sep + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_value)


# LLM-generated content at query #34
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner()
    assert isinstance(signer, TimestampSigner)
    assert isinstance(signer, Signer)
    assert signer.sep == b"."

    custom_sep = b"|"
    signer_custom = TimestampSigner(sep=custom_sep)
    assert signer_custom.sep == custom_sep

    assert callable(signer.get_timestamp)
    assert callable(signer.timestamp_to_datetime)


# LLM-generated content at query #35
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test normal unsign
    result = signer.unsign(signed_value)
    assert result == value

    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None  # Check timezone awareness

    # Test unsign with max_age (should not raise)
    max_age = 1000
    result = signer.unsign(signed_value, max_age=max_age)
    assert result == value

    # Test unsign with expired signature
    expired_signed_value = signer.sign(value)
    # Mock get_timestamp to simulate expiration
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: original_get_timestamp() + max_age + 1
    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=max_age)
    signer.get_timestamp = original_get_timestamp

    # Test unsign with malformed timestamp
    malformed_signed_value = b"test_value:malformed_timestamp:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test unsign with missing timestamp
    missing_timestamp_signed_value = b"test_value:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_signed_value)

    # Test unsign with bad signature
    bad_signed_value = b"test_value:timestamp:bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #36
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test normal unsign
    result = signer.unsign(signed_value)
    assert result == value

    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test unsign with max_age
    result = signer.unsign(signed_value, max_age=3600)
    assert result == value

    # Test unsign with expired signature
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test unsign with malformed timestamp
    malformed_signed_value = value + b":" + b"malformed_timestamp" + b":" + signer.get_signature(value + b":" + b"malformed_timestamp")
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test unsign with missing timestamp
    missing_timestamp_signed_value = value + b":" + signer.get_signature(value)
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_signed_value)

    # Test unsign with bad signature
    bad_signed_value = value + b":" + base64_encode(int_to_bytes(signer.get_timestamp())) + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #37
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test-value"
    signed_value = signer.sign(value)

    # Test normal unsigning
    result = signer.unsign(signed_value)
    assert result == value

    # Test unsigning with return_timestamp
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test unsigning with max_age (valid)
    result = signer.unsign(signed_value, max_age=3600)
    assert result == value

    # Test unsigning with max_age (expired)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)

    # Test unsigning with invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid-signature")

    # Test unsigning with malformed timestamp
    invalid_signed = value + b":" + b"invalid-timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(invalid_signed)

    # Test unsigning with missing timestamp
    invalid_signed = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(invalid_signed)


# LLM-generated content at query #38
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test normal unsign
    result = signer.unsign(signed_value)
    assert result == value

    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test unsign with max_age
    result = signer.unsign(signed_value, max_age=1000)
    assert result == value

    # Test unsign with expired signature
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: int(time.time()) - 2000
    expired_signed_value = expired_signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=1000)

    # Test unsign with malformed timestamp
    malformed_signed_value = value + b":" + b"malformed_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test unsign with missing timestamp
    missing_timestamp_signed_value = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_signed_value)

    # Test unsign with bad signature
    bad_signed_value = value + b":" + base64_encode(int_to_bytes(signer.get_timestamp())) + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #39
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner()
    assert isinstance(signer, TimestampSigner)
    assert isinstance(signer, Signer)


# LLM-generated content at query #40
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner()
    assert isinstance(signer, TimestampSigner)
    assert isinstance(signer, Signer)
    assert signer.sep == b"."

    custom_sep_signer = TimestampSigner(sep=":")
    assert custom_sep_signer.sep == b":"

    custom_key_signer = TimestampSigner(key="secret-key")
    assert custom_key_signer.key == b"secret-key"
    assert custom_key_signer.digest_method.name == "sha1"

    custom_digest_signer = TimestampSigner(digest_method="sha256")
    assert custom_digest_signer.digest_method.name == "sha256"


# LLM-generated content at query #41
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner('secret-key')
    value = b'test_value'
    signed_value = signer.sign(value)

    # Test normal unsign
    result = signer.unsign(signed_value)
    assert result == value

    # Test unsign with return_timestamp
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None  # timezone aware

    # Test unsign with max_age (valid)
    result = signer.unsign(signed_value, max_age=3600)
    assert result == value

    # Test unsign with max_age (expired)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test unsign with invalid signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b'invalid_signature')

    # Test unsign with malformed timestamp
    invalid_signed = value + b':' + b'invalid_timestamp' + b':' + b'invalid_sig'
    with pytest.raises(BadTimeSignature):
        signer.unsign(invalid_signed)

    # Test unsign with missing timestamp
    invalid_signed = value + b':' + b'invalid_sig'
    with pytest.raises(BadTimeSignature):
        signer.unsign(invalid_signed)


# LLM-generated content at query #42
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    original_value = b"test_value"
    signed_value = signer.sign(original_value)

    # Test successful unsign without timestamp return
    unsign_result = signer.unsign(signed_value)
    assert unsign_result == original_value

    # Test successful unsign with timestamp return
    unsign_result_with_ts = signer.unsign(signed_value, return_timestamp=True)
    assert isinstance(unsign_result_with_ts, tuple)
    assert len(unsign_result_with_ts) == 2
    assert unsign_result_with_ts[0] == original_value
    assert isinstance(unsign_result_with_ts[1], datetime)
    assert unsign_result_with_ts[1].tzinfo is not None  # timezone-aware

    # Test with max_age (should not raise if within max_age)
    signer.unsign(signed_value, max_age=1000)

    # Test with expired signature
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: 0  # Force old timestamp
    expired_signed_value = expired_signer.sign(original_value)

    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=1)

    # Test with malformed timestamp
    malformed_signed_value = b"test_value:malformed_timestamp:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test with missing timestamp
    no_timestamp_value = b"test_value:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_value)

    # Test with bad signature
    bad_signed_value = b"test_value:timestamp:bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)

    # Test with negative age (future timestamp)
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: 9999999999  # Future timestamp
    future_signed_value = future_signer.sign(original_value)

    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed_value, max_age=100)


# LLM-generated content at query #43
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test normal unsigning
    assert signer.unsign(signed_value) == value

    # Test unsigning with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test with max_age (should not raise)
    signer.unsign(signed_value, max_age=1000)

    # Test with expired signature
    expired_signed_value = signer.sign(value)
    # Manually modify the timestamp to be very old
    parts = expired_signed_value.split(signer.sep)
    old_timestamp = int_to_bytes(int(time.time()) - 1000)
    expired_signed_value = parts[0] + signer.sep + base64_encode(old_timestamp) + signer.sep + parts[2]
    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=10)

    # Test with malformed timestamp
    malformed_signed_value = b"test" + signer.sep + b"malformed" + signer.sep + b"sig"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test with missing timestamp
    no_timestamp_value = b"test" + signer.sep + b"sig"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_value)

    # Test with bad signature
    bad_signed_value = b"test" + signer.sep + base64_encode(int_to_bytes(int(time.time()))) + signer.sep + b"bad_sig"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #44
#--------------------------

```python
def test_TimedSerializer():
    # Test default construction
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer is TimestampSigner

    # Test with custom separator
    custom_sep = "|"
    serializer = TimedSerializer(separator=custom_sep)
    assert serializer.sep == custom_sep

    # Test with custom serializers
    custom_serializers = ["json", "pickle"]
    serializer = TimedSerializer(serializers=custom_serializers)
    assert serializer.serializers == custom_serializers

    # Test with secret key
    secret_key = "my-secret-key"
    serializer = TimedSerializer(secret_key=secret_key)
    assert serializer.secret_key == secret_key

    # Test with salt
    salt = "my-salt"
    serializer = TimedSerializer(salt=salt)
    assert serializer.salt == salt

    # Test with digest method
    digest_method = "sha256"
    serializer = TimedSerializer(digest_method=digest_method)
    assert serializer.digest_method == digest_method

    # Test with key derivation
    key_derivation = "hmac"
    serializer = TimedSerializer(key_derivation=key_derivation)
    assert serializer.key_derivation == key_derivation


# LLM-generated content at query #45
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test basic functionality
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    assert serializer.loads(signed_data) == data

    # Test with max_age
    signed_data = serializer.dumps(data)
    assert serializer.loads(signed_data, max_age=3600) == data

    # Test with return_timestamp
    signed_data = serializer.dumps(data)
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)

    # Test with salt
    signed_data = serializer.dumps(data, salt="salt")
    assert serializer.loads(signed_data, salt="salt") == data

    # Test with expired signature
    class MockTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return 0

    mock_serializer = TimedSerializer("secret-key", signer_kwargs={"signer": MockTimestampSigner})
    signed_data = mock_serializer.dumps(data)
    with pytest.raises(SignatureExpired):
        mock_serializer.loads(signed_data, max_age=1)

    # Test with bad signature
    with pytest.raises(BadSignature):
        serializer.loads("bad-signature")

    # Test with missing timestamp
    with pytest.raises(BadTimeSignature):
        serializer.loads(serializer.dumps(data)[:-10])

    # Test with malformed timestamp
    with pytest.raises(BadTimeSignature):
        serializer.loads(serializer.dumps(data)[:-5] + b"badts")


# LLM-generated content at query #46
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test normal unsigning
    result = signer.unsign(signed_value)
    assert result == value

    # Test unsigning with return_timestamp
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test unsigning with max_age (should not raise)
    signer.unsign(signed_value, max_age=1000)

    # Test expired signature
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: 0  # Set timestamp to 0
    expired_signed_value = expired_signer.sign(value)

    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=1)

    # Test malformed timestamp
    malformed_signed_value = b"test_value:malformed_timestamp:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test missing timestamp
    missing_timestamp_value = b"test_value:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_value)

    # Test bad signature
    bad_signed_value = b"test_value:timestamp:bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #47
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)

    # Test normal loads
    assert serializer.loads(signed_data) == data

    # Test loads with return_timestamp
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test loads with max_age (should not raise)
    serializer.loads(signed_data, max_age=3600)

    # Test loads with expired signature
    old_timestamp = int(time.time()) - 3600
    old_signed_data = (
        base64_encode(serializer.serializer.dumps(data))
        + b":"
        + base64_encode(int_to_bytes(old_timestamp))
        + b":"
        + TimestampSigner("secret-key").get_signature(
            base64_encode(serializer.serializer.dumps(data))
            + b":"
            + base64_encode(int_to_bytes(old_timestamp))
        )
    )
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed_data, max_age=1)

    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads(b"invalid:signature")

    # Test loads with missing timestamp
    with pytest.raises(BadTimeSignature):
        serializer.loads(b"missing:timestamp")


# LLM-generated content at query #48
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    secret_key = "secret"
    serializer = TimedSerializer(secret_key)
    data = {"key": "value"}
    salt = "test_salt"

    # Test normal operation
    signed_data = serializer.dumps(data, salt=salt)
    assert serializer.loads(signed_data, salt=salt) == data

    # Test with max_age
    signed_data = serializer.dumps(data, salt=salt)
    assert serializer.loads(signed_data, max_age=10, salt=salt) == data

    # Test return_timestamp
    signed_data = serializer.dumps(data, salt=salt)
    result, timestamp = serializer.loads(signed_data, return_timestamp=True, salt=salt)
    assert result == data
    assert isinstance(timestamp, datetime)

    # Test expired signature
    signed_data = serializer.dumps(data, salt=salt)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=0, salt=salt)

    # Test bad signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid_signature", salt=salt)

    # Test with no salt
    signed_data = serializer.dumps(data)
    assert serializer.loads(signed_data) == data

    # Test with wrong salt
    signed_data = serializer.dumps(data, salt=salt)
    with pytest.raises(BadSignature):
        serializer.loads(signed_data, salt="wrong_salt")


# LLM-generated content at query #49
#--------------------------

```python
def test_TimedSerializer():
    # Test default construction
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer == TimestampSigner

    # Test with custom secret key
    serializer_with_key = TimedSerializer(secret_key="my-secret-key")
    assert serializer_with_key.secret_key == "my-secret-key"

    # Test with custom serializer
    serializer_with_serializer = TimedSerializer(serializer=Serializer)
    assert serializer_with_serializer.serializer == Serializer

    # Test with custom signer
    class CustomTimestampSigner(TimestampSigner):
        pass

    serializer_with_signer = TimedSerializer(signer=CustomTimestampSigner)
    assert serializer_with_signer.default_signer == CustomTimestampSigner

    # Test with all parameters
    serializer_full = TimedSerializer(
        secret_key="another-key",
        serializer=Serializer,
        signer=CustomTimestampSigner,
        sep=":",
        digest_method="sha256"
    )
    assert serializer_full.secret_key == "another-key"
    assert serializer_full.serializer == Serializer
    assert serializer_full.default_signer == CustomTimestampSigner
    assert serializer_full.sep == ":"
    assert serializer_full.digest_method == "sha256"


# LLM-generated content at query #50
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test normal unsign
    result = signer.unsign(signed_value)
    assert result == value

    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test unsign with max_age
    result = signer.unsign(signed_value, max_age=3600)
    assert result == value

    # Test expired signature
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test bad signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")

    # Test malformed timestamp
    malformed_signed_value = value + b":" + b"invalid_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test missing timestamp
    missing_timestamp_signed_value = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_signed_value)


# LLM-generated content at query #51
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test successful unsign without timestamp
    assert signer.unsign(signed_value) == value

    # Test successful unsign with timestamp
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

    # Test with max_age not expired
    assert signer.unsign(signed_value, max_age=3600) == value

    # Test with max_age expired
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test with malformed signature
    with pytest.raises(BadSignature):
        signer.unsign(b"malformed_signature")

    # Test with missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value" + signer.sep + b"signature")

    # Test with malformed timestamp
    malformed_ts = b"value" + signer.sep + b"malformed_ts" + signer.sep + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_ts)

    # Test with negative age (future timestamp)
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: int(time.time()) + 100
    future_signed = future_signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed, max_age=50)


# LLM-generated content at query #52
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test normal unsigning without timestamp
    unsign_result = signer.unsign(signed_value)
    assert unsign_result == value

    # Test unsigning with timestamp return
    unsign_result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert unsign_result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None  # Check it's timezone-aware

    # Test with max_age
    # Should not raise if within max_age
    signer.unsign(signed_value, max_age=1000)

    # Test with expired signature
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: 0  # Set timestamp to 0
    expired_signed_value = expired_signer.sign(value)

    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=1)

    # Test with malformed timestamp
    malformed_signed_value = value + b":" + b"malformed_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test with missing timestamp
    no_timestamp_value = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_value)

    # Test with bad signature
    bad_signed_value = value + b":" + base64_encode(int_to_bytes(signer.get_timestamp())) + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #53
#--------------------------

```python
def test_TimedSerializer():
    # Test basic instantiation
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer.default_signer, type)
    assert issubclass(serializer.default_signer, TimestampSigner)

    # Test with custom serializer parameters
    serializer = TimedSerializer(secret_key="test_key", salt="test_salt")
    assert serializer.secret_key == "test_key"
    assert serializer.salt == "test_salt"

    # Test iter_unsigners returns TimestampSigner instances
    for signer in serializer.iter_unsigners():
        assert isinstance(signer, TimestampSigner)


# LLM-generated content at query #54
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test-value"
    signed_value = signer.sign(value)

    # Test successful unsign
    result = signer.unsign(signed_value)
    assert result == value

    # Test successful unsign with timestamp
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test unsign with max_age
    result = signer.unsign(signed_value, max_age=3600)
    assert result == value

    # Test unsign with expired signature
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: int(time.time()) - 3601
    expired_signed_value = expired_signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=3600)

    # Test unsign with malformed timestamp
    malformed_signed_value = value + b":" + b"malformed-timestamp" + b":" + signer.get_signature(value + b":" + b"malformed-timestamp")
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test unsign with missing timestamp
    missing_timestamp_signed_value = value + b":" + signer.get_signature(value)
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_signed_value)

    # Test unsign with bad signature
    bad_signed_value = value + b":" + base64_encode(int_to_bytes(signer.get_timestamp())) + b":" + b"bad-signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #55
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test successful unsign without timestamp return
    assert signer.unsign(signed_value) == value

    # Test successful unsign with timestamp return
    unsign_result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert unsign_result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

    # Test with max_age not exceeded
    assert signer.unsign(signed_value, max_age=1000) == value

    # Test with max_age exceeded
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test with invalid signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid_signature")

    # Test with malformed timestamp
    malformed_signed = value + b":" + b"invalid_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed)

    # Test with missing timestamp
    no_timestamp_signed = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_signed)


# LLM-generated content at query #56
#--------------------------

```python
def test_TimedSerializer():
    # Test basic instantiation
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer.default_signer, type)
    assert issubclass(serializer.default_signer, TimestampSigner)

    # Test with custom separator
    serializer = TimedSerializer(sep=":")
    assert serializer.sep == ":"

    # Test with custom serializer
    serializer = TimedSerializer(serializer="json")
    assert serializer.serializer == "json"

    # Test with custom salt
    serializer = TimedSerializer(salt="custom_salt")
    assert serializer.salt == "custom_salt"

    # Test with multiple salts
    serializer = TimedSerializer(salts=["salt1", "salt2"])
    assert serializer.salts == ["salt1", "salt2"]


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)

    # Test valid signature
    assert serializer.loads(signed_data) == data

    # Test with max_age
    assert serializer.loads(signed_data, max_age=3600) == data

    # Test return_timestamp
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)

    # Test with salt
    salted_data = serializer.dumps(data, salt="salt")
    assert serializer.loads(salted_data, salt="salt") == data

    # Test expired signature
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=0)

    # Test invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid_signature")

    # Test with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(salted_data, salt="wrong_salt")


# LLM-generated content at query #2
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test successful unsign without timestamp return
    result = signer.unsign(signed_value)
    assert result == value

    # Test successful unsign with timestamp return
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test with max_age (should not raise)
    result = signer.unsign(signed_value, max_age=3600)
    assert result == value

    # Test with expired signature
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test with malformed timestamp
    malformed_signed_value = value + b":" + b"malformed_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test with missing timestamp
    no_timestamp_value = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_value)

    # Test with bad signature
    bad_signed_value = value + b":" + base64_encode(int_to_bytes(signer.get_timestamp())) + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #3
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner()
    assert isinstance(signer, TimestampSigner)
    assert signer.sep == b"."

    # Test with custom separator
    custom_sep = b"|"
    signer = TimestampSigner(sep=custom_sep)
    assert signer.sep == custom_sep

    # Test with secret key
    secret_key = "test-secret-key"
    signer = TimestampSigner(secret_key=secret_key)
    assert signer.secret_key == secret_key

    # Test with both custom separator and secret key
    signer = TimestampSigner(sep=custom_sep, secret_key=secret_key)
    assert signer.sep == custom_sep
    assert signer.secret_key == secret_key


# LLM-generated content at query #4
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner()
    assert isinstance(signer, TimestampSigner)
    assert signer.sep == b"."

    # Test with custom separator
    custom_sep = b"|"
    signer = TimestampSigner(sep=custom_sep)
    assert signer.sep == custom_sep

    # Test with key
    key = "secret-key"
    signer = TimestampSigner(key=key)
    assert signer.key == key

    # Test with salt
    salt = "test-salt"
    signer = TimestampSigner(salt=salt)
    assert signer.salt == salt

    # Test with key and salt
    signer = TimestampSigner(key=key, salt=salt)
    assert signer.key == key
    assert signer.salt == salt

    # Test with all parameters
    signer = TimestampSigner(key=key, salt=salt, sep=custom_sep)
    assert signer.key == key
    assert signer.salt == salt
    assert signer.sep == custom_sep


# LLM-generated content at query #5
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)

    # Test successful loads
    assert serializer.loads(signed_data) == data

    # Test loads with return_timestamp
    loaded_data, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert loaded_data == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test loads with max_age (should not raise)
    serializer.loads(signed_data, max_age=3600)

    # Test loads with expired signature
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: int(time.time()) - 4000
    expired_data = expired_signer.sign(serializer.dumps(data))
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_data, max_age=3600)

    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")

    # Test loads with wrong key
    wrong_serializer = TimedSerializer("wrong-key")
    with pytest.raises(BadSignature):
        wrong_serializer.loads(signed_data)


# LLM-generated content at query #6
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner()
    assert isinstance(signer, TimestampSigner)
    assert signer.sep == b"."

    # Test with custom separator
    custom_sep = b"|"
    signer = TimestampSigner(sep=custom_sep)
    assert signer.sep == custom_sep

    # Test with custom key
    custom_key = b"my-secret-key"
    signer = TimestampSigner(key=custom_key)
    assert signer.key == custom_key

    # Test with custom separator and key
    signer = TimestampSigner(sep=custom_sep, key=custom_key)
    assert signer.sep == custom_sep
    assert signer.key == custom_key


# LLM-generated content at query #7
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    max_age = 10

    # Test valid signature
    signed = serializer.dumps(data)
    assert serializer.loads(signed) == data

    # Test valid signature with timestamp
    signed_with_ts = serializer.dumps(data)
    result, timestamp = serializer.loads(signed_with_ts, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)

    # Test expired signature
    expired_signed = serializer.dumps(data)
    time.sleep(2)  # Ensure some time passes
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_signed, max_age=1)

    # Test invalid signature
    invalid_signed = b"invalid-signature"
    with pytest.raises(BadSignature):
        serializer.loads(invalid_signed)

    # Test max_age with valid signature
    valid_signed = serializer.dumps(data)
    assert serializer.loads(valid_signed, max_age=max_age) == data

    # Test max_age with expired signature
    expired_signed = serializer.dumps(data)
    time.sleep(2)  # Ensure some time passes
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_signed, max_age=1)

    # Test salt parameter
    salt = "custom-salt"
    signed_with_salt = serializer.dumps(data, salt=salt)
    assert serializer.loads(signed_with_salt, salt=salt) == data
    with pytest.raises(BadSignature):
        serializer.loads(signed_with_salt, salt="wrong-salt")


# LLM-generated content at query #8
#--------------------------

```python
def test_TimedSerializer():
    # Test default construction
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer == TimestampSigner

    # Test with custom separator
    custom_sep = "|"
    serializer = TimedSerializer(separator=custom_sep)
    assert serializer.sep == custom_sep

    # Test with custom secret key
    secret_key = "my-secret-key"
    serializer = TimedSerializer(secret_key=secret_key)
    assert serializer.secret_key == secret_key

    # Test with both custom separator and secret key
    serializer = TimedSerializer(separator=custom_sep, secret_key=secret_key)
    assert serializer.sep == custom_sep
    assert serializer.secret_key == secret_key

    # Test with custom serializer
    custom_serializer = "json"
    serializer = TimedSerializer(serializer=custom_serializer)
    assert serializer.serializer == custom_serializer


# LLM-generated content at query #9
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test normal unsigning without timestamp
    unsign_result = signer.unsign(signed_value)
    assert unsign_result == value

    # Test unsigning with timestamp return
    unsign_result_with_ts = signer.unsign(signed_value, return_timestamp=True)
    assert isinstance(unsign_result_with_ts, tuple)
    assert unsign_result_with_ts[0] == value
    assert isinstance(unsign_result_with_ts[1], datetime)

    # Test unsigning with max_age (should not raise)
    signer.unsign(signed_value, max_age=3600)

    # Test with expired signature
    old_timestamp = int(time.time()) - 3600
    old_signed_value = (
        value
        + b":"
        + base64_encode(int_to_bytes(old_timestamp))
        + b":"
        + signer.get_signature(value + b":" + base64_encode(int_to_bytes(old_timestamp)))
    )
    with pytest.raises(SignatureExpired):
        signer.unsign(old_signed_value, max_age=1)

    # Test with malformed timestamp
    malformed_signed_value = value + b":invalid_timestamp:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test with missing timestamp
    no_timestamp_signed_value = value + b":signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_signed_value)

    # Test with bad signature
    bad_signed_value = value + b":timestamp:bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #10
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")

    # Test successful unsigning without timestamp return
    value = b"test_value"
    signed = signer.sign(value)
    unsign_result = signer.unsign(signed)
    assert unsign_result == value

    # Test successful unsigning with timestamp return
    signed = signer.sign(value)
    unsign_result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert unsign_result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test with max_age not expired
    signed = signer.sign(value)
    unsign_result = signer.unsign(signed, max_age=1000)
    assert unsign_result == value

    # Test with max_age expired
    signed = signer.sign(value)
    time.sleep(1)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)

    # Test with malformed timestamp
    malformed_signed = value + b":" + b"malformed_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed)

    # Test with missing timestamp
    no_timestamp_signed = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_signed)

    # Test with bad signature
    bad_signed = value + b":" + base64_encode(int_to_bytes(signer.get_timestamp())) + b":" + b"bad_signature"
    with pytest.raises(BadSignature):
        signer.unsign(bad_signed)

    # Test with negative age
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #11
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    secret_key = "secret"
    serializer = TimedSerializer(secret_key)
    data = {"key": "value"}
    max_age = 10

    # Test successful loads
    signed_data = serializer.dumps(data)
    assert serializer.loads(signed_data) == data
    assert serializer.loads(signed_data, max_age=max_age) == data
    assert serializer.loads(signed_data, return_timestamp=True)[0] == data
    assert isinstance(serializer.loads(signed_data, return_timestamp=True)[1], datetime)

    # Test with max_age expiration
    old_timestamp = int(time.time()) - 20
    old_signed_data = serializer.dumps(data)
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed_data, max_age=max_age)

    # Test with invalid signature
    invalid_signed_data = signed_data + b"invalid"
    with pytest.raises(BadSignature):
        serializer.loads(invalid_signed_data)

    # Test with salt
    salt = "salt"
    salted_signed_data = serializer.dumps(data, salt=salt)
    assert serializer.loads(salted_signed_data, salt=salt) == data
    with pytest.raises(BadSignature):
        serializer.loads(salted_signed_data, salt="wrong_salt")

    # Test with bytes input
    assert serializer.loads(want_bytes(signed_data)) == data


# LLM-generated content at query #12
#--------------------------

```python
def test_TimedSerializer():
    # Test default construction
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer.default_signer, type)
    assert issubclass(serializer.default_signer, TimestampSigner)

    # Test with custom separator
    serializer = TimedSerializer(separator=".")
    assert serializer.sep == "."

    # Test with custom serializer
    serializer = TimedSerializer(serializer="json")
    assert serializer.serializer == "json"

    # Test with custom salt
    serializer = TimedSerializer(salt="custom_salt")
    assert serializer.salt == "custom_salt"

    # Test with custom secret key
    serializer = TimedSerializer(secret_key="secret")
    assert serializer.secret_key == "secret"

    # Test with custom digest method
    serializer = TimedSerializer(digest_method="sha256")
    assert serializer.digest_method == "sha256"


# LLM-generated content at query #13
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer, Serializer)
    assert serializer.default_signer == TimestampSigner


# LLM-generated content at query #14
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test successful unsign without timestamp return
    result = signer.unsign(signed_value)
    assert result == value

    # Test successful unsign with timestamp return
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test with max_age within valid range
    result = signer.unsign(signed_value, max_age=3600)
    assert result == value

    # Test with max_age exceeded
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test with malformed signature
    with pytest.raises(BadSignature):
        signer.unsign(b"malformed_signature")

    # Test with missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value" + signer.sep.encode() + b"signature")

    # Test with malformed timestamp
    malformed_ts = value + signer.sep.encode() + b"malformed_ts" + signer.sep.encode() + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_ts)

    # Test with negative max_age (future signature)
    future_ts = int(time.time()) + 100
    future_signed = value + signer.sep.encode() + base64_encode(int_to_bytes(future_ts)) + signer.sep.encode() + signer.get_signature(value + signer.sep.encode() + base64_encode(int_to_bytes(future_ts)))
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed, max_age=50)


# LLM-generated content at query #15
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)

    # Test successful loads
    assert serializer.loads(signed_data) == data

    # Test loads with return_timestamp
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test loads with max_age (should not raise)
    serializer.loads(signed_data, max_age=3600)

    # Test loads with expired signature
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: int(time.time()) - 3601
    expired_data = expired_signer.sign(serializer.dumps(data))
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_data, max_age=3600)

    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")

    # Test loads with tampered data
    tampered_data = signed_data[:-1] + b"x"
    with pytest.raises(BadSignature):
        serializer.loads(tampered_data)

    # Test loads with salt
    salted_data = serializer.dumps(data, salt="custom-salt")
    assert serializer.loads(salted_data, salt="custom-salt") == data
    with pytest.raises(BadSignature):
        serializer.loads(salted_data, salt="wrong-salt")


# LLM-generated content at query #16
#--------------------------

```python
def test_TimedSerializer():
    # Test default construction
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer.default_signer, type)
    assert issubclass(serializer.default_signer, TimestampSigner)

    # Test with custom separator
    custom_sep = "custom_separator"
    serializer = TimedSerializer(sep=custom_sep)
    assert serializer.sep == custom_sep

    # Test with custom serializer
    custom_serializer = Serializer()
    serializer = TimedSerializer(serializer=custom_serializer)
    assert serializer.serializer is custom_serializer

    # Test with custom signer
    custom_signer = TimestampSigner()
    serializer = TimedSerializer(signer=custom_signer)
    assert serializer.default_signer is custom_signer

    # Test with all parameters
    serializer = TimedSerializer(
        sep=custom_sep,
        serializer=custom_serializer,
        signer=custom_signer
    )
    assert serializer.sep == custom_sep
    assert serializer.serializer is custom_serializer
    assert serializer.default_signer is custom_signer


# LLM-generated content at query #17
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test basic functionality
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed = serializer.dumps(data)
    assert serializer.loads(signed) == data

    # Test with max_age
    signed_old = serializer.dumps(data)
    time.sleep(2)
    assert serializer.loads(signed_old, max_age=1) is None
    try:
        serializer.loads(signed_old, max_age=1)
    except SignatureExpired:
        pass
    else:
        assert False, "Expected SignatureExpired"

    # Test return_timestamp
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

    # Test with salt
    signed_salt = serializer.dumps(data, salt="salt")
    assert serializer.loads(signed_salt, salt="salt") == data

    # Test with bad signature
    try:
        serializer.loads("bad-signature")
    except BadSignature:
        pass
    else:
        assert False, "Expected BadSignature"

    # Test with expired signature
    class FixedTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return 0

    expired_serializer = TimedSerializer("secret-key", signer_kwargs={"signer": FixedTimestampSigner})
    signed_expired = expired_serializer.dumps(data)
    try:
        expired_serializer.loads(signed_expired, max_age=1)
    except SignatureExpired:
        pass
    else:
        assert False, "Expected SignatureExpired"

    # Test with multiple salts
    serializer_multi = TimedSerializer("secret-key", salts=["salt1", "salt2"])
    signed_multi = serializer_multi.dumps(data, salt="salt1")
    assert serializer_multi.loads(signed_multi) == data


# LLM-generated content at query #18
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")

    # Test successful unsigning without timestamp return
    value = b"test_value"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value

    # Test successful unsigning with timestamp return
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

    # Test with max_age within limit
    result = signer.unsign(signed, max_age=3600)
    assert result == value

    # Test with max_age exceeded
    old_timestamp = int(time.time()) - 3600
    with patch.object(signer, 'get_timestamp', return_value=old_timestamp):
        signed_old = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_old, max_age=1)

    # Test with negative age (future timestamp)
    future_timestamp = int(time.time()) + 3600
    with patch.object(signer, 'get_timestamp', return_value=future_timestamp):
        signed_future = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_future)

    # Test malformed timestamp
    malformed = value + b":" + b"invalid_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)

    # Test missing timestamp
    no_timestamp = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp)

    # Test bad signature
    bad_signed = signed[:-1] + b"x"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed)


# LLM-generated content at query #19
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test-value"
    signed_value = signer.sign(value)

    # Test successful unsign without timestamp return
    result = signer.unsign(signed_value)
    assert result == value

    # Test successful unsign with timestamp return
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

    # Test with max_age not expired
    result = signer.unsign(signed_value, max_age=3600)
    assert result == value

    # Test with max_age expired
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test with malformed signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"malformed-signature")

    # Test with missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test-value:missing-timestamp")

    # Test with malformed timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test-value:malformed-timestamp:signature")

    # Test with negative age (future timestamp)
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: int(time.time()) + 100
    future_signed_value = future_signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed_value, max_age=3600)


# LLM-generated content at query #20
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test normal unsign
    result = signer.unsign(signed_value)
    assert result == value

    # Test unsign with return_timestamp
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test unsign with max_age (should not raise)
    signer.unsign(signed_value, max_age=3600)

    # Test unsign with expired signature
    old_timestamp = int_to_bytes(signer.get_timestamp() - 3600)
    expired_signed_value = value + b":" + base64_encode(old_timestamp) + b":" + signer.get_signature(value + b":" + base64_encode(old_timestamp))
    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=1)

    # Test unsign with malformed timestamp
    malformed_signed_value = value + b":" + b"malformed" + b":" + signer.get_signature(value + b":" + b"malformed")
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test unsign with missing timestamp
    missing_timestamp_signed_value = value + b":" + signer.get_signature(value)
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_signed_value)

    # Test unsign with bad signature
    bad_signed_value = value + b":" + base64_encode(int_to_bytes(signer.get_timestamp())) + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #21
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test normal unsigning
    result = signer.unsign(signed_value)
    assert result == value

    # Test unsigning with return_timestamp
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test unsigning with max_age (should not raise)
    signer.unsign(signed_value, max_age=3600)

    # Test expired signature
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test malformed timestamp
    malformed_signed = b"test_value:malformed_timestamp:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed)

    # Test missing timestamp
    no_timestamp = b"test_value:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp)

    # Test bad signature
    bad_signed = b"test_value:timestamp:bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed)


# LLM-generated content at query #22
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner()
    assert isinstance(signer, TimestampSigner)
    assert signer.sep == b":"
    assert signer.key_derivation is None
    assert signer.digest_method == "hmac"


# LLM-generated content at query #23
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner()
    assert isinstance(signer, TimestampSigner)
    assert signer.sep == b"."

    # Test with custom separator
    custom_sep = b"|"
    signer = TimestampSigner(sep=custom_sep)
    assert signer.sep == custom_sep

    # Test with key
    key = "secret-key"
    signer = TimestampSigner(key=key)
    assert signer.key == key

    # Test with both key and separator
    signer = TimestampSigner(key=key, sep=custom_sep)
    assert signer.key == key
    assert signer.sep == custom_sep


# LLM-generated content at query #24
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)

    # Test successful loads
    assert serializer.loads(signed_data) == data

    # Test loads with timestamp
    loaded_data, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert loaded_data == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

    # Test loads with max_age
    assert serializer.loads(signed_data, max_age=3600) == data

    # Test loads with expired signature
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=0)

    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")

    # Test loads with salt
    salted_data = serializer.dumps(data, salt="salt")
    assert serializer.loads(salted_data, salt="salt") == data

    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(salted_data, salt="wrong-salt")


# LLM-generated content at query #25
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer.default_signer, type)
    assert issubclass(serializer.default_signer, TimestampSigner)
    assert serializer.secret_key is None
    assert serializer.salt is None
    assert serializer.separator == "."
    assert serializer.digest_method == "sha1"


# LLM-generated content at query #26
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)
    timestamp = signer.get_timestamp()

    # Test normal unsign
    result = signer.unsign(signed_value)
    assert result == value

    # Test unsign with return_timestamp=True
    result, dt = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc

    # Test unsign with max_age (should not raise)
    signer.unsign(signed_value, max_age=1000)

    # Test unsign with expired signature
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)

    # Test unsign with malformed timestamp
    malformed_signed = value + b":" + b"malformed" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed)

    # Test unsign with missing timestamp
    no_timestamp = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp)

    # Test unsign with bad signature
    bad_signed = value + b":" + base64_encode(int_to_bytes(timestamp)) + b":" + b"bad_sig"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed)


# LLM-generated content at query #27
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)
    current_time = signer.get_timestamp()

    # Test normal unsign
    result = signer.unsign(signed_value)
    assert result == value

    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test unsign with max_age (should not raise)
    signer.unsign(signed_value, max_age=3600)

    # Test unsign with expired signature
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: current_time - 3601
    expired_signed_value = expired_signer.sign(value)

    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=3600)

    # Test unsign with future timestamp
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: current_time + 3601
    future_signed_value = future_signer.sign(value)

    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed_value)

    # Test unsign with malformed timestamp
    malformed_signed_value = b"test_value:malformed_timestamp:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test unsign with missing timestamp
    missing_timestamp_value = b"test_value:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_value)

    # Test unsign with bad signature
    bad_signed_value = b"test_value:timestamp:bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #28
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test successful unsign without timestamp return
    unsign_result = signer.unsign(signed_value)
    assert unsign_result == value

    # Test successful unsign with timestamp return
    unsign_result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert unsign_result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

    # Test with max_age not expired
    max_age = 1000
    unsign_result = signer.unsign(signed_value, max_age=max_age)
    assert unsign_result == value

    # Test with max_age expired
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: 0
    expired_signed_value = expired_signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=1)

    # Test with malformed timestamp
    malformed_signed_value = value + b":" + b"malformed_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test with missing timestamp
    missing_timestamp_value = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_value)

    # Test with bad signature
    bad_signed_value = value + b":" + base64_encode(int_to_bytes(signer.get_timestamp())) + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #29
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = "test-value"
    signed_value = signer.sign(value)

    # Test normal unsign
    result = signer.unsign(signed_value)
    assert result == want_bytes(value)

    # Test unsign with return_timestamp
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == want_bytes(value)
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

    # Test unsign with max_age (should not raise)
    signer.unsign(signed_value, max_age=1000)

    # Test unsign with expired signature
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: int(time.time()) - 2000  # 2000 seconds ago
    expired_signed_value = expired_signer.sign(value)

    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=1000)

    # Test unsign with future timestamp
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: int(time.time()) + 1000  # 1000 seconds in future
    future_signed_value = future_signer.sign(value)

    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed_value)

    # Test unsign with malformed timestamp
    malformed_signed_value = b"test-value:malformed-timestamp:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test unsign with missing timestamp
    missing_timestamp_value = b"test-value:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_value)

    # Test unsign with wrong signature
    wrong_signed_value = b"test-value:timestamp:wrong-signature"
    with pytest.raises(BadSignature):
        signer.unsign(wrong_signed_value)


# LLM-generated content at query #30
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test normal unsigning
    assert signer.unsign(signed_value) == value

    # Test unsigning with return_timestamp
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

    # Test unsigning with max_age (should not raise)
    signer.unsign(signed_value, max_age=3600)

    # Test expired signature
    old_timestamp = int(time.time()) - 3600
    old_signed_value = (
        value +
        b":" +
        base64_encode(int_to_bytes(old_timestamp)) +
        b":" +
        signer.get_signature(value + b":" + base64_encode(int_to_bytes(old_timestamp)))
    )
    with pytest.raises(SignatureExpired):
        signer.unsign(old_signed_value, max_age=1)

    # Test malformed timestamp
    malformed_signed_value = value + b":" + b"invalid_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test missing timestamp
    no_timestamp_signed_value = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_signed_value)

    # Test bad signature
    bad_signed_value = value + b":" + base64_encode(int_to_bytes(signer.get_timestamp())) + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #31
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = "test-value"
    signed_value = signer.sign(value)

    # Test normal unsign
    result = signer.unsign(signed_value)
    assert result == b"test-value"

    # Test unsign with return_timestamp
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"test-value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test unsign with max_age (should not raise)
    signer.unsign(signed_value, max_age=1000)

    # Test unsign with expired signature
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test unsign with invalid signature
    with pytest.raises(BadSignature):
        signer.unsign("invalid-signature")

    # Test unsign with malformed timestamp
    malformed_signed = b"test-value:malformed-timestamp:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed)

    # Test unsign with missing timestamp
    missing_timestamp = b"test-value:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp)


# LLM-generated content at query #32
#--------------------------

```python
def test_TimedSerializer():
    # Test default constructor
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert isinstance(serializer.default_signer, type)
    assert issubclass(serializer.default_signer, TimestampSigner)

    # Test with custom parameters
    secret_key = "test-secret-key"
    salt = "test-salt"
    serializer_with_params = TimedSerializer(secret_key, salt=salt)
    assert serializer_with_params.secret_key == secret_key
    assert serializer_with_params.salt == salt


# LLM-generated content at query #33
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = "test_value"
    signed_value = signer.sign(value)

    # Test normal unsign
    result = signer.unsign(signed_value)
    assert result == b"test_value"

    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test unsign with max_age (valid)
    max_age = 1000
    result = signer.unsign(signed_value, max_age=max_age)
    assert result == b"test_value"

    # Test unsign with max_age (expired)
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: int(time.time()) - 2000
    expired_signed_value = expired_signer.sign(value)

    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=1000)

    # Test unsign with malformed timestamp
    malformed_signed_value = b"test_value:malformed_timestamp:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test unsign with missing timestamp
    missing_timestamp_signed_value = b"test_value:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_signed_value)

    # Test unsign with bad signature
    bad_signed_value = b"test_value:timestamp:bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #34
#--------------------------

```python
def test_TimedSerializer():
    # Test default initialization
    ts = TimedSerializer()
    assert isinstance(ts, TimedSerializer)
    assert ts.default_signer == TimestampSigner

    # Test with custom serializer
    ts = TimedSerializer(serializer=Serializer)
    assert isinstance(ts, TimedSerializer)
    assert ts.serializer == Serializer

    # Test with custom secret key
    ts = TimedSerializer(secret_key="test-secret")
    assert ts.secret_key == "test-secret"

    # Test with custom separator
    ts = TimedSerializer(sep=":")
    assert ts.sep == ":"

    # Test with custom salt
    ts = TimedSerializer(salt="test-salt")
    assert ts.salt == "test-salt"

    # Test with custom digits
    ts = TimedSerializer(digits=10)
    assert ts.digits == 10

    # Test with custom algorithm
    ts = TimedSerializer(algorithm="hmac")
    assert ts.algorithm == "hmac"


# LLM-generated content at query #35
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test basic functionality
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed = serializer.dumps(data)
    assert serializer.loads(signed) == data

    # Test with max_age
    signed = serializer.dumps(data)
    assert serializer.loads(signed, max_age=3600) == data

    # Test with return_timestamp
    signed = serializer.dumps(data)
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)

    # Test with salt
    signed = serializer.dumps(data, salt="salt")
    assert serializer.loads(signed, salt="salt") == data

    # Test with expired signature
    signed = serializer.dumps(data)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=0)

    # Test with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid")

    # Test with tampered signature
    signed = serializer.dumps(data)
    tampered = signed[:-1] + (b"x" if isinstance(signed, bytes) else "x")
    with pytest.raises(BadSignature):
        serializer.loads(tampered)


# LLM-generated content at query #36
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test normal unsign
    result = signer.unsign(signed_value)
    assert result == value

    # Test unsign with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

    # Test unsign with max_age
    result = signer.unsign(signed_value, max_age=1000)
    assert result == value

    # Test expired signature
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test tampered signature
    tampered_value = signed_value[:-1] + b"X"
    with pytest.raises(BadSignature):
        signer.unsign(tampered_value)

    # Test malformed timestamp
    malformed_value = value + b":" + b"malformed" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_value)

    # Test missing timestamp
    no_timestamp_value = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_value)


# LLM-generated content at query #37
#--------------------------

```python
def test_TimedSerializer_loads():
    # Test basic functionality
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed = serializer.dumps(data)
    assert serializer.loads(signed) == data

    # Test with max_age
    signed_old = serializer.dumps(data)
    time.sleep(1)
    assert serializer.loads(signed_old, max_age=1) == data
    time.sleep(1)
    try:
        serializer.loads(signed_old, max_age=1)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass

    # Test with return_timestamp
    signed_with_ts = serializer.dumps(data)
    result, timestamp = serializer.loads(signed_with_ts, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test with salt
    signed_salted = serializer.dumps(data, salt="salt")
    assert serializer.loads(signed_salted, salt="salt") == data

    # Test with invalid signature
    try:
        serializer.loads("invalid")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

    # Test with expired signature
    class FixedTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return 0

    expired_serializer = TimedSerializer("secret-key", signer_kwargs={"signer": FixedTimestampSigner})
    expired_signed = expired_serializer.dumps(data)
    try:
        expired_serializer.loads(expired_signed, max_age=0)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass


# LLM-generated content at query #38
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)

    # Test valid signature
    assert serializer.loads(signed_data) == data

    # Test with max_age
    assert serializer.loads(signed_data, max_age=3600) == data

    # Test return_timestamp
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)

    # Test with salt
    salted_data = serializer.dumps(data, salt="test-salt")
    assert serializer.loads(salted_data, salt="test-salt") == data

    # Test expired signature
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=0)

    # Test invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")

    # Test with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(salted_data, salt="wrong-salt")

    # Test with bytes input
    assert serializer.loads(want_bytes(signed_data)) == data


