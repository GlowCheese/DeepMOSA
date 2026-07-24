####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
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
    assert timestamp.tzinfo is not None

    # Test with max_age
    assert serializer.loads(signed_data, max_age=max_age) == data

    # Test with salt
    salted_data = serializer.dumps(data, salt="salt")
    assert serializer.loads(salted_data, salt="salt") == data

    # Test expired signature
    old_timestamp = int(time.time()) - max_age - 1
    old_signed_data = serializer.dumps(data)
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed_data, max_age=max_age)

    # Test bad signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid_signature")

    # Test with bytes input
    signed_bytes = want_bytes(serializer.dumps(data))
    assert serializer.loads(signed_bytes) == data


# LLM-generated content at query #2
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

    # Test with custom key
    key = b"secret-key"
    signer = TimestampSigner(key=key)
    assert signer.key == key

    # Test with both custom separator and key
    signer = TimestampSigner(sep=custom_sep, key=key)
    assert signer.sep == custom_sep
    assert signer.key == key


# LLM-generated content at query #3
#--------------------------

```python
def test_TimedSerializer():
    # Test default constructor
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer == TimestampSigner

    # Test with custom secret_key
    serializer = TimedSerializer(secret_key="custom-secret")
    assert serializer.secret_key == "custom-secret"

    # Test with custom serializer
    serializer = TimedSerializer(serializer="json")
    assert serializer.serializer == "json"

    # Test with custom separator
    serializer = TimedSerializer(sep=":")
    assert serializer.sep == ":"

    # Test with all parameters
    serializer = TimedSerializer(
        secret_key="custom-secret",
        serializer="json",
        sep=":",
        salt="custom-salt",
        digest_method="sha256",
        key_derivation="hmac",
        key_derivation_iterations=100000
    )
    assert serializer.secret_key == "custom-secret"
    assert serializer.serializer == "json"
    assert serializer.sep == ":"
    assert serializer.salt == "custom-salt"
    assert serializer.digest_method == "sha256"
    assert serializer.key_derivation == "hmac"
    assert serializer.key_derivation_iterations == 100000


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

    # Test with key and separator
    signer = TimestampSigner(key=key, sep=custom_sep)
    assert signer.key == key
    assert signer.sep == custom_sep


# LLM-generated content at query #5
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test normal unsign
    unsign_result = signer.unsign(signed_value)
    assert unsign_result == value

    # Test unsign with return_timestamp=True
    unsign_result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert unsign_result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test unsign with max_age
    unsign_result = signer.unsign(signed_value, max_age=1000)
    assert unsign_result == value

    # Test unsign with expired signature
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test unsign with malformed timestamp
    malformed_signed_value = b"test_value" + b":" + b"malformed_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test unsign with missing timestamp
    missing_timestamp_signed_value = b"test_value" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_signed_value)

    # Test unsign with bad signature
    bad_signed_value = b"test_value" + b":" + b"timestamp" + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #6
#--------------------------

```python
def test_TimedSerializer():
    # Test default construction
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer == TimestampSigner

    # Test with custom separator
    serializer = TimedSerializer(sep="|")
    assert serializer.sep == b"|"

    # Test with custom serializer
    serializer = TimedSerializer(serializer=Serializer)
    assert serializer.serializer == Serializer

    # Test with custom secret key
    serializer = TimedSerializer(secret_key="my-secret-key")
    assert serializer.secret_key == b"my-secret-key"

    # Test with custom digest method
    serializer = TimedSerializer(digest_method="sha256")
    assert serializer.digest_method == "sha256"

    # Test with multiple salts
    serializer = TimedSerializer(salts={"salt1": "value1", "salt2": "value2"})
    assert serializer.salts == {"salt1": "value1", "salt2": "value2"}


# LLM-generated content at query #7
#--------------------------

```python
def test_TimedSerializer():
    # Test default constructor
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer == TimestampSigner

    # Test with custom separator
    custom_sep = "|"
    serializer_custom = TimedSerializer(sep=custom_sep)
    assert serializer_custom.sep == custom_sep

    # Test with custom serializer
    custom_serializer = Serializer()
    serializer_with_custom = TimedSerializer(serializer=custom_serializer)
    assert serializer_with_custom.serializer == custom_serializer

    # Test with custom salt
    custom_salt = "custom_salt"
    serializer_salt = TimedSerializer(salt=custom_salt)
    assert serializer_salt.salt == custom_salt

    # Test with custom secret key
    custom_key = "secret_key"
    serializer_key = TimedSerializer(secret_key=custom_key)
    assert serializer_key.secret_key == custom_key

    # Test with custom digest method
    custom_digest = "sha256"
    serializer_digest = TimedSerializer(digest_method=custom_digest)
    assert serializer_digest.digest_method == custom_digest

    # Test with custom key derivation
    custom_key_deriv = "hmac"
    serializer_key_deriv = TimedSerializer(key_derivation=custom_key_deriv)
    assert serializer_key_deriv.key_derivation == custom_key_deriv


# LLM-generated content at query #8
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
    assert timestamp.tzinfo is not None

    # Test with salt
    signed_data_salt = serializer.dumps(data, salt="salt")
    assert serializer.loads(signed_data_salt, salt="salt") == data

    # Test expired signature
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=0)

    # Test invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")

    # Test missing timestamp
    with pytest.raises(BadTimeSignature):
        signer = TimestampSigner("secret-key")
        invalid_signed = signer.sign("data") + b"extra"
        serializer.loads(invalid_signed)


# LLM-generated content at query #9
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)

    # Test successful load
    loaded_data = serializer.loads(signed_data)
    assert loaded_data == data

    # Test with return_timestamp
    loaded_data, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert loaded_data == data
    assert isinstance(timestamp, datetime)

    # Test with max_age (should not raise)
    serializer.loads(signed_data, max_age=3600)

    # Test with expired signature
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: int(time.time()) - 3601
    expired_data = expired_signer.sign(serializer.dumps(data))
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_data, max_age=3600)

    # Test with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")

    # Test with wrong key
    wrong_serializer = TimedSerializer("wrong-key")
    with pytest.raises(BadSignature):
        wrong_serializer.loads(signed_data)


# LLM-generated content at query #10
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
    key = b"secret-key"
    signer = TimestampSigner(key=key)
    assert signer.key == key
    assert signer.sep == b"."

    # Test with key and custom separator
    signer = TimestampSigner(key=key, sep=custom_sep)
    assert signer.key == key
    assert signer.sep == custom_sep


# LLM-generated content at query #11
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
    assert timestamp.tzinfo is not None

    # Test with max_age
    signed_old = signer.sign(b"old value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_old, max_age=0)

    # Test with negative max_age (future signature)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)

    # Test with invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid:signature")

    # Test with malformed timestamp
    malformed = b"value" + b":" + base64_encode(b"not_a_timestamp") + b":" + b"sig"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed)

    # Test with missing timestamp
    no_timestamp = b"value" + b":" + b"sig"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp)

    # Test with string input
    signed_str = signed.decode('utf-8')
    assert signer.unsign(signed_str) == value


# LLM-generated content at query #12
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    secret_key = "secret"
    serializer = TimedSerializer(secret_key)
    data = {"key": "value"}
    max_age = 10

    # Test normal operation
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
    class MockTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return 0  # Old timestamp

    old_serializer = TimedSerializer(secret_key, signer_kwargs={"signer": MockTimestampSigner})
    signed_data = old_serializer.dumps(data)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=max_age)

    # Test with bad signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid_signature")

    # Test with missing timestamp
    with pytest.raises(BadTimeSignature):
        regular_signer = Signer(secret_key)
        signed_data = regular_signer.sign("data")
        serializer.loads(signed_data)


# LLM-generated content at query #13
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

    # Test loads with max_age (valid)
    assert serializer.loads(signed_data, max_age=3600) == data

    # Test loads with max_age (expired)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=0)

    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")

    # Test loads with salt
    salted_data = serializer.dumps(data, salt="custom-salt")
    assert serializer.loads(salted_data, salt="custom-salt") == data

    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(salted_data, salt="wrong-salt")

    # Test loads with bytes input
    assert serializer.loads(want_bytes(signed_data)) == data


# LLM-generated content at query #14
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    max_age = 10

    # Test successful load
    signed_data = serializer.dumps(data)
    assert serializer.loads(signed_data) == data

    # Test successful load with timestamp
    signed_data = serializer.dumps(data)
    loaded_data, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert loaded_data == data
    assert isinstance(timestamp, datetime)

    # Test successful load with max_age
    signed_data = serializer.dumps(data)
    assert serializer.loads(signed_data, max_age=max_age) == data

    # Test signature expired
    signed_data = serializer.dumps(data)
    time.sleep(2)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=1)

    # Test bad signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")

    # Test with salt
    signed_data = serializer.dumps(data, salt="custom-salt")
    assert serializer.loads(signed_data, salt="custom-salt") == data

    # Test with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_data, salt="wrong-salt")


# LLM-generated content at query #15
#--------------------------

```python
def test_TimedSerializer():
    # Test default construction
    ts = TimedSerializer()
    assert isinstance(ts, TimedSerializer)
    assert ts.default_signer == TimestampSigner

    # Test with custom separator
    ts = TimedSerializer(sep=":")
    assert ts.sep == ":"

    # Test with custom serializer
    ts = TimedSerializer(serializer="json")
    assert ts.serializer == "json"

    # Test with custom signer
    class CustomSigner(TimestampSigner):
        pass

    ts = TimedSerializer(signer=CustomSigner)
    assert ts.default_signer == CustomSigner

    # Test with custom salt
    ts = TimedSerializer(salt="custom_salt")
    assert ts.salt == "custom_salt"


# LLM-generated content at query #16
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
    unsign_result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert unsign_result == original_value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None  # Check timezone awareness

    # Test with max_age (should not raise if within max_age)
    signer.unsign(signed_value, max_age=3600)

    # Test with expired signature
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: 0  # Force old timestamp
    expired_signed_value = expired_signer.sign(original_value)

    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=1)

    # Test with malformed timestamp
    malformed_signed_value = original_value + b":" + b"malformed_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test with missing timestamp
    no_timestamp_value = original_value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_value)

    # Test with bad signature
    bad_signed_value = original_value + b":" + base64_encode(int_to_bytes(signer.get_timestamp())) + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #17
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer()
    data = {"key": "value"}
    secret_key = "secret"
    salt = "test_salt"
    max_age = 10

    # Test successful loads
    signed_data = serializer.dumps(data, salt=salt)
    assert serializer.loads(signed_data, salt=salt) == data

    # Test with return_timestamp
    result, timestamp = serializer.loads(signed_data, return_timestamp=True, salt=salt)
    assert result == data
    assert isinstance(timestamp, datetime)

    # Test with max_age (should not raise if within max_age)
    assert serializer.loads(signed_data, max_age=max_age, salt=salt) == data

    # Test with expired signature
    expired_signer = TimestampSigner(key=secret_key)
    expired_signer.get_timestamp = lambda: int(time.time()) - max_age - 1
    expired_data = expired_signer.sign(serializer.dumps(data))
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_data, max_age=max_age)

    # Test with invalid signature
    invalid_data = b"invalid_data"
    with pytest.raises(BadSignature):
        serializer.loads(invalid_data)

    # Test with missing salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_data, salt="wrong_salt")

    # Test with missing timestamp
    no_timestamp_data = b"data_without_timestamp"
    with pytest.raises(BadTimeSignature):
        serializer.loads(no_timestamp_data)


# LLM-generated content at query #18
#--------------------------

```python
def test_TimedSerializer():
    # Test default constructor
    ts = TimedSerializer()
    assert isinstance(ts, TimedSerializer)
    assert ts.default_signer == TimestampSigner

    # Test with custom serializer
    ts = TimedSerializer(serializer=Serializer())
    assert isinstance(ts, TimedSerializer)
    assert isinstance(ts.serializer, Serializer)

    # Test with custom signer
    ts = TimedSerializer(signer=TimestampSigner())
    assert isinstance(ts, TimedSerializer)
    assert isinstance(ts.signer, TimestampSigner)

    # Test with custom separator
    ts = TimedSerializer(sep=":")
    assert ts.sep == ":"

    # Test with custom salt
    ts = TimedSerializer(salt="test_salt")
    assert ts.salt == "test_salt"


# LLM-generated content at query #19
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
    signed = serializer.dumps(data)
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test with salt
    signed_salted = serializer.dumps(data, salt="salt")
    assert serializer.loads(signed_salted, salt="salt") == data

    # Test with bad signature
    try:
        serializer.loads("bad-signature")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

    # Test with expired signature
    class FixedTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return 0

    expired_serializer = TimedSerializer("secret-key", signer_kwargs={"signer": FixedTimestampSigner})
    signed_expired = expired_serializer.dumps(data)
    try:
        expired_serializer.loads(signed_expired, max_age=0)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass

    # Test with malformed timestamp
    class BadTimestampSigner(TimestampSigner):
        def sign(self, value):
            value = want_bytes(value)
            sep = want_bytes(self.sep)
            return value + sep + b"bad-timestamp" + sep + self.get_signature(value)

    bad_ts_serializer = TimedSerializer("secret-key", signer_kwargs={"signer": BadTimestampSigner})
    signed_bad_ts = bad_ts_serializer.dumps(data)
    try:
        bad_ts_serializer.loads(signed_bad_ts)
        assert False, "Should have raised BadTimeSignature"
    except BadTimeSignature:
        pass


# LLM-generated content at query #20
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
        serializer.loads(signed_data, salt="wrong-salt")


# LLM-generated content at query #21
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    max_age = 10

    # Test normal operation
    signed = serializer.dumps(data)
    assert serializer.loads(signed) == data

    # Test with max_age
    signed_with_age = serializer.dumps(data)
    assert serializer.loads(signed_with_age, max_age=max_age) == data

    # Test return_timestamp
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)

    # Test with salt
    salt = "custom-salt"
    signed_with_salt = serializer.dumps(data, salt=salt)
    assert serializer.loads(signed_with_salt, salt=salt) == data

    # Test expired signature
    old_timestamp = int(time.time()) - 20
    old_signed = serializer.dumps(data)
    with pytest.raises(SignatureExpired):
        serializer.loads(old_signed, max_age=10)

    # Test bad signature
    bad_signed = b"bad-signature"
    with pytest.raises(BadSignature):
        serializer.loads(bad_signed)

    # Test malformed timestamp
    malformed_signed = b"data" + b":" + b"malformed" + b":" + b"sig"
    with pytest.raises(BadTimeSignature):
        serializer.loads(malformed_signed)


# LLM-generated content at query #22
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
    loaded_data, timestamp = serializer.loads(signed, return_timestamp=True)
    assert loaded_data == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test with salt
    serializer_salt = TimedSerializer("secret-key", salt="test-salt")
    signed_salt = serializer_salt.dumps(data)
    assert serializer_salt.loads(signed_salt) == data
    try:
        serializer.loads(signed_salt)
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

    # Test with bad signature
    try:
        serializer.loads("bad-signature")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

    # Test with expired signature
    class FixedTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return 0

    serializer_fixed = TimedSerializer("secret-key", signer_kwargs={"signer": FixedTimestampSigner})
    signed_fixed = serializer_fixed.dumps(data)
    try:
        serializer_fixed.loads(signed_fixed, max_age=0)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass


# LLM-generated content at query #23
#--------------------------

```python
def test_TimedSerializer():
    # Test default construction
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer == TimestampSigner

    # Test with custom separator
    serializer = TimedSerializer(separator="|")
    assert serializer.sep == "|"

    # Test with custom serializer
    serializer = TimedSerializer(serializer=Serializer)
    assert serializer.serializer == Serializer

    # Test with custom salt
    serializer = TimedSerializer(salt="custom_salt")
    assert serializer.salt == "custom_salt"

    # Test with custom secret key
    serializer = TimedSerializer(secret_key="secret")
    assert serializer.secret_key == "secret"

    # Test with custom digest method
    serializer = TimedSerializer(digest_method="sha256")
    assert serializer.digest_method == "sha256"


# LLM-generated content at query #24
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    max_age = 10

    # Test normal operation
    signed_data = serializer.dumps(data)
    result = serializer.loads(signed_data)
    assert result == data

    # Test with max_age
    signed_data = serializer.dumps(data)
    result = serializer.loads(signed_data, max_age=max_age)
    assert result == data

    # Test with return_timestamp
    signed_data = serializer.dumps(data)
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)

    # Test with salt
    signed_data = serializer.dumps(data, salt="test-salt")
    result = serializer.loads(signed_data, salt="test-salt")
    assert result == data

    # Test with expired signature
    signed_data = serializer.dumps(data)
    time.sleep(2)  # Ensure some time has passed
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=1)

    # Test with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")

    # Test with missing timestamp
    with pytest.raises(BadTimeSignature):
        signer = TimestampSigner("secret-key")
        invalid_signed_data = signer.sign("data") + b".invalid"
        serializer.loads(invalid_signed_data)


# LLM-generated content at query #25
#--------------------------

```python
def test_TimedSerializer():
    # Test default constructor
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer == TimestampSigner

    # Test with custom secret_key
    serializer = TimedSerializer(secret_key="test-secret-key")
    assert serializer.secret_key == "test-secret-key"

    # Test with custom separator
    serializer = TimedSerializer(separator="|")
    assert serializer.sep == "|"

    # Test with custom serializer
    serializer = TimedSerializer(serializer="json")
    assert serializer.serializer == "json"

    # Test with custom signer_kwargs
    serializer = TimedSerializer(signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}


# LLM-generated content at query #26
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    test_data = {"key": "value"}
    max_age = 10

    # Test normal operation
    signed_data = serializer.dumps(test_data)
    assert serializer.loads(signed_data) == test_data

    # Test with return_timestamp
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == test_data
    assert isinstance(timestamp, datetime)

    # Test with max_age (should not raise)
    serializer.loads(signed_data, max_age=max_age)

    # Test with expired signature
    expired_signer = TimedSerializer("secret-key")
    expired_signer.get_timestamp = lambda: 0  # Set timestamp to 0
    expired_data = expired_signer.dumps(test_data)
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_data, max_age=max_age)

    # Test with bad signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid_signature")

    # Test with salt
    salted_data = serializer.dumps(test_data, salt="test-salt")
    assert serializer.loads(salted_data, salt="test-salt") == test_data
    with pytest.raises(BadSignature):
        serializer.loads(salted_data, salt="wrong-salt")


# LLM-generated content at query #27
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = "test_value"
    signed_value = signer.sign(value)

    # Test successful unsign without timestamp
    unsign_result = signer.unsign(signed_value)
    assert unsign_result == b"test_value"

    # Test successful unsign with timestamp
    unsign_result_with_ts = signer.unsign(signed_value, return_timestamp=True)
    assert isinstance(unsign_result_with_ts, tuple)
    assert unsign_result_with_ts[0] == b"test_value"
    assert isinstance(unsign_result_with_ts[1], datetime)

    # Test with max_age within limit
    max_age = 1000
    unsign_result = signer.unsign(signed_value, max_age=max_age)
    assert unsign_result == b"test_value"

    # Test with max_age exceeded (using a very small max_age)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test with malformed signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"malformed_signature")

    # Test with missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value" + b":" + b"signature")

    # Test with malformed timestamp
    malformed_ts_value = b"value" + b":" + b"malformed_ts" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_ts_value)

    # Test with negative age (future timestamp)
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: int(time.time()) + 100
    future_signed_value = future_signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed_value, max_age=50)


# LLM-generated content at query #28
#--------------------------

```python
def test_TimedSerializer_loads():
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}

    # Test normal signing and unsigning
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
    signed = serializer.dumps(data, salt="custom-salt")
    assert serializer.loads(signed, salt="custom-salt") == data

    # Test with expired signature
    signed = serializer.dumps(data)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=0)

    # Test with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")

    # Test with missing timestamp
    with pytest.raises(BadTimeSignature):
        serializer.loads(b"value:missing-timestamp")


# LLM-generated content at query #29
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
        serializer.loads("invalid-signature")

    # Test with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_data, salt="wrong-salt")


# LLM-generated content at query #30
#--------------------------

```python
def test_TimedSerializer():
    # Test default initialization
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer == TimestampSigner

    # Test with custom serializer parameters
    serializer = TimedSerializer(secret_key="test_key", salt="test_salt", serializer="json")
    assert serializer.secret_key == "test_key"
    assert serializer.salt == "test_salt"
    assert serializer.serializer == "json"

    # Test with custom signer class
    class CustomTimestampSigner(TimestampSigner):
        pass

    serializer = TimedSerializer(default_signer=CustomTimestampSigner)
    assert serializer.default_signer == CustomTimestampSigner

    # Test iter_unsigners returns TimestampSigner instances
    serializer = TimedSerializer()
    for signer in serializer.iter_unsigners():
        assert isinstance(signer, TimestampSigner)


# LLM-generated content at query #31
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
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
    signed_data = serializer.dumps(data, salt="salt")
    assert serializer.loads(signed_data, salt="salt") == data

    # Test expired signature
    class MockTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) - 20  # 20 seconds ago

    expired_serializer = TimedSerializer("secret-key", signer_kwargs={"signer": MockTimestampSigner})
    signed_data = expired_serializer.dumps(data)
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=10)

    # Test invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid_signature")

    # Test with max_age and expired signature
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=5)

    # Test with return_timestamp and expired signature
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, return_timestamp=True, max_age=5)


# LLM-generated content at query #32
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner("secret-key")
    assert signer.sep == b"."
    assert signer.key == "secret-key"

    signer_with_sep = TimestampSigner("secret-key", sep=":")
    assert signer_with_sep.sep == b":"
    assert signer_with_sep.key == "secret-key"

    signer_with_salt = TimestampSigner("secret-key", salt="test-salt")
    assert signer_with_salt.salt == "test-salt"
    assert signer_with_salt.key == "secret-key"


# LLM-generated content at query #33
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    serialized = serializer.dumps(data)

    # Test successful deserialization
    deserialized = serializer.loads(serialized)
    assert deserialized == data

    # Test with return_timestamp
    deserialized, timestamp = serializer.loads(serialized, return_timestamp=True)
    assert deserialized == data
    assert isinstance(timestamp, datetime)

    # Test with max_age (should not raise)
    deserialized = serializer.loads(serialized, max_age=3600)
    assert deserialized == data

    # Test with expired signature
    expired_serialized = serializer.dumps(data)
    time.sleep(2)  # Ensure some time passes
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_serialized, max_age=1)

    # Test with invalid signature
    invalid_serialized = b"invalid-signature"
    with pytest.raises(BadSignature):
        serializer.loads(invalid_serialized)

    # Test with salt
    salted_serialized = serializer.dumps(data, salt="salt")
    deserialized = serializer.loads(salted_serialized, salt="salt")
    assert deserialized == data

    # Test with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(salted_serialized, salt="wrong-salt")


# LLM-generated content at query #34
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
    signed = serializer.dumps(data)
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

    # Test with salt
    signed_salt = serializer.dumps(data, salt="salt")
    assert serializer.loads(signed_salt, salt="salt") == data
    try:
        serializer.loads(signed_salt, salt="wrong-salt")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

    # Test with invalid signature
    try:
        serializer.loads("invalid-signature")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

    # Test with expired signature
    class FixedTimestampSigner(TimestampSigner):
        def get_timestamp(self):
            return 0

    serializer_expired = TimedSerializer("secret-key", signer_kwargs={"signer": FixedTimestampSigner})
    signed_expired = serializer_expired.dumps(data)
    try:
        serializer_expired.loads(signed_expired, max_age=0)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass


# LLM-generated content at query #35
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

    # Test with key and salt
    key = "secret-key"
    salt = "test-salt"
    signer = TimestampSigner(key=key, salt=salt)
    assert signer.key == key
    assert signer.salt == salt

    # Test get_timestamp returns an integer
    assert isinstance(signer.get_timestamp(), int)

    # Test timestamp_to_datetime returns aware datetime
    ts = signer.get_timestamp()
    dt = signer.timestamp_to_datetime(ts)
    assert isinstance(dt, datetime)
    assert dt.tzinfo is not None
    assert dt.tzinfo == timezone.utc


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
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
    expired_signer.get_timestamp = lambda: current_time + 3601
    with pytest.raises(SignatureExpired):
        expired_signer.unsign(signed_value, max_age=3600)

    # Test unsign with future timestamp
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: current_time - 3601
    with pytest.raises(SignatureExpired):
        future_signer.unsign(signed_value, max_age=3600)

    # Test unsign with malformed timestamp
    malformed_signed = value + b":" + b"malformed" + b":" + signer.get_signature(value + b":" + b"malformed")
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed)

    # Test unsign with missing timestamp
    missing_timestamp = value + b":" + signer.get_signature(value)
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp)

    # Test unsign with bad signature
    bad_signed = value + b":" + base64_encode(int_to_bytes(current_time)) + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed)


# LLM-generated content at query #2
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    secret_key = "secret"
    serializer = TimedSerializer(secret_key)

    # Test normal serialization and deserialization
    data = {"key": "value"}
    signed = serializer.dumps(data)
    assert serializer.loads(signed) == data

    # Test with max_age
    signed_old = serializer.dumps(data)
    time.sleep(1)  # Ensure some time passes
    assert serializer.loads(signed_old, max_age=2) == data
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_old, max_age=0)

    # Test return_timestamp
    signed_with_ts = serializer.dumps(data)
    result, timestamp = serializer.loads(signed_with_ts, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test with salt
    salt = "custom-salt"
    signed_salted = serializer.dumps(data, salt=salt)
    assert serializer.loads(signed_salted, salt=salt) == data
    with pytest.raises(BadSignature):
        serializer.loads(signed_salted, salt="wrong-salt")

    # Test invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")

    # Test expired signature
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_old, max_age=0)


# LLM-generated content at query #3
#--------------------------

```python
def test_TimedSerializer():
    # Test default initialization
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer == TimestampSigner

    # Test with custom secret_key
    secret_key = "my-secret-key"
    serializer = TimedSerializer(secret_key=secret_key)
    assert serializer.secret_key == secret_key

    # Test with custom separator
    separator = "|"
    serializer = TimedSerializer(sep=separator)
    assert serializer.sep == separator

    # Test with custom serializer
    custom_serializer = "json"
    serializer = TimedSerializer(serializer=custom_serializer)
    assert serializer.serializer == custom_serializer

    # Test with custom digits
    digits = "abcdefghijklmnopqrstuvwxyz"
    serializer = TimedSerializer(digits=digits)
    assert serializer.digits == digits

    # Test with all parameters
    serializer = TimedSerializer(
        secret_key=secret_key,
        sep=separator,
        serializer=custom_serializer,
        digits=digits
    )
    assert serializer.secret_key == secret_key
    assert serializer.sep == separator
    assert serializer.serializer == custom_serializer
    assert serializer.digits == digits


# LLM-generated content at query #4
#--------------------------

```python
def test_TimedSerializer():
    # Test default construction
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer == TimestampSigner
    assert isinstance(serializer.iter_unsigners().next(), TimestampSigner)

    # Test with custom secret key
    serializer = TimedSerializer(secret_key="custom-secret")
    assert serializer.secret_key == "custom-secret"

    # Test with custom separator
    serializer = TimedSerializer(sep="|")
    assert serializer.sep == "|"

    # Test with custom serializer
    serializer = TimedSerializer(serializer="json")
    assert serializer.serializer == "json"

    # Test with custom digest method
    serializer = TimedSerializer(digest_method="sha256")
    assert serializer.digest_method == "sha256"


# LLM-generated content at query #5
#--------------------------

```python
def test_TimestampSigner():
    # Test default initialization
    signer = TimestampSigner()
    assert signer.sep == b"."

    # Test with custom separator
    custom_sep = b"|"
    signer = TimestampSigner(sep=custom_sep)
    assert signer.sep == custom_sep

    # Test with custom key
    custom_key = b"secret-key"
    signer = TimestampSigner(key=custom_key)
    assert signer.key == custom_key

    # Test with custom separator and key
    signer = TimestampSigner(sep=custom_sep, key=custom_key)
    assert signer.sep == custom_sep
    assert signer.key == custom_key


# LLM-generated content at query #6
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
    assert serializer.loads(signed_data, max_age=3600) == data

    # Test loads with expired signature
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=0)

    # Test loads with invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")

    # Test loads with salt
    salted_serializer = TimedSerializer("secret-key", salt="test-salt")
    salted_data = salted_serializer.dumps(data)
    assert salted_serializer.loads(salted_data) == data

    # Test loads with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(salted_data)


# LLM-generated content at query #7
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
    assert timestamp.tzinfo is not None

    # Test with max_age within valid range
    result = signer.unsign(signed_value, max_age=1000)
    assert result == value

    # Test with max_age expired (should raise SignatureExpired)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test with malformed signature (should raise BadTimeSignature)
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"malformed-signature")

    # Test with missing timestamp (should raise BadTimeSignature)
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value" + signer.sep.encode() + b"signature")

    # Test with negative max_age (should raise SignatureExpired)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)


# LLM-generated content at query #8
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    max_age = 10

    # Test successful deserialization
    signed_data = serializer.dumps(data)
    result = serializer.loads(signed_data)
    assert result == data

    # Test with return_timestamp
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test with max_age (should not raise)
    result = serializer.loads(signed_data, max_age=max_age)
    assert result == data

    # Test expired signature
    old_timestamp = int(time.time()) - max_age - 1
    old_signed_data = serializer.dumps(data)  # This will have current timestamp
    # Manually create expired data by modifying timestamp
    parts = old_signed_data.split(b".")
    if len(parts) == 3:
        base64_data, timestamp_b64, signature = parts
        old_timestamp_b64 = base64_encode(int_to_bytes(old_timestamp))
        expired_data = base64_data + b"." + old_timestamp_b64 + b"." + signature
        with pytest.raises(SignatureExpired):
            serializer.loads(expired_data, max_age=max_age)

    # Test invalid signature
    invalid_data = signed_data + b"invalid"
    with pytest.raises(BadSignature):
        serializer.loads(invalid_data)

    # Test with salt
    salt = "test-salt"
    salted_data = serializer.dumps(data, salt=salt)
    result = serializer.loads(salted_data, salt=salt)
    assert result == data

    # Test without salt when salt was used
    with pytest.raises(BadSignature):
        serializer.loads(salted_data)

    # Test with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(salted_data, salt="wrong-salt")


# LLM-generated content at query #9
#--------------------------

```python
def test_TimedSerializer():
    # Test default construction
    ts = TimedSerializer()
    assert isinstance(ts, TimedSerializer)
    assert ts.default_signer == TimestampSigner

    # Test with custom serializer
    ts = TimedSerializer(serializer=Serializer)
    assert isinstance(ts, TimedSerializer)
    assert ts.default_signer == TimestampSigner

    # Test with custom secret key
    ts = TimedSerializer(secret_key="test-key")
    assert isinstance(ts, TimedSerializer)
    assert ts.secret_key == "test-key"

    # Test with custom separator
    ts = TimedSerializer(sep="|")
    assert isinstance(ts, TimedSerializer)
    assert ts.sep == "|"

    # Test with custom salt
    ts = TimedSerializer(salt="test-salt")
    assert isinstance(ts, TimedSerializer)
    assert ts.salt == "test-salt"

    # Test with custom digits
    ts = TimedSerializer(digits=10)
    assert isinstance(ts, TimedSerializer)
    assert ts.digits == 10

    # Test with custom algorithm
    ts = TimedSerializer(algorithm_name="hmac")
    assert isinstance(ts, TimedSerializer)
    assert ts.algorithm_name == "hmac"


# LLM-generated content at query #10
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

    # Test unsigning with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test unsigning with max_age (should not raise)
    max_age = 1000
    result = signer.unsign(signed_value, max_age=max_age)
    assert result == value

    # Test signature expiration
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


# LLM-generated content at query #11
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

    # Test with key and salt
    key = "secret-key"
    salt = "test-salt"
    signer = TimestampSigner(key=key, salt=salt)
    assert signer.key == key
    assert signer.salt == salt
    assert signer.sep == b"."

    # Test get_timestamp returns an integer
    assert isinstance(signer.get_timestamp(), int)

    # Test timestamp_to_datetime returns a timezone-aware datetime
    timestamp = signer.get_timestamp()
    dt = signer.timestamp_to_datetime(timestamp)
    assert dt.tzinfo is not None
    assert dt.tzinfo == timezone.utc


# LLM-generated content at query #12
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
    key = b"secret-key"
    signer = TimestampSigner(key=key)
    assert signer.key == key

    # Test with both custom separator and key
    signer = TimestampSigner(sep=custom_sep, key=key)
    assert signer.sep == custom_sep
    assert signer.key == key


# LLM-generated content at query #13
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

    # Test unsigning with return_timestamp=True
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test unsigning with max_age (should not raise)
    signer.unsign(signed_value, max_age=1000)

    # Test expired signature
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid_signature")

    # Test malformed timestamp
    malformed_signed = value + b":" + b"invalid_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed)

    # Test missing timestamp
    no_timestamp = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp)


# LLM-generated content at query #14
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)
    max_age = 10

    # Test normal unsigning without timestamp return
    result = signer.unsign(signed_value)
    assert result == value

    # Test normal unsigning with timestamp return
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

    # Test unsigning with max_age within limit
    result = signer.unsign(signed_value, max_age=max_age)
    assert result == value

    # Test unsigning with expired signature
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: int(time.time()) - max_age - 1
    expired_signed_value = expired_signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=max_age)

    # Test unsigning with malformed timestamp
    malformed_signed_value = value + b":" + b"malformed_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test unsigning with missing timestamp
    no_timestamp_signed_value = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_signed_value)

    # Test unsigning with future timestamp
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: int(time.time()) + max_age + 1
    future_signed_value = future_signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed_value, max_age=max_age)


# LLM-generated content at query #15
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

    # Test with secret key
    secret_key = "my-secret-key"
    signer = TimestampSigner(secret_key)
    assert signer.secret_key == secret_key

    # Test with both secret key and custom separator
    signer = TimestampSigner(secret_key, sep=custom_sep)
    assert signer.secret_key == secret_key
    assert signer.sep == custom_sep


# LLM-generated content at query #16
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

    # Test with key and custom separator
    signer = TimestampSigner(key=key, sep=custom_sep)
    assert signer.key == key
    assert signer.sep == custom_sep


# LLM-generated content at query #17
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
    assert timestamp.tzinfo is timezone.utc

    # Test with max_age not expired
    result = signer.unsign(signed_value, max_age=3600)
    assert result == value

    # Test with max_age expired
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)

    # Test with malformed signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"malformed_signature")

    # Test with missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test_value" + signer.sep + b"invalid_base64")

    # Test with negative max_age (future timestamp)
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: int(time.time()) + 100
    future_signed = future_signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed, max_age=50)


# LLM-generated content at query #18
#--------------------------

```python
def test_TimestampSigner():
    # Test default construction
    signer = TimestampSigner()
    assert isinstance(signer, TimestampSigner)
    assert signer.sep == b"."

    # Test with custom separator
    custom_sep = b"|"
    signer = TimestampSigner(sep=custom_sep)
    assert signer.sep == custom_sep

    # Test with key and salt
    key = "secret-key"
    salt = "test-salt"
    signer = TimestampSigner(key=key, salt=salt)
    assert signer.key == key
    assert signer.salt == salt
    assert signer.sep == b"."

    # Test with all parameters
    signer = TimestampSigner(key=key, salt=salt, sep=custom_sep)
    assert signer.key == key
    assert signer.salt == salt
    assert signer.sep == custom_sep


# LLM-generated content at query #19
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

    # Test with max_age expired
    expired_signed_value = signer.sign(value)
    # Mock get_timestamp to return a future time
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: original_get_timestamp() + 3601
    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=3600)
    signer.get_timestamp = original_get_timestamp

    # Test with malformed timestamp
    malformed_signed_value = b"test_value:malformed_timestamp:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test with missing timestamp
    no_timestamp_signed_value = b"test_value:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_signed_value)

    # Test with bad signature
    bad_signed_value = b"test_value:timestamp:bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #20
#--------------------------

```python
def test_TimestampSigner():
    signer = TimestampSigner()
    assert isinstance(signer, TimestampSigner)
    assert signer.sep == b"."
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == "sha1"


# LLM-generated content at query #21
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
    assert timestamp.tzinfo is not None  # Ensure it's timezone-aware

    # Test with max_age (should not raise if within max_age)
    signer.unsign(signed_value, max_age=3600)

    # Test with expired signature
    expired_signed_value = signer.sign(value)
    # Mock get_timestamp to simulate expired signature
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: original_get_timestamp() + 3601
    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=3600)
    signer.get_timestamp = original_get_timestamp

    # Test with malformed timestamp
    malformed_signed_value = b"test_value:malformed_timestamp:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test with missing timestamp
    missing_timestamp_value = b"test_value:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_value)

    # Test with bad signature
    bad_signed_value = b"test_value:timestamp:bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #22
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)

    # Test valid signature
    assert serializer.loads(signed_data) == data

    # Test with return_timestamp
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)

    # Test with max_age (should not raise)
    serializer.loads(signed_data, max_age=3600)

    # Test expired signature
    expired_data = serializer.dumps(data)
    time.sleep(1)  # Ensure some time passes
    with pytest.raises(SignatureExpired):
        serializer.loads(expired_data, max_age=0)

    # Test invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid-signature")

    # Test with salt
    salted_data = serializer.dumps(data, salt="custom-salt")
    assert serializer.loads(salted_data, salt="custom-salt") == data
    with pytest.raises(BadSignature):
        serializer.loads(salted_data, salt="wrong-salt")


# LLM-generated content at query #23
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test successful unsigning without timestamp
    unsign_result = signer.unsign(signed_value)
    assert unsign_result == value

    # Test successful unsigning with timestamp
    unsign_result_with_ts = signer.unsign(signed_value, return_timestamp=True)
    assert unsign_result_with_ts[0] == value
    assert isinstance(unsign_result_with_ts[1], datetime)

    # Test with max_age not expired
    unsign_result_max_age = signer.unsign(signed_value, max_age=1000)
    assert unsign_result_max_age == value

    # Test with max_age expired (using a very small max_age)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test with malformed signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"malformed_signature")

    # Test with missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value" + signer.sep.encode())

    # Test with malformed timestamp
    malformed_ts_value = value + signer.sep.encode() + b"malformed_ts" + signer.sep.encode() + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_ts_value)

    # Test with future timestamp (negative age)
    future_ts = signer.get_timestamp() + 100
    future_ts_bytes = base64_encode(int_to_bytes(future_ts))
    future_signed_value = value + signer.sep.encode() + future_ts_bytes + signer.sep.encode() + signer.get_signature(value + signer.sep.encode() + future_ts_bytes)
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed_value)


# LLM-generated content at query #24
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

    # Test with key and separator
    signer = TimestampSigner(key=key, sep=custom_sep)
    assert signer.key == key
    assert signer.sep == custom_sep


# LLM-generated content at query #25
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test successful unsign without timestamp
    result = signer.unsign(signed_value)
    assert result == value

    # Test successful unsign with timestamp
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test with max_age not expired
    result = signer.unsign(signed_value, max_age=3600)
    assert result == value

    # Test with max_age expired
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test with malformed signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"malformed_signature")

    # Test with missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test_value:missing_timestamp")

    # Test with negative max_age (future timestamp)
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: int(time.time()) + 100
    future_signed_value = future_signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed_value, max_age=50)


# LLM-generated content at query #26
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
    key = b"secret-key"
    signer = TimestampSigner(key=key)
    assert signer.key == key

    # Test with custom separator and key
    signer = TimestampSigner(sep=custom_sep, key=key)
    assert signer.sep == custom_sep
    assert signer.key == key

    # Test get_timestamp returns an integer
    assert isinstance(signer.get_timestamp(), int)

    # Test timestamp_to_datetime returns a timezone-aware datetime
    ts = signer.get_timestamp()
    dt = signer.timestamp_to_datetime(ts)
    assert dt.tzinfo is not None
    assert dt.tzinfo == timezone.utc


# LLM-generated content at query #27
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

    # Test with max_age not expired
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
    missing_timestamp_value = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_value)

    # Test with bad signature
    bad_signed_value = value + b":" + base64_encode(int_to_bytes(signer.get_timestamp())) + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #28
#--------------------------

```python
def test_TimedSerializer_loads():
    # Setup
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    max_age = 10

    # Test valid signature with max_age
    signed_data = serializer.dumps(data)
    result = serializer.loads(signed_data, max_age=max_age)
    assert result == data

    # Test valid signature with return_timestamp
    signed_data = serializer.dumps(data)
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test expired signature
    signed_data = serializer.dumps(data)
    time.sleep(2)  # Ensure some time passes
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=0)

    # Test invalid signature
    with pytest.raises(BadSignature):
        serializer.loads("invalid_signature", max_age=max_age)

    # Test with salt
    signed_data = serializer.dumps(data, salt="salt")
    result = serializer.loads(signed_data, salt="salt", max_age=max_age)
    assert result == data

    # Test with wrong salt
    with pytest.raises(BadSignature):
        serializer.loads(signed_data, salt="wrong_salt", max_age=max_age)


# LLM-generated content at query #29
#--------------------------

```python
def test_TimestampSigner():
    # Test default constructor
    signer = TimestampSigner()
    assert signer.sep == b"."

    # Test with custom separator
    custom_sep = b"|"
    signer = TimestampSigner(sep=custom_sep)
    assert signer.sep == custom_sep

    # Test with custom key
    key = b"secret-key"
    signer = TimestampSigner(key=key)
    assert signer.key == key

    # Test with both custom separator and key
    signer = TimestampSigner(sep=custom_sep, key=key)
    assert signer.sep == custom_sep
    assert signer.key == key


# LLM-generated content at query #30
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test successful unsign without timestamp
    result = signer.unsign(signed_value)
    assert result == value

    # Test successful unsign with timestamp
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test with max_age not expired
    result = signer.unsign(signed_value, max_age=3600)
    assert result == value

    # Test with max_age expired
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test with invalid signature
    invalid_signed_value = signed_value + b"invalid"
    with pytest.raises(BadSignature):
        signer.unsign(invalid_signed_value)

    # Test with malformed timestamp
    malformed_signed_value = value + b"::" + b"malformed_timestamp"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test with missing timestamp
    missing_timestamp_value = value + b"::"
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_value)


# LLM-generated content at query #31
#--------------------------

```python
def test_TimedSerializer():
    # Test default constructor
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer == TimestampSigner

    # Test with custom separator
    serializer = TimedSerializer(separator="|")
    assert serializer.sep == b"|"

    # Test with custom serializer
    serializer = TimedSerializer(serializer=Serializer)
    assert serializer.serializer == Serializer

    # Test with custom digest method
    serializer = TimedSerializer(digest_method="sha256")
    assert serializer.digest_method == "sha256"

    # Test with custom key deriver
    serializer = TimedSerializer(key_derivation="hmac")
    assert serializer.key_derivation == "hmac"

    # Test with secret key
    serializer = TimedSerializer(secret_key="secret")
    assert serializer.secret_key == b"secret"

    # Test with salt
    serializer = TimedSerializer(salt="salt")
    assert serializer.salt == b"salt"


# LLM-generated content at query #32
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)
    current_time = signer.get_timestamp()

    # Test normal unsigning without timestamp return
    unsign_result = signer.unsign(signed_value)
    assert unsign_result == value

    # Test unsigning with timestamp return
    unsign_result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert unsign_result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

    # Test with max_age (should not raise)
    signer.unsign(signed_value, max_age=3600)

    # Test with expired signature (should raise SignatureExpired)
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: current_time - 3601  # 1 hour and 1 second ago
    expired_signed_value = expired_signer.sign(value)

    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=3600)

    # Test with future timestamp (should raise SignatureExpired)
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: current_time + 3601  # 1 hour and 1 second in the future
    future_signed_value = future_signer.sign(value)

    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed_value, max_age=3600)

    # Test with malformed timestamp (should raise BadTimeSignature)
    malformed_signed_value = value + b":" + b"malformed_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test with missing timestamp (should raise BadTimeSignature)
    no_timestamp_signed_value = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_signed_value)

    # Test with bad signature (should raise BadTimeSignature)
    bad_signed_value = value + b":" + base64_encode(int_to_bytes(current_time)) + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #33
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

    # Test unsign with return_timestamp
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

    # Test unsign with max_age (valid)
    max_age = 1000
    result = signer.unsign(signed_value, max_age=max_age)
    assert result == value

    # Test unsign with max_age (expired)
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: current_time + max_age + 1
    with pytest.raises(SignatureExpired):
        expired_signer.unsign(signed_value, max_age=max_age)

    # Test unsign with negative max_age (future timestamp)
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: current_time - 100
    with pytest.raises(SignatureExpired):
        future_signer.unsign(signed_value, max_age=max_age)

    # Test unsign with malformed timestamp
    malformed_signed_value = value + b":" + b"malformed_timestamp" + b":" + signer.get_signature(value + b":" + b"malformed_timestamp")
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test unsign with missing timestamp
    missing_timestamp_signed_value = value + b":" + signer.get_signature(value)
    with pytest.raises(BadTimeSignature):
        signer.unsign(missing_timestamp_signed_value)

    # Test unsign with bad signature
    bad_signed_value = value + b":" + base64_encode(int_to_bytes(current_time)) + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #34
#--------------------------

```python
def test_TimedSerializer():
    serializer = TimedSerializer()
    assert isinstance(serializer, TimedSerializer)
    assert serializer.default_signer == TimestampSigner


# LLM-generated content at query #35
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

    # Test with max_age (should not raise if within max_age)
    max_age = 1000
    result = signer.unsign(signed_value, max_age=max_age)
    assert result == value

    # Test with expired signature (should raise SignatureExpired)
    expired_signer = TimestampSigner("secret-key")
    expired_signer.get_timestamp = lambda: int(time.time()) - 2000
    expired_signed_value = expired_signer.sign(value)

    with pytest.raises(SignatureExpired):
        signer.unsign(expired_signed_value, max_age=1000)

    # Test with malformed timestamp (should raise BadTimeSignature)
    malformed_signed_value = value + b":" + b"malformed_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed_value)

    # Test with missing timestamp (should raise BadTimeSignature)
    no_timestamp_signed_value = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_signed_value)

    # Test with bad signature (should raise BadTimeSignature)
    bad_signed_value = value + b":" + base64_encode(int_to_bytes(signer.get_timestamp())) + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)


# LLM-generated content at query #36
#--------------------------

```python
def test_TimestampSigner_unsign():
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
    assert timestamp.tzinfo is not None

    # Test unsigning with max_age
    unsign_result = signer.unsign(signed_value, max_age=3600)
    assert unsign_result == value

    # Test expired signature
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test malformed signature
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"malformed_signature")

    # Test missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test_value:signature_without_timestamp")


# LLM-generated content at query #37
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)

    # Test normal unsign
    unsign_result = signer.unsign(signed_value)
    assert unsign_result == value

    # Test unsign with return_timestamp=True
    unsign_result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert unsign_result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test unsign with max_age
    unsign_result = signer.unsign(signed_value, max_age=1000)
    assert unsign_result == value

    # Test expired signature
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test bad signature
    with pytest.raises(BadSignature):
        signer.unsign(b"bad_signature")

    # Test malformed timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test_value" + b":" + b"bad_timestamp" + b":" + b"signature")

    # Test missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test_value" + b":" + b"signature")


# LLM-generated content at query #38
#--------------------------

```python
def test_TimestampSigner_unsign():
    signer = TimestampSigner("secret-key")

    # Test successful unsign without timestamp return
    value = b"test_value"
    signed = signer.sign(value)
    unsign_result = signer.unsign(signed)
    assert unsign_result == value

    # Test successful unsign with timestamp return
    signed = signer.sign(value)
    unsign_result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert unsign_result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

    # Test unsign with max_age
    signed = signer.sign(value)
    unsign_result = signer.unsign(signed, max_age=100)
    assert unsign_result == value

    # Test expired signature
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)

    # Test malformed timestamp
    malformed_signed = value + b":" + b"malformed_timestamp" + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_signed)

    # Test missing timestamp
    no_timestamp_signed = value + b":" + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(no_timestamp_signed)

    # Test bad signature
    bad_signed = value + b":" + base64_encode(int_to_bytes(signer.get_timestamp())) + b":" + b"bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed)

    # Test negative age
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #39
#--------------------------

```python
def test_TimestampSigner_unsign():
    # Setup
    signer = TimestampSigner("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)
    current_time = signer.get_timestamp()

    # Test successful unsign without timestamp return
    unsign_result = signer.unsign(signed_value)
    assert unsign_result == value

    # Test successful unsign with timestamp return
    unsign_result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert unsign_result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

    # Test with max_age within valid range
    unsign_result = signer.unsign(signed_value, max_age=60)
    assert unsign_result == value

    # Test with max_age exceeded
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

    # Test with corrupted timestamp
    corrupted_signed_value = signed_value[:-1] + b"A"
    with pytest.raises(BadTimeSignature):
        signer.unsign(corrupted_signed_value)

    # Test with missing separator
    invalid_signed_value = b"test_value"
    with pytest.raises(BadTimeSignature):
        signer.unsign(invalid_signed_value)

    # Test with bad signature
    bad_signed_value = b"test_value" + b":" + base64_encode(int_to_bytes(current_time)) + b":bad_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(bad_signed_value)

    # Test with negative max_age (future timestamp)
    future_timestamp = current_time + 100
    future_signed_value = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(future_signed_value, max_age=0)


# LLM-generated content at query #40
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
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"malformed_signature")

    # Test with missing timestamp
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test_value" + signer.sep + b"invalid_timestamp")

    # Test with negative max_age
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)


