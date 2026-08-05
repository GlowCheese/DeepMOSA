####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_keys():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode("utf-8")

        @staticmethod
        def loads(s):
            return json.loads(s.decode("utf-8"))

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #2
#--------------------------

```python
def test_iter_unsigners_default_signer():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == ["secret-key"]
    assert signers[0].salt == b"itsdangerous"

def test_iter_unsigners_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].salt == b"custom-salt"

def test_iter_unsigners_with_fallback_dict():
    serializer = Serializer("secret-key", fallback_signers=[{"key_derivation": "hmac"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[1].signer_kwargs == {"key_derivation": "hmac"}

def test_iter_unsigners_with_fallback_tuple():
    serializer = Serializer("secret-key", fallback_signers=[(Signer, {"key_derivation": "hmac"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[1].signer_kwargs == {"key_derivation": "hmac"}

def test_iter_unsigners_with_fallback_class():
    serializer = Serializer("secret-key", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_with_multiple_fallbacks():
    serializer = Serializer(
        "secret-key",
        fallback_signers=[
            {"key_derivation": "hmac"},
            (Signer, {"key_derivation": "concat"}),
            Signer
        ]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4
    for signer in signers:
        assert isinstance(signer, Signer)

def test_iter_unsigners_with_key_rotation():
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == ["old-key", "new-key"]

def test_iter_unsigners_with_key_rotation_and_fallback():
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"key_derivation": "hmac"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert signers[0].secret_keys == ["old-key", "new-key"]
    assert signers[1].secret_keys == ["old-key"]
    assert signers[2].secret_keys == ["new-key"]

def test_iter_unsigners_custom_signer_class():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret-key", signer=CustomSigner)
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], CustomSigner)


# LLM-generated content at query #3
#--------------------------

```python
def test_loads_deserializes_payload():
    serializer = _PDataSerializer()
    payload = b'{"key": "value"}'
    result = serializer.loads(payload)
    assert isinstance(result, dict)
    assert result == {"key": "value"}


# LLM-generated content at query #4
#--------------------------

```python
def test_signer_not_none():
    serializer = Serializer(b"secret", signer=Signer)
    assert serializer.signer is Signer


# LLM-generated content at query #5
#--------------------------

```python
def test_serializer_default_serializer():
    serializer = Serializer(b"secret")
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True

def test_serializer_custom_serializer():
    custom_serializer = json
    serializer = Serializer(b"secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True

def test_serializer_text_serializer():
    serializer = Serializer(b"secret", serializer=json)
    assert is_text_serializer(serializer.serializer) is True

def test_serializer_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(data):
            return json.dumps(data).encode("utf-8")

        @staticmethod
        def loads(data):
            return json.loads(data.decode("utf-8"))

    serializer = Serializer(b"secret", serializer=BytesSerializer())
    assert is_text_serializer(serializer.serializer) is False

def test_serializer_secret_key_single():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_secret_key_list():
    serializer = Serializer([b"secret1", b"secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]

def test_serializer_secret_key_str():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_salt_default():
    serializer = Serializer(b"secret")
    assert serializer.salt == b"itsdangerous"

def test_serializer_salt_custom():
    serializer = Serializer(b"secret", salt="custom")
    assert serializer.salt == b"custom"

def test_serializer_salt_none():
    serializer = Serializer(b"secret", salt=None)
    assert serializer.salt is None

def test_serializer_signer_default():
    serializer = Serializer(b"secret")
    assert serializer.signer == Serializer.default_signer

def test_serializer_signer_custom():
    serializer = Serializer(b"secret", signer=Signer)
    assert serializer.signer == Signer

def test_serializer_signer_kwargs_default():
    serializer = Serializer(b"secret")
    assert serializer.signer_kwargs == {}

def test_serializer_signer_kwargs_custom():
    serializer = Serializer(b"secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_fallback_signers_default():
    serializer = Serializer(b"secret")
    assert serializer.fallback_signers == []

def test_serializer_fallback_signers_custom():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer(b"secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_serializer_kwargs_default():
    serializer = Serializer(b"secret")
    assert serializer.serializer_kwargs == {}

def test_serializer_serializer_kwargs_custom():
    serializer = Serializer(b"secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #6
#--------------------------

```python
def test_serializer_constructor_with_string_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_keys():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer={"dumps": lambda x: "custom", "loads": lambda x: {}})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == {"dumps": lambda x: "custom", "loads": lambda x: {}}
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=type("CustomSigner", (), {}))
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == type("CustomSigner", (), {})
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"key": "value"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key": "value"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"key": "value"}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    serializer = Serializer("secret", serializer={"dumps": lambda x: b"custom", "loads": lambda x: {}})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == {"dumps": lambda x: b"custom", "loads": lambda x: {}}
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #7
#--------------------------

```python
def test_salt_is_none():
    serializer = Serializer(secret_key="test", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #8
#--------------------------

```python
def test_load_payload_with_default_serializer():
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_text_serializer():
    class CustomTextSerializer:
        def dumps(self, obj):
            return str(obj)

        def loads(self, payload):
            return {"custom": payload}

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    payload = b'custom_data'
    result = serializer.load_payload(payload)
    assert result == {"custom": "custom_data"}

def test_load_payload_with_custom_binary_serializer():
    class CustomBinarySerializer:
        def dumps(self, obj):
            return bytes(obj, 'utf-8')

        def loads(self, payload):
            return {"binary": payload.decode('utf-8')}

    serializer = Serializer("secret-key", serializer=CustomBinarySerializer())
    payload = b'binary_data'
    result = serializer.load_payload(payload)
    assert result == {"binary": "binary_data"}

def test_load_payload_with_invalid_payload():
    serializer = Serializer("secret-key")
    payload = b'invalid_json'
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #9
#--------------------------

```python
def test_dumps_serializes_object():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, str)


# LLM-generated content at query #10
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = json
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    custom_signer = Signer
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is custom_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode("utf-8")

        @staticmethod
        def loads(s):
            return json.loads(s.decode("utf-8"))

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is not Serializer.default_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #11
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode("utf-8")

        @staticmethod
        def loads(s):
            return json.loads(s.decode("utf-8"))

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #12
#--------------------------

```python
def test_load_payload_with_invalid_data():
    serializer = Serializer(b"secret-key")
    payload = b"invalid-json-data"
    try:
        serializer.load_payload(payload)
    except BadPayload as e:
        assert e.original_error is not None


# LLM-generated content at query #13
#--------------------------

```python
def test_iter_unsigners_with_tuple_fallback():
    serializer = Serializer("secret-key")
    fallback_signers = [(Signer, {"key_derivation": "hmac"})]
    serializer.fallback_signers = fallback_signers
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)


# LLM-generated content at query #14
#--------------------------

```python
def test_serializer_constructor_defaults():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_custom_serializer():
    custom_serializer = {"dumps": lambda x: x, "loads": lambda x: x}
    serializer = Serializer(b"secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer(b"secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_custom_signer_kwargs():
    serializer = Serializer(b"secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_custom_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer(b"secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_constructor_custom_serializer_kwargs():
    serializer = Serializer(b"secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_key_rotation():
    serializer = Serializer([b"old_secret", b"new_secret"])
    assert serializer.secret_keys == [b"old_secret", b"new_secret"]
    assert serializer.secret_key == b"new_secret"

def test_serializer_constructor_custom_salt():
    serializer = Serializer(b"secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_none_salt():
    serializer = Serializer(b"secret", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #15
#--------------------------

```python
def test_salt_none_predicate():
    serializer = Serializer(secret_key="test", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #16
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"sep": ":"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"sep": ":"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": ":"}, (Signer, {"sep": ";"})]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #17
#--------------------------

```python
def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_custom_values():
    serializer = Serializer(
        secret_key="secret",
        salt="custom_salt",
        serializer=json,
        serializer_kwargs={"indent": 2},
        signer=Signer,
        signer_kwargs={"sep": "|"},
        fallback_signers=[{"sep": ":"}]
    )
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"sep": "|"}
    assert serializer.fallback_signers == [{"sep": ":"}]
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_key_rotation():
    serializer = Serializer(["old_secret", "new_secret"])
    assert serializer.secret_keys == [b"old_secret", b"new_secret"]
    assert serializer.secret_key == b"new_secret"

def test_serializer_constructor_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_bytes_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.is_text_serializer is True

def test_serializer_constructor_text_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.is_text_serializer is True

def test_serializer_constructor_fallback_signers_default():
    serializer = Serializer("secret")
    assert serializer.fallback_signers == []

def test_serializer_constructor_fallback_signers_custom():
    serializer = Serializer("secret", fallback_signers=[{"sep": ":"}, (Signer, {"sep": "|"})])
    assert serializer.fallback_signers == [{"sep": ":"}, (Signer, {"sep": "|"})]


# LLM-generated content at query #18
#--------------------------

```python
def test_load_payload_with_exception_raises_bad_payload():
    serializer = Serializer(b"secret-key")
    payload = b"invalid-payload"
    assert isinstance(serializer.load_payload(payload), BadPayload)


# LLM-generated content at query #19
#--------------------------

```python
def test_dumps_with_string_serializer():
    serializer = Serializer("secret-key", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "key" in result

def test_dumps_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=json, serializer_kwargs={"ensure_ascii": False})
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert b"key" in result

def test_dumps_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    result = serializer.dumps({"data": "test"})
    assert isinstance(result, str)
    assert "data" in result

def test_dumps_with_different_secret_key():
    serializer = Serializer("another-secret", salt="test-salt")
    result = serializer.dumps({"test": 123})
    assert isinstance(result, str)
    assert "test" in result

def test_dumps_with_list_secret_key():
    serializer = Serializer(["key1", "key2"], salt="list-salt")
    result = serializer.dumps({"list": "test"})
    assert isinstance(result, str)
    assert "list" in result

def test_dumps_empty_object():
    serializer = Serializer("secret-key")
    result = serializer.dumps({})
    assert isinstance(result, str)
    assert result != ""

def test_dumps_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    result = serializer.dumps({"salt": None})
    assert isinstance(result, str)
    assert "salt" in result


# LLM-generated content at query #20
#--------------------------

```python
def test_dumps_with_string_serializer():
    serializer = Serializer("secret-key", serializer=json)
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert "key" in result

def test_dumps_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=json, serializer_kwargs={"ensure_ascii": False})
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, str)

def test_dumps_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    data = {"key": "value"}
    result = serializer.dumps(data, salt="another-salt")
    assert isinstance(result, str)

def test_dumps_with_bytes_data():
    serializer = Serializer("secret-key")
    data = b"binary-data"
    result = serializer.dumps(data)
    assert isinstance(result, str)

def test_dumps_with_empty_data():
    serializer = Serializer("secret-key")
    data = {}
    result = serializer.dumps(data)
    assert isinstance(result, str)

def test_dumps_with_nested_data():
    serializer = Serializer("secret-key")
    data = {"nested": {"key": "value"}}
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert "nested" in result

def test_dumps_with_list_data():
    serializer = Serializer("secret-key")
    data = [1, 2, 3]
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert "1" in result

def test_dumps_with_none_data():
    serializer = Serializer("secret-key")
    data = None
    result = serializer.dumps(data)
    assert isinstance(result, str)


# LLM-generated content at query #21
#--------------------------

```python
def test_fallback_signers_not_none():
    serializer = Serializer(b"secret", fallback_signers=[{}])
    assert serializer.fallback_signers is not None


# LLM-generated content at query #22
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret-key-1", "secret-key-2"])
    assert serializer.secret_keys == [b"secret-key-1", b"secret-key-2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is not json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}, (Signer, {"key_derivation": "hmac"}), Signer]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt is None
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return bytes(str(obj), "utf-8")

        @staticmethod
        def loads(s):
            return int(s.decode("utf-8"))

    serializer = Serializer("secret-key", serializer=BytesSerializer())
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is not json
    assert serializer.is_text_serializer is False
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #23
#--------------------------

```python
def test_loads_with_valid_payload():
    serializer = _PDataSerializer()
    payload = b'{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}

def test_loads_with_invalid_payload():
    serializer = _PDataSerializer()
    payload = b'invalid_json'
    try:
        serializer.loads(payload)
    except Exception as e:
        assert isinstance(e, ValueError)


# LLM-generated content at query #24
#--------------------------

```python
def test_salt_none_predicate():
    serializer = Serializer(secret_key="test", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #25
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_iterable_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret-key", serializer=json)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret-key", signer=Signer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"sep": "|"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"sep": "|"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"sep": "|"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #26
#--------------------------

```python
def test_dumps_with_bytes_serializer():
    serializer = Serializer(b"secret", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)


# LLM-generated content at query #27
#--------------------------

```python
def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_key_list():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    serializer = Serializer("secret", serializer=b"custom-serializer")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == b"custom-serializer"
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner:
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"key": "value"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key": "value"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #28
#--------------------------

```python
def test_fallback_signers_not_none():
    serializer = Serializer(b"secret-key", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #29
#--------------------------

```python
def test_dumps_serializes_object():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, str)  # Assuming _TSerialized is str for this test


# LLM-generated content at query #30
#--------------------------

```python
def test_load_payload_with_non_text_serializer():
    serializer = Serializer(b"secret", serializer=json)
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #31
#--------------------------

```python
def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"

def test_serializer_constructor_with_key_list():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_bytes_key_list():
    serializer = Serializer([b"secret1", b"secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode("utf-8")

        @staticmethod
        def loads(s):
            return json.loads(s.decode("utf-8"))

    serializer = Serializer("secret", serializer=BytesSerializer)
    assert serializer.serializer == BytesSerializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer == Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_secret_key_property():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_key == b"secret2"


# LLM-generated content at query #32
#--------------------------

```python
def test_loads_with_valid_payload():
    serializer = _PDataSerializer()
    payload = b'{"key": "value"}'
    result = serializer.loads(payload)
    assert isinstance(result, dict)
    assert result == {"key": "value"}

def test_loads_with_empty_payload():
    serializer = _PDataSerializer()
    payload = b''
    result = serializer.loads(payload)
    assert result is None

def test_loads_with_invalid_payload():
    serializer = _PDataSerializer()
    payload = b'invalid_json'
    try:
        serializer.loads(payload)
        assert False, "Expected an exception"
    except Exception:
        pass


# LLM-generated content at query #33
#--------------------------

```python
def test_load_payload_with_non_text_serializer():
    serializer = Serializer(b"secret", serializer=json)
    assert serializer.is_text_serializer is True
    assert serializer.load_payload(b'{"key": "value"}') == {"key": "value"}

def test_load_payload_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def loads(data):
            return data

        @staticmethod
        def dumps(data):
            return data

    serializer = Serializer(b"secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False
    assert serializer.load_payload(b"data") == b"data"


# LLM-generated content at query #34
#--------------------------

```python
def test_iter_unsigners_default_signer():
    serializer = Serializer("secret-key")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_key == b"secret-key"
    assert unsigners[0].salt == b"itsdangerous"

def test_iter_unsigners_with_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_key == b"secret-key"
    assert unsigners[0].salt == b"custom-salt"

def test_iter_unsigners_with_fallback_dict():
    serializer = Serializer("secret-key", fallback_signers=[{"key_derivation": "hmac"}])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].signer_kwargs == {"key_derivation": "hmac"}

def test_iter_unsigners_with_fallback_tuple():
    serializer = Serializer("secret-key", fallback_signers=[(Signer, {"key_derivation": "hmac"})])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].signer_kwargs == {"key_derivation": "hmac"}

def test_iter_unsigners_with_fallback_class():
    serializer = Serializer("secret-key", fallback_signers=[Signer])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)

def test_iter_unsigners_with_multiple_fallbacks():
    serializer = Serializer(
        "secret-key",
        fallback_signers=[
            {"key_derivation": "hmac"},
            (Signer, {"key_derivation": "concat"}),
            Signer
        ]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 4
    assert all(isinstance(unsigner, Signer) for unsigner in unsigners)

def test_iter_unsigners_with_key_rotation():
    serializer = Serializer(["old-key", "new-key"])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_key == b"new-key"

def test_iter_unsigners_with_key_rotation_and_fallback():
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"key_derivation": "hmac"}]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 3
    assert all(isinstance(unsigner, Signer) for unsigner in unsigners)
    assert unsigners[0].secret_key == b"new-key"
    assert unsigners[1].secret_key == b"old-key"
    assert unsigners[2].secret_key == b"new-key"


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_load_payload_with_text_serializer():
    serializer = Serializer("secret", serializer=json)
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def dumps(self, obj):
            return pickle.dumps(obj)

        def loads(self, payload):
            return pickle.loads(payload)

    serializer = Serializer("secret", serializer=BinarySerializer())
    payload = pickle.dumps({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def dumps(self, obj):
            return str(obj).encode()

        def loads(self, payload):
            return int(payload.decode())

    serializer = Serializer("secret", serializer=CustomSerializer())
    payload = b"42"
    result = serializer.load_payload(payload)
    assert result == 42

def test_load_payload_with_invalid_payload():
    serializer = Serializer("secret", serializer=json)
    payload = b"invalid json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #2
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret", salt="salt", serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret", salt=b"salt", serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret1", "secret2"], salt="salt", serializer=json)
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None, serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret", salt="salt", serializer=CustomSerializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"salt"
    assert serializer.serializer == CustomSerializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", salt="salt", serializer=json, serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", salt="salt", serializer=json, signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", salt="salt", serializer=json, signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}, (Signer, {"key_derivation": "hmac"}), Signer]
    serializer = Serializer("secret", salt="salt", serializer=json, fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #3
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret-key-1", "secret-key-2"])
    assert serializer.secret_keys == [b"secret-key-1", b"secret-key-2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: str(x), "loads": lambda x: x}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #4
#--------------------------

```python
def test_loads_deserializes_payload():
    serializer = _PDataSerializer()
    payload = "test_payload"
    result = serializer.loads(payload)
    assert result is not None


# LLM-generated content at query #5
#--------------------------

```python
def test_load_payload_with_non_text_serializer_raises_badpayload():
    serializer = Serializer(b"secret", serializer=JSONSerializer())
    with pytest.raises(BadPayload):
        serializer.load_payload(b"invalid json")


# LLM-generated content at query #6
#--------------------------

```python
def test_serializer_not_none():
    serializer = Serializer(b"secret-key", serializer="custom_serializer")
    assert serializer.serializer == "custom_serializer"


# LLM-generated content at query #7
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_iterable_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": True}}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"sep": "|"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"sep": "|"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": "|"}, (Signer, {"sep": "|"}), Signer]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"bytes", "loads": lambda x: {"bytes": True}}
    serializer = Serializer("secret-key", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #8
#--------------------------

```python
def test_signer_not_none():
    serializer = Serializer(b"secret", signer=Signer)
    assert serializer.signer is Signer


# LLM-generated content at query #9
#--------------------------

```python
def test_dumps_with_string_serializer():
    serializer = Serializer("secret-key", serializer=json)
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert serializer.loads(result) == data

def test_dumps_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=json, serializer_kwargs={"ensure_ascii": False})
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, bytes)
    assert serializer.loads(result) == data

def test_dumps_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert serializer.loads(result, salt="custom-salt") == data

def test_dumps_with_different_salt_fails():
    serializer = Serializer("secret-key", salt="salt1")
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert serializer.loads(result, salt="salt1") == data
    try:
        serializer.loads(result, salt="salt2")
        assert False, "Expected BadSignature"
    except BadSignature:
        pass

def test_dumps_with_empty_data():
    serializer = Serializer("secret-key")
    result = serializer.dumps({})
    assert serializer.loads(result) == {}

def test_dumps_with_none_data():
    serializer = Serializer("secret-key")
    result = serializer.dumps(None)
    assert serializer.loads(result) is None

def test_dumps_with_complex_data():
    serializer = Serializer("secret-key")
    data = {"list": [1, 2, 3], "nested": {"a": 1, "b": 2}}
    result = serializer.dumps(data)
    assert serializer.loads(result) == data

def test_dumps_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert serializer.loads(result) == data

def test_dumps_with_key_rotation():
    serializer = Serializer(["old-key", "new-key"])
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert serializer.loads(result) == data


# LLM-generated content at query #10
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_iterable_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"sep": ";"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"sep": ";"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": ";"}, (Signer, {"sep": ":"})]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            return b"bytes"
        def loads(self, s):
            return {"bytes": s}
    serializer = Serializer("secret-key", serializer=BytesSerializer())
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #11
#--------------------------

```python
def test_load_payload_with_text_serializer():
    serializer = Serializer("secret", serializer=json)
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def dumps(self, obj):
            return pickle.dumps(obj)

        def loads(self, payload):
            return pickle.loads(payload)

    serializer = Serializer("secret", serializer=BinarySerializer())
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def dumps(self, obj):
            return str(obj).encode()

        def loads(self, payload):
            return int(payload.decode())

    serializer = Serializer("secret", serializer=CustomSerializer())
    payload = serializer.dump_payload(42)
    result = serializer.load_payload(payload)
    assert result == 42

def test_load_payload_with_invalid_payload():
    serializer = Serializer("secret", serializer=json)
    try:
        serializer.load_payload(b"invalid json")
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #12
#--------------------------

```python
def test_iter_unsigners_default_signer():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [b"secret-key"]
    assert signers[0].salt == b"itsdangerous"

def test_iter_unsigners_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [b"secret-key"]
    assert signers[0].salt == b"custom-salt"

def test_iter_unsigners_with_fallback_dict():
    serializer = Serializer("secret-key", fallback_signers=[{"key_derivation": "hmac"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[1].signer_kwargs == {"key_derivation": "hmac"}

def test_iter_unsigners_with_fallback_tuple():
    serializer = Serializer("secret-key", fallback_signers=[(Signer, {"key_derivation": "hmac"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[1].signer_kwargs == {"key_derivation": "hmac"}

def test_iter_unsigners_with_fallback_class():
    serializer = Serializer("secret-key", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_with_multiple_fallbacks():
    serializer = Serializer(
        "secret-key",
        fallback_signers=[
            {"key_derivation": "hmac"},
            (Signer, {"key_derivation": "concat"}),
            Signer
        ]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4
    for signer in signers:
        assert isinstance(signer, Signer)

def test_iter_unsigners_with_key_rotation():
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [b"old-key", b"new-key"]

def test_iter_unsigners_with_key_rotation_and_fallbacks():
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"key_derivation": "hmac"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    assert isinstance(signers[1], Signer)
    assert signers[1].secret_keys == [b"old-key"]
    assert isinstance(signers[2], Signer)
    assert signers[2].secret_keys == [b"new-key"]


# LLM-generated content at query #13
#--------------------------

```python
def test_serializer_constructor_default_serializer():
    serializer = Serializer(b"secret-key")
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_custom_serializer():
    serializer = Serializer(b"secret-key", serializer=str)
    assert serializer.serializer == str
    assert serializer.is_text_serializer is True
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_custom_salt():
    serializer = Serializer(b"secret-key", salt="custom-salt")
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_custom_signer():
    serializer = Serializer(b"secret-key", signer=TimedSerializer)
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.signer == TimedSerializer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_custom_signer_kwargs():
    serializer = Serializer(b"secret-key", signer_kwargs={"digest_method": SHA256})
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"digest_method": SHA256}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_custom_fallback_signers():
    fallback_signers = [{"digest_method": SHA256}, TimedSerializer]
    serializer = Serializer(b"secret-key", fallback_signers=fallback_signers)
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_custom_serializer_kwargs():
    serializer = Serializer(b"secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_key_rotation():
    serializer = Serializer([b"old-key", b"new-key"])
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.secret_keys == [b"old-key", b"new-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_no_salt():
    serializer = Serializer(b"secret-key", salt=None)
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt is None
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #14
#--------------------------

```python
def test_salt_is_none():
    serializer = Serializer(secret_key="test")
    assert serializer.salt is None


# LLM-generated content at query #15
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_keys():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode("utf-8")

        @staticmethod
        def loads(s):
            return json.loads(s.decode("utf-8"))

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #16
#--------------------------

```python
def test_serializer_constructor_defaults():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_custom_serializer():
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
    serializer = Serializer(b"secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_custom_signer():
    serializer = Serializer(b"secret", signer=TimedSerializer)
    assert serializer.signer == TimedSerializer

def test_serializer_constructor_custom_signer_kwargs():
    serializer = Serializer(b"secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_custom_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}, (TimedSerializer, {"key_derivation": "hmac"})]
    serializer = Serializer(b"secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_constructor_custom_serializer_kwargs():
    serializer = Serializer(b"secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_key_rotation():
    serializer = Serializer([b"old_secret", b"new_secret"])
    assert serializer.secret_keys == [b"old_secret", b"new_secret"]
    assert serializer.secret_key == b"new_secret"

def test_serializer_constructor_custom_salt():
    serializer = Serializer(b"secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_none_salt():
    serializer = Serializer(b"secret", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #17
#--------------------------

```python
def test_signer_is_not_none():
    serializer = Serializer(b"secret-key", signer=Signer)
    assert serializer.signer is not None


# LLM-generated content at query #18
#--------------------------

```python
def test_load_payload_with_text_serializer():
    serializer = Serializer("secret-key")
    payload = serializer.dumps({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def dumps(self, obj):
            return str(obj).encode("utf-8")

        def loads(self, payload):
            return int(payload.decode("utf-8"))

    serializer = Serializer("secret-key", serializer=BinarySerializer())
    payload = serializer.dumps(42)
    result = serializer.load_payload(payload)
    assert result == 42

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def dumps(self, obj):
            return f"custom-{obj}"

        def loads(self, payload):
            return payload.replace("custom-", "")

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    payload = serializer.dumps("data")
    result = serializer.load_payload(payload)
    assert result == "data"

def test_load_payload_with_invalid_payload():
    serializer = Serializer("secret-key")
    try:
        serializer.load_payload(b"invalid-payload")
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass


# LLM-generated content at query #19
#--------------------------

```python
def test_iter_unsigners_with_dict_fallback():
    serializer = Serializer("secret-key", fallback_signers=[{"key_derivation": "hmac"}])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)


# LLM-generated content at query #20
#--------------------------

```python
def test_load_payload_with_non_text_serializer():
    serializer = Serializer(b"secret-key")
    assert serializer.is_text_serializer is False
    assert serializer.load_payload(b"test") is None


# LLM-generated content at query #21
#--------------------------

```python
def test_iter_unsigners_with_dict_fallback():
    serializer = Serializer("secret", fallback_signers=[{"key": "value"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)


# LLM-generated content at query #22
#--------------------------

```python
def test_salt_none_predicate():
    serializer = Serializer(secret_key="test", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #23
#--------------------------

```python
def test_load_payload_with_non_text_serializer():
    serializer = Serializer(b"secret", serializer=json)
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #24
#--------------------------

```python
def test_load_payload_with_text_serializer():
    serializer = Serializer("secret", serializer=json)
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def dumps(self, obj):
            return pickle.dumps(obj)

        def loads(self, payload):
            return pickle.loads(payload)

    serializer = Serializer("secret", serializer=BinarySerializer())
    payload = pickle.dumps({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')

        def loads(self, payload):
            return int(payload.decode('utf-8'))

    serializer = Serializer("secret", serializer=CustomSerializer())
    payload = b"42"
    result = serializer.load_payload(payload)
    assert result == 42

def test_load_payload_with_invalid_payload():
    serializer = Serializer("secret")
    payload = b"invalid json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #25
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"sep": "|"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"sep": "|"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": "|"}, (Signer, {"sep": "|"}), Signer]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_serializer():
    serializer = Serializer("secret", serializer=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_signer():
    serializer = Serializer("secret", signer=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #26
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode("utf-8")

        @staticmethod
        def loads(s):
            return json.loads(s.decode("utf-8"))

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #27
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret-key-1", "secret-key-2"])
    assert serializer.secret_keys == [b"secret-key-1", b"secret-key-2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret-key", serializer=json)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret-key", signer=Signer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}, (Signer, {"key_derivation": "hmac"})]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #28
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"sep": "|"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"sep": "|"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": "|"}, (Signer, {"sep": "|"})]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_all_parameters():
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
    serializer = Serializer(
        ["secret1", "secret2"],
        salt="custom_salt",
        serializer=custom_serializer,
        serializer_kwargs={"indent": 2},
        signer=Signer,
        signer_kwargs={"sep": "|"},
        fallback_signers=[{"sep": "|"}]
    )
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"sep": "|"}
    assert serializer.fallback_signers == [{"sep": "|"}]
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #29
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_iterable_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {}}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"sep": "|"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"sep": "|"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": "|"}, (Signer, {"sep": "|"}), Signer]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {}}
    serializer = Serializer("secret-key", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #30
#--------------------------

```python
def test_iter_unsigners_with_tuple_fallback():
    serializer = Serializer(b"secret", fallback_signers=[(Signer, {"key_derivation": "hmac"})])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].kwargs == {"key_derivation": "hmac"}


# LLM-generated content at query #31
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_iterable_secret_key():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #32
#--------------------------

```python
def test_serializer_default_serializer():
    serializer = Serializer("secret")
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_serializer_custom_serializer():
    serializer = Serializer("secret", serializer=lambda x: x)
    assert serializer.serializer is not json
    assert serializer.is_text_serializer is True

def test_serializer_default_signer():
    serializer = Serializer("secret")
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}

def test_serializer_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}

def test_serializer_default_fallback_signers():
    serializer = Serializer("secret")
    assert serializer.fallback_signers == []

def test_serializer_custom_fallback_signers():
    fallback = [Signer]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_secret_keys_single():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_secret_keys_multiple():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]

def test_serializer_secret_key_property():
    serializer = Serializer("secret")
    assert serializer.secret_key == b"secret"

def test_serializer_secret_key_property_multiple():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_key == b"secret2"

def test_serializer_salt_default():
    serializer = Serializer("secret")
    assert serializer.salt == b"itsdangerous"

def test_serializer_salt_custom():
    serializer = Serializer("secret", salt="custom")
    assert serializer.salt == b"custom"

def test_serializer_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_serializer_kwargs():
    kwargs = {"key": "value"}
    serializer = Serializer("secret", serializer_kwargs=kwargs)
    assert serializer.serializer_kwargs == kwargs

def test_serializer_signer_kwargs():
    kwargs = {"key": "value"}
    serializer = Serializer("secret", signer_kwargs=kwargs)
    assert serializer.signer_kwargs == kwargs


# LLM-generated content at query #33
#--------------------------

```python
def test_serializer_defaults():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_with_list_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is False

def test_serializer_with_custom_signer():
    serializer = Serializer("secret-key", signer=TimedSerializer)
    assert serializer.signer is TimedSerializer

def test_serializer_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"sep": "|"})
    assert serializer.signer_kwargs == {"sep": "|"}

def test_serializer_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}, (TimedSerializer, {"sep": "|"}), Signer]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_secret_key_property():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_key == b"key2"


# LLM-generated content at query #34
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_iterable_secret_keys():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=TimedSerializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == TimedSerializer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac", "digest_method": "SHA256"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac", "digest_method": "SHA256"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac", "digest_method": "SHA256"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_secret_key_property():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_key == b"secret2"


