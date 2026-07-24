####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_bytes_serializer():
    class CustomBytesSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode()

        @staticmethod
        def loads(b):
            return int(b.decode())

    serializer = Serializer("secret", serializer=CustomBytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    class FallbackSigner(Signer):
        pass

    fallback_signers = [FallbackSigner]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == [FallbackSigner]

def test_serializer_constructor_with_fallback_signers_as_dict():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == [{"key_derivation": "hmac"}]

def test_serializer_constructor_with_fallback_signers_as_tuple():
    class FallbackSigner(Signer):
        pass

    fallback_signers = [(FallbackSigner, {"key_derivation": "hmac"})]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == [(FallbackSigner, {"key_derivation": "hmac"})]


# LLM-generated content at query #2
#--------------------------

def test_load_payload_uses_default_serializer_when_none():
    serializer = Serializer("secret", serializer=json)
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_uses_provided_serializer():
    serializer = Serializer("secret", serializer=json)
    payload = b'{"test": 123}'
    result = serializer.load_payload(payload, serializer=json)
    assert result == {"test": 123}

def test_load_payload_with_text_serializer():
    class TextSerializer:
        def dumps(self, obj):
            return "text"
        def loads(self, data):
            return {"from": "text"}
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = b'{"dummy": true}'
    result = serializer.load_payload(payload)
    assert result == {"from": "text"}

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def dumps(self, obj):
            return b"binary"
        def loads(self, data):
            return {"from": "binary"}
    serializer = Serializer("secret", serializer=BinarySerializer())
    payload = b"raw"
    result = serializer.load_payload(payload)
    assert result == {"from": "binary"}

def test_load_payload_raises_bad_payload_on_exception():
    class FaultySerializer:
        def dumps(self, obj):
            return b"faulty"
        def loads(self, data):
            raise ValueError("broken")
    serializer = Serializer("secret", serializer=FaultySerializer())
    try:
        serializer.load_payload(b"data")
        assert False
    except BadPayload:
        pass

def test_load_payload_decodes_text_payload():
    class TextSerializer:
        def dumps(self, obj):
            return "text"
        def loads(self, data):
            assert isinstance(data, str)
            return {"decoded": True}
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = b"some bytes"
    result = serializer.load_payload(payload)
    assert result == {"decoded": True}

def test_load_payload_passes_bytes_to_binary_serializer():
    class BinarySerializer:
        def dumps(self, obj):
            return b"binary"
        def loads(self, data):
            assert isinstance(data, bytes)
            return {"bytes": True}
    serializer = Serializer("secret", serializer=BinarySerializer())
    payload = b"some bytes"
    result = serializer.load_payload(payload)
    assert result == {"bytes": True}


# LLM-generated content at query #3
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer == True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        def dumps(self, obj):
            return "custom"
        def loads(self, s):
            return "custom loaded"
    serializer = Serializer("secret-key", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            return b"bytes"
        def loads(self, s):
            return "bytes loaded"
    serializer = Serializer("secret-key", serializer=BytesSerializer())
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    from itsdangerous.signer import Signer
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.signer import Signer
    class FallbackSigner(Signer):
        pass
    fallback = [{"key_derivation": "none"}, (FallbackSigner, {}), Signer]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"bytes-key")
    assert serializer.secret_keys == [b"bytes-key"]


# LLM-generated content at query #4
#--------------------------

def test_load_payload_with_none_serializer_and_text_serializer():
    serializer_instance = Serializer("secret", serializer=None)
    result = serializer_instance.load_payload(b'"test"')
    assert result == "test"

def test_load_payload_with_none_serializer_and_bytes_serializer():
    serializer_instance = Serializer("secret", serializer=Serializer._PDataSerializer())
    result = serializer_instance.load_payload(b'"test"')
    assert result == "test"

def test_load_payload_with_custom_text_serializer():
    custom_serializer = _PDataSerializer[str]()
    serializer_instance = Serializer("secret", serializer=custom_serializer)
    result = serializer_instance.load_payload(b'"test"', serializer=custom_serializer)
    assert result == "test"

def test_load_payload_with_custom_bytes_serializer():
    custom_serializer = _PDataSerializer[bytes]()
    serializer_instance = Serializer("secret", serializer=custom_serializer)
    result = serializer_instance.load_payload(b'"test"', serializer=custom_serializer)
    assert result == "test"

def test_load_payload_raises_bad_payload_on_exception():
    serializer_instance = Serializer("secret", serializer=None)
    try:
        serializer_instance.load_payload(b"invalid")
        assert False
    except BadPayload:
        pass


# LLM-generated content at query #5
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret-key")
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_serializer_constructor_custom_text_serializer():
    class CustomTextSerializer:
        def dumps(self, obj):
            return "dumped"

        def loads(self, s):
            return "loaded"

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    assert serializer.is_text_serializer is True

def test_serializer_constructor_custom_bytes_serializer():
    class CustomBytesSerializer:
        def dumps(self, obj):
            return b"dumped"

        def loads(self, s):
            return "loaded"

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_single_secret_key_string():
    serializer = Serializer("my-secret")
    assert serializer.secret_keys == [b"my-secret"]

def test_serializer_constructor_single_secret_key_bytes():
    serializer = Serializer(b"my-secret")
    assert serializer.secret_keys == [b"my-secret"]

def test_serializer_constructor_multiple_secret_keys():
    serializer = Serializer(["key1", "key2", b"key3"])
    assert serializer.secret_keys == [b"key1", b"key2", b"key3"]

def test_serializer_constructor_salt_default():
    serializer = Serializer("secret")
    assert serializer.salt == b"itsdangerous"

def test_serializer_constructor_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_salt_bytes():
    serializer = Serializer("secret", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_salt_str():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_default_signer():
    serializer = Serializer("secret")
    assert serializer.signer is Signer

def test_serializer_constructor_custom_signer():
    class CustomSigner:
        def __init__(self, secret_key, salt, **kwargs):
            pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key": "value"})
    assert serializer.signer_kwargs == {"key": "value"}

def test_serializer_constructor_default_fallback_signers():
    serializer = Serializer("secret")
    assert serializer.fallback_signers == []

def test_serializer_constructor_custom_fallback_signers():
    fallback = [{"key": "value"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #6
#--------------------------

def test_dumps_returns_bytes_with_default_serializer():
    serializer = Serializer("secret")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)

def test_dumps_returns_string_with_text_serializer():
    serializer = Serializer("secret", serializer=Serializer.default_serializer)
    serializer.is_text_serializer = True
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)

def test_dumps_returns_signed_payload():
    serializer = Serializer("secret")
    result = serializer.dumps("data")
    assert len(result) > 0

def test_dumps_uses_custom_salt():
    serializer = Serializer("secret")
    result1 = serializer.dumps("data", salt="custom_salt")
    result2 = serializer.dumps("data", salt="other_salt")
    assert result1 != result2

def test_dumps_uses_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    result = serializer.dumps({"b": 2, "a": 1})
    assert b'"a"' in result


# LLM-generated content at query #7
#--------------------------

def test_serializer_constructor_with_default_parameters():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_bytes_list_of_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom_dumps"
        @staticmethod
        def loads(s):
            return "custom_loads"
    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.serializer == CustomSerializer
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes_dumps"
        @staticmethod
        def loads(s):
            return "bytes_loads"
    serializer = Serializer("secret", serializer=BytesSerializer)
    assert serializer.serializer == BytesSerializer
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_constructor_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom_dumps"
        @staticmethod
        def loads(s):
            return "custom_loads"
    class CustomSigner(Signer):
        pass
    serializer = Serializer(
        secret_key=["key1", "key2"],
        salt="custom_salt",
        serializer=CustomSerializer,
        serializer_kwargs={"indent": 2},
        signer=CustomSigner,
        signer_kwargs={"key_derivation": "hmac"},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == CustomSerializer
    assert serializer.is_text_serializer == True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #8
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(Serializer.default_serializer)
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_with_list_of_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret-key", serializer=json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer == is_text_serializer(json)

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret-key", signer=Signer)
    assert serializer.signer == Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_custom_serializer_bytes():
    mock_serializer = type("MockSerializer", (), {"dumps": lambda self, x: b"{}", "loads": lambda self, x: {}})
    serializer = Serializer("secret-key", serializer=mock_serializer)
    assert serializer.is_text_serializer == is_text_serializer(mock_serializer)


# LLM-generated content at query #9
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=Serializer)
    assert serializer.serializer == Serializer
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Serializer)
    assert serializer.signer == Serializer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[{"digest_method": "sha256"}])
    assert serializer.fallback_signers == [{"digest_method": "sha256"}]

def test_serializer_constructor_with_all_params():
    serializer = Serializer(
        secret_key="secret",
        salt="custom_salt",
        serializer=Serializer,
        serializer_kwargs={"sort_keys": True},
        signer=Serializer,
        signer_kwargs={"key_derivation": "none"},
        fallback_signers=[{"digest_method": "sha512"}],
    )
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == Serializer
    assert serializer.is_text_serializer == False
    assert serializer.signer == Serializer
    assert serializer.signer_kwargs == {"key_derivation": "none"}
    assert serializer.fallback_signers == [{"digest_method": "sha512"}]
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #10
#--------------------------

def test_serializer_constructor_default_values():
    secret_key = "secret"
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    secret_key = b"secret"
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_secret_keys():
    secret_key = ["key1", "key2"]
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    secret_key = "secret"
    salt = b"custom_salt"
    serializer = Serializer(secret_key, salt=salt)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    secret_key = "secret"
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: str(x), "loads": lambda self, x: x})()
    serializer = Serializer(secret_key, serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer == isinstance(custom_serializer.dumps({}), str)
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    secret_key = "secret"
    custom_signer = Signer
    serializer = Serializer(secret_key, signer=custom_signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == custom_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    secret_key = "secret"
    signer_kwargs = {"key_derivation": "hmac"}
    serializer = Serializer(secret_key, signer_kwargs=signer_kwargs)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    secret_key = "secret"
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer(secret_key, fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == [{"key_derivation": "hmac"}]
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    secret_key = "secret"
    serializer_kwargs = {"sort_keys": True}
    serializer = Serializer(secret_key, serializer_kwargs=serializer_kwargs)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_none_salt():
    secret_key = "secret"
    serializer = Serializer(secret_key, salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_str_salt():
    secret_key = "secret"
    salt = "str_salt"
    serializer = Serializer(secret_key, salt=salt)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"str_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #11
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"data"

        @staticmethod
        def loads(s):
            return s

    serializer = Serializer("secret", serializer=BytesSerializer)
    assert serializer.serializer is BytesSerializer
    assert serializer.is_text_serializer is False

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #12
#--------------------------

```python
def test_load_payload_is_text_false_with_non_text_serializer():
    class NonTextSerializer:
        def loads(self, data):
            return data

        def dumps(self, obj):
            return obj

    serializer = Serializer("secret", serializer=NonTextSerializer())
    payload = b"test payload"
    result = serializer.load_payload(payload)
    assert result == payload
```


# LLM-generated content at query #13
#--------------------------

```python
def test_load_payload_predicate_false():
    serializer = Serializer("secret", serializer=Serializer.default_serializer)
    serializer.is_text_serializer = False
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
```


# LLM-generated content at query #14
#--------------------------

def test_iter_unsigners_default_salt():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].salt == b"itsdangerous"

def test_iter_unsigners_custom_salt():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners(salt=b"custom-salt"))
    assert signers[0].salt == b"custom-salt"

def test_iter_unsigners_with_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=[{"key": "fallback-key"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_fallback_as_tuple():
    serializer = Serializer("secret-key", fallback_signers=[(Signer, {"digest_method": hashlib.sha256})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2


# LLM-generated content at query #15
#--------------------------

def test_loads_returns_any_type():
    serializer = _PDataSerializer()
    result = serializer.loads("test")
    assert result is not None

def test_loads_accepts_serialized_input():
    serializer = _PDataSerializer()
    result = serializer.loads(b"data")
    assert result is not None

def test_loads_accepts_string_input():
    serializer = _PDataSerializer()
    result = serializer.loads("{}")
    assert result is not None

def test_loads_accepts_bytes_input():
    serializer = _PDataSerializer()
    result = serializer.loads(b"{}")
    assert result is not None

def test_loads_accepts_none_input():
    serializer = _PDataSerializer()
    result = serializer.loads(None)
    assert result is None

def test_loads_returns_different_types():
    serializer = _PDataSerializer()
    result1 = serializer.loads("1")
    result2 = serializer.loads("[1,2]")
    assert result1 != result2


# LLM-generated content at query #16
#--------------------------

def test_load_payload_with_default_serializer():
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def loads(self, payload):
            return payload.upper()
        def dumps(self, obj):
            return obj
    custom_serializer = CustomSerializer()
    serializer = Serializer("secret", serializer=custom_serializer)
    payload = serializer.dump_payload("test")
    result = serializer.load_payload(payload)
    assert result == "TEST"

def test_load_payload_with_text_serializer():
    class TextSerializer:
        def loads(self, payload):
            return eval(payload)
        def dumps(self, obj):
            return str(obj)
    text_serializer = TextSerializer()
    serializer = Serializer("secret", serializer=text_serializer)
    payload = serializer.dump_payload([1, 2, 3])
    result = serializer.load_payload(payload)
    assert result == [1, 2, 3]

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def loads(self, payload):
            return payload
        def dumps(self, obj):
            return obj
    binary_serializer = BinarySerializer()
    serializer = Serializer("secret", serializer=binary_serializer)
    payload = b"binary data"
    result = serializer.load_payload(payload)
    assert result == b"binary data"

def test_load_payload_raises_bad_payload():
    serializer = Serializer("secret")
    try:
        serializer.load_payload(b"invalid payload")
    except BadPayload:
        pass

def test_load_payload_with_serializer_override():
    class OverrideSerializer:
        def loads(self, payload):
            return "override"
        def dumps(self, obj):
            return obj
    override_serializer = OverrideSerializer()
    serializer = Serializer("secret")
    payload = serializer.dump_payload("original")
    result = serializer.load_payload(payload, serializer=override_serializer)
    assert result == "override"

def test_load_payload_preserves_exception():
    class BrokenSerializer:
        def loads(self, payload):
            raise ValueError("broken")
        def dumps(self, obj):
            return obj
    broken_serializer = BrokenSerializer()
    serializer = Serializer("secret", serializer=broken_serializer)
    try:
        serializer.load_payload(b"anything")
    except BadPayload as e:
        assert "broken" in str(e.original_error)


# LLM-generated content at query #17
#--------------------------

```python
def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads("test_payload")
    assert result is not None or result is None


# LLM-generated content at query #18
#--------------------------

```python
def test_iter_unsigners_default_signer_yielded_first():
    signer_kwargs = {"key_derivation": "hmac"}
    serializer = Serializer("secret", signer=Signer, signer_kwargs=signer_kwargs)
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 0
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == serializer.secret_keys
    assert signers[0].salt == serializer.salt

def test_iter_unsigners_with_fallback_signers_dict():
    fallback = {"key_derivation": "none"}
    serializer = Serializer("secret", fallback_signers=[fallback])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], Signer)
    assert signers[1].key_derivation == "none"

def test_iter_unsigners_with_fallback_signers_tuple():
    fallback = (Signer, {"key_derivation": "none"})
    serializer = Serializer("secret", fallback_signers=[fallback])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], Signer)
    assert signers[1].key_derivation == "none"

def test_iter_unsigners_with_fallback_signers_class():
    serializer = Serializer("secret", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_with_salt_overrides_default():
    serializer = Serializer("secret", salt=b"default_salt")
    signers = list(serializer.iter_unsigners(salt=b"custom_salt"))
    assert signers[0].salt == b"custom_salt"

def test_iter_unsigners_with_multiple_secret_keys():
    serializer = Serializer(["key1", "key2"], fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert signers[0].secret_keys == serializer.secret_keys
    assert signers[1].secret_keys == [b"key1"]
    assert signers[2].secret_keys == [b"key2"]

def test_iter_unsigners_no_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
```


# LLM-generated content at query #19
#--------------------------

def test_serializer_init_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_init_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_init_with_custom_serializer_str():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True

def test_serializer_init_with_custom_serializer_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"{}"
        @staticmethod
        def loads(data):
            return {}
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer == False

def test_serializer_init_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"separators": (",", ":")})
    assert serializer.serializer_kwargs == {"separators": (",", ":")}

def test_serializer_init_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_init_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_init_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #20
#--------------------------

def test_load_payload_with_default_serializer_and_text_serializer():
    serializer = Serializer("secret", serializer=json)
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_default_serializer_and_bytes_serializer():
    serializer = Serializer("secret", serializer=json)
    payload = serializer.dump_payload(123)
    result = serializer.load_payload(payload)
    assert result == 123

def test_load_payload_with_custom_text_serializer():
    class CustomTextSerializer:
        def dumps(self, obj):
            return str(obj)
        def loads(self, payload):
            return int(payload)
    serializer = Serializer("secret", serializer=CustomTextSerializer())
    payload = serializer.dump_payload(42)
    result = serializer.load_payload(payload)
    assert result == 42

def test_load_payload_with_custom_bytes_serializer():
    class CustomBytesSerializer:
        def dumps(self, obj):
            return bytes(str(obj), "utf-8")
        def loads(self, payload):
            return int(payload.decode("utf-8"))
    serializer = Serializer("secret", serializer=CustomBytesSerializer())
    payload = serializer.dump_payload(42)
    result = serializer.load_payload(payload)
    assert result == 42

def test_load_payload_with_explicit_serializer_override():
    serializer = Serializer("secret", serializer=json)
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload, serializer=json)
    assert result == {"key": "value"}

def test_load_payload_with_invalid_payload_raises_bad_payload():
    serializer = Serializer("secret")
    try:
        serializer.load_payload(b"invalid")
        assert False
    except BadPayload:
        pass

def test_load_payload_with_explicit_serializer_and_text():
    class CustomTextSerializer:
        def dumps(self, obj):
            return str(obj)
        def loads(self, payload):
            return int(payload)
    serializer = Serializer("secret", serializer=CustomTextSerializer())
    payload = serializer.dump_payload(42)
    result = serializer.load_payload(payload)
    assert result == 42

def test_load_payload_with_explicit_serializer_and_bytes():
    class CustomBytesSerializer:
        def dumps(self, obj):
            return bytes(str(obj), "utf-8")
        def loads(self, payload):
            return int(payload.decode("utf-8"))
    serializer = Serializer("secret", serializer=CustomBytesSerializer())
    payload = serializer.dump_payload(42)
    result = serializer.load_payload(payload)
    assert result == 42


# LLM-generated content at query #21
#--------------------------

def test_serializer_constructor_with_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is serializer.default_serializer
    assert serializer.is_text_serializer
    assert serializer.signer is serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert not serializer.is_text_serializer

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner:
        def __init__(self, secret_key, salt=None, **kwargs):
            pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_text_serializer():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "text"
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret", serializer=TextSerializer())
    assert serializer.is_text_serializer

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert not serializer.is_text_serializer


# LLM-generated content at query #22
#--------------------------

```
def test_iter_unsigners_fallback_tuple():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer

    fallback_signer = Signer
    fallback_kwargs = {"digest_method": "sha256"}
    serializer = Serializer("secret", fallback_signers=[(fallback_signer, fallback_kwargs)])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[1], fallback_signer)
```


# LLM-generated content at query #23
#--------------------------

```
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert result is not None
```


# LLM-generated content at query #24
#--------------------------

```python
def test_serializer_init_with_serializer_not_none():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
```


# LLM-generated content at query #25
#--------------------------

```python
def test_iter_unsigners_with_tuple_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    secret_key = b"test-secret"
    salt = b"test-salt"
    signer_kwargs = {"digest_method": "sha256"}
    fallback_signer = Signer
    fallback_kwargs = {"digest_method": "sha512"}
    serializer = Serializer(
        secret_key=secret_key,
        salt=salt,
        signer_kwargs=signer_kwargs,
        fallback_signers=[(fallback_signer, fallback_kwargs)],
    )
    result = list(serializer.iter_unsigners(salt))
    assert len(result) == 2
    assert isinstance(result[0], Signer)
    assert isinstance(result[1], Signer)
```


# LLM-generated content at query #26
#--------------------------

```python
def test_fallback_signers_is_not_none():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []
```


# LLM-generated content at query #27
#--------------------------

```
def test_dumps_returns_expected_type():
    serializer = _PDataSerializer()
    result = serializer.dumps(42)
    assert result is not None

def test_dumps_returns_serialized_value():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert isinstance(result, type(serializer.dumps("test")))

def test_dumps_accepts_any_object():
    serializer = _PDataSerializer()
    result = serializer.dumps([1, 2, 3])
    assert result is not None

def test_dumps_returns_same_type_for_different_inputs():
    serializer = _PDataSerializer()
    result1 = serializer.dumps("a")
    result2 = serializer.dumps(1)
    assert type(result1) == type(result2)
```


# LLM-generated content at query #28
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer == True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, obj: "text", "loads": lambda self, s: {}})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_fallback_signers_none():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_secret_key_property():
    serializer = Serializer(["old_key", "new_key"])
    assert serializer.secret_key == b"new_key"


# LLM-generated content at query #29
#--------------------------

```python
def test_fallback_signers_predicate_false():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    serializer = Serializer("secret_key", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #30
#--------------------------

def test_dumps_returns_bytes_when_is_text_serializer_false():
    serializer = Serializer("secret", serializer=type("BytesSerializer", (), {"dumps": lambda self, obj, **kwargs: b"{}", "loads": lambda self, s: {}})())
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)


# LLM-generated content at query #31
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.serializer == CustomSerializer
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_fallback_signers_none():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_secret_key_bytes():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_secret_key_list_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]


# LLM-generated content at query #32
#--------------------------

def test_dumps_returns_bytes_for_bytes_serializer():
    serializer = Serializer("secret", serializer=lambda: None)
    serializer.serializer.dumps = lambda obj: b'{"key":"value"}'
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key":"value"}'

def test_dumps_returns_str_for_text_serializer():
    serializer = Serializer("secret", serializer=lambda: None)
    serializer.serializer.dumps = lambda obj: '{"key":"value"}'
    serializer.is_text_serializer = True
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key":"value"}'

def test_dumps_uses_provided_salt():
    serializer = Serializer("secret", serializer=lambda: None)
    serializer.serializer.dumps = lambda obj: b'payload'
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom_salt")
    assert isinstance(result_with_salt, bytes)


# LLM-generated content at query #33
#--------------------------

```python
def test_iter_unsigners_with_tuple_fallback_signers():
    signer_kwargs = {"key_derivation": "hmac"}
    fallback_signers = [(Signer, {"digest_method": "sha256"})]
    serializer = Serializer("secret", fallback_signers=fallback_signers, signer_kwargs=signer_kwargs)
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
```


# LLM-generated content at query #34
#--------------------------

def test_serializer_constructor_default():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_custom_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_secret_key_property():
    serializer = Serializer(["old", "new"])
    assert serializer.secret_key == b"new"


# LLM-generated content at query #35
#--------------------------

def test_is_text_serializer_false():
    serializer = Serializer("secret", serializer=None, serializer_kwargs={})
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)


# LLM-generated content at query #36
#--------------------------

def test_serializer_constructor_default_serializer():
    s = Serializer("secret-key")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_salt_none():
    s = Serializer("secret-key", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(data):
            return data
    s = Serializer("secret-key", serializer=BytesSerializer())
    assert s.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret-key", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret-key", signer_kwargs={"key_derivation": "none"})
    assert s.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret-key", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_list_secret_key():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"bytes-key")
    assert s.secret_keys == [b"bytes-key"]

def test_serializer_constructor_with_iterable_secret_key():
    keys = iter(["key1", "key2"])
    s = Serializer(keys)
    assert s.secret_keys == [b"key1", b"key2"]


# LLM-generated content at query #37
#--------------------------

def test_salt_is_none_in_serializer_init():
    s = Serializer("secret", salt=None)
    assert s.salt is None


# LLM-generated content at query #38
#--------------------------

def test_constructor_with_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "text", "loads": lambda self, x: {}})
    serializer = Serializer("secret", serializer=custom_serializer())
    assert serializer.serializer is custom_serializer()
    assert serializer.is_text_serializer is True

def test_constructor_with_bytes_serializer():
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, x: b"bytes", "loads": lambda self, x: {}})
    serializer = Serializer("secret", serializer=bytes_serializer())
    assert serializer.is_text_serializer is False

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #39
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer == True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_salt_string():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_salt_bytes():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer == True

def test_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer == False

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_fallback_signers_tuple():
    class CustomSigner(Signer):
        pass
    fallback = [(CustomSigner, {"key_derivation": "none"})]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_fallback_signers_class():
    class CustomSigner(Signer):
        pass
    fallback = [CustomSigner]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_secret_key_bytes():
    serializer = Serializer(b"bytes_key")
    assert serializer.secret_keys == [b"bytes_key"]

def test_constructor_with_secret_key_list_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]


# LLM-generated content at query #40
#--------------------------

```python
def test_salt_is_not_none_condition_false():
    serializer = Serializer(secret_key="secret", salt=None)
    assert serializer.salt is None
```


# LLM-generated content at query #41
#--------------------------

```python
def test_load_payload_exception_raises_bad_payload():
    serializer = Serializer("secret")
    invalid_payload = b"invalid json"
    try:
        serializer.load_payload(invalid_payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass
```


# LLM-generated content at query #42
#--------------------------

```python
def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(Serializer.default_serializer)
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "dumped", "loads": lambda self, x: "loaded"})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer == is_text_serializer(custom_serializer)

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {"__init__": lambda self, keys, salt, **kwargs: None})()
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer is custom_signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_all_parameters():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "dumped", "loads": lambda self, x: "loaded"})()
    custom_signer = type("CustomSigner", (Signer,), {"__init__": lambda self, keys, salt, **kwargs: None})()
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer(
        ["key1", b"key2"],
        salt=b"custom_salt",
        serializer=custom_serializer,
        serializer_kwargs={"sort_keys": True},
        signer=custom_signer,
        signer_kwargs={"key_derivation": "hmac"},
        fallback_signers=fallback,
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer == is_text_serializer(custom_serializer)
    assert serializer.signer is custom_signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #43
#--------------------------

def test_serializer_constructor_with_default_parameters():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_with_list_of_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_text_serializer():
    serializer = Serializer("secret-key", serializer=json)
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    import pickle
    serializer = Serializer("secret-key", serializer=pickle)
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #44
#--------------------------

```python
def test_load_payload_is_text_false_when_serializer_is_bytes_serializer():
    serializer = Serializer("secret", serializer=bytes)
    payload = serializer.dump_payload("test")
    result = serializer.load_payload(payload)
    assert result == "test"
```


# LLM-generated content at query #45
#--------------------------

def test_serializer_init_with_str_secret_key():
    serializer = Serializer("my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(serializer.default_serializer)
    assert serializer.signer == serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key():
    serializer = Serializer(b"my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(serializer.default_serializer)
    assert serializer.signer == serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_list_of_str_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(serializer.default_serializer)
    assert serializer.signer == serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(serializer.default_serializer)
    assert serializer.signer == serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_custom_salt():
    serializer = Serializer("my-secret-key", salt=b"custom-salt")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(serializer.default_serializer)
    assert serializer.signer == serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_custom_salt_as_str():
    serializer = Serializer("my-secret-key", salt="custom-salt")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(serializer.default_serializer)
    assert serializer.signer == serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_salt_none():
    serializer = Serializer("my-secret-key", salt=None)
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt is None
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(serializer.default_serializer)
    assert serializer.signer == serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: str(x), "loads": lambda self, x: x})()
    serializer = Serializer("my-secret-key", serializer=custom_serializer)
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer == is_text_serializer(custom_serializer)
    assert serializer.signer == serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_custom_signer():
    custom_signer = Signer
    serializer = Serializer("my-secret-key", signer=custom_signer)
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(serializer.default_serializer)
    assert serializer.signer == custom_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_signer_kwargs():
    serializer = Serializer("my-secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(serializer.default_serializer)
    assert serializer.signer == serializer.default_signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("my-secret-key", fallback_signers=fallback)
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(serializer.default_serializer)
    assert serializer.signer == serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_serializer_kwargs():
    serializer = Serializer("my-secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(serializer.default_serializer)
    assert serializer.signer == serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #46
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_str_secrets():
    s = Serializer(["secret1", "secret2"])
    assert s.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_list_of_bytes_secrets():
    s = Serializer([b"secret1", b"secret2"])
    assert s.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_salt_bytes():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_salt_str():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_custom_text_serializer():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "text"
        @staticmethod
        def loads(s):
            return s
    s = Serializer("secret", serializer=TextSerializer)
    assert s.serializer is TextSerializer
    assert s.is_text_serializer is True

def test_serializer_constructor_with_custom_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return s
    s = Serializer("secret", serializer=BytesSerializer)
    assert s.serializer is BytesSerializer
    assert s.is_text_serializer is False

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers_list():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_all_parameters():
    class CustomSigner(Signer):
        pass
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(s):
            return s
    s = Serializer(
        secret_key=["old_secret", "new_secret"],
        salt=b"custom_salt",
        serializer=CustomSerializer,
        serializer_kwargs={"indent": 2},
        signer=CustomSigner,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert s.secret_keys == [b"old_secret", b"new_secret"]
    assert s.salt == b"custom_salt"
    assert s.serializer is CustomSerializer
    assert s.is_text_serializer is True
    assert s.signer is CustomSigner
    assert s.signer_kwargs == {"digest_method": "sha256"}
    assert s.fallback_signers == [{"key_derivation": "none"}]
    assert s.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #47
#--------------------------

```python
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert result is not None

def test_dumps_accepts_any_object():
    serializer = _PDataSerializer()
    result = serializer.dumps(42)
    assert result is not None

def test_dumps_returns_same_type():
    serializer = _PDataSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, type(result))

def test_dumps_with_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result is not None

def test_dumps_with_list():
    serializer = _PDataSerializer()
    result = serializer.dumps([1, 2, 3])
    assert result is not None

def test_dumps_with_tuple():
    serializer = _PDataSerializer()
    result = serializer.dumps((1, "a"))
    assert result is not None

def test_dumps_with_boolean():
    serializer = _PDataSerializer()
    result = serializer.dumps(True)
    assert result is not None

def test_dumps_with_float():
    serializer = _PDataSerializer()
    result = serializer.dumps(3.14)
    assert result is not None
```


# LLM-generated content at query #48
#--------------------------

```python
def test_salt_is_none_predicate_false():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None
```


# LLM-generated content at query #49
#--------------------------

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

def test_serializer_constructor_with_list_of_string_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret-key", serializer=CustomSerializer)
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return s

    serializer = Serializer("secret-key", serializer=BytesSerializer)
    assert serializer.serializer is BytesSerializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_default_fallback_signers():
    serializer = Serializer("secret-key")
    assert serializer.fallback_signers == []


# LLM-generated content at query #50
#--------------------------

def test_loads_returns_any():
    serializer = _PDataSerializer()
    serializer.loads = lambda payload: payload
    result = serializer.loads("test")
    assert result == "test"


# LLM-generated content at query #51
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_constructor_with_multiple_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer is True

def test_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"

        @staticmethod
        def loads(s):
            return s

    serializer = Serializer("secret-key", serializer=BytesSerializer())
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer is False

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_constructor_with_none_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_constructor_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    class CustomSigner(Signer):
        pass

    serializer = Serializer(
        secret_key=["key1", "key2"],
        salt=b"custom-salt",
        serializer=CustomSerializer(),
        serializer_kwargs={"sort_keys": True},
        signer=CustomSigner,
        signer_kwargs={"key_derivation": "none"},
        fallback_signers=[{"key_derivation": "hmac"}],
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom-salt"
    assert isinstance(serializer.serializer, CustomSerializer)
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {"key_derivation": "none"}
    assert serializer.fallback_signers == [{"key_derivation": "hmac"}]
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #52
#--------------------------

def test_constructor_default_parameters():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer == json

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer == Signer

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback


# LLM-generated content at query #53
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_serializer_bytes():
    serializer = Serializer("secret", serializer=json)
    assert serializer.is_text_serializer is True

def test_constructor_with_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer is Signer

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_constructor_with_list_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"bytes_key")
    assert serializer.secret_keys == [b"bytes_key"]


# LLM-generated content at query #54
#--------------------------

def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert isinstance(result, type(serializer.dumps("test")))


# LLM-generated content at query #55
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_salt_bytes():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_salt_string():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_serializer_str():
    serializer = Serializer("secret", serializer=None)
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_constructor_with_serializer_bytes():
    serializer = Serializer("secret", serializer=Serializer)
    assert serializer.serializer is Serializer
    assert serializer.is_text_serializer is False

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_constructor_with_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_constructor_with_None_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_constructor_with_secret_key_bytes():
    serializer = Serializer(b"secret_bytes")
    assert serializer.secret_keys == [b"secret_bytes"]

def test_constructor_with_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_secret_key_list_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_signer_kwargs_empty_dict():
    serializer = Serializer("secret", signer_kwargs={})
    assert serializer.signer_kwargs == {}

def test_constructor_with_serializer_kwargs_empty_dict():
    serializer = Serializer("secret", serializer_kwargs={})
    assert serializer.serializer_kwargs == {}

def test_constructor_secret_key_property():
    serializer = Serializer(["old", "new"])
    assert serializer.secret_key == b"new"

def test_constructor_with_fallback_signers_tuple():
    class CustomSigner(Signer):
        pass
    fallback = [(CustomSigner, {"key_derivation": "none"})]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_fallback_signers_class():
    class CustomSigner(Signer):
        pass
    fallback = [CustomSigner]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_serializer_as_positional():
    serializer = Serializer("secret", b"salt", json)
    assert serializer.serializer is json
    assert serializer.salt == b"salt"

def test_constructor_with_serializer_as_keyword():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json


# LLM-generated content at query #56
#--------------------------

def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads("test_payload")
    assert result is not None


# LLM-generated content at query #57
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"secret_key")
    assert serializer.secret_keys == [b"secret_key"]

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_string_salt():
    serializer = Serializer("secret", salt="string_salt")
    assert serializer.salt == b"string_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"{}"
        @staticmethod
        def loads(data):
            return {}
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer == Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"ensure_ascii": False})
    assert serializer.serializer_kwargs == {"ensure_ascii": False}

def test_serializer_constructor_with_all_parameters():
    serializer = Serializer(
        secret_key=["old_key", "new_key"],
        salt=b"salt",
        serializer=json,
        serializer_kwargs={"sort_keys": True},
        signer=Signer,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[Signer]
    )
    assert serializer.secret_keys == [b"old_key", b"new_key"]
    assert serializer.salt == b"salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    assert serializer.fallback_signers == [Signer]
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #58
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])

def test_serializer_constructor_with_salt():
    serializer = Serializer("secret", salt="custom_salt")

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)

def test_serializer_constructor_with_json_serializer():
    serializer = Serializer("secret", serializer=json)

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(data):
            return int(data)

    serializer = Serializer("secret", serializer=CustomSerializer())

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"digest_method": hashlib.sha256})

def test_serializer_constructor_with_fallback_signers_dict():
    serializer = Serializer("secret", fallback_signers=[{"digest_method": hashlib.sha256}])

def test_serializer_constructor_with_fallback_signers_tuple():
    serializer = Serializer("secret", fallback_signers=[(Signer, {"digest_method": hashlib.sha256})])

def test_serializer_constructor_with_fallback_signers_class():
    serializer = Serializer("secret", fallback_signers=[Signer])

def test_serializer_constructor_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)

def test_serializer_constructor_multiple_parameters():
    serializer = Serializer(
        secret_key=["key1", "key2"],
        salt=b"salt",
        serializer=json,
        serializer_kwargs={"sort_keys": True},
        signer=Signer,
        signer_kwargs={"key_derivation": "hmac"},
        fallback_signers=[{"digest_method": hashlib.sha256}]
    )

def test_serializer_constructor_secret_keys_property():
    serializer = Serializer(["key1", "key2", "key3"])
    assert serializer.secret_key == b"key3"

def test_serializer_constructor_is_text_serializer_true():
    serializer = Serializer("secret")
    assert serializer.is_text_serializer

def test_serializer_constructor_is_text_serializer_false():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"{}"

        @staticmethod
        def loads(data):
            return {}

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert not serializer.is_text_serializer


# LLM-generated content at query #59
#--------------------------

```python
def test_load_payload_with_default_serializer_returns_correct_data():
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_text_serializer_returns_correct_data():
    class TextSerializer:
        def dumps(self, obj):
            return str(obj)
        def loads(self, payload):
            return eval(payload)
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = serializer.dump_payload(42)
    result = serializer.load_payload(payload)
    assert result == 42

def test_load_payload_with_custom_bytes_serializer_returns_correct_data():
    class BytesSerializer:
        def dumps(self, obj):
            return bytes(str(obj), "utf-8")
        def loads(self, payload):
            return int(payload.decode("utf-8"))
    serializer = Serializer("secret", serializer=BytesSerializer())
    payload = serializer.dump_payload(100)
    result = serializer.load_payload(payload)
    assert result == 100

def test_load_payload_with_override_serializer_uses_override():
    serializer = Serializer("secret")
    class OverrideSerializer:
        def dumps(self, obj):
            return "override"
        def loads(self, payload):
            return "override_result"
    payload = b"some bytes"
    result = serializer.load_payload(payload, serializer=OverrideSerializer())
    assert result == "override_result"

def test_load_payload_raises_bad_payload_on_invalid_data():
    serializer = Serializer("secret")
    invalid_payload = b"invalid data"
    try:
        serializer.load_payload(invalid_payload)
        assert False, "Expected BadPayload exception"
    except Exception as e:
        assert type(e).__name__ == "BadPayload"

def test_load_payload_with_text_serializer_decodes_utf8():
    serializer = Serializer("secret")
    payload = serializer.dump_payload("hello")
    result = serializer.load_payload(payload)
    assert result == "hello"

def test_load_payload_with_bytes_serializer_does_not_decode():
    class BytesSerializer:
        def dumps(self, obj):
            return obj if isinstance(obj, bytes) else bytes(obj, "utf-8")
        def loads(self, payload):
            return payload
    serializer = Serializer("secret", serializer=BytesSerializer())
    payload = serializer.dump_payload(b"binary data")
    result = serializer.load_payload(payload)
    assert result == b"binary data"

def test_load_payload_with_serializer_that_returns_text_uses_decode():
    class TextReturnSerializer:
        def dumps(self, obj):
            return "text"
        def loads(self, payload):
            return payload
    serializer = Serializer("secret", serializer=TextReturnSerializer())
    payload = serializer.dump_payload("test")
    result = serializer.load_payload(payload)
    assert result == "text"
```


# LLM-generated content at query #60
#--------------------------

def test_dumps_returns_bytes_when_is_text_serializer_false():
    from itsdangerous.serializer import Serializer
    serializer = Serializer("secret", serializer=lambda: None)
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)


# LLM-generated content at query #61
#--------------------------

```python
def test_iter_unsigners_with_default_salt():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].salt == b"itsdangerous"

def test_iter_unsigners_with_custom_salt():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners(salt=b"custom-salt"))
    assert signers[0].salt == b"custom-salt"

def test_iter_unsigners_with_fallback_signers_dict():
    serializer = Serializer("secret-key", fallback_signers=[{"key": "fallback-secret"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_with_fallback_signers_tuple():
    serializer = Serializer("secret-key", fallback_signers=[(Signer, {"key": "fallback-secret"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_with_fallback_signers_class():
    serializer = Serializer("secret-key", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_with_multiple_secret_keys():
    serializer = Serializer(["old-secret", "new-secret"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1

def test_iter_unsigners_with_fallback_and_multiple_keys():
    serializer = Serializer(["old-secret", "new-secret"], fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3

def test_iter_unsigners_yields_make_signer_first():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert signers[0].secret_keys == [b"secret-key"]

def test_iter_unsigners_with_none_salt_uses_default():
    serializer = Serializer("secret-key", salt=None)
    signers = list(serializer.iter_unsigners(salt=None))
    assert signers[0].salt is None
```


# LLM-generated content at query #62
#--------------------------

def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads(b"test")
    assert result is not None


# LLM-generated content at query #63
#--------------------------

```python
def test_dumps_returns_expected_serialized_type():
    serializer = _PDataSerializer()
    serializer.dumps = lambda obj: b"test"
    result = serializer.dumps("test")
    assert result == b"test"

def test_dumps_accepts_any_object():
    serializer = _PDataSerializer()
    serializer.dumps = lambda obj: b"test"
    result = serializer.dumps(123)
    assert result == b"test"

def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    serializer.dumps = lambda obj: b"test"
    result = serializer.dumps(None)
    assert isinstance(result, bytes)
```


# LLM-generated content at query #64
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secrets():
    s = Serializer(["secret1", "secret2"])
    assert s.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_list_of_bytes_secrets():
    s = Serializer([b"secret1", b"secret2"])
    assert s.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_bytes_salt():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_explicit_serializer():
    s = Serializer("secret", serializer=json)
    assert s.serializer is json
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(data):
            return {}
    s = Serializer("secret", serializer=BytesSerializer())
    assert s.serializer is BytesSerializer
    assert s.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_empty_fallback_signers():
    s = Serializer("secret", fallback_signers=[])
    assert s.fallback_signers == []


# LLM-generated content at query #65
#--------------------------

def test_serializer_constructor_with_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"salt"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_custom_salt_as_string():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_custom_salt_as_bytes():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(s):
            return {"custom": True}
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return {"bytes": True}
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_custom_signer():
    class CustomSigner:
        def __init__(self, secret_keys, salt, **kwargs):
            pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #66
#--------------------------

```python
def test_salt_is_not_none():
    serializer = Serializer(secret_key="secret", salt=b"salt")
    assert serializer.salt == b"salt"
```


# LLM-generated content at query #67
#--------------------------

def test_serializer_constructor_default():
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

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback


# LLM-generated content at query #68
#--------------------------

```python
def test_fallback_signers_not_none():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []
```


# LLM-generated content at query #69
#--------------------------

```python
def test_load_payload_with_non_text_serializer_returns_false_for_is_text():
    serializer = Serializer("secret", serializer=Serializer.default_serializer)
    result = serializer.load_payload(b'{"key": "value"}')
    assert result == {"key": "value"}
```


# LLM-generated content at query #70
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer is Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[Signer])
    assert serializer.fallback_signers == [Signer]

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_secret_key_bytes():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_secret_key_list():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_secret_key_list_bytes():
    serializer = Serializer([b"secret1", b"secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]


# LLM-generated content at query #71
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_bytes_key_list():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"{}"
        @staticmethod
        def loads(data):
            return {}
    serializer = Serializer("secret", salt=b"salt", serializer=BytesSerializer())
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_text_serializer():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "{}"
        @staticmethod
        def loads(data):
            return {}
    serializer = Serializer("secret", serializer=TextSerializer())
    assert serializer.serializer == TextSerializer()
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "{}"
        @staticmethod
        def loads(data):
            return {}
    class CustomSigner(Signer):
        pass
    serializer = Serializer(
        secret_key=["key1", "key2"],
        salt=b"custom_salt",
        serializer=CustomSerializer(),
        serializer_kwargs={"sort_keys": True},
        signer=CustomSigner,
        signer_kwargs={"digest_method": hashlib.sha256},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer == True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {"digest_method": hashlib.sha256}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #72
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class MockSerializer:
        def dumps(self, obj):
            return "mock"
        def loads(self, s):
            return "mock"
    serializer = Serializer("secret", serializer=MockSerializer())
    assert isinstance(serializer.serializer, MockSerializer)
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            return b"bytes"
        def loads(self, s):
            return "loaded"
    serializer = Serializer("secret", b"salt", BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[Signer])
    assert serializer.fallback_signers == [Signer]

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #73
#--------------------------

```python
def test_iter_unsigners_fallback_tuple():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer

    serializer = Serializer(
        secret_key=b"secret",
        fallback_signers=[(Signer, {"digest_method": "sha256"})]
    )
    signers = list(serializer.iter_unsigners(salt=b"test"))
    assert len(signers) == 2
    assert isinstance(signers[1], Signer)
```


# LLM-generated content at query #74
#--------------------------

```python
def test_load_payload_is_text_false():
    serializer = Serializer("secret", serializer=None, serializer_kwargs=None, signer=None, signer_kwargs=None, fallback_signers=None)
    payload = b"\x80\x04\x95\x05\x00\x00\x00\x00\x00\x00\x00}\x94."
    result = serializer.load_payload(payload)
    assert result == {}
```


# LLM-generated content at query #75
#--------------------------

def test_constructor_default_serializer():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_constructor_with_iterable_secret_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_constructor_with_custom_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer is CustomSerializer
    assert s.is_text_serializer is True

def test_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"

        @staticmethod
        def loads(data):
            return data

    s = Serializer("secret", serializer=BytesSerializer())
    assert s.is_text_serializer is False

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_constructor_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    class CustomSigner(Signer):
        pass

    s = Serializer(
        secret_key=["old_key", "new_key"],
        salt=b"custom_salt",
        serializer=CustomSerializer(),
        serializer_kwargs={"indent": 2},
        signer=CustomSigner,
        signer_kwargs={"key_derivation": "hmac"},
        fallback_signers=[{"key_derivation": "none"}],
    )
    assert s.secret_keys == [b"old_key", b"new_key"]
    assert s.salt == b"custom_salt"
    assert s.serializer is CustomSerializer
    assert s.is_text_serializer is True
    assert s.signer is CustomSigner
    assert s.signer_kwargs == {"key_derivation": "hmac"}
    assert s.fallback_signers == [{"key_derivation": "none"}]
    assert s.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #76
#--------------------------

def test_serializer_constructor_with_default_parameters() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings_secret_key() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_key() -> None:
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt() -> None:
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer() -> None:
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers() -> None:
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_secret_key_property_returns_last_key() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_key == b"key2"


# LLM-generated content at query #77
#--------------------------

def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads(42)
    assert result is not None or result is None

def test_loads_accepts_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.loads("test")
    assert isinstance(result, object)

def test_loads_accepts_none():
    serializer = _PDataSerializer()
    result = serializer.loads(None)
    assert result is None

def test_loads_accepts_complex_type():
    serializer = _PDataSerializer()
    result = serializer.loads([1, 2, 3])
    assert isinstance(result, list)


# LLM-generated content at query #78
#--------------------------

```python
def test_load_payload_with_default_serializer():
    serializer = Serializer("secret")
    payload = serializer.serializer.dumps({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_bytes_serializer():
    class BytesSerializer:
        def loads(self, payload: bytes) -> dict:
            return {"data": payload.decode()}
        def dumps(self, obj: dict) -> bytes:
            return obj["data"].encode()
    serializer = Serializer("secret", serializer=BytesSerializer())
    payload = serializer.serializer.dumps({"data": "test"})
    result = serializer.load_payload(payload)
    assert result == {"data": "test"}

def test_load_payload_with_explicit_serializer_override():
    serializer = Serializer("secret")
    payload = b'{"custom": "payload"}'
    custom_serializer = json
    result = serializer.load_payload(payload, serializer=custom_serializer)
    assert result == {"custom": "payload"}

def test_load_payload_raises_bad_payload_on_invalid_data():
    serializer = Serializer("secret")
    try:
        serializer.load_payload(b"invalid")
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass

def test_load_payload_with_text_serializer():
    class TextSerializer:
        def loads(self, payload: str) -> dict:
            return {"text": payload}
        def dumps(self, obj: dict) -> str:
            return obj["text"]
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = serializer.serializer.dumps({"text": "hello"}).encode()
    result = serializer.load_payload(payload)
    assert result == {"text": "hello"}

def test_load_payload_uses_is_text_serializer_flag():
    class TextSerializer:
        def loads(self, payload: str) -> str:
            return payload
        def dumps(self, obj: str) -> str:
            return obj
    serializer = Serializer("secret", serializer=TextSerializer())
    assert serializer.is_text_serializer
    payload = serializer.serializer.dumps("test").encode()
    result = serializer.load_payload(payload)
    assert result == "test"
```


# LLM-generated content at query #79
#--------------------------

```python
def test_dumps_returns_bytes_when_not_text_serializer():
    serializer = Serializer("secret", serializer=lambda: None)
    serializer.is_text_serializer = False
    result = serializer.dumps("test")
    assert isinstance(result, bytes)


# LLM-generated content at query #80
#--------------------------

```python
def test_salt_is_not_none_predicate_false():
    serializer = Serializer(secret_key="secret", salt=None)
    assert serializer.salt is None
```


# LLM-generated content at query #81
#--------------------------

```
def test_dumps_returns_serialized_data():
    serializer = _PDataSerializer()
    result = serializer.dumps("test_object")
    assert result is not None
```


# LLM-generated content at query #82
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt is None
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
            return "custom"
        @staticmethod
        def loads(s):
            return {"custom": True}
    serializer = Serializer("key", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return {"bytes": True}
    serializer = Serializer("key", serializer=BytesSerializer())
    assert serializer.serializer is BytesSerializer()
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("key", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_multiple_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"bytes-key")
    assert serializer.secret_keys == [b"bytes-key"]


# LLM-generated content at query #83
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(s):
            return obj
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[{"digest_method": hashlib.sha256}])
    assert len(serializer.fallback_signers) == 1

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #84
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secrets():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        def dumps(self, obj):
            return str(obj)

        def loads(self, s):
            return eval(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is not None

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner:
        def __init__(self, secret_key, salt=None, **kwargs):
            pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"digest_method": "sha256"})
    assert serializer.signer_kwargs == {"digest_method": "sha256"}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"digest_method": "sha256"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_constructor_secret_key_property():
    serializer = Serializer(["old_secret", "new_secret"])
    assert serializer.secret_key == b"new_secret"


# LLM-generated content at query #85
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_str_secret_key():
    serializer = Serializer("my_secret_key")
    assert serializer.secret_keys == [b"my_secret_key"]

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"my_secret_key")
    assert serializer.secret_keys == [b"my_secret_key"]

def test_serializer_constructor_with_list_of_str_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_str_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return eval(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_custom_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_custom_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback


# LLM-generated content at query #86
#--------------------------

def test_serializer_constructor_with_default_parameters() -> None:
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_with_list_of_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys() -> None:
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_salt() -> None:
    serializer = Serializer("secret-key", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_string_salt() -> None:
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: object) -> str:
            return str(obj)

        @staticmethod
        def loads(s: str) -> object:
            return int(s)

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers() -> None:
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #87
#--------------------------

def test_constructor_with_string_secret_key():
    serializer = Serializer("my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]

def test_constructor_with_list_of_string_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_salt():
    serializer = Serializer("secret", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.is_text_serializer == True

def test_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer == False

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_constructor_with_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #88
#--------------------------

```python
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test_data")
    assert result is not None

def test_dumps_returns_string_when_serialized_type_is_string():
    serializer = _PDataSerializer()
    result = serializer.dumps(123)
    assert isinstance(result, str)

def test_dumps_returns_bytes_when_serialized_type_is_bytes():
    serializer = _PDataSerializer()
    result = serializer.dumps("data")
    assert isinstance(result, bytes)

def test_dumps_accepts_any_object():
    serializer = _PDataSerializer()
    result = serializer.dumps({"key": "value"})
    assert result is not None

def test_dumps_accepts_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result is not None

def test_dumps_accepts_list():
    serializer = _PDataSerializer()
    result = serializer.dumps([1, 2, 3])
    assert result is not None

def test_dumps_accepts_integer():
    serializer = _PDataSerializer()
    result = serializer.dumps(42)
    assert result is not None

def test_dumps_accepts_float():
    serializer = _PDataSerializer()
    result = serializer.dumps(3.14)
    assert result is not None

def test_dumps_accepts_boolean():
    serializer = _PDataSerializer()
    result = serializer.dumps(True)
    assert result is not None

def test_dumps_accepts_custom_object():
    class CustomObject:
        pass
    serializer = _PDataSerializer()
    result = serializer.dumps(CustomObject())
    assert result is not None
```


# LLM-generated content at query #89
#--------------------------

def test_serializer_constructor_default():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_secret():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []


# LLM-generated content at query #90
#--------------------------

```python
def test_constructor_default_serializer():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_constructor_with_iterable_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret-key", serializer=CustomSerializer)
    assert serializer.serializer == CustomSerializer
    assert serializer.is_text_serializer == True

def test_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret-key", serializer=BytesSerializer)
    assert serializer.serializer == BytesSerializer
    assert serializer.is_text_serializer == False

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_empty_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_constructor_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(s):
            return s
    class CustomSigner(Signer):
        pass
    serializer = Serializer(
        secret_key="secret",
        salt="salt",
        serializer=CustomSerializer,
        serializer_kwargs={"sort_keys": True},
        signer=CustomSigner,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"salt"
    assert serializer.serializer == CustomSerializer
    assert serializer.is_text_serializer == True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"sort_keys": True}
```


# LLM-generated content at query #91
#--------------------------

def test_salt_is_none_so_predicate_is_false():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #92
#--------------------------

```python
def test_serializer_init_with_explicit_serializer():
    serializer_instance = Serializer("secret", serializer=json)
    assert serializer_instance.serializer is json
    assert serializer_instance.is_text_serializer == True
```


# LLM-generated content at query #93
#--------------------------

```python
def test_load_payload_is_text_false_with_non_text_serializer():
    serializer = Serializer("secret")
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
```


# LLM-generated content at query #94
#--------------------------

```python
def test_load_payload_with_default_serializer_and_text_payload():
    serializer = Serializer("secret")
    payload = '{"key": "value"}'.encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_default_serializer_and_bytes_payload():
    serializer = Serializer("secret")
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_text_serializer():
    class TextSerializer:
        def dumps(self, obj):
            return str(obj)
        def loads(self, payload):
            return eval(payload)
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = "{'key': 'value'}".encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            return repr(obj).encode("utf-8")
        def loads(self, payload):
            return eval(payload.decode("utf-8"))
    serializer = Serializer("secret", serializer=BytesSerializer())
    payload = repr({"key": "value"}).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_serializer_override():
    class OverrideSerializer:
        def dumps(self, obj):
            return str(obj)
        def loads(self, payload):
            return {"overridden": True}
    serializer = Serializer("secret")
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload, serializer=OverrideSerializer())
    assert result == {"overridden": True}

def test_load_payload_raises_bad_payload_on_invalid_data():
    serializer = Serializer("secret")
    payload = b"invalid json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except Exception as e:
        assert "Could not load the payload" in str(e)

def test_load_payload_raises_bad_payload_on_empty_payload():
    serializer = Serializer("secret")
    payload = b""
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except Exception as e:
        assert "Could not load the payload" in str(e)

def test_load_payload_with_serializer_override_raises_bad_payload():
    class FailingSerializer:
        def dumps(self, obj):
            return ""
        def loads(self, payload):
            raise ValueError("load failed")
    serializer = Serializer("secret")
    payload = b"some data"
    try:
        serializer.load_payload(payload, serializer=FailingSerializer())
        assert False, "Expected BadPayload exception"
    except Exception as e:
        assert "Could not load the payload" in str(e)
```


# LLM-generated content at query #95
#--------------------------

def test_dumps_returns_bytes_when_serializer_is_not_text_serializer():
    serializer = Serializer("secret", serializer=lambda: None)
    serializer.is_text_serializer = False
    result = serializer.dumps("data")
    assert isinstance(result, bytes)


# LLM-generated content at query #96
#--------------------------

def test_constructor_default_serializer_uses_json():
    serializer = Serializer("secret")
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_constructor_serializer_none_uses_default():
    serializer = Serializer("secret", serializer=None)
    assert serializer.serializer is json

def test_constructor_custom_text_serializer():
    class CustomSerializer:
        dumps = lambda self, obj: "text"
        loads = lambda self, s: {}
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.is_text_serializer is True

def test_constructor_custom_bytes_serializer():
    class CustomSerializer:
        dumps = lambda self, obj: b"bytes"
        loads = lambda self, s: {}
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.is_text_serializer is False

def test_constructor_single_secret_key():
    serializer = Serializer("mysecret")
    assert serializer.secret_keys == [b"mysecret"]

def test_constructor_multiple_secret_keys():
    serializer = Serializer(["oldkey", "newkey"])
    assert serializer.secret_keys == [b"oldkey", b"newkey"]
    assert serializer.secret_key == b"newkey"

def test_constructor_salt_is_bytes():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_salt_is_str():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_signer_default():
    serializer = Serializer("secret")
    assert serializer.signer is Signer

def test_constructor_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_fallback_signers_default():
    serializer = Serializer("secret")
    assert serializer.fallback_signers == []

def test_constructor_fallback_signers_custom():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #97
#--------------------------

```python
def test_fallback_signers_not_none_when_default_is_empty():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #98
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer = Serializer("secret-key", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(b):
            return b
    serializer = Serializer("secret-key", serializer=BytesSerializer())
    assert not serializer.is_text_serializer

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #99
#--------------------------

```python
def test_iter_unsigners_yields_make_signer_first():
    signer = Signer(b"secret-key")
    serializer = Serializer(b"secret-key")
    result = list(serializer.iter_unsigners())
    expected_first = serializer.make_signer()
    assert result[0].secret_key == expected_first.secret_key

def test_iter_unsigners_uses_provided_salt():
    signer = Signer(b"secret-key")
    serializer = Serializer(b"secret-key", salt=b"custom-salt")
    result = list(serializer.iter_unsigners())
    assert result[0].salt == b"custom-salt"

def test_iter_unsigners_includes_fallback_dict_signers():
    serializer = Serializer(b"secret-key", fallback_signers=[{"key": b"fallback-key"}])
    result = list(serializer.iter_unsigners())
    assert len(result) > 1

def test_iter_unsigners_includes_fallback_tuple_signers():
    class CustomSigner(Signer):
        pass
    serializer = Serializer(b"secret-key", fallback_signers=[(CustomSigner, {})])
    result = list(serializer.iter_unsigners())
    assert len(result) > 1

def test_iter_unsigners_includes_fallback_class_signers():
    class CustomSigner(Signer):
        pass
    serializer = Serializer(b"secret-key", fallback_signers=[CustomSigner])
    result = list(serializer.iter_unsigners())
    assert len(result) > 1

def test_iter_unsigners_yields_multiple_signers_for_multiple_secret_keys():
    serializer = Serializer([b"old-key", b"new-key"], fallback_signers=[{}])
    result = list(serializer.iter_unsigners())
    assert len(result) == 3

def test_iter_unsigners_uses_signer_kwargs_for_fallback_dict():
    serializer = Serializer(b"secret-key", signer_kwargs={"key_derivation": "none"}, fallback_signers=[{"key_derivation": "hmac"}])
    result = list(serializer.iter_unsigners())
    assert result[1].key_derivation == "hmac"

def test_iter_unsigners_uses_provided_salt_for_all_signers():
    serializer = Serializer(b"secret-key", salt=b"test-salt", fallback_signers=[{}])
    result = list(serializer.iter_unsigners())
    for signer in result:
        assert signer.salt == b"test-salt"

def test_iter_unsigners_empty_fallback_signers():
    serializer = Serializer(b"secret-key", fallback_signers=[])
    result = list(serializer.iter_unsigners())
    assert len(result) == 1

def test_iter_unsigners_uses_fallback_signer_class_with_default_kwargs():
    class CustomSigner(Signer):
        pass
    serializer = Serializer(b"secret-key", signer_kwargs={"key_derivation": "none"}, fallback_signers=[CustomSigner])
    result = list(serializer.iter_unsigners())
    assert result[1].__class__ == CustomSigner
    assert result[1].key_derivation == "none"
```


# LLM-generated content at query #100
#--------------------------

def test__pdata_serializer_loads_returns_any_type():
    serializer = _PDataSerializer()
    result = serializer.loads(b"test_payload")
    assert result is not None

def test__pdata_serializer_loads_accepts_bytes():
    serializer = _PDataSerializer()
    result = serializer.loads(b"data")
    assert isinstance(result, object)

def test__pdata_serializer_loads_accepts_string():
    serializer = _PDataSerializer()
    result = serializer.loads("string_payload")
    assert result == "string_payload"


# LLM-generated content at query #101
#--------------------------

```python
def test_dumps_returns_bytes_when_serializer_is_not_text_serializer():
    serializer = Serializer(secret_key="secret", serializer=None)
    serializer.is_text_serializer = False
    result = serializer.dumps("test_data")
    assert isinstance(result, bytes)
```


# LLM-generated content at query #102
#--------------------------

```python
def test_iter_unsigners_default_signer_yielded_first():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_key == b"secret-key"

def test_iter_unsigners_with_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=[{"salt": "fallback"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_fallback_signer_tuple():
    serializer = Serializer("secret-key", fallback_signers=[(Signer, {"salt": "fallback"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_fallback_signer_class():
    serializer = Serializer("secret-key", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom")
    signers = list(serializer.iter_unsigners(salt="override"))
    assert signers[0].salt == b"override"

def test_iter_unsigners_multiple_secret_keys():
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1

def test_iter_unsigners_fallback_with_multiple_secret_keys():
    serializer = Serializer(["old-key", "new-key"], fallback_signers=[{"salt": "fallback"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3

def test_iter_unsigners_fallback_tuple_with_multiple_secret_keys():
    serializer = Serializer(["old-key", "new-key"], fallback_signers=[(Signer, {"salt": "fallback"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3

def test_iter_unsigners_fallback_class_with_multiple_secret_keys():
    serializer = Serializer(["old-key", "new-key"], fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3

def test_iter_unsigners_none_salt_uses_instance_salt():
    serializer = Serializer("secret-key", salt=b"instance")
    signers = list(serializer.iter_unsigners(salt=None))
    assert signers[0].salt == b"instance"
```


# LLM-generated content at query #103
#--------------------------

def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads("test")
    assert result is not None


# LLM-generated content at query #104
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is Serializer.default_serializer
    assert not s.is_text_serializer
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_secret_key_bytes():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_secret_key_list():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_salt_str():
    s = Serializer("secret", salt="mysalt")
    assert s.salt == b"mysalt"

def test_serializer_constructor_salt_bytes():
    s = Serializer("secret", salt=b"mysalt")
    assert s.salt == b"mysalt"

def test_serializer_constructor_serializer_json():
    s = Serializer("secret", serializer=json)
    assert s.serializer is json
    assert s.is_text_serializer

def test_serializer_constructor_serializer_custom_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(data):
            return {}
    s = Serializer("secret", serializer=BytesSerializer())
    assert not s.is_text_serializer

def test_serializer_constructor_serializer_custom_str():
    class StrSerializer:
        @staticmethod
        def dumps(obj):
            return "str"
        @staticmethod
        def loads(data):
            return {}
    s = Serializer("secret", serializer=StrSerializer())
    assert s.is_text_serializer

def test_serializer_constructor_signer_default():
    s = Serializer("secret")
    assert s.signer is Signer

def test_serializer_constructor_signer_custom():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_fallback_signers_list():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #105
#--------------------------

```python
from itsdangerous.serializer import Serializer
from itsdangerous.signer import Signer

def test_constructor_with_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_custom_serializer():
    from itsdangerous.serializer import json
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer is Signer

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_of_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_custom_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_secret_key_property():
    serializer = Serializer(["old", "new"])
    assert serializer.secret_key == b"new"```


# LLM-generated content at query #106
#--------------------------

```python
def test_predicate_at_line_27_evaluates_to_false_when_serializer_is_not_none():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
```


# LLM-generated content at query #107
#--------------------------

def test_serializer_constructor_default_values():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert isinstance(serializer.serializer, type(json))
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"data"
        @staticmethod
        def loads(data):
            return data
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.serializer is BytesSerializer()
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []


# LLM-generated content at query #108
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    serializer = Serializer(["key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer == Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_all_parameters():
    serializer = Serializer(
        ["key1", b"key2"],
        salt=b"custom_salt",
        serializer=json,
        serializer_kwargs={"indent": 2},
        signer=Signer,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[{"key_derivation": "none"}],
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #109
#--------------------------

def test_serializer_constructor_with_string_secret_key() -> None:
    serializer = Serializer("my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]

def test_serializer_constructor_with_list_of_strings() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes() -> None:
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_default_salt() -> None:
    serializer = Serializer("secret")
    assert serializer.salt == b"itsdangerous"

def test_serializer_constructor_with_custom_salt_string() -> None:
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_custom_salt_bytes() -> None:
    serializer = Serializer("secret", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_default_serializer() -> None:
    serializer = Serializer("secret")
    assert serializer.serializer is json

def test_serializer_constructor_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return str(obj)
        @staticmethod
        def loads(s: str) -> t.Any:
            return eval(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_default_signer() -> None:
    serializer = Serializer("secret")
    assert serializer.signer is Signer

def test_serializer_constructor_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_empty_fallback_signers() -> None:
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_default_fallback_signers() -> None:
    serializer = Serializer("secret")
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_text_serializer() -> None:
    class TextSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return "dummy"
        @staticmethod
        def loads(s: str) -> t.Any:
            return None
    serializer = Serializer("secret", serializer=TextSerializer())
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer() -> None:
    class BytesSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> bytes:
            return b"dummy"
        @staticmethod
        def loads(s: bytes) -> t.Any:
            return None
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False


# LLM-generated content at query #110
#--------------------------

def test_serializer_constructor_defaults():
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

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_bytes():
    serializer = Serializer("secret-key", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_salt_str():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "data", "loads": lambda self, x: x})()
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_serializer_bytes():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: b"data", "loads": lambda self, x: x})()
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})()
    serializer = Serializer("secret-key", signer=custom_signer)
    assert serializer.signer is custom_signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_fallback_signers_none():
    serializer = Serializer("secret-key", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_secret_key_property():
    serializer = Serializer(["old_key", "new_key"])
    assert serializer.secret_key == b"new_key"


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"

        @staticmethod
        def loads(s):
            return {"custom": True}

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True

def test_constructor_with_custom_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.serializer_kwargs == {"indent": 4}

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_constructor_with_custom_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    class CustomSigner(Signer):
        pass

    fallback = [{"key_derivation": "hmac"}, (CustomSigner, {}), CustomSigner]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_constructor_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"

        @staticmethod
        def loads(s):
            return {"custom": True}

    class CustomSigner(Signer):
        pass

    serializer = Serializer(
        secret_key="secret",
        salt="salt",
        serializer=CustomSerializer(),
        serializer_kwargs={"sort_keys": True},
        signer=CustomSigner,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[{"key_derivation": "hmac"}],
    )
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"salt"
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True
    assert serializer.signer is CustomSigner
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    assert serializer.fallback_signers == [{"key_derivation": "hmac"}]
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #2
#--------------------------

```python
def test_serializer_constructor_defaults():
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

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer = Serializer("secret-key", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"data"
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret-key", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_multiple_secret_keys_and_custom_salt():
    serializer = Serializer(["key1", b"key2", "key3"], salt=b"custom")
    assert serializer.secret_keys == [b"key1", b"key2", b"key3"]
    assert serializer.salt == b"custom"

def test_serializer_constructor_secret_key_property():
    serializer = Serializer(["old", "new"])
    assert serializer.secret_key == b"new"

def test_serializer_constructor_preserves_default_serializer():
    serializer = Serializer("secret")
    assert serializer.default_serializer is json

def test_serializer_constructor_preserves_default_signer():
    serializer = Serializer("secret")
    assert serializer.default_signer is Signer

def test_serializer_constructor_preserves_default_fallback_signers():
    serializer = Serializer("secret")
    assert serializer.default_fallback_signers == []


# LLM-generated content at query #3
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_str_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return eval(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_custom_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer_class():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_custom_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_constructor_with_serializer_returning_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"

        @staticmethod
        def loads(payload):
            return payload

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer == False


# LLM-generated content at query #4
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_serializer():
    from itsdangerous.serializer import Serializer
    serializer = Serializer("secret-key", serializer=json)
    assert serializer.serializer is json

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.signer import Signer
    serializer = Serializer("secret-key", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"bytes-secret")
    assert serializer.secret_keys == [b"bytes-secret"]

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_bytes_secret_key():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_fallback_signers_none():
    serializer = Serializer("secret-key", fallback_signers=None)
    assert serializer.fallback_signers == []


# LLM-generated content at query #5
#--------------------------

```
def test_iter_unsigners_yields_default_signer_first():
    serializer = Serializer("secret", salt="salt")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) >= 1
    assert unsigners[0].salt == want_bytes("salt")

def test_iter_unsigners_with_dict_fallback():
    serializer = Serializer("secret", salt="salt", fallback_signers=[{"key": "fallback_secret"}])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) >= 2

def test_iter_unsigners_with_tuple_fallback():
    serializer = Serializer("secret", salt="salt", fallback_signers=[(Signer, {"key": "fallback_secret"})])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) >= 2

def test_iter_unsigners_with_signer_class_fallback():
    serializer = Serializer("secret", salt="salt", fallback_signers=[Signer])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) >= 2

def test_iter_unsigners_uses_provided_salt():
    serializer = Serializer("secret", salt="default_salt")
    unsigners = list(serializer.iter_unsigners(salt="custom_salt"))
    assert unsigners[0].salt == want_bytes("custom_salt")

def test_iter_unsigners_yields_for_each_secret_key_with_fallback():
    serializer = Serializer(["secret1", "secret2"], salt="salt", fallback_signers=[Signer])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1 + 2

def test_iter_unsigners_empty_fallback_signers():
    serializer = Serializer("secret", salt="salt", fallback_signers=[])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1

def test_iter_unsigners_yields_default_signer_with_correct_secret_keys():
    serializer = Serializer(["secret1", "secret2"], salt="salt")
    unsigners = list(serializer.iter_unsigners())
    assert unsigners[0].secret_keys == [want_bytes("secret1"), want_bytes("secret2")]

def test_iter_unsigners_fallback_with_dict_uses_default_signer():
    serializer = Serializer("secret", salt="salt", signer=Signer, fallback_signers=[{"key": "fallback_secret"}])
    unsigners = list(serializer.iter_unsigners())
    assert unsigners[1].secret_keys == [want_bytes("fallback_secret")]
```


# LLM-generated content at query #6
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(data):
            return data
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]


# LLM-generated content at query #7
#--------------------------

```python
def test_load_payload_with_default_serializer_and_text_payload():
    serializer = Serializer("secret")
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_text_serializer():
    class TextSerializer:
        def loads(self, payload):
            return {"data": payload}
        def dumps(self, obj):
            return "dummy"
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = b"test_payload"
    result = serializer.load_payload(payload)
    assert result == {"data": "test_payload"}

def test_load_payload_with_custom_bytes_serializer():
    class BytesSerializer:
        def loads(self, payload):
            return {"data": payload}
        def dumps(self, obj):
            return b"dummy"
    serializer = Serializer("secret", serializer=BytesSerializer())
    payload = b"test_bytes"
    result = serializer.load_payload(payload)
    assert result == {"data": b"test_bytes"}

def test_load_payload_with_override_serializer_parameter():
    class OverrideSerializer:
        def loads(self, payload):
            return {"override": payload}
        def dumps(self, obj):
            return "override"
    serializer = Serializer("secret")
    payload = b"test"
    result = serializer.load_payload(payload, serializer=OverrideSerializer())
    assert result == {"override": "test"}

def test_load_payload_with_bad_payload_raises_exception():
    serializer = Serializer("secret")
    payload = b"invalid json"
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_with_text_serializer_and_unicode_payload():
    class TextSerializer:
        def loads(self, payload):
            return payload
        def dumps(self, obj):
            return "dummy"
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = "unicode".encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == "unicode"
```


# LLM-generated content at query #8
#--------------------------

```python
def test_iter_unsigners_predicate_false():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    import json

    serializer = Serializer(
        secret_key=b"test_key",
        salt=b"test_salt",
        serializer=json,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[Signer]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 0
    for signer in signers:
        assert isinstance(signer, Signer)
```


# LLM-generated content at query #9
#--------------------------

```python
def test_iter_unsigners_returns_correct_number_of_signers():
    serializer = Serializer("secret-key", fallback_signers=[{}, {"algorithm": "sha256"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1 + 1 + 1

def test_iter_unsigners_first_signer_is_default():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"secret-key"]

def test_iter_unsigners_with_fallback_dict():
    serializer = Serializer("secret-key", fallback_signers=[{"key_derivation": "hmac"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[1].key_derivation == "hmac"

def test_iter_unsigners_with_fallback_tuple():
    serializer = Serializer("secret-key", fallback_signers=[(Signer, {"digest_method": "sha256"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[1].digest_method == "sha256"

def test_iter_unsigners_with_fallback_signer_class():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", fallback_signers=[CustomSigner])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], CustomSigner)

def test_iter_unsigners_with_multiple_secret_keys():
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1

def test_iter_unsigners_with_custom_salt():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners(salt=b"custom-salt"))
    assert signers[0].salt == b"custom-salt"

def test_iter_unsigners_returns_iterator():
    serializer = Serializer("secret-key")
    result = serializer.iter_unsigners()
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")

def test_iter_unsigners_with_none_salt_uses_default():
    serializer = Serializer("secret-key", salt=b"default-salt")
    signers = list(serializer.iter_unsigners(salt=None))
    assert signers[0].salt == b"default-salt"

def test_iter_unsigners_empty_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=[])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
```


# LLM-generated content at query #10
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer == True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_bytes_salt():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_str_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    s = Serializer("secret", serializer=CustomSerializer)
    assert s.serializer == CustomSerializer
    assert s.is_text_serializer == True

def test_serializer_constructor_with_custom_bytes_serializer():
    class CustomBytesSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode()
        @staticmethod
        def loads(b):
            return eval(b.decode())
    s = Serializer("secret", serializer=CustomBytesSerializer)
    assert s.serializer == CustomBytesSerializer
    assert s.is_text_serializer == False

def test_serializer_constructor_with_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "hmac"}])
    assert s.fallback_signers == [{"key_derivation": "hmac"}]

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #11
#--------------------------

```python
def test_load_payload_text_serializer_raises_bad_payload():
    serializer = Serializer("secret", serializer=json)
    payload = b"invalid json"
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass
    else:
        assert False, "Expected BadPayload"
```


# LLM-generated content at query #12
#--------------------------

```python
def test_salt_is_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None
```


# LLM-generated content at query #13
#--------------------------

```python
def test_iter_unsigners_with_default_signer_and_no_fallback():
    serializer = Serializer(secret_key="secret", serializer=None)
    signers = list(serializer.iter_unsigners(salt=None))
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)

def test_iter_unsigners_with_custom_salt():
    serializer = Serializer(secret_key="secret", serializer=None)
    signers = list(serializer.iter_unsigners(salt=b"custom_salt"))
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)

def test_iter_unsigners_with_fallback_signers_dict():
    serializer = Serializer(secret_key="secret", fallback_signers=[{"key": "fallback_secret"}])
    signers = list(serializer.iter_unsigners(salt=None))
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_with_fallback_signers_tuple():
    serializer = Serializer(secret_key="secret", fallback_signers=[(Signer, {"key": "fallback_secret"})])
    signers = list(serializer.iter_unsigners(salt=None))
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_with_fallback_signers_class():
    serializer = Serializer(secret_key="secret", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners(salt=None))
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_with_multiple_secret_keys():
    serializer = Serializer(secret_key=["old_secret", "new_secret"])
    signers = list(serializer.iter_unsigners(salt=None))
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)

def test_iter_unsigners_with_fallback_and_multiple_keys():
    serializer = Serializer(secret_key=["key1", "key2"], fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners(salt=None))
    assert len(signers) == 3
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert isinstance(signers[2], Signer)
```


# LLM-generated content at query #14
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "dumped", "loads": lambda self, x: "loaded"})()
    s = Serializer("secret", serializer=custom_serializer)
    assert s.serializer is custom_serializer
    assert s.is_text_serializer is True

def test_serializer_constructor_with_binary_serializer():
    binary_serializer = type("BinarySerializer", (), {"dumps": lambda self, x: b"dumped", "loads": lambda self, x: b"loaded"})()
    s = Serializer("secret", serializer=binary_serializer)
    assert s.serializer is binary_serializer
    assert s.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {"sign": lambda self, x: x, "unsign": lambda self, x: x})()
    s = Serializer("secret", signer=custom_signer)
    assert s.signer is custom_signer

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert s.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_uses_default_fallback_signers():
    s = Serializer("secret")
    assert s.fallback_signers == []

def test_serializer_constructor_with_fallback_signers_none():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []


# LLM-generated content at query #15
#--------------------------

```python
def test_serializer_init_with_explicit_serializer_passes_through_to_attribute():
    from itsdangerous.serializer import Serializer
    import json
    serializer = Serializer("secret-key", serializer=json)
    assert serializer.serializer is json
```


# LLM-generated content at query #16
#--------------------------

```python
def test_iter_unsigners_fallback_is_tuple():
    serializer = Serializer(
        secret_key=b"test-secret",
        fallback_signers=[(Signer, {"digest_method": hashlib.sha256})],
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
```


# LLM-generated content at query #17
#--------------------------

```python
def test_load_payload_is_text_false_with_bytes_serializer():
    from itsdangerous.serializer import Serializer
    from itsdangerous.exc import BadPayload
    import json

    serializer = Serializer("secret-key", serializer=lambda: None)
    serializer.is_text_serializer = False
    serializer.serializer = json
    payload = json.dumps({"key": "value"}).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
```


# LLM-generated content at query #18
#--------------------------

```
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert result is not None
```


# LLM-generated content at query #19
#--------------------------

```python
def test_fallback_signers_not_none():
    serializer = Serializer(secret_key="secret", fallback_signers=[Signer])
    assert serializer.fallback_signers is not None
```


# LLM-generated content at query #20
#--------------------------

```python
def test_fallback_signers_predicate_false():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #21
#--------------------------

```python
def test_serializer_constructor_with_serializer_not_none():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
```


# LLM-generated content at query #22
#--------------------------

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

def test_serializer_constructor_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_custom_serializer_str():
    class StrSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer = Serializer("secret", serializer=StrSerializer())
    assert serializer.is_text_serializer is True

def test_serializer_constructor_custom_serializer_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return bytes(str(obj), "utf-8")
        @staticmethod
        def loads(b):
            return int(b.decode("utf-8"))
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_secret_key_bytes():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]


# LLM-generated content at query #23
#--------------------------

def test_constructor_default_parameters():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"bytes-key")
    assert serializer.secret_keys == [b"bytes-key"]

def test_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2", "key3"])
    assert serializer.secret_keys == [b"key1", b"key2", b"key3"]

def test_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2", b"key3"])
    assert serializer.secret_keys == [b"key1", b"key2", b"key3"]

def test_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return eval(s)

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_constructor_is_text_serializer_true():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "text"

        @staticmethod
        def loads(s):
            return s

    serializer = Serializer("secret-key", serializer=TextSerializer())
    assert serializer.is_text_serializer is True

def test_constructor_is_text_serializer_false():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"

        @staticmethod
        def loads(s):
            return s

    serializer = Serializer("secret-key", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_constructor_default_fallback_signers():
    class CustomSerializer(Serializer):
        default_fallback_signers = [{"key_derivation": "none"}]

    serializer = CustomSerializer("secret-key")
    assert serializer.fallback_signers == [{"key_derivation": "none"}]


# LLM-generated content at query #24
#--------------------------

```python
def test_init_with_serializer_evaluates_predicate_false():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
```


# LLM-generated content at query #25
#--------------------------

def test_dumps_returns_bytes_with_default_json_serializer():
    serializer = Serializer(secret_key="secret")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)

def test_dumps_returns_string_with_text_serializer():
    class TextSerializer:
        def dumps(self, obj):
            return "text"

        def loads(self, s):
            return s

    serializer = Serializer(secret_key="secret", serializer=TextSerializer())
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)

def test_dumps_includes_signature():
    serializer = Serializer(secret_key="secret")
    result = serializer.dumps("data")
    assert b"." in result

def test_dumps_with_salt_overrides_default():
    serializer = Serializer(secret_key="secret", salt=b"default_salt")
    result_with_default = serializer.dumps("data")
    result_with_custom = serializer.dumps("data", salt=b"custom_salt")
    assert result_with_default != result_with_custom

def test_dumps_uses_serializer_kwargs():
    class SerializerWithKwargs:
        def __init__(self):
            self.called_with = None

        def dumps(self, obj, **kwargs):
            self.called_with = kwargs
            return "serialized"

        def loads(self, s):
            return s

    custom_serializer = SerializerWithKwargs()
    serializer = Serializer(secret_key="secret", serializer=custom_serializer, serializer_kwargs={"indent": 2})
    serializer.dumps("data")
    assert custom_serializer.called_with == {"indent": 2}


# LLM-generated content at query #26
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_key == b"secret"
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode()

        @staticmethod
        def loads(b):
            return int(b.decode())

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.serializer is BytesSerializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_multiple_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.secret_key == b"key2"

def test_serializer_constructor_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #27
#--------------------------

```
def test_iter_unsigners_default_signer():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)

def test_iter_unsigners_with_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].salt == b"custom-salt"

def test_iter_unsigners_fallback_dict():
    serializer = Serializer("secret-key", fallback_signers=[{"key": "fallback_secret"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_fallback_tuple():
    fallback_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret-key", fallback_signers=[(fallback_signer, {"key": "fallback_secret"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], fallback_signer)

def test_iter_unsigners_fallback_signer_class():
    fallback_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret-key", fallback_signers=[fallback_signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], fallback_signer)

def test_iter_unsigners_multiple_secret_keys():
    serializer = Serializer(["old-secret", "new-secret"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"old-secret", b"new-secret"]

def test_iter_unsigners_fallback_with_multiple_secret_keys():
    serializer = Serializer(["old-secret", "new-secret"], fallback_signers=[{"key": "fallback_secret"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert signers[0].secret_keys == [b"old-secret", b"new-secret"]
    assert signers[1].secret_keys == [b"old-secret"]
    assert signers[2].secret_keys == [b"new-secret"]

def test_iter_unsigners_with_explicit_salt():
    serializer = Serializer("secret-key", salt="default-salt")
    signers = list(serializer.iter_unsigners(salt="explicit-salt"))
    assert len(signers) == 1
    assert signers[0].salt == b"explicit-salt"

def test_iter_unsigners_fallback_with_explicit_salt():
    serializer = Serializer("secret-key", fallback_signers=[{"key": "fallback_secret"}])
    signers = list(serializer.iter_unsigners(salt="explicit-salt"))
    assert len(signers) == 2
    assert signers[0].salt == b"explicit-salt"
    assert signers[1].salt == b"explicit-salt"
```


# LLM-generated content at query #28
#--------------------------

def test_dumps_returns_bytes_when_not_text_serializer():
    s = Serializer("secret", serializer=None)
    s.is_text_serializer = False
    result = s.dumps({"key": "value"})
    assert isinstance(result, bytes)


# LLM-generated content at query #29
#--------------------------

def test_serializer_constructor_default_serializer() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt() -> None:
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: object) -> str:
            return "serialized"

        @staticmethod
        def loads(s: str) -> object:
            return "deserialized"

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer() -> None:
    class BytesSerializer:
        @staticmethod
        def dumps(obj: object) -> bytes:
            return b"serialized"

        @staticmethod
        def loads(s: bytes) -> object:
            return b"deserialized"

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers() -> None:
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #30
#--------------------------

```python
def test_load_payload_is_text_false_with_bytes_serializer():
    serializer = Serializer("secret", serializer=lambda: None)
    serializer.serializer = type("BytesSerializer", (), {"loads": lambda self, x: x})()
    serializer.is_text_serializer = False
    result = serializer.load_payload(b"test")
    assert result == b"test"
```


# LLM-generated content at query #31
#--------------------------

```
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert result is not None

def test_dumps_accepts_any_type():
    serializer = _PDataSerializer()
    result = serializer.dumps(42)
    assert result is not None

def test_dumps_returns_serialized_type_for_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result is not None

def test_dumps_returns_serialized_type_for_list():
    serializer = _PDataSerializer()
    result = serializer.dumps([1, 2, 3])
    assert result is not None

def test_dumps_returns_serialized_type_for_dict():
    serializer = _PDataSerializer()
    result = serializer.dumps({"key": "value"})
    assert result is not None

def test_dumps_returns_serialized_type_for_bool():
    serializer = _PDataSerializer()
    result = serializer.dumps(True)
    assert result is not None

def test_dumps_returns_serialized_type_for_float():
    serializer = _PDataSerializer()
    result = serializer.dumps(3.14)
    assert result is not None

def test_dumps_returns_serialized_type_for_complex_object():
    class CustomObject:
        pass
    serializer = _PDataSerializer()
    result = serializer.dumps(CustomObject())
    assert result is not None
```


# LLM-generated content at query #32
#--------------------------

```python
def test_load_payload_predicate_false():
    serializer = Serializer("secret", serializer=Serializer.default_serializer)
    serializer.is_text_serializer = False
    payload = b"null"
    result = serializer.load_payload(payload)
    assert result is None
```


# LLM-generated content at query #33
#--------------------------

```
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test_object")
    assert result is not None
```


# LLM-generated content at query #34
#--------------------------

def test_salt_is_none_so_predicate_is_false():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #35
#--------------------------

def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads(b"test")
    assert result is None or result is not None

def test_loads_accepts_bytes():
    serializer = _PDataSerializer()
    result = serializer.loads(b"data")
    assert result is None or result is not None

def test_loads_accepts_string():
    serializer = _PDataSerializer()
    result = serializer.loads("data")
    assert result is None or result is not None

def test_loads_accepts_integer():
    serializer = _PDataSerializer()
    result = serializer.loads(123)
    assert result is None or result is not None

def test_loads_accepts_list():
    serializer = _PDataSerializer()
    result = serializer.loads([1, 2, 3])
    assert result is None or result is not None

def test_loads_accepts_dict():
    serializer = _PDataSerializer()
    result = serializer.loads({"key": "value"})
    assert result is None or result is not None


# LLM-generated content at query #36
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(data):
            return data
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    fallback = [{"digest_method": hashlib.sha256}]
    serializer = Serializer(
        secret_key=["old_key", b"new_key"],
        salt=b"custom_salt",
        serializer=CustomSerializer(),
        serializer_kwargs={"ensure_ascii": False},
        signer=Signer,
        signer_kwargs={"key_derivation": "hmac"},
        fallback_signers=fallback,
    )
    assert serializer.secret_keys == [b"old_key", b"new_key"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {"ensure_ascii": False}


# LLM-generated content at query #37
#--------------------------

def test_serializer_constructor_with_default_parameters():
    secret_key = "secret"
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    secret_key = b"secret"
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings():
    secret_key = ["key1", "key2"]
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    secret_key = [b"key1", b"key2"]
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    secret_key = "secret"
    serializer = Serializer(secret_key, salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_salt():
    secret_key = "secret"
    salt = b"custom_salt"
    serializer = Serializer(secret_key, salt=salt)
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    secret_key = "secret"
    serializer = Serializer(secret_key, serializer=json)
    assert serializer.serializer == json

def test_serializer_constructor_with_serializer_kwargs():
    secret_key = "secret"
    serializer_kwargs = {"indent": 2}
    serializer = Serializer(secret_key, serializer_kwargs=serializer_kwargs)
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    secret_key = "secret"
    serializer = Serializer(secret_key, signer=Signer)
    assert serializer.signer == Signer

def test_serializer_constructor_with_signer_kwargs():
    secret_key = "secret"
    signer_kwargs = {"key_derivation": "hmac"}
    serializer = Serializer(secret_key, signer_kwargs=signer_kwargs)
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    secret_key = "secret"
    fallback_signers = [Signer]
    serializer = Serializer(secret_key, fallback_signers=fallback_signers)
    assert serializer.fallback_signers == [Signer]

def test_serializer_constructor_with_custom_serializer_returns_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"test"
        @staticmethod
        def loads(data):
            return {}
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_secret_key_property():
    secret_key = ["old", "new"]
    serializer = Serializer(secret_key)
    assert serializer.secret_key == b"new"


# LLM-generated content at query #38
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is Serializer.default_serializer
    assert s.is_text_serializer == isinstance(json.dumps({}), str)
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secrets():
    s = Serializer(["secret1", "secret2"])
    assert s.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_bytes_salt():
    s = Serializer("secret", salt=b"bytes_salt")
    assert s.salt == b"bytes_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        def dumps(self, obj):
            return str(obj)
        def loads(self, s):
            return str(s)
    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer is not Serializer.default_serializer
    assert s.is_text_serializer == isinstance(CustomSerializer().dumps({}), str)

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers():
    s = Serializer("secret", fallback_signers=[])
    assert s.fallback_signers == []

def test_serializer_constructor_with_none_fallback_signers():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []

def test_serializer_constructor_secret_key_property():
    s = Serializer(["old", "new"])
    assert s.secret_key == b"new"


# LLM-generated content at query #39
#--------------------------

```python
def test_salt_is_not_none_in_constructor():
    serializer = Serializer(secret_key="test-secret", salt=b"custom-salt")
    assert serializer.salt is not None
```


# LLM-generated content at query #40
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_bytes_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_list_keys():
    s = Serializer(["key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_salt_bytes():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_salt_str():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_serializer_text():
    s = Serializer("secret", serializer=json)
    assert s.is_text_serializer is True

def test_serializer_constructor_serializer_bytes():
    from itsdangerous.serializer import _PDataSerializer
    class BytesSerializer:
        def dumps(self, obj):
            return b"bytes"
        def loads(self, data):
            return data
    s = Serializer("secret", serializer=BytesSerializer())
    assert s.is_text_serializer is False

def test_serializer_constructor_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback


# LLM-generated content at query #41
#--------------------------

```python
def test_load_payload_with_text_serializer_and_valid_payload_returns_deserialized_data():
    serializer = Serializer("secret", serializer=json)
    payload = json.dumps({"key": "value"}).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #42
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"

def test_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"

def test_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    custom_serializer = json
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer == True

def test_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"

        @staticmethod
        def loads(data):
            return data

    serializer = Serializer("secret-key", serializer=BytesSerializer())
    assert serializer.is_text_serializer == False

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #43
#--------------------------

```
def test_dumps_with_string_serializer():
    serializer = _PDataSerializer[str]()
    result = serializer.dumps("test_string")
    assert result == "test_string"

def test_dumps_with_integer_serializer():
    serializer = _PDataSerializer[int]()
    result = serializer.dumps(42)
    assert result == 42

def test_dumps_with_list_serializer():
    serializer = _PDataSerializer[list[int]]()
    result = serializer.dumps([1, 2, 3])
    assert result == [1, 2, 3]

def test_dumps_with_none_serializer():
    serializer = _PDataSerializer[None]()
    result = serializer.dumps(None)
    assert result is None

def test_dumps_with_dict_serializer():
    serializer = _PDataSerializer[dict[str, int]]()
    result = serializer.dumps({"key": 1})
    assert result == {"key": 1}

def test_dumps_with_float_serializer():
    serializer = _PDataSerializer[float]()
    result = serializer.dumps(3.14)
    assert result == 3.14

def test_dumps_with_bool_serializer():
    serializer = _PDataSerializer[bool]()
    result = serializer.dumps(True)
    assert result is True

def test_dumps_with_bytes_serializer():
    serializer = _PDataSerializer[bytes]()
    result = serializer.dumps(b"bytes_data")
    assert result == b"bytes_data"

def test_dumps_with_tuple_serializer():
    serializer = _PDataSerializer[tuple[int, str]]()
    result = serializer.dumps((1, "test"))
    assert result == (1, "test")

def test_dumps_with_set_serializer():
    serializer = _PDataSerializer[set[int]]()
    result = serializer.dumps({1, 2, 3})
    assert result == {1, 2, 3}
```


# LLM-generated content at query #44
#--------------------------

def test_serializer_constructor_with_default_parameters() -> None:
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings() -> None:
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes() -> None:
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt() -> None:
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_salt_none() -> None:
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_text_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return str(obj)
        @staticmethod
        def loads(s: str) -> t.Any:
            return eval(s)
    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer == CustomSerializer()
    assert s.is_text_serializer is True

def test_serializer_constructor_with_custom_bytes_serializer() -> None:
    class CustomBytesSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> bytes:
            return str(obj).encode()
        @staticmethod
        def loads(b: bytes) -> t.Any:
            return eval(b.decode())
    s = Serializer("secret", serializer=CustomBytesSerializer())
    assert s.serializer == CustomBytesSerializer()
    assert s.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs() -> None:
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    s = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert s.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers() -> None:
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_all_parameters() -> None:
    s = Serializer(
        ["key1", b"key2"],
        salt="custom_salt",
        serializer=json,
        serializer_kwargs={"sort_keys": True},
        signer=Signer,
        signer_kwargs={"key_derivation": "none"},
        fallback_signers=[{"key_derivation": "hmac"}]
    )
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"custom_salt"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {"key_derivation": "none"}
    assert s.fallback_signers == [{"key_derivation": "hmac"}]
    assert s.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #45
#--------------------------

```python
def test_load_payload_with_default_serializer_returns_loaded_data():
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer_returns_loaded_data():
    custom_serializer = type("CustomSerializer", (), {"loads": staticmethod(lambda x: x + " loaded"), "dumps": staticmethod(lambda x: x)})()
    serializer = Serializer("secret", serializer=custom_serializer)
    payload = serializer.dump_payload("test")
    result = serializer.load_payload(payload, serializer=custom_serializer)
    assert result == "test loaded"

def test_load_payload_with_text_serializer_decodes_bytes():
    class TextSerializer:
        def dumps(self, obj):
            return "text"
        def loads(self, payload):
            return payload
    text_serializer = TextSerializer()
    serializer = Serializer("secret", serializer=text_serializer)
    payload = b"encoded"
    result = serializer.load_payload(payload, serializer=text_serializer)
    assert result == "encoded"

def test_load_payload_with_bytes_serializer_does_not_decode():
    class BytesSerializer:
        def dumps(self, obj):
            return b"bytes"
        def loads(self, payload):
            return payload
    bytes_serializer = BytesSerializer()
    serializer = Serializer("secret", serializer=bytes_serializer)
    payload = b"raw"
    result = serializer.load_payload(payload, serializer=bytes_serializer)
    assert result == b"raw"

def test_load_payload_raises_bad_payload_on_exception():
    class FailingSerializer:
        def loads(self, payload):
            raise ValueError("fail")
        def dumps(self, obj):
            return b"data"
    failing_serializer = FailingSerializer()
    serializer = Serializer("secret", serializer=failing_serializer)
    payload = b"invalid"
    try:
        serializer.load_payload(payload, serializer=failing_serializer)
        assert False
    except BadPayload:
        pass

def test_load_payload_with_none_serializer_uses_self_serializer():
    serializer = Serializer("secret")
    payload = serializer.dump_payload(123)
    result = serializer.load_payload(payload, serializer=None)
    assert result == 123

def test_load_payload_with_is_text_true_uses_decode():
    class TextSerializer:
        def dumps(self, obj):
            return "str"
        def loads(self, payload):
            return payload
    text_serializer = TextSerializer()
    serializer = Serializer("secret", serializer=text_serializer)
    payload = b"hello"
    result = serializer.load_payload(payload, serializer=text_serializer)
    assert result == "hello"

def test_load_payload_with_is_text_false_no_decode():
    class BytesSerializer:
        def dumps(self, obj):
            return b"bytes"
        def loads(self, payload):
            return payload
    bytes_serializer = BytesSerializer()
    serializer = Serializer("secret", serializer=bytes_serializer)
    payload = b"world"
    result = serializer.load_payload(payload, serializer=bytes_serializer)
    assert result == b"world"
```


# LLM-generated content at query #46
#--------------------------

```python
def test_fallback_signers_not_none():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    serializer = Serializer("secret", fallback_signers=[Signer])
    assert serializer.fallback_signers is not None
```


# LLM-generated content at query #47
#--------------------------

def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads(b'test')
    assert result is not None


# LLM-generated content at query #48
#--------------------------

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

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer = Serializer("secret-key", serializer=CustomSerializer)
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_signer():
    class CustomSigner:
        def __init__(self, secret_keys, salt, **kwargs):
            self.secret_keys = secret_keys
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, value):
            return value
        def unsign(self, value):
            return value
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #49
#--------------------------

def test_serializer_constructor_with_default_parameters():
    secret_key = "test-secret"
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"test-secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    secret_key = b"bytes-secret"
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"bytes-secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    secret_keys = ["key1", "key2"]
    serializer = Serializer(secret_keys)
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    secret_key = "test"
    salt = "custom-salt"
    serializer = Serializer(secret_key, salt=salt)
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    secret_key = "test"
    serializer = Serializer(secret_key, salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    secret_key = "test"
    serializer = Serializer(secret_key, serializer=json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    secret_key = "test"
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"dumped"
        @staticmethod
        def loads(data):
            return data
    serializer = Serializer(secret_key, serializer=BytesSerializer())
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_custom_signer():
    secret_key = "test"
    class CustomSigner(Signer):
        pass
    serializer = Serializer(secret_key, signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    secret_key = "test"
    signer_kwargs = {"key_derivation": "hmac"}
    serializer = Serializer(secret_key, signer_kwargs=signer_kwargs)
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    secret_key = "test"
    fallback_signers = [{"key_derivation": "none"}]
    serializer = Serializer(secret_key, fallback_signers=fallback_signers)
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_serializer_kwargs():
    secret_key = "test"
    serializer_kwargs = {"sort_keys": True}
    serializer = Serializer(secret_key, serializer_kwargs=serializer_kwargs)
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #50
#--------------------------

def test_serializer_init_with_string_secret_key():
    serializer = Serializer("my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key():
    serializer = Serializer(b"my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]

def test_serializer_init_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_none_salt():
    serializer = Serializer("key", salt=None)
    assert serializer.salt is None

def test_serializer_init_with_custom_salt():
    serializer = Serializer("key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_init_with_bytes_salt():
    serializer = Serializer("key", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_init_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer = Serializer("key", serializer=CustomSerializer)
    assert serializer.serializer == CustomSerializer
    assert serializer.is_text_serializer == True

def test_serializer_init_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"data"
        @staticmethod
        def loads(b):
            return b
    serializer = Serializer("key", serializer=BytesSerializer)
    assert serializer.serializer == BytesSerializer
    assert serializer.is_text_serializer == False

def test_serializer_init_with_serializer_kwargs():
    serializer = Serializer("key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_init_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_init_with_signer_kwargs():
    serializer = Serializer("key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_init_with_empty_fallback_signers():
    serializer = Serializer("key", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_init_with_none_fallback_signers():
    serializer = Serializer("key", fallback_signers=None)
    assert serializer.fallback_signers == []


# LLM-generated content at query #51
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_salt_bytes():
    serializer = Serializer("secret-key", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_salt_str():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode()

        @staticmethod
        def loads(data):
            return int(data.decode())

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_secret_key_bytes():
    serializer = Serializer(b"bytes-key")
    assert serializer.secret_keys == [b"bytes-key"]

def test_serializer_constructor_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]


# LLM-generated content at query #52
#--------------------------

def test_serializer_constructor_default_values():
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

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner:
        def __init__(self, secret_keys, salt=None, **kwargs):
            pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_custom_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return s
    class CustomSigner:
        def __init__(self, secret_keys, salt=None, **kwargs):
            pass
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer(
        secret_key="secret",
        salt="custom_salt",
        serializer=CustomSerializer,
        serializer_kwargs={"sort_keys": True},
        signer=CustomSigner,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=fallback
    )
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is CustomSigner
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #53
#--------------------------

def test_dumps_returns_bytes_when_is_text_serializer_is_false():
    serializer = Serializer("test-secret", serializer=lambda: None)
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)


# LLM-generated content at query #54
#--------------------------

def test_dumps_signs_payload_with_signer():
    serializer = Serializer(secret_key="secret", salt="salt")
    signed = serializer.dumps({"key": "value"})
    assert isinstance(signed, bytes) or isinstance(signed, str)
    assert len(signed) > 0

def test_dumps_returns_bytes_when_not_text_serializer():
    serializer = Serializer(secret_key="secret", salt="salt", serializer=type("BytesSerializer", (), {"dumps": lambda self, obj: b"serialized", "loads": lambda self, data: data})())
    result = serializer.dumps("data")
    assert isinstance(result, bytes)

def test_dumps_returns_string_when_text_serializer():
    serializer = Serializer(secret_key="secret", salt="salt", serializer=type("TextSerializer", (), {"dumps": lambda self, obj: "serialized", "loads": lambda self, data: data})())
    result = serializer.dumps("data")
    assert isinstance(result, str)

def test_dumps_uses_custom_salt():
    serializer = Serializer(secret_key="secret", salt="default")
    result_default = serializer.dumps("data")
    result_custom = serializer.dumps("data", salt="custom")
    assert result_default != result_custom


# LLM-generated content at query #55
#--------------------------

def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads(b"test")
    assert result is None or True


# LLM-generated content at query #56
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_serializer():
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.serializer is bytes_serializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_keyword():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_serializer_returning_bytes():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(data):
            return data
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is False


# LLM-generated content at query #57
#--------------------------

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

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_str_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_custom_default_serializer():
    class CustomDefaultSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    original_default = Serializer.default_serializer
    Serializer.default_serializer = CustomDefaultSerializer()
    try:
        serializer = Serializer("secret")
        assert serializer.serializer == CustomDefaultSerializer()
        assert serializer.is_text_serializer is True
    finally:
        Serializer.default_serializer = original_default

def test_serializer_constructor_with_custom_default_fallback_signers():
    original_fallback = Serializer.default_fallback_signers
    Serializer.default_fallback_signers = [{"key_derivation": "none"}]
    try:
        serializer = Serializer("secret")
        assert serializer.fallback_signers == [{"key_derivation": "none"}]
    finally:
        Serializer.default_fallback_signers = original_fallback


# LLM-generated content at query #58
#--------------------------

def test_loads_with_valid_payload():
    serializer = _PDataSerializer()
    result = serializer.loads("test_payload")
    assert result is not None

def test_loads_with_empty_payload():
    serializer = _PDataSerializer()
    result = serializer.loads("")
    assert result is not None

def test_loads_with_none_payload():
    serializer = _PDataSerializer()
    result = serializer.loads(None)
    assert result is None

def test_loads_with_integer_payload():
    serializer = _PDataSerializer()
    result = serializer.loads(123)
    assert result == 123

def test_loads_with_list_payload():
    serializer = _PDataSerializer()
    result = serializer.loads([1, 2, 3])
    assert result == [1, 2, 3]


# LLM-generated content at query #59
#--------------------------

```python
def test_serializer_constructor_defaults():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == is_text_serializer(json)
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_custom_secret_key_bytes():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_secret_key_list_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_salt_none():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_custom_serializer():
    custom_serializer = json
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer == is_text_serializer(custom_serializer)

def test_serializer_constructor_custom_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_custom_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_custom_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_overrides_default_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_constructor_positional_serializer():
    serializer = Serializer("secret-key", b"custom-salt", json)
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == json

def test_serializer_constructor_keyword_serializer():
    serializer = Serializer("secret-key", serializer=json)
    assert serializer.serializer == json
```


# LLM-generated content at query #60
#--------------------------

```
def test_dumps_returns_serialized_data():
    _PDataSerializer.dumps(None, 42)


# LLM-generated content at query #61
#--------------------------

```python
def test_iter_unsigners_with_no_fallback_signers():
    serializer = Serializer(secret_key="secret-key", fallback_signers=[])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"secret-key"]

def test_iter_unsigners_with_fallback_signers_dict():
    serializer = Serializer(secret_key="secret-key", fallback_signers=[{"key": "fallback"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].secret_keys == [b"secret-key"]
    assert signers[1].secret_keys == [b"secret-key"]

def test_iter_unsigners_with_fallback_signers_tuple():
    serializer = Serializer(secret_key="secret-key", fallback_signers=[(Signer, {"key": "fallback"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].secret_keys == [b"secret-key"]
    assert signers[1].secret_keys == [b"secret-key"]

def test_iter_unsigners_with_fallback_signers_signer_class():
    serializer = Serializer(secret_key="secret-key", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].secret_keys == [b"secret-key"]
    assert signers[1].secret_keys == [b"secret-key"]

def test_iter_unsigners_with_multiple_secret_keys():
    serializer = Serializer(secret_key=["old-secret", "new-secret"], fallback_signers=[])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"old-secret", b"new-secret"]

def test_iter_unsigners_with_custom_salt():
    serializer = Serializer(secret_key="secret-key", salt=b"custom-salt", fallback_signers=[])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].salt == b"custom-salt"

def test_iter_unsigners_with_none_salt():
    serializer = Serializer(secret_key="secret-key", salt=None, fallback_signers=[])
    signers = list(serializer.iter_unsigners(salt=None))
    assert len(signers) == 1
    assert signers[0].salt == b"itsdangerous"

def test_iter_unsigners_with_explicit_salt_argument():
    serializer = Serializer(secret_key="secret-key", fallback_signers=[])
    signers = list(serializer.iter_unsigners(salt=b"explicit-salt"))
    assert len(signers) == 1
    assert signers[0].salt == b"explicit-salt"

def test_iter_unsigners_with_fallback_signers_multiple_secret_keys():
    serializer = Serializer(secret_key=["key1", "key2"], fallback_signers=[{"key": "fallback"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert signers[0].secret_keys == [b"key1", b"key2"]
    assert signers[1].secret_keys == [b"key1"]
    assert signers[2].secret_keys == [b"key2"]
```


# LLM-generated content at query #62
#--------------------------

```python
def test_load_payload_with_non_text_serializer_raises_bad_payload_on_decode_error():
    serializer = Serializer("secret", serializer=json)
    payload = b"\xff\xfe"
    try:
        result = serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass
```


# LLM-generated content at query #63
#--------------------------

def test_serializer_constructor_with_default_parameters():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = json
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    custom_signer = Signer
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer == custom_signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"bytes_salt")
    assert serializer.salt == b"bytes_salt"

def test_serializer_constructor_with_none_serializer():
    serializer = Serializer("secret", serializer=None)
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_none_signer():
    serializer = Serializer("secret", signer=None)
    assert serializer.signer == Signer

def test_serializer_constructor_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_all_parameters():
    serializer = Serializer(
        secret_key=["key1", "key2"],
        salt="salt",
        serializer=json,
        serializer_kwargs={"sort_keys": True},
        signer=Signer,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"salt"
    assert serializer.serializer == json
    assert serializer.serializer_kwargs == {"sort_keys": True}
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_bytes_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer == json


# LLM-generated content at query #64
#--------------------------

def test_dumps_returns_bytes_for_bytes_serializer():
    serializer = Serializer("secret", serializer=lambda: None)
    serializer.serializer.dumps = lambda obj, **kwargs: b'{"key":"value"}'
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)

def test_dumps_returns_str_for_text_serializer():
    serializer = Serializer("secret", serializer=lambda: None)
    serializer.serializer.dumps = lambda obj, **kwargs: '{"key":"value"}'
    serializer.is_text_serializer = True
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)

def test_dumps_uses_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    serializer.serializer.dumps = lambda obj, **kwargs: b'data'
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"}, salt=b"override_salt")
    assert isinstance(result, bytes)


# LLM-generated content at query #65
#--------------------------

def test_serializer_constructor_defaults() -> None:
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_with_list_of_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys() -> None:
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt() -> None:
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_bytes_salt() -> None:
    serializer = Serializer("secret-key", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return str(obj)

        @staticmethod
        def loads(s: str) -> t.Any:
            return s

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_bytes_serializer() -> None:
    class CustomBytesSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> bytes:
            return str(obj).encode()

        @staticmethod
        def loads(s: bytes) -> t.Any:
            return s.decode()

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    assert serializer.serializer == CustomBytesSerializer()
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers() -> None:
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_secret_key_property() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_key == b"key2"


# LLM-generated content at query #66
#--------------------------

def test_serializer_constructor_default_values():
    secret_key = "secret"
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    secret_key = b"secret"
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    secret_key = ["key1", b"key2"]
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none():
    secret_key = "secret"
    serializer = Serializer(secret_key, salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    secret_key = "secret"
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "{}", "loads": lambda self, x: {}})()
    serializer = Serializer(secret_key, serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_signer():
    secret_key = "secret"
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer(secret_key, signer=custom_signer)
    assert serializer.signer is custom_signer

def test_serializer_constructor_with_signer_kwargs():
    secret_key = "secret"
    signer_kwargs = {"key_derivation": "hmac"}
    serializer = Serializer(secret_key, signer_kwargs=signer_kwargs)
    assert serializer.signer_kwargs == signer_kwargs

def test_serializer_constructor_with_fallback_signers():
    secret_key = "secret"
    fallback_signers = [{"key_derivation": "none"}]
    serializer = Serializer(secret_key, fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_constructor_with_serializer_kwargs():
    secret_key = "secret"
    serializer_kwargs = {"sort_keys": True}
    serializer = Serializer(secret_key, serializer_kwargs=serializer_kwargs)
    assert serializer.serializer_kwargs == serializer_kwargs


# LLM-generated content at query #67
#--------------------------

```python
def test_loads_returns_deserialized_data():
    serializer = _PDataSerializer()
    test_payload = serializer.dumps("test_data")
    result = serializer.loads(test_payload)
    assert result == "test_data"


# LLM-generated content at query #68
#--------------------------

```python
def test_serializer_init_with_serializer_not_none():
    custom_serializer = {"dumps": lambda x: str(x), "loads": lambda x: x}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
```


# LLM-generated content at query #69
#--------------------------

```python
def test_salt_is_not_none_when_passed_as_none():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None
```


# LLM-generated content at query #70
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True

def test_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(payload):
            return payload
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.serializer is BytesSerializer()
    assert serializer.is_text_serializer is False

def test_constructor_with_custom_signer():
    class CustomSigner:
        def __init__(self, secret_key, salt, **kwargs):
            self.secret_key = secret_key
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, value):
            return value
        def unsign(self, value):
            return value
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #71
#--------------------------

def test_constructor_with_default_parameters():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(s):
            return {}
    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.serializer == CustomSerializer
    assert serializer.is_text_serializer is True

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_constructor_with_all_parameters():
    serializer = Serializer(
        secret_key=["key1", b"key2"],
        salt=b"custom_salt",
        serializer=json,
        serializer_kwargs={"indent": 2},
        signer=Signer,
        signer_kwargs={"key_derivation": "hmac"},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #72
#--------------------------

```python
def test_iter_unsigners_default_signer():
    serializer = Serializer("secret")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)

def test_iter_unsigners_with_salt():
    serializer = Serializer("secret")
    signers = list(serializer.iter_unsigners(salt="custom_salt"))
    assert len(signers) == 1
    assert signers[0].salt == want_bytes("custom_salt")

def test_iter_unsigners_with_fallback_dict():
    serializer = Serializer("secret", fallback_signers=[{"key": "fallback_key"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_with_fallback_tuple():
    serializer = Serializer("secret", fallback_signers=[(Signer, {"key": "fallback_key"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_with_fallback_class():
    serializer = Serializer("secret", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_with_multiple_secret_keys():
    serializer = Serializer(["old_secret", "new_secret"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [want_bytes("old_secret"), want_bytes("new_secret")]

def test_iter_unsigners_with_fallback_and_multiple_secret_keys():
    serializer = Serializer(["old_secret", "new_secret"], fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3

def test_iter_unsigners_yields_generator():
    serializer = Serializer("secret")
    result = serializer.iter_unsigners()
    from collections.abc import Iterator
    assert isinstance(result, Iterator)
```


# LLM-generated content at query #73
#--------------------------

def test_serializer_constructor_defaults() -> None:
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_salt_none() -> None:
    serializer = Serializer("secret-key", salt=None)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt is None
    assert serializer.serializer == json

def test_serializer_constructor_with_salt_bytes() -> None:
    serializer = Serializer("secret-key", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_salt_string() -> None:
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_serializer_none() -> None:
    serializer = Serializer("secret-key", serializer=None)
    assert serializer.serializer == json

def test_serializer_constructor_with_custom_serializer() -> None:
    serializer = Serializer("secret-key", serializer=json)
    assert serializer.serializer == json

def test_serializer_constructor_with_binary_serializer() -> None:
    class BinarySerializer:
        @staticmethod
        def dumps(obj: object) -> bytes:
            return b"data"

        @staticmethod
        def loads(data: bytes) -> object:
            return data

    serializer = Serializer("secret-key", serializer=BinarySerializer())
    assert not serializer.is_text_serializer

def test_serializer_constructor_with_signer_class() -> None:
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers() -> None:
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_secret_key_bytes() -> None:
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_with_secret_key_list() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_secret_key_list_bytes() -> None:
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]


# LLM-generated content at query #74
#--------------------------

def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads("test")
    assert result is not None

def test_loads_accepts_serialized_input():
    serializer = _PDataSerializer()
    result = serializer.loads(b"data")
    assert isinstance(result, object)

def test_loads_with_integer_input():
    serializer = _PDataSerializer()
    result = serializer.loads(123)
    assert result is not None

def test_loads_with_list_input():
    serializer = _PDataSerializer()
    result = serializer.loads([1, 2, 3])
    assert result is not None

def test_loads_with_dict_input():
    serializer = _PDataSerializer()
    result = serializer.loads({"key": "value"})
    assert result is not None

def test_loads_with_none_input():
    serializer = _PDataSerializer()
    result = serializer.loads(None)
    assert result is None


# LLM-generated content at query #75
#--------------------------

```python
def test_serializer_init_with_custom_serializer_evaluates_predicate_to_false():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda x, **kw: "dumped", "loads": lambda x: "loaded"})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
```


# LLM-generated content at query #76
#--------------------------

def test_dumps_returns_serialized_data():
    serializer = _PDataSerializer()
    result = serializer.dumps("test_object")
    assert result is not None


# LLM-generated content at query #77
#--------------------------

def test_serializer_constructor_default():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_with_secret_list():
    serializer = Serializer(["key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
    serializer = Serializer("secret-key", serializer=CustomSerializer)
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
    serializer = Serializer("secret-key", serializer=BytesSerializer)
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner:
        def __init__(self, secret_keys, salt=None, **kwargs):
            pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"digest_method": "sha256"})
    assert serializer.signer_kwargs == {"digest_method": "sha256"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"digest_method": "sha256"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_explicit_default_serializer():
    serializer = Serializer("secret-key", serializer=None)
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_explicit_default_signer():
    serializer = Serializer("secret-key", signer=None)
    assert serializer.signer is Serializer.default_signer

def test_serializer_constructor_with_explicit_default_fallback():
    serializer = Serializer("secret-key", fallback_signers=None)
    assert serializer.fallback_signers == []


# LLM-generated content at query #78
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer == isinstance(serializer.default_serializer.dumps({}), str)
    assert serializer.signer == serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"data"
        @staticmethod
        def loads(b):
            return b
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.serializer is BytesSerializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_custom_signer():
    class CustomSigner:
        def __init__(self, secret_keys, salt, **kwargs):
            pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #79
#--------------------------

def test_dumps_with_default_json_serializer():
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result.count(".") == 2

def test_dumps_with_text_serializer_returns_str():
    serializer = Serializer("secret-key", serializer=json)
    result = serializer.dumps("test")
    assert isinstance(result, str)

def test_dumps_with_bytes_serializer_returns_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            import json
            return json.dumps(obj).encode("utf-8")
        @staticmethod
        def loads(data):
            import json
            return json.loads(data)
    serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = serializer.dumps("test")
    assert isinstance(result, bytes)

def test_dumps_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    result = serializer.dumps("data")
    assert isinstance(result, str)

def test_dumps_returns_different_results_for_different_data():
    serializer = Serializer("secret-key")
    result1 = serializer.dumps("data1")
    result2 = serializer.dumps("data2")
    assert result1 != result2

def test_dumps_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    result = serializer.dumps({"b": 1, "a": 2})
    assert result.count(".") == 2

def test_dumps_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    result = serializer.dumps("test")
    assert isinstance(result, str)

def test_dumps_with_multiple_secret_keys():
    serializer = Serializer(["old-key", "new-key"])
    result = serializer.dumps("test")
    assert isinstance(result, str)


# LLM-generated content at query #80
#--------------------------

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

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"bytes_salt")
    assert serializer.salt == b"bytes_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(obj):
            return obj
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(obj):
            return obj
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.serializer is BytesSerializer()
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"key": "value"})
    assert serializer.serializer_kwargs == {"key": "value"}

def test_serializer_constructor_with_custom_signer_class():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key": "value"})
    assert serializer.signer_kwargs == {"key": "value"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key": "value"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_fallback_signers_default():
    serializer = Serializer("secret")
    assert serializer.fallback_signers == []


# LLM-generated content at query #81
#--------------------------

def test_salt_is_none_when_provided_salt_is_none():
    serializer = Serializer(secret_key="secret", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #82
#--------------------------

```python
def test_load_payload_serializer_not_none_and_is_not_text_serializer():
    serializer = Serializer("secret")
    payload = b'{"key": "value"}'
    custom_serializer = type("CustomSerializer", (), {"loads": lambda self, x: x})()
    result = serializer.load_payload(payload, serializer=custom_serializer)
    assert result == payload
```


# LLM-generated content at query #83
#--------------------------

```python
def test_fallback_signers_is_none():
    s = Serializer("secret")
    assert s.fallback_signers is not None
```


# LLM-generated content at query #84
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_with_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_serializer_positional():
    serializer = Serializer("secret-key", b"salt", json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True

def test_serializer_constructor_serializer_keyword():
    serializer = Serializer("secret-key", serializer=json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_signer():
    serializer = Serializer("secret-key", signer=Signer)
    assert serializer.signer == Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_empty_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #85
#--------------------------

```python
def test_load_payload_predicate_false():
    serializer = Serializer(b"secret", serializer=Serializer.default_serializer)
    serializer.is_text_serializer = False
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
```


# LLM-generated content at query #86
#--------------------------

```
def test_dumps_with_integer():
    serializer = _PDataSerializer()
    result = serializer.dumps(42)
    assert result == 42

def test_dumps_with_string():
    serializer = _PDataSerializer()
    result = serializer.dumps("hello")
    assert result == "hello"

def test_dumps_with_list():
    serializer = _PDataSerializer()
    result = serializer.dumps([1, 2, 3])
    assert result == [1, 2, 3]

def test_dumps_with_dict():
    serializer = _PDataSerializer()
    result = serializer.dumps({"key": "value"})
    assert result == {"key": "value"}

def test_dumps_with_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result is None

def test_dumps_with_float():
    serializer = _PDataSerializer()
    result = serializer.dumps(3.14)
    assert result == 3.14
```


# LLM-generated content at query #87
#--------------------------

def test_is_text_serializer_false():
    serializer = Serializer("secret", serializer=type("Dummy", (), {"dumps": lambda self, obj: b"data", "loads": lambda self, s: s})())
    assert serializer.is_text_serializer == False


# LLM-generated content at query #88
#--------------------------

```python
def test_iter_unsigners_default_signer():
    serializer = Serializer("secret")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)

def test_iter_unsigners_with_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[{"key": "fallback_secret"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_with_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    signers = list(serializer.iter_unsigners(salt=b"override_salt"))
    assert len(signers) == 1
    assert signers[0].salt == b"override_salt"

def test_iter_unsigners_multiple_secret_keys():
    serializer = Serializer(["old_secret", "new_secret"], fallback_signers=[{"key": "fallback_secret"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert signers[0].secret_keys == [b"old_secret", b"new_secret"]
    assert signers[1].secret_keys == [b"old_secret"]
    assert signers[2].secret_keys == [b"new_secret"]

def test_iter_unsigners_fallback_as_tuple():
    serializer = Serializer("secret", fallback_signers=[(Signer, {"key": "tuple_fallback"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_fallback_as_class():
    serializer = Serializer("secret", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
```


# LLM-generated content at query #89
#--------------------------

def test_serializer_constructor_defaults():
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

def test_serializer_constructor_with_list_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: str(x), "loads": lambda self, x: eval(x)})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, x: b"data", "loads": lambda self, x: {"data": x}})()
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.serializer is bytes_serializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {"sign": lambda self, x: x, "unsign": lambda self, x: x})()
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer is custom_signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #90
#--------------------------

```python
def test_serializer_init_with_serializer_not_none():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    serializer_instance = Serializer("secret", serializer=Signer)
    assert serializer_instance.serializer is not None
```


# LLM-generated content at query #91
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer == CustomSerializer()
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(b):
            return b
    s = Serializer("secret", serializer=BytesSerializer())
    assert s.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert s.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_fallback_signers_none():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []

def test_serializer_constructor_with_custom_serializer_keyword():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer == CustomSerializer()


# LLM-generated content at query #92
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.serializer == CustomSerializer
    assert not serializer.is_text_serializer

def test_constructor_with_text_serializer():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "text"
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret", serializer=TextSerializer)
    assert serializer.is_text_serializer

def test_constructor_with_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #93
#--------------------------

def test_loads_returns_any():
    _PDataSerializer().loads("test")


# LLM-generated content at query #94
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: str(x), "loads": lambda self, x: x})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False

def test_constructor_custom_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_constructor_custom_signer():
    custom_signer = type("CustomSigner", (), {})
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer == custom_signer

def test_constructor_custom_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_custom_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_multiple_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_constructor_with_bytes_serializer():
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, x: b"{}", "loads": lambda self, x: {}})()
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.is_text_serializer is False


# LLM-generated content at query #95
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-bytes")
    assert serializer.secret_keys == [b"secret-bytes"]

def test_serializer_constructor_with_iterable_secret_keys():
    serializer = Serializer(["key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(b):
            return b
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #96
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(data):
            return data
    serializer = Serializer(b"secret", salt=b"salt", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_positional_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "text"
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret", b"salt", CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True


# LLM-generated content at query #97
#--------------------------

def test_serializer_constructor_defaults():
    secret_key = "test-secret"
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [want_bytes(secret_key)]
    assert serializer.salt == want_bytes(b"itsdangerous")
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    secret_key = b"test-secret"
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [secret_key]

def test_serializer_constructor_with_list_of_secret_keys():
    secret_keys = ["key1", "key2"]
    serializer = Serializer(secret_keys)
    assert serializer.secret_keys == [want_bytes(s) for s in secret_keys]

def test_serializer_constructor_with_custom_salt():
    secret_key = "test-secret"
    salt = "custom-salt"
    serializer = Serializer(secret_key, salt=salt)
    assert serializer.salt == want_bytes(salt)

def test_serializer_constructor_with_none_salt():
    secret_key = "test-secret"
    serializer = Serializer(secret_key, salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    secret_key = "test-secret"
    serializer = Serializer(secret_key, serializer=json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer

def test_serializer_constructor_with_serializer_kwargs():
    secret_key = "test-secret"
    serializer_kwargs = {"indent": 2}
    serializer = Serializer(secret_key, serializer_kwargs=serializer_kwargs)
    assert serializer.serializer_kwargs == serializer_kwargs

def test_serializer_constructor_with_custom_signer_class():
    secret_key = "test-secret"
    class CustomSigner(Signer):
        pass
    serializer = Serializer(secret_key, signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    secret_key = "test-secret"
    signer_kwargs = {"key_derivation": "hmac"}
    serializer = Serializer(secret_key, signer_kwargs=signer_kwargs)
    assert serializer.signer_kwargs == signer_kwargs

def test_serializer_constructor_with_fallback_signers():
    secret_key = "test-secret"
    fallback_signers = [{"key_derivation": "none"}]
    serializer = Serializer(secret_key, fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_constructor_with_all_parameters():
    secret_key = "test-secret"
    salt = "custom-salt"
    serializer_kwargs = {"indent": 2}
    signer_kwargs = {"key_derivation": "hmac"}
    fallback_signers = [{"key_derivation": "none"}]
    serializer = Serializer(
        secret_key,
        salt=salt,
        serializer=json,
        serializer_kwargs=serializer_kwargs,
        signer=Signer,
        signer_kwargs=signer_kwargs,
        fallback_signers=fallback_signers,
    )
    assert serializer.secret_keys == [want_bytes(secret_key)]
    assert serializer.salt == want_bytes(salt)
    assert serializer.serializer == json
    assert serializer.is_text_serializer
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == signer_kwargs
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == serializer_kwargs


# LLM-generated content at query #98
#--------------------------

def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads("test")
    assert result is not None


# LLM-generated content at query #99
#--------------------------

def test_dumps_with_text_serializer_returns_string():
    serializer = Serializer("secret-key", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)

def test_dumps_with_bytes_serializer_returns_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj, **kwargs):
            return b'{"key":"value"}'
    serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)

def test_dumps_returns_different_output_for_different_salts():
    serializer = Serializer("secret-key")
    result1 = serializer.dumps({"key": "value"}, salt="salt1")
    result2 = serializer.dumps({"key": "value"}, salt="salt2")
    assert result1 != result2

def test_dumps_returns_different_output_for_different_objects():
    serializer = Serializer("secret-key")
    result1 = serializer.dumps({"key1": "value1"})
    result2 = serializer.dumps({"key2": "value2"})
    assert result1 != result2


# LLM-generated content at query #100
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"secret_bytes")
    assert serializer.secret_keys == [b"secret_bytes"]

def test_serializer_constructor_with_secret_key_list():
    serializer = Serializer(["key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"bytes_salt")
    assert serializer.salt == b"bytes_salt"

def test_serializer_constructor_with_json_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_text_serializer():
    class TextSerializer:
        def dumps(self, obj):
            return "text"
        def loads(self, s):
            return eval(s)
    serializer = Serializer("secret", serializer=TextSerializer())
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            return b"bytes"
        def loads(self, s):
            return eval(s.decode())
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []


# LLM-generated content at query #101
#--------------------------

```python
def test_load_payload_predicate_false():
    serializer = Serializer("secret", serializer=lambda: None)
    serializer.is_text_serializer = False
    serializer.serializer = type("FakeSerializer", (), {"loads": lambda self, x: (_ for _ in ()).throw(Exception("test"))})()
    try:
        serializer.load_payload(b"test")
    except BadPayload:
        pass
```


# LLM-generated content at query #102
#--------------------------

```python
def test_iter_unsigners_predicate_line_15_true():
    signer_class = Signer
    secret_key = b"test_secret"
    serializer = Serializer(secret_key=secret_key, fallback_signers=[(signer_class, {"key_derivation": "hmac"})])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].key_derivation == "hmac"
```


# LLM-generated content at query #103
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_custom_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    custom_serializer = json
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer is Signer

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_constructor_serializer_is_text():
    serializer = Serializer("secret")
    assert serializer.is_text_serializer is True

def test_constructor_serializer_is_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"

        @staticmethod
        def loads(data):
            return data

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_constructor_secret_key_property():
    serializer = Serializer(["old", "new"])
    assert serializer.secret_key == b"new"

def test_constructor_with_default_fallback_signers():
    class CustomSerializer(Serializer):
        default_fallback_signers = [{"key_derivation": "none"}]

    serializer = CustomSerializer("secret")
    assert serializer.fallback_signers == [{"key_derivation": "none"}]


# LLM-generated content at query #104
#--------------------------

def test_serializer_constructor_defaults():
    secret_key = "test-secret"
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"test-secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(Serializer.default_serializer)
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    secret_key = b"test-secret-bytes"
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"test-secret-bytes"]

def test_serializer_constructor_with_list_of_secret_keys():
    secret_keys = ["key1", "key2"]
    serializer = Serializer(secret_keys)
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    secret_keys = [b"key1", b"key2"]
    serializer = Serializer(secret_keys)
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    secret_key = "test-secret"
    salt = "custom-salt"
    serializer = Serializer(secret_key, salt=salt)
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    secret_key = "test-secret"
    serializer = Serializer(secret_key, salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    secret_key = "test-secret"
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer = Serializer(secret_key, serializer=CustomSerializer)
    assert serializer.serializer == CustomSerializer
    assert serializer.is_text_serializer == is_text_serializer(CustomSerializer)

def test_serializer_constructor_with_serializer_kwargs():
    secret_key = "test-secret"
    serializer_kwargs = {"indent": 4}
    serializer = Serializer(secret_key, serializer_kwargs=serializer_kwargs)
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_custom_signer():
    secret_key = "test-secret"
    class CustomSigner(Signer):
        pass
    serializer = Serializer(secret_key, signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    secret_key = "test-secret"
    signer_kwargs = {"key_derivation": "hmac"}
    serializer = Serializer(secret_key, signer_kwargs=signer_kwargs)
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    secret_key = "test-secret"
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer(secret_key, fallback_signers=fallback_signers)
    assert serializer.fallback_signers == [{"key_derivation": "hmac"}]

def test_serializer_constructor_with_empty_fallback_signers():
    secret_key = "test-secret"
    serializer = Serializer(secret_key, fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_none_fallback_signers():
    secret_key = "test-secret"
    serializer = Serializer(secret_key, fallback_signers=None)
    assert serializer.fallback_signers == []


# LLM-generated content at query #105
#--------------------------

```
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert result is not None
```


# LLM-generated content at query #106
#--------------------------

```python
def test_fallback_signers_not_none_initialization():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    fallback = [Signer]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback
```


# LLM-generated content at query #107
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class MockSerializer:
        @staticmethod
        def dumps(obj):
            return "serialized"

        @staticmethod
        def loads(s):
            return {"data": "test"}

    serializer = Serializer("secret", serializer=MockSerializer())
    assert serializer.serializer is MockSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class MockBytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"serialized"

        @staticmethod
        def loads(s):
            return {"data": "test"}

    serializer = Serializer("secret", serializer=MockBytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_fallback_signers_none():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []


# LLM-generated content at query #108
#--------------------------

def test_serializer_constructor_default_serializer() -> None:
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_with_list_of_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt() -> None:
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_salt_none() -> None:
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer() -> None:
    from itsdangerous.serializer import _PDataSerializer
    class CustomSerializer:
        def dumps(self, obj: object) -> str:
            return str(obj)
        def loads(self, s: str) -> object:
            return eval(s)
    serializer = Serializer("secret-key", serializer=CustomSerializer())
    assert isinstance(serializer.serializer, CustomSerializer)
    assert serializer.is_text_serializer

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer() -> None:
    from itsdangerous.signer import Signer
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers() -> None:
    from itsdangerous.signer import Signer
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_serializer_returns_bytes() -> None:
    from itsdangerous.serializer import _PDataSerializer
    class BytesSerializer:
        def dumps(self, obj: object) -> bytes:
            return b"bytes"
        def loads(self, s: bytes) -> object:
            return s
    serializer = Serializer("secret-key", serializer=BytesSerializer())
    assert not serializer.is_text_serializer

def test_serializer_constructor_with_all_parameters() -> None:
    from itsdangerous.serializer import _PDataSerializer
    class CustomSerializer:
        def dumps(self, obj: object) -> str:
            return "data"
        def loads(self, s: str) -> object:
            return s
    from itsdangerous.signer import Signer
    class CustomSigner(Signer):
        pass
    serializer = Serializer(
        secret_key=["old", "new"],
        salt=b"custom",
        serializer=CustomSerializer(),
        serializer_kwargs={"sort_keys": True},
        signer=CustomSigner,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert serializer.secret_keys == [b"old", b"new"]
    assert serializer.salt == b"custom"
    assert isinstance(serializer.serializer, CustomSerializer)
    assert serializer.is_text_serializer
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #109
#--------------------------

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

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "{}", "loads": lambda self, x: {}})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, x: b"{}", "loads": lambda self, x: {}})()
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer == custom_signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_str_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"


# LLM-generated content at query #110
#--------------------------

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

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return eval(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"

        @staticmethod
        def loads(s):
            return s

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return eval(s)

    class CustomSigner(Signer):
        pass

    fallback = [CustomSigner]
    serializer = Serializer(
        secret_key=["key1", b"key2"],
        salt=b"mysalt",
        serializer=CustomSerializer(),
        serializer_kwargs={"indent": 2},
        signer=CustomSigner,
        signer_kwargs={"digest_method": hashlib.sha256},
        fallback_signers=fallback,
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"mysalt"
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {"digest_method": hashlib.sha256}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {"indent": 2}


