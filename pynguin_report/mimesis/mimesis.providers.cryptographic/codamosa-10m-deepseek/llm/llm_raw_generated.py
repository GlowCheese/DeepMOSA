####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for method certificate_fingerprint of class Cryptographic
def test_Cryptographic_certificate_fingerprint():  # noqa: N802
    """Test method certificate_fingerprint of class Cryptographic."""
    crypto = Cryptographic()
    # Test with default algorithm (sha256)
    fingerprint = crypto.certificate_fingerprint()
    assert isinstance(fingerprint, str)
    # Should be colon-separated hex pairs
    parts = fingerprint.split(":")
    assert len(parts) == 32  # 256 bits = 32 bytes = 64 hex chars = 32 pairs
    for part in parts:
        assert len(part) == 2
        assert all(c in "0123456789ABCDEF" for c in part)
    # Test with sha1
    fingerprint = crypto.certificate_fingerprint(algorithm="sha1")
    assert isinstance(fingerprint, str)
    parts = fingerprint.split(":")
    assert len(parts) == 20  # 160 bits = 20 bytes = 40 hex chars = 20 pairs
    for part in parts:
        assert len(part) == 2
        assert all(c in "0123456789ABCDEF" for c in part)
    # Test with unsupported algorithm
    try:
        crypto.certificate_fingerprint(algorithm="md5")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Unknown algorithm" in str(e)


# LLM-generated content at query #2
#--------------------------

# Unit test for method api_key of class Cryptographic
def test_Cryptographic_api_key():  
    # Test case 1: Default parameters
    crypto = Cryptographic()
    key = crypto.api_key()
    assert isinstance(key, str)
    assert len(key) == 64  # 32 bytes in hex = 64 characters
    
    # Test case 2: With prefix
    key = crypto.api_key(prefix='sk_')
    assert key.startswith('sk_')
    assert len(key) == 64 + 3  # 64 hex chars + 3 prefix chars
    
    # Test case 3: With base64 format
    key = crypto.api_key(fmt='base64')
    assert isinstance(key, str)
    # Base64 length can vary, but should be roughly length * 4/3
    assert 40 <= len(key) <= 50
    
    # Test case 4: With prefix and base64
    key = crypto.api_key(prefix='pk_', fmt='base64')
    assert key.startswith('pk_')
    
    # Test case 5: Invalid format raises ValueError
    try:
        crypto.api_key(fmt='invalid')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# LLM-generated content at query #3
#--------------------------

# Unit test for method api_key of class Cryptographic
def test_Cryptographic_api_key():  # noqa: N802
    crypto = Cryptographic()
    # Test with default parameters
    key = crypto.api_key()
    assert isinstance(key, str)
    assert len(key) == 32  # 32 hex characters = 16 bytes
    # Test with prefix
    key = crypto.api_key(prefix='sk_')
    assert key.startswith('sk_')
    # Test with base64 format
    key = crypto.api_key(fmt='base64')
    assert isinstance(key, str)
    # Test with length
    key = crypto.api_key(length=16)
    assert len(key) == 16
    # Test with prefix and base64
    key = crypto.api_key(prefix='pk_', fmt='base64')
    assert key.startswith('pk_')
    # Test invalid format
    try:
        crypto.api_key(fmt='invalid')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass



# LLM-generated content at query #4
#--------------------------

# Unit test for method certificate_fingerprint of class Cryptographic
def test_Cryptographic_certificate_fingerprint():  # noqa: N802
    """Test method certificate_fingerprint of class Cryptographic."""
    # Test with default algorithm (sha256)
    crypto = Cryptographic()
    fingerprint = crypto.certificate_fingerprint()
    assert isinstance(fingerprint, str)
    # Check format: colon-separated hex pairs, uppercase
    parts = fingerprint.split(":")
    assert len(parts) == 32  # 256 bits = 32 bytes
    for part in parts:
        assert len(part) == 2
        assert part.isupper()
        int(part, 16)  # Should be valid hex

    # Test with sha1
    fingerprint = crypto.certificate_fingerprint(algorithm="sha1")
    parts = fingerprint.split(":")
    assert len(parts) == 20  # 160 bits = 20 bytes
    for part in parts:
        assert len(part) == 2
        assert part.isupper()
        int(part, 16)

    # Test with invalid algorithm
    try:
        crypto.certificate_fingerprint(algorithm="md5")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Unknown algorithm" in str(e)


# LLM-generated content at query #5
#--------------------------

# Unit test for method certificate_fingerprint of class Cryptographic
def test_Cryptographic_certificate_fingerprint():  # noqa: N802
    """Test method certificate_fingerprint of class Cryptographic."""
    # Test with default algorithm (sha256)
    crypto = Cryptographic()
    fingerprint = crypto.certificate_fingerprint()
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 95  # 32 bytes * 2 hex digits + 31 colons
    assert all(c in "0123456789ABCDEF:" for c in fingerprint)
    assert fingerprint.count(":") == 31

    # Test with sha1 algorithm
    fingerprint = crypto.certificate_fingerprint(algorithm="sha1")
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 59  # 20 bytes * 2 hex digits + 19 colons
    assert all(c in "0123456789ABCDEF:" for c in fingerprint)
    assert fingerprint.count(":") == 19

    # Test with invalid algorithm
    try:
        crypto.certificate_fingerprint(algorithm="invalid")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Unknown algorithm" in str(e)


####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for method certificate_fingerprint of class Cryptographic
def test_Cryptographic_certificate_fingerprint():  # noqa: N802
    """Test method certificate_fingerprint of class Cryptographic."""
    crypto = Cryptographic()
    # Test with default algorithm (sha256)
    fingerprint = crypto.certificate_fingerprint()
    assert isinstance(fingerprint, str)
    # Should be colon-separated hex pairs
    parts = fingerprint.split(":")
    assert len(parts) == 32  # 256 bits = 32 bytes
    for part in parts:
        assert len(part) == 2
        int(part, 16)  # Should be valid hex

    # Test with sha1
    fingerprint = crypto.certificate_fingerprint(algorithm="sha1")
    parts = fingerprint.split(":")
    assert len(parts) == 20  # 160 bits = 20 bytes
    for part in parts:
        assert len(part) == 2
        int(part, 16)

    # Test with unsupported algorithm
    try:
        crypto.certificate_fingerprint(algorithm="md5")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown algorithm" in str(e)


# LLM-generated content at query #2
#--------------------------

# Unit test for method api_key of class Cryptographic
def test_Cryptographic_api_key():  
    # Test case 1: default parameters
    crypto = Cryptographic()
    result = crypto.api_key()
    assert isinstance(result, str)
    assert len(result) == 64  # 32 bytes in hex = 64 characters
    assert all(c in '0123456789abcdef' for c in result)

    # Test case 2: with prefix
    result = crypto.api_key(prefix='sk_')
    assert result.startswith('sk_')
    assert len(result) == 64 + 3  # 64 hex chars + 3 prefix chars

    # Test case 3: with base64 format
    result = crypto.api_key(fmt='base64')
    assert isinstance(result, str)
    # Base64 length may vary slightly due to padding removal
    assert 40 <= len(result) <= 44  # 32 bytes in base64

    # Test case 4: with prefix and base64 format
    result = crypto.api_key(prefix='pk_', fmt='base64')
    assert result.startswith('pk_')
    assert 40 + 3 <= len(result) <= 44 + 3

    # Test case 5: custom length
    result = crypto.api_key(length=16)
    assert len(result) == 32  # 16 bytes in hex = 32 characters

    # Test case 6: invalid format should raise ValueError
    try:
        crypto.api_key(fmt='invalid')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown format" in str(e)


# LLM-generated content at query #3
#--------------------------

# Unit test for method certificate_fingerprint of class Cryptographic
def test_Cryptographic_certificate_fingerprint():  # noqa: N802
    """Test certificate_fingerprint method."""
    crypto = Cryptographic()
    # Test with default algorithm (sha256)
    fingerprint = crypto.certificate_fingerprint()
    assert isinstance(fingerprint, str)
    # Check format: colon-separated hex pairs, uppercase
    parts = fingerprint.split(":")
    assert len(parts) == 32  # 256 bits = 32 bytes = 64 hex chars -> 32 pairs
    for part in parts:
        assert len(part) == 2
        assert all(c in "0123456789ABCDEF" for c in part)
    # Test with sha1
    fingerprint_sha1 = crypto.certificate_fingerprint(algorithm="sha1")
    parts_sha1 = fingerprint_sha1.split(":")
    assert len(parts_sha1) == 20  # 160 bits = 20 bytes = 40 hex chars -> 20 pairs
    for part in parts_sha1:
        assert len(part) == 2
        assert all(c in "0123456789ABCDEF" for c in part)
    # Test with unsupported algorithm
    try:
        crypto.certificate_fingerprint(algorithm="md5")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Unknown algorithm" in str(e)


# LLM-generated content at query #4
#--------------------------

# Unit test for method certificate_fingerprint of class Cryptographic
def test_Cryptographic_certificate_fingerprint():  # noqa: N802
    """Test method certificate_fingerprint of class Cryptographic."""
    # Setup
    crypto = Cryptographic()
    # Exercise
    result = crypto.certificate_fingerprint()
    # Verify
    assert isinstance(result, str)
    assert len(result) == 95  # 32 bytes * 2 hex digits + 31 colons
    assert all(c in "0123456789ABCDEF:" for c in result)
    # Cleanup - none necessary



# LLM-generated content at query #5
#--------------------------

# Unit test for method certificate_fingerprint of class Cryptographic
def test_Cryptographic_certificate_fingerprint():  # noqa: N802
    """Test method certificate_fingerprint of class Cryptographic."""
    crypto = Cryptographic()
    # Test with default algorithm (sha256)
    fingerprint = crypto.certificate_fingerprint()
    assert isinstance(fingerprint, str)
    # Should be colon-separated hex pairs
    parts = fingerprint.split(":")
    assert len(parts) == 32  # 256 bits = 32 bytes = 32 hex pairs
    assert all(len(part) == 2 for part in parts)
    assert all(c in "0123456789ABCDEF" for part in parts for c in part)
    # Test with sha1
    fingerprint = crypto.certificate_fingerprint(algorithm="sha1")
    assert isinstance(fingerprint, str)
    parts = fingerprint.split(":")
    assert len(parts) == 20  # 160 bits = 20 bytes = 20 hex pairs
    assert all(len(part) == 2 for part in parts)
    assert all(c in "0123456789ABCDEF" for part in parts for c in part)
    # Test with invalid algorithm
    try:
        crypto.certificate_fingerprint(algorithm="md5")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown algorithm" in str(e)


