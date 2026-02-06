## Problem 1
(a) What Unicode character does chr(0) return?
'\x00'
(b) How does this character’s string representation (__repr__()) differ from its printed representa-
tion?
print is empty, repr shows a string representation of its byte contents
(c) What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:
```
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
The string object has an all-zero byte wherever `chr(0)` appears, but this byte is not displayed in a human-readable form, such as via `print`.

## Problem 2

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.

The range of possible values for utf-16 (0-2^16) and utf-32 (0-2^32) are much larger than utf-8 (0-2^8), making algorithms such as BPE less efficient. Also, it seems encoding to utf-16 and utf-32 also uses up more bytes for the same underlying string, further reducing efficiency of processing these strings.

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into
a Unicode string. Why is this function incorrect? Provide an example of an input byte string
that yields incorrect results.
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'

Example: "'hello! こんにちは!'", and it fails because it doesn't properly handle characters represented by more than 1 byte.

(c) Give a two byte sequence that does not decode to any Unicode character(s).
"b'\xe3\x81'"

These are just the first two bytes of "は", and they don't make sense without the third byte


## Problem (train_bpe_tinystories): BPE Training on TinyStories

(a) 
(b) Pre-tokenising the raw corpus takes the longest
