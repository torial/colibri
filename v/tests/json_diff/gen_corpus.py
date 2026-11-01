#!/usr/bin/env python3
"""Corpus for the json differential test: hand-written edge cases, random valid documents, and byte-level
mutations of both (flips, truncations, insertions of structural and control bytes). usage: gen_corpus.py DIR [N] [SEED]"""
import os, random, sys, json
out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000; random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
os.makedirs(out, exist_ok=True)
edge = [b'', b' ', b'null', b'true', b'false', b'tru', b'nul', b'0', b'-0', b'-', b'+1', b'.5', b'1.', b'1e', b'1e+', b'1E5', b'0123', b'0x1p3',
 b'inf', b'-inf', b'nan', b'1e9999', b'-1e9999', b'1e-9999', b'123456789012345678901234567890', b'0.1', b'3.141592653589793238',
 b'"', b'""', b'"a', b'"a\\', b'"\\u', b'"\\u12', b'"\\u0041"', b'"\\u00e9"', b'"\\u20ac"', b'"\\ud83d\\ude00"', b'"\\ud83d"', b'"\\ud83dx"',
 b'"\\ude00"', b'"\\ud83d\\u0041"', b'"\\ud800\\udc00"', b'"\\udbff\\udfff"', b'"\\ud800\\udbff"', b'"\\ud800\\ud800\\udc00"', b'"\\uD83D\\uDE00"', b'"\\ud800\\udbff\\udfff"', b'"\\u007f\\u0080\\u07ff\\u0800\\uffff"', b'"\\u+123"', b'"\\u 123"', b'"\\u0x12"', b'"\\u-001"', b'"\\uzzzz"', b'"\\u0000"', b'{"\\u0000":1}', b'"\\q"',
 b'"\\n\\t\\r\\b\\f\\/\\\\\\""', b'"tab\there"', b'"a\x01b"', b'"\xff\xfe"', b'[', b']', b'[]', b'[,]', b'[1,]', b'[1 2]', b'[1,2', b'{', b'{}', b'{"a"}',
 b'{"a":}', b'{"a":1,}', b'{a:1}', b'{"a":1 "b":2}', b'{"a":1,"a":2}', b'{"a":{"a":{"a":[]}}}', b' \t\r\n [ 1 , 2 ] \n', b'\x0b[1]', b'[\x0c1]',
 b'[1]x', b'[1] ', b'[1]\x00[2]', b'"a\x00b"', b'[1,\x002]', b'1 2', b'{"k":"v"} trailing', b'[' * 1030 + b']' * 1030, b'[' * 1025, b'[' * 1024 + b']' * 1024,
 b'{"a":' * 1025, b'{"a":' * 1000 + b'1' + b'}' * 1000]
for i, e in enumerate(edge): open(f'{out}/e{i:04d}.json', 'wb').write(e)
def rnd(d=0):
    k = random.random()
    if d > 4 or k < 0.45:
        c = random.choice(['n', 'b', 'i', 'f', 's', 's'])
        if c == 'n': return None
        if c == 'b': return random.random() < .5
        if c == 'i': return random.choice([0, 1, -1, 7, 255, 2**31, -2**31, 2**53, random.randint(-10**9, 10**9)])
        if c == 'f': return random.choice([0.5, -0.25, 1e-7, 1e22, 3.14159, random.uniform(-1e6, 1e6), random.uniform(-1, 1) * 10 ** random.randint(-30, 30)])
        return ''.join(random.choice(['a', 'b', ' ', '"', '\\', '\n', 'é', '€', '😀', '\x01', '/', 'z']) for _ in range(random.randint(0, 12)))
    if k < 0.72: return [rnd(d + 1) for _ in range(random.randint(0, 5))]
    return {''.join(random.choice('abcxyz_é') for _ in range(random.randint(0, 4))): rnd(d + 1) for _ in range(random.randint(0, 5))}
def enc(v):
    return json.dumps(v, ensure_ascii=random.random() < .5, separators=random.choice([(',', ':'), (', ', ': ')]), indent=random.choice([None, None, 1])).encode()
mut_bytes = [b'"', b'\\', b'{', b'}', b'[', b']', b',', b':', b'\x00', b'\x01', b'\n', b'\x0b', b'-', b'e', b'.', b'u', b'0', b'x', b'\xff', b'+']
for i in range(n):
    b = bytearray(enc(rnd()))
    for _ in range(random.choice([0, 0, 1, 1, 2, 3])):
        if not b: break
        m = random.random(); p = random.randrange(len(b) + 1)
        if m < .3 and p < len(b): b[p] = random.choice(mut_bytes)[0]
        elif m < .55: b[p:p] = random.choice(mut_bytes)
        elif m < .75 and p < len(b): del b[p]
        else: del b[p:]
    open(f'{out}/r{i:05d}.json', 'wb').write(bytes(b))
print(len(edge), 'edge cases +', n, 'random ->', out)
