#!/usr/bin/env python3
"""Differential test of v/tokenizer against c/tok.h. For every pre-tokenizer family (cl100k, Qwen one-digit, o200k, Tekken, Kimi,
GPT-2, DeepSeek, and the merge-less rank-BPE form) a tokenizer.json is built here (a byte-level BPE trained on a small corpus, so
real merges happen, plus added tokens), and many texts (words, digits, punctuation, contractions, whitespace runs, accents, CJK, emoji,
invalid UTF-8, added tokens) are encoded and decoded by both; every id and every decoded byte must agree.
usage: run.py [texts per family] [seed]"""
import json, os, random, shutil, subprocess, sys, tempfile
from collections import Counter
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300; random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.abspath(os.path.join(here, '..', '..', '..')); v_dir = os.path.join(root, 'v')
w = tempfile.mkdtemp(prefix='tok_diff.')
subprocess.check_call(['gcc', '-O1', '-I' + os.path.join(root, 'c'), '-o', w + '/c_dump', here + '/c_dump.c', '-lm'])
subprocess.check_call(['v', '-o', w + '/v_dump', 'tests/tok_diff/v_dump.v'], cwd=v_dir, stdout=subprocess.DEVNULL)
# GPT-2 byte <-> unicode map
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
B2U = {b: chr(c) for b, c in zip(bs, cs)}
WORDS = ['the', 'The', 'THE', 'hello', 'World', 'don', 't', 'it', 's', 'we', 're', 'I', 'm', 'll', 'd', 've', 'naïve', 'café', 'Über', 'straße', 'ǅungla', 'x', 'ab', 'abc', 'ABc', 'aBC', 'camelCase', 'snake_case', 'a1b2', '你好', '世界', 'こんにちは', 'カタカナ', 'привет', 'ñandú', 'مرحبا', 'résumé', '😀', '🎉🎉', '́', 'é', '²', '٣', '٤٥٦', '½', 'ⅷ']
PUNCT = ['.', ',', '!', '?', "'", '"', '(', ')', '[', ']', '{', '}', '/', '\\', '-', '_', '+', '=', '*', '&', '$', '#', '@', '~', '^', '|', '<', '>', ':', ';', '`', '…', '—', '€', '©', '»']
def rnd_text():
    parts = []
    for _ in range(random.randint(1, 14)):
        k = random.random()
        if k < .42: parts.append(random.choice(WORDS))
        elif k < .52: parts.append(str(random.choice([0, 7, 42, 123, 4567, 89012, 3141592653])))
        elif k < .67: parts.append(''.join(random.choice(PUNCT) for _ in range(random.randint(1, 4))))
        elif k < .78: parts.append(random.choice([' ', '  ', '   ', '\n', '\n\n', ' \n', '\r\n', '\t', ' \t ', ' ', ' ', '\n  ']))
        elif k < .84: parts.append(random.choice(["'s", "'S", "'t", "'re", "'RE", "'ve", "'m", "'ll", "'d", "'D", "'x", "'"]))
        elif k < .88: parts.append(random.choice(['<|endoftext|>', '<think>', '</think>']))
        else: parts.append(random.choice(['', ' ']))
        if random.random() < .6: parts.append(' ')
    s = ''.join(parts).encode()
    if random.random() < .08:
        b = bytearray(s); 
        for _ in range(random.randint(1, 3)): b[random.randrange(len(b) + 1):0] = bytes([random.choice([0x80, 0xBF, 0xC0, 0xC3, 0xE2, 0xF0, 0xFF, 0xF4])])
        s = bytes(b)
    return s
CORPUS = [rnd_text().decode('utf-8', 'ignore') for _ in range(400)]
def train(num_merges):
    words = Counter()
    for t in CORPUS:
        for tok in t.replace('\n', ' \n ').split(' '):
            if tok: words[tuple(B2U[b] for b in (' ' + tok).encode())] += 1
    merges = []; seqs = dict(words)
    for _ in range(num_merges):
        pairs = Counter()
        for seq, c in seqs.items():
            for a, b in zip(seq, seq[1:]): pairs[(a, b)] += c
        if not pairs: break
        (a, b), _c = pairs.most_common(1)[0]; merges.append((a, b)); new = {}
        for seq, c in seqs.items():
            out = []; i = 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b: out.append(a + b); i += 2
                else: out.append(seq[i]); i += 1
            new[tuple(out)] = new.get(tuple(out), 0) + c
        seqs = new
    return merges
MERGES = train(420)
CL100K = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
FAM = {
  'cl100k': dict(pre={'type': 'Sequence', 'pretokenizers': [{'type': 'Split', 'pattern': {'Regex': CL100K}}, {'type': 'ByteLevel', 'use_regex': False}]}),
  'qwen': dict(pre={'type': 'Sequence', 'pretokenizers': [{'type': 'Split', 'pattern': {'Regex': CL100K.replace(r'\p{N}{1,3}', r'\p{N}').replace('|\\p{N}|', '|\\p{N}|') + '|\\p{N}|'}}, {'type': 'ByteLevel'}]}),
  'o200k': dict(pre={'type': 'Sequence', 'pretokenizers': [{'type': 'Split', 'pattern': {'Regex': r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"}}, {'type': 'ByteLevel'}]}),
  'tekken': dict(pre={'type': 'Sequence', 'pretokenizers': [{'type': 'Split', 'pattern': {'Regex': r"[\p{Lu}] \p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))"}}, {'type': 'ByteLevel'}]}),
  'kimi': dict(pre={'type': 'Sequence', 'pretokenizers': [{'type': 'Split', 'pattern': {'Regex': r"[\p{Han}]+|[\p{Lu}]"}}, {'type': 'ByteLevel'}]}),
  'gpt2': dict(pre={'type': 'ByteLevel', 'add_prefix_space': False, 'use_regex': True}),
  'deepseek': dict(pre={'type': 'Sequence', 'pretokenizers': [{'type': 'Split', 'pattern': {'Regex': r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"}}, {'type': 'ByteLevel', 'use_regex': False}]}),
  'rankbpe': dict(pre={'type': 'Sequence', 'pretokenizers': [{'type': 'Split', 'pattern': {'Regex': CL100K}}, {'type': 'ByteLevel'}]}, rank=True),
}
def build(fam, pair_arrays):
    vocab = {B2U[b]: b for b in range(256)}; nid = 256
    for a, b in MERGES:
        if a + b not in vocab: vocab[a + b] = nid; nid += 1
    added = [{'id': nid + i, 'content': c, 'special': i == 0} for i, c in enumerate(['<|endoftext|>', '<think>', '</think>'])]
    model = {'type': 'BPE', 'vocab': vocab}
    if not FAM[fam].get('rank'): model['merges'] = [[a, b] for a, b in MERGES] if pair_arrays else [a + ' ' + b for a, b in MERGES]
    return {'version': '1.0', 'added_tokens': added, 'pre_tokenizer': FAM[fam]['pre'], 'model': model}
bad = 0; total = 0
for fam in FAM:
    path = '%s/%s.json' % (w, fam); json.dump(build(fam, fam in ('gpt2', 'o200k')), open(path, 'w'), ensure_ascii=False)
    texts = []
    for i in range(N):
        p = '%s/%s_%04d.txt' % (w, fam, i); open(p, 'wb').write(rnd_text()); texts.append(p)
    c = subprocess.run([w + '/c_dump', path] + texts, capture_output=True); v = subprocess.run([w + '/v_dump', path] + texts, capture_output=True)
    total += len(texts)
    if c.returncode != 0 or v.returncode != 0 or c.stdout != v.stdout:
        bad += 1; print('DIFFERENT family', fam, 'rc', c.returncode, v.returncode)
        cl, vl = c.stdout.split(b'\n'), v.stdout.split(b'\n')
        for i, (x, y) in enumerate(zip(cl, vl)):
            if x != y: print(' text', texts[i // 2], '\n  C', x[:200], '\n  V', y[:200]); break
if bad: print('tokenizer differential: DIFFERENT, kept in', w); sys.exit(1)
print('tokenizer differential: %d texts over %d families, ids and decoded bytes identical' % (total, len(FAM))); shutil.rmtree(w)
