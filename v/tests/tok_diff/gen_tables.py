#!/usr/bin/env python3
"""Regenerates v/tokenizer/unicode_tables.v from the C headers' range tables (so both tokenizers use the same code point sets).
usage: python3 tests/tok_diff/gen_tables.py   (from v/)"""
import re
C = '../c/'
out = ['module tokenizer', '',
       '// GENERATED from c/tok_unicode.h, c/tok_unicode_o200k.h and c/tok_unicode_deepseek.h by tests/tok_diff/gen_tables.py:',
       '// the same code point ranges the C tokenizer uses, flat [lo, hi, lo, hi, ...] in ascending order.', '']
for fn, names in [('tok_unicode.h', ['L', 'N', 'S']), ('tok_unicode_o200k.h', ['U', 'X']), ('tok_unicode_deepseek.h', ['Mk', 'Pu', 'Sy'])]:
    s = open(C + fn).read()
    for n in names:
        m = re.search(r'static const uint32_t uni_%s\[\]\[2\] = \{(.*?)\};' % n, s, re.S)
        pairs = re.findall(r'\{(0x[0-9A-Fa-f]+),(0x[0-9A-Fa-f]+)\}', m.group(1))
        cnt = int(re.search(r'uni_%s_n = (\d+);' % n, s).group(1)); assert cnt == len(pairs), (n, cnt, len(pairs))
        flat = [x for p in pairs for x in p]
        out.append('const uni_%s = [' % n.lower())
        for i in range(0, len(flat), 12):
            row = ', '.join(('u32(%s)' % x) if (i == 0 and k == 0) else x for k, x in enumerate(flat[i:i + 12]))
            out.append('\t' + row + ',')
        out.append(']')
        out.append('')
out.append('''// uni_in is a binary search of a flat range table (C: uni_in)
fn uni_in(t []u32, c u32) bool {
	mut lo := 0
	mut hi := t.len / 2 - 1
	for lo <= hi {
		mid := (lo + hi) / 2
		if c < t[2 * mid] {
			hi = mid - 1
		} else if c > t[2 * mid + 1] {
			lo = mid + 1
		} else {
			return true
		}
	}
	return false
}''')
for f in ['l', 'n', 's', 'u', 'x', 'mk', 'pu', 'sy']:
    out.append('\nfn is_%s(c u32) bool {\n\treturn uni_in(uni_%s, c)\n}' % (f, f))
open('tokenizer/unicode_tables.v', 'w').write('\n'.join(out) + '\n')
