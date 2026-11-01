module tokenizer

// A port of c/tok.h: the byte-level BPE tokenizer colibri reads tokenizer.json with. Families of
// pre-tokenizer, picked from the file as the C code picks them: cl100k (GLM, and Qwen with one-digit
// groups), o200k (Inkling) and Tekken, Kimi, GPT-2 (OLMoE, GPT-NeoX), DeepSeek V4. The regexes are
// replayed by hand, in the order the engine tries the alternatives, exactly as the C code does; the code
// point sets come from tables generated out of the C headers (unicode_tables.v). Encoding and decoding
// are compared with the C tokenizer on text corpora by tests/tok_diff.
// Differences: errors are returned (C: exit(1)); `encode` has no output cap; `decode` returns the bytes
// as a string; an embedded NUL in a token string (a \u0000 escape in tokenizer.json) is kept here where the
// C parser's strings stop at it.
import os
import cjson

struct Special {
	str string
	id  int
}

pub struct Tok {
pub mut:
	vocab      map[string]int // byte-level string -> id
	merges     map[string]int // "left\0right" -> rank
	id2str     []string
	has_str    []bool
	id_added   []bool    // an added token: output literally
	id_special []bool    // added with "special": true
	sp         []Special // added tokens, longest first
	n_ids      int
	byte2cp    [256]u32
	byte2str   [256]string
	cp2byte    [1024]i16
	o200k      bool
	tekken     bool
	kimi       bool
	gpt2       bool
	deepseek   bool
	ndig1      bool
	rankbpe    bool // no merges list: merge the pair whose concatenation has the lowest id
}

fn u8_next(s []u8, length int, i int) (u32, int) {
	c := s[i]
	if c < 0x80 {
		return u32(c), 1
	}
	if (c >> 5) == 0x6 && i + 1 < length {
		return (u32(c & 0x1F) << 6) | u32(s[i + 1] & 0x3F), 2
	}
	if (c >> 4) == 0xE && i + 2 < length {
		return (u32(c & 0x0F) << 12) | (u32(s[i + 1] & 0x3F) << 6) | u32(s[i + 2] & 0x3F), 3
	}
	if (c >> 3) == 0x1E && i + 3 < length {
		return (u32(c & 0x07) << 18) | (u32(s[i + 1] & 0x3F) << 12) | (u32(s[i + 2] & 0x3F) << 6) | u32(s[i + 3] & 0x3F), 4
	}
	return u32(c), 1 // an invalid byte stands alone
}

fn u8_put(cp u32) string {
	mut o := []u8{cap: 4}
	if cp < 0x80 {
		o << u8(cp)
	} else if cp < 0x800 {
		o << u8(0xC0 | (cp >> 6))
		o << u8(0x80 | (cp & 0x3F))
	} else if cp < 0x10000 {
		o << u8(0xE0 | (cp >> 12))
		o << u8(0x80 | ((cp >> 6) & 0x3F))
		o << u8(0x80 | (cp & 0x3F))
	} else {
		o << u8(0xF0 | (cp >> 18))
		o << u8(0x80 | ((cp >> 12) & 0x3F))
		o << u8(0x80 | ((cp >> 6) & 0x3F))
		o << u8(0x80 | (cp & 0x3F))
	}
	return o.bytestr()
}

// the GPT-2 / ByteLevel byte <-> unicode map
fn (mut t Tok) build_bytemap() {
	for i in 0 .. 1024 {
		t.cp2byte[i] = -1
	}
	mut isdir := [256]bool{}
	for b in 33 .. 127 {
		isdir[b] = true
	}
	for b in 161 .. 173 {
		isdir[b] = true
	}
	for b in 174 .. 256 {
		isdir[b] = true
	}
	mut n := 0
	for b in 0 .. 256 {
		cp := if isdir[b] { u32(b) } else { u32(256 + n) }
		if !isdir[b] {
			n++
		}
		t.byte2cp[b] = cp
		t.byte2str[b] = u8_put(cp)
		if cp < 1024 {
			t.cp2byte[cp] = i16(b)
		}
	}
}

fn checked_id(v ?&cjson.Value) !int {
	value := v or { return error('tokenizer.json: token id must be an integer between 0 and ${1 << 21}') }
	if value.kind != .num || !(value.num == value.num && value.num - value.num == 0) || value.num < 0
		|| value.num > f64(1 << 21) || value.num != f64(int(value.num)) {
		return error('tokenizer.json: token id must be an integer between 0 and ${1 << 21}')
	}
	return int(value.num)
}

// load reads a tokenizer.json
pub fn load(path string) !Tok {
	text := os.read_file(path) or { return error('${path}: ${err}') }
	return load_text(text)
}

pub fn load_text(text string) !Tok {
	mut t := Tok{}
	t.build_bytemap()
	if text.index_u8(0) >= 0 {
		return error('tokenizer.json: malformed JSON')
	}
	root := cjson.parse_checked(text) or { return error('tokenizer.json: malformed JSON') }
	model := root.get('model') or { &cjson.Value{} }
	vocab := model.get('vocab') or { &cjson.Value{} }
	merges_v := model.get('merges')
	added_v := root.get('added_tokens')
	if vocab.kind != .obj {
		return error('tokenizer.json: model.vocab must be an object')
	}
	mut merges := cjson.Value{}
	mut has_merges := false
	if m := merges_v {
		if m.kind != .arr {
			return error('tokenizer.json: model.merges must be an array')
		}
		merges = *m
		has_merges = true
	}
	mut added := cjson.Value{}
	mut has_added := false
	if a := added_v {
		if a.kind != .arr {
			return error('tokenizer.json: added_tokens must be an array')
		}
		added = *a
		has_added = true
	}
	if !has_merges || merges.kids.len == 0 {
		t.rankbpe = true
		has_merges = false
	}
	mut maxid := 0
	for k in vocab.kids {
		id := checked_id(&k)!
		if id > maxid {
			maxid = id
		}
	}
	if has_added {
		for a in added.kids {
			id := checked_id(a.get('id'))!
			if id > maxid {
				maxid = id
			}
		}
	}
	if maxid > (1 << 21) {
		return error('tokenizer.json: implausible max vocab id ${maxid}')
	}
	t.n_ids = maxid + 1
	t.id2str = []string{len: t.n_ids}
	t.has_str = []bool{len: t.n_ids}
	t.id_added = []bool{len: t.n_ids}
	t.id_special = []bool{len: t.n_ids}
	for i, k in vocab.keys {
		id := int(vocab.kids[i].num)
		t.vocab[k] = id
		t.id2str[id] = k
		t.has_str[id] = true
	}
	if has_merges {
		for i, pr in merges.kids {
			mut l := ''
			mut r := ''
			if pr.kind == .arr && pr.kids.len == 2 && pr.kids[0].kind == .str && pr.kids[1].kind == .str {
				l = cjson_cstr(pr.kids[0].str)
				r = cjson_cstr(pr.kids[1].str)
			} else if pr.kind == .str {
				s := cjson_cstr(pr.str)
				sep := s.index_u8(` `)
				if sep <= 0 || sep + 1 >= s.len {
					return error('tokenizer.json: malformed merge entry ${i}')
				}
				l = s[..sep]
				r = s[sep + 1..]
			} else {
				return error('tokenizer.json: malformed merge entry ${i}')
			}
			t.merges[l + '\0' + r] = i
		}
	}
	if has_added {
		mut sp := []Special{}
		for a in added.kids {
			jc := a.get('content') or { return error('tokenizer.json: malformed added_token') }
			ji := a.get('id') or { return error('tokenizer.json: malformed added_token') }
			if jc.kind != .str || ji.kind != .num {
				return error('tokenizer.json: malformed added_token')
			}
			content := cjson_cstr(jc.str)
			id := int(ji.num)
			if id < 0 || id > maxid {
				return error('tokenizer.json: added id ${id} out of range')
			}
			sp << Special{content, id}
			t.id2str[id] = content
			t.has_str[id] = true
			t.id_added[id] = true
			if sf := a.get('special') {
				if sf.kind == .bool && sf.boolean {
					t.id_special[id] = true
				}
			}
		}
		// the longest first (a stable sort: the C qsort keeps the order of equal lengths in practice)
		sp.sort_with_compare(fn (a &Special, b &Special) int {
			return b.str.len - a.str.len
		})
		t.sp = sp
	}
	if pt := root.get('pre_tokenizer') {
		if pty := pt.get('type') {
			if pty.kind == .str && pty.str == 'ByteLevel' {
				if ur := pt.get('use_regex') {
					if ur.kind == .bool && ur.boolean {
						t.gpt2 = true
					}
				} else {
					t.gpt2 = true
				}
			}
		}
		if ps := pt.get('pretokenizers') {
			if ps.kind == .arr {
				for item in ps.kids {
					pat := item.get('pattern') or { continue }
					rx := pat.get('Regex') or { continue }
					if rx.kind != .str {
						continue
					}
					s := rx.str
					if s.contains('\\p{Lu}') {
						t.o200k = true
					}
					if s.contains('\\p{N}{1,3}(?=(?:\\p{N}{3})*(?!\\p{N}))') {
						t.tekken = true
					}
					if s.contains('\\p{Han}') {
						t.kimi = true
					}
					if s.contains('|\\p{N}|') {
						t.ndig1 = true
					}
					if s.contains('][A-Za-z]+|') && s.contains('[\\p{L}\\p{M}]+') {
						t.deepseek = true
					}
				}
			}
		}
	}
	return t
}

fn cjson_cstr(s string) string {
	nul := s.index_u8(0)
	return if nul >= 0 { s[..nul] } else { s }
}

// bpe_piece: the raw bytes [a, b) of p become ids appended to out
fn (t &Tok) bpe_piece(p []u8, a int, b int, mut out []int) {
	mut s := []u8{cap: 2 * (b - a)}
	for i in a .. b {
		s << t.byte2str[p[i]].bytes()
	}
	whole := t.vocab[s.bytestr()] or { -1 }
	if whole >= 0 {
		out << whole // ignore_merges: the whole piece is a token
		return
	}
	// the symbols start as the code points of the byte-level string
	mut soff := []int{}
	mut slen := []int{}
	mut i := 0
	for i < s.len {
		_, k := u8_next(s, s.len, i)
		soff << i
		slen << k
		i += k
	}
	for {
		mut best := max_int
		mut bp := -1
		for j in 0 .. soff.len - 1 {
			ll := slen[j]
			rl := slen[j + 1]
			rk := if t.rankbpe {
				t.vocab[s[soff[j]..soff[j] + ll + rl].bytestr()] or { -1 } // adjacent in s
			} else {
				t.merges[s[soff[j]..soff[j] + ll].bytestr() + '\0' +
					s[soff[j + 1]..soff[j + 1] + rl].bytestr()] or { -1 }
			}
			if rk >= 0 && rk < best {
				best = rk
				bp = j
			}
		}
		if bp < 0 {
			break
		}
		slen[bp] = soff[bp + 1] + slen[bp + 1] - soff[bp] // merge bp and bp+1 (adjacent in s)
		soff.delete(bp + 1)
		slen.delete(bp + 1)
	}
	for j in 0 .. soff.len {
		id := t.vocab[s[soff[j]..soff[j] + slen[j]].bytestr()] or { -1 }
		if id >= 0 {
			out << id
		}
	}
}

// decode_cps: the code points of p[a..b) and where each starts (off has one more entry, the end)
fn decode_cps(p []u8, a int, b int) ([]u32, []int) {
	mut cp := []u32{}
	mut off := []int{}
	mut i := a
	for i < b {
		c, k := u8_next(p, b, i)
		off << i
		cp << c
		i += k
	}
	off << b
	return cp, off
}

@[inline]
fn is_nl(c u32) bool {
	return c == `\r` || c == `\n`
}

@[inline]
fn low(c u32) u32 {
	return if c >= `A` && c <= `Z` { c + 32 } else { c }
}

// whitespace rules E, F, G of the cl100k / o200k / kimi families at i: the end of the piece, or -1
fn ws_rules(cp []u32, n int, i int) int {
	mut r := i
	for r < n && is_s(cp[r]) {
		r++
	}
	if r > i {
		mut last := -1
		for j in i .. r {
			if is_nl(cp[j]) {
				last = j
			}
		}
		if last >= 0 {
			return last + 1
		}
		mut end := if r < n { r - 1 } else { r }
		if end <= i {
			end = i + 1
		}
		return end
	}
	return -1
}

fn (t &Tok) pretok_cl100k(p []u8, a int, b int, mut out []int) {
	if b - a <= 0 {
		return
	}
	cp, off := decode_cps(p, a, b)
	n := cp.len
	mut i := 0
	for i < n {
		start := i
		c := cp[i]
		// 1) (?i:'s|'t|'re|'ve|'m|'ll|'d)
		if c == `'` && i + 1 < n {
			d := low(cp[i + 1])
			if i + 2 < n {
				d2 := low(cp[i + 2])
				if (d == `r` && d2 == `e`) || (d == `v` && d2 == `e`) || (d == `l` && d2 == `l`) {
					i += 3
					t.bpe_piece(p, off[start], off[i], mut out)
					continue
				}
			}
			if d == `s` || d == `t` || d == `m` || d == `d` {
				i += 2
				t.bpe_piece(p, off[start], off[i], mut out)
				continue
			}
		}
		// 2) [^\r\n\p{L}\p{N}]? \p{L}+
		{
			mut j := i
			mut ok := true
			if !is_l(c) && !is_nl(c) && !is_n(c) {
				if j + 1 < n && is_l(cp[j + 1]) {
					j++
				} else {
					ok = false
				}
			}
			if ok && is_l(cp[j]) {
				for j < n && is_l(cp[j]) {
					j++
				}
				i = j
				t.bpe_piece(p, off[start], off[i], mut out)
				continue
			}
		}
		// 3) \p{N}{1,3}, or \p{N} alone for Qwen (ndig1)
		if is_n(c) {
			mut j := i
			mut k := 0
			kmax := if t.ndig1 { 1 } else { 3 }
			for j < n && is_n(cp[j]) && k < kmax {
				j++
				k++
			}
			i = j
			t.bpe_piece(p, off[start], off[i], mut out)
			continue
		}
		// 4) ' ?[^\s\p{L}\p{N}]+[\r\n]*'
		{
			mut j := i
			if c == ` ` && j + 1 < n && !is_s(cp[j + 1]) && !is_l(cp[j + 1]) && !is_n(cp[j + 1]) {
				j++
			}
			if j < n && !is_s(cp[j]) && !is_l(cp[j]) && !is_n(cp[j]) {
				for j < n && !is_s(cp[j]) && !is_l(cp[j]) && !is_n(cp[j]) {
					j++
				}
				for j < n && is_nl(cp[j]) {
					j++
				}
				i = j
				t.bpe_piece(p, off[start], off[i], mut out)
				continue
			}
		}
		// 5) \s*[\r\n]+, 6) \s+(?!\S), 7) \s+
		e := ws_rules(cp, n, i)
		if e >= 0 {
			i = e
			t.bpe_piece(p, off[start], off[i], mut out)
			continue
		}
		i++
		t.bpe_piece(p, off[start], off[i], mut out)
	}
}

fn o2_contraction(cp []u32, n int, k int) int {
	if k < n && cp[k] == `'` && k + 1 < n {
		d := low(cp[k + 1])
		if k + 2 < n {
			e := low(cp[k + 2])
			if (d == `r` && e == `e`) || (d == `v` && e == `e`) || (d == `l` && e == `l`) {
				return k + 3
			}
		}
		if d == `s` || d == `t` || d == `m` || d == `d` {
			return k + 2
		}
	}
	return k
}

fn o2_s1(c u32) bool {
	return is_u(c) || is_x(c)
}

fn o2_s2(c u32) bool {
	return is_x(c) || (is_l(c) && !is_u(c))
}

fn km_s1(c u32) bool {
	return (is_u(c) || is_x(c)) && !is_han(c)
}

fn km_s2(c u32) bool {
	return (is_x(c) || (is_l(c) && !is_u(c))) && !is_han(c)
}

// the letter matcher of o200k (branches A then B, as the regex engine backtracks) or, with `kimi`,
// the same on Han-masked classes: the end of the match at i, or -1; contr false: no contraction suffix
fn letters(cp []u32, n int, i int, contr bool, kimi bool) int {
	for pfx := 1; pfx >= 0; pfx-- {
		mut j0 := i
		if pfx == 1 {
			c := cp[i]
			if c == `\r` || c == `\n` || is_l(c) || is_n(c) || i + 1 >= n {
				continue
			}
			j0 = i + 1
		}
		mut m1 := j0
		for m1 < n && (if kimi { km_s1(cp[m1]) } else { o2_s1(cp[m1]) }) {
			m1++
		}
		for s := m1; s >= j0; s-- {
			if s < n && (if kimi { km_s2(cp[s]) } else { o2_s2(cp[s]) }) {
				mut k := s + 1
				for k < n && (if kimi { km_s2(cp[k]) } else { o2_s2(cp[k]) }) {
					k++
				}
				return if contr { o2_contraction(cp, n, k) } else { k }
			}
		}
	}
	for pfx := 1; pfx >= 0; pfx-- {
		mut j0 := i
		if pfx == 1 {
			c := cp[i]
			if c == `\r` || c == `\n` || is_l(c) || is_n(c) || i + 1 >= n {
				continue
			}
			j0 = i + 1
		}
		mut m1 := j0
		for m1 < n && (if kimi { km_s1(cp[m1]) } else { o2_s1(cp[m1]) }) {
			m1++
		}
		if m1 > j0 {
			mut k := m1
			for k < n && (if kimi { km_s2(cp[k]) } else { o2_s2(cp[k]) }) {
				k++
			}
			return if contr { o2_contraction(cp, n, k) } else { k }
		}
	}
	return -1
}

fn is_han(c u32) bool {
	if c < 0x2E80 {
		return false
	}
	for r in han_ranges {
		if c >= r[0] && c <= r[1] {
			return true
		}
	}
	return false
}

const han_ranges = [[u32(0x2E80), 0x2E99], [u32(0x2E9B), 0x2EF3], [u32(0x2F00), 0x2FD5],
	[u32(0x3005), 0x3005], [u32(0x3007), 0x3007], [u32(0x3021), 0x3029], [u32(0x3038), 0x303B],
	[u32(0x3400), 0x4DBF], [u32(0x4E00), 0x9FFF], [u32(0xF900), 0xFA6D], [u32(0xFA70), 0xFAD9],
	[u32(0x16FE2), 0x16FE3], [u32(0x16FF0), 0x16FF1], [u32(0x20000), 0x2A6DF], [u32(0x2A700), 0x2B739],
	[u32(0x2B740), 0x2B81D], [u32(0x2B820), 0x2CEA1], [u32(0x2CEB0), 0x2EBE0], [u32(0x2EBF0), 0x2EE5D],
	[u32(0x2F800), 0x2FA1D], [u32(0x30000), 0x3134A], [u32(0x31350), 0x323AF]]

// o200k, Tekken and Kimi: letters (with the contraction tail unless Tekken), numbers, punctuation, whitespace
fn (t &Tok) pretok_o200k_like(p []u8, a int, b int, kimi bool, mut out []int) {
	if b - a <= 0 {
		return
	}
	cp, off := decode_cps(p, a, b)
	n := cp.len
	mut i := 0
	for i < n {
		start := i
		c := cp[i]
		if kimi && is_han(c) {
			mut j := i
			for j < n && is_han(cp[j]) {
				j++
			}
			i = j
			t.bpe_piece(p, off[start], off[i], mut out)
			continue
		}
		e := letters(cp, n, i, !t.tekken || kimi, kimi)
		if e > i {
			i = e
			t.bpe_piece(p, off[start], off[i], mut out)
			continue
		}
		// C: \p{N}{1,3}; Tekken's first group takes what a multiple of three leaves
		if !kimi && is_n(c) && t.tekken {
			mut r := 0
			for i + r < n && is_n(cp[i + r]) {
				r++
			}
			k := if r % 3 != 0 { r % 3 } else { 3 }
			i += k
			t.bpe_piece(p, off[start], off[i], mut out)
			continue
		}
		if is_n(c) {
			mut j := i
			mut k := 0
			for j < n && is_n(cp[j]) && k < 3 {
				j++
				k++
			}
			i = j
			t.bpe_piece(p, off[start], off[i], mut out)
			continue
		}
		// D: ' ?[^\s\p{L}\p{N}]+[\r\n/]*' (Kimi: no '/' in the tail)
		{
			mut j := i
			if c == ` ` && j + 1 < n && !is_s(cp[j + 1]) && !is_l(cp[j + 1]) && !is_n(cp[j + 1]) {
				j++
			}
			if j < n && !is_s(cp[j]) && !is_l(cp[j]) && !is_n(cp[j]) {
				for j < n && !is_s(cp[j]) && !is_l(cp[j]) && !is_n(cp[j]) {
					j++
				}
				for j < n && (is_nl(cp[j]) || (!kimi && cp[j] == `/`)) {
					j++
				}
				i = j
				t.bpe_piece(p, off[start], off[i], mut out)
				continue
			}
		}
		e2 := ws_rules(cp, n, i)
		if e2 >= 0 {
			i = e2
			t.bpe_piece(p, off[start], off[i], mut out)
			continue
		}
		i++
		t.bpe_piece(p, off[start], off[i], mut out)
	}
}

fn (t &Tok) pretok_gpt2(p []u8, a int, b int, mut out []int) {
	if b - a <= 0 {
		return
	}
	cp, off := decode_cps(p, a, b)
	n := cp.len
	mut i := 0
	for i < n {
		start := i
		c := cp[i]
		// 1) 's|'t|'re|'ve|'m|'ll|'d, case-sensitive
		if c == `'` && i + 1 < n {
			d := cp[i + 1]
			if i + 2 < n {
				d2 := cp[i + 2]
				if (d == `r` && d2 == `e`) || (d == `v` && d2 == `e`) || (d == `l` && d2 == `l`) {
					i += 3
					t.bpe_piece(p, off[start], off[i], mut out)
					continue
				}
			}
			if d == `s` || d == `t` || d == `m` || d == `d` {
				i += 2
				t.bpe_piece(p, off[start], off[i], mut out)
				continue
			}
		}
		// 2-4) one optional space, then a run of one class
		{
			mut j := i
			if c == ` ` && j + 1 < n {
				j++
			}
			d := cp[j]
			if is_l(d) {
				for j < n && is_l(cp[j]) {
					j++
				}
				i = j
				t.bpe_piece(p, off[start], off[i], mut out)
				continue
			}
			if is_n(d) {
				for j < n && is_n(cp[j]) {
					j++
				}
				i = j
				t.bpe_piece(p, off[start], off[i], mut out)
				continue
			}
			if !is_s(d) {
				for j < n && !is_s(cp[j]) && !is_l(cp[j]) && !is_n(cp[j]) {
					j++
				}
				i = j
				t.bpe_piece(p, off[start], off[i], mut out)
				continue
			}
		}
		// 5) \s+(?!\S) 6) \s+
		{
			mut r := i
			for r < n && is_s(cp[r]) {
				r++
			}
			if r > i {
				mut end := if r < n { r - 1 } else { r }
				if end <= i {
					end = i + 1
				}
				i = end
				t.bpe_piece(p, off[start], off[i], mut out)
				continue
			}
		}
		i++
		t.bpe_piece(p, off[start], off[i], mut out)
	}
}

fn ds_ascii_punct(c u32) bool {
	return (c >= 0x21 && c <= 0x2F) || (c >= 0x3A && c <= 0x40) || (c >= 0x5B && c <= 0x60)
		|| (c >= 0x7B && c <= 0x7E)
}

fn ds_ascii_alpha(c u32) bool {
	return (c >= `A` && c <= `Z`) || (c >= `a` && c <= `z`)
}

fn ds_cjk(c u32) bool {
	return (c >= 0x4E00 && c <= 0x9FA5) || (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF)
}

fn ds_lm(c u32) bool {
	return is_l(c) || is_mk(c)
}

fn ds_ps(c u32) bool {
	return is_pu(c) || is_sy(c)
}

// the third DeepSeek Split at position i of the piece [i, e): the end of the match, -1 if none
fn ds_match(cp []u32, i int, e int) int {
	c := cp[i]
	if ds_ascii_punct(c) && i + 1 < e && ds_ascii_alpha(cp[i + 1]) {
		mut j := i + 1
		for j < e && ds_ascii_alpha(cp[j]) {
			j++
		}
		return j
	}
	if c != `\r` && c != `\n` && !is_l(c) && !ds_ps(c) && i + 1 < e && ds_lm(cp[i + 1]) {
		mut j := i + 1
		for j < e && ds_lm(cp[j]) {
			j++
		}
		return j
	}
	if ds_lm(c) {
		mut j := i
		for j < e && ds_lm(cp[j]) {
			j++
		}
		return j
	}
	{
		mut j := -1
		if c == ` ` && i + 1 < e && ds_ps(cp[i + 1]) {
			j = i + 1
		} else if ds_ps(c) {
			j = i
		}
		if j >= 0 {
			for j < e && ds_ps(cp[j]) {
				j++
			}
			for j < e && (cp[j] == `\r` || cp[j] == `\n`) {
				j++
			}
			return j
		}
	}
	{
		mut r := i
		for r < e && is_s(cp[r]) {
			r++
		}
		if r > i {
			mut last := -1
			for j in i .. r {
				if cp[j] == `\r` || cp[j] == `\n` {
					last = j
				}
			}
			if last >= 0 {
				return last + 1
			}
			return if r < e && r - 1 > i { r - 1 } else { r }
		}
	}
	return -1
}

fn (t &Tok) ds_split3(p []u8, cp []u32, off []int, s int, e int, mut out []int) {
	mut i := s
	mut pending := s
	for i < e {
		j := ds_match(cp, i, e)
		if j < 0 {
			i++
			continue
		}
		if pending < i {
			t.bpe_piece(p, off[pending], off[i], mut out)
		}
		t.bpe_piece(p, off[i], off[j], mut out)
		i = j
		pending = j
	}
	if pending < e {
		t.bpe_piece(p, off[pending], off[e], mut out)
	}
}

fn (t &Tok) ds_split2(p []u8, cp []u32, off []int, s int, e int, mut out []int) {
	mut i := s
	for i < e {
		mut j := i
		cjk := ds_cjk(cp[i])
		for j < e && ds_cjk(cp[j]) == cjk {
			j++
		}
		t.ds_split3(p, cp, off, i, j, mut out)
		i = j
	}
}

fn (t &Tok) pretok_deepseek(p []u8, a int, b int, mut out []int) {
	if b - a <= 0 {
		return
	}
	cp, off := decode_cps(p, a, b)
	n := cp.len
	mut i := 0
	for i < n {
		mut j := i
		if is_n(cp[i]) {
			for j < n && is_n(cp[j]) && j - i < 3 {
				j++
			}
		} else {
			for j < n && !is_n(cp[j]) {
				j++
			}
		}
		t.ds_split2(p, cp, off, i, j, mut out)
		i = j
	}
}

// encode: text to ids, split on the added tokens first (the longest match at the earliest position), then
// pre-tokenize and BPE each stretch
pub fn (t &Tok) encode(text string) []int {
	p := text.bytes()
	n := p.len
	mut out := []int{}
	mut i := 0
	for i < n {
		mut hitpos := -1
		mut hitlen := 0
		mut hitid := -1
		mut j := i
		for j < n && hitpos < 0 {
			for sp in t.sp {
				sl := sp.str.len
				if sl > 0 && j + sl <= n && sp.str.bytes() == p[j..j + sl] {
					hitpos = j
					hitlen = sl
					hitid = sp.id
					break
				}
			}
			j++
		}
		chunk_end := if hitpos < 0 { n } else { hitpos }
		if chunk_end > i {
			if t.gpt2 {
				t.pretok_gpt2(p, i, chunk_end, mut out)
			} else if t.deepseek {
				t.pretok_deepseek(p, i, chunk_end, mut out)
			} else if t.kimi {
				t.pretok_o200k_like(p, i, chunk_end, true, mut out)
			} else if t.o200k {
				t.pretok_o200k_like(p, i, chunk_end, false, mut out)
			} else {
				t.pretok_cl100k(p, i, chunk_end, mut out)
			}
		}
		if hitpos < 0 {
			break
		}
		out << hitid
		i = hitpos + hitlen
	}
	return out
}

// id_of: the id of an added token given its content, -1 if absent
pub fn (t &Tok) id_of(content string) int {
	for sp in t.sp {
		if sp.str == content {
			return sp.id
		}
	}
	return -1
}

// decode: ids to bytes (the inverse of the byte-level map; added tokens literally; unknown ids skipped)
pub fn (t &Tok) decode(ids []int) string {
	mut o := []u8{}
	for id in ids {
		if id < 0 || id >= t.n_ids || !t.has_str[id] {
			continue
		}
		s := t.id2str[id]
		if t.id_added[id] {
			o << s.bytes()
			continue
		}
		sb := s.bytes()
		mut j := 0
		for j < sb.len {
			c, k := u8_next(sb, sb.len, j)
			j += k
			if c < 1024 && t.cp2byte[c] >= 0 {
				o << u8(t.cp2byte[c])
			}
		}
	}
	return o.bytestr()
}

// unicode_class is the set of classes a code point is in, one bit each: L N S U X Mk Pu Sy Han
// (the sets the pre-tokenizers use; exposed so the tables can be compared with the C ones)
pub fn unicode_class(c u32) int {
	mut m := 0
	if is_l(c) {
		m |= 1
	}
	if is_n(c) {
		m |= 2
	}
	if is_s(c) {
		m |= 4
	}
	if is_u(c) {
		m |= 8
	}
	if is_x(c) {
		m |= 16
	}
	if is_mk(c) {
		m |= 32
	}
	if is_pu(c) {
		m |= 64
	}
	if is_sy(c) {
		m |= 128
	}
	if is_han(c) {
		m |= 256
	}
	return m
}
