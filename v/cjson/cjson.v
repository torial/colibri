module cjson

// A port of c/json.h, the minimal JSON parser colibri reads safetensors headers, config.json and
// tokenizer.json with. The behaviour is the C one on purpose, including its quirks, because the
// differential test (tests/json_diff) compares trees bit for bit against the C parser:
//   - `parse` is the permissive "legacy" entry (json_parse): it never fails, it keeps what it could read;
//   - `parse_checked` is json_parse_checked: none unless the text is one complete, valid document;
//   - whitespace is whatever isspace() says, but only the four JSON characters are valid;
//   - numbers are checked against the JSON grammar but converted, and their extent taken, by strtod
//     (so `0x1p3`, `inf`, `0123` are consumed the way the C code consumes them, and flagged);
//   - `\u` escapes use strtoul on four characters (same quirks), surrogate pairs are joined, a lone
//     surrogate is an error;
//   - nesting deeper than 1024 is an error (the recursive descent must not overflow the stack).
// One deliberate difference: V strings keep an embedded NUL (`"a\u0000b"` is 3 bytes); the C parser's
// NUL-terminated strings stop at it. The test compares the C view (up to the first NUL).

#include <stdlib.h>

fn C.strtod(&char, &&char) f64
fn C.strtoul(&char, &&char, int) u64

pub enum Kind {
	null
	bool
	num
	str
	arr
	obj
}

pub struct Value {
pub mut:
	kind    Kind
	num     f64
	boolean bool
	str     string
	kids    []Value  // array elements, or an object's values
	keys    []string // an object's keys, parallel to kids
}

const max_depth = 1024

struct Parser {
mut:
	s     string // the text up to its first NUL, as a C string would end there
	i     int
	depth int
	error bool
}

@[inline]
fn (p &Parser) at(k int) u8 {
	idx := p.i + k
	return if idx < p.s.len { p.s[idx] } else { u8(0) }
}

@[inline]
fn (p &Parser) cur() u8 {
	return p.at(0)
}

fn is_space(c u8) bool {
	return c == ` ` || c == `\t` || c == `\n` || c == 0x0b || c == 0x0c || c == `\r`
}

fn is_xdigit(c u8) bool {
	return (c >= `0` && c <= `9`) || (c >= `a` && c <= `f`) || (c >= `A` && c <= `F`)
}

fn (mut p Parser) ws() {
	for p.cur() != 0 && is_space(p.cur()) {
		c := p.cur()
		p.i++
		// legacy consumption, but only JSON whitespace is valid
		if c != ` ` && c != `\t` && c != `\r` && c != `\n` {
			p.error = true
		}
	}
}

fn put_utf8(mut out []u8, cp u32) {
	if cp < 0x80 {
		out << u8(cp)
	} else if cp < 0x800 {
		out << u8(0xC0 | (cp >> 6))
		out << u8(0x80 | (cp & 0x3F))
	} else if cp < 0x10000 {
		out << u8(0xE0 | (cp >> 12))
		out << u8(0x80 | ((cp >> 6) & 0x3F))
		out << u8(0x80 | (cp & 0x3F))
	} else {
		out << u8(0xF0 | (cp >> 18))
		out << u8(0x80 | ((cp >> 12) & 0x3F))
		out << u8(0x80 | ((cp >> 6) & 0x3F))
		out << u8(0x80 | (cp & 0x3F))
	}
}

// strtoul(base 16) on four characters, as the C parser calls it (so a sign, blanks or `0x` in them
// behave as they do there); the result is cut to 32 bits like the C `unsigned`
fn strtoul4(a u8, b u8, c u8, d u8) u32 {
	buf := [a, b, c, d, u8(0)]
	return u32(C.strtoul(unsafe { &char(&buf[0]) }, unsafe { nil }, 16))
}

fn (mut p Parser) parse_str_raw(is_key bool) string {
	// SEC (GHSA-2qrj): fail closed if not actually at a quote
	if p.cur() != `"` {
		p.error = true
		return ''
	}
	p.i++
	mut out := []u8{cap: 32}
	for p.cur() != 0 && p.cur() != `"` {
		mut c := p.cur()
		p.i++
		if c < 0x20 {
			p.error = true
		}
		if c == `\\` && p.cur() != 0 {
			e := p.cur()
			p.i++
			match e {
				`n` {
					c = `\n`
				}
				`t` {
					c = `\t`
				}
				`r` {
					c = `\r`
				}
				`b` {
					c = 0x08
				}
				`f` {
					c = 0x0c
				}
				`/` {
					c = `/`
				}
				`\\` {
					c = `\\`
				}
				`"` {
					c = `"`
				}
				`u` {
					if p.at(0) == 0 || p.at(1) == 0 || p.at(2) == 0 || p.at(3) == 0 {
						p.error = true // a truncated \u: do not read past the end
						c = `?`
					} else {
						for k in 0 .. 4 {
							if !is_xdigit(p.at(k)) {
								p.error = true
							}
						}
						mut cp := strtoul4(p.at(0), p.at(1), p.at(2), p.at(3))
						p.i += 4
						// checked keys must keep their identity in C-string lookups
						if is_key && cp == 0 {
							p.error = true
						}
						if cp >= 0xD800 && cp <= 0xDBFF && p.at(0) == `\\` && p.at(1) == `u` && p.at(2) != 0
							&& p.at(3) != 0 && p.at(4) != 0 && p.at(5) != 0 {
							lo := strtoul4(p.at(2), p.at(3), p.at(4), p.at(5))
							if lo >= 0xDC00 && lo <= 0xDFFF {
								cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
								p.i += 6
							}
						}
						// a leftover surrogate is not Unicode; legacy parsing keeps the bytes
						if cp >= 0xD800 && cp <= 0xDFFF {
							p.error = true
						}
						put_utf8(mut out, cp)
						continue
					}
				}
				else {
					p.error = true
					c = e
				}
			}
		}
		out << c
	}
	if p.cur() == `"` {
		p.i++
	} else {
		p.error = true
	}
	return out.bytestr()
}

fn (mut p Parser) parse_val() Value {
	p.ws()
	c := p.cur()
	if c == `"` {
		return Value{
			kind: .str
			str:  p.parse_str_raw(false)
		}
	}
	if c == `{` {
		p.depth++
		if p.depth > max_depth {
			p.error = true
			p.depth--
			return Value{}
		}
		p.i++
		mut v := Value{
			kind: .obj
		}
		p.ws()
		if p.cur() == `}` {
			p.i++
			p.depth--
			return v
		}
		for {
			p.ws()
			if p.cur() != `"` {
				p.error = true // SEC (GHSA-2qrj): an object key must be a quoted string
				break
			}
			key := p.parse_str_raw(true)
			p.ws()
			if p.cur() == `:` {
				p.i++
			} else {
				p.error = true
			}
			val := p.parse_val()
			v.keys << key
			v.kids << val
			p.ws()
			if p.cur() == `,` {
				p.i++
				continue
			}
			if p.cur() == `}` {
				p.i++
				break
			}
			p.error = true
			break
		}
		p.depth--
		return v
	}
	if c == `[` {
		p.depth++
		if p.depth > max_depth {
			p.error = true
			p.depth--
			return Value{}
		}
		p.i++
		mut v := Value{
			kind: .arr
		}
		p.ws()
		if p.cur() == `]` {
			p.i++
			p.depth--
			return v
		}
		for {
			v.kids << p.parse_val()
			p.ws()
			if p.cur() == `,` {
				p.i++
				continue
			}
			if p.cur() == `]` {
				p.i++
				break
			}
			p.error = true
			break
		}
		p.depth--
		return v
	}
	if c == `t` && p.starts('true') {
		p.i += 4
		return Value{
			kind:    .bool
			boolean: true
		}
	}
	if c == `f` && p.starts('false') {
		p.i += 5
		return Value{
			kind: .bool
		}
	}
	if c == `n` && p.starts('null') {
		p.i += 4
		return Value{}
	}
	return p.parse_number()
}

fn (p &Parser) starts(word string) bool {
	return p.i + word.len <= p.s.len && p.s[p.i..p.i + word.len] == word
}

fn (mut p Parser) parse_number() Value {
	start := p.i
	mut q := start
	at := fn [p] (k int) u8 {
		return if k < p.s.len { p.s[k] } else { u8(0) }
	}
	if at(q) == `-` {
		q++
	}
	if at(q) == `0` {
		q++
	} else if at(q) >= `1` && at(q) <= `9` {
		q++
		for at(q) >= `0` && at(q) <= `9` {
			q++
		}
	} else {
		p.error = true
	}
	if at(q) == `.` {
		q++
		if at(q) < `0` || at(q) > `9` {
			p.error = true
		}
		for at(q) >= `0` && at(q) <= `9` {
			q++
		}
	}
	if at(q) == `e` || at(q) == `E` {
		q++
		if at(q) == `+` || at(q) == `-` {
			q++
		}
		if at(q) < `0` || at(q) > `9` {
			p.error = true
		}
		for at(q) >= `0` && at(q) <= `9` {
			q++
		}
	}
	mut end := &char(unsafe { nil })
	d := C.strtod(unsafe { &char(p.s.str + start) }, &end)
	end_i := int(u64(end) - u64(p.s.str))
	// 1e9999 is a legal JSON number and strtod returns inf; checked callers must reject it
	if end_i == start || end_i != q || !is_finite(d) {
		p.error = true
	}
	p.i = end_i
	return Value{
		kind: .num
		num:  d
	}
}

fn is_finite(d f64) bool {
	return d == d && d - d == 0
}

fn new_parser(text string) Parser {
	nul := text.index_u8(0)
	return Parser{
		s: if nul >= 0 { text[..nul].clone() } else { text }
	}
}

// parse is json_parse: permissive, it returns whatever tree it could read (a null on garbage).
pub fn parse(text string) Value {
	mut p := new_parser(text)
	return p.parse_val()
}

// parse_checked is json_parse_checked: none unless `text` is one complete, valid document.
pub fn parse_checked(text string) ?Value {
	mut p := new_parser(text)
	v := p.parse_val()
	p.ws()
	if p.error || p.cur() != 0 {
		return none
	}
	return v
}

// get is json_get: the first value of an object under `key` (compared as C strings, up to a NUL).
pub fn (v &Value) get(key string) ?&Value {
	if v.kind != .obj {
		return none
	}
	k := cstr(key)
	for i, name in v.keys {
		if cstr(name) == k {
			return &v.kids[i]
		}
	}
	return none
}

fn cstr(s string) string {
	nul := s.index_u8(0)
	return if nul >= 0 { s[..nul] } else { s }
}
