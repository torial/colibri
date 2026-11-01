// Reads a file, parses it with the V port both ways and prints the canonical dump that
// c_dump.c prints for the C parser (see run.sh). Numbers as the 16 hex digits of their bits, strings as
// hex of their C view (up to a NUL).
module main

import os
import math
import cjson

fn hex(s string) string {
	nul := s.index_u8(0)
	t := if nul >= 0 { s[..nul] } else { s }
	digits := '0123456789abcdef'
	mut o := []u8{}
	for b in t.bytes() {
		o << digits[b >> 4]
		o << digits[b & 15]
	}
	return o.bytestr()
}

fn dump_value(v cjson.Value, mut out []string) {
	match v.kind {
		.null { out << 'N' }
		.bool { out << if v.boolean { 'B1' } else { 'B0' } }
		.num { out << 'D${math.f64_bits(v.num):016x}' }
		.str { out << 'S${hex(v.str)}' }
		.arr {
			out << 'A${v.kids.len}'
			for k in v.kids {
				dump_value(k, mut out)
			}
		}
		.obj {
			out << 'O${v.kids.len}'
			for i, k in v.kids {
				out << 'K${hex(v.keys[i])}'
				dump_value(k, mut out)
			}
		}
	}
}

fn main() {
	for path in os.args[1..] {
		text := os.read_file(path) or { panic(err) }
		println('#' + os.file_name(path))
		mut a := []string{}
		dump_value(cjson.parse(text), mut a)
		println('legacy ' + a.join(' '))
		if c := cjson.parse_checked(text) {
			mut b := []string{}
			dump_value(c, mut b)
			println('checked ' + b.join(' '))
		} else {
			println('checked NULL')
		}
	}
}
