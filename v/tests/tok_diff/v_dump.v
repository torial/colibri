// V side of the tokenizer differential test (see c_dump.c): encode, decode, id_of for every text file.
module main

import os
import tokenizer

fn main() {
	t := tokenizer.load(os.args[1]) or {
		eprintln('${err}')
		exit(1)
	}
	digits := '0123456789abcdef'
	for path in os.args[2..] {
		text := os.read_file(path) or { panic(err) }
		ids := t.encode(text)
		println('E' + ids.map(' ${it}').join(''))
		d := t.decode(ids)
		mut h := []u8{}
		for b in d.bytes() {
			h << digits[b >> 4]
			h << digits[b & 15]
		}
		println('D ' + h.bytestr())
	}
	println('I ${t.id_of('<|endoftext|>')} ${t.id_of('<think>')}')
}
