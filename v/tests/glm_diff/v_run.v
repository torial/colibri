// The V side of the GLM differential test: greedy generation from a checkpoint and a ref.json, every step's logits
// dumped as raw float32 (what the C engine's DUMP=<path> writes), the generated ids printed.
// usage: v_run <model-dir> <cap> <ref.json> <dump-path> [unused]
module main

import os
import cjson
import glm

fn ids(r &cjson.Value, key string) []int {
	a := r.get(key) or { panic('ref.json: missing array "${key}"') }
	return a.kids.map(int(it.num))
}

fn main() {
	dir := os.args[1]
	cap := os.args[2].int()
	ref := cjson.parse(os.read_file(os.args[3]) or { panic(err) })
	prompt := ids(&ref, 'prompt_ids')
	full := ids(&ref, 'full_ids')
	bits := if os.args.len > 5 { os.args[5].int() } else { 16 }
	mut m := glm.open_bits(dir, cap, bits) or {
		eprintln('${err}')
		exit(1)
	}
	m.idot = !(os.args.len > 6 && os.args[6] == 'noidot')
	m.keep = true
	out := m.generate(prompt, full.len - prompt.len) or {
		eprintln('${err}')
		exit(1)
	}
	mut f := os.create(os.args[4]) or { panic(err) }
	for l in m.logits {
		unsafe { f.write_ptr(l.data, l.len * 4) }
	}
	f.close()
	println(out[prompt.len..].map(it.str()).join(' '))
	println('hits=${m.hits} miss=${m.miss}')
	m.close()
}
