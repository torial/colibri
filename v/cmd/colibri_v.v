// colibri_v: a small command-line front for the V port -- inspect a model directory, try the tokenizer, and measure the
// compute kernels on this machine.
//
//   v -cc gcc -prod -o colibri_v cmd/colibri_v.v      (run kernels/build.sh first; ARCH=native for the machine at hand)
//   colibri_v isa
//   colibri_v info <model-dir> [extra-dir;extra-dir]
//   colibri_v tok <tokenizer.json> "text to encode"
//   colibri_v bench [threads] [seconds-per-kernel]
//
// `bench` is meant for a new machine (the Dell 7820, say): it prints, per kernel, the time of one call, the weight
// bytes streamed per second and the multiply-adds per second, at a decode shape (one token) and a batch shape.
module main

import os
import time
import strconv
import safetensors
import tokenizer
import kernels

fn usage() {
	eprintln('usage: colibri_v isa | info <model-dir> [extra-dirs] | tok <tokenizer.json> <text> | bench [threads] [seconds]')
	exit(2)
}

fn human(n i64) string {
	if n >= i64(1) << 30 {
		return '${f64(n) / f64(i64(1) << 30):.2f} GiB'
	}
	if n >= i64(1) << 20 {
		return '${f64(n) / f64(i64(1) << 20):.1f} MiB'
	}
	return '${n} B'
}

fn cmd_info(dir string, extra string) {
	mut s := safetensors.open_multi(dir, extra) or {
		eprintln('${err}')
		exit(1)
	}
	mut total := i64(0)
	mut by_dtype := map[string]i64{}
	mut counts := map[string]int{}
	for t in s.tensors {
		total += t.nbytes
		k := if t.is_ggml() { 'ggml:${t.ggml}' } else { t.dtype.name() }
		by_dtype[k] += t.nbytes
		counts[k]++
	}
	println('${dir}: ${s.shards.len} shard file(s), ${s.tensors.len} tensors, ${human(total)}')
	if s.gguf_active {
		arch := s.gguf_meta.get_str('general.architecture') or { '?' }
		println('  GGUF, architecture ${arch}')
	}
	mut keys := by_dtype.keys()
	keys.sort()
	for k in keys {
		println('  ${k:-12} ${counts[k]:7} tensors  ${human(by_dtype[k])}')
	}
	if s.fmt_names.len > 0 {
		println('  ${s.fmt_names.len} colibri.fmt stamp(s)')
	}
	if s.n_overlay > 0 {
		println('  ${s.n_overlay} tensor name(s) resolved by model.safetensors.index.json')
	}
	if s.nrep() > 0 {
		println('  ${s.nrep()} mirror(s)')
	}
	s.close()
}

fn cmd_tok(path string, text string) {
	t := tokenizer.load(path) or {
		eprintln('${err}')
		exit(1)
	}
	ids := t.encode(text)
	println(ids.map(it.str()).join(' '))
	println(t.decode(ids))
}

// one benchmark: run `f` for about `secs` seconds, return seconds per call
fn timeit(secs f64, f fn ()) f64 {
	f() // warm the caches and the thread pool
	mut n := 0
	sw := time.new_stopwatch()
	for {
		f()
		n++
		if f64(sw.elapsed().nanoseconds()) / 1e9 >= secs {
			break
		}
	}
	return f64(sw.elapsed().nanoseconds()) / 1e9 / f64(n)
}

fn row(name string, s int, i int, o int, wbytes i64, per f64) {
	macs := f64(s) * f64(i) * f64(o)
	println('${name:-26} S=${s:<3} ${i}x${o}  ${per * 1e3:9.3f} ms  ${f64(wbytes) / per / 1e9:7.2f} GB/s weights  ${macs / per / 1e9:8.2f} GMAC/s')
}

fn cmd_bench(threads int, secs f64) {
	if threads > 0 {
		kernels.set_threads(threads)
	}
	println('kernels built for: ${kernels.isa()}; ${kernels.threads()} thread(s)')
	i := 4096
	o := 4096
	for s in [1, 32] {
		x := []f32{len: s * i, init: f32((index * 7) % 13) * 0.1 - 0.6}
		wf := []f32{len: o * i, init: f32((index * 5) % 17) * 0.05 - 0.4}
		per := timeit(secs, fn [x, wf, s, i, o] () {
			kernels.matmul_f32(x, wf, s, i, o) or { panic(err) }
		})
		row('f32', s, i, o, i64(o) * i * 4, per)
		wq := []i8{len: o * i, init: i8(index % 255 - 127)}
		sc := []f32{len: o, init: 0.01}
		per2 := timeit(secs, fn [x, wq, sc, s, i, o] () {
			kernels.matmul_q(x, wq, sc, s, i, o) or { panic(err) }
		})
		row('int8 + row scale', s, i, o, i64(o) * i, per2)
		w4 := []u8{len: o * i / 2, init: u8(index * 37)}
		per3 := timeit(secs, fn [x, w4, sc, s, i, o] () {
			kernels.matmul_i4(x, w4, sc, s, i, o) or { panic(err) }
		})
		row('int4 + row scale', s, i, o, i64(o) * i / 2, per3)
		wmx := []u8{len: o * i / 2, init: u8(0x22)}
		em := []u8{len: o * i / 32, init: u8(127)}
		per4 := timeit(secs, fn [x, wmx, em, s, i, o] () {
			kernels.matmul_mxfp4(x, wmx, em, s, i, o) or { panic(err) }
		})
		row('MXFP4', s, i, o, i64(o) * i / 2 + i64(o) * i / 32, per4)
		for t in [8, 12, 2] { // Q8_0, Q4_K, Q2_K
			rb := kernels.gq_row_bytes(t, i64(i))
			wg := []u8{len: int(rb) * o, init: u8(index * 131 + 7)}
			per5 := timeit(secs, fn [x, wg, t, s, i, o] () {
				kernels.gq_matmul(x, wg, t, s, i, o) or { panic(err) }
			})
			row('ggml type ${t}', s, i, o, i64(rb) * o, per5)
		}
	}
}

fn main() {
	a := os.args[1..]
	if a.len == 0 {
		usage()
	}
	match a[0] {
		'isa' {
			println('${kernels.isa()}; ${kernels.threads()} thread(s)')
		}
		'info' {
			if a.len < 2 {
				usage()
			}
			cmd_info(a[1], if a.len > 2 { a[2] } else { '' })
		}
		'tok' {
			if a.len < 3 {
				usage()
			}
			cmd_tok(a[1], a[2])
		}
		'bench' {
			th := if a.len > 1 { a[1].int() } else { 0 }
			secs := if a.len > 2 { strconv.atof64(a[2]) or { 0.5 } } else { 0.5 }
			cmd_bench(th, secs)
		}
		else {
			usage()
		}
	}
}
