module safetensors

import os
import math

fn write_shard(dir string, name string, header string, data []u8) ! {
	mut blob := []u8{}
	n := u64(header.len)
	for i in 0 .. 8 {
		blob << u8(n >> (8 * u64(i)))
	}
	blob << header.bytes()
	blob << data
	os.write_file_array('${dir}/${name}', blob)!
}

fn tmpdir(tag string) string {
	d := os.join_path(os.vtmp_dir(), 'st_port_${tag}_${os.getpid()}')
	os.mkdir_all(d) or { panic(err) }
	return d
}

fn test_reads_the_three_float_dtypes() {
	d := tmpdir('ok')
	defer {
		os.rmdir_all(d) or {}
	}
	// bf16 1.0 = 0x3f80, f16 1.0 = 0x3c00, f32 2.0 = 0x40000000
	data := [u8(0x80), 0x3f, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x40]
	write_shard(d, 'model.safetensors', '{"__metadata__":{"format":"pt"},"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"F16","shape":[1],"data_offsets":[2,4]},"c":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}',
		data) or { panic(err) }
	s := open(d) or { panic(err) }
	assert s.tensors.len == 3
	assert (s.read_f32('a') or { panic(err) })[0] == 1.0
	assert (s.read_f32('b') or { panic(err) })[0] == 1.0
	assert (s.read_f32('c') or { panic(err) })[0] == 2.0
	assert s.find('missing') == none
}

fn test_refuses_what_c_refuses() {
	d := tmpdir('bad')
	defer {
		os.rmdir_all(d) or {}
	}
	write_shard(d, 'model.safetensors', '{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}', [
		u8(0),
		0,
		0,
		0,
	]) or {
		panic(err)
	}
	if _ := open(d) {
		assert false, 'numel 2 with 4 bytes of F32 must be refused'
	}
	write_shard(d, 'model.safetensors', '{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,400]}}', [
		u8(0),
		0,
		0,
		0,
	]) or {
		panic(err)
	}
	if _ := open(d) {
		assert false, 'offsets past the end of the file must be refused'
	}
}

fn test_non_float_dtypes_need_read_raw() {
	d := tmpdir('raw')
	defer {
		os.rmdir_all(d) or {}
	}
	write_shard(d, 'model.safetensors', '{"q":{"dtype":"U8","shape":[3],"data_offsets":[0,3]}}', [
		u8(1),
		2,
		3,
	]) or {
		panic(err)
	}
	s := open(d) or { panic(err) }
	assert (s.read_raw('q') or { panic(err) }) == [u8(1), 2, 3]
	if _ := s.read_f32('q') {
		assert false, 'a U8 tensor is not a float tensor'
	}
}

fn one(name string, dt string, n int) string {
	return '"${name}":{"dtype":"${dt}","shape":[${n}],"data_offsets":[0,${n}]}'
}

fn test_search_path_takes_the_first_listed_copy() {
	d := tmpdir('split')
	defer {
		os.rmdir_all(d) or {}
	}
	os.mkdir_all('${d}/a') or { panic(err) }
	os.mkdir_all('${d}/b') or { panic(err) }
	write_shard('${d}/a', 'm-1.safetensors', '{${one('x', 'U8', 2)}}', [u8(1), 2]) or { panic(err) }
	write_shard('${d}/b', 'm-1.safetensors', '{${one('other', 'U8', 2)}}', [u8(9), 9]) or { panic(err) }
	write_shard('${d}/b', 'm-2.safetensors', '{${one('y', 'U8', 1)}}', [u8(7)]) or { panic(err) }
	s := open_multi('${d}/a', ' ${d}/b ; ,') or { panic(err) }
	assert s.shards.len == 2
	assert s.has('x') && s.has('y') && !s.has('other')
	assert s.read_raw('y')! == [u8(7)]
}

fn test_index_decides_between_two_copies() {
	d := tmpdir('overlay')
	defer {
		os.rmdir_all(d) or {}
	}
	write_shard(d, 'a.safetensors', '{${one('w', 'U8', 1)}}', [u8(1)]) or { panic(err) }
	write_shard(d, 'b.safetensors', '{${one('w', 'U8', 1)}}', [u8(2)]) or { panic(err) }
	if _ := open(d) {
		assert false
	}
	os.write_file('${d}/model.safetensors.index.json', '{"weight_map":{"w":"b.safetensors"}}') or {
		panic(err)
	}
	s := open(d) or { panic(err) }
	assert s.tensors.len == 1 && s.n_overlay == 1
	assert s.read_raw('w')! == [u8(2)]
}

fn test_format_stamps() {
	d := tmpdir('stamp')
	defer {
		os.rmdir_all(d) or {}
	}
	m := '"__metadata__":{"colibri.fmt":"{\\"w\\":\\"int4\\"}"},'
	write_shard(d, 'a.safetensors', '{${m}${one('w', 'U8', 1)}}', [u8(1)]) or { panic(err) }
	s := open(d) or { panic(err) }
	assert s.fmt_stamp('w')? == 'int4'
	assert s.fmt_stamp('v') == none
	write_shard(d, 'b.safetensors', '{"__metadata__":{"colibri.fmt":"{\\"w\\":\\"fp8\\"}"},${one('z', 'U8', 1)}}',
		[u8(1)]) or { panic(err) }
	if _ := open(d) {
		assert false // two shards disagree about w
	}
}

fn test_mirror_accepts_only_identical_headers() {
	d := tmpdir('mirror')
	defer {
		os.rmdir_all(d) or {}
	}
	os.mkdir_all('${d}/p') or { panic(err) }
	os.mkdir_all('${d}/m') or { panic(err) }
	hdr := '{${one('w', 'U8', 3)}}'
	write_shard('${d}/p', 'a.safetensors', hdr, [u8(1), 2, 3]) or { panic(err) }
	write_shard('${d}/m', 'a.safetensors', hdr, [u8(4), 5, 6]) or { panic(err) }
	mut s := open('${d}/p') or { panic(err) }
	assert s.mirror_add('${d}/m') == 1 && s.nrep() == 1
	assert s.fd_rep(0, 1) >= 0 && s.fd_rep(0, 2) == -1
	mut buf := []u8{len: 3}
	s.read_range_rep(0, 1, s.find('w')?.off, 3, buf.data, 3, false, false, '') or { panic(err) }
	assert buf == [u8(4), 5, 6] // served by the replica
	s.read_range_rep(0, 0, s.find('w')?.off, 3, buf.data, 3, false, false, '') or { panic(err) }
	assert buf == [u8(1), 2, 3]
	write_shard('${d}/m', 'a.safetensors', '{${one('v', 'U8', 3)}}', [u8(4), 5, 6]) or { panic(err) }
	assert s.mirror_add('${d}/m') == 0 && s.nrep() == 1 // a different header claims no slot
	s.close()
}

fn test_slices_ranges_and_caps() {
	d := tmpdir('slice')
	defer {
		os.rmdir_all(d) or {}
	}
	data := [u8(0), 0, 0x80, 0x3f, 0, 0, 0, 0xc0]
	write_shard(d, 'a.safetensors', '{"f":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"s":{"dtype":"F8_E8M0","shape":[2],"data_offsets":[2,4]}}',
		data) or { panic(err) }
	s := open(d) or { panic(err) }
	assert s.read_slice_f32('f', 1, 1, false)![0] == -2.0
	assert s.read_slice('f', 4, 4, false)! == [u8(0), 0, 0, 0xc0]
	if _ := s.read_slice('f', 5, 4, false) {
		assert false
	}
	if _ := s.read_f32_cap('f', 1, false) {
		assert false
	}
	sc := s.read_scale_f32('s', 2, false)!
	assert sc[0] == f32(2.0) && sc[1] == f32(math.ldexp(1.0, -64))
	m := s.missing_report('model.norm.weight')
	assert m.contains('missing model.norm.weight') && m.contains('indexed 2 tensors from 1 shard file(s)')
}
