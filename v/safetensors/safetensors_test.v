module safetensors

import os

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
