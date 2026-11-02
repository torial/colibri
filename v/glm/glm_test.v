module glm

import os
import safetensors

fn tmp_cfg(text string) string {
	d := os.join_path(os.vtmp_dir(), 'glm_cfg_${os.getpid()}')
	os.mkdir_all(d) or { panic(err) }
	os.write_file('${d}/config.json', text) or { panic(err) }
	return d
}

const base = '"hidden_size":8,"num_hidden_layers":2,"num_attention_heads":2,"n_routed_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":4,"intermediate_size":8,"first_k_dense_replace":1,"q_lora_rank":4,"kv_lora_rank":4,"qk_nope_head_dim":4,"qk_rope_head_dim":4,"v_head_dim":4,"n_shared_experts":1,"vocab_size":10,"n_group":1'

fn test_config_loads() {
	c := load_cfg(tmp_cfg('{${base},"norm_topk_prob":true,"routed_scaling_factor":2.5,"rope_parameters":{"rope_theta":500000}}'))!
	assert c.qk_head == 8 && c.norm_topk && c.routed == 2.5 && c.theta == 500000 && c.eps == f32(1e-5)
}

fn test_config_refusals() {
	if _ := load_cfg(tmp_cfg('{${base.replace('"n_group":1', '"n_group":2')}}')) {
		assert false // this engine requires n_group = 1
	}
	if _ := load_cfg(tmp_cfg('{${base.replace('"qk_rope_head_dim":4', '"qk_rope_head_dim":3')}}')) {
		assert false // an odd rotary width cannot be paired
	}
	if _ := load_cfg(tmp_cfg('{"hidden_size":8}')) {
		assert false
	}
}

fn test_softmax_sigmoid_silu() {
	mut x := [f32(0), 0]
	softmax(mut x, 2)
	assert x[0] == 0.5 && x[1] == 0.5
	assert sigmoidf(0) == 0.5
	assert siluf(0) == 0
}

fn test_the_score_dot_has_its_lane_structure() {
	q := []f32{len: 11, init: f32(index) + 1}
	l := []f32{len: 11, init: 1}
	// all small integers: exact in any order, so both forms agree on the value
	assert lat_dot(q, l, 0, 11, true, 4) == 66 && lat_dot(q, l, 0, 11, false, 0) == 66
}

fn test_container_formats_it_cannot_read_are_refused() {
	// an int4-packed weight (half the bytes) beside a scale per row is a container format this engine does not read yet
	d := os.join_path(os.vtmp_dir(), 'glm_qs_${os.getpid()}')
	os.mkdir_all(d) or { panic(err) }
	mut blob := []u8{}
	hdr := '{"w":{"dtype":"U8","shape":[2,2],"data_offsets":[0,4]},"w.qs":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}'
	n := u64(hdr.len)
	for i in 0 .. 8 {
		blob << u8(n >> (8 * u64(i)))
	}
	blob << hdr.bytes()
	blob << []u8{len: 12}
	os.write_file_array('${d}/m.safetensors', blob) or { panic(err) }
	mut m := Model{
		store: safetensors.open(d)!
	}
	if _ := m.load_container('w', 2, 4) { // O=2, I=4 would need 8 bytes: this is 4, packed int4
		assert false
	} else {
		assert err.msg().contains('not an int8')
	}
	assert (m.load_container('w', 2, 2) or { panic(err) }).q.len == 4
	m.close()
}
