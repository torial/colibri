module glm

import os

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
