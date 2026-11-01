module sample

import math

fn test_greedy_and_ties() {
	mut s := Sampler{}
	assert s.pick_tok([f32(1), 3, 3, 2], -1) == 1 // first largest
	assert argmax_v([]f32{}) == 0
	nan := f32(math.nan())
	assert argmax_v([nan, f32(-1), nan]) == 1
}

fn test_nucleus_truncates() {
	mut s := Sampler{
		temp: 1
		nuc:  0.5
	}
	s.dist_build([f32(5), 0, 0, 0])
	assert s.pbuf[0] > 0.9
	assert s.pbuf[1] == 0
	for _ in 0 .. 20 {
		assert s.dist_sample(-1) == 0
	}
}

fn test_ban_excludes() {
	mut s := Sampler{
		temp: 1
	}
	for _ in 0 .. 50 {
		assert s.pick_tok([f32(1), 1, 1], 1) != 1
	}
}

fn test_non_finite_falls_back_to_argmax() {
	mut s := Sampler{
		temp: 1
		nuc:  0.9
	}
	inf := f32(math.inf(1))
	assert s.pick_tok([-inf, -inf, -inf], -1) == 0
	assert s.warned
}

fn test_stops() {
	st := arm([5, 6], 7, unsafe { nil }, StopOpts{ quiet: true })
	assert st.ids == [5, 6, 7]
	assert st.is_stop(6) && !st.is_stop(8)
	st2 := arm([5, 6], 7, unsafe { nil }, StopOpts{ serve_batch: true, quiet: true })
	assert st2.ids == [7]
}

fn test_logprob_status() {
	inf := f32(math.inf(1))
	nan := f32(math.nan())
	assert logprob_row_checked([f32(1), nan]).status == .nan
	assert logprob_row_checked([f32(1), inf, nan]).status == .nan
	assert logprob_row_checked([f32(1), inf]).status == .pos_inf
	assert logprob_row_checked([f32(1), -inf]).status == .neg_inf
	assert logprob_row_checked([nan, inf]).status == .all_nonfinite
	assert logprob_row_checked([]f32{}).status == .invalid
	r := logprob_row_checked([f32(0), 0])
	st, v := logprob_from_row_checked([f32(0), 0], 0, &r)
	assert st == .finite
	assert v > -0.6932 && v < -0.6930
	st2, _ := logprob_from_row_checked([f32(0), 0], -1, &r)
	assert st2 == .invalid
}

fn test_mt_is_glm() {
	assert mt_is_glm('chatGLM4') && mt_is_glm('glm') && !mt_is_glm('gl') && !mt_is_glm('')
}
