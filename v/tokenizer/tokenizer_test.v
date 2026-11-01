module tokenizer

// A tiny byte-level BPE: the 256 byte tokens plus one merge ("h" "i"), and an added token.
fn tiny_json() string {
	mut vocab := []string{}
	mut id := 0
	for b in 0 .. 256 {
		key := u8_put(byte_cp(b)).replace('\\', '\\\\').replace('"', '\\"')
		vocab << '"${key}":${id}'
		id++
	}
	vocab << '"hi":${id}'
	return '{"added_tokens":[{"id":${id + 1},"content":"<|endoftext|>","special":true}],"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}{1,3}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+"}},{"type":"ByteLevel"}]},"model":{"type":"BPE","vocab":{' +
		vocab.join(',') + '},"merges":["h i"]}}'
}

fn byte_cp(b int) u32 {
	mut t := Tok{}
	t.build_bytemap()
	return t.byte2cp[b]
}

fn test_encode_decode_roundtrip() {
	t := load_text(tiny_json()) or { panic(err) }
	ids := t.encode('hi there')
	assert ids[0] == 256 // the merge "hi"
	assert t.decode(ids) == 'hi there'
	assert t.decode(t.encode('naïve 你好 123')) == 'naïve 你好 123'
}

fn test_added_tokens_are_atomic() {
	t := load_text(tiny_json()) or { panic(err) }
	ids := t.encode('a<|endoftext|>b')
	assert ids.len == 3 && ids[1] == 257
	assert t.id_of('<|endoftext|>') == 257
	assert t.id_of('<nope>') == -1
}

fn refused(text string) bool {
	load_text(text) or { return true }
	return false
}

fn test_bad_files_are_refused() {
	assert refused('[]')
	assert refused('{"model":{"vocab":{"a":-1}}}')
	assert refused('{"model":{"vocab":{"a":0},"merges":["nospace"]}}')
	assert !refused('{"model":{"vocab":{"a":0}}}')
}

fn test_unicode_classes() {
	assert unicode_class(`a`) & 1 != 0
	assert unicode_class(`7`) & 2 != 0
	assert unicode_class(` `) & 4 != 0
	assert unicode_class(u32(0x4E2D)) & 256 != 0 // a Han character
	assert unicode_class(`A`) & 8 != 0
}
