module cjson

fn test_parse_and_get() {
	v := parse_checked('{"a":[1,2.5,"x\\u00e9y",true,null],"b":{"c":-0.0}}') or { panic('no tree') }
	assert v.kind == .obj
	a := v.get('a') or { panic('no a') }
	assert a.kind == .arr && a.kids.len == 5
	assert a.kids[0].num == 1.0 && a.kids[1].num == 2.5
	assert a.kids[2].str == 'xéy'
	assert a.kids[3].boolean && a.kids[4].kind == .null
	c := (v.get('b') or { panic('no b') }).get('c') or { panic('no c') }
	assert c.num == 0.0
	assert v.get('missing') == none
}

fn test_checked_rejects_what_legacy_keeps() {
	assert parse_checked('[1,]') == none
	assert parse_checked('{"a":1} x') == none
	assert parse_checked('1e9999') == none
	assert parse_checked('"\\ud83d"') == none
	assert parse_checked('0123') == none
	l := parse('[1,]')
	assert l.kind == .arr && l.kids.len == 2 // the C parser reads a stray number after the comma
	assert parse('').kind == .num && parse('').num == 0.0 // empty input: the C parser falls into its number branch
}

fn test_surrogate_pair_and_escapes() {
	v := parse_checked('"\\ud83d\\ude00 \\n\\t\\"\\\\"') or { panic('no tree') }
	assert v.str == '😀 \n\t"\\'
}

fn test_depth_limit() {
	assert parse_checked('['.repeat(1024) + ']'.repeat(1024)) != none
	assert parse_checked('['.repeat(1025) + ']'.repeat(1025)) == none
}

fn test_embedded_nul_is_kept_in_values() {
	v := parse_checked('"a\\u0000b"') or { panic('no tree') }
	assert v.str.len == 3 && v.str[1] == 0
	assert parse_checked('{"\\u0000":1}') == none // a NUL in a key would break C-string lookups
}
