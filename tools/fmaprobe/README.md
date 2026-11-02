# fmaprobe

Which summation order did the compiler give this C loop?

A float32 reduction has no single meaning. gcc may keep a plain loop, fuse each multiply-add, split the sum over SIMD lanes and
fold the lanes at the end, or treat whole blocks one way and the remainder another, depending on the loop's length and the
instruction set. Port such code to another language and the last bit of the result depends on which one you got. fmaprobe
answers by behaviour: it compiles your function for several instruction-set levels, runs it on random inputs of many lengths,
and reports, per length, which reference order it reproduces bit for bit.

```
python3 fmaprobe.py FILE.c FUNCTION [--archs v1,v2,v3,v4,native] [--lengths 1-24,64] [--trials 200] [--asm]
```

`FUNCTION` has the shape `float f(const float *x, const float *y, int n, float init)` (a dot product accumulated onto `init`).
References: a plain loop (multiply, then add), one fused chain, 4 / 8 / 16 fused lanes folded the way `hsum256` folds, a pairwise
tree, and "blocks": whole blocks of W products multiplied apart and added one by one, the remainder fused.

## What it found

The loop that started this, `a += x[d] * y[d]` over `d < n` with `n` known only at run time (`examples/rope_dot.c`):

```
-march=x86-64-v2    n = 1..100      seq  (never fused)
-march=x86-64-v3    n = 1-3         one fused chain
                    n = 4, 8, 12    seq  (no remainder, so never fused)
                    n = 5-7, 9-11   blocks of 4 unfused, then a fused remainder
-march=x86-64-v4    n = 4-7         one fused chain
                    n >= 8          blocks of 8 unfused, then a fused remainder
```

So the same source statement is fused for some lengths and not for others. A port that writes the loop the obvious way, or with
an explicit `fma` everywhere, matches the C only for the lengths where the compiler happened to agree. Writing the same
block-and-remainder structure (`hybrid_dot` in `v/glm/glm.v`) made the V port bit-identical to the C engine for the rotary-key
dot product at every length tried. With the engine's prompt path (more than four rows are not absorbed) written the same way, the whole GLM port matched the C engine on every generated checkpoint, bit for bit.

## Limits

x86-64 with gcc and the `-ffp-contract=fast` default only. The references are a small catalogue; "matches no reference" means a
different order, and `--asm` prints instruction counts to start reading it. A reduction nested in a larger function can be
compiled differently from the same loop alone, so use the tool to form a hypothesis and confirm it in the real code.
