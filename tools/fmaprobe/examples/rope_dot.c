/* the loop that cost a day: a scalar dot product accumulated onto a running sum, length known only at run time */
float rope_dot(const float *x, const float *y, int n, float init) { float a = init; for (int d = 0; d < n; d++) a += x[d] * y[d]; return a; }
