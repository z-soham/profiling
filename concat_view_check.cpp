// Verifies ggml CPU CONCAT on NON-contiguous (transposed) inputs - the layout Qwen3.6's GDN conv state uses
// (delta-net-base.cpp: ggml_concat(conv_states, ggml_transpose(qkv_mixed), 0)) - against a naive strided reference.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <initializer_list>
static ggml_tensor * make(ggml_context * ctx, ggml_type t, const int64_t * ne, bool transposed, int seed) {
    ggml_tensor * x;
    if (transposed) {
        int64_t r[4] = {ne[1], ne[0], ne[2], ne[3]};
        x = ggml_new_tensor(ctx, t, 4, r);
    } else {
        x = ggml_new_tensor(ctx, t, 4, ne);
    }
    for (size_t i = 0; i < ggml_nbytes(x); i++) ((uint8_t *) x->data)[i] = (uint8_t)(i * seed + 3);
    return transposed ? ggml_transpose(ctx, x) : x;
}
int main() {
    ggml_backend_load_all();
    long n = 0, bad = 0, cases = 0, strided = 0;
    const ggml_type types[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I8, GGML_TYPE_I32, GGML_TYPE_BF16};
    struct S { int64_t a[4]; int64_t b[4]; int dim; };
    // first case = Qwen conv state: [3,ch,1] ++ transpose([ch,512 tokens]) along dim 0
    for (int generic = 0; generic < 2; generic++)
    for (ggml_type t : types) for (int dim = 0; dim < 4; dim++) for (int tr = 0; tr < 4; tr++) for (int nt : {1, 3, 32}) {
        int64_t A[4], B[4];
        if (generic == 0) {                       // Qwen shape (dim 0 only is the real use; others still valid)
            int64_t a[4] = {3, 512, 1, 1}, b[4] = {512, 512, 1, 1};
            memcpy(A, a, sizeof a); memcpy(B, b, sizeof b);
            if (dim != 0) { B[0] = 3; B[dim] = (dim == 1 ? 512 : 1); }
            if (dim == 1) { B[1] = 40; }
        } else {
            int64_t a[4] = {11, 12, 13, 14}; memcpy(A, a, sizeof a); memcpy(B, a, sizeof a); B[dim] = 7;
        }
        ggml_init_params ip = {256u << 20, nullptr, false};
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a = make(ctx, t, A, tr & 1, 7);
        ggml_tensor * b = make(ctx, t, B, tr & 2, 13);
        const size_t es = ggml_type_size(t);
        if (b->nb[0] != es || a->nb[0] != es) strided++;
        ggml_tensor * c = ggml_concat(ctx, a, b, dim);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, c);
        ggml_graph_compute_with_ctx(ctx, g, nt);
        cases++;
        for (int64_t i3 = 0; i3 < c->ne[3]; i3++) for (int64_t i2 = 0; i2 < c->ne[2]; i2++)
        for (int64_t i1 = 0; i1 < c->ne[1]; i1++) for (int64_t i0 = 0; i0 < c->ne[0]; i0++) {
            int64_t id[4] = {i0, i1, i2, i3};
            const ggml_tensor * s = id[dim] < a->ne[dim] ? a : b;
            if (s == b) id[dim] -= a->ne[dim];
            const char * e = (const char *) s->data + id[0]*s->nb[0] + id[1]*s->nb[1] + id[2]*s->nb[2] + id[3]*s->nb[3];
            const char * r = (const char *) c->data + i0*c->nb[0] + i1*c->nb[1] + i2*c->nb[2] + i3*c->nb[3];
            n++;
            if (memcmp(e, r, es)) bad++;
        }
        ggml_free(ctx);
    }
    printf("cases %ld (%ld with a non-contiguous input), compared %ld elements, %ld mismatches\n", cases, strided, n, bad);
    return bad != 0;
}
