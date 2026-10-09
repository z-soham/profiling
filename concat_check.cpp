// Verifies ggml CPU CONCAT against a naive reference for f32/f16/i8/i32, all dims, odd shapes, many thread counts.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <vector>
int main() {
    ggml_backend_load_all();
    int bad = 0, n = 0;
    const ggml_type types[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I8, GGML_TYPE_I32, GGML_TYPE_BF16};
    for (ggml_type t : types) for (int dim = 0; dim < 4; dim++) for (int nt : {1, 3, 32}) for (int view = 0; view < 2; view++) {
        int64_t ne_a[4] = {11, 12, 13, 14}, ne_b[4] = {11, 12, 13, 14};
        ne_b[dim] = 7;
        if (dim == 0 && view) { ne_a[0] = 1; ne_b[0] = 3; }
        ggml_init_params ip = {64u << 20, nullptr, false};
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a = ggml_new_tensor(ctx, t, 4, ne_a);
        ggml_tensor * b = ggml_new_tensor(ctx, t, 4, ne_b);
        const size_t es = ggml_type_size(t);
        for (size_t i = 0; i < ggml_nbytes(a); i++) ((uint8_t *) a->data)[i] = (uint8_t)(i * 7 + 1);
        for (size_t i = 0; i < ggml_nbytes(b); i++) ((uint8_t *) b->data)[i] = (uint8_t)(i * 13 + 5);
        ggml_tensor * c = ggml_concat(ctx, a, b, dim);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, c);
        ggml_graph_compute_with_ctx(ctx, g, nt);
        for (int64_t i3 = 0; i3 < c->ne[3]; i3++) for (int64_t i2 = 0; i2 < c->ne[2]; i2++)
        for (int64_t i1 = 0; i1 < c->ne[1]; i1++) for (int64_t i0 = 0; i0 < c->ne[0]; i0++) {
            int64_t id[4] = {i0, i1, i2, i3};
            const ggml_tensor * s = id[dim] < ne_a[dim] ? a : b;
            if (s == b) id[dim] -= ne_a[dim];
            const char * e = (const char *) s->data + id[0]*s->nb[0] + id[1]*s->nb[1] + id[2]*s->nb[2] + id[3]*s->nb[3];
            const char * r = (const char *) c->data + i0*c->nb[0] + i1*c->nb[1] + i2*c->nb[2] + i3*c->nb[3];
            n++;
            if (memcmp(e, r, es)) { bad++; }
        }
        ggml_free(ctx);
    }
    printf("compared %d elements, %d mismatches\n", n, bad);
    return bad != 0;
}
