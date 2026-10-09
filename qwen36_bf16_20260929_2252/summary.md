# Qwen3.6-35B-A3B prefill diagnosis: qwen36_bf16_20260929_2252

## Clean prefill throughput (tokens/s, mean ± sd)

| config | pp256 | pp8192 |
|---|---|---|
| llama.cpp stock ub4096 | 251.9 ± 2.4 | 224.0 ± 0.8 |
| llama.cpp stock ub512 | 253.0 ± 2.0 | 259.9 ± 0.2 |
| llama.cpp zendnn ub4096 | 249.1 ± 1.8 | 232.4 ± 0.9 |
| llama.cpp zendnn ub512 | 252.0 ± 2.1 | 248.8 ± 0.8 |
| llama.cpp zendnn_fb0 ub4096 | 236.8 ± 4.7 (last 3: 239.5; 1st 243.1; n=8) | 276.0 ± 0.5 (last 3: 275.7; 1st 276.6; n=8) |
| llama.cpp zendnn_fb0 ub512 | 115.8 ± 38.1 (last 3: 152.2; 1st 52.7; n=8) | 297.1 ± 3.3 (last 3: 299.6; 1st 290.0; n=8) |
| llama.cpp zendnn_fb0_grp1 ub512 | 197.2 ± 28.3 (last 3: 215.5; 1st 132.3; n=8) |  |
| llama.cpp zendnn_fb0_grp3 ub512 | 127.8 ± 41.7 (last 3: 167.2; 1st 58.1; n=8) |  |
| llama.cpp zendnn_fb0_grp5 ub512 | 211.0 ± 18.3 (last 3: 225.8; 1st 174.3; n=8) |  |
| vLLM base (bench latency, out=1) | 436.8 ± 0.4 | 617.0 ± 0.1 |

vLLM t/s = prompt / latency of a 1-token generation (prefill + one sample), batch 1. mnbt8192 = --max-num-batched-tokens 8192 (the whole prompt in one chunk).

## llama.cpp per-op profile (opprof: per-node wall time, ms per prompt)

| run | clean t/s | profiled t/s | sum(nodes) ms |
|---|---|---|---|
| stock_pp256_ub512 | 241.56 | 225.38 | 1130.4 |
| stock_pp8192_ub4096 | 239.42 | 229.55 | 35581.5 |
| stock_pp8192_ub512 | 258.52 | 243.30 | 33530.9 |
| zendnn_fb0_pp256_ub512 | 103.43 | 121.18 | 2106.8 |
| zendnn_fb0_pp8192_ub4096 | 272.39 | 256.72 | 31803.9 |
| zendnn_fb0_pp8192_ub512 | 278.36 | 288.53 | 28252.0 |
| zendnn_pp256_ub512 | 264.59 | 260.95 | 975.5 |
| zendnn_pp8192_ub4096 | 247.19 | 237.14 | 34438.7 |
| zendnn_pp8192_ub512 | 240.88 | 262.04 | 31124.4 |

| bucket (ms, % of nodes; [ms on ZenDNN]) | stock_pp256_ub512 | stock_pp8192_ub4096 | stock_pp8192_ub512 | zendnn_fb0_pp256_ub512 | zendnn_fb0_pp8192_ub4096 | zendnn_fb0_pp8192_ub512 | zendnn_pp256_ub512 | zendnn_pp8192_ub4096 | zendnn_pp8192_ub512 |
|---|---|---|---|---|---|---|---|---|---|
| GDN conv-state CONCAT (single-threaded in ggml-cpu) | 286 (25%) [0] | 14161 (40%) [0] | 8319 (25%) [0] | 231 (11%) [0] | 14688 (46%) [0] | 5534 (20%) [0] | 166 (17%) [0] | 14571 (42%) [0] | 7194 (23%) [0] |
| MoE routed experts (MUL_MAT_ID) | 476 (42%) [0] | 8495 (24%) [0] | 12266 (37%) [0] | 1540 (73%) [1540] | 5739 (18%) [5739] | 10999 (39%) [10999] | 479 (49%) [0] | 8498 (25%) [0] | 12338 (40%) [0] |
| weight matmul: attention / GDN projections | 140 (12%) [0] | 3803 (11%) [0] | 4007 (12%) [0] | 105 (5%) [105] | 2528 (8%) [2528] | 2891 (10%) [2891] | 107 (11%) [103] | 2579 (7%) [2478] | 2949 (9%) [2824] |
| flash attention | 9 (1%) [0] | 3005 (8%) [0] | 2676 (8%) [0] | 9 (0%) [0] | 2993 (9%) [0] | 2657 (9%) [0] | 9 (1%) [0] | 2980 (9%) [0] | 2662 (9%) [0] |
| GDN recurrence + conv1d | 63 (6%) [0] | 2020 (6%) [0] | 2067 (6%) [0] | 63 (3%) [0] | 1978 (6%) [0] | 1991 (7%) [0] | 63 (6%) [0] | 1974 (6%) [0] | 2009 (6%) [0] |
| other (norms, rope, adds, copies) | 51 (5%) [0] | 1312 (4%) [0] | 1339 (4%) [0] | 52 (2%) [0] | 1306 (4%) [0] | 1371 (5%) [0] | 52 (5%) [0] | 1309 (4%) [0] | 1357 (4%) [0] |
| attention output-gate SIGMOID | 30 (3%) [0] | 913 (3%) [0] | 940 (3%) [0] | 30 (1%) [0] | 908 (3%) [0] | 936 (3%) [0] | 30 (3%) [0] | 909 (3%) [0] | 928 (3%) [0] |
| MoE glue (topk, act, weighting) | 24 (2%) [0] | 729 (2%) [0] | 645 (2%) [0] | 31 (1%) [0] | 795 (2%) [0] | 801 (3%) [0] | 24 (2%) [0] | 734 (2%) [0] | 649 (2%) [0] |
| GDN other elementwise | 25 (2%) [0] | 668 (2%) [0] | 726 (2%) [0] | 25 (1%) [0] | 670 (2%) [0] | 744 (3%) [0] | 25 (3%) [0] | 683 (2%) [0] | 742 (2%) [0] |
| weight matmul: MoE router | 10 (1%) [0] | 267 (1%) [0] | 287 (1%) [0] | 6 (0%) [6] | 75 (0%) [75] | 192 (1%) [192] | 6 (1%) [6] | 75 (0%) [75] | 158 (1%) [158] |
| weight matmul: shared expert | 9 (1%) [0] | 190 (1%) [0] | 241 (1%) [0] | 5 (0%) [5] | 107 (0%) [107] | 119 (0%) [119] | 5 (0%) [4] | 107 (0%) [94] | 120 (0%) [105] |
| weight matmul: lm head | 9 (1%) [0] | 18 (0%) [0] | 18 (0%) [0] | 9 (0%) [9] | 18 (0%) [18] | 18 (0%) [18] | 9 (1%) [0] | 18 (0%) [0] | 18 (0%) [0] |

### top rows: stock_pp256_ub512

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| CONCAT | conv_input | CPU | - | - | 30 | 285.94 | 25.30 |
| MUL_MAT_ID | ffn_moe_down | CPU | bf16 | K=512 M=2048 E=256 used=8 tok=256 | 40 | 184.56 | 16.33 |
| MUL_MAT_ID | ffn_moe_gate | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=256 | 40 | 147.08 | 13.01 |
| MUL_MAT_ID | ffn_moe_up | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=256 | 40 | 143.86 | 12.73 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 30 | 49.38 | 4.41 |
| MUL_MAT | (unnamed) | CPU | bf16 | K=2048 M=8192 N=256 B=1 | 30 | 43.42 | 3.89 |
| SIGMOID | gate_sigmoid | CPU | - | - | 10 | 29.73 | 2.63 |
| MUL_MAT | linear_attn_out | CPU | bf16 | K=4096 M=2048 N=256 B=1 | 30 | 24.86 | 2.20 |
| MUL_MAT | z | CPU | bf16 | K=2048 M=4096 N=256 B=1 | 30 | 22.14 | 1.96 |
| MUL_MAT | Qcur_full | CPU | bf16 | K=2048 M=8192 N=256 B=1 | 10 | 14.29 | 1.26 |
| SSM_CONV | conv_output_raw | CPU | - | - | 30 | 13.30 | 1.18 |
| RMS_NORM | norm | CPU | - | - | 131 | 12.51 | 1.11 |
| MUL_MAT | ffn_moe_logits | CPU | f32 | K=2048 M=256 N=256 B=1 | 40 | 10.13 | 0.90 |
| MUL_MAT | ffn_gate | CPU | bf16 | K=2048 M=512 N=256 B=1 | 40 | 10.00 | 0.88 |
| FLASH_ATTN_EXT | (unnamed) | CPU | - | - | 10 | 9.31 | 0.8 |

### top rows: stock_pp8192_ub4096

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| CONCAT | conv_input | CPU | - | - | 60 | 14161.11 | 39.80 |
| MUL_MAT_ID | ffn_moe_down | CPU | bf16 | K=512 M=2048 E=256 used=8 tok=4096 | 80 | 3370.01 | 9.47 |
| FLASH_ATTN_EXT | (unnamed) | CPU | - | - | 20 | 3004.85 | 8.45 |
| MUL_MAT_ID | ffn_moe_up | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=4096 | 80 | 2568.38 | 7.22 |
| MUL_MAT_ID | ffn_moe_gate | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=4096 | 80 | 2556.36 | 7.18 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 60 | 1591.21 | 4.5 |
| MUL_MAT | (unnamed) | CPU | bf16 | K=2048 M=8192 N=4096 B=1 | 60 | 1153.03 | 3.31 |
| SIGMOID | gate_sigmoid | CPU | - | - | 20 | 912.86 | 2.57 |
| MUL_MAT | linear_attn_out | CPU | bf16 | K=4096 M=2048 N=4096 B=1 | 60 | 794.21 | 2.23 |
| MUL_MAT | z | CPU | bf16 | K=2048 M=4096 N=4096 B=1 | 60 | 603.10 | 1.69 |
| SSM_CONV | conv_output_raw | CPU | - | - | 60 | 429.04 | 1.21 |
| MUL_MAT | Qcur_full | CPU | bf16 | K=2048 M=8192 N=4096 B=1 | 20 | 388.12 | 1.09 |
| RMS_NORM | norm | CPU | - | - | 262 | 354.11 | 1.00 |
| MUL | ffn_moe_weighted | CPU | - | - | 80 | 313.91 | 0.88 |
| ADD | (unnamed) | CPU | - | - | 540 | 280.27 | 0.4 |

### top rows: stock_pp8192_ub512

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| CONCAT | conv_input | CPU | - | - | 480 | 8318.91 | 24.81 |
| MUL_MAT_ID | ffn_moe_down | CPU | bf16 | K=512 M=2048 E=256 used=8 tok=512 | 640 | 4966.06 | 14.81 |
| MUL_MAT_ID | ffn_moe_gate | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=512 | 640 | 3672.57 | 10.95 |
| MUL_MAT_ID | ffn_moe_up | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=512 | 640 | 3627.55 | 10.82 |
| FLASH_ATTN_EXT | (unnamed) | CPU | - | - | 160 | 2675.97 | 7.98 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 480 | 1637.08 | 4.82 |
| MUL_MAT | (unnamed) | CPU | bf16 | K=2048 M=8192 N=512 B=1 | 480 | 1268.9 | 3.85 |
| SIGMOID | gate_sigmoid | CPU | - | - | 160 | 940.40 | 2.80 |
| MUL_MAT | linear_attn_out | CPU | bf16 | K=4096 M=2048 N=512 B=1 | 480 | 743.48 | 2.22 |
| MUL_MAT | z | CPU | bf16 | K=2048 M=4096 N=512 B=1 | 480 | 620.93 | 1.85 |
| SSM_CONV | conv_output_raw | CPU | - | - | 480 | 429.73 | 1.28 |
| MUL_MAT | Qcur_full | CPU | bf16 | K=2048 M=8192 N=512 B=1 | 160 | 423.04 | 1.26 |
| RMS_NORM | norm | CPU | - | - | 2096 | 387.59 | 1.16 |
| MUL_MAT | ffn_moe_logits | CPU | f32 | K=2048 M=256 N=512 B=1 | 640 | 286.82 | 0.86 |
| MUL_MAT | ffn_gate | CPU | bf16 | K=2048 M=512 N=512 B=1 | 640 | 251.75 | 0.75 |

### top rows: zendnn_fb0_pp256_ub512

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| MUL_MAT_ID | ffn_moe_gate | ZenDNN | bf16 | K=2048 M=512 E=256 used=8 tok=256 | 40 | 528.27 | 25.07 |
| MUL_MAT_ID | ffn_moe_up | ZenDNN | bf16 | K=2048 M=512 E=256 used=8 tok=256 | 40 | 523.10 | 24.83 |
| MUL_MAT_ID | ffn_moe_down | ZenDNN | bf16 | K=512 M=2048 E=256 used=8 tok=256 | 40 | 489.07 | 23.21 |
| CONCAT | conv_input | CPU | - | - | 30 | 230.98 | 10.96 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 30 | 48.97 | 2.4 |
| MUL_MAT | (unnamed) | ZenDNN | bf16 | K=2048 M=8192 N=256 B=1 | 30 | 33.81 | 1.51 |
| SIGMOID | gate_sigmoid | CPU | - | - | 10 | 30.30 | 1.44 |
| MUL_MAT | linear_attn_out | ZenDNN | bf16 | K=4096 M=2048 N=256 B=1 | 30 | 18.47 | 0.88 |
| MUL_MAT | z | ZenDNN | bf16 | K=2048 M=4096 N=256 B=1 | 30 | 17.49 | 0.83 |
| MUL | ffn_moe_weighted | CPU | - | - | 40 | 15.93 | 0.76 |
| SSM_CONV | conv_output_raw | CPU | - | - | 30 | 13.62 | 0.65 |
| RMS_NORM | norm | CPU | - | - | 131 | 12.78 | 0.61 |
| MUL_MAT | Qcur_full | ZenDNN | bf16 | K=2048 M=8192 N=256 B=1 | 10 | 11.21 | 0.53 |
| FLASH_ATTN_EXT | (unnamed) | CPU | - | - | 10 | 9.15 | 0.4 |
| MUL_MAT | result_output | ZenDNN | bf16 | K=2048 M=248320 N=1 B=1 | 1 | 8.57 | 0.41 |

### top rows: zendnn_fb0_pp8192_ub4096

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| CONCAT | conv_input | CPU | - | - | 60 | 14688.11 | 46.18 |
| FLASH_ATTN_EXT | (unnamed) | CPU | - | - | 20 | 2992.71 | 9.4 |
| MUL_MAT_ID | ffn_moe_down | ZenDNN | bf16 | K=512 M=2048 E=256 used=8 tok=4096 | 80 | 2092.57 | 6.58 |
| MUL_MAT_ID | ffn_moe_up | ZenDNN | bf16 | K=2048 M=512 E=256 used=8 tok=4096 | 80 | 1842.54 | 5.79 |
| MUL_MAT_ID | ffn_moe_gate | ZenDNN | bf16 | K=2048 M=512 E=256 used=8 tok=4096 | 80 | 1803.96 | 5.67 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 60 | 1538.71 | 4.8 |
| SIGMOID | gate_sigmoid | CPU | - | - | 20 | 907.96 | 2.85 |
| MUL_MAT | (unnamed) | ZenDNN | bf16 | K=2048 M=8192 N=4096 B=1 | 60 | 797.59 | 2.4 |
| MUL_MAT | linear_attn_out | ZenDNN | bf16 | K=4096 M=2048 N=4096 B=1 | 60 | 501.04 | 1.58 |
| SSM_CONV | conv_output_raw | CPU | - | - | 60 | 439.38 | 1.38 |
| MUL_MAT | z | ZenDNN | bf16 | K=2048 M=4096 N=4096 B=1 | 60 | 436.56 | 1.37 |
| MUL | ffn_moe_weighted | CPU | - | - | 80 | 363.29 | 1.14 |
| RMS_NORM | norm | CPU | - | - | 262 | 347.89 | 1.09 |
| ADD | (unnamed) | CPU | - | - | 540 | 277.91 | 0.41 |
| MUL_MAT | Qcur_full | ZenDNN | bf16 | K=2048 M=8192 N=4096 B=1 | 20 | 265.73 | 0.84 |

### top rows: zendnn_fb0_pp8192_ub512

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| CONCAT | conv_input | CPU | - | - | 480 | 5534.29 | 19.59 |
| MUL_MAT_ID | ffn_moe_down | ZenDNN | bf16 | K=512 M=2048 E=256 used=8 tok=512 | 640 | 3775.55 | 13.36 |
| MUL_MAT_ID | ffn_moe_up | ZenDNN | bf16 | K=2048 M=512 E=256 used=8 tok=512 | 640 | 3621.26 | 12.82 |
| MUL_MAT_ID | ffn_moe_gate | ZenDNN | bf16 | K=2048 M=512 E=256 used=8 tok=512 | 640 | 3602.07 | 12.75 |
| FLASH_ATTN_EXT | (unnamed) | CPU | - | - | 160 | 2656.99 | 9.4 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 480 | 1559.72 | 5.4 |
| MUL_MAT | (unnamed) | ZenDNN | bf16 | K=2048 M=8192 N=512 B=1 | 480 | 943.88 | 3.31 |
| SIGMOID | gate_sigmoid | CPU | - | - | 160 | 935.69 | 3.31 |
| MUL_MAT | linear_attn_out | ZenDNN | bf16 | K=4096 M=2048 N=512 B=1 | 480 | 537.54 | 1.90 |
| MUL_MAT | z | ZenDNN | bf16 | K=2048 M=4096 N=512 B=1 | 480 | 465.61 | 1.65 |
| SSM_CONV | conv_output_raw | CPU | - | - | 480 | 430.83 | 1.52 |
| RMS_NORM | norm | CPU | - | - | 2096 | 386.66 | 1.37 |
| MUL | ffn_moe_weighted | CPU | - | - | 640 | 362.56 | 1.28 |
| MUL_MAT | Qcur_full | ZenDNN | bf16 | K=2048 M=8192 N=512 B=1 | 160 | 314.80 | 1.11 |
| RMS_NORM | (unnamed) | CPU | - | - | 960 | 222.76 | 0.79 |

### top rows: zendnn_pp256_ub512

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| MUL_MAT_ID | ffn_moe_down | CPU | bf16 | K=512 M=2048 E=256 used=8 tok=256 | 40 | 184.74 | 18.94 |
| CONCAT | conv_input | CPU | - | - | 30 | 166.22 | 17.04 |
| MUL_MAT_ID | ffn_moe_gate | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=256 | 40 | 149.24 | 15.30 |
| MUL_MAT_ID | ffn_moe_up | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=256 | 40 | 144.61 | 14.82 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 30 | 49.16 | 5.1 |
| MUL_MAT | (unnamed) | ZenDNN | bf16 | K=2048 M=8192 N=256 B=1 | 30 | 34.8 | 3.48 |
| SIGMOID | gate_sigmoid | CPU | - | - | 10 | 30.47 | 3.12 |
| MUL_MAT | linear_attn_out | ZenDNN | bf16 | K=4096 M=2048 N=256 B=1 | 30 | 18.72 | 1.92 |
| MUL_MAT | z | ZenDNN | bf16 | K=2048 M=4096 N=256 B=1 | 30 | 17.86 | 1.83 |
| SSM_CONV | conv_output_raw | CPU | - | - | 30 | 13.47 | 1.38 |
| RMS_NORM | norm | CPU | - | - | 131 | 12.84 | 1.32 |
| MUL_MAT | Qcur_full | ZenDNN | bf16 | K=2048 M=8192 N=256 B=1 | 10 | 11.17 | 1.14 |
| FLASH_ATTN_EXT | (unnamed) | CPU | - | - | 10 | 9.28 | 0.95 |
| MUL_MAT | result_output | CPU | bf16 | K=2048 M=248320 N=1 B=1 | 1 | 8.66 | 0.89 |
| MUL | ffn_moe_weighted | CPU | - | - | 40 | 8.23 | 0.84 |

### top rows: zendnn_pp8192_ub4096

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| CONCAT | conv_input | CPU | - | - | 60 | 14571.20 | 42.31 |
| MUL_MAT_ID | ffn_moe_down | CPU | bf16 | K=512 M=2048 E=256 used=8 tok=4096 | 80 | 3358.98 | 9.75 |
| FLASH_ATTN_EXT | (unnamed) | CPU | - | - | 20 | 2980.3 | 8.66 |
| MUL_MAT_ID | ffn_moe_gate | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=4096 | 80 | 2581.94 | 7.50 |
| MUL_MAT_ID | ffn_moe_up | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=4096 | 80 | 2557.16 | 7.43 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 60 | 1540.34 | 4.5 |
| SIGMOID | gate_sigmoid | CPU | - | - | 20 | 909.26 | 2.64 |
| MUL_MAT | (unnamed) | ZenDNN | bf16 | K=2048 M=8192 N=4096 B=1 | 60 | 804.22 | 2.4 |
| MUL_MAT | linear_attn_out | ZenDNN | bf16 | K=4096 M=2048 N=4096 B=1 | 60 | 501.65 | 1.46 |
| MUL_MAT | z | ZenDNN | bf16 | K=2048 M=4096 N=4096 B=1 | 60 | 436.91 | 1.27 |
| SSM_CONV | conv_output_raw | CPU | - | - | 60 | 433.56 | 1.26 |
| RMS_NORM | norm | CPU | - | - | 262 | 344.95 | 1.00 |
| MUL | ffn_moe_weighted | CPU | - | - | 80 | 317.57 | 0.92 |
| ADD | (unnamed) | CPU | - | - | 540 | 283.64 | 0.4 |
| MUL_MAT | Qcur_full | ZenDNN | bf16 | K=2048 M=8192 N=4096 B=1 | 20 | 267.52 | 0.78 |

### top rows: zendnn_pp8192_ub512

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| CONCAT | conv_input | CPU | - | - | 480 | 7193.91 | 23.11 |
| MUL_MAT_ID | ffn_moe_down | CPU | bf16 | K=512 M=2048 E=256 used=8 tok=512 | 640 | 4972.37 | 15.98 |
| MUL_MAT_ID | ffn_moe_gate | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=512 | 640 | 3731.32 | 11.99 |
| MUL_MAT_ID | ffn_moe_up | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=512 | 640 | 3634.42 | 11.68 |
| FLASH_ATTN_EXT | (unnamed) | CPU | - | - | 160 | 2662.15 | 8.56 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 480 | 1565.68 | 5.1 |
| MUL_MAT | (unnamed) | ZenDNN | bf16 | K=2048 M=8192 N=512 B=1 | 480 | 942.72 | 3.02 |
| SIGMOID | gate_sigmoid | CPU | - | - | 160 | 928.50 | 2.98 |
| MUL_MAT | linear_attn_out | ZenDNN | bf16 | K=4096 M=2048 N=512 B=1 | 480 | 532.41 | 1.71 |
| MUL_MAT | z | ZenDNN | bf16 | K=2048 M=4096 N=512 B=1 | 480 | 491.18 | 1.58 |
| SSM_CONV | conv_output_raw | CPU | - | - | 480 | 442.97 | 1.42 |
| RMS_NORM | norm | CPU | - | - | 2096 | 379.16 | 1.22 |
| MUL_MAT | Qcur_full | ZenDNN | bf16 | K=2048 M=8192 N=512 B=1 | 160 | 313.34 | 1.01 |
| RMS_NORM | (unnamed) | CPU | - | - | 960 | 230.84 | 0.6 |
| MUL | ffn_moe_weighted | CPU | - | - | 640 | 215.83 | 0.69 |

## vLLM torch profile: vllm_profile_pp256  (Self CPU time total: 607.571ms)

| bucket (self CPU ms, top-50 ops) | ms |
|---|---|
| MoE routed experts (zentorch_fused_moe / fused_moe) | 356 |
| linear / GEMM (zentorch_linear, mm, addmm) | 122 |
| other (top-50 rows only) | 86 |
| GDN / linear attention | 34 |
| attention | 9 |

| op | self CPU % | self CPU | calls |
|---|---|---|---|
| zentorch::zentorch_fused_moe | 56.29% | 342.012ms | 40 |
| zentorch_linear_unary | 18.61% | 113.063ms | 270 |
| _C::chunk_gated_delta_rule_cpu | 3.61% | 21.918ms | 30 |
| ## Call CompiledFxGraph ficzsnw5o67l4e6rfqartz5zcj62... | 3.58% | 21.721ms | 1 |
| execute_context_1(256)_generation_0(0) | 3.39% | 20.600ms | 1 |
| aten::topk | 2.83% | 17.209ms | 40 |
| aten::copy_ | 1.55% | 9.391ms | 314 |
| vllm::cpu_gdn_attention_core | 1.51% | 9.193ms | 30 |
| zentorch::zentorch_linear_unary | 1.46% | 8.868ms | 271 |
| zentorch::fused_moe::pass2_parallel_memcpy | 1.45% | 8.836ms | 40 |
| _C::cpu_attention_with_kv_cache | 0.72% | 4.359ms | 10 |
| _C::cpu_attn_reshape_and_cache | 0.65% | 3.968ms | 10 |
| zentorch::fused_moe::pass1_active_set_build | 0.53% | 3.246ms | 40 |
| _C::causal_conv1d_fwd_cpu | 0.48% | 2.916ms | 30 |
| aten::cat | 0.40% | 2.405ms | 33 |
| zentorch::fused_moe::scratchpad_allocation | 0.30% | 1.796ms | 40 |
| aten::_index_put_impl_ | 0.19% | 1.134ms | 30 |
| aten::_to_copy | 0.18% | 1.109ms | 79 |
| aten::select | 0.17% | 1.007ms | 115 |
| aten::slice | 0.16% | 977.533us | 525 |

## vLLM torch profile: vllm_profile_pp8192  (Self CPU time total: 13.236s)

| bucket (self CPU ms, top-50 ops) | ms |
|---|---|
| MoE routed experts (zentorch_fused_moe / fused_moe) | 6099 |
| attention | 3280 |
| linear / GEMM (zentorch_linear, mm, addmm) | 2482 |
| other (top-50 rows only) | 748 |
| GDN / linear attention | 626 |

| op | self CPU % | self CPU | calls |
|---|---|---|---|
| zentorch::zentorch_fused_moe | 43.08% | 5.702s | 80 |
| _C::cpu_attention_with_kv_cache | 24.74% | 3.275s | 20 |
| zentorch_linear_unary | 18.60% | 2.462s | 540 |
| _C::chunk_gated_delta_rule_cpu | 3.81% | 504.764ms | 60 |
| zentorch::fused_moe::pass2_parallel_memcpy | 2.42% | 320.652ms | 80 |
| ## Call CompiledFxGraph ficzsnw5o67l4e6rfqartz5zcj62... | 2.26% | 299.250ms | 2 |
| aten::copy_ | 1.79% | 236.770ms | 597 |
| aten::cat | 0.85% | 112.597ms | 66 |
| _C::causal_conv1d_fwd_cpu | 0.69% | 91.180ms | 60 |
| zentorch::fused_moe::pass1_active_set_build | 0.52% | 68.347ms | 80 |
| aten::topk | 0.31% | 40.636ms | 80 |
| vllm::cpu_gdn_attention_core | 0.20% | 27.093ms | 60 |
| execute_context_1(4096)_generation_0(0) | 0.17% | 23.134ms | 2 |
| zentorch::zentorch_linear_unary | 0.15% | 20.391ms | 542 |
| zentorch::fused_moe::scratchpad_allocation | 0.04% | 5.108ms | 80 |
| _C::cpu_attn_reshape_and_cache | 0.03% | 3.595ms | 20 |
| aten::transpose | 0.02% | 2.843ms | 722 |
| zentorch::fused_moe::token_expert_grouping | 0.02% | 2.766ms | 80 |
| _C::fused_gdn_gating_cpu | 0.02% | 2.640ms | 60 |
| aten::slice | 0.02% | 2.510ms | 1042 |

## Op placement (GGML_SCHED_DEBUG=2, last graph): placement_zendnn.txt

```
op           backend  count
ADD          CPU  430
ARGSORT      CPU  40
CLAMP        CPU  40
CONCAT       CPU  30
CONT         CPU  10
CPY          CPU  120
DIV          CPU  40
FLASH_ATTN   CPU  10
GATED_DELT   CPU  30
GET_ROWS     CPU  162
MUL          CPU  281
MUL_MAT      CPU  101
MUL_MAT      ZenDN  290
MUL_MAT_ID   CPU  120
RMS_NORM     CPU  191
ROPE         CPU  20
SCALE        CPU  120
SET_ROWS     CPU  20
SIGMOID      CPU  80
SILU         CPU  60
SOFTPLUS     CPU  30
SOFT_MAX     CPU  40
SSM_CONV     CPU  30
SUM_ROWS     CPU  40
SWIGLU       CPU  80

splits:  CPU=241  ZenDNN=240
```

## Op placement (GGML_SCHED_DEBUG=2, last graph): placement_zendnn_fb0.txt

```
op           backend  count
ADD          CPU  430
ARGSORT      CPU  40
CLAMP        CPU  40
CONCAT       CPU  30
CONT         CPU  10
CPY          CPU  120
DIV          CPU  40
FLASH_ATTN   CPU  10
GATED_DELT   CPU  30
GET_ROWS     CPU  162
MUL          CPU  281
MUL_MAT      ZenDN  391
MUL_MAT_ID   ZenDN  120
RMS_NORM     CPU  191
ROPE         CPU  20
SCALE        CPU  120
SET_ROWS     CPU  20
SIGMOID      CPU  80
SILU         CPU  60
SOFTPLUS     CPU  30
SOFT_MAX     CPU  40
SSM_CONV     CPU  30
SUM_ROWS     CPU  40
SWIGLU       CPU  80

splits:  CPU=381  ZenDNN=381
```

## Where each llama.cpp run's memory lives (numa_maps once the weights are loaded)

| run | resident per node |
|---|---|
| bench_stock | N1=63.0GB N5=0.0GB |
| bench_zendnn | N1=67.9GB N5=0.0GB |
| bench_zendnn_fb0 | N1=75.0GB N5=0.0GB N0=0.0GB |
| bench_zendnn_fb0_grp1 | N1=102.7GB N5=0.0GB N0=0.0GB |
| bench_zendnn_fb0_grp3 | N1=70.5GB N5=0.0GB N0=0.0GB |
| bench_zendnn_fb0_grp5 | N1=99.1GB N5=0.0GB N0=0.0GB |
| opprof_stock_pp256_ub512 |  |
| opprof_stock_pp8192_ub4096 | N1=65.9GB N5=0.0GB N0=0.0GB |
| opprof_stock_pp8192_ub512 | N1=65.1GB N5=0.0GB N0=0.0GB |
| opprof_zendnn_fb0_pp256_ub512 | N1=80.0GB N5=0.0GB N0=0.0GB |
| opprof_zendnn_fb0_pp8192_ub4096 | N1=77.0GB N5=0.0GB N0=0.0GB |
| opprof_zendnn_fb0_pp8192_ub512 | N1=79.3GB N5=0.0GB N0=0.0GB |
| opprof_zendnn_pp256_ub512 |  |
| opprof_zendnn_pp8192_ub4096 | N1=66.7GB N5=0.0GB N0=0.0GB |
| opprof_zendnn_pp8192_ub512 | N1=67.8GB N5=0.0GB N0=0.0GB |

## ZenDNNL calls: zendnnl_zendnn_fb0_pp256.log.gz (one cold pass, includes first-use weight packing)

| group matmul mode | exec_algo | calls | ms | active experts (min/median/max) |
|---|---|---|---|---|
| flat_m_tile_seq_clamp | 1 | 120 | 24734 | 82/132/206 |

| dense matmul kernel | calls | ms |
|---|---|---|
| aocl_dlp_blocked | 391 | 293 |

## Steps

| step | rc | seconds |
|---|---|---|
| environment | 0 | 2 |
| build_opprof | 0 | 1 |
| bench_stock | 0 | 433 |
| bench_zendnn | 0 | 304 |
| bench_zendnn_fb0 | 0 | 602 |
| bench_zendnn_fb0_grp1 | 0 | 74 |
| bench_zendnn_fb0_grp3 | 0 | 64 |
| bench_zendnn_fb0_grp5 | 0 | 38 |
| vllm_pp256 | 0 | 215 |
| vllm_pp8192 | 0 | 269 |
| placement_zendnn | 0 | 174 |
| placement_zendnn_fb0 | 0 | 55 |
| opprof_stock_pp256_ub512 | 0 | 27 |
| opprof_zendnn_pp256_ub512 | 0 | 26 |
| opprof_zendnn_fb0_pp256_ub512 | 0 | 66 |
| opprof_stock_pp8192_ub512 | 0 | 122 |
| opprof_zendnn_pp8192_ub512 | 0 | 124 |
| opprof_zendnn_fb0_pp8192_ub512 | 0 | 221 |
| opprof_stock_pp8192_ub4096 | 0 | 144 |
| opprof_zendnn_pp8192_ub4096 | 0 | 142 |
| opprof_zendnn_fb0_pp8192_ub4096 | 0 | 234 |
| vllm_profile_pp256 | 0 | 223 |
| vllm_profile_pp8192 | 0 | 244 |
| zendnnl_log | 0 | 129 |

