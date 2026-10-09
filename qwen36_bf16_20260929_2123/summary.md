# Qwen3.6-35B-A3B prefill diagnosis: qwen36_bf16_20260929_2123

## Clean prefill throughput (tokens/s, mean ± sd)

- qwen36_bf16_20260929_2123/bench_zendnn_fb0_grp4.json: unreadable (Expecting value: line 1 column 1 (char 0))
| config | pp256 | pp8192 |
|---|---|---|
| llama.cpp stock ub4096 | 262.0 ± 2.5 | 218.1 ± 13.7 |
| llama.cpp stock ub512 | 250.5 ± 2.3 | 250.0 ± 0.9 |
| llama.cpp zendnn ub4096 | 224.7 ± 3.9 | 220.0 ± 0.1 |
| llama.cpp zendnn ub512 | 226.4 ± 2.8 | 242.5 ± 0.5 |
| llama.cpp zendnn_fb0 ub4096 | 195.0 ± 4.3 (last 3: 194.1; 1st 203.8; n=8) | 242.4 ± 1.2 (last 3: 242.0; 1st 242.7; n=8) |
| llama.cpp zendnn_fb0 ub512 | 106.3 ± 35.4 (last 3: 139.9; 1st 47.0; n=8) | 200.0 ± 2.2 (last 3: 201.0; 1st 196.5; n=8) |
| llama.cpp zendnn_fb0_grp1 ub512 | 176.1 ± 14.9 (last 3: 187.3; 1st 145.8; n=8) |  |
| llama.cpp zendnn_fb0_grp3 ub512 | 118.1 ± 39.1 (last 3: 155.5; 1st 53.0; n=8) |  |
| llama.cpp zendnn_fb0_grp5 ub512 | 230.7 ± 22.7 (last 3: 250.0; 1st 189.1; n=8) |  |
| vLLM (bench latency, out=1) | 436.8 ± 0.5 | 615.3 ± 0.4 |

vLLM t/s = prompt / latency of a 1-token generation (prefill + one sample), batch 1.

## llama.cpp per-op profile (opprof: per-node wall time, ms per prompt)

| run | clean t/s | profiled t/s | sum(nodes) ms |
|---|---|---|---|
| stock_pp256_ub512 | 144.26 | 135.25 | 1887.4 |
| stock_pp8192_ub4096 | 213.71 | 208.18 | 39241.4 |
| stock_pp8192_ub512 | 182.59 | 171.64 | 47580.3 |
| zendnn_fb0_pp256_ub512 | 104.53 | 123.29 | 2070.8 |
| zendnn_fb0_pp8192_ub4096 | 275.06 | 256.81 | 31789.9 |
| zendnn_fb0_pp8192_ub512 | 304.45 | 249.99 | 32620.3 |
| zendnn_pp256_ub512 | 152.20 | 157.20 | 1623.2 |
| zendnn_pp8192_ub4096 | 226.80 | 218.59 | 37368.9 |
| zendnn_pp8192_ub512 | 188.00 | 188.49 | 43312.8 |

| bucket (ms, % of nodes; [ms on ZenDNN]) | stock_pp256_ub512 | stock_pp8192_ub4096 | stock_pp8192_ub512 | zendnn_fb0_pp256_ub512 | zendnn_fb0_pp8192_ub4096 | zendnn_fb0_pp8192_ub512 | zendnn_pp256_ub512 | zendnn_pp8192_ub4096 | zendnn_pp8192_ub512 |
|---|---|---|---|---|---|---|---|---|---|
| MoE routed experts (MUL_MAT_ID) | 1110 (59%) [0] | 11773 (30%) [0] | 25953 (55%) [0] | 1466 (71%) [1466] | 5763 (18%) [5763] | 11012 (34%) [11012] | 1106 (68%) [0] | 11772 (32%) [0] | 25901 (60%) [0] |
| GDN conv-state CONCAT (single-threaded in ggml-cpu) | 272 (14%) [0] | 14269 (36%) [0] | 6716 (14%) [0] | 268 (13%) [0] | 14594 (46%) [0] | 9760 (30%) [0] | 167 (10%) [0] | 14133 (38%) [0] | 5553 (13%) [0] |
| weight matmul: attention / GDN projections | 248 (13%) [0] | 4014 (10%) [0] | 5772 (12%) [0] | 105 (5%) [105] | 2555 (8%) [2555] | 2914 (9%) [2914] | 109 (7%) [104] | 2595 (7%) [2494] | 3008 (7%) [2879] |
| flash attention | 9 (0%) [0] | 3002 (8%) [0] | 2667 (6%) [0] | 9 (0%) [0] | 2997 (9%) [0] | 2685 (8%) [0] | 9 (1%) [0] | 3000 (8%) [0] | 2683 (6%) [0] |
| GDN recurrence + conv1d | 68 (4%) [0] | 2020 (5%) [0] | 2055 (4%) [0] | 62 (3%) [0] | 1954 (6%) [0] | 1998 (6%) [0] | 62 (4%) [0] | 1975 (5%) [0] | 2003 (5%) [0] |
| other (norms, rope, adds, copies) | 53 (3%) [0] | 1319 (3%) [0] | 1386 (3%) [0] | 53 (3%) [0] | 1320 (4%) [0] | 1394 (4%) [0] | 53 (3%) [0] | 1334 (4%) [0] | 1389 (3%) [0] |
| attention output-gate SIGMOID | 30 (2%) [0] | 910 (2%) [0] | 944 (2%) [0] | 30 (1%) [0] | 909 (3%) [0] | 942 (3%) [0] | 31 (2%) [0] | 909 (2%) [0] | 940 (2%) [0] |
| MoE glue (topk, act, weighting) | 25 (1%) [0] | 735 (2%) [0] | 696 (1%) [0] | 31 (1%) [0] | 799 (3%) [0] | 812 (2%) [0] | 25 (2%) [0] | 734 (2%) [0] | 720 (2%) [0] |
| GDN other elementwise | 25 (1%) [0] | 679 (2%) [0] | 759 (2%) [0] | 25 (1%) [0] | 695 (2%) [0] | 761 (2%) [0] | 25 (2%) [0] | 683 (2%) [0] | 770 (2%) [0] |
| weight matmul: MoE router | 12 (1%) [0] | 271 (1%) [0] | 310 (1%) [0] | 6 (0%) [6] | 76 (0%) [76] | 198 (1%) [198] | 6 (0%) [6] | 76 (0%) [76] | 167 (0%) [167] |
| weight matmul: shared expert | 10 (1%) [0] | 198 (1%) [0] | 272 (1%) [0] | 5 (0%) [5] | 108 (0%) [108] | 126 (0%) [126] | 5 (0%) [4] | 108 (0%) [93] | 129 (0%) [111] |
| weight matmul: lm head | 25 (1%) [0] | 50 (0%) [0] | 51 (0%) [0] | 9 (0%) [9] | 18 (0%) [18] | 18 (0%) [18] | 25 (2%) [0] | 51 (0%) [0] | 51 (0%) [0] |

### top rows: stock_pp256_ub512

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| MUL_MAT_ID | ffn_moe_down | CPU | bf16 | K=512 M=2048 E=256 used=8 tok=256 | 40 | 405.59 | 21.49 |
| MUL_MAT_ID | ffn_moe_up | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=256 | 40 | 353.63 | 18.74 |
| MUL_MAT_ID | ffn_moe_gate | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=256 | 40 | 350.75 | 18.58 |
| CONCAT | conv_input | CPU | - | - | 30 | 272.01 | 14.41 |
| MUL_MAT | (unnamed) | CPU | bf16 | K=2048 M=8192 N=256 B=1 | 30 | 85.51 | 4.58 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 30 | 53.95 | 2.81 |
| MUL_MAT | linear_attn_out | CPU | bf16 | K=4096 M=2048 N=256 B=1 | 30 | 45.54 | 2.41 |
| MUL_MAT | z | CPU | bf16 | K=2048 M=4096 N=256 B=1 | 30 | 39.37 | 2.09 |
| SIGMOID | gate_sigmoid | CPU | - | - | 10 | 29.58 | 1.57 |
| MUL_MAT | Qcur_full | CPU | bf16 | K=2048 M=8192 N=256 B=1 | 10 | 28.23 | 1.50 |
| MUL_MAT | result_output | CPU | bf16 | K=2048 M=248320 N=1 B=1 | 1 | 25.35 | 1.34 |
| MUL_MAT | attn_output | CPU | bf16 | K=4096 M=2048 N=256 B=1 | 10 | 14.97 | 0.79 |
| MUL_MAT | ffn_gate | CPU | bf16 | K=2048 M=512 N=256 B=1 | 40 | 14.90 | 0.79 |
| SSM_CONV | conv_output_raw | CPU | - | - | 30 | 13.87 | 0.73 |
| RMS_NORM | norm | CPU | - | - | 131 | 12.99 | 0.69 |

### top rows: stock_pp8192_ub4096

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| CONCAT | conv_input | CPU | - | - | 60 | 14269.46 | 36.36 |
| MUL_MAT_ID | ffn_moe_down | CPU | bf16 | K=512 M=2048 E=256 used=8 tok=4096 | 80 | 4469.56 | 11.39 |
| MUL_MAT_ID | ffn_moe_up | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=4096 | 80 | 3664.79 | 9.34 |
| MUL_MAT_ID | ffn_moe_gate | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=4096 | 80 | 3638.49 | 9.27 |
| FLASH_ATTN_EXT | (unnamed) | CPU | - | - | 20 | 3002.19 | 7.64 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 60 | 1591.81 | 4.11 |
| MUL_MAT | (unnamed) | CPU | bf16 | K=2048 M=8192 N=4096 B=1 | 60 | 1276.72 | 3.3 |
| SIGMOID | gate_sigmoid | CPU | - | - | 20 | 909.76 | 2.32 |
| MUL_MAT | linear_attn_out | CPU | bf16 | K=4096 M=2048 N=4096 B=1 | 60 | 785.05 | 2.00 |
| MUL_MAT | z | CPU | bf16 | K=2048 M=4096 N=4096 B=1 | 60 | 637.84 | 1.63 |
| SSM_CONV | conv_output_raw | CPU | - | - | 60 | 428.30 | 1.09 |
| MUL_MAT | Qcur_full | CPU | bf16 | K=2048 M=8192 N=4096 B=1 | 20 | 427.14 | 1.09 |
| RMS_NORM | norm | CPU | - | - | 262 | 355.67 | 0.91 |
| MUL | ffn_moe_weighted | CPU | - | - | 80 | 313.49 | 0.80 |
| ADD | (unnamed) | CPU | - | - | 540 | 283.8 | 0.41 |

### top rows: stock_pp8192_ub512

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| MUL_MAT_ID | ffn_moe_down | CPU | bf16 | K=512 M=2048 E=256 used=8 tok=512 | 640 | 9708.76 | 20.40 |
| MUL_MAT_ID | ffn_moe_up | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=512 | 640 | 8155.15 | 17.14 |
| MUL_MAT_ID | ffn_moe_gate | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=512 | 640 | 8089.25 | 17.00 |
| CONCAT | conv_input | CPU | - | - | 480 | 6716.37 | 14.12 |
| FLASH_ATTN_EXT | (unnamed) | CPU | - | - | 160 | 2666.73 | 5.6 |
| MUL_MAT | (unnamed) | CPU | bf16 | K=2048 M=8192 N=512 B=1 | 480 | 1979.18 | 4.2 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 480 | 1629.67 | 3.31 |
| MUL_MAT | linear_attn_out | CPU | bf16 | K=4096 M=2048 N=512 B=1 | 480 | 1045.71 | 2.20 |
| SIGMOID | gate_sigmoid | CPU | - | - | 160 | 943.71 | 1.98 |
| MUL_MAT | z | CPU | bf16 | K=2048 M=4096 N=512 B=1 | 480 | 915.79 | 1.92 |
| MUL_MAT | Qcur_full | CPU | bf16 | K=2048 M=8192 N=512 B=1 | 160 | 658.12 | 1.38 |
| SSM_CONV | conv_output_raw | CPU | - | - | 480 | 425.17 | 0.89 |
| RMS_NORM | norm | CPU | - | - | 2096 | 396.50 | 0.83 |
| MUL_MAT | attn_output | CPU | bf16 | K=4096 M=2048 N=512 B=1 | 160 | 347.94 | 0.73 |
| MUL_MAT | ffn_gate | CPU | bf16 | K=2048 M=512 N=512 B=1 | 640 | 328.05 | 0.69 |

### top rows: zendnn_fb0_pp256_ub512

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| MUL_MAT_ID | ffn_moe_gate | ZenDNN | bf16 | K=2048 M=512 E=256 used=8 tok=256 | 40 | 531.79 | 25.68 |
| MUL_MAT_ID | ffn_moe_up | ZenDNN | bf16 | K=2048 M=512 E=256 used=8 tok=256 | 40 | 528.95 | 25.54 |
| MUL_MAT_ID | ffn_moe_down | ZenDNN | bf16 | K=512 M=2048 E=256 used=8 tok=256 | 40 | 405.23 | 19.57 |
| CONCAT | conv_input | CPU | - | - | 30 | 268.50 | 12.97 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 30 | 48.67 | 2.4 |
| MUL_MAT | (unnamed) | ZenDNN | bf16 | K=2048 M=8192 N=256 B=1 | 30 | 33.9 | 1.54 |
| SIGMOID | gate_sigmoid | CPU | - | - | 10 | 30.37 | 1.47 |
| MUL_MAT | linear_attn_out | ZenDNN | bf16 | K=4096 M=2048 N=256 B=1 | 30 | 18.53 | 0.89 |
| MUL_MAT | z | ZenDNN | bf16 | K=2048 M=4096 N=256 B=1 | 30 | 17.46 | 0.84 |
| MUL | ffn_moe_weighted | CPU | - | - | 40 | 14.67 | 0.71 |
| SSM_CONV | conv_output_raw | CPU | - | - | 30 | 13.70 | 0.66 |
| RMS_NORM | norm | CPU | - | - | 131 | 12.90 | 0.62 |
| MUL_MAT | Qcur_full | ZenDNN | bf16 | K=2048 M=8192 N=256 B=1 | 10 | 11.29 | 0.55 |
| FLASH_ATTN_EXT | (unnamed) | CPU | - | - | 10 | 9.22 | 0.41 |
| MUL_MAT | result_output | ZenDNN | bf16 | K=2048 M=248320 N=1 B=1 | 1 | 8.57 | 0.41 |

### top rows: zendnn_fb0_pp8192_ub4096

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| CONCAT | conv_input | CPU | - | - | 60 | 14594.50 | 45.91 |
| FLASH_ATTN_EXT | (unnamed) | CPU | - | - | 20 | 2997.12 | 9.42 |
| MUL_MAT_ID | ffn_moe_down | ZenDNN | bf16 | K=512 M=2048 E=256 used=8 tok=4096 | 80 | 2123.34 | 6.68 |
| MUL_MAT_ID | ffn_moe_up | ZenDNN | bf16 | K=2048 M=512 E=256 used=8 tok=4096 | 80 | 1829.96 | 5.76 |
| MUL_MAT_ID | ffn_moe_gate | ZenDNN | bf16 | K=2048 M=512 E=256 used=8 tok=4096 | 80 | 1809.96 | 5.69 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 60 | 1519.86 | 4.8 |
| SIGMOID | gate_sigmoid | CPU | - | - | 20 | 909.30 | 2.86 |
| MUL_MAT | (unnamed) | ZenDNN | bf16 | K=2048 M=8192 N=4096 B=1 | 60 | 809.71 | 2.44 |
| MUL_MAT | linear_attn_out | ZenDNN | bf16 | K=4096 M=2048 N=4096 B=1 | 60 | 505.42 | 1.59 |
| MUL_MAT | z | ZenDNN | bf16 | K=2048 M=4096 N=4096 B=1 | 60 | 440.46 | 1.39 |
| SSM_CONV | conv_output_raw | CPU | - | - | 60 | 434.51 | 1.37 |
| MUL | ffn_moe_weighted | CPU | - | - | 80 | 366.82 | 1.15 |
| RMS_NORM | norm | CPU | - | - | 262 | 351.22 | 1.10 |
| ADD | (unnamed) | CPU | - | - | 540 | 278.21 | 0.4 |
| MUL_MAT | Qcur_full | ZenDNN | bf16 | K=2048 M=8192 N=4096 B=1 | 20 | 266.83 | 0.84 |

### top rows: zendnn_fb0_pp8192_ub512

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| CONCAT | conv_input | CPU | - | - | 480 | 9759.75 | 29.92 |
| MUL_MAT_ID | ffn_moe_down | ZenDNN | bf16 | K=512 M=2048 E=256 used=8 tok=512 | 640 | 3763.34 | 11.54 |
| MUL_MAT_ID | ffn_moe_up | ZenDNN | bf16 | K=2048 M=512 E=256 used=8 tok=512 | 640 | 3632.68 | 11.14 |
| MUL_MAT_ID | ffn_moe_gate | ZenDNN | bf16 | K=2048 M=512 E=256 used=8 tok=512 | 640 | 3616.38 | 11.09 |
| FLASH_ATTN_EXT | (unnamed) | CPU | - | - | 160 | 2684.94 | 8.25 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 480 | 1553.77 | 4.8 |
| MUL_MAT | (unnamed) | ZenDNN | bf16 | K=2048 M=8192 N=512 B=1 | 480 | 952.95 | 3.01 |
| SIGMOID | gate_sigmoid | CPU | - | - | 160 | 942.22 | 2.89 |
| MUL_MAT | linear_attn_out | ZenDNN | bf16 | K=4096 M=2048 N=512 B=1 | 480 | 535.38 | 1.64 |
| MUL_MAT | z | ZenDNN | bf16 | K=2048 M=4096 N=512 B=1 | 480 | 468.55 | 1.44 |
| SSM_CONV | conv_output_raw | CPU | - | - | 480 | 444.40 | 1.36 |
| RMS_NORM | norm | CPU | - | - | 2096 | 386.98 | 1.19 |
| MUL | ffn_moe_weighted | CPU | - | - | 640 | 362.58 | 1.11 |
| MUL_MAT | Qcur_full | ZenDNN | bf16 | K=2048 M=8192 N=512 B=1 | 160 | 316.64 | 0.97 |
| RMS_NORM | (unnamed) | CPU | - | - | 960 | 234.77 | 0.61 |

### top rows: zendnn_pp256_ub512

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| MUL_MAT_ID | ffn_moe_down | CPU | bf16 | K=512 M=2048 E=256 used=8 tok=256 | 40 | 404.15 | 24.90 |
| MUL_MAT_ID | ffn_moe_up | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=256 | 40 | 352.70 | 21.73 |
| MUL_MAT_ID | ffn_moe_gate | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=256 | 40 | 349.24 | 21.52 |
| CONCAT | conv_input | CPU | - | - | 30 | 166.51 | 10.26 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 30 | 48.73 | 3 |
| MUL_MAT | (unnamed) | ZenDNN | bf16 | K=2048 M=8192 N=256 B=1 | 30 | 33.6 | 2.1 |
| SIGMOID | gate_sigmoid | CPU | - | - | 10 | 30.78 | 1.90 |
| MUL_MAT | result_output | CPU | bf16 | K=2048 M=248320 N=1 B=1 | 1 | 25.33 | 1.56 |
| MUL_MAT | linear_attn_out | ZenDNN | bf16 | K=4096 M=2048 N=256 B=1 | 30 | 18.70 | 1.15 |
| MUL_MAT | z | ZenDNN | bf16 | K=2048 M=4096 N=256 B=1 | 30 | 17.90 | 1.10 |
| SSM_CONV | conv_output_raw | CPU | - | - | 30 | 13.15 | 0.81 |
| RMS_NORM | norm | CPU | - | - | 131 | 12.81 | 0.79 |
| MUL_MAT | Qcur_full | ZenDNN | bf16 | K=2048 M=8192 N=256 B=1 | 10 | 11.15 | 0.69 |
| FLASH_ATTN_EXT | (unnamed) | CPU | - | - | 10 | 9.25 | 0.6 |
| MUL | ffn_moe_weighted | CPU | - | - | 40 | 9.06 | 0.56 |

### top rows: zendnn_pp8192_ub4096

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| CONCAT | conv_input | CPU | - | - | 60 | 14132.57 | 37.82 |
| MUL_MAT_ID | ffn_moe_down | CPU | bf16 | K=512 M=2048 E=256 used=8 tok=4096 | 80 | 4456.14 | 11.92 |
| MUL_MAT_ID | ffn_moe_gate | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=4096 | 80 | 3662.36 | 9.80 |
| MUL_MAT_ID | ffn_moe_up | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=4096 | 80 | 3653.13 | 9.78 |
| FLASH_ATTN_EXT | (unnamed) | CPU | - | - | 20 | 2999.5 | 8.03 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 60 | 1541.89 | 4.21 |
| SIGMOID | gate_sigmoid | CPU | - | - | 20 | 909.17 | 2.43 |
| MUL_MAT | (unnamed) | ZenDNN | bf16 | K=2048 M=8192 N=4096 B=1 | 60 | 808.76 | 2.12 |
| MUL_MAT | linear_attn_out | ZenDNN | bf16 | K=4096 M=2048 N=4096 B=1 | 60 | 502.08 | 1.34 |
| MUL_MAT | z | ZenDNN | bf16 | K=2048 M=4096 N=4096 B=1 | 60 | 441.99 | 1.18 |
| SSM_CONV | conv_output_raw | CPU | - | - | 60 | 433.22 | 1.16 |
| RMS_NORM | norm | CPU | - | - | 262 | 354.16 | 0.95 |
| MUL | ffn_moe_weighted | CPU | - | - | 80 | 315.99 | 0.85 |
| ADD | (unnamed) | CPU | - | - | 540 | 286 | 0.4 |
| MUL_MAT | Qcur_full | ZenDNN | bf16 | K=2048 M=8192 N=4096 B=1 | 20 | 268.18 | 0.72 |

### top rows: zendnn_pp8192_ub512

| op | stem | backend | wtype | shape | calls | ms | % |
|---|---|---|---|---|---|---|---|
| MUL_MAT_ID | ffn_moe_down | CPU | bf16 | K=512 M=2048 E=256 used=8 tok=512 | 640 | 9720.34 | 22.44 |
| MUL_MAT_ID | ffn_moe_up | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=512 | 640 | 8146.71 | 18.81 |
| MUL_MAT_ID | ffn_moe_gate | CPU | bf16 | K=2048 M=512 E=256 used=8 tok=512 | 640 | 8033.87 | 18.55 |
| CONCAT | conv_input | CPU | - | - | 480 | 5553.06 | 12.82 |
| FLASH_ATTN_EXT | (unnamed) | CPU | - | - | 160 | 2682.65 | 6.2 |
| GATED_DELTA_NET | (unnamed) | CPU | - | - | 480 | 1572.77 | 3.6 |
| MUL_MAT | (unnamed) | ZenDNN | bf16 | K=2048 M=8192 N=512 B=1 | 480 | 950.17 | 2.12 |
| SIGMOID | gate_sigmoid | CPU | - | - | 160 | 940.05 | 2.17 |
| MUL_MAT | linear_attn_out | ZenDNN | bf16 | K=4096 M=2048 N=512 B=1 | 480 | 532.20 | 1.23 |
| MUL_MAT | z | ZenDNN | bf16 | K=2048 M=4096 N=512 B=1 | 480 | 498.56 | 1.15 |
| SSM_CONV | conv_output_raw | CPU | - | - | 480 | 430.47 | 0.99 |
| RMS_NORM | norm | CPU | - | - | 2096 | 387.75 | 0.90 |
| MUL_MAT | Qcur_full | ZenDNN | bf16 | K=2048 M=8192 N=512 B=1 | 160 | 314.90 | 0.73 |
| MUL | ffn_moe_weighted | CPU | - | - | 640 | 250.74 | 0.58 |
| RMS_NORM | (unnamed) | CPU | - | - | 960 | 226.15 | 0.6 |

## vLLM torch profile: vllm_profile_pp256  (Self CPU time total: 616.849ms)

| bucket (self CPU ms, top-50 ops) | ms |
|---|---|
| MoE routed experts (zentorch_fused_moe / fused_moe) | 358 |
| linear / GEMM (zentorch_linear, mm, addmm) | 123 |
| other (top-50 rows only) | 90 |
| GDN / linear attention | 37 |
| attention | 9 |

| op | self CPU % | self CPU | calls |
|---|---|---|---|
| zentorch::zentorch_fused_moe | 55.81% | 344.246ms | 40 |
| zentorch_linear_unary | 18.43% | 113.670ms | 270 |
| _C::chunk_gated_delta_rule_cpu | 3.59% | 22.149ms | 30 |
| ## Call CompiledFxGraph ficzsnw5o67l4e6rfqartz5zcj62... | 3.57% | 22.006ms | 1 |
| execute_context_1(256)_generation_0(0) | 3.47% | 21.400ms | 1 |
| aten::topk | 2.86% | 17.664ms | 40 |
| vllm::cpu_gdn_attention_core | 1.80% | 11.089ms | 30 |
| aten::copy_ | 1.70% | 10.473ms | 314 |
| zentorch::zentorch_linear_unary | 1.44% | 8.858ms | 271 |
| zentorch::fused_moe::pass2_parallel_memcpy | 1.43% | 8.798ms | 40 |
| _C::cpu_attention_with_kv_cache | 0.70% | 4.326ms | 10 |
| _C::cpu_attn_reshape_and_cache | 0.60% | 3.698ms | 10 |
| _C::causal_conv1d_fwd_cpu | 0.57% | 3.493ms | 30 |
| zentorch::fused_moe::pass1_active_set_build | 0.52% | 3.225ms | 40 |
| aten::cat | 0.38% | 2.354ms | 33 |
| zentorch::fused_moe::scratchpad_allocation | 0.28% | 1.698ms | 40 |
| aten::clone | 0.23% | 1.408ms | 180 |
| aten::_to_copy | 0.19% | 1.196ms | 79 |
| aten::select | 0.19% | 1.174ms | 115 |
| aten::_index_put_impl_ | 0.18% | 1.140ms | 30 |

## vLLM torch profile: vllm_profile_pp8192  (Self CPU time total: 13.337s)

| bucket (self CPU ms, top-50 ops) | ms |
|---|---|
| MoE routed experts (zentorch_fused_moe / fused_moe) | 6174 |
| attention | 3281 |
| linear / GEMM (zentorch_linear, mm, addmm) | 2502 |
| other (top-50 rows only) | 757 |
| GDN / linear attention | 622 |

| op | self CPU % | self CPU | calls |
|---|---|---|---|
| zentorch::zentorch_fused_moe | 43.25% | 5.768s | 80 |
| _C::cpu_attention_with_kv_cache | 24.56% | 3.276s | 20 |
| zentorch_linear_unary | 18.60% | 2.481s | 540 |
| _C::chunk_gated_delta_rule_cpu | 3.75% | 500.359ms | 60 |
| zentorch::fused_moe::pass2_parallel_memcpy | 2.46% | 328.057ms | 80 |
| ## Call CompiledFxGraph ficzsnw5o67l4e6rfqartz5zcj62... | 2.31% | 308.120ms | 2 |
| aten::copy_ | 1.82% | 242.254ms | 597 |
| aten::cat | 0.84% | 111.487ms | 66 |
| _C::causal_conv1d_fwd_cpu | 0.68% | 90.153ms | 60 |
| zentorch::fused_moe::pass1_active_set_build | 0.53% | 70.228ms | 80 |
| aten::topk | 0.31% | 41.008ms | 80 |
| vllm::cpu_gdn_attention_core | 0.21% | 28.455ms | 60 |
| zentorch::zentorch_linear_unary | 0.16% | 21.207ms | 542 |
| execute_context_1(4096)_generation_0(0) | 0.12% | 15.433ms | 2 |
| zentorch::fused_moe::scratchpad_allocation | 0.04% | 5.012ms | 80 |
| aten::transpose | 0.03% | 3.795ms | 722 |
| _C::cpu_attn_reshape_and_cache | 0.03% | 3.750ms | 20 |
| _C::fused_gdn_gating_cpu | 0.02% | 3.076ms | 60 |
| aten::t | 0.02% | 3.021ms | 542 |
| aten::as_strided | 0.02% | 3.019ms | 2402 |

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

## ZenDNNL calls: zendnnl_zendnn_fb0_pp256.log.gz (one cold pass, includes first-use weight packing)

| group matmul mode | exec_algo | calls | ms | active experts (min/median/max) |
|---|---|---|---|---|
| flat_m_tile_seq_clamp | 1 | 120 | 30760 | 82/132/206 |

| dense matmul kernel | calls | ms |
|---|---|---|
| aocl_dlp_blocked | 391 | 299 |

## Steps

| step | rc | seconds |
|---|---|---|
| environment | 0 | 1 |
| build_opprof | 0 | 1 |
| bench_stock | 0 | 290 |
| bench_zendnn | 0 | 296 |
| bench_zendnn_fb0 | 0 | 746 |
| bench_zendnn_fb0_grp1 | 0 | 20 |
| bench_zendnn_fb0_grp3 | 0 | 49 |
| bench_zendnn_fb0_grp4 | 143 | 657 |
| bench_zendnn_fb0_grp5 | 0 | 16 |
| vllm_pp256 | 0 | 277 |
| vllm_pp8192 | 0 | 292 |
| placement_zendnn | 0 | 8 |
| placement_zendnn_fb0 | 0 | 37 |
| opprof_stock_pp256_ub512 | 0 | 9 |
| opprof_zendnn_pp256_ub512 | 0 | 9 |
| opprof_zendnn_fb0_pp256_ub512 | 0 | 49 |
| opprof_stock_pp8192_ub512 | 0 | 142 |
| opprof_zendnn_pp8192_ub512 | 0 | 136 |
| opprof_zendnn_fb0_pp8192_ub512 | 0 | 191 |
| opprof_stock_pp8192_ub4096 | 0 | 138 |
| opprof_zendnn_pp8192_ub4096 | 0 | 131 |
| opprof_zendnn_fb0_pp8192_ub4096 | 0 | 216 |
| vllm_profile_pp256 | 0 | 199 |
| vllm_profile_pp8192 | 0 | 237 |
| zendnnl_log | 0 | 39 |

