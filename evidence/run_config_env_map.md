# run_config.json key -> environment variable

Generated from `train_qat.py`'s `_env(...)` / `os.environ.get(...)` calls. Keys
in `run_config.json` are the Python variable names, which often differ from
the env name. **Defaults differ from the record recipe in dangerous ways**
(e.g. `PJ_FFN_MULT` defaults to 4 but the record is 1, D141; GMP range
defaults to ±0.4 but the record uses ±1.6; `PJ_PRE_NORM` defaults to empty
but the record uses tanh). Always set every structural knob explicitly.

Not in run_config.json but read from env: `PJ_SEED`, `PJ_DISTILL_MIN_ACC`,
`KERAS_BACKEND` (train_qat.py setdefaults it to jax). The LR restart cycle is
derived, not set: `max(int(500 * PJ_EPOCHS / 7000), 10)` epochs, so `PJ_EPOCHS`
controls cycle length (D144; 600 epochs gives the record's 42-epoch cycles).

| run_config key | env var | code default | record (`g-ls-s3`) |
|---|---|---|---|
| `ARCH` | `PJ_ARCH` | (see code) | `phat` |
| `ATTN_ACT` | `PJ_ATTN_ACT` | `"softmax"` | `softmax` |
| `ATTN_ACT_GLOBAL` | `PJ_ATTN_ACT_GLOBAL` | `""` | `None` |
| `ATTN_ACT_LOCAL` | `PJ_ATTN_ACT_LOCAL` | `""` | `None` |
| `ATTN_BETA_SCALE` | `PJ_ATTN_BETA_SCALE` | (see code) | `None` |
| `ATTN_BITS` | `PJ_ATTN_BITS` | (see code) | `None` |
| `ATTN_FLOOR` | `PJ_ATTN_FLOOR` | (see code) | `None` |
| `ATTN_L1` | `PJ_ATTN_L1` | (see code) | `None` |
| `ATTN_LANE_NAMES` | (constant) | | `['local_attn_qk', 'local_attn_softmax', 'local_attn_av', 'patch_attn_qk', 'patch_attn_softmax', 'patch_attn_av']` |
| `AV_L1` | `PJ_AV_L1` | (see code) | `None` |
| `BATCH` | `PJ_BATCH` | `2790` | `2790` |
| `BETA_CONST` | `PJ_BETA_CONST` | (see code) | `None` |
| `BETA_SCALE` | `PJ_BETA_SCALE` | `1.0` | `1.0` |
| `BW_A` | `PJ_BW_A` | `7` | `7` |
| `BW_K` | `PJ_BW_K` | `7` | `7` |
| `DATA` | `PJ_DATA` | `"data/jets_128x3.npz"` | `/data/phatjet-fpga/jets_128x3_aaron_ourunits_kt.npz` |
| `DISTILL_ALPHA` | `PJ_DISTILL_ALPHA` | `0.5` | `0.5` |
| `DISTILL_CKPT` | `PJ_DISTILL_CKPT` | (see code) | `None` |
| `D_MODEL` | `PJ_D_MODEL` | `16` | `16` |
| `EBOPS_TARGET` | `PJ_EBOPS_TARGET` | (see code) | `1700000.0` |
| `EPOCHS` | `PJ_EPOCHS` | `200` | `600` |
| `FFN_L1` | `PJ_FFN_L1` | (see code) | `None` |
| `FFN_MULT` | `PJ_FFN_MULT` | `4` | `1` |
| `FLOAT_CKPT` | `PJ_FLOAT_CKPT` | `"float_phatjet.keras"` | `/nonexistent-train-from-scratch` |
| `GLOBAL_MODE` | `PJ_GLOBAL_MODE` | (see code) | `mha` |
| `GMP2` | `PJ_GMP2` | `"0"` | `False` |
| `GMP_BINS` | `PJ_GMP_BINS` | `8` | `8` |
| `GMP_EDGES` | `PJ_GMP_EDGES` | unset (uniform grid) | `None` |
| `GMP_HI` | `PJ_GMP_HI` | `0.4` | `1.6` |
| `GMP_KERNEL` | `PJ_GMP_KERNEL` | `3` | `3` |
| `GMP_LO` | `PJ_GMP_LO` | `-0.4` | `-1.6` |
| `GMP_MASK_PADDING` | `PJ_GMP_MASK` | `"0"` | `False` |
| `I0_A` | `PJ_I0_A` | `0` | `0` |
| `I0_K` | `PJ_I0_K` | `0` | `0` |
| `INIT_BETA` | `PJ_INIT_BETA` | (see code) | `1e-09` |
| `INPUT_NORM` | `PJ_INPUT_NORM` | (see code) | `pt64` |
| `L1_REG` | `PJ_L1_REG` | `1e-8` | `1e-08` |
| `LABEL_SMOOTH` | `PJ_LABEL_SMOOTH` | `0.0` | `0.02` |
| `LIN_K` | `PJ_LIN_K` | `4` | `4` |
| `LOCAL_MODE` | `PJ_LOCAL_MODE` | `"patch"` | `patch` |
| `LR` | `PJ_LR` | `3e-3` | `0.0003` |
| `NUM_HEADS` | `PJ_HEADS` | `4` | `4` |
| `NUM_PARTICLES` | `PJ_PARTICLES` | `128` | `128` |
| `OUTDIR` | `PJ_OUT` | `"."` | `/data/phatjet-fpga/runs/g-ls-s3` |
| `OVERFLOW` | `PJ_OVERFLOW` | `"wrap"` | `wrap` |
| `OVERFLOW_K` | `PJ_OVERFLOW_K` | `""` | `wrap` |
| `PATCH_SIZE` | `PJ_PATCH` | `8` | `8` |
| `PATCH_TOKENIZER` | `PJ_TOKENIZER` | `"mean"` | `mean` |
| `PID_I` | `PJ_PID_I` | `1e-3` | `0.001` |
| `PID_P` | `PJ_PID_P` | `0.1` | `0.1` |
| `PRE_NORM` | `PJ_PRE_NORM` | `""` | `tanh` |
| `QKV_MODE` | `PJ_QKV_MODE` | `"dense"` | `dense` |
| `RESUME_CKPT` | `PJ_RESUME_CKPT` | (see code) | `/data/phatjet-fpga/runs/vu-long/pareto/epoch=773-val_acc=0.8077-ebops=1536867.keras` |
| `SORT` | `PJ_SORT` | (see code) | `` |
| `TABLE_L1` | `PJ_TABLE_L1` | (see code) | `None` |
| `TRAIN_SUBSET` | (hard-coded None) | | `None` |
| `USE_BN` | `PJ_USE_BN` | `"0"` | `False` |
| `USE_GMP` | `PJ_USE_GMP` | `"1"` | `True` |
| `WV_L1` | `PJ_WV_L1` | (see code) | `None` |
| `W_L1` | `PJ_W_L1` | (see code) | `None` |
