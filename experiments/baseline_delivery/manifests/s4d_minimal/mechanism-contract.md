# Official minimal S4D mechanism contract

Source: state-spaces/s4 at e757cef57d89e448c413de7325ed5601aceaac13, models/s4/s4d.py:S4D. assets.json fixes source/dropout hashes; load_s4d verifies bytes and substitutes only the DropoutNd import path.

Preserve kernel/discretization/convolution/output path and per-parameter _optim metadata. Use d_state=64 and transposed=False. Merge upstream learning rate/decay settings into AdamW groups. Project scheduler, FP32, clipping, validation selection and latest/best separation remain declared common training choices.

This is pedagogical minimal S4D, not full S4/S5 or a strong benchmark recipe. Received CUDA composition evidence lacks an external upstream fixture and S4D GPU recovery; earlier CPU recovery does not certify GPU. Status remains not_verifiable.

Opt-in adaptive-memory execution changes only physical microbatch partitioning under the same logical update budget. It defines a distinct numerical trajectory; it does not establish equivalence to the fixed-policy run or inherit its checkpoint admission. Source-pinned mixer equations and precision remain unchanged. See `docs/baselines/adaptive-memory.md` for rollback, throughput search, bounded recovery coverage and pending hardware validation.
