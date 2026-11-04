# MEGA source-stage contract

Official commit aeaa4b44592cd1d60a9a34554e359eda2a62b03b. strong-source-registry.json pins the LRA encoder, sentence layer, moving-average gated attention, bidirectional multi-head EMA and LRA model/config source identities. strong_sources.official_source verifies the full clean Git checkout and imported package location; mega_encoder constructs the official class with explicit model/norm/pooling/chunk arguments. No operator is reimplemented.

The encoder uses its own sparse/linear input embedding, padding mask, layer stack, optional final normalization and sentence representation. Token mode computes masked mean using src_lengths. Chunking can append padding, so trace alignment must account for the actual returned length. EMA cached kernels/incremental state are not a substitute for training recovery.

Status: source factory only, not a runnable project experiment. A fairseq-compatible isolated environment, common-task head, optimizer/scheduler adaptation, fresh-process complete recovery and real trace replay remain unimplemented/unverified. The current PyTorch Classifier/worker must not silently claim to support MEGA. Source/class import success is not certification.
