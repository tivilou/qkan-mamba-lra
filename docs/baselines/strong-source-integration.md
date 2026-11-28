# Native strong-baseline integration

MEGA now uses the original MegaLRAEncoder with official sparse/real embeddings, masking/mean pooling, LayerNorm and shared pair head. S5 now has a separate native JAX/Flax worker using original StackedEncoderModel/S5SSM, HiPPO eigenbasis, complex bidirectional associative scan and ZOH. No approximate Torch S5 is substituted.

Implementation includes task input/head, full latest-state recovery, validation-best selection, private complete sample-trace.v2 capture/semantic/head replay, isolated environment recipes, direct-source fixed input forward/loss/gradient/update probes and automatic artifact delivery.

The former source-factory-only limitation is closed in code. Runtime import/environment, every-task fresh-process recovery and trace evidence are still pending until the collaborator runs the consolidated entry point. All five committed candidate manifests remain not_verifiable. Conditional local admission requires every check, the reviewed hash-bound policy and successful return before formal training starts; release.json is not globally promoted.

See [consolidated launch](consolidated-launch.md) for the exact handoff and limitations. Author training scores, different author pixel scaling, normalization and recipes are not reproduced or claimed by this common-protocol comparison.
