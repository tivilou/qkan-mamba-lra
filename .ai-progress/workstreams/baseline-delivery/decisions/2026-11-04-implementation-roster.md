# Implementation roster and claim scope

- Date: 2026-11-04
- Scope: baseline-delivery
- Status: active
- Supersedes: None
- Superseded By: None
- Gate: Data qualified for declared internal research; actual model certification and release remain pending
- Authorization: User requested continued release work; no owner model tests or formal training

## Decision

Fix the implementation roster to Transformer, Mamba1, MEGA and S5; minimal official S4D is an auxiliary mechanism control. Preserve unresolved modern candidates and source gaps in cohort-review.json. The roster selects two established task-matched strong reference families with official LRA recipes; it does not claim current global top-two ranking.

## Reason

The direct Mamba backbone and classic attention control alone are insufficient for broad comparative claims. Minimal S4D does not replace a strong S4/S5 reference. MEGA and S5 offer distinct attention/SSM mechanisms and task-matched author artifacts in the bounded existing screen.

## Consequences

Use official native implementations and isolated compatible environments. New factories only verify/import/construct source APIs; native training, recovery, trace and source gate evidence must follow. No formal roster activation or release from selection alone. No global SOTA, quantum advantage or superiority over untested modern neighbors may be inferred.
