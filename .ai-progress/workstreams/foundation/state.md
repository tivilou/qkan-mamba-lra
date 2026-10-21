# 研究恢复、协作交付与科研门户

## Goal

- Restore code provenance, collaborator delivery and the local research portal; prepare the next data/literature stage without formal training.

## Current State

- GitHub 74acc3d recovered to local branch research/collaboration-foundation; original 3080Ti released, replacement explicitly selected via serverctl development role.
- F01/F02/F03 complete: code provenance restored, isolated HTTPS exchange accepted, six portal workbenches validated. sigmoidpp has accepted write access.
- Historical report preserved; original data/JSON/checkpoint not recovered. No formal model training launched.

## In Progress

- D01: restore and qualify five-task data. Find collaborator backups or legitimate upstream sources before declaring any task qualified.
- Foundation publication: commit reviewed compact sources and sync the collaboration branch; formal experiment package remains pending.

## Next Actions

- 定位合作者备份或合法原始来源，恢复五任务数据并登记来源、划分和哈希

## Blockers

- Historical LRA data and Text JSON/checkpoint are not presently available. Dataset qualification remains insufficient.

## Validation

- Seven local artifact exchange tests pass; six live HTTPS exchange checks pass.
- Six portal routes healthy; 57 Playwright assertions pass at 1280px/390px. Registry, plan and JavaScript syntax checks pass.
- Service runtime: independent supervisor qkan-mamba-exchange; app/storage under /srv/qkan-mamba-exchange/. Secrets remain outside Git and portal.
- Local portal runs at loopback port 4191 via scripts/serve_portal.py; restart command is in research-web/README.md.

## Relevant Decisions

- User-approved collaborator execution workflow; no formal training launched.

## Plan Control

- Plan: plans/research-cycle/v001.json
- Plan Version: 1
- Plan Status: active
- Focus Step: D01
- Focus Status: active
- Completed Steps: F01, F02, F03
- Next Action: 定位合作者备份或合法原始来源，恢复五任务数据并登记来源、划分和哈希
- Blocker: None
- Last Checkpoint: 2026-10-21
