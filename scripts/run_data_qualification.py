#!/usr/bin/env python3
"""Run the bounded pair curation and structural qualification audit; never train."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from audit_lra_data import TASKS, audit, sha256_file
from curate_lra_pairs import curate


def write_json(path: Path, value) -> None:
    pending = path.with_name(path.name + ".pending")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def assessment(curated: Path, diagnostics: dict) -> dict:
    tasks = {record["task"]: record for record in diagnostics["tasks"]}
    complete_scope = set(tasks) == set(TASKS)
    no_structural_failures = complete_scope and all(
        not record["missing"] and not record["failures"] for record in tasks.values()
    )
    status = "insufficient" if no_structural_failures else "revise"
    return {
        "schema_version": "qkan-dataset-qualification.v1",
        "profile": "mega-independent-pairclean-v2",
        "status": status,
        "formal_baseline_allowed": False,
        "release_gate": "stop",
        "scope": list(TASKS),
        "evidence": {
            "build_manifest_sha256": sha256_file(curated / "build-manifest.json"),
            "pair_curation_sha256": sha256_file(curated / "pair-curation.json"),
        },
        "gates": {
            "five_task_files_and_hashes": complete_scope,
            "numeric_schema_and_label_range": no_structural_failures,
            "exact_encoded_cross_split_independence": no_structural_failures,
            "per_task_data_license_and_use_boundary": False,
            "retrieval_shared_document_and_near_duplicate_boundary": False,
            "frozen_transform_learnability_evidence": False,
        },
        "claim_boundary": (
            "This is an adapted, structurally audited five-task population. It is not yet "
            "qualified for formal baseline comparison and is not directly comparable with "
            "historical LRA/MEGA scores."
        ),
        "next_actions": [
            "Resolve per-task data license and permitted redistribution/use boundaries.",
            "Audit Retrieval shared-document and near-duplicate groups with source identities when available.",
            "Freeze transforms and obtain a bounded learnability check without using test outcomes for selection.",
        ],
    }


def run(source: Path, curated: Path, output_dir: Path) -> dict:
    source = source.resolve()
    curated = curated.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("Scheduler output directory must be empty")
    pair_report = curate(source, curated)
    diagnostics = audit(curated)
    result = assessment(curated, diagnostics)
    shutil.copyfile(curated / "build-manifest.json", output_dir / "build-manifest.json")
    shutil.copyfile(curated / "pair-curation.json", output_dir / "pair-curation.json")
    write_json(output_dir / "diagnostics.json", diagnostics)
    write_json(output_dir / "qualification-assessment.json", result)
    return {
        "curation_status": pair_report["status"],
        "diagnostic_status": diagnostics["status"],
        "qualification_status": result["status"],
        "formal_baseline_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--curated", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.curated, args.output_dir)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
