#!/usr/bin/env python3
"""Validate the portable sample-trace.v1/v2 Case Study contracts."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


STAGES = [
    "data",
    "preprocess",
    "training",
    "retrieval",
    "scoring",
    "selection",
    "context",
    "generation",
    "evaluation",
    "diagnosis",
]
V2_LANES = {"shared", "training", "evaluation", "replay"}
STATUSES = {"observed", "not_applicable", "unavailable", "failed"}
TARGET_LIKE = {
    "answer", "answers", "gold", "gold_answer", "gold_answers", "target",
    "target_answer", "reference_answer", "reference_answers", "label",
    "labels", "ground_truth", "groundtruth", "expected_answer",
}
SHA256 = re.compile(r"^[A-Fa-f0-9]{64}$")


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _find_target_like(value: Any, path: str = "inputs") -> str | None:
    """Return the first target-like input key, including nested objects."""
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if normalized in TARGET_LIKE or any(token in normalized.split("_") for token in TARGET_LIKE):
                return f"{path}.{key}"
            found = _find_target_like(child, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _find_target_like(child, f"{path}[{index}]")
            if found:
                return found
    return None


def _artifact_refs(value: Any, prefix: str, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        errors.append(f"{prefix} must be a list")
        return
    for index, artifact in enumerate(value):
        item_prefix = f"{prefix}[{index}]"
        if not isinstance(artifact, dict):
            errors.append(f"{item_prefix} must be an object")
            continue
        if not _text(artifact.get("path")):
            errors.append(f"{item_prefix}.path is required")
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            errors.append(f"{item_prefix}.sha256 must be a SHA-256 hex digest")
        if "bytes" in artifact and (type(artifact["bytes"]) is not int or artifact["bytes"] < 0):
            errors.append(f"{item_prefix}.bytes must be a nonnegative integer")
        if "visibility" in artifact and artifact["visibility"] not in {"portal", "exchange_only", "private"}:
            errors.append(f"{item_prefix}.visibility is invalid")


def _help_text(value: Any, prefix: str, errors: list[str]) -> None:
    if not _text(value):
        errors.append(f"{prefix} must be non-empty help text")


def _display(value: Any, prefix: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{prefix} must be an object")
        return
    if not _text(value.get("summary")):
        errors.append(f"{prefix}.summary is required")
    if value.get("default_view") not in {"human_summary", "visual", "table", "text", "not_available"}:
        errors.append(f"{prefix}.default_view is invalid")
    fields = value.get("visible_fields")
    if not isinstance(fields, list):
        errors.append(f"{prefix}.visible_fields must be a list")
    else:
        seen: set[str] = set()
        for index, field in enumerate(fields):
            field_prefix = f"{prefix}.visible_fields[{index}]"
            if not isinstance(field, dict):
                errors.append(f"{field_prefix} must be an object")
                continue
            field_id = field.get("id")
            if not _text(field_id):
                errors.append(f"{field_prefix}.id is required")
            elif field_id in seen:
                errors.append(f"{prefix}.visible_fields contains duplicate id {field_id}")
            else:
                seen.add(field_id)
            if not _text(field.get("label")):
                errors.append(f"{field_prefix}.label is required")
            _help_text(field.get("help"), f"{field_prefix}.help", errors)
    visualizations = value.get("visualizations")
    if not isinstance(visualizations, list):
        errors.append(f"{prefix}.visualizations must be a list")
    else:
        seen: set[str] = set()
        for index, visualization in enumerate(visualizations):
            viz_prefix = f"{prefix}.visualizations[{index}]"
            if not isinstance(visualization, dict):
                errors.append(f"{viz_prefix} must be an object")
                continue
            viz_id = visualization.get("id")
            if not _text(viz_id):
                errors.append(f"{viz_prefix}.id is required")
            elif viz_id in seen:
                errors.append(f"{prefix}.visualizations contains duplicate id {viz_id}")
            else:
                seen.add(viz_id)
            for field in ("kind", "title"):
                if not _text(visualization.get(field)):
                    errors.append(f"{viz_prefix}.{field} is required")
            _help_text(visualization.get("help"), f"{viz_prefix}.help", errors)
            _artifact_refs(visualization.get("artifact_refs"), f"{viz_prefix}.artifact_refs", errors)


def _validate_v2(payload: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    """Validate v2 structure without imposing a domain-specific pipeline."""
    for field in ("trace_id", "experiment", "sample_selection", "coverage", "samples"):
        if field not in payload:
            errors.append(f"missing top-level field: {field}")
    if payload.get("trace_completeness") not in {None, "complete", "partial", "insufficient"}:
        errors.append("trace_completeness is invalid")
    experiment = payload.get("experiment")
    if not isinstance(experiment, dict):
        errors.append("experiment is required")
    else:
        for field in ("run_id", "dataset", "code_revision", "config_sha256"):
            if not _text(experiment.get(field)):
                errors.append(f"experiment.{field} is required")
        if not _text(experiment.get("model_identity")) and not isinstance(experiment.get("model_identity"), dict):
            errors.append("experiment.model_identity is required")
        if _text(experiment.get("config_sha256")) and not SHA256.fullmatch(experiment["config_sha256"]):
            errors.append("experiment.config_sha256 must be a SHA-256 hex digest")
    selection = payload.get("sample_selection")
    if not isinstance(selection, dict):
        errors.append("sample_selection is required")
        selection = {}
    for field in ("rule", "population_scope"):
        if not _text(selection.get(field)):
            errors.append(f"sample_selection.{field} is required")
    selected_count = selection.get("selected_count")
    if type(selected_count) is not int or selected_count < 1:
        errors.append("sample_selection.selected_count must be a positive integer")
    selected_ids = selection.get("selected_sample_ids")
    if selected_ids is not None and (
        not isinstance(selected_ids, list)
        or not selected_ids
        or not all(_text(item) for item in selected_ids)
        or len(selected_ids) != len(set(selected_ids))
    ):
        errors.append("sample_selection.selected_sample_ids must be a unique non-empty string list")
    coverage = payload.get("coverage")
    if not isinstance(coverage, dict):
        errors.append("coverage is required")
        coverage = {}
    for stage in STAGES:
        if coverage.get(stage) not in STATUSES:
            errors.append(f"coverage.{stage} must be one of {sorted(STATUSES)}")

    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append("samples must be a non-empty list")
        return
    sample_ids: list[str] = []
    for sample_index, sample in enumerate(samples):
        prefix = f"samples[{sample_index}]"
        if not isinstance(sample, dict):
            errors.append(f"{prefix} must be an object")
            continue
        sample_id = sample.get("sample_id")
        if not _text(sample_id):
            errors.append(f"{prefix}.sample_id is required")
        else:
            sample_ids.append(sample_id)
        if sample.get("lane") not in V2_LANES:
            errors.append(f"{prefix}.lane must be one of {sorted(V2_LANES)}")
        stages = sample.get("stages")
        if not isinstance(stages, list) or not stages:
            errors.append(f"{prefix}.stages must be a non-empty list")
            continue
        stage_ids: set[str] = set()
        produced: set[str] = set()
        previous_step: int | None = None
        observed_coverage: set[str] = set()
        for stage_index, record in enumerate(stages):
            stage_prefix = f"{prefix}.stages[{stage_index}]"
            if not isinstance(record, dict):
                errors.append(f"{stage_prefix} must be an object")
                continue
            stage_id = record.get("stage_id")
            if not _text(stage_id):
                errors.append(f"{stage_prefix}.stage_id is required")
            elif stage_id in stage_ids:
                errors.append(f"{prefix}.stages contains duplicate stage_id {stage_id}")
            else:
                stage_ids.add(stage_id)
            if not _text(record.get("stage")):
                errors.append(f"{stage_prefix}.stage is required")
            canonical_stage = record.get("canonical_stage")
            if canonical_stage is not None and canonical_stage not in STAGES:
                errors.append(f"{stage_prefix}.canonical_stage is invalid")
            lane = record.get("lane")
            if lane not in V2_LANES:
                errors.append(f"{stage_prefix}.lane is invalid")
            status = record.get("status")
            if status not in STATUSES:
                errors.append(f"{stage_prefix}.status is invalid")
            if not _text(record.get("purpose")):
                errors.append(f"{stage_prefix}.purpose is required")
            step = record.get("step")
            if isinstance(step, bool) or (step is not None and type(step) is not int):
                errors.append(f"{stage_prefix}.step must be an integer when present")
            if type(step) is int:
                if previous_step is not None and step <= previous_step:
                    errors.append(f"{prefix}.stages step values must increase strictly")
                previous_step = step
            for field in ("input_refs", "output_refs"):
                refs = record.get(field)
                if not isinstance(refs, list) or not all(_text(ref) for ref in refs) or len(refs) != len(set(refs)):
                    errors.append(f"{stage_prefix}.{field} must be a unique string list")
                elif field == "input_refs":
                    for ref in refs:
                        if not (ref.startswith(("source:", "external:", "artifact:")) or ref in produced):
                            errors.append(f"{stage_prefix}.input_refs references an unknown prior output {ref}")
                else:
                    for ref in refs:
                        if ref in produced:
                            errors.append(f"{stage_prefix}.output_refs redefines {ref}")
                    produced.update(refs)
            _artifact_refs(record.get("artifact_refs"), f"{stage_prefix}.artifact_refs", errors)
            _display(record.get("display"), f"{stage_prefix}.display", errors)
            if status == "observed":
                if not record.get("input_refs") and not record.get("output_refs") and not record.get("inputs") and not record.get("outputs"):
                    errors.append(f"{stage_prefix} observed stage needs input/output evidence")
            elif not _text(record.get("summary")) and not _text(record.get("reason")):
                warnings.append(f"{stage_prefix} status={status} should include summary or reason")
            if canonical_stage in STAGES and status == "observed":
                observed_coverage.add(canonical_stage)
        for stage in STAGES:
            if coverage.get(stage) == "observed" and stage not in observed_coverage:
                warnings.append(f"{prefix} has no observed stage mapped to coverage.{stage}")
        representations = sample.get("representations", [])
        if not isinstance(representations, list):
            errors.append(f"{prefix}.representations must be a list")
        else:
            representation_ids: set[str] = set()
            for rep_index, representation in enumerate(representations):
                rep_prefix = f"{prefix}.representations[{rep_index}]"
                if not isinstance(representation, dict):
                    errors.append(f"{rep_prefix} must be an object")
                    continue
                rep_id = representation.get("id")
                if not _text(rep_id) or rep_id in representation_ids:
                    errors.append(f"{rep_prefix}.id must be unique and non-empty")
                else:
                    representation_ids.add(rep_id)
                if representation.get("stage_id") not in stage_ids:
                    errors.append(f"{rep_prefix}.stage_id must reference a sample stage")
                if representation.get("status") not in STATUSES:
                    errors.append(f"{rep_prefix}.status is invalid")
                if not _text(representation.get("meaning")):
                    errors.append(f"{rep_prefix}.meaning is required")
                _artifact_refs(representation.get("artifact_refs"), f"{rep_prefix}.artifact_refs", errors)
                _display(representation.get("display"), f"{rep_prefix}.display", errors)
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("sample IDs must be unique")
    if type(selected_count) is int and selected_count != len(samples):
        errors.append("sample_selection.selected_count must match samples length")
    if selected_ids is not None and sample_ids != selected_ids:
        errors.append("sample_selection.selected_sample_ids must match samples exactly and in order")


def _stage(record: Any, prefix: str, errors: list[str], warnings: list[str]) -> str | None:
    if not isinstance(record, dict):
        errors.append(f"{prefix} must be an object")
        return None
    name = record.get("stage")
    if name not in STAGES:
        errors.append(f"{prefix}.stage is invalid")
        return None
    status = record.get("status")
    if status not in STATUSES:
        errors.append(f"{prefix}.status is invalid")
        return name
    _artifact_refs(record.get("artifact_refs"), f"{prefix}.artifact_refs", errors)
    if status == "observed":
        payload_fields = ("inputs", "outputs", "observed_fields", "summary", "artifact_refs")
        if not any(field in record and bool(record.get(field)) for field in payload_fields):
            errors.append(f"{prefix} observed stage needs inputs, outputs, observed_fields, summary, or artifact_refs")
        if "observed_fields" in record and (not isinstance(record["observed_fields"], dict) or not record["observed_fields"]):
            errors.append(f"{prefix}.observed_fields must be a non-empty object")
    elif not _text(record.get("summary")):
        warnings.append(f"{prefix} status={status} should include a summary explaining the missing or failed stage")
    if name == "generation" and status == "observed":
        if record.get("target_access") not in {"outside_generation_input", "post_generation_only", "training_only", "not_available"}:
            errors.append(f"{prefix}.target_access is required for an observed generation stage")
        found = _find_target_like(record.get("inputs", {}))
        if found:
            errors.append(f"{prefix} generation inputs contain a target-like field: {found}")
    return name


def _has_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _validate_value_capture(representation: dict[str, Any], prefix: str, errors: list[str]) -> None:
    """Check the declared full-value capture and coordinate-reader contract."""
    capture = representation.get("value_capture")
    if not isinstance(capture, dict):
        errors.append(f"{prefix}.value_capture is required for a required numeric representation")
        return
    if capture.get("status") != "complete":
        errors.append(f"{prefix}.value_capture.status must be complete")
    shape = capture.get("shape")
    if (
        not isinstance(shape, list)
        or not all(type(dimension) is int and dimension >= 0 for dimension in shape)
    ):
        errors.append(f"{prefix}.value_capture.shape must be a list of nonnegative integers")
        expected_count = None
    else:
        expected_count = math.prod(shape) if shape else 1
        if representation.get("shape") != shape:
            errors.append(f"{prefix}.value_capture.shape must match representation.shape")
    if not _text(representation.get("dtype")):
        errors.append(f"{prefix}.dtype is required for a required numeric representation")
    logical_count = capture.get("logical_element_count")
    stored_count = capture.get("stored_element_count")
    if type(logical_count) is not int or logical_count < 0:
        errors.append(f"{prefix}.value_capture.logical_element_count must be nonnegative")
    if type(stored_count) is not int or stored_count < 0:
        errors.append(f"{prefix}.value_capture.stored_element_count must be nonnegative")
    if expected_count is not None:
        if logical_count != expected_count:
            errors.append(f"{prefix}.value_capture.logical_element_count must match shape")
        if stored_count != expected_count:
            errors.append(f"{prefix}.value_capture.stored_element_count must cover the full shape")
    artifacts = capture.get("artifact_refs")
    _artifact_refs(artifacts, f"{prefix}.value_capture.artifact_refs", errors)
    if not isinstance(artifacts, list) or not artifacts:
        errors.append(f"{prefix}.value_capture.artifact_refs must contain full-value artifacts")
        artifact_hashes: set[str] = set()
    else:
        artifact_hashes = {
            item.get("sha256").lower() for item in artifacts
            if isinstance(item, dict) and isinstance(item.get("sha256"), str)
        }
    projection = capture.get("reader_projection")
    if not isinstance(projection, dict):
        errors.append(f"{prefix}.value_capture.reader_projection is required")
        return
    if projection.get("status") != "available":
        errors.append(f"{prefix}.value_capture.reader_projection.status must be available")
    if projection.get("kind") not in {"coordinate_inspector", "axis_aware_view", "domain_specific"}:
        errors.append(f"{prefix}.value_capture.reader_projection.kind is invalid")
    if projection.get("coordinate_access") not in {"arbitrary", "bounded_tiles"}:
        errors.append(f"{prefix}.value_capture.reader_projection must expose coordinate access beyond a fixed preview")
    if not _text(projection.get("reader_ref")):
        errors.append(f"{prefix}.value_capture.reader_projection.reader_ref is required")
    source_hashes = projection.get("source_sha256")
    if (
        not isinstance(source_hashes, list)
        or not source_hashes
        or not all(isinstance(item, str) and SHA256.fullmatch(item) for item in source_hashes)
        or len(source_hashes) != len({item.lower() for item in source_hashes})
    ):
        errors.append(f"{prefix}.value_capture.reader_projection.source_sha256 must be a unique digest list")
    elif {item.lower() for item in source_hashes} != artifact_hashes:
        errors.append(f"{prefix}.value_capture.reader_projection source hashes must match full-value artifacts")


def _validate_semantic_contract(payload: dict[str, Any], errors: list[str]) -> None:
    """Validate the project-owned semantic readiness extension."""
    contract = payload.get("task_contract")
    if not isinstance(contract, dict):
        errors.append("--require-complete requires task_contract")
        return
    if not _text(contract.get("task_type")):
        errors.append("task_contract.task_type is required")
    required_roles = contract.get("required_roles")
    if (
        not isinstance(required_roles, list)
        or not required_roles
        or not all(_text(role) for role in required_roles)
        or len(required_roles) != len(set(required_roles))
    ):
        errors.append("task_contract.required_roles must be a unique non-empty string list")
        required_roles = []
    required_representations = contract.get("required_representations")
    if (
        not isinstance(required_representations, list)
        or not all(_text(item) for item in required_representations)
        or len(required_representations) != len(set(required_representations))
    ):
        errors.append("task_contract.required_representations must be a unique string list")
        required_representations = []
    required_splits = contract.get("required_splits")
    if (
        not isinstance(required_splits, list)
        or not required_splits
        or not all(_text(item) for item in required_splits)
        or len(required_splits) != len(set(required_splits))
    ):
        errors.append("task_contract.required_splits must be a unique non-empty string list")
        required_splits = []
    required_checkpoints = contract.get("required_checkpoints")
    if (
        not isinstance(required_checkpoints, list)
        or not required_checkpoints
        or not all(_text(item) for item in required_checkpoints)
        or len(required_checkpoints) != len(set(required_checkpoints))
    ):
        errors.append("task_contract.required_checkpoints must be a unique non-empty string list")
        required_checkpoints = []
    if payload.get("trace_completeness") != "complete":
        errors.append("--require-complete requires trace_completeness=complete")

    samples = payload.get("samples")
    observed_splits: set[str] = set()
    observed_checkpoints: set[str] = set()
    if not isinstance(samples, list):
        return
    for index, sample in enumerate(samples):
        prefix = f"samples[{index}]"
        if not isinstance(sample, dict):
            continue
        split = sample.get("split")
        if not _text(split):
            errors.append(f"{prefix}.split is required by --require-complete")
        else:
            observed_splits.add(split)
        checkpoint = sample.get("checkpoint")
        if not _text(checkpoint):
            errors.append(f"{prefix}.checkpoint is required by --require-complete")
        else:
            observed_checkpoints.add(checkpoint)
        roles = sample.get("roles")
        if not isinstance(roles, dict):
            errors.append(f"{prefix}.roles is required by --require-complete")
        else:
            for role in required_roles:
                if role not in roles or not _has_value(roles[role]):
                    errors.append(f"{prefix}.roles.{role} is required")
        representations = sample.get("representations")
        if not isinstance(representations, list):
            errors.append(f"{prefix}.representations is required by --require-complete")
            continue
        by_id: dict[str, dict[str, Any]] = {}
        for rep_index, representation in enumerate(representations):
            rep_prefix = f"{prefix}.representations[{rep_index}]"
            if not isinstance(representation, dict):
                errors.append(f"{rep_prefix} must be an object")
                continue
            rep_id = representation.get("id")
            if not _text(rep_id):
                errors.append(f"{rep_prefix}.id is required")
                continue
            if rep_id in by_id:
                errors.append(f"{prefix}.representations contains duplicate id {rep_id}")
            by_id[rep_id] = representation
            if representation.get("stage") not in STAGES:
                errors.append(f"{rep_prefix}.stage is invalid")
            if representation.get("status") not in STATUSES:
                errors.append(f"{rep_prefix}.status is invalid")
            _artifact_refs(representation.get("artifact_refs"), f"{rep_prefix}.artifact_refs", errors)
        for representation_id in required_representations:
            representation = by_id.get(representation_id)
            if representation is None:
                errors.append(f"{prefix}.representations is missing required id {representation_id}")
                continue
            if representation.get("status") != "observed":
                errors.append(f"{prefix}.representations.{representation_id} must be observed")
            if not any(
                _has_value(representation.get(field))
                for field in ("summary", "shape", "dtype", "artifact_refs")
            ):
                errors.append(f"{prefix}.representations.{representation_id} needs summary, shape, dtype, or artifact_refs")
    missing_splits = sorted(set(required_splits) - observed_splits)
    for split in missing_splits:
        errors.append(f"samples is missing required split {split}")
    missing_checkpoints = sorted(set(required_checkpoints) - observed_checkpoints)
    for checkpoint in missing_checkpoints:
        errors.append(f"samples is missing required checkpoint {checkpoint}")


def _validate_v2_completeness(payload: dict[str, Any], errors: list[str]) -> None:
    """Apply the opt-in semantic readiness gate to a v2 trace."""
    contract = payload.get("task_contract")
    if not isinstance(contract, dict):
        errors.append("--require-complete requires task_contract")
        return
    required_roles = contract.get("required_roles")
    required_representations = contract.get("required_representations")
    required_value_representations = contract.get("required_value_representations")
    required_splits = contract.get("required_splits")
    required_checkpoints = contract.get("required_checkpoints")
    for field, value in (
        ("required_roles", required_roles),
        ("required_representations", required_representations),
        ("required_value_representations", required_value_representations),
        ("required_splits", required_splits),
        ("required_checkpoints", required_checkpoints),
    ):
        if not isinstance(value, list) or not all(_text(item) for item in value) or len(value) != len(set(value)):
            errors.append(f"task_contract.{field} must be a unique string list")
    valid_required_representations = (
        required_representations
        if isinstance(required_representations, list) and all(_text(item) for item in required_representations)
        else []
    )
    valid_required_value_representations = (
        required_value_representations
        if isinstance(required_value_representations, list) and all(_text(item) for item in required_value_representations)
        else []
    )
    if valid_required_value_representations:
        required_representation_set = set(valid_required_representations)
        for representation_id in valid_required_value_representations:
            if representation_id not in required_representation_set:
                errors.append(f"task_contract.required_value_representations.{representation_id} must also be required_representations")
    if not _text(contract.get("task_type")):
        errors.append("task_contract.task_type is required")
    samples = payload.get("samples")
    observed_splits: set[str] = set()
    observed_checkpoints: set[str] = set()
    if not isinstance(samples, list):
        return
    for index, sample in enumerate(samples):
        prefix = f"samples[{index}]"
        if not isinstance(sample, dict):
            continue
        if _text(sample.get("split")):
            observed_splits.add(sample["split"])
        if _text(sample.get("checkpoint")):
            observed_checkpoints.add(sample["checkpoint"])
        roles = sample.get("roles")
        if not isinstance(roles, dict):
            errors.append(f"{prefix}.roles is required by --require-complete")
        else:
            for role in required_roles if isinstance(required_roles, list) else []:
                if role not in roles or not _has_value(roles[role]):
                    errors.append(f"{prefix}.roles.{role} is required")
        representations = sample.get("representations")
        by_id = {
            item.get("id"): item
            for item in representations
            if isinstance(item, dict) and _text(item.get("id"))
        } if isinstance(representations, list) else {}
        if not isinstance(representations, list):
            errors.append(f"{prefix}.representations is required by --require-complete")
        for representation_id in valid_required_representations:
            representation = by_id.get(representation_id)
            if representation is None:
                errors.append(f"{prefix}.representations is missing required id {representation_id}")
            elif representation.get("status") != "observed":
                errors.append(f"{prefix}.representations.{representation_id} must be observed")
        for representation_id in valid_required_value_representations:
            representation = by_id.get(representation_id)
            if representation is not None:
                _validate_value_capture(
                    representation,
                    f"{prefix}.representations.{representation_id}",
                    errors,
                )
    if isinstance(required_splits, list):
        for split in sorted(set(required_splits) - observed_splits):
            errors.append(f"samples is missing required split {split}")
    if isinstance(required_checkpoints, list):
        for checkpoint in sorted(set(required_checkpoints) - observed_checkpoints):
            errors.append(f"samples is missing required checkpoint {checkpoint}")


def validate(payload: dict[str, Any], *, require_complete: bool = False) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, dict):
        return ["trace must be a JSON object"], warnings
    schema_version = payload.get("schema_version")
    if schema_version == "sample-trace.v2":
        _validate_v2(payload, errors, warnings)
        if require_complete:
            if payload.get("trace_completeness") != "complete":
                errors.append("--require-complete requires trace_completeness=complete")
            _validate_v2_completeness(payload, errors)
        return errors, warnings
    if schema_version != "sample-trace.v1":
        errors.append("schema_version must be sample-trace.v1 or sample-trace.v2")
    experiment = payload.get("experiment")
    if not isinstance(experiment, dict):
        errors.append("experiment is required")
    else:
        for field in ("run_id", "dataset", "code_revision", "config_sha256"):
            if not _text(experiment.get(field)):
                errors.append(f"experiment.{field} is required")
        if not _text(experiment.get("model_identity")) and not isinstance(experiment.get("model_identity"), dict):
            errors.append("experiment.model_identity is required")
        if _text(experiment.get("config_sha256")) and not SHA256.fullmatch(experiment["config_sha256"]):
            errors.append("experiment.config_sha256 must be a SHA-256 hex digest")
    selection = payload.get("sample_selection")
    if not isinstance(selection, dict):
        errors.append("sample_selection is required")
        selection = {}
    for field in ("rule", "population_scope"):
        if not _text(selection.get(field)):
            errors.append(f"sample_selection.{field} is required")
    selected_count = selection.get("selected_count")
    if type(selected_count) is not int or selected_count < 1:
        errors.append("sample_selection.selected_count must be a positive integer")
    selected_ids = selection.get("selected_sample_ids")
    if selected_ids is not None:
        if not isinstance(selected_ids, list) or not selected_ids or not all(_text(item) for item in selected_ids) or len(selected_ids) != len(set(selected_ids)):
            errors.append("sample_selection.selected_sample_ids must be a unique non-empty string list")
    coverage = payload.get("coverage")
    if not isinstance(coverage, dict):
        errors.append("coverage is required")
        coverage = {}
    for stage in STAGES:
        if coverage.get(stage) not in STATUSES:
            errors.append(f"coverage.{stage} must be one of {sorted(STATUSES)}")

    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append("samples must be a non-empty list")
        samples = []
    sample_ids: list[str] = []
    for index, sample in enumerate(samples):
        prefix = f"samples[{index}]"
        if not isinstance(sample, dict):
            errors.append(f"{prefix} must be an object")
            continue
        sample_id = sample.get("sample_id")
        if not _text(sample_id):
            errors.append(f"{prefix}.sample_id is required")
        else:
            sample_ids.append(sample_id)
        records = sample.get("stages")
        if not isinstance(records, list) or not records:
            errors.append(f"{prefix}.stages must be a non-empty list")
            continue
        seen: set[str] = set()
        positions: list[int] = []
        for stage_index, record in enumerate(records):
            name = _stage(record, f"{prefix}.stages[{stage_index}]", errors, warnings)
            if name is None:
                continue
            if name in seen:
                errors.append(f"{prefix}.stages contains duplicate stage {name}")
            seen.add(name)
            positions.append(STAGES.index(name))
        if positions != sorted(positions):
            errors.append(f"{prefix}.stages must follow canonical order: {' -> '.join(STAGES)}")
        for stage in STAGES:
            if coverage.get(stage) == "observed" and stage not in seen:
                errors.append(f"{prefix}.stages is missing observed coverage stage {stage}")
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("sample IDs must be unique")
    if type(selected_count) is int and selected_count != len(samples):
        errors.append("sample_selection.selected_count must match samples length")
    if selected_ids is not None and sample_ids != selected_ids:
        errors.append("sample_selection.selected_sample_ids must match samples exactly and in order")
    if require_complete:
        _validate_semantic_contract(payload, errors)
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require the project-owned semantic trace contract and complete readiness",
    )
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.trace.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    errors, warnings = validate(payload, require_complete=args.require_complete)
    if args.strict:
        errors.extend(f"strict: {warning}" for warning in warnings)
    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print("OK: sample trace validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
