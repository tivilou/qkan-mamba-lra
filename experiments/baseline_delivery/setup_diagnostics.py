"""Allowlisted installation failure facts; never return raw setup output."""
import re
from pathlib import Path

from .common import ROOT, digest


def summarize(path):
    path=Path(path)
    if not path.is_file(): return {'available':False}
    # Bound memory even if pip/compiler emits a very large log.
    with path.open('rb') as handle:
        handle.seek(max(0,path.stat().st_size-256*1024))
        text=handle.read().decode('utf8',errors='replace')
    lower=' '.join(text.lower().replace("'",'').replace('"','').split())
    patterns={
        'ensurepip_missing':('ensurepip is not available','no module named ensurepip'),
        'pip_missing':('no module named pip',),
        'conda_unavailable':('no conda is available',),
        'conda_channel_terms':('condatosnoninteractiveerror','terms of service have not been accepted'),
        'dependency_conflict':('resolutionimpossible','conflicting dependencies'),
        'distribution_unavailable':('no matching distribution found','could not find a version that satisfies'),
        'compiler_failure':('failed building wheel','error compiling objects','command gcc failed'),
        'network_failure':('certificate_verify_failed','connection timed out','could not resolve host','temporary failure in name resolution'),
        'upstream_checkout_mismatch':('upstream checkout must be clean','upstream revision differs','upstream source hash differs'),
    }
    # Only fixed labels, known requirement names, exception class names and
    # numeric exit codes leave the machine. No URLs, paths, argv or log lines.
    known=set()
    for name in ('requirements-mega.txt','requirements-s5.txt'):
        for line in (ROOT/name).read_text(encoding='utf8').splitlines():
            match=re.match(r'([A-Za-z0-9_.-]+)(?:\[.*?\])?==',line)
            if match: known.add(match[1].lower())
    unresolved=[]
    for match in re.finditer(r'(?:No matching distribution found for|satisfies the requirement)\s+([A-Za-z0-9_.-]+)',text,re.I):
        if match[1].lower() in known: unresolved.append(match[1].lower())
    stages=re.findall(r'^QKAN_SETUP_STAGE (bootstrap_python|(?:mega|s5):(?:source|environment|dependencies|build|verify))$',text,re.M)
    return dict(available=True,sha256=digest(path),bytes=path.stat().st_size,
        last_stage=stages[-1] if stages else 'unavailable_in_log_tail',
        signals=[key for key,terms in patterns.items() if any(term in lower for term in terms)],
        unresolved_packages=sorted(set(unresolved)),
        exception_types=sorted(set(re.findall(r'^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):',text,re.M))),
        exit_codes=sorted(set(map(int,re.findall(r'returned non-zero exit status (\d+)',text)))),
        tail_bytes=min(path.stat().st_size,256*1024),privacy='fixed labels and package identities only; no raw logs')
