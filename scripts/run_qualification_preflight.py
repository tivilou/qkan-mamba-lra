#!/usr/bin/env python3
"""Scheduler entry for focused synthetic tests; no real data or training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("Require an empty scheduler output directory")
    root=Path(__file__).resolve().parents[1]
    results=[]
    for pattern in ("test_data_pipeline.py","test_qualification.py","test_pixel_source.py","test_exchange.py"):
        command=[sys.executable,"-B","-m","unittest","discover","-s","tests","-p",pattern,"-v"]
        result=subprocess.run(command,cwd=root,capture_output=True,text=True)
        (args.output_dir/(pattern+".log")).write_text(result.stdout+result.stderr,encoding="utf-8")
        results.append({"pattern":pattern,"returncode":result.returncode,"argv":command[2:]})
    passed=all(item["returncode"]==0 for item in results)
    (args.output_dir/"preflight.json").write_text(json.dumps({"schema":"qkan-qualification-preflight.v1","passed":passed,"cpu_only":True,"synthetic":True,"tests":results},indent=2)+"\n",encoding="utf-8")
    return 0 if passed else 1


if __name__=="__main__":
    raise SystemExit(main())
