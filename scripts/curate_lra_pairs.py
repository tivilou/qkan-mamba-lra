#!/usr/bin/env python3
"""Create a separate, conservative unordered-pair evaluation population; preserve source."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

import numpy as np
import zipfile

from audit_lra_data import ArrayStream, BLOCK_BYTES, sha256_file
from rebuild_mega_lra import save_arrays


def blocks(path):
    with ExitStack() as stack:
        archive=stack.enter_context(zipfile.ZipFile(path))
        arrays={k:ArrayStream(archive,k+'.npy',stack) for k in ['inputs1','inputs2','labels']}
        n=arrays['labels'].shape[0]
        size=max(1,BLOCK_BYTES//sum(a.row_values*8 for a in arrays.values()))
        for start in range(0,n,size):
            take=min(size,n-start)
            yield start,{k:a.read(take) for k,a in arrays.items()}


def identity(a,b,unordered=True):
    sides=[]
    for value in (a,b):
        row=np.asarray(value,dtype='<f8').reshape(-1).copy();row[row==0]=0
        sides.append(hashlib.sha256(str(row.size).encode()+b':'+row.tobytes()).digest())
    return hashlib.sha256(b''.join(sorted(sides) if unordered else sides)).digest()


def curate(source,out):
    source=source.resolve();out=out.resolve()
    if out.exists() or source==out or out.is_relative_to(source):
        raise ValueError('Use a new sibling version; source is never changed')
    expected=json.loads((source/'build-manifest.json').read_text())
    for task in expected['tasks']:
        for split,r in task['splits'].items():
            if sha256_file(source/task['task']/(split+'.npz'))!=r['sha256']:
                raise ValueError('Source build identity changed')
    out.mkdir(parents=True)
    # Copy audited bytes for the four untouched task populations.
    for name in ['listops','text','image','pathfinder']:
        shutil.copytree(source/name,out/name)
    shutil.copytree(source/'metadata',out/'metadata'/'source-v1')
    report={'schema_version':'qkan-pair-curation.v1','profile':'mega-independent-pairclean-v2',
            'status':'curated_not_qualified','source_build_sha256':sha256_file(source/'build-manifest.json'),
            'code_sha256':sha256_file(Path(__file__)),
            'scope':'Conservative population of unambiguous unordered encoded document pairs; not an unchanged official AAN benchmark.',
            'policy':['Exclude every pair group with conflicting labels across any split.',
                      'For remaining pair identities preserve test before validation before train.',
                      'Keep only first source row for duplicate identities within a split.',
                      'No model predictions or performance used; source data and previous outputs retained.'],
            'splits':{},'limitations':['Curating labels is dataset construction, not model selection; formal test remains unused by model fitting.',
                                      'Shared single documents and near duplicates require separate claim boundaries.']}
    with tempfile.TemporaryDirectory(prefix='qkan-pair-curation-') as tmp:
        con=sqlite3.connect(str(Path(tmp)/'pairs.sqlite'))
        try:
            con.execute('CREATE TABLE rows (pair BLOB, ordered BLOB, split TEXT, row_id INTEGER, label INTEGER, priority INTEGER)')
            for priority,split in enumerate(['test','val','train']):
                for start,block in blocks(source/'retrieval'/(split+'.npz')):
                    con.executemany('INSERT INTO rows VALUES(?,?,?,?,?,?)',[(identity(a,b),identity(a,b,False),split,start+i,int(label),priority) for i,(a,b,label) in enumerate(zip(block['inputs1'],block['inputs2'],block['labels']))])
                    con.commit()
            con.execute('CREATE INDEX by_pair ON rows(pair,priority,row_id)')
            con.execute('CREATE INDEX by_ordered ON rows(ordered,label)')
            report['ordered_conflicting_groups']=con.execute('SELECT COUNT(*) FROM (SELECT ordered FROM rows GROUP BY ordered HAVING COUNT(DISTINCT label)>1)').fetchone()[0]
            con.execute('CREATE TABLE conflicts AS SELECT pair FROM rows GROUP BY pair HAVING COUNT(DISTINCT label)>1')
            con.execute('CREATE UNIQUE INDEX conflicts_key ON conflicts(pair)')
            report['unordered_conflicting_groups']=con.execute('SELECT COUNT(*) FROM conflicts').fetchone()[0]
            con.execute('CREATE TABLE keep AS SELECT split,row_id FROM (SELECT r.*,ROW_NUMBER() OVER(PARTITION BY pair ORDER BY priority,row_id) AS rank FROM rows r WHERE pair NOT IN (SELECT pair FROM conflicts)) WHERE rank=1')
            con.commit()
            for split in ['train','val','test']:
                selected=np.asarray([r[0] for r in con.execute('SELECT row_id FROM keep WHERE split=? ORDER BY row_id',(split,))],dtype=np.int64)
                before=con.execute('SELECT COUNT(*) FROM rows WHERE split=?',(split,)).fetchone()[0]
                ambiguous=con.execute('SELECT COUNT(*) FROM rows WHERE split=? AND pair IN (SELECT pair FROM conflicts)',(split,)).fetchone()[0]
                length=next(t['padded_length'] for t in expected['tasks'] if t['task']=='retrieval')
                with tempfile.TemporaryDirectory(prefix='pair-output-',dir=out) as work:
                    arrays={k:np.lib.format.open_memmap(Path(work)/(k+'.npy'),mode='w+',dtype='<u2',shape=(len(selected),length)) for k in ['inputs1','inputs2']}
                    arrays['labels']=np.empty(len(selected),dtype=np.int64)
                    dest=0
                    for start,block in blocks(source/'retrieval'/(split+'.npz')):
                        end=start+len(block['labels'])
                        positions=selected[(selected>=start)&(selected<end)]-start
                        count=len(positions)
                        for key in arrays:
                            arrays[key][dest:dest+count]=block[key][positions]
                        dest+=count
                    if dest!=len(selected):raise ValueError('Incomplete pair output')
                    result=save_arrays(out/'retrieval'/(split+'.npz'),arrays)
                    for value in arrays.values():
                        if isinstance(value,np.memmap):value._mmap.close()
                save_arrays(out/'metadata'/('retrieval-'+split+'-curation-rows.npz'),{'source_v1_row':selected})
                report['splits'][split]={'original_rows':before,'kept_rows':len(selected),'excluded_ambiguous_rows':ambiguous,'excluded_duplicate_or_overlap_rows':before-len(selected)-ambiguous,'output':result}
        finally:con.close()
    build=json.loads(json.dumps(expected))
    build['profile']=report['profile'];build['status']='curated_not_qualified'
    build['parent_build_sha256']=report['source_build_sha256']
    build['curation_code_sha256']=report['code_sha256']
    for task in build['tasks']:
        if task['task']=='retrieval':
            task['splits']={s:r['output'] for s,r in report['splits'].items()}
            task['curation']='pair-curation.json; unambiguous unordered input-pair population'
    (out/'build-manifest.json').write_text(json.dumps(build,indent=2)+'\n')
    (out/'pair-curation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(curate(args.source,args.output)),flush=True)
