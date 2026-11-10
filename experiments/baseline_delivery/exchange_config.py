"""Process-local exchange settings from environment or a private local profile.

No network, framework imports, shell evaluation or credential-bearing output.
Run with --configure once; all check_exchange callers then reuse the profile.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import shlex
import stat
import sys
import tempfile
from urllib.parse import urlsplit

KEYS=('PROJECT_EXCHANGE_URL','PROJECT_EXCHANGE_TOKEN','PROJECT_EXCHANGE_CA_FILE')
ROOT=Path(__file__).resolve().parents[2]


def private_directory():
    try: return Path.home()/'qkan-private'
    except RuntimeError: return ROOT/'.bootstrap'


def profile_path():
    override=os.environ.get('QKAN_EXCHANGE_CONFIG')
    if override:
        path=Path(override).expanduser()
        if not path.is_absolute(): raise ValueError('QKAN_EXCHANGE_CONFIG must be an absolute private file path')
        return path
    return private_directory()/'exchange.json'


def validate(values):
    for key in KEYS:
        value=values.get(key)
        if not isinstance(value,str) or not value or any(c in value for c in '\r\n\x00'):
            raise ValueError(f'Missing or invalid {key}; run python -B -m experiments.baseline_delivery.exchange_config --configure once in the executor checkout')
    try: parsed=urlsplit(values[KEYS[0]])
    except ValueError: raise ValueError('Invalid bare HTTPS exchange origin') from None
    if (parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in ('','/') or parsed.query or parsed.fragment or any(c.isspace() for c in values[KEYS[0]])):
        raise ValueError('PROJECT_EXCHANGE_URL must be a bare HTTPS origin')
    try: parsed.port
    except ValueError: raise ValueError('Invalid exchange HTTPS port') from None
    ca=Path(values[KEYS[2]]).expanduser()
    if not ca.is_file(): raise ValueError('PROJECT_EXCHANGE_CA_FILE does not point to a certificate file')
    return {KEYS[0]:values[KEYS[0]].rstrip('/'),KEYS[1]:values[KEYS[1]],KEYS[2]:str(ca.resolve())}


def read_private(path):
    # Parent .bootstrap may be a shared worktree asset; the secret file itself
    # must be an ordinary, owner-only local file on the Linux executor.
    if path.is_symlink(): raise ValueError('Exchange profile must not be a symlink')
    info=path.stat()
    if (not stat.S_ISREG(info.st_mode) or info.st_size>32768
            or (os.name=='posix' and (info.st_uid!=os.getuid() or info.st_mode&0o077))):
        raise ValueError('Exchange profile must be an owner-only regular file (chmod 600)')
    try:
        raw=path.read_text(encoding='utf8')
        if path.suffix=='.json': values=json.loads(raw)
        else:
            values={}
            for line in raw.splitlines():
                line=line.strip()
                if line.startswith('export '): line=line[7:].lstrip()
                key,sep,value=line.partition('=')
                if sep and key.strip() in KEYS:
                    parts=shlex.split(value,comments=True,posix=True)
                    if len(parts)!=1: raise ValueError('Invalid exchange assignment')
                    values[key.strip()]=parts[0]  # Literal only; never source/eval.
        if not isinstance(values,dict): raise ValueError('Invalid exchange object')
        return validate(values)
    except (OSError,UnicodeError,ValueError):
        # Parsing exceptions can contain the private document. Never propagate.
        raise ValueError('Invalid private exchange profile; check its three settings locally') from None


def load():
    inherited={key:os.environ.get(key,'') for key in KEYS}
    if all(inherited.values()):
        result=validate(inherited)
    else:
        paths=[profile_path()]
        if not os.environ.get('QKAN_EXCHANGE_CONFIG'):
            paths.extend([ROOT/'.bootstrap/exchange.json',private_directory()/'exchange.env',ROOT/'.env'])
        found=next((path for path in paths if path.exists() or path.is_symlink()),None)
        if found is None:
            if os.environ.get('QKAN_EXCHANGE_CONFIG'):
                raise ValueError('QKAN_EXCHANGE_CONFIG file is missing; configure that private profile first')
            result=validate(inherited)
        else:
            result=read_private(found)
            # Never send a saved token to a different partly inherited origin.
            for key,value in inherited.items():
                if not value: continue
                same=(value.rstrip('/')==result[key] if key==KEYS[0] else
                    (str(Path(value).expanduser().resolve())==result[key] if key==KEYS[2] else value==result[key]))
                if not same: raise ValueError('Partial exchange environment conflicts with private profile; use one complete configuration')
    os.environ.update(result)
    return result


def save(values):
    values=validate(values);path=profile_path()
    path.parent.mkdir(parents=True,mode=0o700,exist_ok=True)
    if path.exists() or path.is_symlink():
        if read_private(path)!=values: raise ValueError('Existing private exchange profile differs; preserve it and choose an explicit QKAN_EXCHANGE_CONFIG path')
        return path
    fd,temporary=tempfile.mkstemp(prefix='.exchange-',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf8',newline='\n') as handle:
            json.dump(values,handle);handle.write('\n');handle.flush();os.fsync(handle.fileno())
        # No overwrite of a concurrent configuration. File has mkstemp's0600.
        os.link(temporary,path)
    finally: Path(temporary).unlink(missing_ok=True)
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configure',action='store_true',help='Save existing settings or prompt once; token input is hidden')
    args=p.parse_args()
    try:
        if args.configure:
            try: values=load()
            except ValueError:
                if not sys.stdin.isatty():
                    raise ValueError('Private exchange settings are incomplete; run --configure in an interactive terminal') from None
                values={key:os.environ.get(key,'') for key in KEYS}
                for key in KEYS[:2]:
                    if not values[key]:
                        values[key]=(getpass.getpass('项目访问 token（隐藏输入）: ') if key==KEYS[1]
                            else input('文件共享服务 HTTPS 地址: ').strip())
                if not values[KEYS[2]]:
                    default=private_directory()/'exchange-ca.crt'
                    values[KEYS[2]]=str(default) if default.is_file() else input('已有 CA 证书完整路径: ').strip()
            save(values)
            print('私密配置已保存；此账户的新终端和隔离工作目录会自动加载。凭证未写入 Git 或实验报告。')
        else:
            load();print('文件共享配置检查通过。')
    except (ValueError,OSError,EOFError) as error:
        # Generic I/O exceptions can include paths; print only controlled values.
        p.exit(2,(str(error) if type(error) is ValueError else 'Private exchange configuration could not be read or saved')+'\n')


if __name__=='__main__': main()
