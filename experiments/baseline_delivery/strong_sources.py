"""Verified entry points for official MEGA/S5 sources, not training runners.

Keep each source context open for the lifetime of its isolated worker. No
dependency installation, fallback implementation or model execution occurs here.
"""
from contextlib import contextmanager
import importlib
from pathlib import Path
import subprocess
import sys

from .common import ROOT, digest, read


def verify_checkout(model, checkout):
    registry = read(ROOT/'experiments/baseline_delivery/strong-source-registry.json')
    if model not in registry['models']:
        raise ValueError('Unknown strong baseline: '+model)
    record = registry['models'][model]
    checkout = Path(checkout).resolve(strict=True)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(checkout), *args], text=True).strip()
    if Path(git('rev-parse', '--show-toplevel')).resolve() != checkout:
        raise ValueError('Expected the upstream repository root')
    if git('rev-parse', 'HEAD') != record['revision']:
        raise ValueError('Upstream revision differs from the selected source')
    if git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('Upstream checkout must be clean; preserve modifications separately')
    for name, expected in record['files'].items():
        path = checkout/name
        if path.is_symlink() or not path.resolve().is_relative_to(checkout):
            raise ValueError('Upstream source path escapes repository')
        if digest(path) != expected['sha256']:
            raise ValueError('Upstream source hash differs: '+name)
    return checkout, record


@contextmanager
def official_source(model, checkout):
    """Import the actual upstream API in a clean, framework-specific worker."""
    checkout, record = verify_checkout(model, checkout)
    package = record['package']
    for name, module in tuple(sys.modules.items()):
        if name == package or name.startswith(package+'.'):
            filename = getattr(module, '__file__', None)
            if filename is None or not Path(filename).resolve().is_relative_to(checkout):
                raise ValueError('Conflicting upstream package already imported: '+name)
    previous = list(sys.path)
    sys.path.insert(0, str(checkout))
    try:
        if model == 's5':
            # Upstream uses DeviceArray only in evaluated type annotations.
            # JAX >=0.4 exposes Array instead; this alias changes no operation,
            # parameter, initializer, discretization or scan implementation.
            import jax
            import jax.numpy as jnp
            if not hasattr(jnp, 'DeviceArray'):
                jnp.DeviceArray = jax.Array
        modules = {name: importlib.import_module(path) for name, path in record['modules'].items()}
        for module in modules.values():
            if not Path(module.__file__).resolve().is_relative_to(checkout):
                raise ValueError('Imported module did not come from verified checkout')
        yield modules
    finally:
        sys.path[:] = previous
        # Do not unload live upstream classes. A worker must not switch source
        # checkouts; the conflict check above prevents accidental reuse.


def mega_encoder(modules, config):
    """Construct the official encoder from explicit reviewed constructor kwargs."""
    required = {'padding_idx', 'vocab_size', 'embedding_type', 'embedding_dim',
                'num_encoder_layers', 'hidden_dim', 'ffn_hidden_dim', 'z_dim',
                'n_dim', 'norm_type', 'normalize_before', 'chunk_size',
                'max_seq_len', 'sen_rep_type', 'dropout'}
    if not required.issubset(config):
        raise ValueError('Missing explicit MEGA constructor fields: '+','.join(sorted(required-set(config))))
    if config['sen_rep_type'] != 'mp':
        raise ValueError('Project comparison requires declared masked mean pooling')
    return modules['encoder'].MegaLRAEncoder(**config)


def s5_encoder(modules, ssm_config, encoder_config):
    """Use native Flax S5; never replace its scan with a PyTorch approximation."""
    required = {'H', 'P', 'Lambda_re_init', 'Lambda_im_init', 'V', 'Vinv',
                'C_init', 'discretization', 'dt_min', 'dt_max', 'conj_sym',
                'clip_eigs', 'bidirectional'}
    if not required.issubset(ssm_config):
        raise ValueError('S5 initialization/discretization must be explicit')
    if not {'d_model', 'n_layers', 'training', 'batchnorm', 'prenorm', 'dropout', 'activation'}.issubset(encoder_config):
        raise ValueError('S5 layer and normalization choices must be explicit')
    if ssm_config['H'] != encoder_config['d_model']:
        raise ValueError('S5 state-space input width differs from encoder width')
    ssm = modules['ssm'].init_S5SSM(**ssm_config)
    return modules['encoder'].StackedEncoderModel(ssm=ssm, **encoder_config)
