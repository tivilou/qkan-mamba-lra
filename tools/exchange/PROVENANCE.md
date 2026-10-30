# Artifact exchange source

`service/` and `web/` are copied together, without changes, from the canonical
`collaborator-artifact-exchange/templates/` release (`qore-upload-v4`).
The Lucide asset retains its bundled license. Exact file SHA-256 values are
recorded in `source-manifest.json` by the foundation audit.

`scripts/serve_exchange.py` adds TLS without changing the service protocol.
Deployment paths, endpoint, service account, credentials and private keys are
machine-local configuration. Do not expose another project's namespace.
