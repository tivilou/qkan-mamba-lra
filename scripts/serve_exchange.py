#!/usr/bin/env python3
"""TLS adapter around the pinned, unmodified artifact exchange implementation."""
import argparse
import importlib.util
from pathlib import Path
import ssl
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--tls-cert", required=True)
    parser.add_argument("--tls-key", required=True)
    args, remaining = parser.parse_known_args()
    spec = importlib.util.spec_from_file_location("exchange_service", ROOT / "tools/exchange/service/qore_upload_server.py")
    service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(args.tls_cert, args.tls_key)
    original = service.UploadServer

    class TLSServer(original):
        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(120)
            try:
                return context.wrap_socket(connection, server_side=True), address
            except Exception:
                connection.close()
                raise

    service.UploadServer = TLSServer
    sys.argv = [sys.argv[0], *remaining]
    service.main()


if __name__ == "__main__":
    main()
