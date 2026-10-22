"""Separate-process entry point; stdout is one machine-readable result."""
import argparse
import json
import sys

from .certification import certify_artifact, load_revision_artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description='Certify unchanged frozen-producer artifacts with mandatory receipt lineage.')
    commands = parser.add_subparsers(dest='operation',required=True)
    certify = commands.add_parser('certify')
    for option in ('source','destination-root','gate-bundle','job-directory','diagnosis'):
        certify.add_argument('--'+option,required=True)
    check = commands.add_parser('check')
    check.add_argument('path')
    check.add_argument('--receipt-sha256',required=True,help='Digest pinned outside the artifact directory')
    args = parser.parse_args(argv)
    try:
        if args.operation == 'certify':
            result = certify_artifact(source=args.source,destination_root=args.destination_root,
                gate_bundle=args.gate_bundle,job_directory=args.job_directory,diagnosis=args.diagnosis)
            output = {'status':'certified','path':str(result.path),'receipt_sha256':result.receipt_sha256}
        else:
            manifest,_,rows,_ = load_revision_artifacts(args.path,receipt_sha256=args.receipt_sha256)
            output = {'status':'passed','run_id':manifest['run_id'],'rows':len(rows),
                      'receipt_sha256':args.receipt_sha256}
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__,'error':str(exc)}),file=sys.stderr)
        return 1
    print(json.dumps(output,sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
