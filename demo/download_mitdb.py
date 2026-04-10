"""
download_mitdb.py

Download a subset of the MIT-BIH Arrhythmia Database from PhysioNet.

Usage:
    python demo/download_mitdb.py
    python demo/download_mitdb.py --output data/mitdb_raw
    python demo/download_mitdb.py --output data/mitdb_raw --records 100 101 103
"""

import argparse
import os
import sys

# Default 16-record subset covering a good mix of rhythms and arrhythmias
DEFAULT_RECORDS = [
    '100', '101', '103', '105', '106', '108',
    '109', '111', '112', '113', '114', '115',
    '116', '117', '118', '119'
]

# Required file extensions per record
REQUIRED_EXTS = ['.dat', '.hea', '.atr']


def download_records(records, output_dir):
    """Download MIT-BIH records via wfdb."""
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb not installed. Run: pip install wfdb")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading {len(records)} MIT-BIH records to: {output_dir}")
    print(f"Records: {', '.join(records)}\n")

    failed = []
    for rec in records:
        rec_path = os.path.join(output_dir, rec)
        # Skip if already downloaded
        if all(os.path.exists(rec_path + ext) for ext in REQUIRED_EXTS):
            print(f"  [SKIP] {rec} — already downloaded")
            continue
        try:
            wfdb.dl_database('mitdb', dl_dir=output_dir, records=[rec])
            print(f"  [OK]   {rec}")
        except Exception as e:
            print(f"  [FAIL] {rec}: {e}")
            failed.append(rec)

    return failed


def verify_downloads(records, output_dir):
    """Check that all required files exist for each record."""
    print(f"\nVerifying downloads in: {output_dir}")
    ok = []
    missing = []
    for rec in records:
        rec_path = os.path.join(output_dir, rec)
        files_present = [os.path.exists(rec_path + ext) for ext in REQUIRED_EXTS]
        if all(files_present):
            ok.append(rec)
        else:
            absent = [ext for ext, p in zip(REQUIRED_EXTS, files_present) if not p]
            missing.append((rec, absent))
            print(f"  [MISSING] {rec}: {absent}")

    print(f"\nSummary: {len(ok)}/{len(records)} records complete")
    if missing:
        print(f"  {len(missing)} records failed: {[r for r, _ in missing]}")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description='Download MIT-BIH Arrhythmia Database subset from PhysioNet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo/download_mitdb.py
  python demo/download_mitdb.py --output data/mitdb_raw
  python demo/download_mitdb.py --records 100 101 103 106
        """
    )
    parser.add_argument(
        '--output', '-o',
        default='data/mitdb_raw',
        help='Output directory for downloaded records (default: data/mitdb_raw)'
    )
    parser.add_argument(
        '--records', '-r',
        nargs='+',
        default=DEFAULT_RECORDS,
        help='Record numbers to download (default: 16-record subset)'
    )
    args = parser.parse_args()

    # Resolve path relative to project root (one level up from demo/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, args.output)

    failed = download_records(args.records, output_dir)
    remaining = [r for r in args.records if r not in failed]
    ok = verify_downloads(remaining, output_dir)

    if len(ok) == len(args.records):
        print("\nAll records downloaded successfully.")
        print(f"\nNext step:")
        print(f"  python demo/preprocess_mitdb.py --raw-dir {args.output} --output-dir data/mitdb_processed")
    else:
        print(f"\nWarning: {len(args.records) - len(ok)} record(s) could not be downloaded.")
        print("You can still proceed with the available records.")
        sys.exit(1 if len(ok) == 0 else 0)


if __name__ == '__main__':
    main()
