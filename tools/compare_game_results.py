#!/usr/bin/env python3
"""
Compare TypeScript vs C++ game results

Parses output from both and compares territory values.
"""

import json
import subprocess
import sys
from pathlib import Path


def parse_cpp_output(output: str) -> dict:
    """Parse C++ RESULT blocks"""
    results = {}
    lines = output.strip().split('\n')

    current_file = None
    for line in lines:
        if line.startswith('FILE='):
            current_file = line.split('=')[1]
            results[current_file] = {}
        elif current_file and '=' in line:
            if line.startswith('[RESULT]'):
                continue
            key, value = line.split('=', 1)
            try:
                results[current_file][key] = int(value)
            except ValueError:
                results[current_file][key] = value

    return results


def compare_results(ts_json_file: str, cpp_output: str):
    """Compare TypeScript JSON with C++ output"""
    # Load TypeScript results
    with open(ts_json_file) as f:
        ts_results = json.load(f)

    # Parse C++ results
    cpp_results = parse_cpp_output(cpp_output)

    print("=" * 70)
    print("Cross-Language Validation Results")
    print("=" * 70)

    matches = 0
    mismatches = []

    for ts_game in ts_results:
        tgn_file = ts_game['tgnFile']

        if tgn_file not in cpp_results:
            mismatches.append(f"{tgn_file}: Missing in C++ results")
            continue

        cpp_game = cpp_results[tgn_file]

        # Check if there's an error
        if 'error' in ts_game:
            mismatches.append(f"{tgn_file}: TS error: {ts_game['error']}")
            continue

        # Compare move count (excluding pass moves/nulls)
        ts_moves = len([m for m in ts_game['moves'] if m is not None])
        cpp_moves = cpp_game.get('MOVES', -1)

        if ts_moves != cpp_moves:
            mismatches.append(
                f"{tgn_file}: Move count mismatch (TS={ts_moves}, C++={cpp_moves})"
            )
            continue

        # Compare territory
        ts_territory = ts_game['territory']
        cpp_black = cpp_game.get('BLACK', -1)
        cpp_white = cpp_game.get('WHITE', -1)
        cpp_neutral = cpp_game.get('NEUTRAL', -1)

        if (ts_territory['black'] != cpp_black or
            ts_territory['white'] != cpp_white or
            ts_territory['neutral'] != cpp_neutral):
            mismatches.append(
                f"{tgn_file}: Territory mismatch\n"
                f"  TS:  BLACK={ts_territory['black']}, WHITE={ts_territory['white']}, NEUTRAL={ts_territory['neutral']}\n"
                f"  C++: BLACK={cpp_black}, WHITE={cpp_white}, NEUTRAL={cpp_neutral}"
            )
            continue

        # All checks passed
        matches += 1
        print(f"✓ {tgn_file}: Move count={ts_moves}, Territory B={cpp_black} W={cpp_white} N={cpp_neutral}")

    print("\n" + "=" * 70)
    if mismatches:
        print(f"❌ VALIDATION FAILED: {len(mismatches)} mismatches, {matches} matches\n")
        for mismatch in mismatches:
            print(f"  • {mismatch}")
    else:
        print(f"✅ VALIDATION PASSED: All {matches} games match perfectly!")
    print("=" * 70)

    return len(mismatches) == 0


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_game_results.py <ts_json> <cpp_executable>")
        print("")
        print("Example:")
        print("  python3 compare_game_results.py /tmp/tgn_moves.json build/test_game_replay")
        sys.exit(1)

    ts_json_file = sys.argv[1]
    cpp_executable = sys.argv[2]

    if not Path(ts_json_file).exists():
        print(f"Error: TypeScript JSON not found: {ts_json_file}")
        sys.exit(1)

    if not Path(cpp_executable).exists():
        print(f"Error: C++ executable not found: {cpp_executable}")
        sys.exit(1)

    # Run C++ executable
    print(f"Running C++ replay: {cpp_executable} {ts_json_file}\n")
    result = subprocess.run(
        [cpp_executable, ts_json_file],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error running C++ executable:")
        print(result.stderr)
        sys.exit(1)

    # Compare results
    success = compare_results(ts_json_file, result.stdout)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
