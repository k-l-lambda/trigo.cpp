#!/usr/bin/env python3
"""
Cross-Language Game Equivalence Validation

Generates random games using TypeScript, replays them in both TS and C++,
and compares:
1. Final board state
2. Territory calculation results

This validates that the C++ port behaves identically to the TypeScript implementation.
"""

import subprocess
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


# Paths
SCRIPT_DIR = Path(__file__).parent
TRIGO_CPP_DIR = SCRIPT_DIR.parent
TRIGO_WEB_DIR = TRIGO_CPP_DIR.parent / "trigoRL" / "third_party" / "trigo" / "trigo-web"
BUILD_DIR = TRIGO_CPP_DIR / "build"


def generate_random_games(count: int, board_shape: str = "5*5*1", output_dir: Path = None) -> List[Path]:
    """Generate random TGN games using TypeScript tool"""
    print(f"\n[1/4] Generating {count} random games...")

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="trigo_validation_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Run TypeScript game generator
    cmd = [
        "npm", "run", "generate:games", "--",
        "--count", str(count),
        "--board", board_shape,
        "--moves", "10-30",  # Use specific move range for faster generation
        "--output", str(output_dir)
    ]

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=TRIGO_WEB_DIR,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"  Error generating games: {result.stderr}")
        sys.exit(1)

    # Get generated TGN files
    tgn_files = list(output_dir.glob("*.tgn"))
    print(f"  ✓ Generated {len(tgn_files)} games in {output_dir}")

    return tgn_files


def run_typescript_validation(tgn_files: List[Path]) -> Dict[str, Any]:
    """Run TypeScript validation script"""
    print(f"\n[2/4] Running TypeScript validation...")

    # Compile TypeScript validation script
    ts_script = SCRIPT_DIR / "validate_game_equivalence.ts"
    js_script = SCRIPT_DIR / "validate_game_equivalence.js"

    # Compile with tsc
    compile_cmd = [
        "npx", "tsc",
        "--module", "esnext",
        "--target", "es2020",
        "--moduleResolution", "node",
        "--esModuleInterop",
        str(ts_script)
    ]

    result = subprocess.run(
        compile_cmd,
        cwd=TRIGO_WEB_DIR,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"  Error compiling TS: {result.stderr}")
        sys.exit(1)

    # Run validation script
    cmd = ["node", str(js_script)] + [str(f) for f in tgn_files]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=TRIGO_WEB_DIR
    )

    if result.returncode != 0:
        print(f"  Error running TS validation: {result.stderr}")
        sys.exit(1)

    # Parse JSON output
    try:
        ts_results = json.loads(result.stdout)
        print(f"  ✓ Processed {len(ts_results)} games in TypeScript")
        return {r["tgnFile"]: r for r in ts_results}
    except json.JSONDecodeError as e:
        print(f"  Error parsing TS output: {e}")
        print(f"  Output: {result.stdout[:500]}")
        sys.exit(1)


def run_cpp_validation(tgn_files: List[Path]) -> Dict[str, Any]:
    """Run C++ validation (to be implemented)"""
    print(f"\n[3/4] Running C++ validation...")
    print("  [TODO] C++ validation not yet implemented")

    # Placeholder: return dummy data
    cpp_results = {}
    for tgn_file in tgn_files:
        cpp_results[tgn_file.name] = {
            "tgnFile": tgn_file.name,
            "error": "C++ implementation pending"
        }

    return cpp_results


def compare_results(ts_results: Dict[str, Any], cpp_results: Dict[str, Any]) -> None:
    """Compare TypeScript and C++ results"""
    print(f"\n[4/4] Comparing results...")

    mismatches = []
    matches = 0

    for tgn_file, ts_result in ts_results.items():
        if tgn_file not in cpp_results:
            mismatches.append(f"{tgn_file}: Missing in C++ results")
            continue

        cpp_result = cpp_results[tgn_file]

        # Check for errors
        if "error" in ts_result:
            mismatches.append(f"{tgn_file}: TS error: {ts_result['error']}")
            continue

        if "error" in cpp_result:
            mismatches.append(f"{tgn_file}: C++ error: {cpp_result['error']}")
            continue

        # Compare board shapes
        if ts_result["boardShape"] != cpp_result.get("boardShape"):
            mismatches.append(f"{tgn_file}: Board shape mismatch")
            continue

        # Compare move counts
        if ts_result["moveCount"] != cpp_result.get("moveCount"):
            mismatches.append(f"{tgn_file}: Move count mismatch")
            continue

        # Compare territory
        ts_territory = ts_result["territory"]
        cpp_territory = cpp_result.get("territory", {})

        if ts_territory != cpp_territory:
            mismatches.append(
                f"{tgn_file}: Territory mismatch\n"
                f"  TS:  {ts_territory}\n"
                f"  C++: {cpp_territory}"
            )
            continue

        # Compare board state
        ts_board = np.array(ts_result["board"])
        cpp_board = np.array(cpp_result.get("board", []))

        if not np.array_equal(ts_board, cpp_board):
            diff_count = np.sum(ts_board != cpp_board)
            mismatches.append(
                f"{tgn_file}: Board state mismatch ({diff_count} positions differ)"
            )
            continue

        matches += 1

    # Print results
    print("\n" + "=" * 70)
    if mismatches:
        print(f"❌ VALIDATION FAILED: {len(mismatches)} mismatches, {matches} matches\n")
        for mismatch in mismatches[:10]:  # Show first 10
            print(f"  • {mismatch}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
    else:
        print(f"✅ VALIDATION PASSED: All {matches} games match perfectly!")
    print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-language validation for Trigo game engine"
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=10,
        help="Number of random games to generate (default: 10)"
    )
    parser.add_argument(
        "--board", "-b",
        type=str,
        default="5*5*1",
        help="Board shape (default: 5*5*1)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory for TGN files (default: temp dir)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Cross-Language Trigo Game Validation")
    print("=" * 70)
    print(f"Board shape: {args.board}")
    print(f"Game count: {args.count}")

    # Step 1: Generate random games
    tgn_files = generate_random_games(
        count=args.count,
        board_shape=args.board,
        output_dir=args.output_dir
    )

    if not tgn_files:
        print("\n❌ No games generated!")
        sys.exit(1)

    # Step 2: Run TypeScript validation
    ts_results = run_typescript_validation(tgn_files)

    # Step 3: Run C++ validation
    cpp_results = run_cpp_validation(tgn_files)

    # Step 4: Compare results
    compare_results(ts_results, cpp_results)


if __name__ == "__main__":
    main()
