#!/usr/bin/env python3
"""
Normalize branch_consistency scores in trajectories.jsonl

This script handles both old and new trajectory formats:

OLD FORMAT (branch_consistency = raw logits):
  1. Reads raw branch_consistency logits
  2. Applies sigmoid normalization to get [0,1] values
  3. Overwrites branch_consistency with normalized values
  4. Stores raw in branch_consistency_raw
  5. Recomputes escalated flag: escalated = (branch_consistency < threshold)

NEW FORMAT (branch_consistency_raw + branch_consistency):
  1. Keeps both raw and normalized values as-is
  2. Recomputes escalated flag based on normalized branch_consistency
  3. Avoids double-normalization

Usage:
    python normalize_branch_consistency.py <input_file> [output_file] [threshold]

Example:
    python normalize_branch_consistency.py results/gsm8k/trajectories.jsonl
    python normalize_branch_consistency.py results/gsm8k/trajectories.jsonl results/gsm8k/trajectories_normalized.jsonl 0.6
    python normalize_branch_consistency.py results/gsm8k/trajectories.jsonl results/gsm8k/trajectories_normalized.jsonl 0.7
"""

import json
import sys
from pathlib import Path
from scipy.special import expit  # sigmoid function
import numpy as np


def sigmoid(x):
    """Apply sigmoid function to normalize logits to [0, 1]"""
    return float(expit(x))


def normalize_trajectories(input_file, output_file=None, threshold=0.6):
    """
    Read trajectories, normalize branch_consistency, and recompute escalated flag.

    Handles both old and new trajectory formats:
    - OLD: branch_consistency contains raw logits → normalize via sigmoid
    - NEW: branch_consistency_raw and branch_consistency both present → keep as-is, recompute escalated

    Parameters
    ----------
    input_file : str or Path
        Path to input trajectories.jsonl file
    output_file : str or Path, optional
        Path to output file. If None, overwrites input file.
    threshold : float, optional
        Escalation threshold in [0,1]. Default: 0.6
        Escalated will be True if normalized branch_consistency < threshold

    Returns
    -------
    dict
        Summary statistics:
        - n_trajectories: number of trajectories processed
        - n_steps: total number of steps
        - n_with_branch_consistency: steps that had branch_consistency
        - n_escalated_changed: number of steps where escalated flag was changed
        - stats: dict with min, max, mean, std of normalized values
        - escalation_stats: dict with counts before/after normalization
        - format_stats: dict tracking old vs new format steps
    """

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_file is None:
        output_path = input_path
    else:
        output_path = Path(output_file)

    # Validate threshold
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}")

    # Temporary file for writing
    temp_path = output_path.parent / f"{output_path.stem}_temp{output_path.suffix}"

    print(f"Reading from:  {input_path}")
    print(f"Writing to:    {output_path}")
    print(f"Threshold:     {threshold}")
    print()

    n_trajectories = 0
    n_steps = 0
    n_with_branch_consistency = 0
    n_escalated_changed = 0
    normalized_values = []

    # Track escalation changes
    escalated_before_count = 0
    escalated_after_count = 0

    # Track format (old vs new)
    n_old_format = 0  # Only branch_consistency (raw logits)
    n_new_format = 0  # Both branch_consistency_raw and branch_consistency

    # Read and process
    with open(input_path, 'r') as infile, open(temp_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            if not line.strip():
                continue

            try:
                traj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

            n_trajectories += 1

            # Process each step in trajectory
            if 'trajectory' in traj:
                for step in traj['trajectory']:
                    n_steps += 1

                    # Check format and handle accordingly
                    has_raw = 'branch_consistency_raw' in step
                    has_bc = 'branch_consistency' in step

                    if has_bc:
                        n_with_branch_consistency += 1

                        # Track before
                        if step.get('escalated', False):
                            escalated_before_count += 1

                        if has_raw:
                            # NEW FORMAT: Both raw and normalized present
                            # Keep both as-is, just recompute escalated flag
                            n_new_format += 1
                            normalized_score = step['branch_consistency']

                        else:
                            # OLD FORMAT: Only branch_consistency (raw logits)
                            # Normalize and store both
                            n_old_format += 1
                            raw_score = step['branch_consistency']
                            normalized_score = sigmoid(raw_score)
                            step['branch_consistency_raw'] = raw_score
                            step['branch_consistency'] = normalized_score

                        # Recompute escalated flag (use normalized value)
                        new_escalated = normalized_score < threshold
                        old_escalated = step.get('escalated', False)

                        if new_escalated != old_escalated:
                            n_escalated_changed += 1

                        step['escalated'] = new_escalated

                        # Track after
                        if new_escalated:
                            escalated_after_count += 1

                        normalized_values.append(normalized_score)

            # Write to output file
            outfile.write(json.dumps(traj) + '\n')

            # Progress indicator
            if line_num % 100 == 0:
                print(f"  Processed {line_num} trajectories...")

    # Replace original with temp file
    if output_path != input_path:
        # Different output file - just rename temp
        temp_path.rename(output_path)
    else:
        # Overwriting input - be careful
        input_path.unlink()  # Delete original
        temp_path.rename(output_path)

    # Compute statistics
    if normalized_values:
        stats = {
            'min': float(np.min(normalized_values)),
            'max': float(np.max(normalized_values)),
            'mean': float(np.mean(normalized_values)),
            'median': float(np.median(normalized_values)),
            'std': float(np.std(normalized_values)),
        }
    else:
        stats = {}

    escalation_stats = {
        'before': escalated_before_count,
        'after': escalated_after_count,
        'changed': n_escalated_changed,
    }

    format_stats = {
        'old_format (raw only)': n_old_format,
        'new_format (raw + normalized)': n_new_format,
    }

    return {
        'n_trajectories': n_trajectories,
        'n_steps': n_steps,
        'n_with_branch_consistency': n_with_branch_consistency,
        'n_escalated_changed': n_escalated_changed,
        'stats': stats,
        'escalation_stats': escalation_stats,
        'format_stats': format_stats,
    }


def main():
    """Main entry point"""

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6

    try:
        results = normalize_trajectories(input_file, output_file, threshold)

        print("\n" + "="*80)
        print("NORMALIZATION COMPLETE")
        print("="*80)
        print(f"\nSummary:")
        print(f"  Trajectories processed: {results['n_trajectories']}")
        print(f"  Total steps: {results['n_steps']}")
        print(f"  Steps with branch_consistency: {results['n_with_branch_consistency']}")

        print(f"\nFormat Detection:")
        print(f"  Old format (raw logits only):        {results['format_stats']['old_format (raw only)']}")
        print(f"  New format (raw + normalized):       {results['format_stats']['new_format (raw + normalized)']}")

        if results['stats']:
            print(f"\nNormalized branch_consistency statistics:")
            print(f"  Min:    {results['stats']['min']:.4f}")
            print(f"  Max:    {results['stats']['max']:.4f}")
            print(f"  Mean:   {results['stats']['mean']:.4f}")
            print(f"  Median: {results['stats']['median']:.4f}")
            print(f"  Std:    {results['stats']['std']:.4f}")

        print(f"\nEscalation flag recomputation:")
        print(f"  Changed escalation flags: {results['n_escalated_changed']}")
        print(f"  Escalated before: {results['escalation_stats']['before']}")
        print(f"  Escalated after:  {results['escalation_stats']['after']}")

        output_path = output_file if output_file else input_file
        print(f"\nOutput saved to: {output_path}")
        print("\nFields in each step:")
        print("  - branch_consistency_raw (raw NLI logits)")
        print("  - branch_consistency (normalized [0,1])")
        print("  - escalated (recomputed based on threshold)")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
