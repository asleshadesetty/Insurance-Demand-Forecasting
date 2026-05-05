"""
run_pipeline.py
---------------
Single entry point that runs the full project end to end:

  Step 1 — Generate synthetic dataset
  Step 2 — Train Prophet models and produce forecasts
  Step 3 — Generate all visualizations

Usage:
    python run_pipeline.py
"""

import subprocess
import sys

steps = [
    ("Generating dataset",       "src/generate_data.py"),
    ("Running forecast pipeline","src/forecast.py"),
    ("Generating visualizations","src/visualize.py"),
]

for label, script in steps:
    print(f"\n{'='*60}")
    print(f"  {label}...")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script], check=True)

print("\n" + "="*60)
print("  Pipeline complete.")
print("  Outputs  ->  outputs/")
print("  Charts   ->  visuals/")
print("="*60 + "\n")
