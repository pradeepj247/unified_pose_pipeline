#!/usr/bin/env python3
"""
Version validation for Unified Pose Pipeline
Ensures all dependencies are at correct versions
"""

import pkg_resources
import sys

# Expected versions
expected_versions = {
    'onnxruntime-gpu': '1.23.0',
    'ultralytics': '8.3.228',
    'boxmot': '15.0.10',
    'rtmlib': None,  # Any version
}

print("🔍 Validating dependency versions...")
print("=" * 50)

all_correct = True

for package, expected_version in expected_versions.items():
    try:
        installed_version = pkg_resources.get_distribution(package).version
        if expected_version and installed_version != expected_version:
            print(f"❌ {package}: expected {expected_version}, got {installed_version}")
            all_correct = False
        else:
            print(f"✅ {package}: {installed_version}")
    except pkg_resources.DistributionNotFound:
        print(f"❌ {package}: NOT INSTALLED")
        all_correct = False

print("=" * 50)
if all_correct:
    print("🎉 All dependencies are at correct versions!")
    sys.exit(0)
else:
    print("❌ Some dependencies have incorrect versions")
    print("Run: pip install -r requirements.txt to fix")
    sys.exit(1)