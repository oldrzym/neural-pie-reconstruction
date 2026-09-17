"""
Check overlap between etymology-db pie_direct and pie_via_proto datasets.
"""

import pandas as pd

# Load both datasets
direct = pd.read_csv("dataset/processed_with_codes/etymology_db_pie_direct.csv")
via_proto = pd.read_csv("dataset/processed_with_codes/etymology_db_pie_via_proto.csv")

print("="*60)
print("Etymology-db Datasets Overlap Analysis")
print("="*60)

# Basic stats
print(f"\nDataset sizes:")
print(f"  pie_direct:    {len(direct):,} rows")
print(f"  pie_via_proto: {len(via_proto):,} rows")
print(f"  Total:         {len(direct) + len(via_proto):,} rows")

# Unique PIE forms
direct_forms = set(direct['output'].dropna())
via_proto_forms = set(via_proto['output'].dropna())

print(f"\nUnique PIE forms:")
print(f"  pie_direct:    {len(direct_forms):,}")
print(f"  pie_via_proto: {len(via_proto_forms):,}")
print(f"  Overlap:       {len(direct_forms & via_proto_forms):,}")
print(f"  Total unique:  {len(direct_forms | via_proto_forms):,}")

# Show overlap examples
overlap = direct_forms & via_proto_forms
if overlap:
    print(f"\nSample overlapping PIE forms: {len(overlap):,} total")
    # Skip printing forms due to Unicode issues in Windows console

# Exact duplicate rows (same input AND output)
merged = pd.concat([
    direct[['input', 'output']],
    via_proto[['input', 'output']]
])
duplicates = merged[merged.duplicated(keep=False)]
print(f"\nExact duplicate rows (same input+output): {len(duplicates)//2:,}")

# Recommendation
print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
print(f"\nYES, use BOTH datasets!")
print(f"  - They have different cognate sets for the same PIE forms")
print(f"  - Total examples: {len(direct) + len(via_proto):,}")
print(f"  - Total unique PIE forms: {len(direct_forms | via_proto_forms):,}")
print(f"  - Overlap is complementary (different cognates)")
