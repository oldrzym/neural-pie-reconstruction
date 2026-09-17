"""Helper script to run validation in chunks."""
import sys
import subprocess

def run_chunk(start_row, chunk_size=100):
    """Run validation for a specific chunk."""
    print(f"\n{'='*80}")
    print(f"RUNNING CHUNK: rows {start_row}-{start_row + chunk_size - 1}")
    print('='*80 + "\n")

    # Update START_ROW and CHUNK_SIZE in validate_wiktionary.py
    script_path = "dataset/scripts/validate_wiktionary.py"

    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace START_ROW and CHUNK_SIZE
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if line.startswith('START_ROW = '):
            new_lines.append(f'START_ROW = {start_row}         # 0-indexed, skip first N rows')
        elif line.startswith('CHUNK_SIZE = '):
            new_lines.append(f'CHUNK_SIZE = {chunk_size}      # process N rows from START_ROW (None = all remaining rows)')
        else:
            new_lines.append(line)

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))

    # Run the script
    result = subprocess.run([sys.executable, script_path], capture_output=False)

    if result.returncode == 0:
        print(f"\n✓ Chunk {start_row}-{start_row + chunk_size - 1} completed successfully!")
    else:
        print(f"\n✗ Chunk {start_row}-{start_row + chunk_size - 1} failed!")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_chunk.py START_ROW [CHUNK_SIZE]")
        print("Example: python run_chunk.py 0 100")
        print("Example: python run_chunk.py 100 100")
        sys.exit(1)

    start = int(sys.argv[1])
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    run_chunk(start, size)
