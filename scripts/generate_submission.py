import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ensemble.inference.predict import generate_submission

if __name__ == '__main__':
    print("Generating submission file...")
    submission_df = generate_submission()
    print("\n✓ Done!")
