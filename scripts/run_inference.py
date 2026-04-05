import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import argparse


def run_inference():
    print("=" * 60)
    print("RUNNING INFERENCE ON TEST SET")
    print("=" * 60)
    
    from ensemble.inference.predict import generate_submission
    
    print("\nLoading models and extracting features...")
    submission_df = generate_submission()
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    
    return submission_df


def main():
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/submissions/submission.csv',
        help='Output path for submission file'
    )
    
    args = parser.parse_args()
    
    from ensemble.inference.predict import predict_test_set, generate_submission
    from ensemble.config.config import get_config
    
    config = get_config()
    
    print("Running inference...")
    results_df, probabilities = predict_test_set(config)
    
    submission_df = results_df[['id', 'label']].copy()
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(args.output, index=False)
    
    print(f"\n✓ Submission saved to {args.output}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nClass distribution:")
    print(submission_df['label'].value_counts())


if __name__ == '__main__':
    main()
