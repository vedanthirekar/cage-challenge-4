"""
Evaluation script for DQN solution using official CybORG evaluation
"""

import sys
import os
import tempfile
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from CybORG.Evaluation.evaluation import run_evaluation, load_submission


def evaluate_solution(max_episodes=10,
                      output_dir="cage_dqn_solution/evaluation_results"):
    """
    Evaluate the DQN solution using official evaluation
    """
    print("=" * 60)
    print("CAGE Challenge 4 - DQN Evaluation")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp(prefix="cage_dqn_submission_")

    try:
        # Copy submission files
        files_to_copy = ["submission.py", "dqn_agent.py"]
        for file_name in files_to_copy:
            src = os.path.join(current_dir, file_name)
            dst = os.path.join(temp_dir, file_name)
            shutil.copy2(src, dst)

        # Copy trained model
        model_src = os.path.join(current_dir, "trained_model")
        model_dst = os.path.join(temp_dir, "trained_model")
        if os.path.exists(model_src):
            shutil.copytree(model_src, model_dst)

        with open(os.path.join(temp_dir, "metadata"), "w") as f:
            pass

        print(f"Submission prepared in: {temp_dir}")

        submission = load_submission(temp_dir)
        print(f"Loaded: {submission.NAME}")
        print(f"Team: {submission.TEAM}")
        print(f"Technique: {submission.TECHNIQUE}")

        os.makedirs(output_dir, exist_ok=True)

        print(f"\nRunning evaluation ({max_episodes} episodes)...")
        run_evaluation(
            submission=submission,
            log_path=output_dir,
            max_eps=max_episodes,
            write_to_file=True,
            seed=42,
        )

        summary_file = os.path.join(output_dir, "summary.txt")
        if os.path.exists(summary_file):
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            with open(summary_file, "r") as f:
                print(f.read())

        scores_file = os.path.join(output_dir, "scores.txt")
        if os.path.exists(scores_file):
            with open(scores_file, "r") as f:
                print(f.read())

        print(f"✅ Evaluation completed! Results saved to: {output_dir}")
        return True

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate DQN solution")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes")
    parser.add_argument("--output-dir", type=str,
                        default="cage_dqn_solution/evaluation_results",
                        help="Output directory")

    args = parser.parse_args()

    success = evaluate_solution(
        max_episodes=args.episodes,
        output_dir=args.output_dir
    )

    sys.exit(0 if success else 1)
