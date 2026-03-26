from model.randomforest import RandomForest
from Config import Config
from sklearn.metrics import accuracy_score
import pandas as pd
from utils import _TargetData


def model_predict(data, df, name):
    print("\n" + "=" * 70)
    print("DESIGN CHOICE 1: CHAINED MULTI-OUTPUT CLASSIFICATION")
    print(f"Model Run: {name}")
    print("=" * 70)

    available_targets = data.get_all_targets()

    if not available_targets:
        print("No valid targets found. Skipping modeling.")
        return

    results = {}

    for target_name in ['y2', 'y2_y3', 'y2_y3_y4']:
        if target_name not in available_targets:
            print(f"\n[WARNING] Target '{target_name}' not available. Skipping...")
            continue

        print(f"\n{'=' * 70}")
        print(f"TARGET: {target_name}")
        print(f"{'=' * 70}")

        target_split = data.get_target_split(target_name)

        print(f"Training samples : {len(target_split['y_train'])}")
        print(f"Testing samples  : {len(target_split['y_test'])}")
        print(f"Unique classes   : {pd.Series(target_split['y_train']).nunique()}")

        # Instantiate RandomForest for this target
        model = RandomForest(
            model_name=f"RandomForest_{target_name}",
            embeddings=data.get_embeddings(),
            y=target_split['y_train']
        )

        # Wrap split into object that RandomForest expects
        target_data = _TargetData(target_split)

        # Train → Predict → Evaluate
        print(f"\nTraining {model.model_name}...")
        model.train(target_data)

        print("Making predictions...")
        model.predict(target_data.X_test)

        print(f"\nEvaluation Results for {target_name}:")
        print("-" * 70)
        model_evaluate(model, target_data)

        results[target_name] = model.predictions

    # ── Accuracy summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHAINED ACCURACY SUMMARY")
    print("=" * 70)
    print("(All models tested on the SAME rows — results are directly comparable)")
    print("-" * 70)
    for target_name, preds in results.items():
        split = data.get_target_split(target_name)
        acc = accuracy_score(split['y_test'], preds)
        print(f"  {target_name:<15} → {acc*100:.2f}%")

    return results


def model_evaluate(model, data):
    model.print_results(data)
