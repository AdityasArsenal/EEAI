from model.randomforest import RandomForest
from Config import Config


def model_predict(data, df, name):
    print("\n" + "=" * 70)
    print(f"DESIGN CHOICE 1: CHAINED MULTI-OUTPUT CLASSIFICATION")
    print(f"Model Run: {name}")
    print("=" * 70)
    
    # Get all available targets
    available_targets = data.get_all_targets()
    
    if not available_targets:
        print("No valid targets found. Skipping modeling.")
        return
    
    # Train a separate model for each chained target
    results = {}
    
    for target_name in ['y2', 'y2_y3', 'y2_y3_y4']:
        if target_name not in available_targets:
            print(f"\n[WARNING] Target '{target_name}' not available. Skipping...")
            continue
        
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target_name}")
        print(f"{'=' * 70}")
        
        # Get train/test split for this target
        target_split = data.get_target_split(target_name)
        
        if target_split is None:
            print(f"Insufficient data for target '{target_name}'. Skipping...")
            continue
        
        # Display target information
        print(f"Training samples: {len(target_split['y_train'])}")
        print(f"Testing samples: {len(target_split['y_test'])}")
        print(f"Number of classes: {len(target_split['classes'])}")
        print(f"Classes: {list(target_split['classes'][:5])}{'...' if len(target_split['classes']) > 5 else ''}")
        
        # Initialize RandomForest model for this target
        model = RandomForest(
            model_name=f"RandomForest_{target_name}",
            embeddings=data.get_embeddings(),
            y=target_split['y_all']
        )
        
        # Create a temporary data object for this specific target
        class TargetData:
            def __init__(self, split):
                self.X_train = split['X_train']
                self.X_test = split['X_test']
                self.y_train = split['y_train']
                self.y_test = split['y_test']
        
        target_data = TargetData(target_split)
        
        # Train the model
        print(f"\nTraining {model.model_name}...")
        model.train(target_data)
        
        # Make predictions
        print(f"Making predictions...")
        model.predict(target_data.X_test)
        
        # Evaluate the model
        print(f"\nEvaluation Results for {target_name}:")
        print("-" * 70)
        model_evaluate(model, target_data)
        
        # Store results
        results[target_name] = {
            'model': model,
            'predictions': model.predictions
        }
    
    print("\n" + "=" * 70)
    print("CHAINED MULTI-OUTPUT CLASSIFICATION COMPLETE")
    print("=" * 70)
    
    return results


def model_evaluate(model, data):
    model.print_results(data)
