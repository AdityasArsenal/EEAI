from model.randomforest import RandomForest


def model_predict(data, df, name):
    """
    Train and evaluate the RandomForest model
    Args:
        data: Data object containing train/test splits
        df: Original dataframe (not used in basic version)
        name: Name identifier for the model run
    """
    # Check if data has valid train/test split
    if data.X_train is None:
        print("Skipping modeling - insufficient data")
        return
    
    print("=" * 50)
    print(f"Training RandomForest Model: {name}")
    print("=" * 50)
    
    # Initialize RandomForest model
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    
    # Train the model
    model.train(data)
    
    # Make predictions on test set
    model.predict(data.X_test)
    
    # Print evaluation results
    model_evaluate(model, data)
    
    print("=" * 50)


def model_evaluate(model, data):
    """
    Evaluate and print model performance metrics
    Args:
        model: Trained model instance
        data: Data object with test labels
    """
    model.print_results(data)
