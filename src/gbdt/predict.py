def predict_one(model, row_df):
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        return model.predict(row_df, iteration_range=(0, model.best_iteration + 1))[0]
    return model.predict(row_df)[0]

