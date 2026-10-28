import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_models(trained_models, X_test, y_test_price, last_observed_price_series):
    """Calculates chronological out-of-sample metrics for predicted price levels.
    
    y_pred_level = last_observed_price + delta_pred
    Evaluated against the true observed retail prices y_test_price.
    """
    results = {}
    naive_mae = None
    
    for name, model in trained_models.items():
        delta_pred = model.predict(X_test)
        y_pred_price = last_observed_price_series.values + delta_pred
        
        mae = float(mean_absolute_error(y_test_price, y_pred_price))
        mse = float(mean_squared_error(y_test_price, y_pred_price))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test_price, y_pred_price))
        
        if "Naive" in name:
            naive_mae = mae
            
        results[name] = {
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4),
            'y_pred': y_pred_price
        }
        
    for name in results:
        if naive_mae and naive_mae > 0:
            imp = ((naive_mae - results[name]['MAE']) / naive_mae) * 100.0
            results[name]['vs_Naive_Improvement_%'] = round(imp, 2)
        else:
            results[name]['vs_Naive_Improvement_%'] = 0.0
            
    return results

