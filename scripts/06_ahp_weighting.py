"""
Script 06: AHP Weighting
Calculates and validates Analytic Hierarchy Process (AHP) weights for criteria

Input: config.py (AHP_WEIGHTS and EXPERT_MATRICES)
Output: output/ahp_weights.csv

Steps:
1. Load expert pairwise comparison matrices
2. Calculate priority vectors using eigenvalue method
3. Compute Consistency Index (CI) and Consistency Ratio (CR)
4. Aggregate expert weights using geometric mean
5. Export final weights
"""

import pandas as pd
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

# Random Index (RI) values for different matrix sizes (Saaty, 1980)
RI_VALUES = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49
}

def calculate_priority_vector(matrix):
    """
    Calculate priority vector using eigenvalue method
    Returns: priority vector (normalized eigenvector)
    """
    matrix = np.array(matrix)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # Get index of maximum eigenvalue
    max_idx = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[max_idx].real
    
    # Get corresponding eigenvector
    priority_vector = eigenvectors[:, max_idx].real
    
    # Normalize
    priority_vector = priority_vector / priority_vector.sum()
    
    return priority_vector, lambda_max

def calculate_consistency(matrix, lambda_max):
    """
    Calculate Consistency Index (CI) and Consistency Ratio (CR)
    CI = (λmax - n) / (n - 1)
    CR = CI / RI
    """
    n = len(matrix)
    
    # Consistency Index
    CI = (lambda_max - n) / (n - 1)
    
    # Consistency Ratio
    RI = RI_VALUES.get(n, 1.0)
    CR = CI / RI
    
    return CI, CR

def geometric_mean_aggregation(weights_list):
    """
    Aggregate multiple weight vectors using geometric mean
    Recommended method for AHP group decision making
    """
    weights_array = np.array(weights_list)
    
    # Geometric mean
    geo_mean = np.prod(weights_array, axis=0) ** (1 / len(weights_array))
    
    # Normalize
    geo_mean = geo_mean / geo_mean.sum()
    
    return geo_mean

def compute_ahp_weights(output_file):
    """Main AHP weighting pipeline"""
    
    print("="*60)
    print("STEP 06: AHP WEIGHTING")
    print("="*60)
    
    criteria_names = [
        'Topic_Frequency',
        'User_Importance_Strength', 
        'Review_Recency',
        'App_Version_Spread'
    ]
    
    print(f"Criteria being evaluated:")
    for i, name in enumerate(criteria_names, 1):
        print(f"  {i}. {name}")
    
    # Process expert matrices
    print(f"Processing {len(config.EXPERT_MATRICES)} expert matrices...")
    
    expert_weights = []
    results = []
    
    for expert_name, expert_data in config.EXPERT_MATRICES.items():
        print(f"Processing {expert_name}:")
        
        matrix = np.array(expert_data['matrix'])
        
        # Calculate priority vector
        priority_vector, lambda_max = calculate_priority_vector(matrix)
        
        # Calculate consistency
        CI, CR = calculate_consistency(matrix, lambda_max)
        
        # Validate consistency
        if CR < 0.10:
            status = "Acceptable"
        elif CR < 0.15:
            status = "Marginal"
        else:
            status = "Inconsistent"
        
        print(f"  λ_max: {lambda_max:.4f}")
        print(f"  CI:    {CI:.4f}")
        print(f"  CR:    {CR:.4f} {status}")
        print(f"  Weights: {priority_vector}")
        
        expert_weights.append(priority_vector)
        
        # Store results
        results.append({
            'expert': expert_name,
            'lambda_max': lambda_max,
            'CI': CI,
            'CR': CR,
            **{criteria_names[i]: priority_vector[i] for i in range(len(criteria_names))}
        })
    
    # Aggregate weights using geometric mean
    print(f"Aggregating expert weights using geometric mean...")
    aggregated_weights = geometric_mean_aggregation(expert_weights)
    
    print(f"Final Aggregated Weights:")
    for name, weight in zip(criteria_names, aggregated_weights):
        print(f"  {name:30s}: {weight:.4f}")
    print(f"  Sum: {aggregated_weights.sum():.4f}")
    
    # Compare with config weights
    print(f"Comparison with config.py weights:")
    config_weights = [config.AHP_WEIGHTS[name] for name in criteria_names]
    print(f"{'Criterion':<30} {'Calculated':<12} {'Config':<12} {'Difference':<12}")
    print("-" * 66)
    for name, calc, conf in zip(criteria_names, aggregated_weights, config_weights):
        diff = calc - conf
        print(f"{name:<30} {calc:<12.4f} {conf:<12.4f} {diff:<+12.4f}")
    
    # Add aggregated weights to results
    results.append({
        'expert': 'Aggregated',
        'lambda_max': np.nan,
        'CI': np.nan,
        'CR': np.nan,
        **{criteria_names[i]: aggregated_weights[i] for i in range(len(criteria_names))}
    })
    
    # Create dataframe
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    print("\n" + "="*60)
    print("AHP WEIGHTING SUMMARY")
    print("="*60)
    print(f"Number of experts: {len(config.EXPERT_MATRICES)}")
    print(f"All CR < 0.10:     {all(r['CR'] < 0.10 for r in results[:-1])}")
    print(f"\nFinal weights (to be used in TOPSIS):")
    for name, weight in zip(criteria_names, aggregated_weights):
        print(f"  {name}: {weight:.4f}")
    
    # Save results
    print(f"Saving AHP weights to: {output_file}")
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Saved AHP analysis results")
    
    # Save final weights separately
    weights_file = output_file.replace('.csv', '_final.csv')
    final_weights_df = pd.DataFrame({
        'criterion': criteria_names,
        'weight': aggregated_weights
    })
    final_weights_df.to_csv(weights_file, index=False, encoding='utf-8-sig')
    print(f"Saved final weights to: {weights_file}")
    
    return True

if __name__ == "__main__":
    success = compute_ahp_weights(config.AHP_WEIGHTS_FILE)
    
    if success:
        print("\n" + "="*60)
        print("AHP WEIGHTING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nNext step: Run 07_topsis_ranking.py")
    else:
        print("\n" + "="*60)
        print("AHP WEIGHTING FAILED")
        print("="*60)
        sys.exit(1)
