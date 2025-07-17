import matplotlib.pyplot as plt
import numpy as np

# CORRECTED: Based on actual business model behavior
def estimate_active_parameters(total_params, alpha):
    """Corrected estimation formula based on empirical revenue model data"""
    
    if alpha <= 0.5:
        return int(total_params * 0.75)     # Keep ~75% (corrected from 80%)
    elif alpha <= 1.0:
        return int(total_params * 0.45)     # Keep ~45% (corrected from 40%)  
    elif alpha <= 2.0:
        return int(total_params * 0.28)     # Keep ~28% (corrected from 20%)
    elif alpha <= 5.0:
        return int(total_params * 0.18)     # Keep ~18% (corrected from 10% - was way off!)
    elif alpha <= 10.0:
        return int(total_params * 0.12)     # Keep ~12% (corrected from 6%)
    elif alpha <= 20.0:
        return int(total_params * 0.08)     # Keep ~8% (corrected from 4%)
    else:
        return int(total_params * 0.04)     # Keep ~4% (corrected from 2%)

def plot_alpha_curve(total_params=152, data_points=60):
    """Plot alpha vs active parameters curve"""
    
    # Create range of alpha values
    alphas = np.logspace(-1, 1.5, 50)  # From 0.1 to ~30
    
    # Calculate estimated active parameters for each alpha
    active_params = [estimate_active_parameters(total_params, alpha) for alpha in alphas]
    
    # Calculate ratios
    ratios = [data_points / active for active in active_params]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Alpha vs Active Parameters
    ax1.semilogx(alphas, active_params, 'b-o', markersize=4, linewidth=2)
    ax1.axhline(y=12, color='r', linestyle='--', alpha=0.7, label='Target (12 active for 5:1 ratio)')
    ax1.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Target (10 active for 6:1 ratio)')
    ax1.axhline(y=6, color='green', linestyle='--', alpha=0.7, label='Target (6 active for 10:1 ratio)')
    
    # Add empirical validation point if using 77 parameters
    if total_params == 77:
        ax1.plot(5.0, 15, 'ro', markersize=8, label='Your Actual Result (α=5, 15 active)')
    
    ax1.set_xlabel('Alpha (Regularization Strength)', fontsize=12)
    ax1.set_ylabel('Active Parameters', fontsize=12)
    ax1.set_title(f'Alpha vs Active Parameters\n(Total Parameters: {total_params})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for key points
    key_alphas = [1.0, 2.0, 5.0, 10.0, 20.0]
    for alpha in key_alphas:
        active = estimate_active_parameters(total_params, alpha)
        ax1.annotate(f'α={alpha}\n{active} active', 
                    xy=(alpha, active), 
                    xytext=(alpha*1.5, active+5),
                    arrowprops=dict(arrowstyle='->', alpha=0.7),
                    fontsize=9,
                    ha='center')
    
    # Plot 2: Alpha vs Data Ratio
    ax2.semilogx(alphas, ratios, 'g-o', markersize=4, linewidth=2)
    ax2.axhline(y=5.0, color='r', linestyle='--', alpha=0.7, label='Safe Ratio (5:1)')
    ax2.axhline(y=10.0, color='orange', linestyle='--', alpha=0.7, label='Ideal Ratio (10:1)')
    
    # Add empirical validation point if using 77 parameters
    if total_params == 77:
        ax2.plot(5.0, 4.0, 'ro', markersize=8, label='Your Actual Result (α=5, 4:1 ratio)')
    
    ax2.set_xlabel('Alpha (Regularization Strength)', fontsize=12)
    ax2.set_ylabel('Data-to-Parameter Ratio', fontsize=12)
    ax2.set_title(f'Alpha vs Safety Ratio\n(Data Points: {data_points})', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add annotations for key ratios
    for alpha in key_alphas:
        active = estimate_active_parameters(total_params, alpha)
        ratio = data_points / active
        color = 'green' if ratio >= 5.0 else 'orange' if ratio >= 3.0 else 'red'
        ax2.annotate(f'α={alpha}\nRatio={ratio:.1f}', 
                    xy=(alpha, ratio), 
                    xytext=(alpha*1.5, ratio+1),
                    arrowprops=dict(arrowstyle='->', alpha=0.7, color=color),
                    fontsize=9,
                    ha='center',
                    color=color)
    
    plt.tight_layout()
    plt.show()
    
    return alphas, active_params, ratios

def run_estimate(total_params=152, plot=False):
    """Run estimation with optional plotting"""
    
    print(f"CORRECTED Estimation for {total_params} total parameters:")
    print("="*50)
    
    # Examples with specified parameters
    for alpha in [0.5, 1.0, 2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0]:
        estimated = estimate_active_parameters(total_params, alpha)
        ratio = 60 / estimated
        safety = "✅" if ratio >= 5.0 else "⚠️" if ratio >= 3.0 else "🚨"
        print(f"Alpha {alpha:4.1f}: ~{estimated:2d} active parameters, ratio {ratio:.1f}:1 {safety}")
    
    # Show validation against known results
    if total_params == 77:
        print(f"\n🎯 VALIDATION: Your actual alpha=5.0 result was ~15 active parameters")
        predicted = estimate_active_parameters(77, 5.0)
        error = abs(15 - predicted)
        print(f"   Corrected formula predicts: {predicted} active (error: {error})")
        print(f"   Accuracy: {'✅ Much better!' if error <= 3 else '❌ Still off'}")
    
    if plot:
        print("\nGenerating corrected alpha curve plot...")
        alphas, active_params, ratios = plot_alpha_curve(total_params)
        
        # Find optimal alphas for different targets
        target_ratios = [5.0, 6.0, 10.0]
        print(f"\nCORRECTED alpha recommendations:")
        for target_ratio in target_ratios:
            target_active = 60 / target_ratio
            best_idx = np.argmin(np.abs(np.array(active_params) - target_active))
            optimal_alpha = alphas[best_idx]
            actual_active = active_params[best_idx]
            actual_ratio = ratios[best_idx]
            print(f"For {target_ratio}:1 ratio: Use alpha ≈ {optimal_alpha:.1f} "
                  f"({actual_active} active, {actual_ratio:.1f}:1 ratio)")


if __name__ == "__main__":
    # Run basic estimation
    run_estimate(total_params=135, plot=True)
    
    print("\n" + "="*60)