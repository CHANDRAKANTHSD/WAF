import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def load_report(path='performance_report.json'):
    """Load performance report"""
    with open(path, 'r') as f:
        return json.load(f)

def plot_model_comparison(report):
    """Create comprehensive visualization of model comparison"""
    metrics_data = report['models']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('WAF Authentication Detection - Model Comparison', fontsize=16, fontweight='bold')
    
    models = list(metrics_data.keys())
    
    # 1. Performance Metrics Comparison
    ax1 = axes[0, 0]
    metrics = ['precision', 'recall', 'f1_score', 'auc_roc']
    x = np.arange(len(metrics))
    width = 0.35
    
    xgb_values = [metrics_data['XGBoost'][m] for m in metrics]
    lstm_values = [metrics_data['LSTM'][m] for m in metrics]
    
    ax1.bar(x - width/2, xgb_values, width, label='XGBoost', color='#2ecc71')
    ax1.bar(x + width/2, lstm_values, width, label='LSTM', color='#3498db')
    
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Performance Metrics Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Precision', 'Recall', 'F1-Score', 'AUC-ROC'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Add value labels
    for i, (xgb_val, lstm_val) in enumerate(zip(xgb_values, lstm_values)):
        ax1.text(i - width/2, xgb_val + 0.02, f'{xgb_val:.3f}', ha='center', fontsize=9)
        ax1.text(i + width/2, lstm_val + 0.02, f'{lstm_val:.3f}', ha='center', fontsize=9)
    
    # 2. Inference Latency Comparison
    ax2 = axes[0, 1]
    latencies = [metrics_data[model]['latency_ms'] for model in models]
    colors = ['#2ecc71' if lat < 100 else '#e74c3c' for lat in latencies]
    
    bars = ax2.bar(models, latencies, color=colors, alpha=0.7)
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100ms Threshold')
    ax2.set_ylabel('Latency (ms)', fontweight='bold')
    ax2.set_title('Inference Latency Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, lat in zip(bars, latencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{lat:.2f}ms', ha='center', va='bottom', fontweight='bold')
    
    # 3. F1-Score vs Latency Trade-off
    ax3 = axes[1, 0]
    f1_scores = [metrics_data[model]['f1_score'] for model in models]
    
    scatter = ax3.scatter(latencies, f1_scores, s=300, alpha=0.6, 
                         c=['#2ecc71', '#3498db'], edgecolors='black', linewidth=2)
    
    for i, model in enumerate(models):
        ax3.annotate(model, (latencies[i], f1_scores[i]), 
                    fontsize=12, fontweight='bold', ha='center', va='center')
    
    ax3.set_xlabel('Inference Latency (ms)', fontweight='bold')
    ax3.set_ylabel('F1-Score', fontweight='bold')
    ax3.set_title('F1-Score vs Latency Trade-off', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='100ms Threshold')
    ax3.legend()
    
    # 4. Overall Score Radar Chart
    ax4 = axes[1, 1]
    categories = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    
    xgb_values_radar = [metrics_data['XGBoost'][m] for m in ['precision', 'recall', 'f1_score', 'auc_roc']]
    xgb_values_radar += xgb_values_radar[:1]
    
    lstm_values_radar = [metrics_data['LSTM'][m] for m in ['precision', 'recall', 'f1_score', 'auc_roc']]
    lstm_values_radar += lstm_values_radar[:1]
    
    ax4.plot(angles, xgb_values_radar, 'o-', linewidth=2, label='XGBoost', color='#2ecc71')
    ax4.fill(angles, xgb_values_radar, alpha=0.25, color='#2ecc71')
    
    ax4.plot(angles, lstm_values_radar, 'o-', linewidth=2, label='LSTM', color='#3498db')
    ax4.fill(angles, lstm_values_radar, alpha=0.25, color='#3498db')
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 1)
    ax4.set_title('Performance Radar Chart', fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to model_comparison.png")
    plt.close()

def create_summary_table(report):
    """Create summary table"""
    metrics_data = report['models']
    
    df = pd.DataFrame(metrics_data).T
    df = df.round(4)
    
    # Add pass/fail for latency
    df['Latency Status'] = df['latency_ms'].apply(lambda x: '✓ PASS' if x < 100 else '✗ FAIL')
    
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY TABLE")
    print("="*70)
    print(df.to_string())
    
    # Save to CSV
    df.to_csv('performance_summary.csv')
    print("\n✓ Summary table saved to performance_summary.csv")

if __name__ == "__main__":
    report = load_report()
    plot_model_comparison(report)
    create_summary_table(report)
