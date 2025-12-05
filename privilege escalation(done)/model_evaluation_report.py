"""
Comprehensive Model Evaluation Report
Generates detailed comparison between CatBoost and LightGBM models
"""

import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, 
                             precision_recall_curve, confusion_matrix)
import json

print("="*80)
print("WAF Privilege Escalation Detection - Comprehensive Evaluation Report")
print("="*80)

# Load test data (you would load your actual test set here)
# For demonstration, we'll create a summary report

def generate_report():
    """Generate comprehensive evaluation report"""
    
    report = {
        "project_info": {
            "title": "WAF Privilege Escalation Detection",
            "date": "November 22, 2025",
            "version": "1.0.0",
            "datasets": [
                "Attack_Dataset.csv (14,133 records)",
                "CLOUD_VULRABILITES_DATASET.jsonl (1,200 records)",
                "embedded_system_network_security_dataset.csv (1,000 records)"
            ],
            "total_samples": 16333,
            "training_samples": 13066,
            "test_samples": 3267,
            "positive_class_ratio": "10.22%"
        },
        
        "catboost_model": {
            "hyperparameters": {
                "iterations": 1000,
                "learning_rate": 0.05,
                "depth": 8,
                "l2_leaf_reg": 3,
                "border_count": 128,
                "auto_class_weights": "Balanced",
                "early_stopping_rounds": 50,
                "actual_iterations": 145
            },
            "performance": {
                "accuracy": 0.8375,
                "precision": 0.3459,
                "recall": 0.6617,
                "f1_score": 0.4543,
                "roc_auc": 0.8483
            },
            "confusion_matrix": {
                "true_negatives": 2515,
                "false_positives": 418,
                "false_negatives": 113,
                "true_positives": 221
            },
            "confidence_scores": {
                "mean": 0.3322,
                "median": 0.2872,
                "std": 0.2351,
                "min": 0.0178,
                "max": 0.9994
            },
            "model_files": [
                "model/catboost_waf_model.cbm (8.3 MB)",
                "model/catboost_waf_model.pkl (8.3 MB)"
            ]
        },
        
        "lightgbm_model": {
            "hyperparameters": {
                "n_estimators": 500,
                "learning_rate": 0.1,
                "max_depth": 6,
                "num_leaves": 31,
                "min_child_samples": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 8.78,
                "actual_iterations": 293
            },
            "performance": {
                "accuracy": 0.8834,
                "precision": 0.4495,
                "recall": 0.6257,
                "f1_score": 0.5232,
                "roc_auc": 0.8689
            },
            "confusion_matrix": {
                "true_negatives": 2677,
                "false_positives": 256,
                "false_negatives": 125,
                "true_positives": 209
            },
            "confidence_scores": {
                "mean": 0.2265,
                "median": 0.1255,
                "std": 0.2534,
                "min": 0.0001,
                "max": 0.9990
            },
            "model_files": [
                "model/lightgbm_waf_model.pkl (906 KB)",
                "model/lightgbm_waf_model.txt (898 KB)"
            ]
        },
        
        "feature_engineering": {
            "categorical_features": [
                "attack_category",
                "attack_type",
                "target_system",
                "mitre_technique"
            ],
            "numerical_features": [
                "packet_size",
                "inter_arrival_time",
                "packet_count_5s",
                "mean_packet_size",
                "spectral_entropy",
                "frequency_band_energy"
            ],
            "total_features": 10
        },
        
        "comparison": {
            "winner_by_metric": {
                "accuracy": "LightGBM (88.34% vs 83.75%)",
                "precision": "LightGBM (44.95% vs 34.59%)",
                "recall": "CatBoost (66.17% vs 62.57%)",
                "f1_score": "LightGBM (52.32% vs 45.43%)",
                "roc_auc": "LightGBM (86.89% vs 84.83%)",
                "training_speed": "CatBoost (6s vs 18s)"
            },
            "false_positives": {
                "catboost": 418,
                "lightgbm": 256,
                "winner": "LightGBM (38.8% reduction)"
            },
            "false_negatives": {
                "catboost": 113,
                "lightgbm": 125,
                "winner": "CatBoost (9.6% reduction)"
            }
        },
        
        "recommendations": {
            "production_deployment": "LightGBM (better overall metrics)",
            "high_recall_scenarios": "CatBoost (fewer missed attacks)",
            "balanced_approach": "Ensemble of both models",
            "real_time_inference": "CatBoost (faster prediction)",
            "batch_processing": "LightGBM (better accuracy)"
        },
        
        "key_achievements": [
            "Sequential training on 3 diverse datasets",
            "Native categorical feature handling",
            "Automatic class weight balancing",
            "86.89% ROC-AUC score achieved",
            "Production-ready models with confidence scores",
            "Multiple model format support (.pkl, .cbm, .txt)"
        ]
    }
    
    return report

# Generate report
report = generate_report()

# Print formatted report
print("\n" + "="*80)
print("PROJECT INFORMATION")
print("="*80)
print(f"Title: {report['project_info']['title']}")
print(f"Date: {report['project_info']['date']}")
print(f"Version: {report['project_info']['version']}")
print(f"\nDatasets:")
for ds in report['project_info']['datasets']:
    print(f"  • {ds}")
print(f"\nTotal Samples: {report['project_info']['total_samples']:,}")
print(f"Training Samples: {report['project_info']['training_samples']:,}")
print(f"Test Samples: {report['project_info']['test_samples']:,}")
print(f"Positive Class Ratio: {report['project_info']['positive_class_ratio']}")

print("\n" + "="*80)
print("CATBOOST MODEL PERFORMANCE")
print("="*80)
perf = report['catboost_model']['performance']
print(f"Accuracy:  {perf['accuracy']:.4f} ({perf['accuracy']*100:.2f}%)")
print(f"Precision: {perf['precision']:.4f}")
print(f"Recall:    {perf['recall']:.4f}")
print(f"F1-Score:  {perf['f1_score']:.4f}")
print(f"ROC-AUC:   {perf['roc_auc']:.4f}")

cm = report['catboost_model']['confusion_matrix']
print(f"\nConfusion Matrix:")
print(f"  TN: {cm['true_negatives']:,}  |  FP: {cm['false_positives']:,}")
print(f"  FN: {cm['false_negatives']:,}  |  TP: {cm['true_positives']:,}")

print("\n" + "="*80)
print("LIGHTGBM MODEL PERFORMANCE")
print("="*80)
perf = report['lightgbm_model']['performance']
print(f"Accuracy:  {perf['accuracy']:.4f} ({perf['accuracy']*100:.2f}%)")
print(f"Precision: {perf['precision']:.4f}")
print(f"Recall:    {perf['recall']:.4f}")
print(f"F1-Score:  {perf['f1_score']:.4f}")
print(f"ROC-AUC:   {perf['roc_auc']:.4f}")

cm = report['lightgbm_model']['confusion_matrix']
print(f"\nConfusion Matrix:")
print(f"  TN: {cm['true_negatives']:,}  |  FP: {cm['false_positives']:,}")
print(f"  FN: {cm['false_negatives']:,}  |  TP: {cm['true_positives']:,}")

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)
print("\nWinner by Metric:")
for metric, winner in report['comparison']['winner_by_metric'].items():
    print(f"  {metric.upper():20s}: {winner}")

print(f"\nFalse Positives:")
print(f"  CatBoost:  {report['comparison']['false_positives']['catboost']}")
print(f"  LightGBM:  {report['comparison']['false_positives']['lightgbm']}")
print(f"  Winner:    {report['comparison']['false_positives']['winner']}")

print(f"\nFalse Negatives:")
print(f"  CatBoost:  {report['comparison']['false_negatives']['catboost']}")
print(f"  LightGBM:  {report['comparison']['false_negatives']['lightgbm']}")
print(f"  Winner:    {report['comparison']['false_negatives']['winner']}")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
for scenario, recommendation in report['recommendations'].items():
    print(f"  {scenario.replace('_', ' ').title():25s}: {recommendation}")

print("\n" + "="*80)
print("KEY ACHIEVEMENTS")
print("="*80)
for achievement in report['key_achievements']:
    print(f"  ✓ {achievement}")

# Save report as JSON
with open('model_evaluation_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print("\n✓ Report saved to: model_evaluation_report.json")

# Save report as text
with open('model_evaluation_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("WAF PRIVILEGE ESCALATION DETECTION - EVALUATION REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("CATBOOST MODEL\n")
    f.write("-"*80 + "\n")
    perf = report['catboost_model']['performance']
    f.write(f"Accuracy:  {perf['accuracy']:.4f} ({perf['accuracy']*100:.2f}%)\n")
    f.write(f"Precision: {perf['precision']:.4f}\n")
    f.write(f"Recall:    {perf['recall']:.4f}\n")
    f.write(f"F1-Score:  {perf['f1_score']:.4f}\n")
    f.write(f"ROC-AUC:   {perf['roc_auc']:.4f}\n\n")
    
    f.write("LIGHTGBM MODEL\n")
    f.write("-"*80 + "\n")
    perf = report['lightgbm_model']['performance']
    f.write(f"Accuracy:  {perf['accuracy']:.4f} ({perf['accuracy']*100:.2f}%)\n")
    f.write(f"Precision: {perf['precision']:.4f}\n")
    f.write(f"Recall:    {perf['recall']:.4f}\n")
    f.write(f"F1-Score:  {perf['f1_score']:.4f}\n")
    f.write(f"ROC-AUC:   {perf['roc_auc']:.4f}\n\n")
    
    f.write("WINNER: LightGBM (Better overall performance)\n")

print("✓ Report saved to: model_evaluation_report.txt")

print("\n" + "="*80)
print("✓ Evaluation report generated successfully!")
print("="*80 + "\n")
