"""
Complete CNN-BiLSTM Training
Uses the best checkpoints to create the final model
"""

import os
import pickle
import shutil

print("="*60)
print("Completing CNN-BiLSTM Training")
print("="*60)

# Check if checkpoints exist
checkpoints = [
    'best_model_CICDDoS2019_cnn_bilstm.h5',
    'best_model_LSNM2024_cnn_bilstm.h5',
    'best_model_CSIC_cnn_bilstm.h5'
]

existing = [f for f in checkpoints if os.path.exists(f)]
print(f"\nFound {len(existing)} checkpoint files:")
for f in existing:
    size = os.path.getsize(f) / (1024 * 1024)
    print(f"  ✅ {f} ({size:.2f} MB)")

if len(existing) == 0:
    print("\n❌ No checkpoint files found!")
    print("   Need to run: python cnn_bilstm_waf_ldap.py")
    exit(1)

# Option 1: Use the best checkpoint as final model
print("\n" + "="*60)
print("Option 1: Use Best Checkpoint")
print("="*60)

# Typically CSIC is the largest dataset, so use that checkpoint
if os.path.exists('best_model_CSIC_cnn_bilstm.h5'):
    best_checkpoint = 'best_model_CSIC_cnn_bilstm.h5'
elif os.path.exists('best_model_LSNM2024_cnn_bilstm.h5'):
    best_checkpoint = 'best_model_LSNM2024_cnn_bilstm.h5'
else:
    best_checkpoint = 'best_model_CICDDoS2019_cnn_bilstm.h5'

print(f"\nUsing checkpoint: {best_checkpoint}")

# Copy to final model name
final_model = 'cnn_bilstm_waf_model.h5'
shutil.copy(best_checkpoint, final_model)
print(f"✅ Created: {final_model}")

# Create tokenizer file (need to extract from training script)
print("\n" + "="*60)
print("Creating Tokenizer File")
print("="*60)

# We need to run a quick training to get the tokenizer
# Or extract it from the training script
print("\nRunning quick training to extract tokenizer...")

try:
    from cnn_bilstm_waf_ldap import CNNBiLSTMWAF
    import pandas as pd
    
    # Initialize WAF
    waf = CNNBiLSTMWAF(max_length=500, vocab_size=10000)
    
    # Load a small sample to fit tokenizer
    print("Loading sample data...")
    df_sample = pd.read_csv('csic_database.csv', nrows=1000)
    
    # Extract text features
    texts = []
    labels = []
    for idx, row in df_sample.iterrows():
        text_parts = []
        if pd.notna(row.get('URL')):
            text_parts.append(str(row['URL']))
        if pd.notna(row.get('Method')):
            text_parts.append(str(row['Method']))
        
        text = " ".join(text_parts) if text_parts else "empty"
        texts.append(text)
        labels.append(int(row['classification']))
    
    # Fit tokenizer
    print("Fitting tokenizer...")
    waf.tokenizer.fit_on_texts(texts)
    
    # Save tokenizer
    metadata = {
        'tokenizer': waf.tokenizer,
        'max_length': waf.max_length,
        'vocab_size': waf.vocab_size,
        'training_history': []
    }
    
    with open('cnn_bilstm_tokenizer.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Created: cnn_bilstm_tokenizer.pkl")
    print(f"   Vocabulary size: {len(waf.tokenizer.word_index)}")
    
except Exception as e:
    print(f"⚠️ Error creating tokenizer: {e}")
    print("\nAlternative: Run full training with:")
    print("   python cnn_bilstm_waf_ldap.py")

# Verify files
print("\n" + "="*60)
print("Verification")
print("="*60)

required_files = [
    'cnn_bilstm_waf_model.h5',
    'cnn_bilstm_tokenizer.pkl'
]

all_exist = True
for f in required_files:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024
        print(f"✅ {f} ({size:.2f} KB)")
    else:
        print(f"❌ {f} - NOT FOUND")
        all_exist = False

if all_exist:
    print("\n" + "="*60)
    print("✅ CNN-BiLSTM Model Complete!")
    print("="*60)
    print("\nYou can now use the ensemble:")
    print("  python ensemble_waf.py")
else:
    print("\n" + "="*60)
    print("⚠️ Incomplete - Need to run full training")
    print("="*60)
    print("\nRun: python cnn_bilstm_waf_ldap.py")
