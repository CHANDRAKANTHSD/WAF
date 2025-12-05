"""
CNN-BiLSTM with Attention Mechanism for LDAP Injection Detection
Deep learning WAF with character-level sequence processing
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class AttentionLayer(layers.Layer):
    """Custom Attention Layer"""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                shape=(input_shape[-1], input_shape[-1]),
                                initializer='glorot_uniform',
                                trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                shape=(input_shape[-1],),
                                initializer='zeros',
                                trainable=True)
        self.u = self.add_weight(name='attention_u',
                                shape=(input_shape[-1],),
                                initializer='glorot_uniform',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # x shape: (batch_size, time_steps, features)
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        a = tf.nn.softmax(ait, axis=1)
        a = tf.expand_dims(a, -1)
        weighted_input = x * a
        output = tf.reduce_sum(weighted_input, axis=1)
        return output, a
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()


class CNNBiLSTMWAF:
    def __init__(self, max_length=500, vocab_size=10000):
        self.model = None
        self.tokenizer = Tokenizer(char_level=True, oov_token='<OOV>')
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.training_history = []
        self.attention_weights = None
        
    def preprocess_text(self, texts):
        """Convert text to sequences"""
        # Fit tokenizer if not fitted
        if not self.tokenizer.word_index:
            self.tokenizer.fit_on_texts(texts)
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        return padded
    
    def extract_text_features(self, df, dataset_name):
        """Extract text features from different datasets"""
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} Dataset")
        print(f"{'='*60}")
        print(f"Original shape: {df.shape}")
        
        texts = []
        labels = []
        
        if dataset_name == "CICDDoS2019":
            # Create text representation from network features
            for idx, row in df.iterrows():
                # Combine key features into text
                text_parts = []
                for col in df.columns[:20]:  # Use first 20 features
                    if col != 'Label':
                        text_parts.append(f"{col}:{row[col]}")
                texts.append(" ".join(text_parts))
                labels.append(0 if row['Label'] == 'NetBIOS' else 1)
        
        elif dataset_name == "LSNM2024":
            # Use HTTP/DNS query information
            for idx, row in df.iterrows():
                text_parts = []
                
                # Add protocol info
                if pd.notna(row.get('Protocol')):
                    text_parts.append(f"proto:{row['Protocol']}")
                
                # Add HTTP info
                if pd.notna(row.get('HTTP Request URI')):
                    text_parts.append(str(row['HTTP Request URI']))
                if pd.notna(row.get('HTTP Request Method')):
                    text_parts.append(str(row['HTTP Request Method']))
                
                # Add DNS info
                if pd.notna(row.get('DNS Query Name')):
                    text_parts.append(str(row['DNS Query Name']))
                
                # Add Info field
                if pd.notna(row.get('Info')):
                    text_parts.append(str(row['Info']))
                
                text = " ".join(text_parts) if text_parts else "empty"
                texts.append(text)
                labels.append(0 if row['label'] == 'normal' else 1)
        
        elif dataset_name == "CSIC":
            # Use URL and HTTP headers
            for idx, row in df.iterrows():
                text_parts = []
                
                # URL is the main feature
                if pd.notna(row.get('URL')):
                    text_parts.append(str(row['URL']))
                
                # Add method
                if pd.notna(row.get('Method')):
                    text_parts.append(str(row['Method']))
                
                # Add content
                if pd.notna(row.get('content')):
                    text_parts.append(str(row['content']))
                
                text = " ".join(text_parts) if text_parts else "empty"
                texts.append(text)
                labels.append(int(row['classification']))
        
        labels = np.array(labels)
        print(f"Texts extracted: {len(texts)}")
        print(f"Class distribution: {pd.Series(labels).value_counts().to_dict()}")
        
        return texts, labels
    
    def build_model(self):
        """Build CNN-BiLSTM with Attention architecture"""
        print("\nBuilding CNN-BiLSTM with Attention model...")
        
        # Input layer
        input_layer = layers.Input(shape=(self.max_length,))
        
        # Embedding layer
        embedding = layers.Embedding(
            input_dim=min(len(self.tokenizer.word_index) + 1, self.vocab_size),
            output_dim=128,
            input_length=self.max_length,
            mask_zero=True
        )(input_layer)
        
        # Dual CNN channels
        # Channel 1: kernel size 3
        conv1 = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(embedding)
        conv1 = layers.MaxPooling1D(pool_size=2)(conv1)
        conv1 = layers.Dropout(0.3)(conv1)
        
        # Channel 2: kernel size 5
        conv2 = layers.Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(embedding)
        conv2 = layers.MaxPooling1D(pool_size=2)(conv2)
        conv2 = layers.Dropout(0.3)(conv2)
        
        # Concatenate CNN channels
        concat = layers.Concatenate()([conv1, conv2])
        
        # BiLSTM layer
        bilstm = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
        )(concat)
        
        # Attention layer
        attention_output, attention_weights = AttentionLayer()(bilstm)
        
        # Dense layers
        dense1 = layers.Dense(64, activation='relu')(attention_output)
        dense1 = layers.Dropout(0.5)(dense1)
        dense2 = layers.Dense(32, activation='relu')(dense1)
        dense2 = layers.Dropout(0.3)(dense2)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid')(dense2)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
        )
        
        print("\nModel Architecture:")
        model.summary()
        
        self.model = model
        return model
    
    def train_on_dataset(self, texts, labels, dataset_name, epochs=50, batch_size=32):
        """Train model on a single dataset"""
        print(f"\n{'='*60}")
        print(f"Training on {dataset_name}")
        print(f"{'='*60}")
        
        # Preprocess texts
        X = self.preprocess_text(texts)
        y = labels
        
        # Split data: 70% train, 15% val, 15% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 ≈ 0.15/0.85
        )
        
        print(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")
        print(f"Train labels: {pd.Series(y_train).value_counts().to_dict()}")
        
        # Build model if not exists
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'best_model_{dataset_name}_cnn_bilstm.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train model
        print("\nTraining CNN-BiLSTM model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        self.plot_training_history(history, dataset_name)
        
        # Evaluate
        results = self.evaluate(X_test, y_test, dataset_name)
        
        # Store training history
        self.training_history.append({
            'dataset': dataset_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'results': results,
            'epochs_trained': len(history.history['loss'])
        })
        
        return results
    
    def plot_training_history(self, history, dataset_name):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Training History - {dataset_name}')
        plt.tight_layout()
        plt.savefig(f'training_history_{dataset_name}_cnn_bilstm.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training history saved: training_history_{dataset_name}_cnn_bilstm.png")
    
    def evaluate(self, X_test, y_test, dataset_name):
        """Evaluate model performance"""
        print(f"\n{'='*60}")
        print(f"Evaluation Results for {dataset_name}")
        print(f"{'='*60}")
        
        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # False Positive Rate
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"FPR:       {fpr:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=['Benign', 'Attack'],
                   yticklabels=['Benign', 'Attack'])
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{dataset_name}_cnn_bilstm.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved: confusion_matrix_{dataset_name}_cnn_bilstm.png")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'fpr': fpr,
            'confusion_matrix': cm.tolist()
        }
    
    def visualize_attention(self, sample_texts, sample_labels, dataset_name, num_samples=5):
        """Visualize attention weights"""
        print(f"\nVisualizing attention weights for {dataset_name}...")
        
        # Create attention model
        attention_model = Model(
            inputs=self.model.input,
            outputs=[self.model.output, self.model.layers[-5].output]  # Get attention layer output
        )
        
        # Get samples
        X_sample = self.preprocess_text(sample_texts[:num_samples])
        
        try:
            predictions, attention_weights = attention_model.predict(X_sample, verbose=0)
            
            fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
            if num_samples == 1:
                axes = [axes]
            
            for i in range(num_samples):
                # Get attention weights for this sample
                attn = attention_weights[i].flatten()
                
                # Plot
                axes[i].bar(range(len(attn)), attn)
                axes[i].set_title(f'Sample {i+1} - Label: {sample_labels[i]} - Pred: {predictions[i][0]:.4f}')
                axes[i].set_xlabel('Sequence Position')
                axes[i].set_ylabel('Attention Weight')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'attention_weights_{dataset_name}_cnn_bilstm.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Attention visualization saved: attention_weights_{dataset_name}_cnn_bilstm.png")
            
        except Exception as e:
            print(f"Could not visualize attention: {e}")
    
    def predict_realtime(self, query_text):
        """Real-time prediction for incoming queries"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Preprocess
        X = self.preprocess_text([query_text])
        
        # Predict
        prediction_proba = self.model.predict(X, verbose=0)[0][0]
        prediction = int(prediction_proba > 0.5)
        
        return {
            'is_attack': bool(prediction),
            'attack_probability': float(prediction_proba),
            'benign_probability': float(1 - prediction_proba)
        }
    
    def save_model(self, model_filename='cnn_bilstm_waf_model.h5', 
                   tokenizer_filename='cnn_bilstm_tokenizer.pkl'):
        """Save trained model and tokenizer"""
        # Save Keras model
        self.model.save(model_filename)
        print(f"\nModel saved: {model_filename}")
        print(f"File size: {os.path.getsize(model_filename) / 1024:.2f} KB")
        
        # Save tokenizer and metadata
        metadata = {
            'tokenizer': self.tokenizer,
            'max_length': self.max_length,
            'vocab_size': self.vocab_size,
            'training_history': self.training_history
        }
        
        with open(tokenizer_filename, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Tokenizer saved: {tokenizer_filename}")
        print(f"File size: {os.path.getsize(tokenizer_filename) / 1024:.2f} KB")
    
    @staticmethod
    def load_model(model_filename='cnn_bilstm_waf_model.h5',
                   tokenizer_filename='cnn_bilstm_tokenizer.pkl'):
        """Load trained model and tokenizer"""
        waf = CNNBiLSTMWAF()
        
        # Load model with custom objects
        waf.model = keras.models.load_model(
            model_filename,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        
        # Load tokenizer and metadata
        with open(tokenizer_filename, 'rb') as f:
            metadata = pickle.load(f)
        
        waf.tokenizer = metadata['tokenizer']
        waf.max_length = metadata['max_length']
        waf.vocab_size = metadata['vocab_size']
        waf.training_history = metadata['training_history']
        
        return waf


def main():
    print("="*60)
    print("CNN-BiLSTM with Attention WAF for LDAP Injection Detection")
    print("="*60)
    
    waf = CNNBiLSTMWAF(max_length=500, vocab_size=10000)
    all_results = {}
    
    # Dataset 1: CICDDoS2019 LDAP
    try:
        print("\n\n" + "="*60)
        print("DATASET 1: CICDDoS2019 LDAP")
        print("="*60)
        
        df_train = pd.read_parquet('cicddos_2019/LDAP-training.parquet')
        df_test = pd.read_parquet('cicddos_2019/LDAP-testing.parquet')
        df_cicddos = pd.concat([df_train, df_test], ignore_index=True)
        
        texts, labels = waf.extract_text_features(df_cicddos, "CICDDoS2019")
        results = waf.train_on_dataset(texts, labels, "CICDDoS2019", epochs=10, batch_size=128)
        all_results['CICDDoS2019'] = results
        
        # Visualize attention
        waf.visualize_attention(texts[:5], labels[:5], "CICDDoS2019")
        
    except Exception as e:
        print(f"Error processing CICDDoS2019: {e}")
        import traceback
        traceback.print_exc()
    
    # Dataset 2: LSNM2024
    try:
        print("\n\n" + "="*60)
        print("DATASET 2: LSNM2024")
        print("="*60)
        
        # Load all LSNM2024 files with sample to avoid memory issues
        df_benign = pd.read_csv('LSNM2024 Dataset/Benign/normal_data.csv', nrows=5000)
        df_fuzzing = pd.read_csv('LSNM2024 Dataset/Malicious/Fuzzing/Fuzzing.csv', nrows=5000)
        df_sql1 = pd.read_csv('LSNM2024 Dataset/Malicious/SQL injection/SQL Injection.csv', nrows=5000)
        df_sql2 = pd.read_csv('LSNM2024 Dataset/Malicious/SQL injection/SQL-Injection2.csv', nrows=5000)
        df_sql3 = pd.read_csv('LSNM2024 Dataset/Malicious/SQL injection/SQLinjection-UPDATED.csv', nrows=5000)
        
        df_lsnm = pd.concat([df_benign, df_fuzzing, df_sql1, df_sql2, df_sql3], ignore_index=True)
        
        texts, labels = waf.extract_text_features(df_lsnm, "LSNM2024")
        results = waf.train_on_dataset(texts, labels, "LSNM2024", epochs=10, batch_size=128)
        all_results['LSNM2024'] = results
        
        # Visualize attention
        waf.visualize_attention(texts[:5], labels[:5], "LSNM2024")
        
    except Exception as e:
        print(f"Error processing LSNM2024: {e}")
        import traceback
        traceback.print_exc()
    
    # Dataset 3: CSIC
    try:
        print("\n\n" + "="*60)
        print("DATASET 3: CSIC Database")
        print("="*60)
        
        df_csic = pd.read_csv('csic_database.csv')
        
        texts, labels = waf.extract_text_features(df_csic, "CSIC")
        results = waf.train_on_dataset(texts, labels, "CSIC", epochs=10, batch_size=128)
        all_results['CSIC'] = results
        
        # Visualize attention
        waf.visualize_attention(texts[:5], labels[:5], "CSIC")
        
    except Exception as e:
        print(f"Error processing CSIC: {e}")
        import traceback
        traceback.print_exc()
    
    # Save model
    waf.save_model('cnn_bilstm_waf_model.h5', 'cnn_bilstm_tokenizer.pkl')
    
    # Print summary
    print("\n\n" + "="*60)
    print("TRAINING SUMMARY - CNN-BiLSTM WAF")
    print("="*60)
    
    for dataset, results in all_results.items():
        print(f"\n{dataset}:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1_score']:.4f}")
        print(f"  FPR:       {results['fpr']:.4f}")
    
    # Demo real-time prediction
    print("\n\n" + "="*60)
    print("REAL-TIME PREDICTION DEMO")
    print("="*60)
    
    # Example queries
    test_queries = [
        "GET /index.html HTTP/1.1",
        "GET /admin.php?id=1' OR '1'='1 HTTP/1.1",
        "ldap://server/cn=*)(uid=*))(|(cn=*"
    ]
    
    for query in test_queries:
        prediction = waf.predict_realtime(query)
        print(f"\nQuery: {query}")
        print(f"  Is Attack: {prediction['is_attack']}")
        print(f"  Attack Probability: {prediction['attack_probability']:.4f}")
    
    print("\n" + "="*60)
    print("CNN-BiLSTM WAF Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
