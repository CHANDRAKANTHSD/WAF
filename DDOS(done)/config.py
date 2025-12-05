# Configuration file for DDoS Detection System

# Dataset paths - Update these to match your file locations
DATASETS = [
    # CICIDS2017
    {
        'path': r"Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        'name': "CICIDS2017-Friday"
    },
    {
        'path': r"Monday-WorkingHours.pcap_ISCX.csv",
        'name': "CICIDS2017-Monday"
    },
    {
        'path': r"Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        'name': "CICIDS2017-Thursday"
    },
    
    # CSE-CIC-IDS2018
    {
        'path': r"DDOS attack-HOIC.csv",
        'name': "CSE-CIC-IDS2018-HOIC"
    },
    {
        'path': r"DDOS attack-LOIC-UDP.csv",
        'name': "CSE-CIC-IDS2018-LOIC-UDP"
    },
    {
        'path': r"DDoS attacks-LOIC-HTTP.csv",
        'name': "CSE-CIC-IDS2018-LOIC-HTTP"
    },
    
    # TON_IoT
    {
        'path': r"ton_iot_network.csv",
        'name': "TON_IoT"
    }
]

# Model hyperparameters
LIGHTGBM_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 10,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

XGBOOST_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss'
}

# Feature selection
N_FEATURES = 30  # Number of top features to select

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Output paths
MODEL_OUTPUT_DIR = '.'
PLOT_OUTPUT_DIR = '.'
