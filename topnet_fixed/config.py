"""
TopNet training configuration (fixed Transformer version).

Dataset: Baidu Cloud (code: 4zf9) / Dropbox
SOPA encoder: Baidu Cloud (code: 1x3n) / Dropbox
"""

# ---- Paths ----
DATA_DIR = './data/data'
BG_DIR   = f'{DATA_DIR}/bg'
FG_DIR   = f'{DATA_DIR}/fg'
MASK_DIR = f'{DATA_DIR}/mask'
TRAIN_JSON = f'{DATA_DIR}/train_pair_new.json'
TEST_JSON  = f'{DATA_DIR}/test_pair_new.json'
SOPA_WEIGHT = f'{DATA_DIR}/SOPA.pth.tar'

CHECKPOINT_DIR = './checkpoints'
LOG_DIR = './logs'

# ---- Training ----
EPOCHS       = 25
BATCH_SIZE   = 8        # ObPlaNet is large (113M params)
INPUT_SIZE   = 256
NUM_WORKERS  = 4

LR           = 1e-5
WEIGHT_DECAY = 1e-4
LR_DECAY     = 0.5      # multiplicative factor
DECAY_EVERY  = 2        # decay LR every N epochs

# ---- Loss ----
IGNORE_INDEX = 255       # unlabeled pixels in target

# ---- Logging & Saving ----
SAVE_FREQ    = 5         # save checkpoint every N epochs
PRINT_FREQ   = 20        # print loss every N batches
LOG_TO_TENSORBOARD = True
