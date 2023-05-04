# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 500
NUM_EPOCHS = 10

# Dataset
DATA_DIR = r"C:\Users\keris\Desktop\Postdoc"
# DATA_DIR = r"/scratch/ramonpr/3NoiseModelling/"
FILES_TRAIN = ["BalderRIR.mat", "FrejaRIR.mat"]
FILES_VAL = ["MuninRIR.mat"]
PARAM_DOWNSAMPLING = dict(ratio_t=1, ratio_x=0.5, kernel=[32, 32], stride=[32, 32])
NUM_WORKERS = 4

# Model
N_CHANNELS, HIN, WIN, HOUT, WOUT = 1, 32, 32, 32, 32
N_FILTERS = [5, 5, 5]
CONV_KERN = [3, 5, 5]
CONV_PAD  = [1, 2, 2]
CONV_STR  = [1, 1, 1]

MODEL_NAME = "CNN_basic"
SAVE_NAME = f"{MODEL_NAME}_{N_FILTERS[0]}_{N_FILTERS[1]}_2"
VAR2MONITOR = "train_loss"


# Compute related
ACCELERATOR = "cpu"
DEVICES = "auto"
PRECISION = 32

