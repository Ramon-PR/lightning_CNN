# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 3



# Dataset
DATA_DIR = r"C:\Users\keris\Desktop\Postdoc"
FILES_TRAIN = ["BalderRIR.mat", "FrejaRIR.mat"]
FILES_VAL = ["MuninRIR.mat"]
PARAM_DOWNSAMPLING = dict(ratio_t=1, ratio_x=0.5, kernel=[32, 32], stride=[32, 32])
NUM_WORKERS = 4


# Model
N_CHANNELS, HIN, WIN, HOUT, WOUT = 1, 32, 32, 32, 32



# Compute related
ACCELERATOR = "cpu"
DEVICES = "auto"
PRECISION = 32