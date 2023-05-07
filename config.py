# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 500
NUM_EPOCHS = 500

# Dataset
# DATA_DIR = r"C:\Users\keris\Desktop\Postdoc"
DATA_DIR = r"/scratch/ramonpr/3NoiseModelling/"
FILES_TRAIN = ["BalderRIR.mat", "FrejaRIR.mat"]
FILES_VAL = ["MuninRIR.mat"]
PARAM_DOWNSAMPLING = dict(ratio_t=1, ratio_x=0.5, kernel=[32, 32], stride=[32, 32])
NUM_WORKERS = 4

# Model
N_CHANNELS, HIN, WIN, HOUT, WOUT = 3, 32, 32, 32, 32
N_FILTERS = [10]
CONV_KERN = [ 3]
CONV_PAD  = [ 1]
CONV_STR  = [ 1]

N_FILTERS_B2 = [10]
CONV_KERN_B2 = [ 5]
CONV_PAD_B2  = [ 2]


MODEL_NAME = "CNN_2B"
SAVE_NAME = f"{MODEL_NAME}_L_{len(N_FILTERS)}_K_{CONV_KERN[0]}_F_{N_FILTERS[0]}_C_{N_CHANNELS}"
# SAVE_NAME = f"{MODEL_NAME}_L_{len(N_FILTERS)}_K_{CONV_KERN[0]}-{CONV_KERN[1]}_F_{N_FILTERS[0]}"
VAR2MONITOR = "train_loss"


# Compute related
ACCELERATOR = "cpu"
DEVICES = "auto"
PRECISION = 32

