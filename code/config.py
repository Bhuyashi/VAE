import torch

class Config:
    # Dataset
    DATASET = "mnist"   # "mnist", "fashion_mnist"
    BATCH_SIZE = 64
    SEED = 31

    # Model 
    INPUT_DIM = 784
    H_DIM = 400
    Z_DIM = 20

    # Training
    LR_RATE = 1e-3  #5e-4  #3e-4  # Karpathy constant
    NUM_EPOCHS = 30

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    MODEL_SAVE_PATH = "./models/"
    DATA_PATH = "./data/"

    # Inference
    LOAD_PRETRAINED = True
    INFERENCE = True