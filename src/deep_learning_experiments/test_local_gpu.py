import torch
import numpy as np

def main():
    mps_available = torch.backends.mps.is_available()
    if mps_available:
        device = torch.device("mps")
        print("PyTorch will use the integrated GPU on your Apple Silicon chip using the 'mps' backend.")
    else:
        device = torch.device("cpu")
        print("MPS not detected. Defaulting to CPU.")
        
if __name__ == "__main__":
    main()