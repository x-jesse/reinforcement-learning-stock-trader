import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class FeedForward(nn.Module):
    def __init__(self, input_dim=128, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)




def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    msft = yf.Ticker("MSFT")

    # get all stock info
    msft.info

    # get historical market data
    hist = msft.history(period="1mo")

    # show meta information about the history (requires history() to be called first)
    print(msft.history_metadata)


if __name__ == "__main__":
    main()
