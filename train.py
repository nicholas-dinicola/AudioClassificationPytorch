import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from urbansounddataset import UrbanSoundDataset
from cnn import CNNNetwork
import os

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "data/UrbanSound8K.csv"
AUDIO_DIR = "data/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            DEVICE)

    train_dataloader = create_data_loader(usd, BATCH_SIZE)

    # construct model and assign it to device
    cnn = CNNNetwork().to(DEVICE)
    print(cnn)

    # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, DEVICE, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "cnnnet.pth")
    print("Trained CNN net saved at cnnnet.pth")