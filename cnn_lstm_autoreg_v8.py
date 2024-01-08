"""Baseline linear model for wind power prediction."""

from datetime import datetime
from sklearn import model_selection
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import TemporalAttention, ResNetBlock, TemporalFeatureExtractor
import argparse
import joblib
import numpy as np
import os
import torch
import torch.nn as nn
import tqdm


#
# Helper functions not directly related to data
#


class AverageMetric(object):
    def __init__(self, scale=1.0):
        self.count = 0
        self.value = 0
        self.scale = scale

    def update(self, value, n):
        if n == 0:
            return
        self.value += n / (self.count + n) * (value * self.scale - self.value)
        self.count += n


def read_pickle(src):
    return joblib.load(src)


def write_pickle(model, dst):
    joblib.dump(model, dst)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Running on device %s" % device)
    return device


def kfold_chunk(n, n_splits, chunk_size, random_state=None):
    """
    Divides data using sklearn's cv split
    """
    n_chunks = n / chunk_size
    if n % chunk_size:
        n_chunks += 1

    def _chunk(k):
        return range(k * chunk_size, min((k + 1) * chunk_size, n))

    kf = model_selection.KFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    cv = []
    for t_chunk, s_chunk in kf.split(np.arange(n_chunks)):
        t = [i for k in t_chunk for i in _chunk(k)]
        s = [i for k in s_chunk for i in _chunk(k)]
        cv.append((t, s))
    return cv


def load_model(model_dir):
    checkpoint_file = os.path.join(model_dir, "checkpoint.pth.tar")
    assert os.path.exists(checkpoint_file), "No checkpoint found in %s" % model_dir

    params = read_pickle(os.path.join(model_dir, "params.pkl"))
    state_dict = torch.load(checkpoint_file)
    model = Model(**params["model"])
    model.load_state_dict(state_dict["model"])

    print("Previous training completed %i epochs" % state_dict["epoch"])
    return model, params


#
# Data preprocessing
#

NAM_SHIFT = np.array(
    [
        109.68943293843417,  # Pa
        128.74438657930364,  # m/s
        128.74438657930364,  # m/s
    ],
    dtype=np.float32,
)

NAM_SCALE = np.array(
    [
        73.87423140863105,  # Pa
        17.35445118879509,  # m/s
        17.35445118879509,  # m/s
    ],
    dtype=np.float32,
)

WIND_POWER_SHIFT = np.array([29.906836194977963], dtype=np.float32)
WIND_POWER_SCALE = np.array([38.9876854204175], dtype=np.float32)


def get_data(dfs, params):
    """Reads in data subject to data parameters."""

    index = dfs["X"].index
    time_horizon = params["data"]["horizon"]
    if time_horizon > 24:
        raise RuntimeError("longer time_horizon changes the dataset split and corrupts the val set")

    output = {}
    for k, v in dfs.items():
        idx = np.nonzero(index[: -time_horizon + 1].hour == 0)[0]
        idx = idx.reshape(-1, 1) + np.arange(0, time_horizon)
        output[k] = v.values[idx]

    output["X"] = output["X"].reshape(
        *output["X"].shape[:2],
        params["data"]["height"],
        params["data"]["width"],
        params["data"]["channels"],
    )

    # Include previous wind power output as part of the input features
    # Assuming 'y' contains the wind power data
    # Shift the wind power data to align with the input features
    output["prev_y"] = np.roll(output["y"], shift=1, axis=1)
    # Set the first time step of wind power to 0 as there's no previous data for the first time step
    output["prev_y"][:, 0, :] = 0


    print("_" * 50)
    print("features:", output["X"].shape)
    print("previous wind power:", output["prev_y"].shape)
    print("targets:", output["y"].shape)
    print("weights:", output["w"].shape)
    print("_" * 50)

    # Scale targets
    output["y"] -= WIND_POWER_SHIFT
    output["y"] /= WIND_POWER_SCALE

    # Scale prev_y 
    output["prev_y"] -= WIND_POWER_SHIFT
    output["prev_y"] /= WIND_POWER_SCALE

    # NOTE: We defer scaling the X features until inside the training loop
    # because scaling them here would result in a type conversion from uint8 to
    # float32 and subsequent increase in memory usage

    print("Data read and preprocessed")

    return output


def split_data(data, params):
    """Pass data to split into train, val sets."""
    n = data["X"].shape[0]

    chunk_size = 2
    t, s = kfold_chunk(n, 5, chunk_size, random_state=0)[0]
    print("Train samples:", len(t))
    print("Val samples:", len(s))

    train_data, val_data = [], []
    for d in ["X", "prev_y", "y", "w"]:
        val_data.append(torch.from_numpy(data[d][s]))
        train_data.append(torch.from_numpy(data[d][t]))

    return train_data, val_data


class Model(nn.Module):
    def __init__(self, in_dim, out_dim, lstm_hidden_dim, conv_channels):
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dropout = nn.Dropout(p=0.3)
        
        # LSTM for temporal feature extraction
        self.NAM_temporal_feature_extractor = TemporalFeatureExtractor(input_size=32 * 3 * 3, hidden_dim=lstm_hidden_dim)

        # Wind power features
        self.wind_power_feature_extractor = TemporalFeatureExtractor(out_dim, lstm_hidden_dim)

        # Fully connected layer for prediction
        self.fc_combine = nn.Linear(lstm_hidden_dim + lstm_hidden_dim, out_dim)

        self.register_buffer("features_shift", torch.from_numpy(NAM_SHIFT))
        self.register_buffer("features_scale", torch.from_numpy(NAM_SCALE))

    def preprocess_features(self, features):
        features = (features.float() - self.features_shift) / self.features_scale
        features = features.permute(0, 4, 1, 2, 3).contiguous()  # NTHWC --> NCTHW
        return features

    def forward(self, features, prev_y):
        features = self.preprocess_features(features)
        b, c, t, h, w = features.shape
        features = features.transpose(1, 2)  # b, t, c, h, w
        # Spatial features
        c_out = [self.cnn(features[:, ti, :, :, :]).view(b, -1) for ti in range(t)]
        c_out = torch.stack(c_out, dim=1)
    
        # Temporal features
        x_out = self.NAM_temporal_feature_extractor(c_out)
        x_out = self.dropout(x_out)    
        
        # Wind power temporal features
        prev_y = prev_y.to(torch.float32)
        wind_power_features = self.wind_power_feature_extractor(prev_y)

        # Combine features by concatenation -> can use feature fusion as well
        combined_features = torch.cat((x_out, wind_power_features), dim=-1)

        # Output layer
        out = self.fc_combine(combined_features)
        return out

    def metrics(self):
        return {
            "loss": AverageMetric(),
            "mse": AverageMetric(scale=WIND_POWER_SCALE),
            "mae": AverageMetric(scale=WIND_POWER_SCALE),
        }

    def loss(self, features, prev_y, targets, weights, metrics=None):
        """Makes prediction of targets based on features. Returns MSE loss,
        using weights to mask out points that should be ignored (due to windfarm
        not yet being operational or curtailment).

        Arguments:
        features: [b, t, h, w, c]
        targets:  [b, t, n]
        weights:  [b, t, n]

        where
        b: batch_size
        c: input channels
        t: time horizon
        h: spatial feature height
        w: spatial feature width
        n: number of wind farms

        Returns a scalar representing the average loss of the batch
        """
        pred = self(features, prev_y)
        mse = ((targets - pred).square() * weights).sum() / weights.sum()
        mae = ((targets - pred).abs() * weights).sum() / weights.sum()

        if metrics is not None:
            n = weights.sum().item()  # n included samples
            metrics["loss"].update(mse.item(), n)
            metrics["mse"].update(mse.item(), n)
            metrics["mae"].update(mae.item(), n)

        return mse


def get_params():
    params = {
        "train_epochs": 80,
        "optimizer": {"lr": 1e-3}, 
        "loader": {"batch_size": 32},
        "model": {
            "in_dim": 14 * 14 * 3,  # h * w * c, the dimensions of the features
            "out_dim": 237,  # the number of farms
            "lstm_hidden_dim": 128,
            "conv_channels": [3, 16, 32]
        },
        "data": {
            "horizon": 24,  # n hours to predict
            "channels": 3,  # before preprocessing (add icing feats)
            "height": 14,  # height of the input
            "width": 14,  # width of the input
        },
    }
    return params

def train_model(train_data, model, model_dir, params, val_data=None):
    print("Training %s" % model_dir)
    device = get_device()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), **params["optimizer"])

    train_loader = DataLoader(
        torch.utils.data.TensorDataset(*train_data),
        **params["loader"],
        shuffle=True,
        pin_memory=True,
    )
    train_writer = SummaryWriter(model_dir + "/train")

    val_loader = None
    val_writer = None
    if val_data is not None:
        val_loader = DataLoader(
            torch.utils.data.TensorDataset(*val_data),
            **params["loader"],
            pin_memory=True,
        )
        val_writer = SummaryWriter(model_dir + "/eval")

    print(model)
    print(
        "Number of parameters: %i" % sum(np.prod(x.shape) for x in model.parameters())
    )
    write_pickle(params, model_dir + "/params.pkl")
    checkpoint_file = model_dir + "/checkpoint.pth.tar"

    if os.path.exists(checkpoint_file):
        state_dict = torch.load(checkpoint_file)
        start_epoch = state_dict["epoch"]
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        print("Previous training completed %i epochs" % start_epoch)
    else:
        start_epoch = 0

    for epoch in range(start_epoch, params["train_epochs"]):
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            epoch + 1,
            train_writer,
            device,
            leave=val_loader is None,
        )
        if val_loader is not None:
            evaluate(
                val_loader,
                model,
                epoch + 1,
                val_writer,
                device,
            )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
        }
        torch.save(checkpoint, checkpoint_file)

    write_pickle("", model_dir + "/DONE")


def train_one_epoch(
    loader,
    model,
    optimizer,
    epoch,
    writer,
    device,
    leave=False,
):
    model.train()
    metrics = model.metrics()
    gen = tqdm.tqdm(loader, leave=leave)
    for features, prev_y, targets, weights in gen:
        features = features.to(device, non_blocking=True)
        prev_y = prev_y.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)

        optimizer.zero_grad()
        loss = model.loss(features, prev_y, targets, weights, metrics=metrics)
        loss.backward()
        optimizer.step()

        gen.set_description(
            "Epoch %d, train loss: %.3f" % (epoch, metrics["loss"].value)
        )

    for k, metric in metrics.items():
        writer.add_scalar(k, metric.value, epoch)
    writer.flush()


def evaluate(loader, model, epoch, writer, device, leave=False):
    model.eval()
    metrics = model.metrics()
    gen = tqdm.tqdm(loader, leave=leave)
    with torch.no_grad():
        for features, prev_y, targets, weights in gen:
            features = features.to(device, non_blocking=True)
            prev_y = prev_y.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)

            model.loss(features, prev_y, targets, weights, metrics=metrics)

            gen.set_description(
                "Epoch %d, val loss: %.3f" % (epoch, metrics["loss"].value)
            )

    for k, metric in metrics.items():
        writer.add_scalar(k, metric.value, epoch)
    writer.flush()


if __name__ == "__main__":
    curr_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.path.join(curr_dir, "models", "linear_v1"),
    )
    parser.add_argument("--model-run", default=datetime.now().strftime("%Y%m%d%H%M%S"))
    args = parser.parse_args()

    params = get_params()
    model_dir = os.path.join(args.model_path, args.model_run)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print("Model dir: ", model_dir)

    data_dir = os.path.join(curr_dir, "dfs14.pkl")
    print("Reading data...")
    dfs = read_pickle(data_dir)
    print("Done reading data")

    data = get_data(dfs, params)
    train_data, val_data = split_data(data, params)
    model = Model(**params["model"])
    train_model(train_data, model, model_dir, params, val_data=val_data)