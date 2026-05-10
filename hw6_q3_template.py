import os
import torch
import numpy as np
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import time
import json
import matplotlib.pyplot as plt

try:
    from thop import clever_format
    from thop import profile
    THOP_AVAILABLE = True
except Exception:
    THOP_AVAILABLE = False

from sklearn.metrics import accuracy_score

# data dir is the mnist folder
"""
Load data to pytorch data loader
"""
def get_dataloader(data_dir, batch_size=32):
    """
    :param data_dir: folder directory of the data
    :param batch_size: batch size
    :return: a list of [trainloader, testloader]
    """

    def find_file(base_dir, candidates):
        for name in candidates:
            path = os.path.join(base_dir, name)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"None of {candidates} found in {base_dir}")

    def get_split_loader(split):
        # support both `name.npy` and `name(1).npy`
        data_path = find_file(data_dir, [f"{split}_data_merged.npy", f"{split}_data_merged(1).npy"])
        label_path = find_file(data_dir, [f"{split}_labels_merged.npy", f"{split}_labels_merged(1).npy"])

        data_images = np.load(data_path)

        # labels[:, 0]: figures of the first image
        # labels[:, 1]: figures of the second image
        # labels[:, 2]: figures of the mul of the previous two
        labels = np.load(label_path)  # (B, 3)
        labels[:, 2] = labels[:, 0] * labels[:, 1]  # do the multiplication

        images1 = data_images[:, 0, :]  # (B, 784), 0-1 scale flattened image
        images2 = data_images[:, 1, :]  # (B, 784), 0-1 scale flattened image
        images1 = images1.reshape([-1, 1, 28, 28])
        images2 = images2.reshape([-1, 1, 28, 28])

        torch_dataset = utils.TensorDataset(
            torch.from_numpy(images1).float(),
            torch.from_numpy(images2).float(),
            torch.from_numpy(labels).long()
        )

        if split == "train":
            torch_loader = utils.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
        else:
            torch_loader = utils.DataLoader(torch_dataset, batch_size=batch_size)

        return torch_loader

    return get_split_loader("train"), get_split_loader("val"), get_split_loader("test")


def create_model(fusion):
    if fusion == "early_fusion":
        return EarlyFusion()
    elif fusion == "late_fusion":
        return LateFusion()
    else:
        raise ValueError("Invalid fusion type")


def train(model, device, train_loader, val_loader, test_loader, optimizer, epoch):
    model.train()

    train_loss = 0

    loss_fn = nn.CrossEntropyLoss()

    for batch_idx, (image1, image2, merged_labels) in enumerate(train_loader):
        image1, image2, mul_labels = image1.to(device), image2.to(device), merged_labels[:, 2].to(device)

        optimizer.zero_grad()

        output = model(image1, image2)

        loss = loss_fn(output, mul_labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:  # Print loss every 100 batch
            print('Train Epoch: {}\tLoss: {:.6f}'.format(
                epoch, loss.item()))

        train_loss += loss.item()

    train_loss /= len(train_loader)

    val_loss, val_acc = test(model, device, val_loader)
    _, train_acc = test(model, device, train_loader)
    test_loss, test_acc = test(model, device, test_loader)

    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc


def test(model, device, torch_loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    epoch_loss = 0
    gt_labels = []
    pred_labels = []

    with torch.no_grad():
        for image1, image2, merged_labels in torch_loader:
            image1, image2, mul_labels = image1.to(device), image2.to(device), merged_labels[:, 2].to(device)
            output = model(image1, image2)
            loss = loss_fn(output, mul_labels)
            epoch_loss += loss.item()

            gt_labels.append(mul_labels.detach().cpu().numpy())
            pred_labels.append(np.argmax(output.detach().cpu().numpy(), axis=1))

    epoch_loss /= len(torch_loader)
    gt_labels = np.concatenate(gt_labels)
    pred_labels = np.concatenate(pred_labels)

    return epoch_loss, accuracy_score(gt_labels, pred_labels)


def main(fusion, optimizer_type):
    seed = 42
    """
    Fix the random seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    """
    Alternatively, you can also use the following code to select the device
    Init the device based on your hardware:

    1.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # <- if your device support cuda

    2.
    device = torch.device("cpu") # <- if you want to force the device to use cpu

    3.
    device = "mps" if torch.backends.mps.is_available() else "cpu" # <- if you are running on mac with m chip
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    """
    Training settings
    """
    NumEpochs = 10
    batch_size = 32
    os.makedirs(f"{fusion}_{optimizer_type}_report", exist_ok=True)  # make relative path
    output_dir = f"{fusion}_{optimizer_type}_report"

    model = create_model(fusion=fusion).to(device)

    # optimizer selection
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.999, 0.999))
    elif optimizer_type == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, centered=False, weight_decay=0.001)
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.99, 0.999), weight_decay=0.01)
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")

    # use mnist_data subfolder if it exists, otherwise use the current folder
    data_dir = "mnist_data" if os.path.isdir("mnist_data") else "."
    train_loader, val_loader, test_loader = get_dataloader(data_dir=data_dir, batch_size=batch_size)

    best_val_acc = 0

    # historical performance
    train_losses = []
    val_losses = []
    test_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []

    for epoch in range(NumEpochs):
        print("############# epoch: ", epoch)

        train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = train(
            model, device, train_loader, val_loader, test_loader, optimizer, epoch
        )

        print('\nTrain loss: {:.6f}, acc: {:.6f}\n'.format(train_loss, train_acc))
        print('\nVal loss:   {:.6f}, acc: {:.6f}\n'.format(val_loss, val_acc))
        print('\nTest loss:  {:.6f}, acc: {:.6f}\n'.format(test_loss, test_acc))
        print("############# End of epoch: ", epoch)
        print()

        # record epoch metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        np.save(os.path.join(output_dir, "train_losses.npy"), train_losses)
        np.save(os.path.join(output_dir, "train_accuracies.npy"), train_accuracies)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        np.save(os.path.join(output_dir, "val_losses.npy"), val_losses)
        np.save(os.path.join(output_dir, "val_accuracies.npy"), val_accuracies)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        np.save(os.path.join(output_dir, "test_losses.npy"), test_losses)
        np.save(os.path.join(output_dir, "test_accuracies.npy"), test_accuracies)

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss
                },
                os.path.join(output_dir, "mnist_model.pt"))

        """
        Diagnostic per-run plots (not the report plot)
        """
        plt.plot(range(epoch + 1), train_losses, label='train loss')
        plt.plot(range(epoch + 1), val_losses, label='val loss')
        plt.plot(range(epoch + 1), test_losses, label='test loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "train_val_loss.png"))
        plt.close()

        plt.plot(range(epoch + 1), train_accuracies, label='train acc')
        plt.plot(range(epoch + 1), val_accuracies, label='val acc')
        plt.plot(range(epoch + 1), test_accuracies, label='test acc')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "train_val_acc.png"))
        plt.close()

    # final test eval using best (val-selected) checkpoint
    best_cp = torch.load(os.path.join(output_dir, "mnist_model.pt"))
    model.load_state_dict(best_cp["state_dict"])
    test_loss, test_acc = test(model, device, test_loader)
    print("Best epoch: {}, Test loss:  {:.6f}, Test acc:  {:.6f}\n".format(best_cp["epoch"], test_loss, test_acc))
    with open(os.path.join(output_dir, "test_summary.json"), "w") as f:
        json.dump({"test_loss": test_loss, "test_acc": test_acc,
                   "val_acc": best_cp["val_acc"], "val_loss": best_cp["val_loss"],
                   "epoch": best_cp["epoch"],
                   "final_train_acc": train_accuracies[-1],
                   "final_test_acc": test_accuracies[-1]}, f, indent=4)

    return {
        "fusion": fusion,
        "optimizer": optimizer_type,
        "final_train_acc": train_accuracies[-1],
        "final_test_acc": test_accuracies[-1],
        "best_val_epoch": best_cp["epoch"],
        "best_val_acc": best_cp["val_acc"],
        "best_ckpt_test_acc": test_acc,
    }


# Early Fusion: place the two images side-by-side along the width axis,
# then run a single CNN over the fused (1, 28, 56) tensor (Table 2 of HW6).
class EarlyFusion(nn.Module):
    _shape_check_done = False

    def __init__(self):
        super(EarlyFusion, self).__init__()
        # input after side-by-side concat: (B, 1, 28, 56)
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.25)
        # after pool: (B, 4, 12, 26) -> flatten 4*12*26 = 1248
        self.fc1 = nn.Linear(1248, 128)
        self.fc2 = nn.Linear(128, 82)

    def forward(self, x1, x2):
        # x1, x2: (B, 1, 28, 28)
        x = torch.cat((x1, x2), dim=3)   # (B, 1, 28, 56)  side-by-side
        x = F.relu(self.conv1(x))        # (B, 2, 26, 54)
        x = F.relu(self.conv2(x))        # (B, 4, 24, 52)
        x = self.pool(x)                 # (B, 4, 12, 26)
        x = self.dropout(x)
        x = torch.flatten(x, 1)          # (B, 1248)
        x = F.relu(self.fc1(x))          # (B, 128)
        x = self.fc2(x)                  # (B, 82)

        # one-time runtime shape check (prints from rank 0 the first time
        # any forward pass runs, then disables itself for the rest of training)
        if not EarlyFusion._shape_check_done:
            EarlyFusion._shape_check_done = True
            with torch.no_grad():
                t1 = torch.zeros(1, 1, 28, 28, device=x1.device)
                t2 = torch.zeros(1, 1, 28, 28, device=x2.device)
                cat = torch.cat((t1, t2), dim=3)
                a = F.relu(self.conv1(cat))
                b = F.relu(self.conv2(a))
                c = self.pool(b)
                print("[EarlyFusion shape check]")
                print(f"  inputs:       x1={tuple(t1.shape)}  x2={tuple(t2.shape)}")
                print(f"  after concat: {tuple(cat.shape)}      (expect (1,1,28,56))")
                print(f"  after conv1:  {tuple(a.shape)}      (expect (1,2,26,54))")
                print(f"  after conv2:  {tuple(b.shape)}      (expect (1,4,24,52))")
                print(f"  after pool:   {tuple(c.shape)}      (expect (1,4,12,26))")
                print(f"  after flatten:(1, {c.numel()})              (expect (1,1248))")
        return x


# Late Fusion: process each image with a shared CNN, then concatenate features
class LateFusion(nn.Module):
    def __init__(self):
        super(LateFusion, self).__init__()
        # shared feature extractor
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7 * 2, 128)
        self.fc2 = nn.Linear(128, 82)

    def extract(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32, 7, 7)
        return x.view(x.size(0), -1)          # (B, 1568)

    def forward(self, x1, x2):
        f1 = self.extract(x1)
        f2 = self.extract(x2)
        x = torch.cat([f1, f2], dim=1)        # (B, 3136)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def plot_metric(model_name, list_of_optimizers, metric, ylabel, fname):
    """
    Single figure: one curve per optimizer for the requested metric.

    metric: 'train_accuracies' | 'val_accuracies' | 'test_accuracies'
    """
    colors = {"Adam": "#1f77b4", "RMSprop": "#d62728", "AdamW": "#2ca02c"}
    markers = {"Adam": "o", "RMSprop": "s", "AdamW": "^"}

    fig, ax = plt.subplots(figsize=(10, 6.5))
    pretty = {"train_accuracies": "Training accuracy",
              "val_accuracies":   "Validation accuracy",
              "test_accuracies":  "Test accuracy"}.get(metric, ylabel)
    pretty_model = "Early Fusion" if model_name == "early_fusion" else "Late Fusion"

    n_epochs = 0
    for opt_name in list_of_optimizers:
        report_dir = f"{model_name}_{opt_name}_report"
        path = os.path.join(report_dir, f"{metric}.npy")
        if not os.path.exists(path):
            print(f"Skipping {opt_name}: missing {path}")
            continue
        arr = np.load(path)
        epochs = np.arange(1, len(arr) + 1)
        n_epochs = max(n_epochs, len(arr))
        ax.plot(epochs, arr,
                color=colors.get(opt_name, None),
                marker=markers.get(opt_name, "o"),
                markersize=7, linewidth=2.2,
                label=f"{opt_name}  (final = {arr[-1]:.3f})")
        # annotate the final-epoch value
        ax.annotate(f"{arr[-1]:.3f}",
                    xy=(epochs[-1], arr[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=9, color=colors.get(opt_name, "black"),
                    va="center")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(pretty, fontsize=12)
    ax.set_title(f"{pretty_model}: {pretty.lower()} vs epoch across optimizers",
                 fontsize=13, pad=10)
    if n_epochs:
        ax.set_xticks(range(1, n_epochs + 1))
        ax.set_xlim(0.5, n_epochs + 0.7)  # room for the annotation
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=11, title="Optimizer")
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)


def plot(model_name, list_of_optimizers):
    """
    One figure per fusion model that overlays train and test accuracy curves
    for all three optimizers across epochs.
    """
    plt.figure(figsize=(10, 6))
    for opt_name in list_of_optimizers:
        report_dir = f"{model_name}_{opt_name}_report"
        train_path = os.path.join(report_dir, "train_accuracies.npy")
        test_path = os.path.join(report_dir, "test_accuracies.npy")
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            print(f"Skipping {opt_name}: missing accuracy files in {report_dir}")
            continue
        train_acc = np.load(train_path)
        test_acc = np.load(test_path)
        epochs = range(1, len(train_acc) + 1)
        line, = plt.plot(epochs, train_acc, label=f"{opt_name} train")
        plt.plot(epochs, test_acc, linestyle="--", color=line.get_color(),
                 label=f"{opt_name} test")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name}: train vs test accuracy across optimizers")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_name}_accuracy_comparison.png")
    plt.close()


def complexity(model):
    """
    Print per-layer parameter counts and split totals between
    convolutional and fully-connected layers.
    """
    print(f"\n=== Model: {model.__class__.__name__} ===")

    conv_total = 0
    fc_total = 0
    other_total = 0
    print(f"{'Layer':<20} {'Type':<10} {'Params':>10}")
    print("-" * 44)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            n = sum(p.numel() for p in module.parameters() if p.requires_grad)
            kind = "conv" if isinstance(module, nn.Conv2d) else "fc"
            print(f"{name:<20} {kind:<10} {n:>10}")
            if kind == "conv":
                conv_total += n
            else:
                fc_total += n

    # any other parametric layers (e.g. BN) get bucketed separately
    counted = {id(p) for m in model.modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))
               for p in m.parameters()}
    for p in model.parameters():
        if p.requires_grad and id(p) not in counted:
            other_total += p.numel()

    total = conv_total + fc_total + other_total
    print("-" * 44)
    print(f"{'Conv total':<20} {'':<10} {conv_total:>10}")
    print(f"{'FC   total':<20} {'':<10} {fc_total:>10}")
    if other_total:
        print(f"{'Other total':<20} {'':<10} {other_total:>10}")
    print(f"{'Grand total':<20} {'':<10} {total:>10}")

    if total > 0:
        print(f"Conv fraction: {conv_total / total:.4f}  ({100 * conv_total / total:.2f}%)")
        print(f"FC   fraction: {fc_total / total:.4f}  ({100 * fc_total / total:.2f}%)")

    # FLOPs / MACs via thop (optional)
    if THOP_AVAILABLE:
        try:
            dummy1 = torch.randn(1, 1, 28, 28)
            dummy2 = torch.randn(1, 1, 28, 28)
            macs, params = profile(model, inputs=(dummy1, dummy2), verbose=False)
            macs_str, params_str = clever_format([macs, params], "%.3f")
            print(f"MACs: {macs_str}, Params (thop): {params_str}")
        except Exception as e:
            print(f"thop profiling failed: {e}")
    else:
        print("thop not available, skipping FLOPs/MACs computation")

    return {
        "model": model.__class__.__name__,
        "conv_params": conv_total,
        "fc_params": fc_total,
        "other_params": other_total,
        "total_params": total,
        "conv_fraction": conv_total / total if total else 0.0,
        "fc_fraction": fc_total / total if total else 0.0,
    }


def aggregate_summary():
    """Read each run's test_summary.json and produce one combined summary."""
    runs = []
    for fusion_name in ("early_fusion", "late_fusion"):
        for opt_name in ("Adam", "RMSprop", "AdamW"):
            d = f"{fusion_name}_{opt_name}_report"
            sp = os.path.join(d, "test_summary.json")
            if not os.path.exists(sp):
                print(f"[skip] {d}: no test_summary.json")
                continue
            with open(sp) as f:
                s = json.load(f)
            runs.append({
                "fusion": fusion_name,
                "optimizer": opt_name,
                "final_train_acc": s.get("final_train_acc"),
                "final_test_acc": s.get("final_test_acc"),
                "best_val_epoch": s.get("epoch"),
                "best_val_acc": s.get("val_acc"),
                "best_ckpt_test_acc": s.get("test_acc"),
            })
    return runs


if __name__ == '__main__':
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        # Single-run mode: `python hw6_q3_template.py run <fusion> <optimizer>`
        main(sys.argv[2], sys.argv[3])
        sys.exit(0)

    if len(sys.argv) >= 2 and sys.argv[1] == "aggregate":
        # Aggregate mode: read existing per-run summaries and emit final report
        runs = aggregate_summary()
        plot("early_fusion", ["Adam", "RMSprop", "AdamW"])
        plot("late_fusion", ["Adam", "RMSprop", "AdamW"])
        # Per-Table-2 deliverables — train-acc and val-acc plots
        for fusion_name in ("early_fusion", "late_fusion"):
            plot_metric(fusion_name, ["Adam", "RMSprop", "AdamW"],
                        "train_accuracies", "Training accuracy",
                        f"{fusion_name}_train_accuracy.png")
            plot_metric(fusion_name, ["Adam", "RMSprop", "AdamW"],
                        "val_accuracies", "Validation accuracy",
                        f"{fusion_name}_val_accuracy.png")
        complexity_results = [complexity(EarlyFusion()), complexity(LateFusion())]

        print("\n================ FINAL SUMMARY ================")
        print(f"{'fusion':<14} {'optimizer':<10} {'final_train_acc':>16} {'final_test_acc':>15} "
              f"{'best_val_epoch':>15} {'best_ckpt_test_acc':>20}")
        for s in runs:
            print(f"{s['fusion']:<14} {s['optimizer']:<10} {s['final_train_acc']:>16.4f} "
                  f"{s['final_test_acc']:>15.4f} {s['best_val_epoch']:>15} {s['best_ckpt_test_acc']:>20.4f}")

        with open("final_summary.json", "w") as f:
            json.dump({"runs": runs, "complexity": complexity_results}, f, indent=4)
        print("\nWrote final_summary.json")
        sys.exit(0)

    # Default: run all six configs in this process, then aggregate.
    summaries = []
    for fusion_name in ("early_fusion", "late_fusion"):
        for opt_name in ("Adam", "RMSprop", "AdamW"):
            summaries.append(main(fusion_name, opt_name))

    plot("early_fusion", ["Adam", "RMSprop", "AdamW"])
    plot("late_fusion", ["Adam", "RMSprop", "AdamW"])
    for fusion_name in ("early_fusion", "late_fusion"):
        plot_metric(fusion_name, ["Adam", "RMSprop", "AdamW"],
                    "train_accuracies", "Training accuracy",
                    f"{fusion_name}_train_accuracy.png")
        plot_metric(fusion_name, ["Adam", "RMSprop", "AdamW"],
                    "val_accuracies", "Validation accuracy",
                    f"{fusion_name}_val_accuracy.png")

    complexity_results = [complexity(EarlyFusion()), complexity(LateFusion())]

    print("\n================ FINAL SUMMARY ================")
    print(f"{'fusion':<14} {'optimizer':<10} {'final_train_acc':>16} {'final_test_acc':>15} "
          f"{'best_val_epoch':>15} {'best_ckpt_test_acc':>20}")
    for s in summaries:
        print(f"{s['fusion']:<14} {s['optimizer']:<10} {s['final_train_acc']:>16.4f} "
              f"{s['final_test_acc']:>15.4f} {s['best_val_epoch']:>15} {s['best_ckpt_test_acc']:>20.4f}")

    with open("final_summary.json", "w") as f:
        json.dump({"runs": summaries, "complexity": complexity_results}, f, indent=4)
    print("\nWrote final_summary.json")
