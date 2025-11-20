import os
import torch
import matplotlib.pyplot as plt
from models import MakeupGCN, post_processor
from data import MakeupDataset


def visualize_results(dataset, model, device, save_dir="results_test", max_show=10):
    os.makedirs(save_dir, exist_ok=True)
    print("\nRunning inference...\n")

    for idx in range(min(max_show, len(dataset))):
        data = dataset[idx]
        if data is None:
            print(f"Skipping invalid sample {idx}")
            continue

        img_src = data.img_src
        img_ref = data.img_ref

        result = post_processor(model, img_src, img_ref, device)
        if result is None:
            print(f"Failed to process sample {idx}")
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img_src)
        axes[0].set_title("Source")
        axes[0].axis("off")

        axes[1].imshow(img_ref)
        axes[1].set_title("Reference")
        axes[1].axis("off")

        axes[2].imshow(result)
        axes[2].set_title("Result")
        axes[2].axis("off")

        plt.tight_layout()

        out_path = os.path.join(save_dir, f"result_{idx}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved → {out_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load test split
    test_dataset = MakeupDataset(
        makeup_csv="/content/makeup_split.csv",
        non_csv="/content/non_split.csv",
        device=device,
        split="test",
        max_samples=None
    )

    # Load trained GCN model
    model = MakeupGCN(
        in_channels=9,
        hidden_channels=128,
        num_layers=4
    ).to(device)

    checkpoint_path = "makeup_gcn_best.pth"
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Run inference + save results
    visualize_results(
        dataset=test_dataset,
        model=model,
        device=device,
        save_dir="results_test",
        max_show=20
    )

    print("\nTesting complete.\n")
