import argparse
import torch.nn as nn

class LossFunction:
    def get_loss_fn(loss_name: str):
        """
        Returns the loss function based on user string input.
        """
        loss_map = {
            "L1Loss": nn.L1Loss(),
            "MSELoss": nn.MSELoss(),
            "CrossEntropyLoss": nn.CrossEntropyLoss()
        }
        
        if loss_name in loss_map:
            return loss_map[loss_name]
        else:
            raise ValueError(f"Loss '{loss_name}' not supported. Choose from: {list(loss_map.keys())}")

# --- Usage with Argparse Logic ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Loss Function related script")
    parser.add_argument("--loss", type=str, default="CrossEntropyLoss", 
                        choices=["L1Loss", "MSELoss", "CrossEntropyLoss"],
                        help="Select the loss function")

    args = parser.parse_args()

    # Setup Loss
    criterion = LossFunction.get_loss_fn(args.loss)

    print(f"Loss Function: {criterion}")