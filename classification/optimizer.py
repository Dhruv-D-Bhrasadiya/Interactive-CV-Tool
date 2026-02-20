import argparse
import torch.optim
from model import alexnet 

class OptimizerFactory:
    @staticmethod
    def get_optimizer(optimizer_name: str, model_name: str):
        # 1. Instantiate the model object based on string
        if model_name == "Alex":
            model = alexnet.AlexNet()
        else:
            raise ValueError(f"Model {model_name} not supported.")
        
        # 2. Setup Optimizer with parameters
        if optimizer_name == "Adam":
            return torch.optim.Adam(model.parameters())
        elif optimizer_name == "SGD":
            return torch.optim.SGD(model.parameters(), lr=0.01) # Added LR
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported.")

# --- Usage with Argparse Logic ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimizer related function script')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['Adam', 'SGD'], help='Select the optimizer')
    parser.add_argument('--model', type=str, default='Alex', choices=['Alex', 'ConvNext'], help='Select the model')
    args = parser.parse_args()

    # Setup Optimizer - Pass both strings to factory
    optimizer = OptimizerFactory.get_optimizer(args.optimizer, args.model)
    print(f"Optimizer: {optimizer}")