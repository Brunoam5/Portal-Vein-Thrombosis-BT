from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch


class Trainer_v1_2(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        # Call the parent class's __init__ to inherit its setup
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, unpack_dataset=unpack_dataset, device=device)

        self.num_epochs = 300