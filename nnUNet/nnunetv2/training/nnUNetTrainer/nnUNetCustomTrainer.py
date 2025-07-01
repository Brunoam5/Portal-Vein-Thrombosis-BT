from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch


class nnUNetCustomTrainer(nnUNetTrainer):
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        # Call the parent class's __init__ to inherit its setup
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, unpack_dataset=unpack_dataset, device=device)

        # 🧩 1. Modify number of epochs
        self.num_epochs = 100
        print(f"Number of epochs set to {self.num_epochs}")
    """

        
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
        

    def _print_trainable_layers(self):
        """
        Utility function to print trainable layers for debugging purposes.
        """
        print("\nTrainable layers:")
        for name, param in self.network.named_parameters():
            #if param.requires_grad:
             print(name)
  
    def initialize(self):  
        super().initialize()
# 🧩 4. Print a summary of which layers are trainable
        self._print_trainable_layers()
       #self.freeze_unfreeze_layers()


    def freeze_unfreeze_layers(self):
# 🧩 2. Freeze all layers by default
        print("Freezing all layers...")
        for param in self.network.parameters():
            param.requires_grad = False

        # 🧩 3. Unfreeze specific layers (example: decoder layers)
        print("Unfreezing decoder layers...")
        for name, param in self.network.named_parameters():
            if "decoder" in name:
                param.requires_grad = True
                print(f"Unfrozen layer: {name}")


