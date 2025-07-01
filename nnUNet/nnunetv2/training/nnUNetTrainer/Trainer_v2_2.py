from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch


class Trainer_v2_2(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        # Call the parent class's __init__ to inherit its setup
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, unpack_dataset=unpack_dataset, device=device)

        self.num_epochs = 200
        self.initial_lr = 1e-4
        self.layers_to_freeze = ["_orig_mod.encoder.stages.0","_orig_mod.encoder.stages.1","_orig_mod.encoder.stages.2"]#,"_orig_mod.encoder.stages.3"]
        print(f"Number of epochs set to {self.num_epochs}")
    

    """    
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
    """
        

    def _print_trainable_layers(self):
        """
        Utility function to print trainable layers for debugging purposes.
        """
        print("\nTrainable layers:")
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                print(name)
        print("\nFrozen layers:")
        for name, param in self.network.named_parameters():
            if not param.requires_grad:
                print(name)
  
    def initialize(self):  
        super().initialize()
        self.freeze_layers()
        self._print_trainable_layers()


    def freeze_layers(self):
        for name, param in self.network.named_parameters():
            if any(layer in name for layer in self.layers_to_freeze):
                param.requires_grad = False
                print(f"Layer frozen: {name}")