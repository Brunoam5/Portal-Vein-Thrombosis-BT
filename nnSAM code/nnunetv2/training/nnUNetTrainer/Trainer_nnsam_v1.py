from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch


class Trainer_nnsam_v1(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        # Call the parent class's __init__ to inherit its setup
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, unpack_dataset=unpack_dataset, device=device)

        self.num_epochs = 100



    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes