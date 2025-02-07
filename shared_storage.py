import copy

import ray
import torch

import discord_io


@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self, checkpoint, config):
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)
        # CHANGED ----------------------------------------------------------
        discord_io.shared_storage_send("Initialized!")
        # ------------------------------------------------------------------

    def save_checkpoint(self, path=None):
        if not path:
            path = self.config.results_path / "model.checkpoint"

        torch.save(self.current_checkpoint, path)

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError

    def set_list_info(self, keys, index, values=None):
        if isinstance(keys, str) and isinstance(index, int) and values is not None:
            self.current_checkpoint[keys][index] = values
        else:
            raise TypeError