import os
import re
import torch


class BaseAgent:
    def __init__(self):
        pass
    def save_weights(self, target_directory: str) -> None:
        assert self.networks is not None
        weights_directory = os.path.join(target_directory, 'model_weights')
        if not os.path.exists(weights_directory):
            os.makedirs(weights_directory)

        for k, v in self.networks.items():
            fname = os.path.join(weights_directory, k + "_weights.pth")
            torch.save(v.state_dict(), fname)

    def load_weights(self, target_directory: str) -> None:
        contents = os.listdir(target_directory)
        patt = re.compile(".*.pth")
        weights_files = list(filter(lambda x: patt.match(x) is not None, contents)      )
        assert len(weights_files) == len(self.networks), "Expected one weights file for each network."
        for filename in weights_files:
            filepath = os.path.join(target_directory, filename)
            network_name = (
                re.compile("(.*)(_weights.pth)")
                .match(filename)
                .groups()[0]
                )
            (
                self.networks[network_name]
                .load_state_dict(
                    torch.load(filepath, 
                    map_location=lambda storage, 
                    loc: storage)
                    )
            )

