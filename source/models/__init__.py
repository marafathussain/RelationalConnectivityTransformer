from omegaconf import DictConfig
from .RBNT import RelationalBrainNetworkTransformer, RelationalBrainNetworkTransformer2, RelationalBrainNetworkTransformer3, RelationalBrainNetworkTransformer4


def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config).cuda()