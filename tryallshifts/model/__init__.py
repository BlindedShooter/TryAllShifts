from tryallshifts.model.net import *
from tryallshifts.model.rssm import *
from tryallshifts.model.world_model import *
from tryallshifts.model.dynamics import *



MODEL_MAPPING: Dict[str, Union[WorldModel, DynamicsEncoder]] = {}

for cls_name, cl in locals().copy().items():
    try:
        if issubclass(cl, WorldModel) or issubclass(cl, DynamicsEncoder):
            if cl.name is not None:
                MODEL_MAPPING[cl.name] = cl
    except TypeError:
        pass