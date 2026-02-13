import os
from typing import Callable
from mup import get_shapes, make_base_shapes, set_base_shapes

def ensure_base_shapes(model_builder: Callable[[int, int], object], identifier: str, 
                       width: int, depth: int, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{identifier}_dmodel_{width}_layer{depth}_256_512.bsh')

    if os.path.exists(path):
        return path

    base_model = model_builder(256, depth)
    delta_model = model_builder(512, depth)

    base_shapes = get_shapes(base_model)
    delta_shapes = get_shapes(delta_model)
    make_base_shapes(base_shapes, delta_shapes, savefile=path)
    return path

def apply_mup(target_module: object, model_builder: Callable[[int, int], object], 
              identifier: str, width: int, depth: int, save_dir: str) -> None:
    path = ensure_base_shapes(model_builder, identifier, width, depth, save_dir)
    set_base_shapes(target_module, path)