from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from ultralytics.utils.plotting import Annotator, colors


def plot_obb_boxes_on_image(
    image: npt.NDArray,
    obb_cls: List[int],
    obb_boxes: List,
    obb_names: Optional[Dict[int, str]] = None,
    single_color: Optional[Tuple[int, int, int]] = None,
    line_width: int = 1,
) -> npt.NDArray:
    if obb_names is None:
        obb_names = defaultdict(str)

    ann = Annotator(
        im=np.ascontiguousarray(image),
        line_width=line_width,  # default auto-size
        font_size=None,  # default auto-size
        font="Arial.ttf",  # must be ImageFont compatible
        pil=False,  # use PIL, otherwise uses OpenCV
    )

    for i, cls_idx in enumerate(obb_cls):
        ann.box_label(
            box=obb_boxes[i],
            label=obb_names.get(int(cls_idx)),
            color=single_color or colors(cls_idx, True),
        )

    image_with_obb = ann.result()

    return image_with_obb
