# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class TIB_GROUNG1Dataset(BaseSegDataset):
    """TIB_GROUNG1 dataset.

    In segmentation map annotation for TIB_GROUNG1, Train-IDs of the 10k version
    are from 1 to 171, where 0 is the ignore index, and Train-ID of TIB_GROUNG1
    164k is from 0 to 170, where 255 is the ignore index. So, they are all 171
    semantic categories. ``reduce_zero_label`` is set to True and False for the
    10k and 164k versions, respectively. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    METAINFO = dict(
        classes=(
            'concreteroad', 'road_curb', 'redbrickroad', 'zebracrossing', 
            'stone_pier', 'soil', 'yellowbrick_road'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30]])

    def __init__(self,
                 img_suffix='.jpg',
                 reduce_zero_label: bool = False,
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, 
            seg_map_suffix=seg_map_suffix, 
            reduce_zero_label=reduce_zero_label,
            **kwargs)
