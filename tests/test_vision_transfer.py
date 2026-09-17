from copy import deepcopy
import numpy as np

from jepa_forge.vision_transfer import moving_shapes, visible_pixels
from jepa_forge.selection import propose_candidates
from jepa_forge.datasets import make_splits


def test_video_entity_split_and_masked_pixel_noninterference():
    data = moving_shapes(seed=8, groups=12, clips_per_group=2)
    splits = make_splits(data, 99)
    group_sets = [set(data.groups[v]) for v in splits.values()]
    assert all(not (a & b) for i,a in enumerate(group_sets) for b in group_sets[i+1:])
    for task in propose_candidates(data):
        poisoned = deepcopy(data)
        hidden = sorted(set(range(data.X.shape[1]))-set(task.context))
        poisoned.X[:,hidden] = 2000
        np.testing.assert_array_equal(visible_pixels(data,task),visible_pixels(poisoned,task))
        assert len(task.context) == data.X.shape[1]//2
