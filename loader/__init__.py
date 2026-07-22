from .heightmap_feasibility import (
    HeightmapFeasibilityDataset,
    HeightmapFeasibilityNPZDataset,
    candidate_heightmap_stack,
    resize_heightmap,
    stone_heightmaps_from_mesh,
)

try:
    from .heightmap import (
        HeightmapH5Dataset,
        MixedBatchSampler,
        xy_to_bin,
        resolve_h5_paths,
        get_heightmap_dataloaders,
    )
    from .rl import (
        RLDataset,
        get_rl_dataloaders,
    )
except ModuleNotFoundError as exc:
    if exc.name != "h5py":
        raise
    HeightmapH5Dataset = None
    MixedBatchSampler = None
    xy_to_bin = None
    resolve_h5_paths = None
    get_heightmap_dataloaders = None
    RLDataset = None
    get_rl_dataloaders = None
