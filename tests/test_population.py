import numpy as np

from prototypes.v30_video_apex.targets import encode_mask_raster
from tubetracker.population import banked_grains, build_population_report, census_tiles


def test_banked_join_decodes_real_raster_and_scopes_ruling_to_one_grain():
    registry = {"grains": {
        "m|track-a": {"movie": "m", "grain_native": [10, 20], "first_seen_frame": 30},
        "m|b": {"movie": "m", "grain_native": [30, 20], "first_seen_frame": 30},
        "other|c": {"movie": "other", "grain_native": [10, 20], "first_seen_frame": 30}},
        "aliases": {"owner|owner-a": "m|track-a"}}
    paint = np.zeros((64, 64), bool); paint[20:24, 14:24] = True
    data = {"body_masks": [{"mask_uuid": "mask", "movie": "m", "source_frame": 20,
        "source_obs_uuid": "a", "mask_revision": 3, "mask_raster": encode_mask_raster(paint)}],
        "observations": [{"obs_uuid": "obs", "movie": "m", "source_frame": 30,
        "owner_uuid": "owner-a", "direct_state": "direct_visible", "direct_xy": [24, 22]}]}
    rulings = [{"uuid": "ruling", "revision": 4, "source": "human-db#ruling", "data": {
        "movie": "m", "grain_owner": "a", "ruling": "started germinated and never grew",
        "reviewed_frames": [20, 30]}}]
    grains, audit = banked_grains(registry, data, rulings=rulings)
    assert audit["mask_decoding"][0]["decoded_painted_pixels"] == 40
    report = build_population_report(grains, [], movie="m")
    assert report["census"]["n_grains"] == report["germination"]["denominator_grains"] == 2
    assert report["germination"]["numerator_germinated"] == 1
    interval = next(iv for iv in report["intervals"] if iv["grain_id"] == "m|track-a")
    assert interval["first_verified_present"]["frame"] == 20
    assert interval["classification_revision"] == 4
    assert report["germination"]["population_fraction"] is None
    later = build_population_report(grains, [], movie="m", scope={
        "movie": "m", "roi_xywh": [0, 0, 64, 64], "frames": [30, 40]})
    assert later["germination"]["numerator_germinated"] == 0


def test_snapshot_tiles_use_actual_frame_extent_and_protocol():
    row = {"task_uuid": "tile", "movie": "m", "source_frame": 30,
        "tile_xywh": [10, 20, 80, 90], "task_revision": 7, "complete": True,
        "class_scopes": ["grains"], "border_policy": "centre-inside",
        "clump_policy": "individual-physical-grains", "tile_derived": False}
    tile = census_tiles([row])[0]
    assert (tile.frame, tile.revision, tile.box_xywh) == (30, 7, (10, 20, 80, 90))
    assert tile.exhaustive
    assert not census_tiles([dict(row, tile_derived=True)])[0].exhaustive
