"""Prediction tags: tag -> (v2 variant, B3 variant or None). A variant is "id" (the plain cache) or <set><p|l>
(tta.SETS: t2 = e + r180, t4 = the four flips, t4r = the four rotations, t8 = all eight; p: mean probability,
l: mean logit). Every tag is decoded by decode5.decode (repository fuse.fuse + reach.analyze, current defaults)."""
TAGS = {"tta_id": ("id", "id")}  # must reproduce R5 default exactly
for s in ("t2", "t4", "t4r", "t8"):
    for m in ("p", "l", "m"):  # m (max) exploratory
        TAGS[f"tta_v2{s}{m}"] = (f"{s}{m}", "id")          # TTA on v2 only
        TAGS[f"tta_both{s}{m}"] = (f"{s}{m}", f"{s}{m}")    # TTA on v2 and B3
