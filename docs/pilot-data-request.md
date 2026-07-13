# WT / LLG pilot data request

For the first analysis pass, request a small representative set before asking
for the entire experiment.

## Videos

- Two or three WT videos and two or three LLG-overexpression videos.
- Include one clean/easy video and one difficult video from each genotype.
- Send original or losslessly exported files when possible.
- Keep filenames unique and avoid embedding spaces or special characters.

Suggested name:

```text
YYYYMMDD_genotype_plant-replicate_field-replicate.avi
```

## Metadata required for every video

- Genotype and transgenic line.
- Plant or biological replicate identifier.
- Field/video technical replicate identifier.
- Imaging date/session and microscope/camera setup.
- Biological time represented by one frame.
- Pixel-to-micrometer calibration at the recorded magnification.
- Definition of time zero: media addition, recording start, or another event.
- Any frame averaging, downsampling, cropping, rotation, or contrast processing.
- Temperature and treatment conditions.

Playback FPS in an AVI/MP4 is not necessarily the biological frame interval and
must not be substituted for it.

## Manual reference measurements

For one WT and one LLG video, request:

- Germination frame for approximately 10 clearly visible grains.
- One-hour tube length for the same grains, traced in ImageJ if possible.
- Notes identifying grains that leave the field, overlap, burst, or become
  impossible to follow.

These annotations provide an initial accuracy check; they are not intended to
be a full training dataset.

## Experimental structure

Ask how many independent plants, imaging sessions, and videos are available.
Pollen grains within one video are nested observations and should not be
treated as independent biological replicates without an appropriate model.
