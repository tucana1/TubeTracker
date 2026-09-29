# Movie 1 (ld_v1) blind retest, 29 Sep 2026

The annotator repeated 15 FULL traces blind (random.Random(20260927), one per included isolated grain). The 7-grain onset retest is from Session A.
Tolerances are the scorer's: length within max(2 px, 10%), apex within max(5 px, 10%), onset within 600 frames.

```
onset retest: 4/7 first-visible bins within +/-600 frames of the first answer
length retest: 15 traces repeated, 14 full both times; length within max(2 px, 10%): 11/14; length and apex within max(5 px, 10%): 11/14; median |difference| 1.66 px
   {'trace': 'g029:22', 'first': 6.88, 'repeat': 5.88, 'diff': -1.0, 'tip_px': 0.28, 'length_ok': True, 'tip_ok': True}
   {'trace': 'g027:122', 'first': 33.12, 'repeat': 36.62, 'diff': 3.5, 'tip_px': 1.0, 'length_ok': False, 'tip_ok': True}
   {'trace': 'g030:70', 'first': 24.83, 'repeat': 23.66, 'diff': -1.17, 'tip_px': 0.63, 'length_ok': True, 'tip_ok': True}
   {'trace': 'g011:174', 'first': 91.19, 'repeat': 87.64, 'diff': -3.55, 'tip_px': 0.89, 'length_ok': True, 'tip_ok': True}
   {'trace': 'g036:122', 'first': 51.74, 'repeat': 52.44, 'diff': 0.7, 'tip_px': 1.6, 'length_ok': True, 'tip_ok': True}
   {'trace': 'g033:31', 'first': 9.12, 'repeat': 9.34, 'diff': 0.22, 'tip_px': 0.89, 'length_ok': True, 'tip_ok': True}
   {'trace': 'g025:174', 'first': 43.49, 'repeat': 45.62, 'diff': 2.13, 'tip_px': 2.43, 'length_ok': True, 'tip_ok': True}
   {'trace': 'g028:122', 'first': 34.98, 'repeat': 32.05, 'diff': -2.93, 'tip_px': 1.65, 'length_ok': True, 'tip_ok': True}
   {'trace': 'g038:174', 'first': 73.49, 'repeat': 72.44, 'diff': -1.05, 'tip_px': 0.89, 'length_ok': True, 'tip_ok': True}
   {'trace': 'g032:70', 'first': 20.13, 'repeat': 21.62, 'diff': 1.49, 'tip_px': 2.53, 'length_ok': True, 'tip_ok': True}
   {'trace': 'g021:122', 'first': 34.58, 'repeat': 32.76, 'diff': -1.82, 'tip_px': 2.68, 'length_ok': True, 'tip_ok': True}
   {'trace': 'g039:62', 'first': 4.87, 'repeat': 7.2, 'diff': 2.33, 'tip_px': 2.8, 'length_ok': False, 'tip_ok': True}
   {'trace': 'g014:174', 'first': 'full', 'repeat': 'no_tube'}
   {'trace': 'g004:122', 'first': 33.88, 'repeat': 38.32, 'diff': 4.44, 'tip_px': 4.62, 'length_ok': False, 'tip_ok': True}
   {'trace': 'g012:174', 'first': 91.44, 'repeat': 91.61, 'diff': 0.17, 'tip_px': 1.13, 'length_ok': True, 'tip_ok': True}
```

Human ceiling on these traces: 11/14 lengths (79%), 11/14 length and apex.
For comparison, SparseTrack 0.5.3 on all 104 FULL ld traces: 62/104 lengths (60%), 51/104 length and tip (49%).
The misses are young or short tubes (4.9 -> 7.2 px, 33 -> 37 px), where 2 px or 10% is tight; g014 at bin 174 flipped from full to no_tube.
