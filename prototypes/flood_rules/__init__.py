"""The flood's start and stop rules for the BatchNorm tube maps, tuned on one movie and checked on the other (30 Sep 2026).

Maps: the leave-one-out tn_bn_r3v6 networks of prototypes/tube_net (bg 96): movie 2 (m2) read with
runs/tube_net/tn_bn_r3v6_ld.pt (trained with the dev movie's traces), the dev movie (ld) with tn_bn_r3v6_m2.pt.
``bench serve`` holds one movie's probability movie in memory and reads variants from a queue; its default run
reproduces runs/tube_net/e2e_tn_bn_r3v6_*.json and (maps "shipped") runs/lab_checks_2026-09-29/final_070.json exactly.
Record: runs/flood_rules/ (q_*/done: one dump per run, q_*/pred: predictions, stage1.json: the table below).

Result: no tuned rule set transfers between the movies. Tuned on m2 (hybrid; pre-registered: settings gaining >= 2
length hits, combined greedily) -> flood_frac 0.5: m2 26/54 (+2 in sample), ld -1 (hybrid, CI -3..+0). Tuned on ld
(flood everywhere, the only way the dev movie reads many grains with the flood) -> flood_p 0.6 + flood_tip radial
+ flood_give_up 80 + flood_exit_edge: ld 67/104 flood everywhere (+10 in sample), m2 23/54 (-1, CI -10..+7). The two
movies want opposite thresholds (P >= 0.6: ld +8, m2 -2; halo 1 px: ld +3, m2 -5; frac 0.5: m2 +2, ld -1; exit
edge: ld +2, m2 -1), as on the old maps. One rule helps both, the tip the length is read to (flood_tip "radial"):
a young stub's blob widens along the rim, pieces joining late get the greatest rim distance, and ``from_exit`` then
measured to a rim pixel (ld g012@31: flood reach 9.8 px, read 3.6, human 7.2). m2 +1 length (CI -2..+4), +2 length
and tip (+0..+5); ld hybrid +1 (+0..+3), +2 (+0..+5); ld flood everywhere +4 (+1..+8), +7 (+3..+12). On the old
maps it loses (m2 -4): a rule for these maps only. It was found by looking at both movies' grains, so it is not a
leave-one-out selection. Dispatch: keep the hybrid (flood everywhere: ld -16, CI -27..-5; m2 +0).

Settings tried (each on m2 hybrid, ld hybrid and ld flood everywhere; changes vs the default rules on these maps,
m2 hybrid / ld flood everywhere lengths): flood_tip radial +1/+4, radial_zone +1/+3, flood_exit_edge -1/+2, halo 2
-2/+0, halo 1 -5/+3, start band 2 -1/-2, 6 -1/+0, min_len 4 +0/+1, 12 -3/+0, give_up 20 -1/+1, 80 +0/+3, never
+0/+3, arc 45 +0/+0, 90 +0/+0, old-far test off +0/+1, 20 px +0/+1, look-back off +0/-1 (m2 onsets +1, reported-onset
accidental hits 3 -> 0), 0.4 +0/+0, persist 5 +0/+0, frac 0.5 +2/-1, P 0.4 +1/-1, 0.6 -2/+8, recent 6 -5/-3, 24
+0/-1, bridge 3 -1/+2, 6 -1/+2, tip offset 1 px -3/+0: 27 settings; then 6 greedy combinations on ld (on m2 the
pre-registered threshold passed one setting: nothing to combine) and, post hoc with a threshold of +1, 2 more on m2
(frac 0.5 + radial tip: m2 28/54, 25 length and tip; ld hybrid +0, ld flood everywhere +2).

Accidental length hits (the flood's first claim > 10 bins before the annotator's last-absent bin / its reported
onset, after the look-back, that early), m2: 0.7.0 6/8 of 23; new maps, default rules 0/3 of 24; radial tip 0/4.
"""
