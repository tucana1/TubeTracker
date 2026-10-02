"""TubeTracker: the lab's desktop app for pollen germination movies, with SparseTrack inside.

Open a movie, say how long it ran (and the pixel size), let SparseTrack analyse it (in a child process the window
starts and follows), then look at every grain and tube on the movie, go through the readings the model is least
sure of, correct them where needed and export the results. ``python -m tubetracker`` (``window.main``).

Plain-Python parts (no display needed): ``units`` (time and length units), ``runfolder`` (a movie's analysis
folder and its setup), ``model`` (an analysis as the window shows it: grains, tubes, events, the check list),
``overlay`` (where to draw a grain and its tube at a time), ``corrections`` (fixes saved to the review labels file
through the labelling tool's own store), ``imaging`` (frames and close-ups as 8-bit images), ``jobs`` / ``worker``
(the analysis in a child process) and ``exports`` (tables and figures, several movies side by side). The window:
``window``, ``panels``, ``canvas``, ``charts``, ``dialogs``, ``guide`` (the help window), ``theme``.
"""
