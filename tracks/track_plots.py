'''
plotting and diagnostics track is always track object.
'''
from ..graphics.graphics import plot_all_tracks, plot_sandro_ptcri
from ..graphics.graphics import plot_track, quick_hrd, annotate_plot
from ..graphics.diagnostics import diag_plots, check_ptcris
from ..graphics.kippenhahn import kippenhahn


class TrackPlots(object):
    '''a class for plotting tracks'''
    def __init__(self):
        pass

    def annotate_plot(self, *args, **kwargs):
        return annotate_plot(*args, **kwargs)

    def kippenhahn(self, *args, **kwargs):
        return kippenhahn(*args, **kwargs)

    def hrd(self, *args, **kwargs):
        return quick_hrd(*args, **kwargs)

    def plot_track(self, *args, **kwargs):
        return plot_track(*args, **kwargs)

    def diag_plots(self, *args, **kwargs):
        return diag_plots(*args, **kwargs)

    def plot_all_tracks(self, *args, **kwargs):
        return plot_all_tracks(*args, **kwargs)

    def check_ptcris(self, *args, **kwargs):
        return check_ptcris(*args, **kwargs)

    def plot_sandro_ptcri(self, *args, **kwargs):
        return plot_sandro_ptcri(*args, **kwargs)
