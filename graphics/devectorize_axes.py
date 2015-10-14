import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib.transforms import Bbox

try:
    from cStringIO import StringIO
except ImportError:
    #py3k
    from io import StringIO


def devectorize_axes(ax=None, dpi=None, transparent=True):
    """Convert axes contents to a png.

    This is useful when plotting many points, as the size of the saved file
    can become very large otherwise.

    Parameters
    ----------
    ax : Axes instance (optional)
        Axes to de-vectorize.  If None, this uses the current active axes
        (plt.gca())
    dpi: int (optional)
        resolution of the png image.  If not specified, the default from
        'savefig.dpi' in rcParams will be used
    transparent : bool (optional)
        if True (default) then the PNG will be made transparent

    Returns
    -------
    ax : Axes instance
        the in-place modified Axes instance

    Examples
    --------
    The code can be used in the following way::

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> fig, ax = plt.subplots()
        >>> x, y = np.random.random((2, 10000))
        >>> ax.scatter(x, y)
        >>> devectorize_axes(ax)
        >>> plt.savefig('devectorized.pdf')

    The resulting figure will be much smaller than the vectorized version.
    """
    if ax is None:
        ax = plt.gca()

    fig = ax.figure
    axlim = ax.axis()

    # setup: make all visible spines (axes & ticks) & text invisible
    # we need to set these back later, so we save their current state
    _sp = {}
    _txt_vis = [t.get_visible() for t in ax.texts]
    for k in ax.spines:
        _sp[k] = ax.spines[k].get_visible()
        ax.spines[k].set_visible(False)
    for t in ax.texts:
        t.set_visible(False)

    _xax = ax.xaxis.get_visible()
    _yax = ax.yaxis.get_visible()
    _patch = ax.axesPatch.get_visible()
    ax.axesPatch.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # convert canvas to PNG
    extents = ax.bbox.extents / fig.dpi
    sio = StringIO()
    plt.savefig(sio, format='png', dpi=dpi,
                transparent=transparent,
                bbox_inches=Bbox([extents[:2], extents[2:]]))
    sio.reset()
    im = image.imread(sio)

    # clear everything on axis (but not text)
    ax.lines = []
    ax.patches = []
    ax.tables = []
    ax.artists = []
    ax.images = []
    ax.collections = []

    # Show the image
    ax.imshow(im, extent=axlim, aspect='auto', interpolation='nearest')

    # restore all the spines & text
    for k in ax.spines:
        ax.spines[k].set_visible(_sp[k])
    for t, v in zip(ax.texts, _txt_vis):
        t.set_visible(v)
    ax.axesPatch.set_visible(_patch)
    ax.xaxis.set_visible(_xax)
    ax.yaxis.set_visible(_yax)

    if plt.isinteractive():
        plt.draw()

    return ax
