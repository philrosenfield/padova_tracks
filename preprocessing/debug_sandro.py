import os
import numpy as np
import matplotlib.pylab as plt

from ..graphics.graphics import annotate_plot, hrd, plot_sandro_ptcri
from ..eep.critical_point import CriticalPoint
from ..config import mass, inte_mass, high_mass

from ..fileio import get_dirs, get_files
from ..tracks.track import Track
plt.ion()


def check_eep_hrd(tracks, ptcri_loc, between_ptcris='default', sandro=True):
    '''
    a simple debugging tool.
    Load in tracks (string or Track obj)
    give the location of the ptcri file
    and choose which set of ptcris to plot.
    returns the track set and the axs (one for each Z)

    '''
    if between_ptcris == 'default':
        between_ptcris = [0, -2]
    from ..track_set import TrackSet
    if type(tracks[0]) is str:
        from ..tracks.track import Track
        tracks = [Track(t) for t in tracks]
    ts = TrackSet()
    ts.tracks = tracks
    if not hasattr(tracks[0], 'sptcri'):
        ts._load_ptcri(ptcri_loc, sandro=True)
    if not hasattr(tracks[0], 'iptcri'):
        ts._load_ptcri(ptcri_loc, sandro=False)

    zs = np.unique([t.Z for t in tracks])
    axs = [plt.subplots()[1] for i in range(len(zs))]
    [axs[list(zs).index(t.Z)].set_title(t.Z) for t in tracks]

    for t in tracks:
        ax = axs[list(zs).index(t.Z)]
        plot_track(t, logT, logL, sandro=sandro, ax=ax,
                   between_ptcris=between_ptcris, add_ptcris=True,
                   add_mass=True)

        ptcri_names = Eep().eep_list[between_ptcris[0]: between_ptcris[1] + 1]
        annotate_plot(t, ax, logT, logL, ptcri_names=ptcri_names)

    [ax.set_xlim(ax.get_xlim()[::-1]) for ax in axs]
    return ts, axs


def check_ptcri(CriticalPoint, mass_, arr):
    ndefined = len(np.nonzero(arr > 0)[0])
    needed = 15

    if (inte_mass < mass_ <= high_mass) and ndefined != needed:
        print('check_ptcri error: M{:.3f} does not have enough EEPs'
              .format(mass_))
        try:
            masses = np.array([f.split('F7_M')[1].replace('.PMS', '')
                               for f in CriticalPoint.fnames], dtype=float)
        except:
            masses = np.array([f.split('F7_M')[1].replace('.DAT', '')
                               for f in CriticalPoint.fnames], dtype=float)

        inds, = np.nonzero(mass_ == masses)
        print('files in question:')
        print(np.array(CriticalPoint.fnames)[inds])
        for ind in inds:
            fix_ptcri(np.array(CriticalPoint.fnames)[ind])


def guessandcheck(ptname, track, pt=None):
    """
    interactively plot an input point on a track's HRD.
    Enter the index of the array to plot, and 0 to return that index.

    A big assumption is that the ptcri file is in a folder called data
    and that the corresponding track is in ../tracks/
    Parameters
    ----------
    ptname : string
        name of the track point, for stout messages only

    pt : int optional
        a starting guess. If not None, will plot it first.

    Returns
    -------
    pt : int
        the final value decided
    """
    go_on = 1

    outmsg = '%s MODE: %i'
    if pt is not None:
        # plot guess first
        print(outmsg % (ptname, track.data[MODE][pt]))
        go_on, pt = plot_point(ptname, pt)

    while go_on != 0:
        go_on, pt = plot_point(ptname, pt)

    print(outmsg % (ptname, track.data[MODE][pt]))
    return track.data[MODE][pt]


def plot_point(ptname, pt):
    inmsg = 'Enter new value for %s or 0 to move on: ' % ptname
    ax.plot(track.data[logT][pt], track.data[logL][pt], '*')
    plt.draw()
    go_on = raw_input(inmsg)
    try:
        go_on = int(go_on)
    except:
        import pdb
        pdb.set_trace()
    if go_on != 0:
        # don't overwrite with "move on" value
        pt = go_on
    return go_on, pt


def fix_ptcri(CriticalPoint, fname=None, iptcri=None, track=None):
    """
    print better values of sandro's EEPs (the basis for Phil's EEPs)
    (It is expected you will hand code the proper values in the ptcri file
    or a copy of the ptcri file.)

    This was made for inte_mass tracks, 12>M>20 that had errors.
    The errors were typically:
    1) MS_BEG is at the point in the track that should be POINT_C
    2) POINT_C is at the point in the track that should be LOOP_B
    3) RG_TIP is at the point in the track that should probably be the
       final EEP.
    4) LOOP_A, B, C, TPAGB/CBUR EEPS are all 0.
        This final issue could be accounted for in the ptcri reader, but
        it is taken as a sign that the other EEPs are probably wrong.)

    Paramaters
    ----------
    fnane : string
        filename of the F7 file from the bad line of the ptcri file

    Returns
    -------
        sys.exit()
        prints EEPs to stout and exits upon completion.
    """
    if track is None:
        # track_dir = CriticalPoint.base.replace('data', 'tracks')
        track_dir, tname = os.path.split(fname)
        estr = 'PMS'
        hb = 'hb' in fname.lower()
        if hb:
            estr = 'HB'
        z = tname.split('Z')[1].split('Y')[0].replace('_', '')
        mass_ = tname.split('M')[1].replace('.DAT', '') \
                     .replace('.P', '').split('.HB')[0]
        try:
            track_file, = get_files(track_dir,
                                    '*M{}*{}'.format(mass_, estr))
        except ValueError:
            # the above search will not distiguish M030.000 and M130.000
            track_files = get_files(track_dir, '*{}*'.format(mass_))
            # cull mass values
            try:
                tms = np.array(['.'.join(os.path.split(t)[1].split('M')[1]
                                         .split('.')[:-1])
                                for t in track_files], dtype=float)
            except:
                tms = np.array(['.'.join(os.path.split(t)[1].split('M')[1]
                                         .split('.')[:-3])
                                for t in track_files], dtype=float)

            # select by mass
            track_file, = \
                np.array(track_files)[np.nonzero(float(mass_) == tms)]

        track = Track(track_file)
    else:
        tname = track.name

    if iptcri is not None:
        track.iptcri = iptcri

    if hasattr(track, 'iptcri'):
        ax = hrd(track)
        ax = hrd(track, ax=ax, inds=iptcri)
        # annotate_plot(track, ax, logT, logL)
    else:
        ax = plot_sandro_ptcri(track, ptcri=CriticalPoint)

    print('Open sandros ptcri file and get ready to edit. {}'
          .format(tname))
    print('Current values:', track.sptcri)
    if len(track.sptcri) <= 3:
        print('this ptcri looks especially messed up.')
        import pdb
        pdb.set_trace()

    pms_beg = guessandcheck('pms_beg', track, pt=track.sptcri[0])
    pms_min = guessandcheck('pms_min', track, pt=track.sptcri[1])
    pms_end = guessandcheck('pms_end', track, pt=track.sptcri[2])
    near_zam = guessandcheck('near_zam', track, pt=track.sptcri[4])
    ms_beg = guessandcheck('ms_beg', track, pt=track.sptcri[4])

    # experience has shown that Sandro's code sets MS_BEG as what should
    # be point C
    point_c = track.sptcri[4]
    point_c = guessandcheck('point_c', track, pt=point_c)

    # MS_TMIN is the min LOG_TE between MS_BEG and POINT_C
    inds = np.arange(ms_beg, point_c, dtype=int)
    point_b = ms_beg + np.argmin(track.data[logT][inds])
    point_b = guessandcheck('point_b', track, pt=point_b)

    # RG_BASE is probably the lowest LOG_L after POINT_C
    inds = np.arange(point_c, len(track.data), dtype=int)
    rg_base = point_c + np.argmin(track.data[logL][inds])
    rg_base = guessandcheck('rg_base', track, pt=rg_base)

    # RG_TIP is the peak LOG_L after RG_BASE, probably happens soon
    # in high mass tracks.
    inds = np.arange(rg_base, rg_base + 200, dtype=int)
    rg_tip = rg_base + np.argmax(track.data[logL][inds])
    rg_tip = guessandcheck('rg_tip', track, pt=rg_tip)

    # There is no RG_BMP when there is core fusion. Pick equally spaced
    # points
    rgbmp1, rgbmp2 = np.linspace(rg_base, rg_tip, 4, endpoint=False)[1:-1]

    # Take the final point of the track, either the final, or where
    # Sandro cut it
    fin = track.data[MODE][-1]
    fin_sandro = track.sptcri[np.nonzero(track.sptcri > 0)][-1]
    if fin > fin_sandro:
        print('last point in track {}, Sandro cut at {}.'
              .format(fin, fin_sandro))
        fin = guessandcheck('final point', track, pt=fin_sandro)

    # I don't use Sandro's loops, so I don't care, here are non zero points
    # that ensure age increase.
    loopa, loopb, loopc = np.linspace(rg_tip, fin, 5, endpoint=False)[1:-1]

    print(track.sptcri[:3])
    print('suggested line ptcri line:')
    print(''.join(['%10i' % i for i in
                   (pms_beg, pms_min, pms_end, near_zam, ms_beg, point_b,
                    point_c, rg_base, int(rgbmp1), int(rgbmp2), rg_tip,
                    loopa, loopb, loopc, fin)]))

    sys.exit()
