import os
import fileio

def add_comments_to_header(tracks_base, prefix, search_term):
    '''
    insert a # at every line before MODE. genfromtxt will skip the
    footer and doesn't need #... but that's a place to improve this
    function.
    '''
    tracks = os.path.join(tracks_base, prefix)
    track_names = fileIO.get_files(tracks, search_term)

    for name in track_names:
        with open(name, 'r') as t:
            lines = t.readlines()
        try:
            imode, = [i for (i, l) in enumerate(lines)
                      if l.strip().startswith('MODE ')]
        except ValueError:
            print '\n %s \n' % name

        lines[:imode + 1] = ['# ' + l for l in lines[:imode + 1]]

        oname = '%s.dat' % name

        with open(oname, 'w') as t:
            t.writelines(lines)


def quick_color_em(tracks_base, prefix, photsys='UVbright',
                   search_term='*F7_*PMS', ):
    '''
    This goes quickly through each directory and adds a [search_term].dat file
    that has # in the header and a [search_term].dat.[photsys]] file that is
    the output of Leo's fromHR2mags.

    sometimes leo's code has bad line endings or skips lines, i donno. so
    when reading in as TrackSet, you'll get loads of warnings...

    ex:
    tracks_base = '/Users/phil/research/parsec2match/S12_set/CAF09_S12D_NS/'
    prefix = 'S12D_NS_Z0.0001_Y0.249'
    quick_color_em(tracks_base, prefix)
    '''

    tracks = os.path.join(tracks_base, prefix)
    track_names = fileIO.get_files(tracks, search_term)

    for name in track_names:
        # this is set for .PMS and .PMS.HB tracks
        z = float(name.replace('.dat','').upper().split('Z')[1].split('_Y')[0])
        color_tracks(track_names, comments=True, z=z, photsys=photsys)


def color_tracks(filename, fromHR2mags=None, logl=5, logte=6, mass=2, z=None,
                 comments=True, photsys='UVbright'):
    if fromHR2mags is None:
        fromHR2mags = '~/research/padova_apps/fromHR2mags/fromHR2mags'

    cmd = '%s %s ' % (fromHR2mags, photsys)
    cmd += '%s %i %i %i %.4f'

    if comments:
        add_comments_to_header(tracks_base, prefix, search_term)
        filename +=  '.dat'

    os.system(cmd % (filename, logl, logte, mass, z))

