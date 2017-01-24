
import os
import glob
import json
from ..utils import is_numeric
import collections
from ast import literal_eval

import logging

__all__ = ['ensure_dir', 'ensure_file', 'get_files', 'load_input', 'get_dirs',
           'load_eepdefs', 'replace_ext', 'tfm_indict', 'ts_indict',
           'save_ptcri']


def save_ptcri(line, loc=None, prefix=None):
    '''save parsec2match EEPs in similar format as sandro's'''
    import operator
    from ..eep.critical_point import Eep
    print('fileio.save_ptcri is broken.')
    loc == loc or os.getcwd()
    prefix = prefix or ''
    if len(prefix) > 0:
        prefix = '_{0:s}'.format(prefix)

    filename = os.path.join(loc, 'p2m_{0:s}'.format(prefix))
    eep_list = Eep().eep_list

    if hb:
        filename = filename.replace('p2m', 'p2m_hb')

    if hb:
        pdict = Eep().pdict_hb

    # sort the dictionary by values (which are ints)
    sorted_keys, _ = list(zip(*sorted(list(pdict.items()), key=operator.itemgetter(1))))

    cols = ' '.join(list(eep_list))
    header = '# EEPs defined by sandro, basti, mist, and phil \n'
    header += '# M Z {0:s}\n'.format(cols)
    linefmt = '{0:s} \n'
    with open(filename, 'w') as f:
        f.write(header)
        f.write(linefmt.format(line))
    return filename


def tfm_indict():
    '''
    Load default inputs, the eep lists, and number of equally spaced points
    between eeps. Input file will overwrite these. These should be all possible
    input options.
    '''
    base = os.path.split(os.path.split(__file__)[0])[0]
    inp_par = os.path.join(base, 'inputs/tracks4match.json')
    with open(inp_par, 'r') as inp:
        indict = json.load(inp)

    # Set default locations to here
    # for k, v in indict.items():
    #    if k.endswith('dir'):
    #        indict[k] = os.getcwd()
    return indict


def load_eepdefs():
    base = os.path.split(os.path.split(__file__)[0])[0]
    inp_par = os.path.join(base, 'inputs/eeps.json')
    with open(inp_par, 'r') as inp:
        indict = json.load(inp, object_pairs_hook=collections.OrderedDict)

    eep_list, eep_lengths = list(zip(*list(indict.items())))

    eep_list = list(eep_list)
    eep_lengths = list(eep_lengths)[:-1]
    assert (len(eep_list) == len(eep_lengths) + 1), 'Bad eep definitions'
    return eep_list, eep_lengths


def ts_indict():
    '''
    Load default inputs, the eep lists, and number of equally spaced points
    between eeps. Input file will overwrite these. These should be all possible
    input options.
    '''
    base = os.path.split(os.path.split(__file__)[0])[0]
    inp_par = os.path.join(base, 'inputs/trackset.json')
    with open(inp_par, 'r') as inp:
        indict = json.load(inp)
    return indict


def load_input(filename, comment_char='#', list_sep=',', default_dict=None):
    '''
    read an input file into a dictionary

    Ignores all lines that start with #
    each line in the file has format key  value
    True and False are interpreted as bool
    converts values to float, string, or list
    also accepts dictionary with one key and one val
        e.g: inp_dict      {'key': val1}

    Parameters
    ----------
    filename : string
        filename to parse
    comment_char : string
        skip line if it starts with comment_char
    list_sep : string
        within a value, if it's a list, split it by this value
        if it's numeric, it will make a np.array of floats.
    Returns
    -------
    d : dict
        parsed information from filename
    '''
    default_dict = default_dict or None
    d = default_dict.copy()
    with open(filename) as f:
        # skip comment_char, empty lines, strip out []
        lines = [l.strip().replace('[', '').replace(']', '')
                 for l in f.readlines() if not l.startswith(comment_char) and
                 len(l.strip()) > 0]

    # fill the dict
    for line in lines:
        key, val = line.partition(' ')[0::2]
        d[key] = is_numeric(val.replace(' ', ''))

    # check the values
    for key in list(d.keys()):
        # is_numeric already got the floats and ints
        if isinstance(d[key], float) or isinstance(d[key], int) or \
           isinstance(d[key], bool) or d[key] is None:
            continue
        # check for a comma separated list
        temp = d[key].split(list_sep)
        if len(temp) > 1:
            try:
                # assume list of floats.
                d[key] = [is_numeric(t) for t in temp]
            except:
                d[key] = temp
        # check for a dictionary
        elif len(d[key].split(':')) > 1:
            temp1 = d[key].split(':')
            d[key] = {is_numeric(temp1[0]): is_numeric(temp1[1])}
        else:
            val = temp[0]
            # check bool
            true = val.upper().startswith('TRUE')
            false = val.upper().startswith('FALSE')
            none = val.title().startswith('None')
            if true or false or none:
                val = literal_eval(val)
            d[key] = val
    return d


def replace_ext(filename, ext):
    '''replace a filename's current extension with ext'''
    return split_file_extention(filename)[0] + ext


def split_file_extention(filename):
    '''split the filename from its extension'''
    return '.'.join(filename.split('.')[:-1]), filename.split('.')[-1]


def ensure_file(f, mad=True):
    '''Return bool test if file exists. If mad, throw assertion error'''
    fileis = os.path.isfile(f)
    if not fileis:
        msg = '{} not found'.format(f)
        logging.warning(msg)
        if mad:
            assert(fileis), msg
    return fileis


def ensure_dir(f):
    '''if directory f does not exist, make it'''
    if not f.endswith('/'):
        f += '/'

    d = os.path.dirname(f)
    if not os.path.isdir(d):
        os.makedirs(d)
        logging.info('made dirs: {}'.format(d))


def get_dirs(src, criteria=None):
    """
    return a list of directories in src, optional simple cut by criteria

    Parameters
    ----------
    src : str
        abs path of directory to search in
    criteria : str
        simple if criteria in d to select within directories in src

    Returns
    -------
    dirs : abs path of directories found
    """
    dirs = [os.path.join(src, l) for l in os.listdir(src)
            if os.path.join(src, l)]
    if criteria is not None:
        dirs = [d for d in dirs if criteria in d]
    return dirs


def get_files(src, search_string):
    '''return a list of files, similar to ls src/search_string'''
    if not src.endswith('/'):
        src += '/'
    try:
        files = glob.glob1(src, search_string)
    except IndexError:
        logging.error('Can''t find %s in %s' % (search_string, src))
        raise
    files = [os.path.join(src, f)
             for f in files if ensure_file(os.path.join(src, f), mad=False)]
    return files
