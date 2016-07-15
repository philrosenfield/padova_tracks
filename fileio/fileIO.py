from __future__ import print_function, division
import os
import glob
from pprint import pprint
from ..utils import is_numeric
from ast import literal_eval

import logging

__all__ = ['InputFile', 'InputParameters', 'ensure_dir', 'ensure_file',
           'get_files', 'load_input', 'replace_ext', 'get_dirs']


class InputParameters(object):
    '''
    need to make a dictionary of all the possible parameters
        (in the ex: rsp.TrilegalUtils.galaxy_input_dict())
    need to make a formatted string with dictionary printing
        (in the ex: rsp.TrilegalUtils.galaxy_input_fmt())

    example
    import ResolvedStellarPops as rsp
    inp = rsp.fileIO.input_parameters(default_dict=galaxy_input_dict())
    send any replacement params as kwargs.
    inp.write_params('test', rsp.TrilegalUtils.galaxy_input_fmt())
    $ cat test

    use print inp to see what current values are in cmd line.
    '''
    def __init__(self, default_dict=None):
        self.possible_params(default_dict)

    def possible_params(self, default_dict=None):
        '''
        assign key as attribute name and value as attribute value from
        dictionary
        '''
        default_dict = default_dict or {}
        [self.__setattr__(k, v) for k, v in default_dict.items()]

    def update_params(self, new_dict, loud=False):
        '''only overwrite attributes that already exist from dictionary'''
        if loud:
            self.check_keys('not updated', new_dict)
        [self.__setattr__(k, v)
         for k, v in new_dict.items() if hasattr(self, k)]

    def add_params(self, new_dict, loud=False):
        '''add or overwrite attributes from dictionary'''
        if loud:
            self.check_keys('added', new_dict)
        [self.__setattr__(k, v) for k, v in new_dict.items()]

    def write_params(self, new_file, formatter=None, loud=False):
        '''write self.__dict__ to new_file with format from formatter'''
        with open(new_file, 'w') as f:
            if formatter is not None:
                f.write(formatter % self.__dict__)
            else:
                for k in sorted(self.__dict__):
                    f.write('{0: <16} {1}\n'.format(k, str(self.__dict__[k])))
        if loud:
            logging.info('wrote {}'.format(new_file))

    def check_keys(self, msg, new_dict):
        """ check if new_dict.keys() are already attributes """
        new_keys = [k for k, v in new_dict.items() if not hasattr(self, k)]
        logging.info('{}: {}'.format(msg, new_keys))

    def __str__(self):
        '''pprint self.__dict__'''
        pprint(self.__dict__)
        return ""


class InputFile(object):
    '''
    a class to replace too many kwargs from the input file.
    does two things:
    1. sets a default dictionary (see input_defaults) as attributes
    2. unpacks the dictionary from load_input as attributes
        (overwrites defaults).
    '''
    def __init__(self, filename, default_dict=None):
        if default_dict is not None:
            self.set_defaults(default_dict)
        self.in_dict = load_input(filename)
        self.unpack_dict()

    def set_defaults(self, in_def):
        self.unpack_dict(udict=in_def)

    def unpack_dict(self, udict=None):
        if udict is None:
            udict = self.in_dict
        [self.__setattr__(k, v) for k, v in udict.items()]


def load_input(filename, comment_char='#', list_sep=','):
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
    d = {}
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
    for key in d.keys():
        # is_numeric already got the floats and ints
        if type(d[key]) == float or type(d[key]) == int:
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
