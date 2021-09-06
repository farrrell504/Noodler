from dataclasses import dataclass, field
from typing import List
from . import notes
from . import intervals
from .notes import Note, make_note
from .utils import _whole, _half, _roman_dict

@dataclass
class Scale:
    root: str
    type: str
    notes: List[Note] = field(init=False, repr=False)
    steps: List[str] = field(init=False)
    numerals: List[str] = field(init=False)
    modes: List[str] = field(init=False)

    def __post_init__(self):
        self.notes = _make_scale((self.root,self.type))

        _notes = []
        for note in self.notes:
            if type(note) != Note:
                _notes.append(make_note(note))
            else:
                _notes.append(note)
        self.notes = _notes

        self.steps = scales[self.type]["_intervals"]
        self.steps = scales[self.type]["numerals"]
        self.numerals = get_numerals_of_scale(self.type)
        self.modes = scales[self.type]["modes"]

    def __len__(self):
        return len(self.notes)

    def __iter__(self):
        return iter([note.name for note in self.notes])


scales = {}
scales['major'] = {
    '_intervals': [_whole,_whole,_half,_whole,_whole,_whole,_half],
    'modes': ['ionian','dorian','phrygian','lydian','mixolydian','aeolian','locrian'],
    'numerals': [
        'M',
        'm',
        'm',
        'M',
        'M',
        'm',
        'm,o',
    ]
}

scales['minor'] = {
    '_intervals': [_whole,_half,_whole,_whole,_half,_whole,_whole],
    'modes': ['aeolian','locrian','ionian','dorian','phrygian','lydian','mixolydian'],
    'numerals': [
        'm',
        'm,o',
        'M',
        'm',
        'm',
        'M',
        'M',
    ]    
}

def get_scale_of_mode(mode):
    try:
        # find mode index relative to major/ionian
        idx = scales['major']['modes'].index(mode)
    except ValueError:
        print('Valid modes are {:}'.format(scales['major']['modes']))


    mode_scale = []
    mode_modes = []
    mode_numerals = []
    for n in range(len(scales['major']['_intervals'])):
        n_idx = (idx+n) % len(scales['major']['_intervals'])
        mode_scale.append(scales['major']['_intervals'][n_idx])
        mode_modes.append(scales['major']['modes'][n_idx])
        mode_numerals.append(scales['major']['numerals'][n_idx])

    new_mode = {
        '_intervals': mode_scale,
        'modes': mode_modes,
        'numerals': mode_numerals,
    }
    
    return new_mode

def get_modes_of_scale(scale_type):
    
    if scale_type in scales:
        return scales[scale_type]['modes']

    mode_dict = get_scale_of_mode(scale_type)

    return mode_dict['modes']

def get_numerals_of_scale(scale_type):
    
    if scale_type in scales:
        scale_numerals = scales[scale_type]['numerals']
    else:
        mode_dict = get_scale_of_mode(scale_type)
        scale_numerals = mode_dict['numerals']
    
    numerals = []
    for idx, numeral_change in enumerate(scale_numerals):
        numeral = _roman_dict[idx+1]
        if numeral_change[0].isupper():
            numeral = numeral.upper()
        if numeral_change.find(',') > -1:
            add_this = numeral_change.split(',')[1]
            numeral = numeral + add_this
        numerals.append(numeral)

    return numerals

def get_intervals_of_scale(scale_type):
    if scale_type in scales:
        scale_intervals = scales[scale_type]['_intervals']
    else:
        mode_dict = get_scale_of_mode(scale_type)
        scale_intervals = mode_dict['_intervals']
        
    return scale_intervals

def make_scale(scale_name,  *args, with_next_octave=False):
    '''
    makes a scale, should probably better explain
    
    keyword arguments:
    scale_name -- [for now] tuple: (root_note, scale_type)
    '''    
    if type(scale_name) == tuple:
        root_note = scale_name[0].lower()
        scale_type = scale_name[1].lower()
    else:
        root_note = scale_name
        scale_type = args[0]

    return Scale(root_note, scale_type)

def _make_scale(scale_name,  *args, with_next_octave=False):
    '''
    makes a scale, should probably better explain
    
    keyword arguments:
    scale_name -- [for now] tuple: (root_note, scale_type)
    '''    
    if type(scale_name) == tuple:
        root_note = scale_name[0].lower()
        scale_type = scale_name[1].lower()
    else:
        root_note = scale_name
        scale_type = args[0]

    # make blank alphabet/note list
    scale = notes._make_note_list(root_note[0],only_natural=True)
    
    # replace first with correct note,
    # in case first note was sharp/flat
    scale[0] = root_note
    
    if scale_type not in scales:
        scale_dict = get_scale_of_mode(scale_type)
    else:
        scale_dict = scales[scale_type]
    
    # grab where the root note is
    root_idx = notes._notes.index(root_note)
    cur_idx = root_idx
    for scale_idx, semitones in enumerate(scale_dict['_intervals']):
        cur_idx = semitones+cur_idx
        
        if cur_idx > len(notes._notes)-1:
            cur_idx = cur_idx - len(notes._notes)       
        if scale_idx == len(scale_dict['_intervals'])-1:
            if with_next_octave:
                scale.append(notes._notes[cur_idx])
        else:
            scale[scale_idx+1] = notes._notes[cur_idx]
    
    return scale

def get_major_scale_by_circle(root_note):
    '''
    give root note
    creates major/ionic scale
    '''
    root_note = root_note.lower()
    root_idx = notes._notes.index(root_note)
    # make blank alphabet/note list
    scale = notes._make_note_list(root_note[0],only_natural=True)
    
    # replace first with correct note,
    # in case first note was sharp/flat
    scale[0] = root_note
    
    # get circle of fifths and order of sharps
    circ, order = intervals.make_circle_of_fifths()
    
    # the position in circle says how many sharps are in the scale
    num_sharps = circ.index(root_note)
    
    for idx in range(num_sharps):
        sharp_note_idx = scale.index(order[idx])
        scale[sharp_note_idx] = scale[sharp_note_idx]+'♯'

    return scale
