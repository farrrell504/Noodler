import pandas as pd
from dataclasses import dataclass, field
from typing import List
from . import scales
from . import notes
from . import intervals
from .notes import Note

@dataclass
class Chord:
    root: str = field(init=False)
    type: str = field(init=False)
    notes: List[Note] = field(repr=False)
    intervals: List[str] = field(init=False)
    numbers: List[str] = field(init=False)

    def __post_init__(self):
        self.type = get_chord_type(self.notes)
        self.root = self.notes[0].lower()

        _notes = []
        for note in self.notes:
            if type(note) != notes.Note:
                _notes.append(notes.make_note(note))
            else:
                _notes.append(note)
        self.notes = _notes

        self.intervals = chords[self.type]["_intervals"]
        self.numbers = chords[self.type]["numbers"]


chords = {}
chords['major'] = {
    '_intervals': ('M3','P5'),
    'numbers': ('1','3','5'),
    'chord_type': 'major',
}
chords['minor'] = {
    '_intervals': ('m3','P5'),
    'numbers': ('1','3b','5'),
    'chord_type': 'minor',
    
}
chords['diminished'] = {
    '_intervals': ('m3','T'),
    'numbers': ('1','3b','5b'), 
    'chord_type': 'diminished',
}
chords['augmented'] = {
    '_intervals': ('M3','T'),
    'numbers': ('1','3♯','5♯'), 
    'chord_type': 'augmented',
}
chords['sus2'] = {
    '_intervals': ('M2','P5'),
    'numbers': ('1','2','5'),
    'chord_type': 'sus2',
}
chords['sus4'] = {
    '_intervals': ('P4','P5'),
    'numbers': ('1','4','5'),
    'chord_type': 'sus4',
}

chords_df = pd.DataFrame(chords).T
chords_df

def make_chord(chord_name, *args):
    '''find_chord_type_by_intervals
    makes a chord, should probably better explain
    
    keyword arguments:
    chord_name -- [for now] tuple: (root_note, scale_type)
    '''
    if type(chord_name) == tuple:
        root_note = chord_name[0].lower()
        chord_type = chord_name[1].lower()
    else:
        root_note = chord_name
        chord_type = args[0]
        
    chord_intervals = chords[chord_type]['_intervals']
    
    chord_notes = [root_note]
    for interval in chord_intervals:
        chord_notes.append(intervals.notes_for_interval(root_note,interval)[1])
        
    return Chord(notes=chord_notes)

def get_chords_root_note_for_major_key_by_circle(root_note):
    root_note = root_note.lower()
    
    chords = {}
    
    # get circle of fifths and order of sharps
    circ, order = intervals.make_circle_of_fifths()
    
    # the position in circle says how many sharps are in the scale
    root_idx = circ.index(root_note)
    
    # get list of roman numerals representing chord number
    numerals = scales.get_numerals_of_scale('major')
    
    chords[numerals[0]] = root_note
    chords[numerals[3]] = circ[root_idx+1]
    chords[numerals[4]] = circ[root_idx-1]
    
    return chords

def get_chosrds_for_major_key(root_note):
    root_note = root_note.lower()
    
    chords = {}
    
    # get scale of root note
    scale = scales.get_major_scale_by_circle(root_note)
    numerals = scales.get_numerals_of_scale('major')
    
    chords[numerals[0]] = make_chord((scale[0],'major'))
    chords[numerals[1]] = make_chord((scale[1], 'minor'))
    chords[numerals[2]] = make_chord((scale[2], 'minor'))
    chords[numerals[3]] = make_chord((scale[3], 'major'))
    chords[numerals[4]] = make_chord((scale[4], 'major'))
    chords[numerals[5]] = make_chord((scale[5], 'minor'))
    chords[numerals[6]] = make_chord((scale[6], 'diminished'))
    
    return chords

def get_chord_type_by_intervals(_intervals):
    matches = chords_df[chords_df._intervals == tuple(_intervals)]
    
    if len(matches) > 0:
        # assuming only one match
        chord_type = matches.chord_type.to_list()[0]
    else:
        chord_type = 'Not Found'
    
    return chord_type

def get_chord_type(chord_notes):
    '''
    give this either:
    list of _notes in desired chord
    name of chord
    
    overloaded (idk if thats what this term means but it does to me so stick that in your pipe and lick it)
    '''
    if (type(chord_notes) == list) or (type(chord_notes) == tuple):
        if len(chord_notes) > 2:
            chord_intervals = []
            for idx in range(len(chord_notes)-1):
                chord_intervals.append(intervals.interval_for_notes(chord_notes[0],chord_notes[idx+1]))

            return get_chord_type_by_intervals(tuple(chord_intervals))
        else:
            return 'Function needs a list or tuple of notes'
    else:
        return 'Function needs a list or tuple of notes'
    
def get_chords_for_key(key_name):
    '''
    makes chords for a specified key, should probably better explain
    
    keyword arguments:
    key_name -- [for now] tuple: (root_note, scale_type)
    '''
    
    root_note = key_name[0].lower()
    scale_type = key_name[1].lower()
    
    chords = {}
    
    # get scale of root note
    scale = scales._make_scale((root_note,scale_type))
    numerals = scales.get_numerals_of_scale(scale_type)
    
    for idx, note in enumerate(scale):
        chord = [scale[idx],scale[(idx+2)%len(scale)],scale[(idx+4)%len(scale)]]
        chord_type = get_chord_type(chord)
        chord_symbol = numerals[idx]
        chords[chord_symbol] = chord
   
    return chords
