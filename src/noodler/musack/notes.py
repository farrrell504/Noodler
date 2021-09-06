
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
# from . import intervals


# fundamentals
_notes = ['a','a♯','b','c','c♯','d','d♯','e','f','f♯','g','g♯']

@dataclass
class Note:
    name: str
    _accidental: str = field(default=None, repr=False)

    def __post_init__(self):
        self.name = self.name.lower()
        self.name = _hash_to_sharp(self.name)
        self._accidental = _find_accidental(self.name)

    def flip_accidental(self):
        if self._accidental == "b":
            self.name = _flat_to_sharp(self.name)
            self._accidental = "♯"
        elif self._accidental == "♯":
            self.name = _sharp_to_flat(self.name)
            self._accidental = "b"

@dataclass
class ScientificNote(Note):
    frequency: float = field(default=None, init=False)
    note: str = field(default="", init=False)
    octave: int = field(default=0, init=False)
    midi: int = field(default=0, init=False)
    
    def __post_init__(self):
        self.note, self.octave = split_note(self.name)
        self.frequency = note_to_frequency(self.note, self.octave)
        self.midi = _get_midi_number(self.note+str(self.octave))

    def __add__(self, note):
        from . import intervals
        return intervals.Interval(self, note)

def make_note(note_name):
    note, octave = split_note(note_name)

    if octave is None:
        return Note(note_name)
    else:
        return ScientificNote(note_name)

# note functions
def _make_note_list(root,only_natural=False):
    '''returns list whose starting note is the note given'''
    note_list = []
    
    root = _hash_to_sharp(root)

    root_idx = _notes.index(root.lower())

    for idx in range(len(_notes)):
        interval_idx = root_idx + idx
        
        # check if we are at the end of the note alphabet, 
        # if so, start back at beginning
        if interval_idx > len(_notes)-1:
            interval_idx = interval_idx - len(_notes)
        
        new_note = _notes[interval_idx]
        if only_natural:
            # dont add sharps or flats
            if len(new_note) == 1:
                note_list.append(new_note)
        else:
            note_list.append(new_note)
        
    return note_list

def _sharp_to_flat(note):
    '''converts a sharp note to an equivalent flat note'''
    idx = _notes.index(note.lower())
    if idx+1 > len(_notes)-1:
        idx = -1
    flat_note = "{:}b".format(_notes[idx+1])
    return flat_note

def _flat_to_sharp(note):
    '''converts a flat note to an equivalent sharp note'''
    idx = _notes.index(note[0].lower())
    if idx+1 > len(_notes)-1:
        idx = -1
    sharp_note = _notes[idx-1]
    return sharp_note

def _hash_to_sharp(note):
    return note.replace("#","♯")

def _find_accidental(note):
    if any(x in note for x in ["#", "♯", "b"]):
        if "b" in note:
            return "b"
        else:
            return "♯"
    else:
        return "♮"

def _get_midi_number(note):
    note, octave = split_note(note)

    if octave is not None:
        midi = (octave + 1)*12
        midi += _make_note_list('c').index(note)
        return midi
    return None

def split_note(note):
    if len(note) == 1:
        # easy peasy, 
        # you gave us a single note
        note = note
        octave = None
    elif len(note) == 2:
        if any(x in note for x in ["#", "♯", "b"]):
            # almost got me!
            # you sly fox
            # you passed a note WITH an accidental! (maybe)
            if note[1].isnumeric():
                # ooo you so sneaky!!
                # handle when note is b
                octave = int(note[1:])
                note = note[0]
            else:
                note = note
                octave = None
        else:
            # handle when note is b
            octave = int(note[1:])
            note = note[0]                
    elif any(x in note for x in ["#", "♯", "b"]):
        # handle if a sharp or flat was given
        octave = int(note[2:])
        note = note[0:2]
    else:
        octave = int(note[1:])
        note = note[0]
        
    return note, octave

def note_to_frequency(note,octave=None):
    '''
    assumes use of A4 = 440Hz
    
    keyword arguments:
    note -- either note, 'a♯' or scientific pitch notation, 'a♯3'
    
    optional keyword agruments:
    octave -- if providing just a note above, must provide octave number here
    '''
    note = note.lower()
    note = _hash_to_sharp(note)

    if octave is None:
        note, octave = split_note(note)
        if octave is None:
            print('ERROR: note_to_frequency requires notation of notefrequency, for example E4 or E,4')
            return
            
    f0 = 440
    
    # root note is at octave 4
    octave_change = octave - 4

    # in semitones
    octave_change = octave_change * 12
    
    note_idx = _make_note_list('c').index(note)
    
    n = note_idx + octave_change

    # offset by 9 semitones since C4 is our target
    r = 2 ** ((n-9)/len(_notes))
    f = f0 * (r)
  
    return f
