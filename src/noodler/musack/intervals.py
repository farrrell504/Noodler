from . import notes
from dataclasses import dataclass, field

# step sizes
_whole = 2
_whole_step = _whole
_half = 1
_half_step = _half

# _intervals
_intervals = {
    "u":0,  # union
    "m2":1,
    "M2":2,
    "m3":3,
    "M3":4,
    "P4":5,
    "T":6,  # tritone
    "P5":7,
    "m6":8,
    "M6":9,
    "m7":10,
    "M7":11,
    "o":12,  # octave
}

_intervals_by_semitones = {}
for interval_name, semitone_count in _intervals.items():
    _intervals_by_semitones[semitone_count] = interval_name

# make it possible to call semitone count and return interval name
_intervals.update(_intervals_by_semitones)


# interval functions
def _decompose_interval(interval):
    whole_count = int(_intervals[interval]/2)
    half_count = _intervals[interval]%2
        
    decomposed_intervals=[]
    [decomposed_intervals.append('_whole')for count in range(whole_count)]
    [decomposed_intervals.append('_half')for count in range(half_count)]
    
    return decomposed_intervals

def notes_for_interval(root,interval):
    notes_returned = [root.lower()]
    
    semitones = _intervals[interval]
    note_list = notes._make_note_list(root.lower())
    interval_note = note_list[semitones]
    
    notes_returned.append(interval_note)
    
    return notes_returned

def interval_for_notes(note_1, note_2):
    note_1_name, note_1_octave = notes.split_note(note_1)
    note_2_name, note_2_octave = notes.split_note(note_2)
    
    note_1_idx = notes._make_note_list('c').index(note_1_name.lower())
    note_2_idx = notes._make_note_list('c').index(note_2_name.lower())
    
    if (note_1_octave is not None) and (note_2_octave is not None):
        note_1_semitones = note_1_idx + (len(notes._notes)*note_1_octave)
        note_2_semitones = note_2_idx + (len(notes._notes)*note_2_octave)

        interval_semitones = note_2_semitones - note_1_semitones

        if interval_semitones > 0:
            direction = '+'
        elif interval_semitones < 0:
            direction = '-'
        else:
            # interval_semitones == 0?
            return 'u'

        interval = _intervals[abs(interval_semitones)%len(notes._notes)]
        if direction == '-':
            interval = direction + interval
        
        for octave in range(abs(int(interval_semitones/len(notes._notes)))):
            interval = interval+direction+'o'   
    else:
        # handle note 1 being higher in alphabet
        # and note 2 being wrapped
        if note_1_idx > note_2_idx:
            note_1_idx = -(len(notes._notes) - note_1_idx)
        interval = _intervals[note_2_idx-note_1_idx]
    
    return interval

def add_interval(note, interval):
    note = note.lower()
    octave = None
    
    if type(interval) == str:
        # going to prob break if you mix + and - _intervals, but that's future me problems
        interval_dist = 0
        if interval.find('-') > -1:
            for an_interval in interval.split('-'):
                # for catching if - is the first character
                if len(an_interval) > 0:
                    interval_dist = interval_dist - _intervals[an_interval]   
        elif interval.find('+') > -1:
            for an_interval in interval.split('+'):
                # for catching if + is the first character
                if len(an_interval) > 0:
                    interval_dist = interval_dist + _intervals[an_interval]
        else:
            interval_dist = interval_dist + _intervals[interval]
    else:
        # assumes an int was passed
        interval_dist = interval
    
    if note in notes._notes:
        note_idx = notes._make_note_list('c').index(note)
    else:
        note, octave = notes.split_note(note)
        note_idx = notes._make_note_list('c').index(note)
    
    new_note_idx = note_idx + interval_dist
    new_note = notes._make_note_list('c')[new_note_idx%len(notes._notes)]
    
    if octave is None:
        return new_note
    else:
        add_octave = new_note_idx//len(notes._notes)
        new_note_octave = add_octave+octave
        return new_note+str(new_note_octave)
  

# circles functions
def make_circle_of_fifths():
    circle_for_sharps = []
    first_note = 'c'
    note = first_note
    circle_for_sharps.append(first_note)
    for idx in range(int(len(notes._notes)/2)):
        fifth = notes_for_interval(note,'P5')
        note = fifth[1]
        circle_for_sharps.append(note)
        
    sharp_order =[]
    first_note = 'f'
    note = first_note
    sharp_order.append(first_note)
    for idx in range(int(len(notes._notes)/2)):
        fifth = notes_for_interval(note,'P5')
        note = fifth[1]
        sharp_order.append(note)
    
    return circle_for_sharps, sharp_order

def make_circle_of_fourths():
    circle_for_flats = []
    first_note = 'c'
    note = first_note
    circle_for_flats.append(first_note)
    for idx in range(int(len(notes._notes)/2)):
        fourth = notes_for_interval(note,'P4')
        note = fourth[1]
        if ("#" in note) or ("♯" in note):
            circle_for_flats.append(notes._sharp_to_flat(note))
        else:
            circle_for_flats.append(note)
            
    flats_order =[]
    first_note = 'b'
    note = first_note
    flats_order.append(first_note)
    for idx in range(int(len(notes._notes)/2)):
        fourth = notes_for_interval(note,'P4')
        note = fourth[1]
        if ("#" in note) or ("♯" in note):
            flats_order.append(notes._sharp_to_flat(note))
        else:
            flats_order.append(note)
        
    return circle_for_flats, flats_order

@dataclass
class Interval:
    name: str = field(init=False)
    first: notes.ScientificNote
    second: notes.ScientificNote

    def __post_init__(self):
        self.name = interval_for_notes(self.first.name, self.second.name)
