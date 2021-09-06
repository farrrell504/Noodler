import pandas as pd
from . import scales
from . import chords

def make_key(key_name):
    '''
    makes DataFrame for a specified key, should probably better explain
    
    keyword arguments:
    key_name -- [for now] tuple: (root_note, scale_type)
    '''
    import pandas as pd
    
    if (hasattr(key_name,'__iter__')) and (not type(key_name) == str):
        root_note = key_name[0]
        scale_type = key_name[1]
    else:
        root_note = key_name
        scale_type = 'major'
    
    note = root_note.lower()
    
    key_scale = scales.make_scale((note,scale_type))
    key_modes = scales.get_modes_of_scale(scale_type)
    key_numerals = scales.get_numerals_of_scale(scale_type)
    
    key_chords = chords.get_chords_for_key(key_name)

    key_dict = {}

    for idx, key_note, key_chord_numeral in zip(range(len(key_scale)), key_scale, key_chords):
        key_dict[idx+1] = {
            'note': key_note,
            'numeral': key_numerals[idx],
            'chord_triad': key_chords[key_chord_numeral],
            'type': chords.get_chord_type(key_chords[key_chord_numeral]),
            'mode': key_modes[idx]
        }
        
    return pd.DataFrame(key_dict).T