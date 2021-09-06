from dataclasses import dataclass, field
from . import pickup_electrical
from .pickup_magnetics import Pickup_Magnetics
from .strings_and_frets import get_fret_distances, get_action_per_string
from .strings_and_frets import String
from .musack.notes import split_note, _notes
from .musack.intervals import interval_for_notes
from .musack.chords import make_chord, chords
from .musack.keys import make_key
from .musack.scales import make_scale, get_intervals_of_scale


@dataclass
class GuitarProperties:
    number_of_strings: int
    number_of_frets: int
    scale_length: int
    distance_between_strings: float
    nut_distance: float = field(init=False)

    def __post_init__(self):
        self.nut_distance = self.scale_length


class Guitar:
    def __init__(
        self,
        tuning,
        gauges,
        scale_length,
        num_of_frets,
        action_recommendation,
        dist_between_strings=0.0104775,
        num_of_strings=6,
    ):
        """
        add more detail if you care later

        intuition says do this:
        +X -> thickest string to thinnest string
        +Y -> from bridge towards nut/head stock
        +Z -> up from pickups to strings

        math formulas would be easier though if
        +X -> from bridge towards nut/head stock
        +Y -> thinnest string to thickest string
        +Z -> up from pickups to strings
            this makes it so you can say at position x, i.e. which pickup
            using this for now
        """
        self.strings = {}
        self.pickups = {}

        # copy so that the global versions arent modified when reversed is called
        self.tuning = list(tuning)
        self.gauges = list(gauges)
        self.tuning.reverse()
        self.gauges.reverse()

        self.properties = GuitarProperties(
            number_of_strings=num_of_strings,
            number_of_frets=num_of_frets,
            scale_length=scale_length,
            distance_between_strings=dist_between_strings,
        )

        self.action_recommendation = action_recommendation

        self._setup()

    def _setup(self):
        ind = "    "
        print("Making Guitar:")
        print(f"{ind}Number of Strings:{self.properties.number_of_strings}")
        print(f"{ind}Number of Frets:{self.properties.number_of_frets}")
        print(f"{ind}Scale Length:{self.properties.scale_length}")
        print(f"{ind}Tuning:{self.tuning}")
        print(f"{ind}Gauges:{self.gauges}")
        print(f"{ind}Action Recommendation: {self.action_recommendation}")

        self.fret_distances = get_fret_distances(self.properties.number_of_frets, self.properties.scale_length)

        self._make_strings()

        self._get_frets_as_dataframe()

        self._make_keys()

        self._make_pickup(name="bridge", distance_between_bridge_and_pickup=0.05)

        self._make_volume_tone_pots()

    def _make_volume_tone_pots(self):
        from .toan import Toans
        print('ADD IN SPICE OF LC YA AWESOME HUMAN')

        self.toan_part = Toans()

    def _make_strings(self):
        import seaborn as sns
        import numpy as np

        colors = sns.color_palette("colorblind", n_colors=6).as_hex()
        action_per_string = np.flip(
            get_action_per_string(self.action_recommendation, self.properties.number_of_strings)
        )
        dist_between_strings = np.flip(
            np.linspace(
                -self.properties.distance_between_strings * (self.properties.number_of_strings - 1),
                0,
                self.properties.number_of_strings,
            )
        )

        for number, color, tune, gauge, action, dist in zip(
            range(self.properties.number_of_strings),
            colors,
            self.tuning,
            self.gauges,
            action_per_string,
            dist_between_strings,
        ):
            string = String(
                number + 1,
                tune,
                gauge,
                self.properties.scale_length,
                self.properties.number_of_frets,
                self.fret_distances,
                action,
                dist,
                color,
            )

            self.strings[number + 1] = string

    def _get_frets_as_dataframe(self):
        import pandas as pd

        dfs = []
        for string_num, string in self.strings.items():
            dfs.append(string.at_fret_df)

        self.frets = pd.concat(dfs)

        self.at_fret = {}
        for fret in range(self.properties.number_of_frets):
            self.at_fret[fret] = self.frets.loc[fret].set_index('note')

    def _plot_strings_yz_initial(self, return_fig=False):
        """
        plots initial location of strings,
        from perspective of looking from bridge to neck/headstock
        """
        import plotly.graph_objects as go

        fig = go.Figure()

        # # Set axes properties
        fig.update_xaxes(
            title="meters",
            range=[-(1.1 * self.properties.distance_between_strings * self.properties.number_of_strings), 0.01],
        )

        plot_range = np.abs(
            0.01 + -(1.1 * self.properties.distance_between_strings * self.properties.number_of_strings)
        )

        y_min = 0
        y_max = 0
        for string_num, string in self.strings.items():
            # Create scatter trace of text labels
            fig.add_trace(
                go.Scatter(
                    x=[string.y_position],
                    y=[string.z_position],
                    name=string.tuning,
                    line_color=string.color,
                    mode="markers",
                    marker=dict(size=string.diameter * 1e3),
                )
            )

            x0 = string.y_position - string.diameter / 2
            x1 = string.y_position + string.diameter / 2
            y0 = string.z_position - string.diameter / 2
            y1 = string.z_position + string.diameter / 2

            if y1 > y_max:
                y_max = y1 * 1.1
            if y0 < y_min:
                y_min = y0 * 1.1

            # Add circles
            fig.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line_color=string.color,
                fillcolor="Turquoise",
            )

        y_mid = np.mean((y_max, y_min))
        fig.update_yaxes(
            title="Height/Action from Pickup Magnet (meters)",
            range=[y_mid - (plot_range / 2), y_mid + (plot_range / 2)],
        )

        # Set figure size
        fig.update_layout(
            title="String Distance/Size", title_x=0.5, width=800, height=800
        )

        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )

        if return_fig:
            return fig
        else:
            fig.show(renderer="png")

    def plot_blank_fretboard(
        self,
        with_strings=True,
        with_fret_dots=True,
        with_fret_numbers=True,
        with_bridge=False,
        title="Fret Distances",
        x_title="Distance from nut (meters)",
        return_fig=False,
    ):
        import notebook_helpers as nh

        # if title hasn't been changed, lets name it something saying about this is the fret board
        if with_fret_numbers:
            if title == "Fret Distances":
                title = "Fretboard View"

        fig = nh.quick_plot(
            np.full(len(self.fret_distances), 0),
            self.fret_distances,
            title=title,
            type="scatter",
            return_fig=True,
        )

        fret_dots = [0, 3, 5, 7, 9, 12, 15, 17, 19]
        tickvals = []
        ticktext = []
        for idx, dist in enumerate(self.fret_distances):
            fig.add_vline(x=dist, layer="below")
            if idx in fret_dots:
                tickvals.append(dist)
                ticktext.append(str(idx))

        if with_fret_numbers:
            fig.update_layout(
                xaxis=dict(tickmode="array", tickvals=tickvals, ticktext=ticktext)
            )
            x_title = None

        if with_strings:
            for string_num, string in self.strings.items():
                fig.add_hline(
                    string.y_position,
                    line_color=string.color,
                    annotation_text=string.tuning,
                    layer="below",
                )

            last_string_idx = list(self.strings)[-1]
            fig.update_yaxes(
                range=[self.strings[last_string_idx].y_position * 1.1, 0.005]
            )

        if with_fret_dots:
            fig = self._add_fret_dots(fig)

        if with_bridge:
            fig.add_vline(0, line_width=4)
            fig.update_xaxes(range=[-self.properties.scale_length * 1.02, 0.005])

        fig.add_vline(-self.properties.scale_length, line_width=4)
        fig.update_xaxes(
            showgrid=False,
            zeroline=True,
            zerolinewidth=4,
            zerolinecolor="black",
            title=x_title,
        )
        fig.update_yaxes(
            showgrid=False, zeroline=False, showticklabels=False, title=None
        )

        if return_fig:
            return fig
        else:
            fig.show(renderer="png", width=1300, height=500)

    def _add_fret_dots(self, fig):
        fret_dots = [5, 7, 9, 15, 17, 19]

        for fret_number in fret_dots:
            x0, x1, y0, y1 = self._make_fret_dot_coord(
                3, 4, fret_number - 1, fret_number
            )
            fig.add_shape(
                type="circle",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                opacity=0.7,
                fillcolor="black",
                line_color="black",
                layer="below",
            )

        # make double dots on 12
        x0, x1, y0, y1 = self._make_fret_dot_coord(1, 2, 11, 12)
        fig.add_shape(
            type="circle",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            opacity=0.7,
            fillcolor="black",
            line_color="black",
            layer="below",
        )

        x0, x1, y0, y1 = self._make_fret_dot_coord(5, 6, 11, 12)
        fig.add_shape(
            type="circle",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            opacity=0.7,
            fillcolor="black",
            line_color="black",
            layer="below",
        )

        return fig

    def find_note(self, note):
        from .musack.notes import Note, ScientificNote

        if type(note) == Note:
            note = note.name
        elif type(note) == ScientificNote:
            note = note.note

        note = note.lower()
        fnd_notes = {}

        note_name, octave = split_note(note)

        for idx, (string_num, string) in enumerate(self.strings.items()):
            somethin = string.at_fret_df.note_name
            fnd_notes[string_num] = somethin[somethin == note_name].keys().to_list()

        return fnd_notes

    def find_chord(self, chord_name, *args):
        from .musack.chords import Chord

        if type(chord_name) != Chord:
            chord_name = make_chord(chord_name, *args)
        
        chord_notes = chord_name.notes

        return self._finish_finding(chord_notes)

    def find_scale(self, scale_name, *args):
        from .musack.scales import Scale

        if type(scale_name) != Scale:
            scale_name = make_chord(scale_name, *args)

        scale_notes = scale_name.notes

        return self._finish_finding(scale_notes)

    def _finish_finding(self, notes):
        import pandas as pd

        notes = [note.name for note in notes]

        found_possibilities_dict = {}

        for note in notes:
            found_possibilities_dict[note] = self.find_note(note)

        found_possibilities = pd.DataFrame(
            found_possibilities_dict, columns=notes
        )

        return found_possibilities        

    def plot_key(
        self, key_name, show_as="numeral", colormap="ch:start=.15,rot=-.4,dark=0.5"
    ):
        import seaborn as sns

        root_note = key_name[0]

        found_scale_possibilities = self.find_scale(key_name)

        title = "Key: {:} {:}".format(key_name[0].upper(), key_name[1])

        scale_type = key_name[
            1
        ]  # will break when you graduate to not using just tuples for chords

        if show_as == "number":
            note_text_list = list(
                zip(range(len(get_intervals_of_scale(scale_type)) + 1))
            )
            note_text_list.pop(0)
        elif show_as == "numeral":
            note_text_list = self.keys[root_note].numeral.to_list()
        elif show_as == "note":
            note_text_list = make_scale(key_name)
        else:
            raise ValueError(
                "Invalid value for show_as, use either number, numeral, or note"
            )

        colors = sns.color_palette(
            colormap, n_colors=len(found_scale_possibilities.columns)
        ).as_hex()

        self.plot_notes_on_fretboard(
            found_scale_possibilities, title, note_text_list, show_as, colors
        )

    def plot_scale(
        self, scale_name, show_as="number", colormap="ch:start=.15,rot=-.4,dark=0.5"
    ):
        import seaborn as sns

        found_scale_possibilities = self.find_scale(scale_name)

        title = "Scale: {:} {:}".format(scale_name[0].upper(), scale_name[1])

        scale_type = scale_name[
            1
        ]  # will break when you graduate to not using just tuples for chords
        if show_as == "number":
            note_text_list = list(
                zip(range(len(get_intervals_of_scale(scale_type)) + 1))
            )
            note_text_list.pop(0)
        elif show_as == "numeral":
            note_text_list = self.keys[scale_name[0].lower()].numeral.to_list()
        elif show_as == "note":
            note_text_list = make_scale(scale_name)
        else:
            raise ValueError(
                "Invalid value for show_as, use either number, numeral, or note"
            )

        colors = sns.color_palette(
            colormap, n_colors=len(found_scale_possibilities.columns)
        ).as_hex()

        self.plot_notes_on_fretboard(
            found_scale_possibilities, title, note_text_list, show_as, colors
        )

    def plot_chord(self, chord_name, show_as="number"):
        found_chord_possibilities = self.find_chord(chord_name)

        title = "Chord: {:} {:}".format(chord_name[0].upper(), chord_name[1])

        chord_type = chord_name[
            1
        ]  # will break when you graduate to not using just tuples for chords
        note_text_list = chords[chord_type]["numbers"]

        self.plot_notes_on_fretboard(
            found_chord_possibilities, title, note_text_list, show_as
        )

    def plot_notes_on_fretboard(
        self, what_notes, title, note_text_list=None, show_as="note", colors=None
    ):
        import seaborn as sns
        import plotly.graph_objects as go

        fig = self.plot_blank_fretboard(x_title=None, title=title, return_fig=True)

        if colors is None:
            if hasattr(what_notes, "columns"):
                n_to_make = len(what_notes.columns)
            else:
                n_to_make = len(what_notes)
            colors = sns.color_palette(
                "colorblind", n_colors=self.properties.number_of_strings + n_to_make
            ).as_hex()[self.properties.number_of_strings :]

        for idx, (note, string_stuff) in enumerate(what_notes.items()):
            first_time = True
            for string, s_val in string_stuff.items():
                for fret in s_val:
                    # get position to put the note dot
                    xpos = self.strings[string].at_fret[fret]["x_position (m)"]

                    if not fret == 0:
                        xpos_pre = self.strings[string].at_fret[fret - 1]["x_position (m)"]
                        xpos = (xpos + xpos_pre) / 2

                    yz = self.strings[string].y_position

                    # toggle what to show over the note dot
                    if show_as == "note":
                        note_text = note
                    else:
                        note_text = note_text_list[idx]

                    # add dot and note_text to fretboard
                    fig.add_trace(
                        go.Scatter(
                            x=[xpos],
                            y=[yz],
                            name=note,
                            text=note_text,
                            line_color=colors[idx],
                            mode="markers+text",
                            legendgroup=note,
                            showlegend=first_time,
                            marker=dict(size=30),
                            textfont=dict(
                                size=20,
                            ),
                        )
                    )

                    # makes sure duplicate values aren't added to the legend
                    if first_time:
                        first_time = False
        fig.show(renderer="png", width=1300, height=500)

    def _make_fret_dot_coord(self, s0, s1, f0, f1):
        """
        s0 -- bottom string number
        s1 -- top string number
        f0 -- back fret number
        f1 -- front fret number
        """
        x0 = self.strings[s0].at_fret[f0]["x_position (m)"]
        x1 = self.strings[s0].at_fret[f1]["x_position (m)"]
        y0 = self.strings[s0].y_position
        y1 = self.strings[s1].y_position

        # mid point of circle
        xmid = (x1 + x0) / 2
        ymid = (y1 + y0) / 2

        # radius of circle in x and y
        xrad = xmid - x1
        yrad = ymid - y1

        # radius of circle averaged
        rad = (xrad + yrad) / 2

        # scale factor
        scf = 0.3

        # top and bottom in x and y axes
        x0 = xmid - (xrad * scf)
        x1 = xmid + (xrad * scf)
        y0 = ymid - (yrad * scf)
        y1 = ymid + (yrad * scf)

        return x0, x1, y0, y1

    def _make_keys(self):
        key_dict = {}
        for note in _notes:
            key_dict[note] = make_key((note, "major"))

        self.keys = key_dict

    def _search_keys_for_note(self, note):
        """return list of keys with give note"""
        note = note.lower()

        possible_keys = []
        for key, key_df in self.keys.items():
            if key_df["note"].str.fullmatch(note).any():
                possible_keys.append(key)

        return possible_keys

    def _search_keys_for_notes(self, _notes):
        """ builds on above"""

        first = True
        for note in _notes:
            note = note.lower()
            pos_key = self._search_keys_for_note(note)
            if first:
                pos_keys = pos_key
                first = False
            else:
                pos_keys = set(pos_keys).intersection(pos_key)

        return pos_keys

    def guess_chord(self, frets, plot=False, plot_possible_keys=False):
        """
        give list or tuple of ints representing fret number
        order is from thickest string to thinnest

        negative number means ignore

        Optional keyword arguments:
        plot -- plot frets as _notes (default False)
        plot_possible_keys -- plot frets in possible keys (default False)
        """
        import pandas as pd

        top_down_fret_numbers = self._reverse_fret_list(frets)

        fret_list = []
        for fret, (string_num, string) in zip(
            top_down_fret_numbers, self.strings.items()
        ):
            # None is passed when no frets are played on a string
            if fret is not None:
                # if multiple frets played on a single string (think ornaments or grace _notes like hammer ons or pull offs, etc.)
                if hasattr(fret, "__iter__"):
                    for a_fret in fret:
                        fret_list.append(string.at_fret[a_fret])
                else:
                    fret_list.append(string.at_fret[fret])

        notes_df = pd.DataFrame(fret_list)

        pos_keys = self._search_keys_for_notes(notes_df.note_name)
        print("Found possible keys: ", pos_keys)

        #         note_df_dict = {}
        #         for string_idx, fret in enumerate(top_down_fret_numbers):
        #             note = self.strings[string_idx].at_fret[fret]['note']
        #             note_df_dict[note] = {}
        #             string_number_adj = notes_df[notes_df.note.str.fullmatch(note)].string_number.values[0]-1
        #             note_df_dict[note][string_number_adj] = [notes_df[notes_df.note.str.fullmatch(note)].fret.values[0]]
        #         note_df_dict

        pos_chords = {}
        for a_root_note in reversed(notes_df.note.to_list()):
            # root_note = df.note.to_list()[-1]
            pos_chord_intervals = []
            for a_note in reversed(notes_df.note.to_list()):
                interval = interval_for_notes(a_root_note, a_note)
                #         if interval.split('+')[0] not in pos_chord_intervals and not interval.split('+')[0] == 'u':
                if interval.find("+") > 0:
                    pos_chord_intervals.append(interval.split("+")[0])
                elif interval.find("-") > -1:
                    pos_chord_intervals.append("-" + interval.split("-")[1])
                else:
                    pos_chord_intervals.append(interval.split("+")[0])
            pos_chords[a_root_note] = pos_chord_intervals
        pos_chords

        pos_chords_df = pd.DataFrame(pos_chords, index=list(pos_chords.keys()))

        # find unique _notes?
        unique_notes = []
        for a_note in notes_df.note_name.to_list():
            if a_note not in unique_notes:
                unique_notes.append(a_note)
        print("unique _notes: ", unique_notes)

        pos_chords = []
        for key in pos_keys:
            for idx, chord in enumerate(self.keys[key].chord_triad):
                if all(item in chord for item in unique_notes):
                    print(
                        "found matching chord! key: ",
                        key,
                        " chord: ",
                        chord,
                        " type: ",
                        self.keys[key].type.iloc[idx],
                        " numeral ",
                        self.keys[key].numeral.iloc[idx],
                    )
                    # tuple( key, chord, type, numeral )
                    pos_chords.append(
                        (
                            key,
                            chord,
                            self.keys[key].type.iloc[idx],
                            self.keys[key].numeral.iloc[idx],
                        )
                    )
        if plot:
            self.plot_frets(frets, title=f"Possible keys: {pos_keys}")

        if plot_possible_keys:
            for key in pos_keys:
                self.plot_key((key, "major"))

        return notes_df, pos_chords_df

    def _reverse_fret_list(self, frets):
        """
        reverses list of frets to make it intuitive (to me) to type

        allows for writing frets in terms of thickets to thinnest strings ie the direction you are looking down onto the guitar while playing
        """
        top_down_fret_numbers = []
        for fret in reversed(frets):
            if (type(fret) == tuple) and (not fret):
                fret = None
            elif fret is None:
                fret = None
            elif type(fret) == str:
                fret = None
            elif hasattr(fret, "__iter__"):
                fret = list(fret)
            top_down_fret_numbers.append(fret)

        return top_down_fret_numbers

    def plot_frets(self, frets, title="Some frets plotted"):
        """
        give list or tuple of ints representing fret number
        order is from thickest string to thinnest

        negative number means ignore
        """

        top_down_fret_numbers = self._reverse_fret_list(frets)

        note_df_dict = {}
        for string_idx, fret in enumerate(top_down_fret_numbers):
            # None is passed when no frets are played on a string
            if fret is not None:
                # if multiple frets played on a single string (think ornaments or grace _notes like hammer ons or pull offs, etc.)
                if hasattr(fret, "__iter__"):
                    for a_fret in fret:
                        note = self.strings[string_idx + 1].at_fret[a_fret]["note"]
                        if note not in note_df_dict:
                            note_df_dict[note] = {}
                        if string_idx + 1 not in note_df_dict[note]:
                            note_df_dict[note][string_idx + 1] = [a_fret]
                        else:
                            note_df_dict[note][string_idx + 1].append(a_fret)
                else:
                    note = self.strings[string_idx + 1].at_fret[fret]["note"]
                    note_df_dict[note] = {}
                    note_df_dict[note][string_idx + 1] = [fret]

        self.plot_notes_on_fretboard(note_df_dict, title=title)

    def plot_pluck(
        self,
        at_string,
        at_fret,
        pick_x=0.1,
        pick_y=0.002,
        x=0.029,
        axis_of_interest=None,
        fs=48000,
        n=1,
        sum_n=1,
        title=None,
        plot_time=True,
        plot_frequency=True,
        plot_y=True,
        plot_z=True,
        return_fig=False,
    ):
        """
        plots string movement

        Keyword arguments:
        at_string -- either single string number or iterable of string numbers
        at_fret -- either single fret number or iterable of fret numbers

        Optional keyword arguments
        Yi -- y axis initial [m] (default 0.01)
        Zi -- z axis initial [m] (default 0.005)
        x -- position to simulate on string. estimate of bridge pickup for now [m] (default 0.029)
        axis_of_interest -- if only interested in single axis, just give it here (default None)
        fs -- sampling rate [Hz] (default 48000)
        l -- length [s] (default 0.1)
        return_fig -- returns fig instead of showing it (default False)
        """

        import seaborn as sns

        if hasattr(at_fret, "__iter__"):
            frets = len(at_fret)
        else:
            frets = 1

        if hasattr(at_string, "__iter__"):
            colors = sns.color_palette(
                "colorblind", n_colors=frets * len(at_string)
            ).as_hex()
            first = True
            for s, a_string_num in enumerate(at_string):
                idx = s * (frets)
                start = idx
                end = frets + idx
                if first:
                    fig = self.strings[a_string_num].plot_pluck(
                        at_fret,
                        pick_x=pick_x,
                        pick_y=pick_y,
                        x=x,
                        axis_of_interest=axis_of_interest,
                        fs=fs,
                        n=n,
                        sum_n=sum_n,
                        colors=colors[start:end],
                        plot_time=plot_time,
                        plot_frequency=plot_frequency,
                        plot_y=plot_y,
                        plot_z=plot_z,
                        return_fig=True,
                    )
                    first = False
                else:
                    fig = self.strings[a_string_num].plot_pluck(
                        at_fret,
                        pick_x=pick_x,
                        pick_y=pick_y,
                        x=x,
                        axis_of_interest=axis_of_interest,
                        fs=fs,
                        n=n,
                        sum_n=sum_n,
                        colors=colors[start:end],
                        plot_time=plot_time,
                        plot_frequency=plot_frequency,
                        plot_y=plot_y,
                        plot_z=plot_z,
                        fig=fig,
                        return_fig=True,
                    )
        else:
            colors = sns.color_palette("colorblind", n_colors=frets).as_hex()
            fig = self.strings[at_string].plot_pluck(
                at_fret,
                pick_x=pick_x,
                pick_y=pick_y,
                x=x,
                axis_of_interest=axis_of_interest,
                fs=fs,
                n=n,
                sum_n=sum_n,
                colors=colors,
                plot_time=plot_time,
                plot_frequency=plot_frequency,
                plot_y=plot_y,
                plot_z=plot_z,
                return_fig=True,
            )

        if return_fig:
            return fig
        else:
            fig.show(renderer="png", width=1300, height=500)

    def _make_pickup(self, name, distance_between_bridge_and_pickup):
        from .pickup import Pickup

        self.pickups[name] = Pickup(
            name,
            num_of_magnets=self.properties.number_of_strings,
            distance_between_magnets=self.properties.distance_between_strings,
            distance_between_bridge_and_pickup=distance_between_bridge_and_pickup,
        )

    def _make_pickup_magnetics(self, distance_between_bridge_and_pickup=0.029):
        return Pickup_Magnetics(
            distance_between_magnets=self.properties.distance_between_strings,
            distance_between_bridge_and_pickup=distance_between_bridge_and_pickup,
            num_of_magnets=self.properties.number_of_strings,
        )

    def plot_pickup_mag(
        self,
        pickup_name,
        mag_num,
        string_number=None,
        show_mag_field_direction=True,
        from_string_perspective=True,
    ):
        import numpy as np
        import magpylib as magpy
        import matplotlib.pyplot as plt
        from magpylib import displaySystem
        import seaborn as sns

        s = self.pickups[pickup_name].magnetics.magnets.sources[mag_num]

        if string_number is None:
            string_number = mag_num + 1

        mag_height = s.dimension[1]
        window_length = s.position[1] + (mag_height * 2)

        if from_string_perspective:
            string_center = mag_height + (self.strings[string_number].z_position * 1e3)
        else:
            string_center = 0

        x_d = [-10, 10]
        y_d = self.strings[string_number].y_position * 1e3
        z_d = [-10 + string_center, 10 + string_center]

        fig1 = plt.figure(figsize=(10, 15))
        axsA = fig1.add_subplot(2, 1, 1, projection="3d")
        axsB = fig1.add_subplot(2, 1, 2)

        # position grid
        xs = np.linspace(x_d[0], x_d[1], 50)
        zs = np.linspace(z_d[0], z_d[1], 50)
        posis = np.array([(x, y_d, z) for z in zs for x in xs])
        X, Y = np.meshgrid(xs, zs)

        # for i,s in enumerate(c.sources):
        # display system on respective axes, use marker to zoom out
        magpy.displaySystem(
            s,
            subplotAx=axsA,
            markers=[(10, 0, mag_height + 10)],
            suppress=True,
            direc=show_mag_field_direction,
        )
        axsA.plot(
            [x_d[0], x_d[1], x_d[1], x_d[0], x_d[0]],
            [y_d, y_d, y_d, y_d, y_d],
            [z_d[0], z_d[0], z_d[1], z_d[1], z_d[0]],
        )

        # plot field on respective axes
        B = np.array([s.getB(p) for p in posis]).reshape(50, 50, 3)
        #     cmap = sns.color_palette("rocket", as_cmap=True)
        cmap = plt.cm.get_cmap("coolwarm")
        cs = axsB.pcolor(
            X, Y, np.linalg.norm(B, axis=2), cmap=cmap, shading="auto"
        )  # amplitude
        axsB.streamplot(
            X, Y, B[:, :, 0], B[:, :, 2], color="k", linewidth=1
        )  # field lines
        fig1.colorbar(cs)

        plt.show()

    def plot_pickup_mags(
        self,
        pickup_name,
        x_n=33,
        z_n=44,
        show_mag_field_direction=True,
        from_string_perspective=True,
        string_number=1,
        mag_dist=None,
    ):
        """
        plots all magnets
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import magpylib as mp
        # from magpylib import displaySystem

        c = self.pickups[pickup_name].magnetics.magnets

        mag_height = c.sources[string_number - 1].dimension[1]

        if from_string_perspective:
            string_center = mag_height + (self.strings[string_number].z_position * 1e3)
        else:
            string_center = 0

        if mag_dist is None:
            mag_dist = self.properties.distance_between_strings * 1e3
            # calculate because lazy me forgot to pass it. lets hope they are all the same...
            # mag_dist = c.sources[1].position[1] - c.sources[0].position[1]

        x_d = [-10, 10]
        y_d = (string_number - 1) * mag_dist
        z_d = [-10 + string_center, 10 + string_center]

        # calculate B-field on a grid
        xs = np.linspace(x_d[0], x_d[1], x_n)
        zs = np.linspace(z_d[0], z_d[1], z_n)
        POS = np.array([(x, y_d, z) for z in zs for x in xs])
        Bs = c.getB(POS).reshape(z_n, x_n, 3)  # <--VECTORIZED

        # create figure
        fig = plt.figure(figsize=(9, 5))
        ax1 = fig.add_subplot(121, projection="3d")  # 3D-axis
        ax2 = fig.add_subplot(122)  # 2D-axis

        # display system geometry on ax1
        # mp.display(c, subplotAx=ax1, suppress=True, direc=show_mag_field_direction)
        mp.display(c, axis=ax1, show_direction=True)
        ax1.plot(
            [x_d[0], x_d[1], x_d[1], x_d[0], x_d[0]],
            [y_d, y_d, y_d, y_d, y_d],
            [z_d[0], z_d[0], z_d[1], z_d[1], z_d[0]],
        )

        # display field in xz-plane using matplotlib
        X, Z = np.meshgrid(xs, zs)
        U, V = Bs[:, :, 0], Bs[:, :, 2]
        ax2.streamplot(X, Z, U, V, color=np.log(U ** 2 + V ** 2))

        plt.show()

class StandardNoodle(Guitar):
    def __init__(self):
        from .strings_and_frets import tunings, gauges, scale_lengths

        num_of_frets = 22
        tuning = tunings["standard"]
        gauges = gauges["heavy_bottom"]
        scale_length = scale_lengths["stratocaster"]

        super().__init__(tuning=tuning, gauges=gauges, scale_length=scale_length, num_of_frets=num_of_frets, action_recommendation="fralin")    

class SevenStringNoodle(Guitar):
    def __init__(self):
        from .strings_and_frets import tunings, gauges, scale_lengths

        num_of_frets = 22
        tuning = tunings["standard_7"]
        gauges = gauges["heavy_bottom_7"]
        scale_length = scale_lengths["stratocaster"]

        super().__init__(tuning=tuning, gauges=gauges, scale_length=scale_length, num_of_frets=num_of_frets, action_recommendation="fralin")    

class Strandberg(Guitar):
    def __init__(self):
        from .strings_and_frets import tunings, gauges, scale_lengths

        num_of_frets = 22
        tuning = tunings["standard"]
        gauges = gauges["regular_slinky"]
        scale_length = scale_lengths["stratocaster"]

        self.scale_lengths = [.635, .637, .6393, .6414, .6434, .6477]
        self.tensions = [7.36, 6.98, 7.52, 8.31, 8.6, 7.63]

        super().__init__(tuning=tuning, gauges=gauges, scale_length=scale_length, num_of_frets=num_of_frets, action_recommendation="fralin")    
