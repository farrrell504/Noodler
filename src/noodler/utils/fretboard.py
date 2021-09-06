## put blank fretboard printing class here

class Fretboard:
    def __init__(self,
        num,
        scale_length,
    ):
        self.num_of_frets = num
        self.scale_length = scale_length

    def make_frets(self):
        # making a dict for now
        # will want properties like how it impacts tension
        # frequency and yada later I imagine
        import pandas as pd
        import numpy as np
        from ..musack import note_to_frequency, add_interval, split_note
        from ..strings_and_frets import get_fret_distances
        
        self.distances = get_fret_distances(self.num_of_frets, self.scale_length)

        at_fret = {}
        for fret in range(self.num_of_frets):
            # im lazy, theres a better way to do this im sure
            note = add_interval(self.tuning, fret)
            note_name, octave = split_note(note)

            at_fret[fret] = {
                "note": note,
                "fret": fret,
                "string_number": self.number,
                "note_name": note_name,
                "note_octave": octave,
                "frequency_fundamental (Hz)": note_to_frequency(note),
                "x_position (m)": self.distances[fret],
                "y_position (m)": self.y_position,
                "position (m)": np.array([self.distances[fret], self.y_position]),
                "linear_mass_density (kg/m)": self.get_linear_mass_density_at_fret(
                    fret
                ),
                "tension (N)": self.get_tension_at_fret(fret),
            }

        self.at_fret_df = pd.DataFrame(at_fret).T

        self.at_fret = {}
        for fret in range(self.num_of_frets):
            self.at_fret[fret] = self.at_fret_df.loc[fret]

    def plot(
            with_strings=True,
            with_fret_dots=True,
            with_fret_numbers=True,
            with_bridge=False,
            title="Fret Distances",
            x_title="Distance from nut (meters)",
            return_fig=False,
    ):
        import notebook_helpers as nh
        import numpy as np

        # if title hasn't been changed, lets name it something saying about this is the fret board
        if with_fret_numbers:
            if title == "Fret Distances":
                title = "Fretboard View"

        fig = nh.quick_plot(
            np.full(len(self.distances), 0),
            self.distances,
            title=title,
            type="scatter",
            return_fig=True,
        )

        fret_dots = [0, 3, 5, 7, 9, 12, 15, 17, 19]
        tickvals = []
        ticktext = []
        for idx, dist in enumerate(self.distances):
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
            fig.update_xaxes(range=[-self.scale_length * 1.02, 0.005])

        fig.add_vline(-self.scale_length, line_width=4)
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
            x0, x1, y0, y1 = self._make_fret_dot_coord(fret_number - 1, fret_number)
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
        x0, x1, y0, y1 = self._make_fret_dot_coord(11, 12)
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

        x0, x1, y0, y1 = self._make_fret_dot_coord(11, 12)
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

    def _make_fret_dot_coord(self, f0, f1):
        """
        f0 -- back fret number
        f1 -- front fret number
        """
        x0 = self.at_fret[f0]["x_position (m)"]
        x1 = self.at_fret[f1]["x_position (m)"]

        # mid point of circle
        xmid = (x1 + x0) / 2

        # radius of circle in x and y
        xrad = xmid - x1

        # radius of circle averaged
        rad = xrad

        # scale factor
        scf = 0.3

        # top and bottom in x and y axes
        x0 = xmid - (xrad * scf)
        x1 = xmid + (xrad * scf)
        y0 = self.y_position - (self.y_position * scf * 0.1)
        y1 = self.y_position + (self.y_position * scf * 0.1)

        return x0, x1, y0, y1
