import numpy as np
from .musack.notes import _notes, note_to_frequency, split_note
from .musack.intervals import add_interval

standard_tuning = ["e2", "a2", "d3", "g3", "b3", "e4"]

tunings = {
    "standard": standard_tuning,
    "drop_d": ["d2", "a2", "d3", "g3", "b3", "e4"],
    "standard_7": ["b1", "e2", "a2", "d3", "g3", "b3", "e4"],
    "baritone_P4": ["b1", "e2", "a2", "d3", "f#3", "b3"],
    "baritone_P5": ["a1", "d2", "g2", "c3", "e3", "a3"],
    "baritone_M3": ["c2", "f2", "bb2", "eb3", "g3", "c4"],
    "bass_standard": ["e1", "a1", "d2", "g2"],
    "bass_5_standard": ["b0", "e1", "a1", "d2", "g2"],
}

scale_lengths = {
    "stratocaster": 0.6477,  # based off typical Fender length of 25.5", put in meters
    "p_bass": 0.8636,  # based off 34" fender p bass
    "bass_vi": 0.762,  # based off 30" fender bass vi
    "jaguar_baritone": 0.7239,  # based off 28.5" fender jaguar barione
}

_r = 2 ** (1 / len(_notes))
_magic_number = _r / (_r - 1)
_magic_number

# fret stuff


def get_fret_distance(fret_number, scale_length=scale_lengths["stratocaster"]):
    """
    to get distance for a specific fret
    """
    dist_if_nut_0 = scale_length * (1 - (1 / (_r ** fret_number)))
    # offset since bridge is 0 for us
    dist = dist_if_nut_0 - scale_length
    return dist


def get_fret_distances(num_of_frets, scale_length=scale_lengths["stratocaster"]):
    import numpy as np

    distance = 0
    fret_distances = [0]
    for fret in range(num_of_frets):
        location = scale_length - distance
        scaling_factor = location / _magic_number
        distance = distance + scaling_factor
        fret_distances.append(distance)

    # x = 0 is the bridge, so offset by scale length
    fret_distance_arr = np.array(fret_distances) - scale_length

    return fret_distance_arr


# string stuff

# needs to go
action = {
    "fralin": {
        "thickest_string_action": (1 / 8) / 39.37,
        "thinnest_string_action": (1 / 16) / 39.37,
    }
}


def get_action_per_string(recommendation, num_of_strings):
    """
    Currently does a linear interpolation between min and max strings action

    TODO account for the slight arch

    Keyword agruments:
    recommendation -- whose recommendation to use (possibilities found in action dict)
    num_of_strings -- number of strings
    """
    import numpy as np

    thickest_string_action = action[recommendation]["thickest_string_action"]
    thinnest_string_action = action[recommendation]["thinnest_string_action"]
    action_per_string = np.linspace(
        thickest_string_action, thinnest_string_action, num_of_strings
    )

    return action_per_string


gauges = {
    "heavy_bottom": [52, 42, 30, 17, 13, 10],  # based off heavy bottom
    "regular_slinky": [46, 36, 26, 17, 13, 10],  # based off regular slinky
    "heavy_bottom_7": [62, 52, 42, 30, 17, 13, 10],  # based off heavy bottom 7 strings
    "baritone_slinky": [72, 56, 44, 30, 18, 13],  # based off regular slinky
    "bass_regular_slinky": [105, 85, 70, 50],  # based off bass regular slinky
    "bass_regular_slinky_5": [130, 100, 80, 65, 45],  # based off bass regular slinky
}


def string_gauge_to_meters(string_gauges):
    """
    converts commonly given string gauge to useful metric of the strings diameter in millimeters
    """
    string_diameter = []
    [string_diameter.append(string_gauge * 2.54e-5) for string_gauge in string_gauges]
    return string_diameter


# linear mass densities
# from https://www.daddario.com/globalassets/pdfs/accessories/tension_chart_13934.pdf
string_linear_mass_density = {
    10: 0.00002215 * 0.0115212,
    13: 0.00003744 * 0.0115212,
    17: 0.00006402 * 0.0115212,
    18: 0.00007177 * 0.0115212,
    26: 0.00012671 * 0.0115212,
    30: 0.00017236 * 0.0115212,
    36: 0.00023964 * 0.0115212,
    42: 0.00032279 * 0.0115212,
    44: 0.00035182 * 0.0115212,
    46: 0.00038216 * 0.0115212,
    52: 0.00048109 * 0.0115212,
    56: 0.00057598 * 0.0115212,
    62: 0.00070697 * 0.0115212,
    72: 0.00094124 * 0.0115212,
}


class String:
    def __init__(
        self,
        number,
        tuning,
        gauge,
        scale_length,
        num_of_frets,
        fret_distances,
        action,
        dist_to_first_string,
        color,
    ):
        import numpy as np

        self.number = number  # guytar players love referring to their strings by number
        self.tuning = tuning
        self.frequency_fundamental_open = note_to_frequency(tuning)
        self.gauge = gauge
        self.linear_mass_density = string_linear_mass_density[self.gauge]

        # idk which is more intuitive..?
        self.scale_length = scale_length
        self.x_length = self.scale_length
        self.x_max = self.scale_length

        self.num_of_frets = num_of_frets
        self.fret_distances = fret_distances
        self.color = color

        self.diameter = string_gauge_to_meters([gauge])[0]
        self.y_position = dist_to_first_string

        self.yz_area = np.pi * (self.diameter / 2) ** 2
        self.area = (
            self.yz_area
        )  # idk which is better, one is annoying to look at, but descriptive, other is simple, but dont know which axes its describing

        self.action = action
        self.z_position = self.action

        self._make_frets()

        open_movement = self.pluck(0)

        self.y_movement_open = open_movement[0]
        self.z_movement_open = open_movement[1]

        self._make_properties()

    def __repr__(self):
        return f"string {self.number}"

    def _make_properties(self):
        import pandas as pd

        properties = {
            "tuning": self.tuning,
            "gauge": self.gauge,
            "action": self.action,  # (m)
            "diameter": self.diameter,  # (m)
            "area": self.area,  # (m^2)
            "length": self.x_length,  # (m)
            "x_max": self.x_max,  # (m)
            "y_position": self.y_position,  # (m)
            "z_position": self.z_position,  # (m)
            "frequency_fundamental": note_to_frequency(self.tuning),  # (Hz)
            "linear_mass_density": self.linear_mass_density,  # (kg/m)
            "tension": self._get_tension_at_fret(0),  # (N)
        }

        self.properties = pd.DataFrame(properties, index=[self.number])

    def pluck(
        self,
        at_fret,
        pick_x=-0.1,
        pick_y=-0.002,
        x=0.05,
        axis_of_interest=None,
        fs=48000,
        l=0.1,
        n=1,
        sum_n=1,
        auto_n=True,
    ):
        """
        Keyword arguments
        at_fret -- self explanatoryweroiuwehf

        Optional keyword arguments
        pick_x -- distance from bridge where pick touches string
        pick_y -- distance pick has moved the string before releasing string
        x -- position to simulate on string. estimate of bridge pickup for now [m] (default 0.029)
        axis_of_interest -- if only interested in single axis, just give it here (default None)
        fs -- sampling rate [Hz] (default 48000)
        l -- length [s] (default 0.1)
        n -- harmonic number (default 1)
        sum_n -- total number of harmonics to model past n (default 1)
        auto_n -- automatically determine the number of harmonics, based off frequemcy upto 20kHz, top end of human hearing (default True)
        """
        import notebook_helpers as nh  # temp

        t = np.arange(l * fs) / fs

        # we just care about the difference
        pick_x = np.abs(pick_x)
        pick_y = np.abs(pick_y)

        L = np.abs(self.at_fret[at_fret]["x_position"])

        def initial_position(n, pick_x, pick_y):
            """
            for derivation see: https://www.acs.psu.edu/drussell/Demos/Pluck-Fourier/Pluck-Fourier.html

            Keyword arguments
            n -- harmonic number
            pick_x -- distance from bridge where pick touches string
            pick_y -- distance pick has moved the string before releasing string
            """
            first_term = (2 * pick_y) / ((n ** 2) * (np.pi ** 2))
            second_term = (L ** 2) / (pick_x * (L - pick_x))
            third_term = np.sin((pick_x * n * np.pi) / L)
            initial = first_term * second_term * third_term
            # print(
            #     "initial: {:}, n: {:}, pick_x, {:}, pick_y: {:}, L: {:}".format(
            #         initial, n, pick_x, pick_y, L
            #     )
            # )
            return initial

        def movement(n):
            w_n = self.get_w_n(at_fret, n)
            y_initial = initial_position(n, pick_x, pick_y)
            y_movement = y_initial * np.cos(w_n * t) * np.sin((np.pi * x * n) / L)
            z_movement = (
                0.05 * y_initial * np.cos(w_n * t) * np.sin((np.pi * x * n) / L)
            )
            # print(" y max: ", np.max(y_movement))
            return y_movement, z_movement

        if auto_n:
            f = note_to_frequency(self.at_fret[at_fret]["note"])
            f20k = 20e3
            sum_n = int(np.ceil(f20k / f))
            n = 1

        y = np.full(len(t), self.y_position)
        z = np.full(len(t), self.z_position)
        for idn in range(sum_n):
            y_n, z_n = movement(idn + n)
            y += y_n
            z += z_n

        if axis_of_interest is None:
            return y, z, t
        elif axis_of_interest.lower() == "y":
            return y
        elif axis_of_interest.lower() == "z":
            return z

    def plot_pluck(
        self,
        at_fret,
        pick_x=-0.1,
        pick_y=-0.002,
        x=0.05,
        axis_of_interest=None,
        fs=48000,
        n=1,
        sum_n=1,
        title=None,
        colors=None,
        plot_time=True,
        plot_frequency=True,
        plot_animation=True,
        plot_y=True,
        plot_z=True,
        fig=None,
        return_fig=False,
    ):
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots
        import seaborn as sns

        height_mult = 0
        rows = 0
        cols = 0
        subplot_titles = []
        rows_time = []
        cols_time = []
        rows_freq = []
        cols_freq = []
        if plot_y:
            cols += 1
        if plot_z:
            cols += 1
        if plot_time:
            height_mult += 1
            rows += 1
            cols_time = [col + 1 for col in range(cols)]
            rows_time.append(rows)
            if plot_y:
                subplot_titles.append("Y-Axis Movement over Time")
            if plot_z:
                subplot_titles.append("Z-Axis Movement over Time")

        if plot_frequency:
            height_mult += 1
            rows += 1
            cols_freq = [col + 1 for col in range(cols)]
            rows_freq.append(rows)
            if plot_y:
                subplot_titles.append("Y-Axis Movement over Frequency")
            if plot_z:
                subplot_titles.append("Z-Axis Movement over Frequency")

        # print(f"plot_y: {plot_y}, plot_z: {plot_z}")
        # print(f"plot_time: {plot_time}, plot_frequency: {plot_frequency}")

        # print(f"row: {rows}, cols: {cols}")
        # print(f"rows_time: {rows_time}, cols_time: {cols_time}")
        # print(f"rows_freq: {rows_freq}, cols_freq: {cols_freq}")

        if fig is None:
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=subplot_titles,
            )
            if plot_time:
                fig.update_yaxes(
                    title_text="Distance (m)", row=rows_time[0], col=cols_time[0]
                )
                if plot_y and plot_z:
                    fig.update_yaxes(
                        title_text="Distance (m)", row=rows_time[0], col=cols_time[1]
                    )
                fig.update_xaxes(
                    title_text="Time (s)", row=rows_time[0], col=cols_time[0]
                )
                if plot_y and plot_z:
                    fig.update_xaxes(
                        title_text="Time (s)", row=rows_time[0], col=cols_time[1]
                    )
            if plot_frequency:
                fig.update_yaxes(
                    title_text="dBDistance/Hz (m)", row=rows_freq[0], col=cols_freq[0]
                )
                if plot_y and plot_z:
                    fig.update_yaxes(
                        title_text="dBDistance/Hz (m)",
                        row=rows_freq[0],
                        col=cols_freq[1],
                    )
                fig.update_xaxes(
                    title_text="Frequency (Hz)", row=rows_freq[0], col=cols_freq[0]
                )
                if plot_y and plot_z:
                    fig.update_xaxes(
                        title_text="Frequency (Hz)", row=rows_freq[0], col=cols_freq[1]
                    )
            fig.update_layout(
                title="String {:}; Fret: {:}; X position: {:}".format(
                    self.number, at_fret, x
                ),
                title_x=0.5,
            )
        else:
            title = fig.layout.title.text
            title_strings = title.split(";")[0]
            title_frets = title.split(";")[1]
            title_xpos = title.split(";")[2]
            title_strings = title_strings + "," + str(self.number)
            title = title_strings + ";" + title_frets + ";" + title_xpos
            fig.update_layout(
                title=title,
                title_x=0.5,
            )

        def add_pluck(
            at_this_fret,
            color,
            pick_x=pick_x,
            pick_y=pick_y,
            x=x,
            axis_of_interest=axis_of_interest,
            fs=fs,
            n=n,
            sum_n=sum_n,
        ):
            from scipy import signal

            trace_name = "str:{:},fret:{:}".format(self.number, at_this_fret)

            if plot_time:
                f = note_to_frequency(self.at_fret[at_this_fret]["note"])
                y, z, t = self.pluck(
                    at_this_fret,
                    pick_x=pick_x,
                    pick_y=pick_y,
                    x=x,
                    axis_of_interest=axis_of_interest,
                    fs=fs,
                    l=2 * (1 / f),
                    n=n,
                    sum_n=sum_n,
                )
                col_idx = 0
                if plot_y:
                    fig.add_trace(
                        go.Scatter(
                            y=y,
                            x=t,
                            mode="lines",
                            name=trace_name,
                            legendgroup=trace_name,
                            showlegend=True,
                            line=dict(color=color),
                        ),
                        row=rows_time[0],
                        col=cols_time[col_idx],
                    )
                    col_idx += 1

                if plot_z:
                    fig.add_trace(
                        go.Scatter(
                            y=z,
                            x=t,
                            mode="lines",
                            name=trace_name,
                            legendgroup=trace_name,
                            showlegend=False,
                            line=dict(color=color),
                        ),
                        row=rows_time[0],
                        col=cols_time[col_idx],
                    )

            if plot_frequency:
                y, z, t = self.pluck(
                    at_this_fret,
                    pick_x=pick_x,
                    pick_y=pick_y,
                    x=x,
                    axis_of_interest=axis_of_interest,
                    fs=fs,
                    l=1,
                    n=n,
                    sum_n=sum_n,
                )
                col_idx = 0
                if plot_y:
                    f, Pxx_den = signal.periodogram(y, fs)
                    plot_this = 10 * np.log10(np.abs(Pxx_den))
                    fig.add_trace(
                        go.Scatter(
                            y=plot_this,
                            x=f,
                            mode="lines",
                            name=trace_name,
                            legendgroup=trace_name,
                            showlegend=False,
                            line=dict(color=color),
                        ),
                        row=rows_freq[0],
                        col=cols_freq[col_idx],
                    )
                    fig.update_xaxes(
                        title_text="Frequency (Hz)",
                        type="log",
                        range=[np.log10(10), np.log10(fs / 2)],
                        row=rows_freq[0],
                        col=cols_freq[col_idx],
                    )
                    col_idx += 1

                if plot_z:
                    f, Pxx_den = signal.periodogram(z, fs)
                    plot_this = 10 * np.log10(np.abs(Pxx_den))
                    fig.add_trace(
                        go.Scatter(
                            y=plot_this,
                            x=f,
                            mode="lines",
                            name=trace_name,
                            legendgroup=trace_name,
                            showlegend=False,
                            line=dict(color=color),
                        ),
                        row=rows_freq[0],
                        col=cols_freq[col_idx],
                    )
                    fig.update_xaxes(
                        title_text="Frequency (Hz)",
                        type="log",
                        range=[np.log10(10), np.log10(fs / 2)],
                        row=rows_freq[0],
                        col=cols_freq[col_idx],
                    )

        if hasattr(at_fret, "__iter__"):
            if colors is None:
                colors = sns.color_palette("colorblind", n_colors=len(at_fret)).as_hex()
            for idx, at_this_fret in enumerate(at_fret):
                add_pluck(at_this_fret, color=colors[idx])
        else:
            if colors is None:
                colors = sns.color_palette("colorblind", n_colors=1).as_hex()
            add_pluck(at_fret, color=colors[0])

        # adjust height of graph based off how many versions are being plotted
        fig.update_layout(height=int(200 * height_mult))

        if return_fig:
            return fig
        else:
            fig.show(renderer="png", width=1300, height=500)

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

    def animate_pluck(self, at_fret):
        import plotly.graph_objects as go
        import numpy as np

        def add_fret(at_fret, fig, trace_idx, frames_data, frames_traces):
            L = np.abs(self.at_fret[at_fret]["x_position (m)"])
            x = np.linspace(0, np.abs(L), 100)

            y, z, t = self.pluck((at_fret), l=1, x=0.1)

            y_of_x = np.zeros((len(y), len(x)))

            for idx, x_pos in enumerate(x):
                y_of_x[:, idx] = self.pluck(
                    (at_fret), x=x_pos, l=1, axis_of_interest="y"
                )

            # TODO fix equation so that it's already flipped for us
            fy_of_x = np.zeros((len(y), len(x)))
            for idy, y_pos in enumerate(y_of_x):
                flip_y_of_x = np.flip(y_pos)
                fy_of_x[idy] = flip_y_of_x

            fx = x - np.max(x)

            x = fx
            y = fy_of_x
            ym = np.min(y)
            yM = np.max(y)

            for k in range(len(x)):
                if k not in frames_data:
                    # print(f"add {k} to frames_data and traces")
                    frames_data[k] = []
                    frames_traces[k] = []
                frames_data[k].append(go.Scatter(x=x, y=y[k], line_color=self.color))
                frames_traces[k].append(trace_idx)

            # update fig section
            fig.add_scatter(
                x=x, y=y[0], line_color=self.color, name=str(at_fret), visible=True
            )

            fig.update_yaxes(range=[ym, yM])

        fig = self.plot_blank_fretboard(
            with_strings=False, with_bridge=True, return_fig=True
        )

        frames = []
        frames_data = {}
        frames_traces = {}
        trace_idx = 1
        for at_this_fret in at_fret:
            add_fret(at_this_fret, fig, trace_idx, frames_data, frames_traces)
            trace_idx += 1

        for k in range(len(frames_data)):
            frames.append(go.Frame(data=frames_data[k], traces=frames_traces[k]))

        frame_duration = 50
        button_anim = dict(
            label="Play Animation",
            method="animate",
            args=[
                None,
                dict(
                    frame=dict(duration=frame_duration, redraw=False),
                    transition=dict(duration=1),
                    fromcurrent=True,
                    mode="immediate",
                ),
            ],
        )

        fig.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [button_anim],
                    "xanchor": "left",
                    "yanchor": "top",
                }
            ]
        )

        fig.frames = frames

        fig.show()

    def _get_linear_mass_density_at_fret(self, fret):
        import numpy as np

        length_ratio = np.abs(self.fret_distances[fret] / self.scale_length)

        return self.linear_mass_density * length_ratio

    def _get_tension_at_fret(self, fret):
        linear_mass_density = self._get_linear_mass_density_at_fret(fret)

        note = add_interval(self.tuning, fret)
        frequency = note_to_frequency(note)

        tension = ((2 * frequency * self.scale_length) ** 2) * linear_mass_density

        return tension

    def get_w_n(self, fret, n):
        """
        returns natural frequency of nth mode/harmonic
        can be verified by comparing against note_to_frequency(self.at_fret[fret]["note"])

        Keyword arguments
        fret -- fret number
        n -- harmonic number/ harmonic of interest
        """
        T = self._get_tension_at_fret(fret)
        u = self._get_linear_mass_density_at_fret(fret)
        v = np.sqrt(T / u)  # speed
        L = np.abs(self.at_fret[fret]["x_position"])
        return (2 * np.pi) * ((n * v) / (2 * L))

    def _make_frets(self):
        # making a dict for now
        # will want properties like how it impacts tension
        # frequency and yada later I imagine
        import pandas as pd
        import numpy as np

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
                "frequency_fundamental": note_to_frequency(note), # (Hz)
                "x_position": self.fret_distances[fret], # (m)
                "y_position": self.y_position, # (m)
                "position": np.array([self.fret_distances[fret], self.y_position]), # (m)
                "linear_mass_density": self._get_linear_mass_density_at_fret(
                    fret # (kg/m)
                ),
                "tension": self._get_tension_at_fret(fret), # (N)
            }

        self.at_fret_df = pd.DataFrame(at_fret).T

        self.at_fret = {}
        for fret in range(self.num_of_frets):
            self.at_fret[fret] = self.at_fret_df.loc[fret]

    def find_note_occurrences(self, note):
        note_name, note_octave = split_note(note)


class Neck:
    # put strings and fretboard here 
    # dont give up
    # youll be happy when its done
    def __init__(
        self,
        num_of_frets,
        num_of_strings,
        scale_length,
        distance_between_strings=0.0104775, # cite this
        ):
        from .utils.fretboard import Fretboard

        self.fretboard = Fretboard(num=num_of_frets, scale_length=scale_length)

        self.num_of_strings = num_of_strings
        self.distance_between_strings = distance_between_strings
        self.make_strings('fralin')
        
    def make_strings(self, action_recommendation):
        import seaborn as sns
        import numpy as np

        colors = sns.color_palette("colorblind", n_colors=6).as_hex()
        action_per_string = np.flip(
            get_action_per_string(action_recommendation, self.num_of_strings)
        )
        dist_between_strings = np.flip(
            np.linspace(
                -self.distance_between_strings * (self.num_of_strings - 1),
                0,
                self.num_of_strings,
            )
        )

        for number, color, tune, gauge, action, dist in zip(
            range(self.num_of_strings),
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
                self.fretboard.scale_length,
                self.fretboard.num_of_frets,
                self.fretboard.distances,
                action,
                dist,
                color,
            )

            self.strings[number + 1] = string