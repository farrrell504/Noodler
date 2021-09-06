import numpy as np

mu_0 = 4 * np.pi * 1e-7

# magnet material: [millitesla, oersteds, BHmax]
magnet_materials = {
    "material_name": ["millitesla", "oersteds", "BHmax"],
    "alnico_1": [720, 470, 1.4],
    "alnico_2": [750, 560, 1.70],
    "alnico_3": [700, 480, 1.35],
    "alnico_4": [560, 720, 1.35],
    "alnico_5": [1280, 640, 5.50],
}

magnet_lengths = {
    "strats_staggered": [17.04, 17.04, 18.03, 18.03, 16.51, 17.04],
    "strats_70s": [17.04, 17.04, 17.04, 17.04, 17.04, 17.04],
    "teles_staggered_bridge": [16.0, 16.0, 17.48, 17.48, 16.0, 16.0],
    "teles_neck": [16.0, 16.0, 16.0, 16.0, 16.0, 16.0],
}


def create_pickup_magnets(
    B=150, diameter=4.95, length=np.tile(17.04, 6), dist=10.4775, num_mags=6
):
    """
    Optional keyword agruments:
    B -- mag field strength (mT)
    diameter -- diameter of magnet (mm)
    length -- length of magnets [num_mag element array] (mm)
    num_mags -- number of magnets in pickup
    dist -- distance between magnets
    """
    from magpylib.magnet import Cylinder
    from magpylib import Collection
    import numpy as np

    if not hasattr(B, "__iter__"):
        B = [B] * num_mags

    if not hasattr(diameter, "__iter__"):
        diameter = [diameter] * num_mags

    if not hasattr(length, "__iter__"):
        length = [length] * num_mags

    #     print('B[0] {:}, diameter[0] {:}, length[0] {:}'.format(B[0], diameter[0], length[0]))
    mag = Cylinder(magnetization=(0, 0, B[0]), dimension=(diameter[0], length[0]))
    c = Collection(mag)
    for idx in range(num_mags - 1):
        mag = Cylinder(magnetization=(0, 0, B[idx + 1]), dimension=(diameter[idx + 1], length[idx + 1]))
        mag.move((0, -dist * (idx + 1), 0))
        c.addSources(mag)
    return c


def plot_mags(
    c,
    x_n=33,
    z_n=44,
    show_mag_field_direction=True,
    from_string_perspective=True,
    string_to_pickup=3.175,
    string_number=1,
    mag_dist=None,
):
    """
    plots all magnets
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from magpylib import displaySystem

    mag_height = c.sources[0].dimension[1]

    if from_string_perspective:
        string_center = mag_height + string_to_pickup
    else:
        string_center = 0

    if mag_dist is None:
        # calculate because lazy me forgot to pass it. lets hope they are all the same...
        mag_dist = c.sources[1].position[1] - c.sources[0].position[1]

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
    displaySystem(c, subplotAx=ax1, suppress=True, direc=show_mag_field_direction)
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


def plot_indv_mag(
    s,
    show_mag_field_direction=True,
    from_string_perspective=True,
    string_to_pickup=3.175,
    string_number=1,
    dist=10.4775,
):
    import numpy as np
    import magpylib as magpy
    import matplotlib.pyplot as plt
    from magpylib import displaySystem

    mag_height = s.dimension[1]
    window_length = s.position[1] + (mag_height * 2)

    if from_string_perspective:
        string_center = mag_height + string_to_pickup
    else:
        string_center = 0

    mag_dist = dist

    x_d = [-10, 10]
    y_d = (string_number - 1) * mag_dist
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
    markers = [(10, 0, mag_height + 10)]
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
    axsB.pcolor(
        X, Y, np.linalg.norm(B, axis=2), cmap=plt.cm.get_cmap("coolwarm")
    )  # amplitude
    axsB.streamplot(X, Y, B[:, :, 0], B[:, :, 2], color="k", linewidth=1)  # field lines

    plt.show()


class Pickup_Magnetics:
    def __init__(
        self,
        name=None,
        magnet_material=None,
        B=0.150,
        diameter=0.00495,
        #         length=np.tile(0.01704,6),
        length=np.array([0.01704, 0.01704, 0.01803, 0.01803, 0.01651, 0.01704]),
        distance_between_magnets=0.0104775,
        distance_between_bridge_and_pickup=0.029,
        num_of_magnets=6,
    ):
        """
        creates pickup magnetics with the given properties

        TODO bad naming needs cleaned up
        """
        self.name = name

        self.distance_between_magnets = distance_between_magnets
        self.num_of_magnets = num_of_magnets

        self.distance_between_bridge_and_pickup = distance_between_bridge_and_pickup
        self.x_position = self.distance_between_bridge_and_pickup

        self.make_magnets(
            magnet_material,
            B,
            diameter,
            length,
            self.num_of_magnets,
            self.distance_between_magnets,
        )

    def make_magnets(self, magnet_material, B, diameter, length, num_mags, dist):
        from magpylib.magnet import Cylinder
        from magpylib import Collection
        import numpy as np

        if not hasattr(B, "__iter__"):
            B = [B] * num_mags

        if not hasattr(diameter, "__iter__"):
            diameter = [diameter] * num_mags

        if not hasattr(length, "__iter__"):
            length = [length] * num_mags

        dist_in_mm = dist * 1e3
        diameter_in_mm = np.array(diameter) * 1e3
        length_in_mm = np.array(length) * 1e3
        B_in_mT = np.array(B) * 1e3

        # print("length_in_mm ", length_in_mm)
        # print("diameter_in_mm ", diameter_in_mm)
        # print("B_in_mT ", B_in_mT)

        #     print('B[0] {:}, diameter[0] {:}, length[0] {:}'.format(B[0], diameter[0], length[0]))
        mag = Cylinder(magnetization=(0, 0, B_in_mT[0]), dimension=(diameter_in_mm[0], length_in_mm[0]))
        c = Collection(mag)
        for idx in range(num_mags - 1):
            mag = Cylinder(
                magnetization=(0, 0, B_in_mT[idx + 1]),
                dimension=(diameter_in_mm[idx + 1], length_in_mm[idx + 1]),
            )
            mag.move((0, -dist_in_mm * (idx + 1), 0))
            c.sources.append(mag)
        self.magnets = c