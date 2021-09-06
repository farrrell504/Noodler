from .pickup_magnetics import Pickup_Magnetics
from .pickup_electrical import Pickup_Electrical


class Pickup:
    def __init__(
        self,
        name,
        num_of_magnets,
        distance_between_magnets,
        distance_between_bridge_and_pickup,
    ):
        self.name = name
        self.num_of_magnets = num_of_magnets
        self.distance_between_bridge_and_pickup = distance_between_bridge_and_pickup
        self.x_position = -self.distance_between_bridge_and_pickup
        self.distance_between_magnets = distance_between_magnets

        self.magnetics = Pickup_Magnetics(
            self.name,
            distance_between_magnets=self.distance_between_magnets,
            distance_between_bridge_and_pickup=self.distance_between_bridge_and_pickup,
            num_of_magnets=self.num_of_magnets,
        )
        self.electrical = Pickup_Electrical(self.name)
