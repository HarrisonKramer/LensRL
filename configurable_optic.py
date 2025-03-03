"""LensRL - ConfigurableOptic module.

This module defines a configurable optic that can be used for reinforcement
learning. The optic is based on an Optiland `Optic` object, but with additional
methods to make it easier to interact with the optic.

Kramer Harrison, 2025
"""
import numpy as np
from optiland import optic, materials
from optiland.fields import FieldGroup
from .glass import VALID_GLASSES


class ConfigurableOptic(optic.Optic):
    """A configurable optic for reinforcement learning."""

    def __init__(self, f_number, max_lenses=5):
        super().__init__()
        self.f_number = f_number
        self.max_lenses = max_lenses
        self.reset()

        self.rms = 1e6  # placeholder value for RMS spot size

    @property
    def num_lenses(self):
        return len(self.surface_group.surfaces) // 2 - 1

    def increase_field_of_view(self, increment):
        self.field_of_view += increment

        self.fields = FieldGroup()
        self.add_field(y=0)
        if self.field_of_view > 0:
            self.add_field(y=self.field_of_view * 0.7)
            self.add_field(y=self.field_of_view)

    def scale_to_unity_focal_length(self):
        """Scale the system to unity focal length"""
        f = self.paraxial.f2()
        self.scale_system(1/f)

    def reset(self, seed=None):
        super().__init__()
        self.set_aperture(aperture_type='imageFNO', value=self.f_number)

        self.set_field_type(field_type='angle')
        self.field_of_view = 0.0
        self.add_field(y=0)

        self.add_wavelength(value=0.4861)
        self.add_wavelength(value=0.5876, is_primary=True)
        self.add_wavelength(value=0.6563)

        # add random starting lens
        if seed is not None:
            np.random.seed(seed)

        # object surface
        self.add_surface(index=0, radius=np.inf, thickness=np.inf)

        # surface 1
        radius = 1000 * np.random.rand()
        thickness = 10 * np.random.rand()
        material = np.random.choice(VALID_GLASSES)
        self.add_surface(index=1, radius=radius, thickness=thickness,
                         material=material, is_stop=True)

        # surface 2
        radius = -1000 * np.random.rand()
        thickness = 100 * np.random.rand()
        self.add_surface(index=2, radius=radius, thickness=thickness)

        # image surface
        self.add_surface(index=3)

    def add_lens(self, lens_idx, radii, thicknesses, material):
        idx = lens_idx * 2 + 1
        self.add_surface(index=idx, radius=radii[0], thickness=thicknesses[0],
                         material=material)
        self.add_surface(index=idx+1, radius=radii[1],
                         thickness=thicknesses[1])

        # workaround
        self.set_thickness(thicknesses[1], idx+1)

    def move_stop(self, surface_idx):
        for idx, surf in enumerate(self.surface_group.surfaces):
            if idx == surface_idx:
                surf.is_stop = True
            else:
                surf.is_stop = False

    def change_glass(self, lens_idx, material):
        idx = lens_idx * 2 + 1
        new_mat = materials.Material(material)
        self.surface_group.surfaces[idx].material_post = new_mat

    def complexity(self):
        return max([len(self.surface_group.surfaces) - 2, 0])

    def get_raw_observation(self):
        t = np.diff(self.surface_group.positions.ravel())

        data = []
        for k, surf in enumerate(self.surface_group.surfaces[1:-1]):
            if isinstance(surf.material_post, materials.Material):
                n = surf.material_post.n(0.58756)
                v = surf.material_post.abbe()

                data += [n, v]

            r0 = surf.geometry.radius
            data += [r0, t[k+1]]

        data = np.array(data)

        # limit size of data
        if len(data) > self.max_lenses * 6:
            data = data[:self.max_lenses * 6]

        # pad with 0 to max number of lenses
        max_elements = self.max_lenses * 6
        data = np.pad(data, (0, max_elements - len(data)), 'constant')

        # prepend metadata
        num_surfaces = len(self.surface_group.surfaces)
        data = np.concatenate([[self.rms, self.f_number, self.field_of_view,
                                num_surfaces], data])

        return data
