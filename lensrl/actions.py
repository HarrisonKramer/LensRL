"""LensRL - Actions module.

This module defines the actions that can be performed on a lens system.
These include:
- Optimizing radii of curvature and thicknesses
- Adding a lens
- Changing the glass material of a lens
- Increasing the field of view
- Moving the stop

Kramer Harrison, 2025
"""
import warnings
from optiland import optimization
from optiland.materials import Material
from .glass import VALID_GLASSES


class BaseAction:
    """
    Base class for all actions that can be performed on a lens system.

    A fixed set of parameters is expected regardless of the action. The
    parameters are expected to be a list of values.

    Definition of parameters:
    0: lens index
    1: surface index
    2: material index
    3: radius0
    4: radius1
    5: thickness0
    6: thickness1
    """
    def validate(self, lens, params):
        """Validate the action and parameters."""
        raise NotImplementedError

    def execute(self, lens, params):
        """Execute the action."""
        raise NotImplementedError


class OptimizeRadiiAction(BaseAction):
    """"
    Optimize the radii of curvature of all surfaces in the lens, as well as the
    thickness to the image plane.
    """
    def validate(self, lens, params):
        return True

    def execute(self, lens, params):
        problem = optimization.OptimizationProblem()

        # add RMS spot size operand
        for field in lens.fields.get_field_coords():
            input_data = {'optic': lens, 'surface_number': -1, 'Hx': field[0],
                          'Hy': field[1], 'num_rays': 5, 'wavelength': 0.55,
                          'distribution': 'hexapolar'}
            problem.add_operand(operand_type='rms_spot_size', target=0,
                                weight=1, input_data=input_data)

        # add variables - all lens radii of curvature
        # (exclude object and image surfaces)
        num_surfaces = lens.surface_group.num_surfaces
        for surf_idx in range(1, num_surfaces-1):
            problem.add_variable(lens, 'radius', surface_number=surf_idx)

        # add thickness variable for the last surface
        problem.add_variable(lens, 'thickness', surface_number=num_surfaces-2)

        optimizer = optimization.OptimizerGeneric(problem)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer.optimize(maxiter=10)


class OptimizeAllAction(BaseAction):
    """
    Optimize all lens radii of curvature and thicknesses between lenses and
    image surface.
    """
    def validate(self, lens, params):
        return True

    def execute(self, lens, params):
        problem = optimization.OptimizationProblem()

        # add RMS spot size operand
        for field in lens.fields.get_field_coords():
            input_data = {'optic': lens, 'surface_number': -1, 'Hx': field[0],
                          'Hy': field[1], 'num_rays': 5, 'wavelength': 0.55,
                          'distribution': 'hexapolar'}
            problem.add_operand(operand_type='rms_spot_size', target=0,
                                weight=1, input_data=input_data)

        # add variables - all lens radii of curvature
        # (exclude object and image surfaces)
        num_surfaces = lens.surface_group.num_surfaces
        for surf_idx in range(1, num_surfaces-1):
            problem.add_variable(lens, 'radius', surface_number=surf_idx)

        # optimize thicknesses between lenses and image surface
        n = lens.n(0.587)
        for surf_idx in range(1, len(n)-1):
            if n[surf_idx] == 1:
                problem.add_variable(lens, 'thickness',
                                     surface_number=surf_idx)

        optimizer = optimization.OptimizerGeneric(problem)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer.optimize(maxiter=10)


class OptimizeNone(BaseAction):
    """
    Do not optimize any parameters.
    """
    def validate(self, lens, params):
        return True

    def execute(self, lens, params):
        pass


class AddLensAction(BaseAction):
    """
    Add a lens to the lens system.
    """
    def validate(self, lens, params):
        lens_idx = params[0]
        mat_idx = params[2]
        radius0 = params[3]
        radius1 = params[4]
        thickness0 = params[5]
        thickness1 = params[6]

        if thickness0 < 0 or thickness1 < 0:
            return False

        if mat_idx < 0 or mat_idx >= len(VALID_GLASSES):
            return False

        if radius0 == 0 or radius1 == 0:
            return False

        n = lens.surface_group.num_surfaces
        num_lens = (n - 2) // 2

        if lens_idx < 0 or lens_idx > num_lens:
            return False

        if num_lens >= lens.max_lenses:
            return False

        return True

    def execute(self, lens, params):
        lens_idx = params[0]
        material = Material(VALID_GLASSES[params[2]])
        radii = [params[3], params[4]]
        thicknesses = [params[5], params[6]]
        lens.add_lens(lens_idx, radii, thicknesses, material)


class ChangeGlassAction(BaseAction):
    """
    Change the glass material of a lens.
    """
    def validate(self, lens, params):
        lens_idx = params[0]
        n = lens.surface_group.num_surfaces
        num_lens = (n - 2) // 2

        if lens_idx < 0 or lens_idx >= num_lens:
            return False

        glass_idx = params[0]
        return glass_idx >= 0 and glass_idx < len(VALID_GLASSES)

    def execute(self, lens, params):
        lens_idx = params[0]
        material = VALID_GLASSES[params[2]]
        lens.change_glass(lens_idx, material)


class IncreaseFOVAction(BaseAction):
    """
    Increase the field of view of the lens system.
    """
    def validate(self, lens, params):
        return True

    def execute(self, lens, params):
        increment = 4.0  # default for now...
        lens.increase_field_of_view(increment)


class MoveStopAction(BaseAction):
    """
    Move the stop to a different surface.
    """
    def validate(self, lens, params):
        surf_idx = params[1]
        if surf_idx == 0 or surf_idx >= len(lens.surface_group.surfaces)-1:
            return False
        return True

    def execute(self, lens, params):
        lens.move_stop(params[1])


OPTIMIZATION_ACTIONS = [
    OptimizeRadiiAction,
    OptimizeAllAction,
    OptimizeNone
]

UPDATE_ACTIONS = [
    AddLensAction,
    ChangeGlassAction,
    IncreaseFOVAction,
    MoveStopAction
]
