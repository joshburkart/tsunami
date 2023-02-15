import abc
import dataclasses

import numpy as np
import scipy.interpolate

CoordArray = np.ndarray[float]

ScalarFieldArray = np.ndarray[float]  # num_x_points [x num_y_points] x num_z_points
VecFieldArray = np.ndarray[float]  # num_x_points [x num_y_points] x num_z_points x num_horiz_dims

HorizScalarFieldArray = np.ndarray[float]  # num_x_points [x num_y_points]
HorizVecFieldArray = np.ndarray[float]  # num_x_points [x num_y_points] x num_horiz_dims

Mask = np.ndarray[bool]

t_max = 4
t_rain_stop = 3
num_x_points = 200
num_z_points = 5
rain_rate = 0.
g = 10

faces, dx = np.linspace(0, 1, num_x_points, retstep=True)
centers = (faces[1:] + faces[:-1]) / 2


@dataclasses.dataclass
class Axis:
    faces: CoordArray
    centers: CoordArray
    spacing: float


@dataclasses.dataclass
class Grid:
    x_axis: Axis
    y_axis: Axis

    num_z_cells: int

    @property
    def num_horiz_axis(self) -> int:
        return len(self.axes)

    def make_mesh(self, terrain_height: HorizScalarFieldArray,
                             water_height: HorizScalarFieldArray) -> 'Mesh':
        z_centers =
            terrain_height[:, np.newaxis] +
            water_height[:, np.newaxis] * np.linspace(0, 1, self.num_z_points + 1)[np.newaxis, :])

    def compute_cell_vert_normals(self, terrain_height: HorizScalarFieldArray,
                                  water_height: HorizScalarFieldArray) -> VecFieldArray:
        # num_x_points x num_y_points x (num_z_points + 1)
        z_boundaries = self.compute_z_boundaries(
            terrain_height=terrain_height, water_height=water_height)

    def interp_to_horiz_boundaries(self, axis: int,
                                   field: HorizVecFieldArray) -> HorizVecFieldArray:
        raise NotImplementedError
        interpolant = scipy.interpolate.InterpolatedUnivariateSpline(self.centers, field, k=1)
        return interpolant(self.faces)


@dataclasses.dataclass
class Mesh:
    grid: Grid
    z_boundaries: 


def _horiz_grad(field):
    spline = scipy.interpolate.Akima1DInterpolator(centers, field)
    return spline.derivative()(centers)


def _laplacian(field: VecFieldArray | ScalarFieldArray, grid: Grid,
               terrain: Terrain) -> VecFieldArray | ScalarFieldArray:
    raise NotImplementedError


@dataclasses.dataclass
class Terrain:
    height: HorizScalarFieldArray
    grad_height: HorizVecFieldArray

    _HORIZ_SCALE: float = 0.2
    _VERT_SCALE: float = 0.3

    @classmethod
    def make_mountain_1d(cls, x: Axis) -> 'Terrain':
        midpoint = np.mean(x.centers)
        height = (1.5 * cls._VERT_SCALE * np.exp(-(x.centers - midpoint)**2 / cls._HORIZ_SCALE**2) +
                  0.3 * cls._VERT_SCALE * np.sin(14 * np.pi * (x.centers - midpoint) + 1))
        return cls._make(height=height)

    @classmethod
    def _make(cls, height: HorizScalarFieldArray) -> 'Terrain':
        return cls(height=height, grad_height=_horiz_grad(height))


class Rain(abc.ABC):
    @abc.abstractmethod
    def compute_rain_rate(self, t: float, grid: Grid) -> HorizScalarFieldArray:
        pass


@dataclasses.dataclass
class Fields:
    height: HorizScalarFieldArray
    velocity: VecFieldArray
    pressure: ScalarFieldArray

    def compute_cfl_timestep(self) -> float:
        MAX_VELOCITY = 1
        np.abs(self.velocity)

    @property
    def active_mask(self) -> Mask:
        return self.height > 1e-8

    def compute_horiz_velocity_column_int(self, z_boundaries: CoordArray) -> HorizVecFieldArray:
        delta_z = z_boundaries[..., :-1] - z_boundaries[..., 1:]
        return np.sum(delta_z[..., np.newaxis] * self.velocity, axis=-2)[..., :-1]

    @classmethod
    def zeros(cls) -> 'Fields':
        return cls(
            height=np.zeros(num_x_points),
            velocity=np.zeros(num_x_points, num_z_points, 2),
            pressure=np.zeros(num_x_points, num_z_points),
        )


def take_step(t, grid: Grid, terrain: Terrain, rain: Rain, fields: Fields) -> Fields:
    dt = fields.compute_cfl_timestep()

    height_flux_centers = fields.compute_horiz_velocity_column_int(
        z_boundaries=grid.compute_z_boundaries())
    height_flux_boundaries = grid.interp_to_horiz_boundaries(height_flux_centers)
    rain_rate = rain.compute_rain_rate(t, grid)

    new_fields = Fields.zeros()
    new_fields.height[:] = fields.height
    new_fields.height[1:-1] += -dt / dx * (height_flux_boundaries[1:] - height_flux_boundaries[:-1])
    new_fields.height += dt * rain_rate

    return new_fields
