import unittest

import numpy as np

from wavelet_nerf.camera import configured_render_path, render_camera_path


class RenderPathTests(unittest.TestCase):
    def test_orbit_controls_change_radius_and_reject_invalid_values(self):
        original = {'type': 'orbit', 'elevation': -30., 'radius': 4.}
        settings = configured_render_path(original, {'render_orbit_radius': 2.0, 'render_orbit_elevation': 0.0})
        poses = render_camera_path(settings, 4)
        np.testing.assert_allclose(np.linalg.norm(poses[:, :3, 3], axis=1), 2)
        self.assertEqual(original['radius'], 4)
        for value in ('0', '-1', 'nan'):
            with self.assertRaises(ValueError):
                configured_render_path(original, {'render_orbit_radius': value})

    def test_spiral_controls_are_serializable_and_preserve_saved_defaults(self):
        original = {'type': 'spiral', 'average_pose': np.eye(4).tolist(), 'up': [0, 1, 0],
                    'radii': [1, 2, 3], 'focus_depth': 5, 'rotations': 2, 'zrate': .5}
        settings = configured_render_path(original, {'render_spiral_rotations': 1.0,
                                                    'render_spiral_zrate': 0.0,
                                                    'render_spiral_radius_scale': 0.5})
        poses = render_camera_path(settings, 4)
        np.testing.assert_allclose(poses[0, :3, 3], [.5, 0, 0])
        np.testing.assert_allclose(poses[:, 2, 3], 0)
        self.assertEqual(settings, configured_render_path(settings, {}))
        self.assertEqual(original['rotations'], 2)
        for key in ('render_spiral_rotations', 'render_spiral_radius_scale'):
            with self.assertRaises(ValueError):
                configured_render_path(original, {key: '0'})
        with self.assertRaises(ValueError):
            configured_render_path(original, {'render_spiral_zrate': 'inf'})


if __name__ == '__main__':
    unittest.main()
