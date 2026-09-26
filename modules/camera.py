import numpy as np


def translate_by_t_along_z(t):
    tform = np.eye(4).astype(np.float32)
    tform[2][3] = t
    return tform


def rotate_by_phi_along_x(phi):
    tform = np.eye(4).astype(np.float32)
    tform[1, 1] = tform[2, 2] = np.cos(phi)
    tform[1, 2] = -np.sin(phi)
    tform[2, 1] = -tform[1, 2]
    return tform


def rotate_by_theta_along_y(theta):
    tform = np.eye(4).astype(np.float32)
    tform[0, 0] = tform[2, 2] = np.cos(theta)
    tform[0, 2] = -np.sin(theta)
    tform[2, 0] = -tform[0, 2]
    return tform


def pose_spherical(theta, phi, radius):
    c2w = translate_by_t_along_z(radius)
    c2w = rotate_by_phi_along_x(phi / 180.0 * np.pi) @ c2w
    c2w = rotate_by_theta_along_y(theta / 180 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def _unit(vector):
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-10:
        raise ValueError('Camera orientation is degenerate')
    return vector / length


def view_pose(back, up, position):
    """Construct an OpenGL camera pose from its backward axis and approximate up."""
    back = _unit(back)
    right = _unit(np.cross(up, back))
    up = _unit(np.cross(back, right))
    pose = np.eye(4)
    pose[:3, :4] = np.stack((right, up, back, position), axis=-1)
    return pose


def average_pose(poses):
    return view_pose(poses[:, :3, 2].sum(axis=0), poses[:, :3, 1].sum(axis=0),
                     poses[:, :3, 3].mean(axis=0))


def render_camera_path(settings, count):
    """Generate orbit or LLFF spiral poses from serializable path settings."""
    if count < 1:
        raise ValueError('Number of render poses must be positive')
    if settings['type'] == 'orbit':
        return np.stack([pose_spherical(angle, settings['elevation'], settings['radius'])
                         for angle in np.linspace(-180, 180, count, endpoint=False)]).astype(np.float32)
    if settings['type'] != 'spiral':
        raise ValueError(f'Unknown camera path type: {settings["type"]}')
    average = np.asarray(settings['average_pose'])
    up = np.asarray(settings['up'])
    radii = np.asarray(settings['radii'])
    focus = average[:3, 3] - settings['focus_depth'] * average[:3, 2]
    poses = []
    for angle in np.linspace(0, 4 * np.pi, count, endpoint=False):
        offset = radii * [np.cos(angle), -np.sin(angle), -np.sin(angle * 0.5)]
        position = average[:3, :3] @ offset + average[:3, 3]
        poses.append(view_pose(position - focus, up, position))
    return np.stack(poses).astype(np.float32)
