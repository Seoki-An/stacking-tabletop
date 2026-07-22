"""Open3D viewer that runs in a dedicated subprocess.

Open3D's ``Visualizer`` and PySide6's QApplication both initialize OpenGL
state at the process level and conflict with each other on Linux (the legacy
viewer uses GLFW; Qt uses the ``xcb`` / ``wayland`` platform plugin).  Hosting
the viewer in its own Python process completely avoids that interaction.

Geometries are sent over an ``mp.Queue`` as numpy arrays.  The parent process
never imports :mod:`open3d.visualization` itself.
"""

import multiprocessing as _mp
import queue as _queue
import time

import numpy as np

SHADED_POINT_RADIUS = 0.018
SHADED_POINT_MAX_POINTS = 600
SHADED_WIREFRAME_RADIUS = 0.012
SHADED_WIREFRAME_MAX_EDGES = 800
POINT_CLOUD_RENDER_MODE = "raw"  # "raw" or "spheres"


def _mesh_to_dict(m, default_color=None, as_wireframe=False):
    """Serialize an ``o3d.geometry.TriangleMesh`` to picklable numpy arrays.

    ``default_color`` is applied in the subprocess (via
    :py:meth:`paint_uniform_color`) when the source mesh has no per-vertex
    colors of its own.  Normals are intentionally NOT shipped — they are
    recomputed in the subprocess, which is faster and more reliable than
    trusting the round-trip.

    When ``as_wireframe=True``, the subprocess renders the geometry as a tube
    mesh instead of a LineSet, so it still receives lighting.
    """
    if as_wireframe:
        return {
            "type": "wireframe",
            "vertices": np.asarray(m.vertices, dtype=np.float64).copy(),
            "triangles": np.asarray(m.triangles, dtype=np.int32).copy(),
            "default_color": default_color,
        }
    has_colors = m.has_vertex_colors()
    return {
        "type": "triangle_mesh",
        "vertices": np.asarray(m.vertices, dtype=np.float64).copy(),
        "triangles": np.asarray(m.triangles, dtype=np.int32).copy(),
        "vertex_colors": (
            np.asarray(m.vertex_colors, dtype=np.float64).copy()
            if has_colors
            else None
        ),
        "default_color": default_color,
    }


def _pcd_to_dict(pcd, default_color=None):
    has_colors = pcd.has_colors()
    return {
        "type": "point_cloud",
        "points": np.asarray(pcd.points, dtype=np.float64).copy(),
        "colors": (
            np.asarray(pcd.colors, dtype=np.float64).copy() if has_colors else None
        ),
        "default_color": default_color,
    }


def _viewer_process_main(cmd_queue, window_name: str):
    """Subprocess entrypoint: owns the Open3D Visualizer."""
    # Local imports — keep Open3D entirely out of the parent process.
    import open3d as o3d_p

    def _dict_to_geometry(d):
        if d.get("type") == "point_cloud":
            if POINT_CLOUD_RENDER_MODE == "spheres":
                try:
                    return _point_cloud_as_spheres(d)
                except Exception:
                    return _point_cloud_raw(d)
            else:
                return _point_cloud_raw(d)
        if d.get("type") == "wireframe":
            try:
                return _wireframe_as_tubes(d)
            except Exception:
                return _wireframe_raw(d)
        m = o3d_p.geometry.TriangleMesh()
        m.vertices = o3d_p.utility.Vector3dVector(d["vertices"])
        m.triangles = o3d_p.utility.Vector3iVector(d["triangles"])
        if d.get("vertex_colors") is not None:
            m.vertex_colors = o3d_p.utility.Vector3dVector(d["vertex_colors"])
        elif d.get("default_color") is not None:
            m.paint_uniform_color(d["default_color"])
        # Always recompute normals from the geometry. Transferred normals can
        # be stale or non-unit-length and produce flat (unlit) shading.
        m.compute_vertex_normals()
        m.compute_triangle_normals()
        return m

    def _point_cloud_raw(d):
        pcd = o3d_p.geometry.PointCloud()
        pcd.points = o3d_p.utility.Vector3dVector(d["points"])
        if d.get("colors") is not None:
            pcd.colors = o3d_p.utility.Vector3dVector(d["colors"])
        elif d.get("default_color") is not None:
            pcd.paint_uniform_color(d["default_color"])
        return pcd

    def _wireframe_raw(d):
        mesh = o3d_p.geometry.TriangleMesh()
        mesh.vertices = o3d_p.utility.Vector3dVector(d["vertices"])
        mesh.triangles = o3d_p.utility.Vector3iVector(d["triangles"])
        ls = o3d_p.geometry.LineSet.create_from_triangle_mesh(mesh)
        color = d.get("default_color")
        if color is not None:
            ls.paint_uniform_color(color)
        return ls

    def _point_cloud_as_spheres(d):
        points = np.asarray(d["points"], dtype=np.float64)
        if len(points) == 0:
            return o3d_p.geometry.TriangleMesh()
        if len(points) > SHADED_POINT_MAX_POINTS:
            indices = np.linspace(
                0, len(points) - 1, SHADED_POINT_MAX_POINTS, dtype=np.int64
            )
            points = points[indices]
            colors = d.get("colors")
            if colors is not None:
                colors = np.asarray(colors, dtype=np.float64)[indices]
        else:
            colors = d.get("colors")
            if colors is not None:
                colors = np.asarray(colors, dtype=np.float64)

        default_color = d.get("default_color")
        merged = o3d_p.geometry.TriangleMesh()
        sphere_template = o3d_p.geometry.TriangleMesh.create_sphere(
            radius=SHADED_POINT_RADIUS, resolution=6
        )
        sphere_template.compute_vertex_normals()
        for i, point in enumerate(points):
            sphere = o3d_p.geometry.TriangleMesh(sphere_template)
            if colors is not None:
                sphere.paint_uniform_color(colors[i])
            elif default_color is not None:
                sphere.paint_uniform_color(default_color)
            sphere.translate(point)
            merged += sphere
        merged.compute_vertex_normals()
        merged.compute_triangle_normals()
        return merged

    def _wireframe_as_tubes(d):
        vertices = np.asarray(d["vertices"], dtype=np.float64)
        triangles = np.asarray(d["triangles"], dtype=np.int32)
        if len(vertices) == 0 or len(triangles) == 0:
            return o3d_p.geometry.TriangleMesh()

        edges = set()
        for tri in triangles:
            a, b, c = map(int, tri)
            edges.add(tuple(sorted((a, b))))
            edges.add(tuple(sorted((b, c))))
            edges.add(tuple(sorted((c, a))))
        edges = list(edges)
        if len(edges) > SHADED_WIREFRAME_MAX_EDGES:
            indices = np.linspace(
                0, len(edges) - 1, SHADED_WIREFRAME_MAX_EDGES, dtype=np.int64
            )
            edges = [edges[i] for i in indices]

        color = d.get("default_color")
        merged = o3d_p.geometry.TriangleMesh()
        z_axis = np.array([0.0, 0.0, 1.0])
        for i0, i1 in edges:
            p0 = vertices[i0]
            p1 = vertices[i1]
            delta = p1 - p0
            length = float(np.linalg.norm(delta))
            if length <= 1e-9:
                continue
            cylinder = o3d_p.geometry.TriangleMesh.create_cylinder(
                radius=SHADED_WIREFRAME_RADIUS,
                depth=length,
                resolution=6,
                split=1,
            )
            direction = delta / length
            rotation = _rotation_between_vectors(z_axis, direction)
            cylinder.rotate(rotation, center=np.zeros(3))
            cylinder.translate((p0 + p1) * 0.5)
            if color is not None:
                cylinder.paint_uniform_color(color)
            merged += cylinder
        merged.compute_vertex_normals()
        merged.compute_triangle_normals()
        return merged

    def _rotation_between_vectors(source, target):
        v = np.cross(source, target)
        c = float(np.dot(source, target))
        if c > 1.0 - 1e-9:
            return np.eye(3)
        if c < -1.0 + 1e-9:
            return np.diag([1.0, -1.0, -1.0])
        vx = np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ]
        )
        return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))

    vis = o3d_p.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1024, height=768)

    opt = vis.get_render_option()
    opt.light_on = True
    opt.background_color = np.asarray([0.9, 0.9, 0.9])
    opt.mesh_show_back_face = True
    opt.point_size = 4.0
    try:
        opt.mesh_color_option = o3d_p.visualization.MeshColorOption.Color
        opt.mesh_shade_option = o3d_p.visualization.MeshShadeOption.Default
    except AttributeError:
        # Older Open3D builds may not expose these enums; defaults are fine.
        pass

    current = []
    first_geom = True
    running = True

    try:
        while running:
            try:
                while True:
                    cmd = cmd_queue.get_nowait()
                    if cmd[0] == "stop":
                        running = False
                        break
                    if cmd[0] == "set":
                        for g in current:
                            vis.remove_geometry(g, reset_bounding_box=False)
                        current = []
                        for d in cmd[1]:
                            try:
                                current.append(_dict_to_geometry(d))
                            except Exception:
                                continue
                        for g in current:
                            vis.add_geometry(g, reset_bounding_box=first_geom)
                            first_geom = False
            except _queue.Empty:
                pass

            if not running:
                break

            if not vis.poll_events():
                # Window closed by the user — exit the rendering loop but
                # leave the queue draining alive for graceful shutdown.
                running = False
                break
            vis.update_renderer()
            time.sleep(0.01)
    finally:
        try:
            vis.destroy_window()
        except Exception:
            pass


class Open3DViewer:
    """Handle to an Open3D viewer subprocess.

    Geometries can be sent as bare ``o3d.geometry.TriangleMesh`` objects or as
    ``(mesh, [r, g, b])`` tuples to specify a uniform colour.
    """

    def __init__(self, window_name: str = "Excavator Stacking Viewer"):
        # ``spawn`` gives a fresh interpreter with no Qt/CUDA/Ray state.
        ctx = _mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=8)
        self._proc = ctx.Process(
            target=_viewer_process_main,
            args=(self._queue, window_name),
            daemon=True,
        )
        self._proc.start()

    def set_geometries(self, geoms):
        """Replace the displayed geometry set.

        Each entry may be:
        * a bare ``o3d.geometry.TriangleMesh``
        * a bare ``o3d.geometry.PointCloud``
        * ``(mesh, color)`` to apply a uniform RGB color
        * ``(point_cloud, color)`` to apply a uniform RGB color
        * ``(mesh, color, "wireframe")`` to render the mesh as a LineSet
          (used as a ghost/preview indicator).
        """
        dicts = []
        for entry in geoms:
            g, color, as_wireframe = entry, None, False
            if isinstance(entry, tuple):
                if len(entry) == 3:
                    g, color, style = entry
                    as_wireframe = style == "wireframe"
                elif len(entry) == 2:
                    g, color = entry
                else:
                    continue
            if hasattr(g, "vertices") and hasattr(g, "triangles"):
                try:
                    dicts.append(
                        _mesh_to_dict(
                            g, default_color=color, as_wireframe=as_wireframe
                        )
                    )
                except Exception:
                    continue
            elif hasattr(g, "points"):
                try:
                    dicts.append(_pcd_to_dict(g, default_color=color))
                except Exception:
                    continue
        if not self._proc.is_alive():
            return
        try:
            self._queue.put(("set", dicts), timeout=1.0)
        except _queue.Full:
            pass

    def poll(self):
        # The subprocess polls its own event loop; nothing to do here.
        pass

    def close(self):
        if not self._proc.is_alive():
            return
        try:
            self._queue.put(("stop",), timeout=0.5)
        except Exception:
            pass
        self._proc.join(timeout=3.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=1.0)
