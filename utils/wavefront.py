from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Union

import numpy as np
import yaml
from scipy.spatial import ConvexHull


def format_matrix(mat, indent=2):
    rows = []
    for i, row in enumerate(mat):
        row_str = ", ".join(f"{v:.6g}" for v in row)

        if i == 0:
            rows.append(f"[[{row_str}],")
        elif i == len(mat) - 1:
            rows.append(" " * indent + f"[{row_str}]]")
        else:
            rows.append(" " * indent + f"[{row_str}],")

    return "\n".join(rows)


def format_params(params):
    lines = []

    for k, v in params.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            mat_str = format_matrix(v)
            mat_str = mat_str.replace("\n", "\n# ")
            lines.append(f"# {k}: {mat_str}")
        else:
            lines.append(f"# {k}: {v}")

    return "\n".join(lines)


class WavefrontExporter:
    def __init__(self, path, params=None):
        self.path = path
        self.n_object = 0
        self.n_vertex = 0
        self.n_texcoord = 0
        self.n_normal = 0

        with open(path, "w") as file:
            if params is not None:
                file.write(format_params(params))
                file.write("\n\n")

    def add_object(
        self,
        vertices,
        faces,
        texcoords=None,
        normals=None,
        sharpness=None,
        mu=None,
        name=None,
        tag=None,
    ):
        with open(self.path, "a") as file:
            if self.n_object > 0:
                file.write("\n")

            if tag is not None:
                file.write(f"# {', '.join(tag)}\n")

            if name is None:
                name = str(self.n_object)
            file.write(f"o {name}\n")

            if sharpness is not None:
                file.write(f"# sharpness: {sharpness}\n")
            if mu is not None:
                file.write(f"# mu: {mu}\n")

            for vertex in vertices:
                file.write(f"v {' '.join(map(str, vertex))}\n")

            if texcoords is not None:
                for texcoord in texcoords:
                    file.write(f"vt {' '.join(map(str, texcoord))}\n")

            if normals is not None:
                for normal in normals:
                    file.write(f"vn {' '.join(map(str, normal))}\n")

            for face in faces:
                elements = []
                for idx in face:
                    if isinstance(idx, np.integer):
                        elements.append(f"{idx + self.n_vertex}")
                    else:
                        v, vt, vn = idx
                        elements.append(
                            f"{v + self.n_vertex}"
                            f"/{vt + self.n_texcoord if vt is not None else ''}"
                            f"/{vn + self.n_normal if vn is not None else ''}"
                        )
                file.write(f"f {' '.join(elements)}\n")

            self.n_object += 1
            self.n_vertex += len(vertices)
            self.n_texcoord += len(texcoords) if texcoords is not None else 0
            self.n_normal += len(normals) if normals is not None else 0

    def add_convex(self, vertices, sharpness, mu, name=None):
        if name is None:
            name = f"convex.{self.n_object}"

        try:
            hull = ConvexHull(vertices)
        except Exception as exc:
            raise ValueError("Input vertices do not appear to be convex") from exc

        faces = hull.simplices
        center = np.mean(vertices, axis=0)
        for i, face in enumerate(faces):
            edge_01 = vertices[face[1]] - vertices[face[0]]
            edge_02 = vertices[face[2]] - vertices[face[0]]
            normal = np.cross(edge_01, edge_02)
            if np.dot(normal, vertices[face[0]] - center) < 0:
                faces[i] = face[::-1]
        faces += 1

        self.add_object(vertices, faces, sharpness=sharpness, mu=mu, name=name)


@dataclass
class Object:
    vertices: np.ndarray
    texcoords: np.ndarray
    normals: np.ndarray
    faces: np.ndarray


@dataclass
class Convex(Object):
    face_centers: np.ndarray
    face_normals: np.ndarray
    sharpness: float = 0.0


class WavefrontImporter:
    class Mode(Enum):
        PARAM = auto()
        DATA = auto()

    def __init__(self, path):
        self.n_object = 0
        self.n_vertex = 0
        self.n_texcoord = 0
        self.n_normal = 0
        self.objects = {}
        self.__init_internal_vars()

        with open(path, "r") as file:
            for line in file:
                self.__process_line(line)
        self.__flush()

    def get_object(self, idx) -> Union[Object, Convex]:
        try:
            return list(self.objects.values())[idx]
        except Exception as exc:
            raise IndexError(f"No object corresponding to the given index: {idx}") from exc

    def get_objects(self) -> List[Union[Object, Convex]]:
        return list(self.objects.values())

    def __init_internal_vars(self):
        self.__mode = __class__.Mode.PARAM
        self.__yaml_str = ""
        self.__tag = None
        self.__object_name = None
        self.__object_data = {}

    def __process_line(self, line: str):
        if self.__mode == __class__.Mode.PARAM:
            if self.__is_comment(line):
                self.__yaml_str += self.__read_as_param(line)
            elif self.__yaml_str and line.startswith((" ", "\t")):
                self.__yaml_str += line
            else:
                try:
                    self.params = yaml.safe_load(self.__yaml_str)
                    self.__yaml_str = ""
                except Exception as exc:
                    raise ValueError("Unable to parse comments as yaml") from exc
                self.__mode = __class__.Mode.DATA

        elif self.__mode == __class__.Mode.DATA:
            if self.__is_empty(line):
                return
            elif self.__is_comment(line):
                self.__flush()
                self.__tag = self.__read_as_tag(line)
                if self.__tag == ["params"]:
                    self.__mode = __class__.Mode.PARAM
                yaml_str = self.__read_as_param(line)
                param = yaml.safe_load(yaml_str)
                if isinstance(param, dict) and param.get("sharpness") is not None:
                    self.__object_data["sharpness"] = param["sharpness"]

            elif self.__is_named_object(line):
                self.__flush()
                _, self.__object_name = self.__read_as_data(line)
            else:
                key, val = self.__read_as_data(line)
                if key in self.__object_data:
                    self.__object_data[key].append(val)
                else:
                    self.__object_data[key] = [val]

    def __is_empty(self, line: str):
        return line.strip(" ") == "\n"

    def __is_comment(self, line: str):
        return line.startswith("#")

    def __is_named_object(self, line: str):
        return line.startswith("o ")

    def __read_as_tag(self, line: str):
        return line[1:].strip().split(", ")

    def __read_as_param(self, line: str):
        return line[1:]

    def __read_as_data(self, line: str):
        words = line.strip().split()
        for i, word in enumerate(words):
            if word.startswith("#"):
                words = words[:i]
                break

        header, body = words[0], words[1:]
        if header == "o":
            return header, line[1:].strip()
        elif header == "v" or header == "vt" or header == "vn":
            return header, list(map(float, body))
        elif header == "f":
            face = []
            for element in body:
                element = element.split("/")
                face.append(
                    int(element[0])
                    if len(element) == 1
                    else [int(e) if e else 0 for e in element]
                )
            return header, face
        else:
            raise f"Not defined for given header: {header}"

    def __flush(self):
        vertices, texcoords, normals, faces, sharpness = (
            np.array(self.__object_data.get(key, []))  #
            for key in ["v", "vt", "vn", "f", "sharpness"]
        )
        if vertices.size == 0:
            return

        if faces.ndim == 2:
            offset = np.array([self.n_vertex])
        elif faces.ndim == 3:
            offset = np.array([self.n_vertex, self.n_texcoord, self.n_normal])
        faces -= offset + 1

        if normals.size == 0:
            face_vertices = vertices[faces, :]
            face_centers = face_vertices.mean(1)
            face_normals = np.cross(
                face_vertices[:, 1, :] - face_vertices[:, 0, :],
                face_vertices[:, 2, :] - face_vertices[:, 0, :],
            )
            face_normals /= np.linalg.norm(face_normals, axis=1)[:, None]

            object = Convex(
                vertices,
                texcoords,
                normals,
                faces,
                face_centers,
                face_normals,
                sharpness,
            )
        else:
            object = Object(vertices, texcoords, normals, faces)

        if self.__object_name is None:
            self.__object_name = str(self.n_object)
        self.objects.update({self.__object_name: object})

        self.n_object += 1
        self.n_vertex += len(vertices)
        self.n_texcoord += len(texcoords)
        self.n_normal += len(normals)
        self.__tag = None
        self.__object_name = None
        self.__object_data = {}


class FlowStyleListDumper(yaml.Dumper):
    def represent_sequence(self, tag, sequence, flow_style=True):
        return super().represent_sequence(tag, sequence, flow_style)
