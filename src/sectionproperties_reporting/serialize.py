import json
import io
import pathlib
from pydantic import TypeAdapter, BaseModel, RootModel, Field, field_serializer, field_validator
from typing import Optional, Any, Union, ClassVar, TextIO, Annotated

from sectionproperties.pre.geometry import Geometry, CompoundGeometry
from sectionproperties.pre.pre import Material, DEFAULT_MATERIAL
import numpy as np
import shapely
from shapely import wkt


def to_json(section_geometry: Geometry | CompoundGeometry, filepath: str | pathlib.Path) -> None:
    """
    Serializes the model to a new JSON file at 'filepath'.
    """
    filepath = pathlib.Path(filepath)
    with open(filepath, 'w') as file:
        dump(section_geometry, file)


def from_json(filepath: str | pathlib.Path) -> Geometry | CompoundGeometry:
    """
    Reads the JSON file at 'filepath' and returns the Pynite.FEModel3D.
    """
    with open(filepath, 'r') as file:
        model = load(file)
    return model


def dump(section_geometry: Geometry | CompoundGeometry, file_io: TextIO, indent: int = 2) -> None:
    """
    Writes the 'model' as a JSON data to the file-handler object, 'file_io'.

    'indent': the number of spaces to indent in the file.
    """
    model_dict = dump_dict(section_geometry)
    json.dump(model_dict, fp=file_io, indent=indent)


def dumps(section_geometry: Geometry | CompoundGeometry, indent: int = 2) -> str:
    """
    Returns the model as JSON string.
    """
    model_schema = get_model_schema(section_geometry)
    return model_schema.model_dump_json(indent=indent)


def dump_dict(section_geometry: Geometry | CompoundGeometry) -> dict:
    """
    Returns a Python dictionary representing the model.

    The Python dictionary is serializable to JSON.
    """
    model_schema = get_model_schema(section_geometry)
    return model_schema.model_dump()


def load(file_io: TextIO) -> Geometry | CompoundGeometry:
    """
    Returns an FEModel3D from the json data contained within the file.
    """
    json_data = json.load(file_io)
    model_adapter = TypeAdapter(SectionPropertiesSchema)
    model_schema = model_adapter.validate_python(json_data)
    # return model_schema.to_femodel3d()
    return model_schema


def loads(model_json: str) -> Geometry | CompoundGeometry:
    """
    Returns an FEModel3D based on the provided 'model_json'.

    'model_json': a JSON-serialized str representing an FEModel3D
    """
    model_adapter = TypeAdapter(SectionPropertiesSchema)
    model_schema = model_adapter.validate_json(model_json)
    # femodel3d = model_schema.to_femodel3d()
    return model_schema


def load_dict(model_dict: dict) -> Geometry | CompoundGeometry:
    """
    Returns an FEModel3D based on the provided 'model_dict'.

    'model_dict': A JSON-serializable dict representing an FEModel3D
    """
    model_adapter = TypeAdapter(SectionPropertiesSchema)
    model_schema = model_adapter.validate_python(model_dict)
    femodel3d = model_schema.to_femodel3d()
    return femodel3d


def get_model_schema(section_geometry: Geometry | CompoundGeometry) -> dict[str, dict]:
    """
    Returns an SectionPropertiesSchema based on the supplied model.
    """
    model_adapter = TypeAdapter(SectionPropertiesSchema)
    model_schema = model_adapter.validate_python(section_geometry, from_attributes=True)
    return model_schema


class ExporterMixin:
    def to_init_dict(self):
        init_dict = {}
        if self._init_attrs is None:
            return init_dict
        for attr_name in self._init_attrs:
            attr_value = getattr(self, attr_name)
            if hasattr(attr_value, "_sectionproperties_class"):
                attr_value = attr_value._sectionproperties_class(**attr_value.to_init_dict())
            init_dict.update({attr_name: attr_value})
        return init_dict



class MaterialSchema(BaseModel, ExporterMixin):
    name: str
    elastic_modulus: float
    poissons_ratio: float
    yield_strength: float
    density: float
    color: str
    _init_attrs: ClassVar[Optional[list[str]]] = [
        'name',
        'elastic_modulus',
        'poissons_ratio',
        'yield_strength',
        'density',
        'color',
    ]
    _sectionproperties_class: ClassVar[type] = Material


class GeometrySchema(BaseModel, ExporterMixin):
    geom: str
    material: Optional[MaterialSchema]
    control_points: Optional[list[str] | list[tuple[float, float]]] = None
    tol: int = 12
    points: Optional[list[tuple[float, float]]] = None
    facets: Optional[list[tuple[int, int]]] = None
    holes: Optional[list[tuple[float, float]]] = None
    mesh: Optional[dict[str, list | float]] = None
    _init_attrs: ClassVar[Optional[list[str]]] = ['geom', 'material', 'control_points', 'tol']
    _sectionproperties_class: ClassVar[type] = Geometry

    
    @field_validator("geom", mode="before")
    @classmethod
    def validate_geom(cls, geom: shapely.Polygon):
        return wkt.dumps(geom, trim=True)


    @field_validator("control_points", mode="before")
    @classmethod
    def validate_control_points(cls, ctrl_pts: list[tuple[float, float]] | shapely.Point):
        if isinstance(ctrl_pts, shapely.Point):
            return list(ctrl_pts.coords)
        else:
            return ctrl_pts
        
    @field_validator("mesh", mode="before")
    @classmethod
    def validate_mesh(cls, mesh: dict[str, Any]):
        if mesh is not None:
            serialized_mesh = {}
            serialized_mesh['vertices'] = mesh['vertices'].tolist()
            serialized_mesh['vertex_markers'] = mesh['vertex_markers'].tolist()
            serialized_mesh['triangles'] = mesh['triangles'].tolist()
            serialized_mesh['triangle_attributes'] = mesh['triangle_attributes'].tolist()
            serialized_mesh['segments'] = mesh['segments']
            serialized_mesh['segment_markers'] = mesh['segment_markers']
            serialized_mesh['regions'] = mesh['regions']
            return serialized_mesh
        
    def to_sectionproperties(self):
        sec_prop_class = self._sectionproperties_class
        geom = wkt.loads(self.geom)
        init_dict = self.to_init_dict()
        init_dict.update({"geom": geom})
        new_inst = sec_prop_class(**init_dict)

        for attr_name, attr_value in self:
            if attr_name in init_dict: continue
            if attr_name == "mesh" and attr_value is not None:
                attr_value['vertices'] = np.array(attr_value['vertices'])
                attr_value['vertex_markers'] = np.array(attr_value['vertex_markers'])
                attr_value['triangles'] = np.array(attr_value['triangles'])
                attr_value['triangle_attributes'] = np.array(attr_value['triangle_attributes'])
            setattr(new_inst, attr_name, attr_value)
        return new_inst

    


class CompoundGeometrySchema(GeometrySchema):
    geoms: list[GeometrySchema] | str

    _init_attrs: ClassVar[Optional[list[str]]] = ['geoms']
    _sectionproperties_class: ClassVar[type] = CompoundGeometry


class SectionPropertiesSchema(RootModel, ExporterMixin):
    """
    A container to hold the schema, whether it is Geometry or CompoundGeometry
    object
    """
    root: GeometrySchema | CompoundGeometrySchema 
    _init_attrs: ClassVar[Optional[list[str]]] = None

    # def to_femodel3d(self):
    #     model_object_classes = {
    #         "nodes": Node3D.Node3D,
    #         "materials": Material.Material,
    #         "sections": Section.Section,
    #         "springs": Spring3D.Spring3D,
    #         "members": PhysMember.PhysMember,
    #         "quads": Quad3D.Quad3D,
    #         "plates": Plate3D.Plate3D,
    #         "meshes": Mesh.Mesh,
    #         "load_combos": LoadCombo.LoadCombo,
    #     }
    #     femodel3d = FEModel3D()
    #     for key, schema_objects in self.__dict__.items():
    #         model_object_class = model_object_classes[key]
    #         model_objects = {}
    #         for key_name, schema_object in schema_objects.items():
    #             schema_init_dict = schema_object.to_init_dict()

    #             # Modify the init dict with special case attributes
    #             if "model" in schema_init_dict:
    #                 # Need to add the model as an attr to several object types
    #                 schema_init_dict.update({"model": femodel3d})
    #             if "material" in schema_init_dict:
    #                 # Need to use the material_name (not the material object) as the init value
    #                 material_name = schema_init_dict['material'].name
    #                 schema_init_dict.pop("material")
    #                 schema_init_dict.update({"material_name": material_name})
    #             if "section" in schema_init_dict:
    #                 # Same as material_name above but with the section
    #                 section_name = schema_init_dict['section'].name
    #                 schema_init_dict.pop("section")
    #                 schema_init_dict.update({"section_name": section_name})
                    
    #             # Create the new object with their init values
    #             new_object = model_object_class(**schema_init_dict)
                
    #             # Add in all of the other attrs excluded from the init process
    #             for attr_name, attr_value in schema_object.__dict__.items():
    #                 if attr_name == "model":
    #                     attr_value = femodel3d
    #                 if schema_init_dict is None or attr_name not in schema_init_dict:
    #                     setattr(new_object, attr_name, attr_value)

    #             # For attr_values that reference nodes, they must reference the original
    #             # node in the model (an new-but-equal instance will not suffice because it will 
    #             # not have the correct .ID attribute).
    #             for attr_name, attr_value in new_object.__dict__.items():
    #                 if 'node' in attr_name:
    #                     node_name = attr_value.name
    #                     orig_node = femodel3d.nodes[node_name]
    #                     setattr(new_object, attr_name, orig_node)
                    
    #             model_objects.update({key_name: new_object})
    #         setattr(femodel3d, key, model_objects)
    #     return femodel3d