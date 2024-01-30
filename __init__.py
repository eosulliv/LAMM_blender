"""Module providing means of editing face shape via facial landmarks.
"""

import os
import json
import pickle
import sys

from math import radians
from mathutils import Vector, Quaternion

import numpy as np

import bpy
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                       IntProperty, PointerProperty, StringProperty)
from bpy.types import PropertyGroup

# Add path to packages
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
CHECKPOINT_DIR = os.path.join(CURR_DIR, 'LAMM', 'assets', 'checkpoints')
# TODO: Change this so that it is alphabetically the first checkpoint in the dir
DEFAULT_CHECKPOINT = 'mtt_256-11regions_wholehead_fromAE_lr1e4_wregion6_rand-b'

sys.path.append(os.path.join(CURR_DIR, 'packages'))
import trimesh as tm
import torch

# Add path to model utils
# sys.path.append(os.path.join(CURR_DIR, 'utils'))
sys.path.append(CURR_DIR)
from LAMM.utils.config_files_utils import read_yaml
from LAMM.utils.torch_utils import load_from_checkpoint
from LAMM.models import LAMM

# Blender plug-in information
bl_info = {
    "name": "Locally Adaptive 3D Morphable Models",
    "author": "Eimear O' Sullivan",
    "version": (2023, 10, 23),
    "blender": (3, 3, 1),
    "location": "Viewport > Right panel",
    "description": "Locally Adaptive 3D Morphable Models for Blender",
    "wiki_url": "",
    "category": "LAMM"
}

##################################### Globals #####################################
# FACE_BASEMESH = 'mean.obj'

# LANDMARK_FILE = 'data\\resources\\FaceLandmarks.blend'
# LANDMARK_SET = 'facial_landmarks'

# FACE_LM_NAMES = [
#     'left_nostril', 'nose1', 'nose2', 'nose3', 'nose4', 'nose5', 'nose6', 'right_nostril',  # 0-7
#     'left_ear1', 'left_cheek1', 'left_cheek2', 'jaw1', 'jaw2', 'left_cheek3',  # 8-13
#     'jaw3', 'jaw4', 'left_cheek4', 'left_cheek5', 'chin1', 'chin2', 'left_cheek6',  # 14-20
#     'left_eye1', 'left_eye2', 'left_eye3', 'left_eye4', 'left_eye5',  # 21-25
#     'right_eye1', 'right_cheek1', 'right_eye2', 'right_eye3', 'right_eye4', 'right_eye5',  # 26-31
#     'left_ear2', 'left_ear3', 'left_ear4', 'left_ear5',  # 32-35
#     'left_ear6', 'left_ear7', 'left_ear8', 'left_ear9',  # 36-39
#     'right_ear1', 'right_ear2', 'right_ear3', 'right_ear4', 'right_ear5',  # 40-44
#     'right_ear6', 'right_ear7', 'right_ear8', 'cranium1', 'cranium2', 'cranium3',  # 45-50
#     'right_ear9', 'jaw5', 'chin3', 'chin4', 'right_cheek2', 'right_cheek3',  # 51-56
#     'jaw6', 'jaw7', 'right_cheek4', 'right_cheek3', 'right_cheek6',  # 57-61
#     'jaw8', 'chin5', 'mouth1', 'mouth2', 'mouth3', 'mouth4', 'mouth5', 'mouth6', 'mouth7',  # 62-70
#     'mouth8', 'mouth9', 'mouth10', 'mouth11', 'mouth12', 'mouth13', 'mouth14',  # 71-77
#     'cranium4', 'cranium5', 'cranium6'  # 78-80
# ]
# NUM_FACE_LMS = len(FACE_LM_NAMES)


#################################### Functions ####################################
def clear_object_selection():
    """Deselect all objects"""
    bpy.ops.object.select_all(action='DESELECT')


def get_face_landmark_vertices(mesh):
    """Load landmarks for the mean mesh"""
    control_lms = config['MODEL']['regions']
    lms = []
    for vals in control_lms.values():
        for idx in vals:
            lms.append(mesh.data.vertices[idx].co)

    edges = []
    if 'region_edges' in config['MODEL']:
        edges = config['MODEL']['region_edges']

    lms_mesh = bpy.data.meshes.new('lms')
    lms_mesh.from_pydata(lms, edges, [])
    lms_mesh.update()
    return lms_mesh


def get_object(name):
    """Returns an object or none"""
    if isinstance(name, str):
        obj = bpy.context.scene.objects.get(name)
        if obj:
            return obj
    return None


def set_active_object(obj, select=True):
    """Select the object"""
    if isinstance(obj, str):
        obj = bpy.data.objects.get(obj)
    if obj:
        if select:
            obj.select_set(state=True)
        bpy.context.view_layer.objects.active = obj
        return {'FINISHED'}

    print(f'WARNING! Object {obj.name} does not exist')
    return {'CANCELLED'}


##################################### Classes #####################################
class PG_FaceProperties(PropertyGroup):
    """Properties"""
    face_region: EnumProperty(
        name = 'Region',
        description = 'Mesh Regions',
        items = [ ('0', 'Left Side', ''), ('1', 'Right Side', ''), ('3', 'Ears', ''),
                  ('6', 'Skull', ''), ('7', 'Forehead', ''), ('8', 'Eyes', ''),
                  ('9', 'Nose', ''), ('10','Mouth',  '') ]
    )


class FaceAddMeanMesh(bpy.types.Operator):
    """Add model mean mesh to scene"""
    bl_idname = 'object.face_add_mean_mesh'
    bl_label = 'Add'
    bl_description = 'Add the model mean mesh to the scene'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        """Check whether the panel can be drawn"""
        try:
            if context.active_object is None or context.active_object.mode == 'OBJECT':
                return True
            return False
        except:
            return False

    def execute(self, context):
        """Execute"""
        print('Adding mean model mesh...')
        control_lms = config['MODEL']['regions']
        delta = []
        for idx in control_lms.keys():
            delta.append(torch.zeros(3 * len(control_lms[idx]), device=device).reshape(1, -1))

        z_mean = np.array([gid_dict['mean']])
        vertices = model.decode(torch.tensor(z_mean).unsqueeze(0), delta)[-1][0].detach().numpy()
        vertices = vertices * (mean_std['std'] + 1e-7) + mean_std['mean']

        new_mesh = bpy.data.meshes.new('mean')
        new_mesh.from_pydata(vertices, [], faces)
        new_mesh.update()

        new_object = bpy.data.objects.new('mean', new_mesh)
        bpy.context.collection.objects.link(new_object)

        lms = get_face_landmark_vertices(new_object)
        lms_object = bpy.data.objects.new(f'{new_object.name}_lms', lms)
        bpy.context.collection.objects.link(lms_object)
        lms_object.parent = new_object
        set_active_object(new_object)

        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = new_object
        bpy.data.objects[new_object.name].select_set(True)

        bpy.context.active_object.rotation_mode = 'XYZ'
        bpy.context.active_object.rotation_euler = (radians(90), 0, 0)
        print('Done.')

        return {'FINISHED'}


class FaceAddRandomShape(bpy.types.Operator):
    """Apply random shape to face
    """
    bl_idname = "object.face_add_random_shape"
    bl_label = "Random"
    bl_description = "Sets all shape blend shape keys to a random value"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        """Check whether the panel can be drawn"""
        try:  # Enable only if in object mode
            if context.active_object is None or context.active_object.mode == 'OBJECT':
                return True
            return False
        except:
            return False

    def execute(self, context):
        """Execute"""
        print('Adding random model mesh...')
        control_lms = config['MODEL']['regions']
        mu = gid_dict['mean']
        sigma = gid_dict['sigma']

        # Sample randomly from model distribution
        z = torch.tensor(np.random.multivariate_normal(mu, 1 * sigma, 1),
                         dtype=torch.float32).to(device)

        # Regions: 0: 'left side', 1: 'right side', 3: 'ears', 6: 'skull',
        #          7: 'forehead', 8: 'eyes', 9: 'nose', 10: 'mouth'
        delta = []
        for idx in control_lms.keys():  # [9, 0, 8, 3, 7, 1, 10, 6]:
            delta.append(torch.zeros(3 * len(control_lms[idx]), device=device).reshape(1, -1))

        vertices = model.decode(z.unsqueeze(0), delta)[-1][0].detach().numpy()
        vertices = vertices * (mean_std['std'] + 1e-7) + mean_std['mean']

        # Create new object with decoded vertices and rotate to correct orientation
        new_mesh = bpy.data.meshes.new('random_shape')
        new_mesh.from_pydata(vertices, [], faces)
        new_mesh.update()

        new_object = bpy.data.objects.new('random_shape', new_mesh)
        bpy.context.collection.objects.link(new_object)

        lms = get_face_landmark_vertices(new_object)
        lms_object = bpy.data.objects.new(f'{new_object.name}_lms', lms)
        bpy.context.collection.objects.link(lms_object)
        lms_object.parent = new_object
        set_active_object(new_object)

        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = new_object
        bpy.data.objects[new_object.name].select_set(True)

        bpy.context.active_object.rotation_mode = 'XYZ'
        bpy.context.active_object.rotation_euler = (radians(90), 0, 0)

        return {'FINISHED'}


class FaceEditLandmarks(bpy.types.Operator):
    """Edit the landmarks"""
    bl_idname = 'scene.face_edit_landmarks'
    bl_label = 'Edit Landmarks'
    bl_description = 'Edit landmark positions'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        """Check whether the panel can be drawn"""
        try:  # Enable button only if a mesh the active object
            return context.object.type == 'MESH'
        except:
            return False

    def execute(self, context):
        """Execute"""
        print('Enable landmark editing...')
        if context.active_object.name.endswith('_lms'):
            set_active_object(context.active_object.name.split('_lms')[0])
        mesh_name = context.active_object.name

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        clear_object_selection()
        set_active_object(f'{mesh_name}_lms')

        # Put object in edit move and set cursor to move setting
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.wm.tool_set_by_id(name="builtin.move")

        return {'FINISHED'}


class FaceLoadModel(bpy.types.Operator):
    """Add model to scene
    """
    bl_idname = "scene.face_load_model"
    bl_label = "Load Model"
    bl_description = "Load selected face model to scene"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        """Check whether the panel can be drawn"""
        try:  # Enable button only if in Object Mode
            if context.active_object is None or context.active_object.mode == 'OBJECT':
                return True
            return False
        except:
            return False

    def load_model(self, path):
        """Load Model"""
        print(CURR_DIR)
        global model
        global config
        global device
        checkpoint_name = 'checkpoint.pth'
        device_ids = [-1]
        device = torch.device("cpu" if device_ids[0] == -1 else f"cuda:{device_ids[0]:%d}")

        # Load config file and update paths
        config_file = os.path.join(path, 'config_file.yaml')
        config_file = config_file.replace('\\', '/')
        config = read_yaml(config_file)
        config_json = json.dumps(config, indent=1, ensure_ascii=True)
        config['local_device_ids'] = device_ids
        config['MODEL']['face_part_ids_file'] = os.path.join(
            CURR_DIR, 'LAMM', config['MODEL']['face_part_ids_file'])
        config['MODEL']['manipulation'] = True

        # # Set Enum Properties for facial regions
        # bpy.context.window_manager.face_tool = EnumProperty(
        #     name = 'Region',
        #     description = 'Mesh Regions',
        #     items = [ ('0', 'Left Side', ''), ('1', 'Right Side', ''), ('3', 'Ears', '') ]
        # )
        # bpy.types.WindowManager.face_tool.face_region.items.append[
        #     ('0', 'Left Side', ''), ('1', 'Right Side', '')]

        # Load model
        model = LAMM(config['MODEL']).to(device)

        checkpoint = os.path.join('/'.join(config_file.split('/')[:-1]), checkpoint_name)
        load_from_checkpoint(model, checkpoint, partial_restore=False)

        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.to(device)
        return model

    def get_model_stats(self, path):
        """Load model attributes"""
        global gid_dict
        with open(os.path.join(path, 'gaussian_id.pickle'), 'rb') as f:
            gid_dict = pickle.load(f)

        global mean_std
        with open(os.path.join(path, 'mean_std.pickle'), 'rb') as f:
            mean_std = pickle.load(f, encoding='latin1')

        global disp_stats
        with open(os.path.join(path, 'displacement_stats.pickle'), 'rb') as f:
            disp_stats = pickle.load(f)

        global faces
        head_mesh = tm.load(os.path.join(path, 'template.obj'))
        faces = head_mesh.faces

    def execute(self, context):
        """Execute"""
        model_path = context.scene.conf_path
        model_name = model_path.split('\\')[-1]
        print(f'Loading model: {model_name}...')

        self.load_model(model_path)
        self.get_model_stats(os.path.join(model_path, 'files'))
        print('Loaded.')
        return {'FINISHED'}


class FaceRandomiseRegion(bpy.types.Operator):
    """Randomise the shape of a specific region"""
    bl_idname = 'object.face_randomise_region'
    bl_label = 'Randomise Region'
    bl_description = 'Randomise the shape for a given mesh region'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        """Check whether the panel can be drawn"""
        try:  # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except:
            return False

    def update_face_landmarks(self, mesh):
        """Update face landmarks"""
        new_lms = get_face_landmark_vertices(mesh)
        mesh_lms = get_object(f'{mesh.name}_lms')

        for i, vert in enumerate(new_lms.vertices):
            mesh_lms.data.vertices[i].co = vert.co

    def execute(self, context):
        """Execute"""
        control_lms = config['MODEL']['regions']
        region = int(context.window_manager.face_tool.face_region)
        print(f'Sampling region: {region}')

        # Get existing mesh
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        if context.active_object.name.endswith('_lms'):
            set_active_object(context.active_object.name.split('_lms')[0])
        mesh_name = context.active_object.name
        mesh = get_object(mesh_name)

        # Randomise deltas in the region of interest
        deltas = []
        for idx in control_lms.keys():
            if idx != region:
                deltas.append(torch.zeros(3 * len(control_lms[idx]), device=device).reshape(1, -1))
            else:
                deltas.append(torch.tensor(np.random.multivariate_normal(disp_stats[region]['mean'],
                                          disp_stats[region]['std'], 1),
                                          dtype=torch.float32, device=device))

        # Get mesh vertices
        vertices = np.ones(len(mesh.data.vertices) * 3)
        mesh.data.vertices.foreach_get("co", vertices)
        vertices = torch.tensor(vertices.reshape(1, -1, 3), dtype=torch.float32)
        vertices = (vertices - mean_std['mean']) / (mean_std['std'] + 1e-7)
        vertices = vertices.to(torch.float32).to(device)

        id_token, _ = model.encode(vertices)
        y = model.decode(id_token, deltas)[-1][0].detach().numpy()
        y = (y * (mean_std['std'] + 1e-7) + mean_std['mean'])

        # Update the mesh vertices
        for i, y_vert in enumerate(y):
            mesh.data.vertices[i].co = y_vert
        self.update_face_landmarks(mesh)

        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = mesh
        bpy.data.objects[mesh.name].select_set(True)

        bpy.context.active_object.rotation_mode = 'XYZ'
        bpy.context.active_object.rotation_euler = (radians(90), 0, 0)

        return {'FINISHED'}


class FaceResetShape(bpy.types.Operator):
    """Reset face texture"""
    bl_idname = "object.face_reset_shape"
    bl_label = "Reset"
    bl_description = "Resets all blend shape keys for shape"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        """Check whether the panel can be drawn"""
        try:  # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except:
            return False

    def update_face_landmarks(self, mesh):
        """Update face landmarks"""
        new_lms = get_face_landmark_vertices(mesh)
        mesh_lms = get_object(f'{mesh.name}_lms')

        for i, vert in enumerate(new_lms.vertices):
            mesh_lms.data.vertices[i].co = vert.co

    def execute(self, context):
        """Execute"""
        print('Resetting mesh to mean shape...')
        # Ensure the mesh is set as the active object
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        if context.active_object.name.endswith('_lms'):
            set_active_object(context.active_object.name.split('_lms')[0])
        mesh_name = context.active_object.name
        mesh = get_object(mesh_name)

        # Get mean vertices
        control_lms = config['MODEL']['regions']
        delta = []
        for idx in control_lms.keys():
            delta.append(torch.zeros(3 * len(control_lms[idx]), device=device).reshape(1, -1))

        z_mean = np.array([gid_dict['mean']])
        vertices = model.decode(torch.tensor(z_mean).unsqueeze(0), delta)[-1][0].detach().numpy()
        vertices = vertices * (mean_std['std'] + 1e-7) + mean_std['mean']

        # Update the mesh vertices and corresponding landmarks
        for i, vert in enumerate(vertices):
            mesh.data.vertices[i].co = vert
        self.update_face_landmarks(mesh)

        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = mesh
        bpy.data.objects[mesh.name].select_set(True)

        bpy.context.active_object.rotation_mode = 'XYZ'
        bpy.context.active_object.rotation_euler = (radians(90), 0, 0)
        print('Done.')

        return {'FINISHED'}


class FaceUpdateShape(bpy.types.Operator):
    """Get the landmark deltas and update face shape
    """
    bl_idname = 'object.face_update_shape'
    bl_label = 'Update Face Shape'
    bl_description = 'Update face shape based on identity and landmark deltas'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        """Check whether the panel can be drawn"""
        try:  # Enable button only if a mesh the active object
            return context.object.type == 'MESH'
        except:
            return False

    def get_landmark_deltas(self, orig_lms, curr_lms):
        """Calculate landmark displacements and normalise appropriately."""
        delta_list = []
        for curr_lm, orig_lm in zip(curr_lms, orig_lms):
            delta_list.append(curr_lm.co - orig_lm.co)

        control_lms = config['MODEL']['regions']
        delta_tensor = []
        sum_deltas = 0

        start_lm = 0
        for idx in control_lms.keys():
            n_lms = len(control_lms[idx])
            d = torch.tensor(delta_list[start_lm:(start_lm + n_lms)]).reshape(-1)
            indices = config['MODEL']['regions'][idx]
            std_ = torch.tensor((mean_std['std'][indices] + 1e-7).reshape(-1), dtype=torch.float32)

            delta_tensor.append((d / std_).reshape(1, -1))
            sum_deltas += torch.sum(torch.abs(delta_tensor[-1]))
            start_lm += n_lms
        return delta_tensor, sum_deltas

    def update_face_landmarks(self, mesh):
        """Update face landmarks"""
        new_lms = get_face_landmark_vertices(mesh)
        mesh_lms = get_object(f'{mesh.name}_lms')

        for i, vert in enumerate(new_lms.vertices):
            mesh_lms.data.vertices[i].co = vert.co

    def execute(self, context):
        """Execute"""
        print('Updating mesh from landmark deltas...')
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        if context.active_object.name.endswith('_lms'):
            set_active_object(context.active_object.name.split('_lms')[0])
        mesh_name = context.active_object.name
        mesh = get_object(mesh_name)

        # Check original landmark positions
        curr_lms = bpy.data.objects[f'{mesh_name}_lms'].data.vertices
        orig_lms = get_face_landmark_vertices(mesh).vertices
        deltas, sum_deltas = self.get_landmark_deltas(orig_lms, curr_lms)
        if sum_deltas == 0:  # Exit early if landmarks have not been moved
            return {'FINISHED'}

        # Get mesh vertices
        vertices = np.ones(len(mesh.data.vertices) * 3)
        mesh.data.vertices.foreach_get("co", vertices)
        vertices = torch.tensor(vertices.reshape(1, -1, 3), dtype=torch.float32)
        vertices = (vertices - mean_std['mean']) / (mean_std['std'] + 1e-7)
        vertices = vertices.to(torch.float32).to(device)

        id_token, _ = model.encode(vertices)
        y = model.decode(id_token, deltas)[-1][0].detach().numpy()
        y = (y * (mean_std['std'] + 1e-7) + mean_std['mean'])

        # Update the mesh vertices
        for i, y_vert in enumerate(y):
            mesh.data.vertices[i].co = y_vert
        self.update_face_landmarks(mesh)

        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = mesh
        bpy.data.objects[mesh.name].select_set(True)

        bpy.context.active_object.rotation_mode = 'XYZ'
        bpy.context.active_object.rotation_euler = (radians(90), 0, 0)
        print('Done.')

        return {'FINISHED'}


################################## Panel Classes ##################################
class LAMM_PT_Model(bpy.types.Panel):
    """Face loading UI
    """
    bl_label = "Face Model"
    bl_category = "LAMM"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        """Draw panel"""
        layout = self.layout
        col = layout.column(align=True)

        col.prop(context.scene, 'conf_path')
        col.operator("scene.face_load_model", text="Load Model")

        col.separator()
        col.operator("object.face_add_mean_mesh", text="Add Model Mean")

        col.separator()
        col.label(text='Sample:')
        col.operator("object.face_add_random_shape", text="Add Random Instance")

        col.separator()
        col.prop(context.window_manager.face_tool, "face_region")
        col.operator("object.face_randomise_region", text="Randomise Region")

        col.separator()
        col.operator("object.face_reset_shape", text="Reset To Mean")


class LAMM_PT_Landmarks(bpy.types.Panel):
    """Landmarks UI"""
    bl_label = "Landmarks"
    bl_category = "LAMM"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        """Draw panel"""
        layout = self.layout
        col = layout.column(align=True)

        col.separator()
        col.operator("scene.face_edit_landmarks", text="Edit Landmarks")

        col.separator()
        col.operator("object.face_update_shape", text="Update Shape")


classes = [
    PG_FaceProperties,
    FaceAddMeanMesh,
    FaceAddRandomShape,
    FaceEditLandmarks,
    FaceLoadModel,
    FaceRandomiseRegion,
    FaceResetShape,
    FaceUpdateShape,
    LAMM_PT_Model,
    LAMM_PT_Landmarks,
]


def register():
    """Run when enabling add-on"""
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.conf_path = bpy.props.StringProperty(
        name='Model',
        default=os.path.join(CHECKPOINT_DIR, DEFAULT_CHECKPOINT),
        description='Define the root path of the project',
        subtype = 'DIR_PATH'
    )

    # Store properties under WindowManager (not Scene) so that they are not saved
    #     in .blend files and always show default values after loading
    bpy.types.WindowManager.face_tool = PointerProperty(type=PG_FaceProperties)
    print('CUDA:', torch.cuda.is_available())


def unregister():
    """Run when diabling add-on"""
    for cls in classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.WindowManager.face_tool
    del bpy.types.Scene.conf_path


if __name__ == "__main__":
    register()
