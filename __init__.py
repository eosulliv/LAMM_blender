"""Module providing means of editing mesh shape via contorl points or landmarks.
"""

import os
import json
import pickle
import sys

from math import radians
from mathutils import Vector, Quaternion

import numpy as np
import trimesh as tm

import bpy
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                       IntProperty, PointerProperty, StringProperty)
from bpy.types import PropertyGroup, MeshVertex

# Add path to packages
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
CHECKPOINT_DIR = os.path.join(CURR_DIR, 'LAMM', 'assets', 'checkpoints')
if len(os.listdir(CHECKPOINT_DIR)) >= 1:
    DEFAULT_CHECKPOINT = sorted(os.listdir(CHECKPOINT_DIR))[0]
else:
    DEFAULT_CHECKPOINT = ''

sys.path.append(os.path.join(CURR_DIR, 'packages'))
import trimesh as tm
import torch

# Add path to model utils
sys.path.append(CURR_DIR)
from LAMM.utils.config_utils import read_yaml
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


#################################### Functions ####################################
def clear_object_selection():
    """Deselect all objects"""
    bpy.ops.object.select_all(action='DESELECT')


def get_landmark_vertices(mesh):
    """Get mesh landmarks from control landmark indices"""
    control_lms = config['MODEL']['control_vertices']
    lms = []
    for vals in control_lms.values():
        for idx in vals:
            lms.append(mesh.data.vertices[idx].co)

    lms_mesh = bpy.data.meshes.new('lms')
    lms_mesh.from_pydata(lms, landmark_edges, [])
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
class PG_LAMMProperties(PropertyGroup):
    """Properties"""
    lamm_region: EnumProperty(
        name = 'Region',
        description = 'Mesh Regions',
        items = [ ('0', '0', ''), ('1', '1', ''), ('2', '2', ''), ('3', '3', ''),
                  ('4', '4', ''), ('5', '5', ''), ('6', '6', ''), ('7', '7', ''),
                  ('8', '8', ''), ('9', '9', ''), ('10','10', ''), ('11', '11', ''),
                  ('12', '12', ''), ('13', '13', ''), ('14','14', ''), ('15', '15', '') ]
    )

    enable_smoothing: BoolProperty(
        name="Enable Smoothing",
        description="Enable smoothing on region boundaries",
        default=True
    )

    smoothing_iterations: IntProperty(name='Iterations', default=10, min=1, max=200)

    smoothing_rings: IntProperty(name='Smoothing Rings', default=1, min=1, max=10)


class LAMMAddMeanMesh(bpy.types.Operator):
    """Add model mean mesh to scene"""
    bl_idname = 'object.lamm_add_mean_mesh'
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
        control_lms = config['MODEL']['control_vertices']
        delta = []
        for idx in control_lms.keys():
            delta.append(torch.zeros(3 * len(control_lms[idx]), device=device).reshape(1, -1))

        # Create mesh
        z_mean = np.array([gid_dict['mean']])
        vertices = model.decode(torch.tensor(z_mean).unsqueeze(0), delta)[-1][0].detach().numpy()
        vertices = vertices * (mean_std['std'] + 1e-7) + mean_std['mean'] - vertex_mean

        new_mesh = bpy.data.meshes.new('mean')
        new_mesh.from_pydata(vertices, [], faces)
        new_mesh.update()

        mesh_object = bpy.data.objects.new('mean', new_mesh)
        bpy.context.collection.objects.link(mesh_object)

        # Get mesh landmarks
        lms = get_landmark_vertices(mesh_object)
        lms_object = bpy.data.objects.new(f'{mesh_object.name}_lms', lms)
        bpy.context.collection.objects.link(lms_object)
        lms_object.parent = mesh_object
        set_active_object(mesh_object)

        # Add id and original mesh landmarks to the mesh as an attribute
        mesh_object['id'] = np.array([gid_dict['mean']])
        print(lms)
        mesh_object['orig_lms'] = lms.copy()

        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = mesh_object
        bpy.data.objects[mesh_object.name].select_set(True)

        bpy.context.active_object.rotation_mode = 'XYZ'
        bpy.context.active_object.rotation_euler = (radians(90), 0, 0)
        print('Done.')

        return {'FINISHED'}


class LAMMAddRandomShape(bpy.types.Operator):
    """Apply random mesh shape
    """
    bl_idname = "object.lamm_add_random_shape"
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
        control_lms = config['MODEL']['control_vertices']
        mu = gid_dict['mean']
        sigma = gid_dict['sigma']

        # Sample randomly from model distribution
        z = torch.tensor(np.random.multivariate_normal(mu, 1 * sigma, 1),
                         dtype=torch.float32).to(device)

        delta = []
        for idx in control_lms.keys():
            delta.append(torch.zeros(3 * len(control_lms[idx]), device=device).reshape(1, -1))

        vertices = model.decode(z.unsqueeze(0), delta)[-1][0].detach().numpy()
        vertices = vertices * (mean_std['std'] + 1e-7) + mean_std['mean'] - vertex_mean

        # Create new object with decoded vertices and rotate to correct orientation
        new_mesh = bpy.data.meshes.new('random_shape')
        new_mesh.from_pydata(vertices, [], faces)
        new_mesh.update()

        new_object = bpy.data.objects.new('random_shape', new_mesh)
        bpy.context.collection.objects.link(new_object)

        # Get mesh landmarks
        lms = get_landmark_vertices(new_object)
        lms_object = bpy.data.objects.new(f'{new_object.name}_lms', lms)
        bpy.context.collection.objects.link(lms_object)
        lms_object.parent = new_object
        set_active_object(new_object)

        # Add id and mesh deltas to the mesh as an attribute
        new_object['id'] = np.array([np.asarray(z).reshape((-1,))])
        new_object['orig_lms'] = lms.copy()

        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = new_object
        bpy.data.objects[new_object.name].select_set(True)

        bpy.context.active_object.rotation_mode = 'XYZ'
        bpy.context.active_object.rotation_euler = (radians(90), 0, 0)

        return {'FINISHED'}


class LAMMEditLandmarks(bpy.types.Operator):
    """Edit the landmarks"""
    bl_idname = 'scene.lamm_edit_landmarks'
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


class LAMMLoadMesh(bpy.types.Operator):
    """Add model to scene
    """
    bl_idname = "scene.lamm_load_mesh"
    bl_label = "Load Mesh"
    bl_description = "Load selected mesh to scene"
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

    def execute(self, context):
        """Execute"""
        mesh_path = context.scene.mesh_path
        if not os.path.isfile(mesh_path):
            print('File does not exist... Can\'t load mesh...')
            return {'FINISHED'}

        mesh_name = mesh_path.split('\\')[-1].split('.')[0]
        print(f'Loading mesh: {mesh_name}...')
        mesh = tm.load(mesh_path)

        # Load mesh and centre it about the origin
        new_mesh = bpy.data.meshes.new(mesh_name)
        vertices = mesh.vertices - np.mean(mesh.vertices, axis=0)
        new_mesh.from_pydata(vertices, [], mesh.faces)
        new_mesh.update()

        mesh_object = bpy.data.objects.new(mesh_name, new_mesh)
        bpy.context.collection.objects.link(mesh_object)

        # Get mesh landmarks
        lms = get_landmark_vertices(mesh_object)
        lms_object = bpy.data.objects.new(f'{mesh_object.name}_lms', lms)
        bpy.context.collection.objects.link(lms_object)
        lms_object.parent = mesh_object
        set_active_object(mesh_object)

        # # Get mesh vertices
        verts = np.ones(len(mesh_object.data.vertices) * 3)
        mesh_object.data.vertices.foreach_get("co", verts)
        verts = torch.tensor(mesh.vertices.reshape(1, -1, 3), dtype=torch.float32)
        verts += vertex_mean
        verts = (verts - mean_std['mean']) / (mean_std['std'] + 1e-7)
        verts = verts.to(torch.float32).to(device)

        # Get current id token and decode with updated deltas
        mesh_object['id'], _ = model.encode(verts)
        mesh_object['orig_lms'] = lms.copy()

        # bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = mesh_object
        bpy.data.objects[mesh_object.name].select_set(True)

        bpy.context.active_object.rotation_mode = 'XYZ'
        bpy.context.active_object.rotation_euler = (radians(90), 0, 0)

        print('Loaded.')
        return {'FINISHED'}


class LAMMLoadModel(bpy.types.Operator):
    """Add model to scene
    """
    bl_idname = "scene.lamm_load_model"
    bl_label = "Load Model"
    bl_description = "Load selected model to scene"
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
        config['MODEL']['region_ids_file'] = os.path.join(
            CURR_DIR, 'LAMM', config['MODEL']['region_ids_file'])
        config['MODEL']['manipulation'] = True

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
        mesh = tm.load(os.path.join(path, 'template.obj'))
        faces = mesh.faces

        global vertex_mean
        vertex_mean = np.asarray(np.mean(mesh.vertices, axis=0))

        global landmark_edges
        landmark_file = os.path.join(path, 'landmark_edges.yaml')
        landmark_edges = []
        if os.path.isfile(landmark_file):
            landmark_json = read_yaml(landmark_file)
            landmark_edges = landmark_json['landmark_edges']

        # Pinned vertices for smoothing
        global pinned_verts
        boundaries_file = os.path.join(path, 'region_boundaries.pickle')
        pinned_verts = []
        if os.path.isfile(boundaries_file):
            with open(boundaries_file, 'rb') as f:
                region_boundaries = pickle.load(f)

            boundary_verts_list = [region_boundaries[key] for key in region_boundaries.keys()]
            boundary_verts = [vert for verts in boundary_verts_list for vert in verts]

            pinned_verts = np.ones(mesh.vertices.shape[0], dtype=bool)
            pinned_verts[boundary_verts] = 0
            pinned_verts = np.where(pinned_verts)[0]

    def execute(self, context):
        """Execute"""
        model_path = context.scene.model_path
        model_name = model_path.split('\\')[-1]
        print(f'Loading model: {model_name}...')

        self.load_model(model_path)
        self.get_model_stats(os.path.join(model_path))
        print('Loaded.')
        return {'FINISHED'} 


class LAMMRandomiseRegion(bpy.types.Operator):
    """Randomise the shape of a specific region"""
    bl_idname = 'object.lamm_randomise_region'
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

    def update_lamm_landmarks(self, mesh):
        """Update mesh landmarks"""
        new_lms = get_landmark_vertices(mesh)
        mesh_lms = get_object(f'{mesh.name}_lms')

        for i, vert in enumerate(new_lms.vertices):
            mesh_lms.data.vertices[i].co = vert.co

    def execute(self, context):
        """Execute"""
        control_lms = config['MODEL']['control_vertices']
        region = int(context.window_manager.lamm_tool.lamm_region)
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
        vertices += vertex_mean
        vertices = (vertices - mean_std['mean']) / (mean_std['std'] + 1e-7)
        vertices = vertices.to(torch.float32).to(device)

        # Get the mesh id token and update shape
        id_token =  torch.tensor([[context.active_object['id'].to_list()]])
        y = model.decode(id_token, deltas)[-1][0].detach().numpy()
        y = y * (mean_std['std'] + 1e-7) + mean_std['mean'] - vertex_mean

        # Smooth mesh vertices in boundary regions if smoothing enabled
        if context.window_manager.lamm_tool.enable_smoothing:
            mesh_tm = tm.Trimesh(y, faces)
            laplacian_op = tm.smoothing.laplacian_calculation(
                mesh_tm, equal_weight=False, pinned_vertices=pinned_verts)
            mesh_tm = tm.smoothing.filter_taubin(mesh_tm, laplacian_operator=laplacian_op)
            y = mesh_tm.vertices

        # Update the mesh vertices
        for i, y_vert in enumerate(y):
            mesh.data.vertices[i].co = y_vert
        self.update_lamm_landmarks(mesh)

        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = mesh
        bpy.data.objects[mesh.name].select_set(True)

        bpy.context.active_object.rotation_mode = 'XYZ'
        bpy.context.active_object.rotation_euler = (radians(90), 0, 0)

        return {'FINISHED'}


class LAMMResetShape(bpy.types.Operator):
    """Reset mesh shape"""
    bl_idname = "object.lamm_reset_shape"
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

    def update_lamm_landmarks(self, mesh):
        """Update mesh landmarks"""
        new_lms = get_landmark_vertices(mesh)
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
        mesh['deltas'] = np.zeros((len(bpy.data.objects['mean_lms'].data.vertices), 3))

        # Get mean vertices
        control_lms = config['MODEL']['control_vertices']
        delta = []
        for idx in control_lms.keys():
            delta.append(torch.zeros(3 * len(control_lms[idx]), device=device).reshape(1, -1))

        z_mean = np.array([gid_dict['mean']])
        vertices = model.decode(torch.tensor(z_mean).unsqueeze(0), delta)[-1][0].detach().numpy()
        vertices = vertices * (mean_std['std'] + 1e-7) + mean_std['mean'] - vertex_mean

        # Update the mesh vertices and corresponding landmarks
        for i, vert in enumerate(vertices):
            mesh.data.vertices[i].co = vert
        self.update_lamm_landmarks(mesh)

        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = mesh
        bpy.data.objects[mesh.name].select_set(True)

        bpy.context.active_object.rotation_mode = 'XYZ'
        bpy.context.active_object.rotation_euler = (radians(90), 0, 0)
        print('Done.')

        return {'FINISHED'}


class LAMMUpdateShape(bpy.types.Operator):
    """Get the landmark deltas and update mesh shape
    """
    bl_idname = 'object.lamm_update_shape'
    bl_label = 'Update Mesh Shape'
    bl_description = 'Update mesh shape based on identity and landmark deltas'
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

        control_lms = config['MODEL']['control_vertices']
        delta_tensor = []
        sum_deltas = 0

        start_lm = 0
        for idx in control_lms.keys():
            n_lms = len(control_lms[idx])
            d = torch.tensor(delta_list[start_lm:(start_lm + n_lms)]).reshape(-1)
            indices = config['MODEL']['control_vertices'][idx]
            std_ = torch.tensor((mean_std['std'][indices] + 1e-7).reshape(-1), dtype=torch.float32)

            delta_tensor.append((d / std_).reshape(1, -1))
            sum_deltas += torch.sum(torch.abs(delta_tensor[-1]))
            start_lm += n_lms
        return delta_tensor, sum_deltas

    def update_lamm_landmarks(self, mesh):
        """Update mesh landmarks"""
        new_lms = get_landmark_vertices(mesh)
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
        orig_lms = mesh['orig_lms'].vertices  # get_landmark_vertices(mesh).vertices
        deltas, sum_deltas = self.get_landmark_deltas(orig_lms, curr_lms)
        if sum_deltas == 0:  # Exit early if landmarks have not been moved
            print('No landmark updates required...')
            return {'FINISHED'}

        # Get mesh vertices
        vertices = np.ones(len(mesh.data.vertices) * 3)
        mesh.data.vertices.foreach_get("co", vertices)
        vertices = torch.tensor(vertices.reshape(1, -1, 3), dtype=torch.float32)
        vertices += vertex_mean
        vertices = (vertices - mean_std['mean']) / (mean_std['std'] + 1e-7)
        vertices = vertices.to(torch.float32).to(device)

        # Get current id token and decode with updated deltas
        id_token = torch.tensor([[context.active_object['id'].to_list()]])
        y = model.decode(id_token, deltas)[-1][0].detach().numpy()
        y = y * (mean_std['std'] + 1e-7) + mean_std['mean'] - vertex_mean

        # Smooth mesh vertices in boundary regions if smoothing enabled
        if context.window_manager.lamm_tool.enable_smoothing:
            mesh_tm = tm.Trimesh(y, faces)
            n_iterations = context.window_manager.lamm_tool.smoothing_iterations
            laplacian_op = tm.smoothing.laplacian_calculation(
                mesh_tm, equal_weight=False, pinned_vertices=pinned_verts)
            mesh_tm = tm.smoothing.filter_taubin(mesh_tm, iterations=n_iterations, laplacian_operator=laplacian_op)
            y = mesh_tm.vertices

        # Update the mesh vertices
        for i, y_vert in enumerate(y):
            mesh.data.vertices[i].co = y_vert
        self.update_lamm_landmarks(mesh)

        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = mesh
        bpy.data.objects[mesh.name].select_set(True)

        bpy.context.active_object.rotation_mode = 'XYZ'
        bpy.context.active_object.rotation_euler = (radians(90), 0, 0)
        print('Done.')

        return {'FINISHED'}


################################## Panel Classes ##################################
class LAMM_PT_Model(bpy.types.Panel):
    """Mesh Loading UI
    """
    bl_label = "Model"
    bl_category = "LAMM"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        """Draw panel"""
        layout = self.layout
        col = layout.column(align=True)

        col.prop(context.scene, 'model_path')
        col.operator("scene.lamm_load_model", text="Load Model")

        col.separator()
        col.operator("object.lamm_add_mean_mesh", text="Add Model Mean")

        # col.separator()
        # col.prop(context.scene, 'mesh_path')
        # col.operator("scene.lamm_load_mesh", text="Load Mesh")

        col.separator()
        col.label(text='Sample:')
        col.operator("object.lamm_add_random_shape", text="Add Random Instance")


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
        col.operator("scene.lamm_edit_landmarks", text="Edit Landmarks")

        col.separator()
        col.operator("object.lamm_update_shape", text="Update Shape")
        col.prop(context.window_manager.lamm_tool, "enable_smoothing")
        col.prop(context.window_manager.lamm_tool, "smoothing_iterations")

        col.separator()
        col.prop(context.window_manager.lamm_tool, "lamm_region")
        col.operator("object.lamm_randomise_region", text="Randomise Region")

        col.separator()
        col.operator("object.lamm_reset_shape", text="Reset To Mean")


classes = [
    PG_LAMMProperties,
    LAMMAddMeanMesh,
    LAMMAddRandomShape,
    LAMMEditLandmarks,
    LAMMLoadModel,
    LAMMLoadMesh,
    LAMMRandomiseRegion,
    LAMMResetShape,
    LAMMUpdateShape,
    LAMM_PT_Model,
    LAMM_PT_Landmarks,
]


def register():
    """Run when enabling add-on"""
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.model_path = bpy.props.StringProperty(
        name='Model',
        default=os.path.join(CHECKPOINT_DIR, DEFAULT_CHECKPOINT),
        description='Define the root path of the project',
        subtype='DIR_PATH'
    )

    bpy.types.Scene.mesh_path = bpy.props.StringProperty(
        name='Mesh',
        default='',
        description='Load a mesh from files',
        subtype='FILE_PATH'
    )

    # Store properties under WindowManager (not Scene) so that they are not saved
    #     in .blend files and always show default values after loading
    bpy.types.WindowManager.lamm_tool = PointerProperty(type=PG_LAMMProperties)
    print('CUDA:', torch.cuda.is_available())


def unregister():
    """Run when diabling add-on"""
    for cls in classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.WindowManager.lamm_tool
    del bpy.types.Scene.model_path
    del bpy.types.Scene.mesh_path


if __name__ == "__main__":
    register()  
