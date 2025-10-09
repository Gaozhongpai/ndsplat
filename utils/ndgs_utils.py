import numpy as np
import os
import math
import torch
import cv2
import glfw
import OpenGL.GL as gl
# import flash_gaussian_splatting
from utils.graphics_utils import fov2focal

class Rasterizer:
    # 构造函数中分配内存
    def __init__(self, num_vertex, device):
        
        MAX_NUM_RENDERED = 2 ** 24
        SORT_BUFFER_SIZE = 2 ** 30
        MAX_NUM_TILES = 2 ** 20
        self.max_num_rendered = MAX_NUM_RENDERED
        # 24byte
        self.gaussian_keys_unsorted = torch.empty(MAX_NUM_RENDERED, device=device, dtype=torch.int64)
        self.gaussian_values_unsorted = torch.empty(MAX_NUM_RENDERED, device=device, dtype=torch.int32)
        self.gaussian_keys_sorted = torch.empty(MAX_NUM_RENDERED, device=device, dtype=torch.int64)
        self.gaussian_values_sorted = torch.empty(MAX_NUM_RENDERED, device=device, dtype=torch.int32)

        self.list_sorting_space = torch.empty(SORT_BUFFER_SIZE, device=device, dtype=torch.int8)
        self.ranges = torch.empty((MAX_NUM_TILES, 2), device=device, dtype=torch.int32)
        self.curr_offset = torch.empty(1, device=device, dtype=torch.int32)

        self.device = device
        # 9*4byte
        self.points_xy = torch.empty((num_vertex, 2), device=device, dtype=torch.float32)
        self.depths = torch.empty(num_vertex, device=device, dtype=torch.float32)
        self.rgb = torch.empty((num_vertex, 3), device=device, dtype=torch.float32)
        self.conic_opacity = torch.empty((num_vertex, 4), device=device, dtype=torch.float32)

    # 前向传播（应用层封装）
    def forward(self, position, shs, opacity, cov3d, camera, bg_color):
        self.curr_offset.fill_(0)
        
        camera_position = camera.camera_center
        camera_rotation = camera.rotation

        camera_height = int(camera.image_height)
        camera_width = int(camera.image_width)
        camera_focal_x = fov2focal(camera.FoVx, camera.image_width)
        camera_focal_y = fov2focal(camera.FoVy, camera.image_height)
        camera_zFar = camera.zfar
        camera_zNear = camera.znear
                
        # 属性预处理 + 键值绑定
        flash_gaussian_splatting.ops.preprocess(position, shs.view(shs.shape[0], -1), opacity, cov3d,
                                    camera_width, camera_height, 32, 16,
                                    camera_position, camera_rotation,
                                    camera_focal_x, camera_focal_y, camera_zFar, camera_zNear,
                                    self.points_xy, self.depths, self.rgb, self.conic_opacity,
                                    self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                                    self.curr_offset)
        
        # 键值对数量判断 + 处理键值对过多的异常情况
        num_rendered = int(self.curr_offset.cpu()[0])
        # print(num_rendered)
        if num_rendered >= self.max_num_rendered:
            raise

        flash_gaussian_splatting.ops.sort_gaussian(num_rendered, camera_width, camera_height, 32, 16,
                                            self.list_sorting_space,
                                            self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                                            self.gaussian_keys_sorted, self.gaussian_values_sorted)
        # 排序 + 像素着色 + 混色阶段
        out_color = torch.zeros((camera_height, camera_width, 3), device=self.device, dtype=torch.uint8)
        flash_gaussian_splatting.ops.render_32x16(num_rendered, camera_width, camera_height,
                                self.points_xy, self.depths, self.rgb, self.conic_opacity,
                                self.gaussian_keys_sorted, self.gaussian_values_sorted,
                                self.ranges, bg_color, out_color)
        out_color = (out_color.float() / 255).permute(2, 0, 1).contiguous()

        return out_color
    
    
# self.anisotropy = nn.Parameter(torch.ones(num_gaussians) * 0.3)
def ct_color_model(base_color, view_direction, normal, anisotropy):
    cos_theta = torch.sum(view_direction * normal, dim=-1)
    
    # Henyey-Greenstein phase function with per-Gaussian anisotropy
    g = torch.clamp(anisotropy, -0.9, 0.9)  # Clamp to typical range
    
    phase = (1 - g**2) / (4 * math.pi * (1 + g**2 - 2*g*cos_theta)**1.5)
    
    factor = 0.5
    # Apply scattering effect to base color
    color = base_color * (1 + phase[:, None] * factor)  # Subtle scattering influence
    # Ensure color stays in valid range
    color = torch.sigmoid(color)
    return color

def ggx(roughness, NoH, NoV, NoL):
    alpha = roughness * roughness
    D = alpha / (math.pi * torch.pow(NoH * NoH * (alpha - 1) + 1, 2))
    G = 2 * NoL * NoV / (NoL + NoV - NoL * NoV + 1e-5)
    return D * G

# Microfacet BRDF: For more complex materials, we can implement a simplified
def microfacet_brdf(base_color, view_direction, normal, 
                    roughness=0.5, metallic=0.0, light_direction=torch.tensor([0, 1, 0])):
    
    H = torch.nn.functional.normalize(view_direction + light_direction, dim=-1)
    NoV = torch.clamp(torch.sum(normal * view_direction, dim=-1), 0.0, 1.0)
    NoL = torch.clamp(torch.sum(normal * light_direction, dim=-1), 0.0, 1.0)
    NoH = torch.clamp(torch.sum(normal * H, dim=-1), 0.0, 1.0)
    
    F0 = 0.04 * (1 - metallic) + base_color * metallic
    F = fresnel(F0, view_direction, normal)
    
    specular = ggx(roughness, NoH, NoV, NoL) * F
    diffuse = base_color * (1 - metallic) * (1 - F) * NoL / math.pi
    color = diffuse + specular
    return color

# Fresnel Effect: In the real world, surfaces tend to become more reflective at grazing
def fresnel(base_color, view_direction, normal, F0=0.04):
    cosTheta = torch.clamp(torch.sum(view_direction * normal, dim=-1), 0.0, 1.0)
    fresnel_factor = F0 + (1.0 - F0) * torch.pow(1.0 - cosTheta, 5.0)
    color = base_color * (1 - fresnel_factor) + torch.ones_like(base_color) * fresnel_factor
    return color

# Subsurface Scattering: For materials like skin or wax, we can approximate subsurface scattering:
def subsurface_approximation(base_color, view_direction, normal, thickness=0.5):
    NoV = torch.clamp(torch.sum(normal * view_direction, dim=-1), 0.0, 1.0)
    scatter = torch.exp(-thickness * (1 - NoV))
    color = base_color * (1 - scatter) + base_color * scatter * 0.25
    return color

# Anisotropic Reflections: For materials like brushed metal or hair:
def anisotropic_reflection(base_color, view_direction, normal, tangent, 
                        roughness=0.5, anisotropy=0.5, light_direction=torch.tensor([0, 1, 0])):
    bitangent = torch.cross(normal, tangent, dim=-1)
    H = torch.nn.functional.normalize(view_direction + light_direction, dim=-1)
    NoH = torch.clamp(torch.sum(normal * H, dim=-1), 0.0, 1.0)
    ToH = torch.sum(tangent * H, dim=-1)
    BoH = torch.sum(bitangent * H, dim=-1)
    
    roughness_x = roughness * (1 + anisotropy)
    roughness_y = roughness * (1 - anisotropy)
    
    D = torch.sqrt((ToH * ToH) / (roughness_x * roughness_x) + (BoH * BoH) / (roughness_y * roughness_y) + NoH * NoH)
    aniso_highlight = 1.0 / (roughness_x * roughness_y * D * D)
    
    color = base_color + aniso_highlight
    return color

# Iridescence: For materials like soap bubbles or certain insect wings:
def iridescence(base_color, view_direction, normal, iridescence_strength=0.5):
    import colorsys
    
    NoV = torch.clamp(torch.sum(normal * view_direction, dim=-1), 0.0, 1.0)
    hue = (NoV * 0.5 + 0.5) * 360  # Map NoV to hue
    iridescent_color = torch.tensor(colorsys.hsv_to_rgb(hue, 1, 1))
    color = base_color * (1 - iridescence_strength) + iridescent_color * iridescence_strength
    return color


# OpenGL & GLFW
def extract_rotation_scale_from_cov(cov):
    # Reconstruct full 3x3 covariance matrix
    full_cov = torch.zeros(cov.shape[0], 3, 3, device=cov.device)
    full_cov[:, 0, 0] = cov[:, 0]
    full_cov[:, 1, 0] = full_cov[:, 0, 1] = cov[:, 1]
    full_cov[:, 2, 0] = full_cov[:, 0, 2] = cov[:, 2]
    full_cov[:, 1, 1] = cov[:, 3]
    full_cov[:, 2, 1] = full_cov[:, 1, 2] = cov[:, 4]
    full_cov[:, 2, 2] = cov[:, 5]

    # Perform eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(full_cov)

    # Extract scale (square root of eigenvalues)
    scale = torch.sqrt(torch.abs(eigenvalues))

    # Extract rotation (eigenvectors)
    rotation = eigenvectors

    return scale, rotation

def create_cholesky_v2(diag: torch.Tensor, l_triang: torch.Tensor) -> torch.Tensor:
    L = torch.diag_embed(diag)
    N = diag.size(1)
    tril_indices = torch.tril_indices(N, N, offset=-1, device=diag.device)
    L[:, tril_indices[0], tril_indices[1]] = l_triang.view(diag.size(0), -1) * diag[:, tril_indices[0]]
    return torch.bmm(L, L.transpose(-1, -2)) 

def strip_lower_diag(L):
    return torch.stack([
        L[:, 0, 0].abs(), 
        L[:, 0, 1], 
        L[:, 0, 2],
        L[:, 1, 1].abs(), 
        L[:, 1, 2], 
        L[:, 2, 2].abs()
    ], dim=1)

def slice_gaussian(m_1, m_2, q, v, c_dim, lambda_opc=0.35):
    v_11 = v[:, :c_dim, :c_dim]
    v_12 = v[:, :c_dim, c_dim:]
    v_21 = v[:, c_dim:, :c_dim]
    v_22 = v[:, c_dim:, c_dim:]

    v_22_inv = torch.linalg.inv(v_22)
    v_regr = torch.bmm(v_12, v_22_inv)

    x = q - m_2

    direction_influence = torch.einsum('bi,bij,bj->b', x, v_22_inv, x).unsqueeze(-1)
    ## (0.5 : 168,087 : 33.91) -> (0.35 : 179,776 : 34.05) (0.25 : 192,511 : 34.22)
    # lambda_opc = 0.35 ## smaller -> more points -> higher quality 
    scale = torch.exp(-lambda_opc*direction_influence) 

    m_cond = m_1 + torch.bmm(v_regr, x.unsqueeze(-1)).squeeze(-1)
    v_cond = (v_11 - torch.bmm(v_regr, v_21)) # * scale.unsqueeze(-1) or (2 - scale.unsqueeze(-1))
    
    cov3D_precomp = strip_lower_diag(v_cond)
    return m_cond, cov3D_precomp, scale

def slice_gaussian_test(m_1, m_2, q, v_22_inv, v_regr, lambda_opc=0.35):
    x = q - m_2
    direction_influence = torch.einsum('bi,bij,bj->b', x, v_22_inv, x).unsqueeze(-1)
    scale = torch.exp(-lambda_opc*direction_influence)
    m_cond = m_1 + torch.bmm(v_regr, x.unsqueeze(-1)).squeeze(-1)
    return m_cond, scale

def optimized_slice_gaussian(m_1, m_2, q, diag, l_triang, c_dim=3):
    batch_size, N = diag.shape
    device = diag.device
    
    # Create L matrix more efficiently
    L = torch.zeros(batch_size, N, N, device=device)
    L.diagonal(dim1=-2, dim2=-1)[:] = diag
    tril_indices = torch.tril_indices(N, N, offset=-1, device=device)
    L[:, tril_indices[0], tril_indices[1]] = l_triang * diag[:, tril_indices[0]]
    
    # Split L into L_c and L_r
    L_c = L[:, :c_dim]
    L_r = L[:, c_dim:]
    
    # Compute v_12 directly
    v_12 = torch.bmm(L_c, L_r.transpose(-1, -2))
    
    # Compute v_22_L (Cholesky of v_22) directly from L_r
    v_22_L = torch.linalg.cholesky(torch.bmm(L_r, L_r.transpose(-1, -2)))
    
    # Compute x
    x = q - m_2
    
    # Solve systems and compute scale in one go
    temp = torch.linalg.solve_triangular(v_22_L, x.unsqueeze(-1), upper=False)
    scale = torch.exp(-0.5 * torch.sum(temp**2, dim=1, keepdim=True))
    
    # Combine v_regr and m_cond computations
    v_12_transformed = torch.linalg.solve_triangular(v_22_L, v_12.transpose(-1, -2), upper=False)
    m_cond = m_1 + torch.bmm(v_12_transformed.transpose(-1, -2), temp).squeeze(-1)
    
    # Compute v_cond more efficiently
    v_11 = torch.bmm(L_c, L_c.transpose(-1, -2))
    v_cond = (v_11 - torch.bmm(v_12_transformed.transpose(-1, -2), v_12_transformed)) * scale
    
    # Compute cov3D_precomp
    cov3D_precomp = torch.stack([
        v_cond[:, 0, 0].abs(), v_cond[:, 0, 1], v_cond[:, 0, 2],
        v_cond[:, 1, 1].abs(), v_cond[:, 1, 2], v_cond[:, 2, 2].abs()
    ], dim=1)
    
    return m_cond, cov3D_precomp, scale.squeeze(-1)


def bind_texture(texture_img, resolution):
    texture_id = gl.glGenTextures(1)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB16F, resolution[0], resolution[1], 0, gl.GL_RGB, gl.GL_FLOAT, texture_img)

    return texture_id


def replace_texture(texture_img, texture_id, resolution, exposure=1.0):
    texture_img = texture_img.detach()[0, :, :, :] * exposure
    texture_img = texture_img.float().cpu().numpy()
    texture_img = cv2.resize(texture_img, (resolution[0], resolution[1]), interpolation=cv2.INTER_NEAREST)

    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, resolution[0], resolution[1], gl.GL_RGB, gl.GL_FLOAT, texture_img)


def impl_glfw_init(width, height, window_name):
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window


def load_tensor_from_exr(filename, target_resolution=None):
    image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # OpenCV loads in BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if target_resolution is not None:
        image = cv2.resize(image, target_resolution, interpolation=cv2.INTER_AREA)

    # OpenCV has row major convention
    image = image.transpose(1, 0, 2)
    image = torch.from_numpy(image)

    return image


def create_3d_grid(resolution):
    x, y, z = torch.meshgrid([torch.linspace(0, 1, resolution[0]), torch.linspace(0, 1, resolution[1]), torch.linspace(0, 1, resolution[2])])
    grid = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)

    return grid


def create_2d_grid(resolution, start=0, end=0):
    x, y = torch.meshgrid([torch.linspace(start, end, resolution[0]), torch.linspace(start, end, resolution[1])])
    grid = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)

    return grid


def print_params(self):
    print("Number of model parameters:")
    params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print("%d" % params)


def get_params(variables, custom_values):
    params = []

    for i in range(len(variables)):
        # Don't inform the network of the camera params
        if 'sensor' in variables[i].id():
            continue
        for j in range(variables[i].num_parameters()):
            params.append(custom_values[i][j])

    return params


def stack_inputs_tensor(buffers, variables, custom_values):
    resolution = (buffers[0].shape[0], buffers[0].shape[1], 1)

    variable_buffers = []

    for i in range(len(variables)):
        # Don't inform the network of the camera params
        if 'sensor' in variables[i].id():
            continue
        for j in range(variables[i].num_parameters()):
            variable_buffers.append(torch.full(resolution, custom_values[i][j] * 2.0 - 1.0, device='cuda'))

    inputs = torch.cat([*buffers, *variable_buffers], 2)

    return inputs


def create_custom_values_tensor(variables, custom_values):
    res = torch.tensor([])

    for i in range(len(variables)):
        for j in range(variables[i].num_parameters()):
            res = torch.cat((res, torch.tensor(custom_values[i][j]).unsqueeze(0)))

    return res


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def linear_to_srgb(x, gamma=2.4):
    x = torch.where(x <= 0.0031308, x * 12.92, 1.055 * abs(x) ** (1 / gamma) - 0.055)

    return x


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def softplus(x, b=1.0):
    return torch.log(1 + torch.exp(b * x)) / b


def inverse_softplus(x, b=1.0):
    return torch.log(torch.exp(b * x) - 1) / b


def create_diag(diag):
    B, N = diag.shape

    L = torch.zeros(B, N, N, dtype=diag.dtype, device=diag.device)
    L.view(B, -1)[:, ::N + 1] = diag

    return L


def create_cholesky(diag, l_triang):
    L = create_triang(diag, l_triang)

    symm_matrix = torch.bmm(L, L.transpose(-1, -2))

    return symm_matrix


def compute_lr(step, lr_init, lr_final, max_steps=30000):
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return log_lerp


def create_cholesky_matrix(L):
    B, N, N = L.shape

    symm_matrix = torch.bmm(L, L.transpose(-1, -2))

    return symm_matrix


def get_cholesky(symm_matrix):
    B, N, N = symm_matrix.shape

    diag = symm_matrix[:, torch.arange(N), torch.arange(N)]

    s_inv = torch.zeros(B, N, N, dtype=diag.dtype, device=diag.device)
    s_inv.view(B, -1)[:, ::N + 1] = 1.0/(diag+1e-6)

    symm_matrix = torch.bmm(s_inv, symm_matrix)

    l_triang = symm_matrix[:, torch.tril_indices(N, N, offset=-1)[0], torch.tril_indices(N, N, offset=-1)[1]]

    return diag, l_triang


def create_triang(diag, l_triang):
    B, N = diag.shape

    L = torch.eye(N, dtype=diag.dtype, device=diag.device).unsqueeze(0).repeat(B, 1, 1)

    L[:, torch.tril_indices(N, N, offset=-1)[0], torch.tril_indices(N, N, offset=-1)[1]] = l_triang

    S = create_diag(diag)

    L = S @ L

    return L


def sample_from_gs(m, diags, l_triangs, n_samples):
    L = create_triang(diags, l_triangs).unsqueeze(1).repeat(1, n_samples, 1, 1)
    samples = torch.randn([m.shape[0], n_samples, m.shape[-1]], device=m.device)
    samples = torch.einsum('ijk,ijkm->ijk', samples, L) + m.unsqueeze(1).repeat(1, n_samples, 1)

    return samples


def copy_diag(matrix):
    B, N, _ = matrix.shape

    L = torch.zeros(B, N, N, dtype=matrix.dtype, device=matrix.device)
    L.view(B, -1)[:, ::N + 1] = matrix.view(B, -1)[:, ::N + 1]

    return L


def copy_l_triang(matrix):
    B, N, _ = matrix.shape

    L = torch.zeros(B, N, N, dtype=matrix.dtype, device=matrix.device)

    L[:, torch.tril_indices(N, N, offset=-1)[0], torch.tril_indices(N, N, offset=-1)[1]] = matrix[:, torch.tril_indices(N, N, offset=-1)[0], torch.tril_indices(N, N, offset=-1)[1]]

    return L


def matrix_to_list(mat):
    return list(map(list, list(mat)))


def update_learning_rate(optimizer, lr, param_name):
    for param_group in optimizer.param_groups:
        if param_group["name"] == param_name:
            param_group['lr'] = lr


def mask_optimizer(optimizer, mask, group_names):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if group.get('name') in group_names:
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


def remove_group_optimizer(optimizer, group_names):
    # Find the index of the parameter group with the specified name
    index_to_remove = None
    for i, group in enumerate(optimizer.param_groups):
        if group.get('name') in group_names:
            index_to_remove = i
            break

    # Remove the parameter group if found
    if index_to_remove is not None:
        optimizer.param_groups.pop(index_to_remove)


def cat_optimizer(optimizer, tensors_dict):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        assert len(group["params"]) == 1
        if group["name"] in tensors_dict:
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors


def normalize_vecs_np(vector):
    return vector / (np.linalg.norm(vector, axis=-1, keepdims=True))


def normalize_vecs_torch(vector):
    return vector / (torch.norm(vector, dim=-1, keepdim=True))


def translation_matrix(translation):
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    return matrix


def rotation_matrix(axis, theta):
    theta = theta * np.pi / 180.0
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
                     [0, 0, 0, 1]])


def camera_to_world(origin, target, up):
    forward = normalize_vecs_np(target - origin)

    right = -normalize_vecs_np(np.cross(up, forward))
    up = normalize_vecs_np(np.cross(right, forward))

    rotation = np.eye(4)
    rotation[:3, :3] = np.stack((right, up, forward), axis=-1)

    translation = translation_matrix(origin)

    transformation = (translation @ rotation)

    return transformation


# TODO: cleanup
def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)


def set_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def hash_vectors_cosine(n_buckets, vecs, rotations, n_hashes=8):
    batch_size = vecs.shape[0]
    device = vecs.device

    # Normalize for LSH
    norm_rotations = rotations / rotations.norm(dim=0, keepdim=True)
    norm_vecs = vecs / vecs.norm(dim=-1, keepdim=True)

    assert n_buckets % 2 == 0

    rot_size = n_buckets

    rotated_vecs = torch.mm(norm_vecs, norm_rotations)

    rotated_vecs = rotated_vecs.reshape(-1, n_hashes, rot_size // 2).permute(1, 0, 2)

    rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
    buckets = torch.argmax(rotated_vecs, dim=-1).type(torch.int32).permute(1, 0)

    return buckets


def project_vectors_gaussians(vecs, projection_vecs, cov=None, n_hashes=8):
    vecs = vecs.unsqueeze(0).repeat(n_hashes, 1, 1)

    projections = torch.einsum("ijk,ik->ji", vecs, projection_vecs)
    projections_range = torch.zeros_like(projections)

    # Compute range for gaussians
    if cov is not None:
        cov = cov.unsqueeze(0).repeat(n_hashes, 1, 1, 1)
        projection_vecs = projection_vecs.unsqueeze(1).unsqueeze(-1).repeat(1, vecs.shape[1], 1, 1)

        eigen_projections = torch.einsum("ijum,ijmk->ijuk", projection_vecs.permute(0, 1, 3, 2), cov)
        eigen_projections = torch.einsum("ijum,ijmu->ij", eigen_projections, projection_vecs)

        projections_range = (3.0 * torch.sqrt(torch.abs(eigen_projections))).permute(1, 0)

    return projections, projections_range
