# Copyright 2019 RnD at Spoon Radio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch

def generate_max_filter(img_tensor, 
                        source_locs,
                        dest_locs,
                        filter_stride=2):
    batch_size, image_height, image_width, image_channel = img_tensor.shape
    max_filter = torch.ones_like(img_tensor[0]) #[1024, 1024, 3]
    rows_extract = [i for i in range(image_height) if (i // filter_stride) % 2 == 0]
    cols_extract = [i for i in range(image_height) if (i // filter_stride) % 2 == 1]

    max_filter[rows_extract, :, :] = 0
    max_filter[:, cols_extract, :] = 1 - max_filter[:, cols_extract, :]
    max_filter = max_filter.unsqueeze(dim=0) # [1, 1024, 1024, 3]
    
    max_filter, _ = sparse_image_warp(max_filter, source_locs[0:1], dest_locs[0:1],
                                      interpolation_order=1)

    return max_filter

def avg_batch_sparse_image_warp_by_filter(img_tensor, 
                      batch_source_control_point_locations,
                      batch_dest_control_point_locations,
                      interpolation_order=2,
                      regularization_weight=0.0,
                      num_boundaries_points=0,
                      max_filter=None):
    assert len(batch_source_control_point_locations) == len(batch_dest_control_point_locations)
    batch_size, image_height, image_width, image_channel = img_tensor.shape
    warp_size = len(batch_source_control_point_locations)
    grid_locations = get_grid_locations(image_height, image_width)
    flattened_grid_locations = torch.tensor(flatten_grid_locations(grid_locations, image_height, image_width), device = img_tensor.device)
    assert batch_size == 1
    batch_control_point_flows = (batch_dest_control_point_locations - batch_source_control_point_locations)
    warped_image_batch = []
    dense_flows_batch = []

    for warp_id in range(warp_size):
        dest_control_point_locations = batch_dest_control_point_locations[warp_id: warp_id+1]
        control_point_flows = batch_control_point_flows[warp_id: warp_id+1]
        flattened_flows = interpolate_spline(
            dest_control_point_locations,
            control_point_flows,
            flattened_grid_locations,
            interpolation_order,
            regularization_weight)
        dense_flows = create_dense_flows(flattened_flows, batch_size, image_height, image_width)
        #warped_image, mask = dense_image_warp(img_tensor, dense_flows)
        warped_image = dense_image_warp(img_tensor, dense_flows)
        warped_image_batch.append(warped_image)
        dense_flows_batch.append(dense_flows)

    warped_image_max, _ = torch.stack(warped_image_batch, dim=0).max(dim=0)
    warped_image_min, _ = torch.stack(warped_image_batch, dim=0).min(dim=0)
    warped_image = warped_image_max * max_filter + warped_image_min * (1 - max_filter)
    dense_flows = None
    return warped_image, dense_flows

def avg_batch_sparse_image_warp(img_tensor, 
                      batch_source_control_point_locations,
                      batch_dest_control_point_locations,
                      interpolation_order=2,
                      regularization_weight=0.0,
                      num_boundaries_points=0,
                      filter_stride=2):
    assert len(batch_source_control_point_locations) == len(batch_dest_control_point_locations)
    batch_size, image_height, image_width, image_channel = img_tensor.shape
    warp_size = len(batch_source_control_point_locations)
    grid_locations = get_grid_locations(image_height, image_width)
    flattened_grid_locations = torch.tensor(flatten_grid_locations(grid_locations, image_height, image_width), device = img_tensor.device)
    assert batch_size == 1
    batch_control_point_flows = (batch_dest_control_point_locations - batch_source_control_point_locations)
    warped_image_batch = []
    dense_flows_batch = []

    for warp_id in range(warp_size):
        dest_control_point_locations = batch_dest_control_point_locations[warp_id: warp_id+1]
        control_point_flows = batch_control_point_flows[warp_id: warp_id+1]
        flattened_flows = interpolate_spline(
            dest_control_point_locations,
            control_point_flows,
            flattened_grid_locations,
            interpolation_order,
            regularization_weight)
        dense_flows = create_dense_flows(flattened_flows, batch_size, image_height, image_width)
        #warped_image, mask = dense_image_warp(img_tensor, dense_flows)
        warped_image = dense_image_warp(img_tensor, dense_flows)
        warped_image_batch.append(warped_image)
        dense_flows_batch.append(dense_flows)

    #warped_image = torch.stack(warped_image_batch, dim=0).mean(dim=0)
    #dense_flows = torch.stack(dense_flows_batch, dim=0).mean(dim=0)
    
    warped_image_max, _ = torch.stack(warped_image_batch, dim=0).max(dim=0)
    #dense_flows_max, _ = torch.stack(dense_flows_batch, dim=0).max(dim=0)
    warped_image_min, _ = torch.stack(warped_image_batch, dim=0).min(dim=0)
    #dense_flows_min, _ = torch.stack(dense_flows_batch, dim=0).min(dim=0)

    max_filter = torch.ones_like(warped_image_batch[0][0]) #[1024, 1024, 3]
    rows_extract = [i for i in range(image_height) if (i // filter_stride) % 2 == 0]
    cols_extract = [i for i in range(image_height) if (i // filter_stride) % 2 == 1]

    max_filter[rows_extract, :, :] = 0
    max_filter[:, cols_extract, :] = 1 - max_filter[:, cols_extract, :]
    max_filter = max_filter.unsqueeze(dim=0)
    warped_image = warped_image_max * max_filter + warped_image_min * (1 - max_filter)
    dense_flows = None

    return warped_image, dense_flows

def sparse_image_warp(img_tensor,
                      source_control_point_locations,
                      dest_control_point_locations,
                      interpolation_order=2,
                      regularization_weight=0.0,
                      num_boundaries_points=0):
    control_point_flows = (dest_control_point_locations - source_control_point_locations)

    batch_size, image_height, image_width, image_channel = img_tensor.shape
    grid_locations = get_grid_locations(image_height, image_width)
    flattened_grid_locations = torch.tensor(flatten_grid_locations(grid_locations, image_height, image_width), device = img_tensor.device)
    #print(dest_control_point_locations.detach().numpy().shape)
    flattened_flows = interpolate_spline(
        dest_control_point_locations,
        control_point_flows,
        flattened_grid_locations,
        interpolation_order,
        regularization_weight)
    dense_flows = create_dense_flows(flattened_flows, batch_size, image_height, image_width)
    #warped_image, mask = dense_image_warp(img_tensor, dense_flows)
    warped_image = dense_image_warp(img_tensor, dense_flows)

    #return warped_image, dense_flows, mask
    return warped_image, dense_flows

def get_grid_locations(image_height, image_width):
    """Wrapper for np.meshgrid."""

    y_range = np.linspace(0, image_height - 1, image_height)
    x_range = np.linspace(0, image_width - 1, image_width)
    y_grid, x_grid = np.meshgrid(y_range, x_range, indexing='ij')
    return np.stack((y_grid, x_grid), -1)

def flatten_grid_locations(grid_locations, image_height, image_width):
    return np.reshape(grid_locations, [image_height * image_width, 2])

def create_dense_flows(flattened_flows, batch_size, image_height, image_width):
    # possibly .view
    #return torch.reshape(flattened_flows, [batch_size, image_height, image_width, 2])
    return torch.reshape(flattened_flows, [1, image_height, image_width, 2])

def interpolate_spline(train_points, train_values, query_points, order, regularization_weight=0.0, ):
    # First, fit the spline to the observed data.
    
    w, v = solve_interpolation(train_points, train_values, order, regularization_weight)
    # Then, evaluate the spline at the query locations.


    query_values = apply_interpolation(query_points, train_points, w, v, order)
    return query_values

def solve_interpolation(train_points, train_values, order, regularization_weight):
    device = train_points.device
    #print(train_points.shape)
    b, n, d = train_points.shape
    k = train_values.shape[-1]

    # First, rename variables so that the notation (c, f, w, v, A, B, etc.)
    # follows https://en.wikipedia.org/wiki/Polyharmonic_spline.
    # To account for python style guidelines we use
    # matrix_a for A and matrix_b for B.

    c = train_points
    f = train_values.float()

    matrix_a = phi(cross_squared_distance_matrix(c, c), order)  # [b, n, n]

    # Append ones to the feature values for the bias term in the linear model.
    ones = torch.ones([b, n, 1], dtype=train_points.dtype, device = device)#.view([-1, 1, 1])
    matrix_b = torch.cat((c, ones), 2).float()  # [b, n, d + 1]
    # [b, n + d + 1, n]

    left_block = torch.cat((matrix_a, torch.transpose(matrix_b, 2, 1)), 1)

    num_b_cols = matrix_b.shape[2]  # d + 1

    # In Tensorflow, zeros are used here. Pytorch gesv fails with zeros for some reason we don't understand.
    # So instead we use very tiny randn values (variance of one, zero mean) on one side of our multiplication.
    lhs_zeros = torch.randn((b, num_b_cols, num_b_cols), device = device) / 1e10
    #lhs_zeros = torch.zeros((b, num_b_cols, num_b_cols), device = device)
    right_block = torch.cat((matrix_b, lhs_zeros),
                            1)  # [b, n + d + 1, d + 1]
    lhs = torch.cat((left_block, right_block),
                    2)  # [b, n + d + 1, n + d + 1]
    rhs_zeros = torch.zeros((b, d + 1, k), dtype=train_points.dtype, device = device).float()
    rhs = torch.cat((f, rhs_zeros), 1)  # [b, n + d + 1, k]

    # Then, solve the linear system and unpack the results.
    lhs[lhs<0] = 0
    lhs = lhs.float()
    X = torch.linalg.solve(lhs, rhs)
    #X, LU = torch.solve(rhs, lhs)
    #X, LU = torch.gesv(rhs, lhs)#pytorch1.0
    #X = np.linalg.solve(lhs.cpu().numpy(), rhs.cpu().numpy())
    #X = torch.from_numpy(X).cuda()
    w = X[:, :n, :]
    v = X[:, n:, :]
    return w, v

def cross_squared_distance_matrix(x, y):
    """Pairwise squared distance between two (batch) matrices' rows (2nd dim).
        Computes the pairwise distances between rows of x and rows of y
        Args:
        x: [batch_size, n, d] float `Tensor`
        y: [batch_size, m, d] float `Tensor`
        Returns:
        squared_dists: [batch_size, n, m] float `Tensor`, where
        squared_dists[b,i,j] = ||x[b,i,:] - y[b,j,:]||^2
    """
    x_norm_squared = torch.sum(torch.mul(x, x), dim=2).unsqueeze(2)
    y_norm_squared = torch.sum(torch.mul(y, y), dim=2).unsqueeze(1)

    x_y_transpose = torch.matmul(x.squeeze(0), y.squeeze(0).transpose(0, 1))
    # squared_dists[b,i,j] = ||x_bi - y_bj||^2 = x_bi'x_bi- 2x_bi'x_bj + x_bj'x_bj
    squared_dists = x_norm_squared - 2 * x_y_transpose + y_norm_squared
    return squared_dists.float()

def phi(r, order):
    """Coordinate-wise nonlinearity used to define the order of the interpolation.
    See https://en.wikipedia.org/wiki/Polyharmonic_spline for the definition.
    Args:
    r: input op
    order: interpolation order
    Returns:
    phi_k evaluated coordinate-wise on r, for k = r
    """
    EPSILON = torch.tensor(1e-10, device=r.device)
    # using EPSILON prevents log(0), sqrt0), etc.
    # sqrt(0) is well-defined, but its gradient is not
    if order == 1:
        r = torch.max(r, EPSILON)
        r = torch.sqrt(r)
        return r
    elif order == 2:
        return 0.5 * r * torch.log(torch.max(r, EPSILON))
    elif order == 4:
        return 0.5 * torch.square(r) * torch.log(torch.max(r, EPSILON))
    elif order % 2 == 0:
        r = torch.max(r, EPSILON)
        return 0.5 * torch.pow(r, 0.5 * order) * torch.log(r)
    else:
        r = torch.max(r, EPSILON)
        return torch.pow(r, 0.5 * order)


def apply_interpolation(query_points, train_points, w, v, order):
    """Apply polyharmonic interpolation model to data.
    Given coefficients w and v for the interpolation model, we evaluate
    interpolated function values at query_points.
    Args:
    query_points: `[b, m, d]` x values to evaluate the interpolation at
    train_points: `[b, n, d]` x values that act as the interpolation centers
                    ( the c variables in the wikipedia article)
    w: `[b, n, k]` weights on each interpolation center
    v: `[b, d, k]` weights on each input dimension
    order: order of the interpolation
    Returns:
    Polyharmonic interpolation evaluated at points defined in query_points.
    """
    query_points = query_points.unsqueeze(0)
    # First, compute the contribution from the rbf term.
    pairwise_dists = cross_squared_distance_matrix(query_points.float(), train_points.float())
    phi_pairwise_dists = phi(pairwise_dists, order)
    rbf_term = torch.matmul(phi_pairwise_dists, w)

    # Then, compute the contribution from the linear term.
    # Pad query_points with ones, for the bias term in the linear model.
    ones = torch.ones_like(query_points[..., :1])
    query_points_pad = torch.cat((
        query_points,
        ones
    ), 2).float()
    linear_term = torch.matmul(query_points_pad, v)

    return rbf_term + linear_term


def dense_image_warp(image, flow):
    """Image warping using per-pixel flow vectors.
    Apply a non-linear warp to the image, where the warp is specified by a dense
    flow field of offset vectors that define the correspondences of pixel values
    in the output image back to locations in the  source image. Specifically, the
    pixel value at output[b, j, i, c] is
    images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].
    The locations specified by this formula do not necessarily map to an int
    index. Therefore, the pixel value is obtained by bilinear
    interpolation of the 4 nearest pixels around
    (b, j - flow[b, j, i, 0], i - flow[b, j, i, 1]). For locations outside
    of the image, we use the nearest pixel values at the image boundary.
    Args:
    image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
    flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
    name: A name for the operation (optional).
    Note that image and flow can be of type tf.half, tf.float32, or tf.float64,
    and do not necessarily have to be the same type.
    Returns:
    A 4-D float `Tensor` with shape`[batch, height, width, channels]`
    and same type as input image.
    Raises:
    ValueError: if height < 2 or width < 2 or the inputs have the wrong number
    of dimensions.
    """
    #image = image.unsqueeze(3)  # add a single channel dimension to image tensor
    batch_size, height, width, channels = image.shape
    device = image.device

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y = torch.meshgrid(
        torch.arange(width, device = device), torch.arange(height, device = device))

    #stacked_grid = torch.stack((grid_y, grid_x), dim=2).float()
    stacked_grid = torch.stack((grid_x, grid_y), dim=2).float()

    batched_grid = stacked_grid.unsqueeze(-1).permute(3, 1, 0, 2)

    query_points_on_grid = batched_grid - flow
    if batch_size != 1:
        query_points_on_grid_list = []
        for _ in range(batch_size):
            query_points_on_grid_list.append(query_points_on_grid)
        query_points_on_grid = torch.cat(query_points_on_grid_list, dim=0)

    query_points_flattened = torch.reshape(query_points_on_grid,
                                           [batch_size, height * width, 2])

    image = image.permute([0, 3, 1 ,2])
    # may wrong h w
    #a = torch.Tensor(np.array([width - 1, height - 1]), device = image.device)
    
    query_points_on_grid = query_points_on_grid / (width-1) *2 -1
    interpolated = torch.nn.functional.grid_sample(image, query_points_on_grid, padding_mode='reflection')
    interpolated = interpolated.permute([0,2,3,1])

    #mask = torch.autograd.Variable(torch.ones(image.size()))
    #mask = torch.nn.functional.grid_sample(mask, query_points_on_grid)
    #mask[mask < 0.9999] = 0
    #mask[mask > 0] = 1

    #return interpolated, mask
    return interpolated
