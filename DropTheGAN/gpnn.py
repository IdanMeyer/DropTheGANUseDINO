import math
import numpy as np
from typing import List, Optional, Sequence, Tuple
from PIL import Image

import utils_local
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms

import fold
import resize_right
import tqdm
import os
from pathlib import Path
import faiss
import sys ; sys.path.append("../dino-vit-features")
import sys ; sys.path.append("dino-vit-features")
from extractor import ViTExtractor

_MAX_MEMORY_SIZE = 1 << 30
_INF = float('inf')

# Debugging function which shows the query, key, value and output images to the screen.
# Also supports passing the query and key clustering
def show_images(query, key, value, output, query_clustering=None, key_clustering=None):
    query_image = query.squeeze().permute(1, 2, 0)
    query_image = utils_local.to_numpy((255 * query_image).clamp(0, 255).to(dtype=torch.uint8))

    key_image = key.squeeze().permute(1, 2, 0)
    key_image = utils_local.to_numpy((255 * key_image).clamp(0, 255).to(dtype=torch.uint8))

    value_image = value.squeeze().permute(1, 2, 0)
    value_image = utils_local.to_numpy((255 * value_image).clamp(0, 255).to(dtype=torch.uint8))

    output_image = output.squeeze().permute(1, 2, 0)
    output_image = utils_local.to_numpy((255 * output_image).clamp(0, 255).to(dtype=torch.uint8))

    if query_clustering is not None:
        query_clustring_image = query_clustering.squeeze().permute(1, 2, 0)
        query_clustring_image = utils_local.to_numpy((255 * query_clustring_image).clamp(0, 255).to(dtype=torch.uint8))

    if key_clustering is not None:
        key_clustring_image = key_clustering.squeeze().permute(1, 2, 0)
        key_clustring_image = utils_local.to_numpy((255 * key_clustring_image).clamp(0, 255).to(dtype=torch.uint8))


    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)

    if query_clustering is not None:
        ax5 = fig.add_subplot(235)
    if key_clustering is not None:
        ax6 = fig.add_subplot(236)

    ax1.title.set_text('Query')
    ax1.imshow(query_image)
    ax2.title.set_text('Key')
    ax2.imshow(key_image)
    ax3.title.set_text('Value')
    ax3.imshow(value_image)
    ax4.title.set_text('Output')
    ax4.imshow(output_image)

    if query_clustering is not None:
        ax5.title.set_text('Query Clustering')
        ax5.imshow(query_clustring_image)
    if query_clustering is not None:
        ax6.title.set_text('Key Clustering')
        ax6.imshow(key_clustring_image)

    plt.show()


def gpnn(pyramid: Sequence[torch.Tensor],
         initial_guess: torch.Tensor,
         downscale_ratio: float = 0.75,
         patch_size: int = 7,
         alpha: float = _INF,
         output_pyramid_shape: Optional[Sequence[torch.Size]] = None,
         mask_pyramid: Optional[Sequence[torch.Tensor]] = None,
         num_iters_in_level: int = 10,
         num_iters_in_coarsest_level: int = 1,
         reduce: str = 'weighted_mean',
         should_use_our_code=True) -> torch.Tensor:
    if output_pyramid_shape is None:
        output_pyramid_shape = [image.shape for image in pyramid]
    if mask_pyramid is None:
        mask_pyramid = [None] * len(pyramid)
    generated = initial_guess
    coarsest_level = len(pyramid) - 1
    for level in range(coarsest_level, -1, -1):

        # Handle coarsest level
        if level == coarsest_level:
            for i in range(num_iters_in_coarsest_level):
                should_show_images = True
                should_run_dino = should_use_our_code
                original_weight = 10
                dino_weight = 90
                generated = pnn(generated,
                                key=pyramid[level],
                                value=pyramid[level],
                                mask=mask_pyramid[level],
                                # Modified patch size to be 3 for first iteration which has a high dino weight
                                patch_size=3,
                                alpha=alpha,
                                reduce=reduce,
                                should_show_images=should_show_images,
                                should_run_dino=should_run_dino,
                                original_weight=original_weight,
                                dino_weight=dino_weight)
        else:
            # At the last levels, do not iterate in order to save run-time (adding iterations over here has minimal actual impact on final result, but is heavy from run-time point of view on weak machines)
            if level < 3:
                num_iters_in_level = 1

            # Generate blurred
            blurred = resize_right.resize(pyramid[level + 1],
                                          1 / downscale_ratio,
                                          pyramid[level].shape)

            for i in range(num_iters_in_level):
                current_patch_size = patch_size
                should_show_images = i==0
                should_run_dino = should_use_our_code and i%2 == 0 and level > coarsest_level-3

                if should_use_our_code:
                    if i%2 == 1 and level == coarsest_level-1:
                        current_patch_size = 5

                original_weight = 70
                dino_weight = 30

                if level == coarsest_level-1:
                    original_weight = 30
                    dino_weight = 70
                elif level == coarsest_level-2:
                    original_weight = 60
                    dino_weight = 40

                # Call actual pnn module
                print(f"Level: {level}. coarsest_level: {coarsest_level}. iter: {i+1}/{num_iters_in_level}")
                generated = pnn(generated,
                                key=blurred,
                                value=pyramid[level],
                                mask=mask_pyramid[level],
                                patch_size=current_patch_size,
                                alpha=alpha,
                                reduce=reduce,
                                should_show_images=should_show_images,
                                should_run_dino=should_run_dino,
                                original_weight=original_weight,
                                dino_weight=dino_weight)

        # Resize generated output to match next level input
        if level > 0:
            generated = resize_right.resize(generated, 1 / downscale_ratio,
                                            output_pyramid_shape[level - 1])
    return generated

# Based on logic taken from dino-vit-features
def find_clustering(image_paths: List[str], elbow: float = 0.975, load_size: int = 224, layer: int = 11,
                        facet: str = 'key', bin: bool = False, thresh: float = 0.065, model_type: str = 'dino_vits8',
                        stride: int = 4, votes_percentage: int = 75, sample_interval: int = 100,
                        remove_outliers: bool = False, outliers_thresh: float = 0.7, low_res_saliency_maps: bool = True,
                        save_dir: str = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Use the VitExtractor defined in dino-vit-features
    extractor = ViTExtractor(model_type, stride, device=device)
    descriptors_list = []
    saliency_maps_list = []
    image_pil_list = []
    num_patches_list = []
    load_size_list = []
    if low_res_saliency_maps:
        saliency_extractor = ViTExtractor(model_type, stride=8, device=device)
    else:
        saliency_extractor = extractor
    if remove_outliers:
        cls_descriptors = []
    num_images = len(image_paths)

    # Create save_dir if needed
    if save_dir is not None:
        save_dir = Path(save_dir)

    # extract descriptors and saliency maps for each image
    for image_path in image_paths:
        image_batch, image_pil = extractor.preprocess(image_path, load_size)
        image_pil_list.append(image_pil)
        include_cls = remove_outliers  # removing outlier images requires the cls descriptor.
        descs = extractor.extract_descriptors(image_batch.to(device), layer, facet, bin, include_cls)
        descs = descs.cpu().detach().numpy()
        curr_num_patches, curr_load_size = extractor.num_patches, extractor.load_size
        num_patches_list.append(curr_num_patches)
        load_size_list.append(curr_load_size)
        if remove_outliers:
            cls_descriptor, descs = torch.from_numpy(descs[:, :, 0, :]), descs[:, :, 1:, :]
            cls_descriptors.append(cls_descriptor)
        descriptors_list.append(descs)
        if low_res_saliency_maps:
            if load_size is not None:
                low_res_load_size = (curr_load_size[0] // 2, curr_load_size[1] // 2)
            else:
                low_res_load_size = curr_load_size
            image_batch, _ = saliency_extractor.preprocess(image_path, low_res_load_size)

        saliency_map = saliency_extractor.extract_saliency_maps(image_batch.to(device)).cpu().detach().numpy()
        curr_sal_num_patches, curr_sal_load_size = saliency_extractor.num_patches, saliency_extractor.load_size
        if low_res_saliency_maps:
            reshape_op = transforms.Resize(curr_num_patches, transforms.InterpolationMode.NEAREST)
            saliency_map = np.array(reshape_op(Image.fromarray(saliency_map.reshape(curr_sal_num_patches)))).flatten()
        else:
            saliency_map = saliency_map[0]
        saliency_maps_list.append(saliency_map)

    # Remove outliers if needed
    if remove_outliers:
        all_cls_descriptors = torch.stack(cls_descriptors, dim=2)[0, 0]
        mean_cls_descriptor = torch.mean(all_cls_descriptors, dim=0)[None, ...]
        cos_sim = torch.nn.CosineSimilarity(dim=1)
        similarities_to_mean = cos_sim(all_cls_descriptors, mean_cls_descriptor)
        inliers_idx = torch.where(similarities_to_mean >= outliers_thresh)[0]
        inlier_descriptors, outlier_descriptors = [], []
        inlier_image_paths, outlier_image_paths = [], []
        inlier_image_pil, outlier_image_pil = [], []
        inlier_num_patches, outlier_num_patches = [], []
        inlier_saliency_maps, outlier_saliency_maps = [], []
        inlier_load_size, outlier_load_size = [], []
        for idx, (image_path, descriptor, saliency_map, pil_image, num_patches, load_size) in enumerate(zip(image_paths,
                descriptors_list, saliency_maps_list, image_pil_list, num_patches_list, load_size_list)):
            (inlier_image_paths if idx in inliers_idx else outlier_image_paths).append(image_path)
            (inlier_descriptors if idx in inliers_idx else outlier_descriptors).append(descriptor)
            (inlier_saliency_maps if idx in inliers_idx else outlier_saliency_maps).append(saliency_map)
            (inlier_image_pil if idx in inliers_idx else outlier_image_pil).append(pil_image)
            (inlier_num_patches if idx in inliers_idx else outlier_num_patches).append(num_patches)
            (inlier_load_size if idx in inliers_idx else outlier_load_size).append(load_size)
        image_paths = inlier_image_paths
        descriptors_list = inlier_descriptors
        saliency_maps_list = inlier_saliency_maps
        image_pil_list = inlier_image_pil
        num_patches_list = inlier_num_patches
        load_size_list = inlier_load_size
        num_images = len(inliers_idx)

    # Cluster all images using k-means:
    all_descriptors = np.ascontiguousarray(np.concatenate(descriptors_list, axis=2)[0, 0])
    normalized_all_descriptors = all_descriptors.astype(np.float32)
    # normalized_all_descriptors in place!!!
    faiss.normalize_L2(normalized_all_descriptors)

    sampled_descriptors_list = [x[:, :, ::sample_interval, :] for x in descriptors_list]

    all_sampled_descriptors = np.ascontiguousarray(np.concatenate(sampled_descriptors_list, axis=2)[0, 0])
    normalized_all_sampled_descriptors = all_sampled_descriptors.astype(np.float32)

    # normalized_all_sampled_descriptors in place!!!
    faiss.normalize_L2(normalized_all_sampled_descriptors)

    sum_of_squared_dists = []
    n_cluster_range = list(range(1, 15))
    for n_clusters in n_cluster_range:
        algorithm = faiss.Kmeans(d=normalized_all_sampled_descriptors.shape[1], k=n_clusters, niter=300, nredo=10)
        algorithm.train(normalized_all_sampled_descriptors.astype(np.float32))
        squared_distances, labels = algorithm.index.search(normalized_all_descriptors.astype(np.float32), 1)
        objective = squared_distances.sum()
        sum_of_squared_dists.append(objective / normalized_all_descriptors.shape[0])
        if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow * sum_of_squared_dists[-2]):
            break

    num_labels = np.max(n_clusters) + 1
    num_descriptors_per_image = [num_patches[0]*num_patches[1] for num_patches in num_patches_list]
    labels_per_image = np.split(labels, np.cumsum(num_descriptors_per_image))

    output_images = []
    # Use color map of tab20 in order to have enough discrete colors
    cmap = 'tab20'
    print(f"Number of labels:{num_labels}")

    # Iterate over images, and save the clustering to file.
    for image_path, num_patches, label_per_image in zip(image_paths, num_patches_list, labels_per_image):
        output_path = os.path.join(Path(image_path).parent, f"{Path(image_path).stem}_clustering.png")
        output_images.append(output_path)
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(label_per_image.reshape(num_patches), vmin=0, vmax=num_labels-1, cmap=cmap)
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # Return paths of output images, as needed by the calling function
    return output_images

def dino_vit_get_clustering(image1, image2, load_size=100, sample_interval=10, stride=2, elbow=0.975):
    # Save image1 and image2 to local disk
    image_path_a = "/tmp/image1.png"
    image_path_b = "/tmp/image2.png"
    utils_local.imwrite(image_path_a, image1.squeeze())
    utils_local.imwrite(image_path_b, image2.squeeze())

    # Find the clustering
    clustering_paths = find_clustering([image_path_a, image_path_b], load_size=load_size,
                                               sample_interval=sample_interval,
                                               stride=stride, elbow=elbow)
    # Get path of each clustering
    image1_clustering = (utils_local.imread(clustering_paths[0])[0:3])
    image2_clustering = (utils_local.imread(clustering_paths[1])[0:3])

    # Resize clustering to original image size
    T = transforms.Resize(size=(tuple(image1.shape[2:])))
    image1_clustering = T(image1_clustering)
    image2_clustering = T(image2_clustering)

    # Run garbage collector
    import gc;gc.collect()
    return image1_clustering.unsqueeze(0), image2_clustering.unsqueeze(0)


def pnn(query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patch_size: int = 7,
        alpha: float = _INF,
        reduce: str = 'weighted_mean',
        should_show_images=False,
        should_run_dino=False,
        original_weight=50,
        dino_weight=50) -> torch.Tensor:

    # Run garbage collector, clears up non-needed memory from GPU
    import gc;gc.collect()

    query = query.unsqueeze(0)
    key = key.unsqueeze(0)
    value = value.unsqueeze(0)
    if mask is not None:
        mask = mask.unsqueeze(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)

    query_patches = fold.unfold2d(query, patch_size)
    query_patches_column, query_patches_size = fold.view_as_column(
        query_patches)
    key_patches_column, _ = fold.view_as_column(fold.unfold2d(key, patch_size))
    value_patches_column, _ = fold.view_as_column(
        fold.unfold2d(value, patch_size))

    # Handle masks, not relevant for structural analogy
    if mask is not None:
        mask = (mask > 0.5).to(query)
        mask_patches_column, _ = fold.view_as_column(
            fold.unfold2d(mask, patch_size))
        valid_patches_mask = mask_patches_column.sum(
            dim=2) > mask_patches_column.shape[2] - 0.5
        key_patches_column = key_patches_column.squeeze(0)[
            valid_patches_mask.squeeze(0)].unsqueeze(0)
        value_patches_column = value_patches_column.squeeze(0)[
            valid_patches_mask.squeeze(0)].unsqueeze(0)


    query_clustering, key_clustering = None, None
    query_clustering_patches_column = None
    key_clustering_patches_column = None

    if should_run_dino:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Get query clustering and key clustering from query and key via dino-vit
        query_clustering, key_clustering = dino_vit_get_clustering(query, key)
        query_clustering_patches = fold.unfold2d(query_clustering, patch_size)
        query_clustering_patches_column, _ = fold.view_as_column(
            query_clustering_patches)
        key_clustering_patches_column, _ = fold.view_as_column(fold.unfold2d(key_clustering, patch_size))
        query_clustering_patches_column = query_clustering_patches_column.to(device)
        key_clustering_patches_column = key_clustering_patches_column.to(device)

    # Commented out line below and replaced it with an internal function that supports DINO
    # _, indices = find_normalized_nearest_neighbors(query_patches_column,
    #                                                key_patches_column, alpha)
    _, indices = find_weighted_nearest_neighbors(query_patches_column, key_patches_column,
                                                 queries_clustering=query_clustering_patches_column,
                                                 keys_clustering=key_clustering_patches_column,
                                                 original_weight=original_weight,
                                                 dino_weight=dino_weight,
                                                 )

    # Generate output from value patchs and indices
    out_patches_column = F.embedding(indices.squeeze(2),
                                     value_patches_column.squeeze(0))
    out_patches = fold.view_as_image(out_patches_column, query_patches_size)
    output = fold.fold2d(out_patches, reduce=reduce)

    # Show images via plt.show if needed
    if should_show_images or should_run_dino:
        show_images(query, key, value, output, query_clustering, key_clustering)

    return output.squeeze(0)


def make_pyramid(image: torch.Tensor, num_levels: int,
                 downscale_ratio: float) -> List[torch.Tensor]:
    scale_factor = (1, ) * (image.ndim - 2) + (downscale_ratio,
                                               downscale_ratio)
    pyramid = [image]
    for level in range(1, num_levels + 1):
        output_shape = (*image.shape[:-2],
                        math.ceil(image.shape[-2] * downscale_ratio**level),
                        math.ceil(image.shape[-1] * downscale_ratio**level))
        pyramid.append(
            resize_right.resize(pyramid[-1], scale_factor, output_shape))
    return pyramid


def _find_tile_size(height: int, width: int, cell_size: int,
                    max_tile_size: int) -> Tuple[int, int]:
    best_tile_height = 1
    best_tile_width = 1
    best_tile_size = cell_size
    for tile_height in range(2, height + 1):
        if tile_height * cell_size > max_tile_size:
            break
        tile_width = min(width, max_tile_size // (tile_height * cell_size))
        tile_size = tile_height * tile_width * cell_size
        if tile_size == max_tile_size:
            return tile_height, tile_width
        elif tile_size < max_tile_size and tile_size > best_tile_size:
            best_tile_height, best_tile_width = tile_height, tile_width
            best_tile_size = tile_size
    return best_tile_height, best_tile_width


def _compute_dist_matrix(queries: torch.Tensor,
                         keys: torch.Tensor) -> torch.Tensor:
    """Computes a matrix of MSE between each query and each key."""
    # x2 = torch.einsum('bid,bid->bi',queries, queries).unsqueeze(2)
    # y2 = torch.einsum('bjd,bjd->bj', keys, keys).unsqueeze(1)
    # xy = torch.einsum('bid,bjd->bij', queries, keys)
    # return (x2 + y2 - 2 * xy) / queries.shape[-1]
    return torch.cdist(queries, keys, p=2).pow(2) / queries.shape[-1]


def _find_tile_height(height: int, width: int, cell_size: int,
                      max_tile_size: int) -> Tuple[int, int]:
    row_size = width * cell_size
    return min(height, (max_tile_size + row_size - 1) // row_size)


def _slice_weights(weights: Optional[torch.Tensor], start: int,
                   stop: int) -> Optional[torch.Tensor]:
    if weights is None:
        return None
    if weights.shape[1] == 1:
        return weights
    return weights[:, start:stop, :]


def _find_weighted_nearest_neighbors(
    queries: torch.Tensor,
    keys: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    queries_clustering=None,
    keys_clustering=None,
    original_weight=50,
    dino_weight=50,
) -> Tuple[torch.Tensor, torch.Tensor]:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Calculate original (MSE) distance
    original_dists = _compute_dist_matrix(queries, keys)
    original_dists = original_dists.to(device)

    if queries_clustering is not None and keys_clustering is not None:
        print("Using dino matrix calculation")
        dino_dists = []
        # Below is a slow code which works serially. Left it here because it is much easier to debug.
        # for x in [x for x in queries_clustering[0]]:
        #     for y in [x for x in keys_clustering[0]]:
        #         x_y_diff = x - y
        #         # Every pixels which are different (not same segment), mark the distance as a constant value
        #         x_y_diff[x!=y] = 1
        #         dino_dists.append(x_y_diff.norm())
        # Conver distances to tensor
        # queries_clustering.shape[1] is the number of clusters
        # dino_dists = torch.tensor(dino_dists).reshape(queries_clustering.shape[1],
                                                      # queries_clustering.shape[1]).unsqueeze(0)
        # Normalize values according to patch size
        # dino_dists /= queries_clustering.shape[2]
        # Move dino_dists to device
        # dino_dists = dino_dists.to(device)
        # dino_dists = _compute_dist_matrix(queries_clustering, keys_clustering)

        # Calculate DINO-dists (semantic distance)
        A_expanded = queries_clustering.squeeze(0).unsqueeze(1)
        B_expanded = keys_clustering.squeeze(0).unsqueeze(0)
        C = A_expanded-B_expanded
        C[A_expanded != B_expanded] = 1
        dino_dists = C.norm(dim=2).unsqueeze(0)/queries_clustering.shape[2]

        print(f"original_weight={original_weight}, dino_weight={dino_weight}")

        # Calculate weighted distance according to the provided weights for dino dists and original (mse) dists
        dists = (original_weight*original_dists + dino_weight*dino_dists)/(original_weight+dino_weight)
    else:
        dists = original_dists

    if weights is not None:
        dists *= weights
    # Find the minimum distance according to axis=2 (nearest patch)
    return dists.min(dim=2, keepdim=True)


def find_weighted_nearest_neighbors(
    queries: torch.Tensor,
    keys: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    max_memory_usage: int = _MAX_MEMORY_SIZE,
    queries_clustering=None,
    keys_clustering=None,
    original_weight=50,
    dino_weight=50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    size_in_bytes = torch.finfo(queries.dtype).bits // 8
    batch_size = queries.shape[0]
    num_queries = queries.shape[1]
    num_keys = keys.shape[1]
    if weights is not None:
        max_memory_usage //= 2

    # Split to tiles if needed
    tile_height = _find_tile_height(num_queries, num_keys,
                                    batch_size * size_in_bytes,
                                    max_memory_usage)
    values = []
    indices = []
    # Iterate over tiles and call _find_weighted_nearest_neighbors for each one
    for start in range(0, num_queries, tile_height):
        if queries_clustering is not None:
            value, idx = _find_weighted_nearest_neighbors(
                queries[:, start:start + tile_height], keys,
                _slice_weights(weights, start, start + tile_height),
                queries_clustering[:, start:start + tile_height], keys_clustering,
                original_weight=original_weight,
                dino_weight=dino_weight,
            )
        else:
            value, idx = _find_weighted_nearest_neighbors(
                queries[:, start:start + tile_height], keys,
                _slice_weights(weights, start, start + tile_height),
                original_weight=original_weight,
                dino_weight=dino_weight,
            )
        values.append(value)
        indices.append(idx)
    return torch.cat(values, dim=1), torch.cat(indices, dim=1)

    # values = queries.new_zeros(size=(batch_size, num_queries, 1))
    # indices = torch.zeros(size=(batch_size, num_queries, 1), dtype=torch.int64, device=queries.device)
    # for start in range(0, num_queries, tile_height):
    #     value, idx = _find_weighted_nearest_neighbors(
    #         queries[:, start:start + tile_height], keys,
    #         _slice_weights(weights, start, start + tile_height))
    #     values[:, start:start + tile_height] = value
    #     indices[:, start:start + tile_height] = idx
    # return values, indices


def find_normalized_nearest_neighbors(
    queries: torch.Tensor,
    keys: torch.Tensor,
    alpha: float = _INF,
    max_memory_usage: int = _MAX_MEMORY_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha < _INF:
        # compute min distance where queries<-keys, and keys<-queries
        normalizer, _ = find_weighted_nearest_neighbors(
            keys, queries, None, max_memory_usage)
        normalizer += alpha
        normalizer = 1 / normalizer
        normalizer = normalizer.transpose(1, 2)  # "keys" <-> "queries"
    else:
        normalizer = None
    return find_weighted_nearest_neighbors(queries, keys, normalizer,
                                           max_memory_usage)
