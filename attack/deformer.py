import torch
from pytorch_msssim import MS_SSIM


class Deformer(object):
    def __init__(self, clean_cv2_image, num_split=3, fix_corner=True, src_rand_range=0.2, dst_rand_range=0.5, deform_guidance='patch'):
        self.clean_cv2_image = clean_cv2_image
        self.num_split = num_split
        self.fix_corner = fix_corner
        self.src_rand_range = src_rand_range
        self.dst_rand_range = dst_rand_range
        self.original_h, self.original_w = clean_cv2_image.shape[:2]

        self.img_h, self.img_w = 1024, 1024
        self.block_h = int(self.img_h / self.num_split)
        self.block_w = int(self.img_w / self.num_split)

        if fix_corner:
            source_centers = [[0,0], [self.img_h-1, self.img_w-1], [0, self.img_w-1], [self.img_h-1, 0]]
        else:
            source_centers = []
        for idx in range(self.num_split):
            for jdx in range(self.num_split):
                source_centers.append([int((idx+0.5)*self.block_h), int((jdx+0.5)*self.block_w)])
        self.source_centers = source_centers
        self.deform_guidance = deform_guidance
        channel_normalized_range = [256.0/58.395, 256.0/57.12, 256.0/57.375]
        self.ms_ssim_fn = MS_SSIM(data_range=channel_normalized_range, size_average=True, channel=3)

    def show_src_dest_points(self, ax, source_locs, dest_locs):
        scale_w = self.original_w / 1024
        scale_h = self.original_h / 1024
        source_locs[:, 0] = source_locs[:, 0] * scale_w
        source_locs[:, 1] = source_locs[:, 1] * scale_h
        dest_locs[:, 0] = dest_locs[:, 0] * scale_w
        dest_locs[:, 1] = dest_locs[:, 1] * scale_h
        ax.scatter(source_locs[:, 0], source_locs[:, 1], color='royalblue')
        ax.scatter(dest_locs[:, 0], dest_locs[:, 1], color='orangered')
        for i in range(source_locs.shape[0]):
            ax.plot([source_locs[i, 0], dest_locs[i, 0]], [source_locs[i, 1], dest_locs[i, 1]], color='orangered')


    def deform_loss(self, img_tensor, warp_tensor):
        if self.deform_guidance == 'patch':
            result = -1 * (img_tensor - warp_tensor).abs().mean()
        elif self.deform_guidance == 'ssim':
            result = self.ms_ssim_fn(img_tensor, warp_tensor)
        else:
            raise NotImplementedError(f"Add a new implementation of {self.deform_guidance} deformation guidance in attack/deformer.py")
        return result
    
    def control_loss(self, source_locs, dest_locs, translation_lambda=0.5):
        mov_vec = dest_locs - source_locs
        mov_vec_std_within = mov_vec.std(dim=1).mean()
        mov_vec_std_across = 0 if len(dest_locs) == 1 else mov_vec.std(dim=0).mean()
        return mov_vec_std_within - translation_lambda * mov_vec_std_across
    