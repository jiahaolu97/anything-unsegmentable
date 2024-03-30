import torch
import torchvision
import numpy as np
import cv2
import copy
from tqdm import tqdm, trange
from pytorch_msssim import MS_SSIM

from segment_anything import SamPredictor
from segment_anything.modeling import Sam
from config import *
from .deformer import Deformer
from .tps import (generate_max_filter, sparse_image_warp, avg_batch_sparse_image_warp_by_filter)

class SamAttacker(SamPredictor):
    def __init__(self, sam_model: Sam, debug_mode=False):
        super().__init__(sam_model)
        self.adv_epsilon = 8 / 255.0
        self.adv_alpha = 2 / 255.0
        self.adv_iters = 40
        self.select_feature_layers = [4, 5, 6, 7]

        if debug_mode:
            self.adv_epsilon = 64 / 255.0
            self.adv_alpha = 16 / 255.0
            self.adv_iters = 4

    def transform_coord(self, points, boxes):
        # points: numpy array of shape (N, 2)
        # boxes: numpy array of shape (M, 4)
        coords_torch, labels_torch, box_torch = None, None, None
        if points is not None:
            labels = np.array([1] * len(points))
            points_coords = self.transform.apply_coords(points, self.original_size)
            coords_torch = torch.as_tensor(points_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if boxes is not None:
            boxes = self.transform.apply_boxes(boxes, self.original_size)
            box_torch = torch.as_tensor(boxes, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        points_torch = None if coords_torch is None else (coords_torch, labels_torch)
        return points_torch, box_torch
    
    def sample_points(self, mask=None, sample_size=10):
        # mask is a two dimensional numpy array (binary mask over original image size)
        if mask is None: # sample in the whole image
            mask = np.ones(self.original_size, dtype=np.uint8)
        inmask_pixel_positions = np.flip(np.argwhere(mask == True), axis=1)
        sample_size = min(sample_size, inmask_pixel_positions.shape[0])
        sampled_pixel_id = np.random.choice(inmask_pixel_positions.shape[0], sample_size, replace=False)
        sampled_pixel_pos = inmask_pixel_positions[sampled_pixel_id]
        return sampled_pixel_pos
        
    def set_clean_cv2_image(self, clean_cv2_image):
        if isinstance(clean_cv2_image, np.ndarray):
            self.clean_cv2_image = clean_cv2_image
            input_image_torch = self.set_image(clean_cv2_image)
        elif isinstance(clean_cv2_image, str):
            clean_cv2_image = cv2.imread(clean_cv2_image)
            clean_cv2_image = cv2.cvtColor(clean_cv2_image, cv2.COLOR_BGR2RGB)
            input_image_torch = self.set_image(clean_cv2_image)
        return input_image_torch

    def get_cv2_from_torch(self, img_torchtensor):
        h, w = self.input_size
        torch_img = self.model.de_preprocess(img_torchtensor, h, w)
        numpy_img = torch_img.cpu()[0].numpy().astype(np.uint8)
        cv2_img = np.transpose(numpy_img, (1, 2, 0))
        cv2_img = cv2.resize(cv2_img, (self.original_size[1], self.original_size[0]))
        return cv2_img

    def attack_fn(self, attackname):
        if attackname == 'attack_sam':
            return self.attack_sam
        elif attackname == 'attack_sam_k':
            return self.attack_sam_K
        elif attackname == 'tap':
            return self.TAP
        elif attackname == 'aa':
            return self.AA
        elif attackname == 'ilpd':
            return self.ILPD
        elif attackname == 'pata':
            return self.PATA
        elif attackname == 'pata++':
            return self.PATA_plusplus
        elif attackname == 'uad':
            return self.UAD
        else:
            raise ValueError(f"Unknown attackname {attackname}, go define it in attack/attacker.py")

    # "Attack-sam: Towards evaluating adversarial robustness of segment anything model" (Arxiv 23.05)
    def attack_sam(self, clean_cv2_image, points=None, boxes=None):
        # points: numpy array of shape (N, 2)
        # boxes: numpy array of shape (M, 4)
        assert points is not None or boxes is not None
        input_image = self.set_clean_cv2_image(clean_cv2_image)
        original_clean_image = input_image.data
        points_torch, box_torch = self.transform_coord(points, boxes)
        with trange(self.adv_iters, desc='Attack-SAM') as pbar:
            for adv_iter in pbar:
                input_image.requires_grad = True
                image_embeddings = self.model.image_encoder(input_image)
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=points_torch, boxes=box_torch, masks=None)
                low_res_masks, _ = self.model.mask_decoder(
                    image_embeddings = image_embeddings,
                    image_pe = self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings = sparse_embeddings,
                    dense_prompt_embeddings = dense_embeddings,
                    multimask_output = False
                )
                target_low_res_masks = torch.clamp(low_res_masks.detach(), max=-10)
                loss = torch.nn.MSELoss()(low_res_masks, target_low_res_masks)
                loss.backward()
                pbar.set_postfix({'loss': loss.item()})

                perturbation = self.adv_alpha * input_image.grad.data.sign() * 255 / self.model.pixel_std
                adv_image_unclipped = input_image.data - perturbation
                clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image, 
                                                    min=-self.adv_epsilon*255/self.model.pixel_std, 
                                                    max=self.adv_epsilon*255/self.model.pixel_std)
                input_image = torch.clamp(original_clean_image + clipped_perturbation,
                                            min=-self.model.pixel_mean / self.model.pixel_std,
                                            max=(255 - self.model.pixel_mean) / self.model.pixel_std).detach()
        return input_image
            
    # "Attack-sam: Towards evaluating adversarial robustness of segment anything model" (Arxiv 23.05)
    def attack_sam_K(self, clean_cv2_image, K=400):
        input_image = self.set_clean_cv2_image(clean_cv2_image)
        original_clean_image = input_image.data
        sampled_pixel_pos = self.sample_points(mask=None, sample_size=K)
        points_torch, box_torch = self.transform_coord(sampled_pixel_pos, None)
        with trange(self.adv_iters, desc='Attack-SAM-K') as pbar:
            for adv_iter in pbar:
                input_image.requires_grad = True
                image_embeddings = self.model.image_encoder(input_image)
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=points_torch, boxes=box_torch, masks=None)
                low_res_masks, _ = self.model.mask_decoder(
                    image_embeddings = image_embeddings,
                    image_pe = self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings = sparse_embeddings,
                    dense_prompt_embeddings = dense_embeddings,
                    multimask_output = False
                )
                target_low_res_masks = torch.clamp(low_res_masks.detach(), max=-10)
                loss = torch.nn.MSELoss()(low_res_masks, target_low_res_masks)
                loss.backward()
                pbar.set_postfix({'loss': loss.item()})

                perturbation = self.adv_alpha * input_image.grad.data.sign() * 255 / self.model.pixel_std
                adv_image_unclipped = input_image.data - perturbation
                clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image, 
                                                    min=-self.adv_epsilon*255/self.model.pixel_std, 
                                                    max=self.adv_epsilon*255/self.model.pixel_std)
                input_image = torch.clamp(original_clean_image + clipped_perturbation,
                                            min=-self.model.pixel_mean / self.model.pixel_std,
                                            max=(255 - self.model.pixel_mean) / self.model.pixel_std).detach()
        return input_image

    # "Transferable ad-versarial perturbations" (ECCV 18)
    def TAP(self, clean_cv2_image, TAP_alpha=0.5, TAP_eta=0.01):
        input_image = self.set_clean_cv2_image(clean_cv2_image)
        original_features = self.model.image_encoder.hook_feature_list.detach()[self.select_feature_layers] # [L, 1, 64, 64, C]
        T_original_features = torch.sign(original_features) * torch.pow(torch.abs(original_features), TAP_alpha)

        original_clean_image = input_image.data
        input_image = input_image + torch.randn_like(input_image) * 0.01 # add tiny difference to avoid zero gradient
        with trange(self.adv_iters, desc='TAP') as pbar:
            for adv_iter in pbar:
                input_image.requires_grad = True
                self.model.image_encoder(input_image) # foward passing adversarial sample to collect features
                adv_features = self.model.image_encoder.hook_feature_list[self.select_feature_layers]
                T_adv_features = torch.sign(adv_features) * torch.pow(torch.abs(adv_features), TAP_alpha)
                feat_loss = torch.norm(T_adv_features - T_original_features, p=2) * -1
                smooth_loss = TAP_eta * torch.abs(torch.nn.AvgPool2d(3)(original_clean_image - input_image)).sum()
                attack_loss = feat_loss + smooth_loss
                attack_loss.backward()
                pbar.set_postfix({'loss': attack_loss.item()})

                perturbation = self.adv_alpha * input_image.grad.data.sign() * 255 / self.model.pixel_std
                adv_image_unclipped = input_image.data - perturbation
                clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image, 
                                                    min=-self.adv_epsilon*255/self.model.pixel_std, 
                                                    max=self.adv_epsilon*255/self.model.pixel_std)
                input_image = torch.clamp(original_clean_image + clipped_perturbation,
                                            min=-self.model.pixel_mean / self.model.pixel_std,
                                            max=(255 - self.model.pixel_mean) / self.model.pixel_std).detach()
        return input_image
    
    # "Improving adversarial transferability by intermediate-level perturbation decay" (NeurIPS 2023)
    def ILPD(self, clean_cv2_image, TAP_alpha=0.5, TAP_eta=0.01, ILPD_gamma=0.5):
        input_image = self.set_clean_cv2_image(clean_cv2_image)
        original_features = self.model.image_encoder.hook_feature_list.detach()[self.select_feature_layers] # [L, 1, 64, 64, C]
        T_original_features = torch.sign(original_features) * torch.pow(torch.abs(original_features), TAP_alpha)

        original_clean_image = input_image.data
        input_image = input_image + torch.randn_like(input_image) * 0.01 # add tiny difference to avoid zero gradient
        with trange(self.adv_iters, desc='ILPD') as pbar:
            for adv_iter in pbar:
                input_image.requires_grad = True
                self.model.image_encoder(input_image) # foward passing adversarial sample to collect features
                adv_features = self.model.image_encoder.hook_feature_list[self.select_feature_layers]
                mix_features = ILPD_gamma * adv_features +(1 - ILPD_gamma) * original_features
                T_adv_features = torch.sign(mix_features) * torch.pow(torch.abs(mix_features), TAP_alpha)
                feat_loss = torch.norm(T_adv_features - T_original_features, p=2) * -1
                smooth_loss = TAP_eta * torch.abs(torch.nn.AvgPool2d(3)(original_clean_image - input_image)).sum()
                attack_loss = feat_loss + smooth_loss
                attack_loss.backward()
                pbar.set_postfix({'loss': attack_loss.item()})

                perturbation = self.adv_alpha * input_image.grad.data.sign() * 255 / self.model.pixel_std
                adv_image_unclipped = input_image.data - perturbation
                clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image, 
                                                    min=-self.adv_epsilon*255/self.model.pixel_std, 
                                                    max=self.adv_epsilon*255/self.model.pixel_std)
                input_image = torch.clamp(original_clean_image + clipped_perturbation,
                                            min=-self.model.pixel_mean / self.model.pixel_std,
                                            max=(255 - self.model.pixel_mean) / self.model.pixel_std).detach()
        return input_image

    # "Feature space perturbations yield more transferable adversarial examples" (CVPR 19)
    def AA(self, clean_cv2_image, target_cv2_image=None):
        if target_cv2_image is None:
            random_ref_img_id = np.random.randint(1, 1001)
            target_cv2_image = cv2.imread(f"{SAM_DATASET_PATH}sa_{random_ref_img_id}.jpg")
            target_cv2_image = cv2.cvtColor(target_cv2_image, cv2.COLOR_BGR2RGB)
        self.set_clean_cv2_image(target_cv2_image)
        target_ref_features = self.model.image_encoder.hook_feature_list.detach()[self.select_feature_layers] # [L, 1, 64, 64, C]

        input_image = self.set_clean_cv2_image(clean_cv2_image)
        original_clean_image = input_image.data
        
        with trange(self.adv_iters, desc='AA') as pbar:
            for adv_iter in pbar:
                input_image.requires_grad = True
                self.model.image_encoder(input_image)
                adv_features = self.model.image_encoder.hook_feature_list[self.select_feature_layers]
                attack_loss = torch.norm(adv_features - target_ref_features, p=2)
                attack_loss.backward()
                pbar.set_postfix({'loss': attack_loss.item()})

                perturbation = self.adv_alpha * input_image.grad.data.sign() * 255 / self.model.pixel_std
                adv_image_unclipped = input_image.data - perturbation
                clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image, 
                                                    min=-self.adv_epsilon*255/self.model.pixel_std, 
                                                    max=self.adv_epsilon*255/self.model.pixel_std)
                input_image = torch.clamp(original_clean_image + clipped_perturbation,
                                            min=-self.model.pixel_mean / self.model.pixel_std,
                                            max=(255 - self.model.pixel_mean) / self.model.pixel_std).detach()
        return input_image
    
    # "Black-box targeted adversarial attack on segment anything" (Arxiv 23.10)
    def PATA(self, clean_cv2_image, PATA_lambda=0.1):
        L = len(self.select_feature_layers)
        random_ref_img_id = np.random.randint(1, 1001)
        target_cv2_image = cv2.imread(f"{SAM_DATASET_PATH}sa_{random_ref_img_id}.jpg")
        target_cv2_image = cv2.cvtColor(target_cv2_image, cv2.COLOR_BGR2RGB)
        self.set_image(target_cv2_image)
        target_ref_features = self.model.image_encoder.hook_feature_list.detach()[self.select_feature_layers]

        random_cpt_id = np.random.randint(1, 1001)
        cpt_cv2_image = cv2.imread(f"{SAM_DATASET_PATH}sa_{random_cpt_id}.jpg")
        cpt_cv2_image = cv2.cvtColor(cpt_cv2_image, cv2.COLOR_BGR2RGB)
        self.set_image(cpt_cv2_image)
        cpt_features = self.model.image_encoder.hook_feature_list.detach()[self.select_feature_layers]
        cpt_features_flatten = cpt_features.reshape(L, -1)

        cosfn = torch.nn.CosineSimilarity(dim=-1)
        input_image = self.set_clean_cv2_image(clean_cv2_image)
        original_clean_image = input_image.data
        
        with trange(self.adv_iters, desc='PATA') as pbar:
            for adv_iter in pbar:
                input_image.requires_grad = True
                self.model.image_encoder(input_image)
                adv_features = self.model.image_encoder.hook_feature_list[self.select_feature_layers]
                feature_mse_loss = torch.norm(adv_features - target_ref_features, p=2)

                mix_features_flatten = (adv_features + cpt_features).reshape(L, -1)
                adv_features_flatten = adv_features.reshape(L, -1)
                feature_dominance_reg_loss = (cosfn(adv_features_flatten, mix_features_flatten) - cosfn(cpt_features_flatten, mix_features_flatten)).mean()
                attack_loss = feature_mse_loss - PATA_lambda * feature_dominance_reg_loss
                attack_loss.backward()
                pbar.set_postfix({'loss': attack_loss.item()})

                perturbation = self.adv_alpha * input_image.grad.data.sign() * 255 / self.model.pixel_std
                adv_image_unclipped = input_image.data - perturbation
                clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image, 
                                                    min=-self.adv_epsilon*255/self.model.pixel_std, 
                                                    max=self.adv_epsilon*255/self.model.pixel_std)
                input_image = torch.clamp(original_clean_image + clipped_perturbation,
                                            min=-self.model.pixel_mean / self.model.pixel_std,
                                            max=(255 - self.model.pixel_mean) / self.model.pixel_std).detach()
        return input_image

    # "Black-box targeted adversarial attack on segment anything" (Arxiv 23.10)
    def PATA_plusplus(self, clean_cv2_image, PATA_lambda=0.1):
        L = len(self.select_feature_layers)
        random_ref_img_id = np.random.randint(1, 1001)
        target_cv2_image = cv2.imread(f"{SAM_DATASET_PATH}sa_{random_ref_img_id}.jpg")
        target_cv2_image = cv2.cvtColor(target_cv2_image, cv2.COLOR_BGR2RGB)
        self.set_image(target_cv2_image)
        target_ref_features = self.model.image_encoder.hook_feature_list.detach()[self.select_feature_layers]


        cosfn = torch.nn.CosineSimilarity(dim=-1)
        input_image = self.set_clean_cv2_image(clean_cv2_image)
        original_clean_image = input_image.data
        
        with trange(self.adv_iters, desc='PATA') as pbar:
            for adv_iter in pbar:
                random_cpt_id = np.random.randint(1, 1001)
                cpt_cv2_image = cv2.imread(f"{SAM_DATASET_PATH}sa_{random_cpt_id}.jpg")
                cpt_cv2_image = cv2.cvtColor(cpt_cv2_image, cv2.COLOR_BGR2RGB)
                self.set_image(cpt_cv2_image)
                cpt_features = self.model.image_encoder.hook_feature_list.detach()[self.select_feature_layers]
                cpt_features_flatten = cpt_features.reshape(L, -1)

                self.set_clean_cv2_image(clean_cv2_image)
                input_image.requires_grad = True
                self.model.image_encoder(input_image)
                adv_features = self.model.image_encoder.hook_feature_list[self.select_feature_layers]
                feature_mse_loss = torch.norm(adv_features - target_ref_features, p=2)

                mix_features_flatten = (adv_features + cpt_features).reshape(L, -1)
                adv_features_flatten = adv_features.reshape(L, -1)
                feature_dominance_reg_loss = (cosfn(adv_features_flatten, mix_features_flatten) - cosfn(cpt_features_flatten, mix_features_flatten)).mean()
                attack_loss = feature_mse_loss - PATA_lambda * feature_dominance_reg_loss
                attack_loss.backward()
                pbar.set_postfix({'loss': attack_loss.item()})

                perturbation = self.adv_alpha * input_image.grad.data.sign() * 255 / self.model.pixel_std
                adv_image_unclipped = input_image.data - perturbation
                clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image, 
                                                    min=-self.adv_epsilon*255/self.model.pixel_std, 
                                                    max=self.adv_epsilon*255/self.model.pixel_std)
                input_image = torch.clamp(original_clean_image + clipped_perturbation,
                                            min=-self.model.pixel_mean / self.model.pixel_std,
                                            max=(255 - self.model.pixel_mean) / self.model.pixel_std).detach()
        return input_image
    
    # "Unsegment Anything by Simulating Deformation" (CVPR 24)
    def UAD(self, clean_cv2_image,
            control_loss_alpha=0.05, fidelity_loss_alpha=1, deform_epochs=40, est_fidelity_iter=4,
            warp_size=4, warp_filter_stride=300, num_split=6, src_rand_range=0.0, dst_rand_range=0.1,
            use_DI=False, DI_noise_std=4.0/255, use_MI=False, MI_momentum=1):
        clean_cv2_image = cv2.imread(clean_cv2_image)
        clean_cv2_image = cv2.cvtColor(clean_cv2_image, cv2.COLOR_BGR2RGB)
        deformer = Deformer(clean_cv2_image=clean_cv2_image,
                            num_split=num_split, 
                            fix_corner=False,
                            src_rand_range=src_rand_range, 
                            dst_rand_range=dst_rand_range)
        base_locs = np.array(deformer.source_centers)
        fix_idx = 4 if deformer.fix_corner else 0
        control_point_size = base_locs.shape[0] - fix_idx
        source_locs = np.repeat(base_locs[np.newaxis, :], warp_size, axis=0)
        source_locs[:, fix_idx:,0] = np.minimum(np.maximum(source_locs[:, fix_idx:,0]  + 
                                            np.random.randn(warp_size, control_point_size)*deformer.block_h*deformer.src_rand_range, 0), 1023)
        source_locs[:, fix_idx:,1] = np.minimum(np.maximum(source_locs[:, fix_idx:,1]  + 
                                            np.random.randn(warp_size, control_point_size)*deformer.block_w*deformer.src_rand_range, 0), 1023)                
        dest_locs = copy.copy(source_locs)
        dest_locs[:, fix_idx:,0]  = np.minimum(np.maximum(source_locs[:, fix_idx:,0]  + 
                                            np.random.randn(warp_size, control_point_size)*deformer.block_h*deformer.dst_rand_range, 0), 1023)
        dest_locs[:, fix_idx:,1]  = np.minimum(np.maximum(source_locs[:, fix_idx:,1]  + 
                                            np.random.randn(warp_size, control_point_size)*deformer.block_w*deformer.dst_rand_range, 0), 1023)
        source_locs = torch.Tensor(source_locs).to(self.device)
        dest_locs = torch.Tensor(dest_locs).to(self.device)
        deform_optimizer = torch.optim.Adam([source_locs, dest_locs], lr=10)

        input_image = self.set_image(clean_cv2_image)
        original_input = input_image.detach().clone()
        original_features = self.model.image_encoder.hook_feature_list.detach()[self.select_feature_layers]
        L, B, C, H, W = original_features.shape
        deform_filter = generate_max_filter(original_input.permute(0, 2, 3, 1), source_locs, dest_locs, filter_stride=warp_filter_stride)
        cosfn = torch.nn.CosineSimilarity(dim=-1)
        with trange(deform_epochs, desc='UAD') as pbar:
            for deform_epoch in pbar:
                deform_optimizer.zero_grad()
                source_locs.requires_grad_(True)
                dest_locs.requires_grad_(True)
                if warp_size == 1:
                    warped_input, dense_flows = sparse_image_warp(
                        original_input.permute(0, 2, 3, 1),
                        source_locs, dest_locs,
                        interpolation_order=1,
                        regularization_weight=0.0,
                        num_boundaries_points=0
                    )
                elif warp_size > 1:
                    warped_input, dense_flows = avg_batch_sparse_image_warp_by_filter(
                        original_input.permute(0, 2, 3, 1),
                        source_locs, dest_locs,
                        interpolation_order=1,
                        regularization_weight=0.0,
                        num_boundaries_points=0,
                        max_filter=deform_filter
                    )
                deform_loss = deformer.deform_loss(original_input, warped_input.permute(0, 3, 1, 2))
                control_loss = deformer.control_loss(source_locs, dest_locs)
                deformation_loss = deform_loss + control_loss_alpha * control_loss
                deformation_loss.backward()
                deform_optimizer.step()
                with torch.no_grad():
                    source_locs.data = torch.clamp(source_locs.data, 0, 1023)
                    dest_locs.data = torch.clamp(dest_locs.data, 0, 1023)
                self.model.image_encoder(warped_input.permute(0, 3, 1, 2))
                warped_features = self.model.image_encoder.hook_feature_list.detach()[self.select_feature_layers]
                adv_input = original_input.data
                adv_iters = self.adv_iters if deform_epoch == deform_epochs - 1 else est_fidelity_iter
                accumulated_grad = torch.zeros_like(adv_input)
                for adv_iter in range(adv_iters):
                    if adv_iter == 0:
                        deform_optimizer.zero_grad()
                    if use_DI:
                        adv_input = adv_input + torch.randn_like(adv_input) * DI_noise_std
                    adv_input.requires_grad_(True)
                    self.model.image_encoder(adv_input)
                    adv_features = self.model.image_encoder.hook_feature_list[self.select_feature_layers]

                    original_features_flatten = original_features.reshape(L, -1)
                    adv_features_flatten = adv_features.reshape(L, -1)
                    warped_features_flatten = warped_features.reshape(L, -1)

                    fidelity_loss = (- cosfn(adv_features_flatten, warped_features_flatten).mean() 
                                        + cosfn(adv_features_flatten, original_features_flatten).mean()) * fidelity_loss_alpha

                    fidelity_loss.backward(inputs=[adv_input, source_locs, dest_locs])

                    if use_MI:
                        accumulated_grad = MI_momentum * accumulated_grad + adv_input.grad.data / torch.norm(adv_input.grad.data, p=1)
                        perturbation = self.adv_alpha * accumulated_grad.sign() * 255 / self.model.pixel_std
                    else:
                        perturbation = self.adv_alpha * adv_input.grad.data.sign() * 255 / self.model.pixel_std
                    adv_image_unclipped = adv_input.data - perturbation
                    clipped_perturbation = torch.clamp(adv_image_unclipped - original_input, 
                                                        min=-self.adv_epsilon*255/self.model.pixel_std, 
                                                        max=self.adv_epsilon*255/self.model.pixel_std)
                    adv_input = torch.clamp(original_input + clipped_perturbation,
                                                min=-self.model.pixel_mean / self.model.pixel_std,
                                                max=(255 - self.model.pixel_mean) / self.model.pixel_std).detach()
        self.warped_input = warped_input.permute(0, 3, 1, 2)
        return adv_input

