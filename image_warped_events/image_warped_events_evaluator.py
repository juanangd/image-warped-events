import rospy
import torch
import numpy as np
from torch.func import jacrev
from torchvision.transforms import GaussianBlur
from .sharpness_loss_fns import SharpnessLossFunctionSuite

torch.set_grad_enabled(True)  # Context-manager


class ImageWarpedEventsEvaluator:

    def __init__(self, camera_intrinsics, camera_intrinsics_inv, img_size, gauss_kernel_size, gaussian_sigma,
                 sharpness_function_type='variance', use_polarity=False, motion_model="rotation",
                 param_to_eval=[True, True, True], img_area_kernel="exponential", approximate_rmatrix=False,
                 use_bilinear_voting=False, iwe_padding=(10, 10), use_gpu=False):

        self.use_polarity = use_polarity
        self.motion_model = motion_model
        self.param_to_eval = param_to_eval
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.sharpness_function = SharpnessLossFunctionSuite(sharpness_function_type, img_area_kernel)
        self.camera_intrinsics = camera_intrinsics.to(self.device).to(torch.float64)
        self.camera_intrinsics_inv = camera_intrinsics_inv.to(self.device).to(torch.float64)
        self.img_size = img_size  # height, width
        self.approximate_rotation_matrix = approximate_rmatrix
        self.derivative_loss_fn = jacrev(self.loss_fn)
        self.gaussian_smoother = GaussianBlur(gauss_kernel_size, list(gaussian_sigma)).to(self.device)
        self.center_kernel = torch.Tensor([gauss_kernel_size // 2, gauss_kernel_size // 2]).to(self.device)
        self.gaussian_sigma = gaussian_sigma.clone().detach().to(self.device)
        self.idx_kernel = self.get_idx_kernel(gauss_kernel_size).to(self.device)
        self.use_bilinear_voting = use_bilinear_voting
        self.iwe_padding = iwe_padding

    @staticmethod
    def get_idx_kernel(k_size):
        rows, cols = np.indices((k_size, k_size))
        kernel_idx = np.column_stack([
            rows.ravel(), cols.ravel()
        ])
        return torch.Tensor(kernel_idx)

    def rotation_matrix(self, rot_vel, time):

        time.to(self.device)
        rotation = rot_vel * time

        if self.approximate_rotation_matrix:

            rot_x = rotation[0] if self.param_to_eval[0] else torch.tensor(0)
            rot_y = rotation[1] if self.param_to_eval[1] else torch.tensor(0)
            rot_z = rotation[2] if self.param_to_eval[2] else torch.tensor(0)

            rot_matrix = torch.stack([
                torch.stack([torch.tensor(0), -rot_z, rot_y]),
                torch.stack([rot_z, torch.tensor(0), -rot_x]),
                torch.stack([-rot_y, rot_x, torch.tensor(0)])
            ]).to(self.device)

            return torch.eye(3).to(self.device) + rot_matrix  # First order approximation of expm

        else:

            R_matrix = torch.eye(3, dtype=torch.float64).to(self.device)
            if self.param_to_eval[0]:
                rot_x = torch.stack([
                    torch.tensor([1., 0., 0.]),
                    torch.stack([torch.tensor(0.), torch.cos(rotation[0]), -torch.sin(rotation[0])]),
                    torch.stack([torch.tensor(0.), torch.sin(rotation[0]), torch.cos(rotation[0])])
                ]).to(self.device)
                R_matrix = torch.mm(R_matrix, rot_x)

            if self.param_to_eval[1]:
                rot_y = torch.stack([
                    torch.stack([torch.cos(rotation[1]), torch.tensor(0.), torch.sin(rotation[1])]),
                    torch.tensor([0., 1., 0.]),
                    torch.stack([-torch.sin(rotation[1]), torch.tensor(0.), torch.cos(rotation[1])])
                ]).to(self.device)

                R_matrix = torch.mm(R_matrix, rot_y)

            if self.param_to_eval[2]:
                rot_z = torch.stack([
                    torch.stack([torch.cos(rotation[2]), -torch.sin(rotation[2]), torch.tensor(0.)]),
                    torch.stack([torch.sin(rotation[2]), torch.cos(rotation[2]), torch.tensor(0.)]),
                    torch.tensor([0., 0., 1.])
                ]).to(self.device)

                R_matrix = torch.mm(R_matrix, rot_z)

            return R_matrix

    def rotate_event(self, event, rot_vel):
        time = event[2]
        rotation_m = self.rotation_matrix(rot_vel, time)
        point_2d = torch.stack([event[0], event[1], torch.tensor(1.0)]).to(torch.float64).to(self.device)
        point_3d = torch.matmul(self.camera_intrinsics_inv, point_2d)
        point_3d_rot = torch.matmul(rotation_m, point_3d)
        point_2d_warped = torch.matmul(self.camera_intrinsics, point_3d_rot)

        return point_2d_warped[:-1]

    def accumulate_events_bilinear_voting(self, warped_events, polarities):

        iwe_size = torch.Size((self.img_size[0] + self.iwe_padding[0], self.img_size[1] + self.iwe_padding[1]))
        iwe = torch.zeros(iwe_size, dtype=torch.float64).to(self.device)

        x_float = warped_events[:, 0]
        y_float = warped_events[:, 1]

        x_int = x_float.floor()
        y_int = y_float.floor()

        dx = (x_float - x_int).to(torch.float64).to(self.device)
        dy = (y_float - y_int).to(torch.float64).to(self.device)

        x_ind = (x_int + self.iwe_padding[1]).long()
        y_ind = (y_int + self.iwe_padding[0]).long()

        zero = torch.tensor([0.]).to(self.device)
        one = torch.tensor([1.]).to(self.device)

        valid_indexes = torch.where(x_ind > 0, one, zero) * torch.where(y_ind > 0, one, zero) * torch.where(
            x_ind < iwe_size[1] - 1, one, zero) * torch.where(
            iwe_size[0] - 1 > y_ind, one, zero)
        valid_indexes = valid_indexes.bool()

        y_ind = y_ind[valid_indexes]
        x_ind = x_ind[valid_indexes]
        dx = dx[valid_indexes]
        dy = dy[valid_indexes]
        polarities = polarities[valid_indexes]

        if self.use_polarity:
            iwe.index_put_((y_ind, x_ind), polarities * (1. - dx) * (1. - dy), accumulate=True)
            iwe.index_put_((y_ind, x_ind + 1), polarities * dx * (1. - dy), accumulate=True)
            iwe.index_put_((y_ind + 1, x_ind), polarities * (1. - dx) * dy, accumulate=True)
            iwe.index_put_((y_ind + 1, x_ind + 1), polarities * dx * dy, accumulate=True)
        else:
            iwe.index_put_((y_ind, x_ind), (1. - dx) * (1. - dy), accumulate=True)
            iwe.index_put_((y_ind, x_ind + 1), dx * (1. - dy), accumulate=True)
            iwe.index_put_((y_ind + 1, x_ind), (1. - dx) * dy, accumulate=True)
            iwe.index_put_((y_ind + 1, x_ind + 1), dx * dy, accumulate=True)
        return iwe

    def gaussian_event_kernel(self, warped_events):

        event_int = warped_events.long()
        decimals = warped_events - event_int
        event_int = torch.where(event_int > 0.5, event_int + 1, event_int)
        decimals = torch.where(decimals > 0.5, decimals - 0.5, decimals)

        gaussian_mu = self.center_kernel + decimals
        inside_exp = (self.idx_kernel - gaussian_mu) ** 2 / self.gaussian_sigma ** 2
        sum_xy = torch.sum(inside_exp, 1)
        gaussian = torch.exp(-sum_xy)
        gaussian_norm = gaussian / torch.sum(gaussian)
        idx_in_img = event_int - self.center_kernel + self.idx_kernel
        return idx_in_img.long(), gaussian_norm

    def accumulate_events_wo_voting(self, events):

        img = torch.zeros(self.img_size, dtype=torch.float64).to(self.device)
        img = img.index_put((events[:, 1].long(), events[:, 0].long()), torch.ones((events.shape[0]))).to(self.device)
        return img

    def accumulate_events_gaussian_voting(self, warped_events, polarities):

        iwe_size = torch.Size((self.img_size[0] + self.iwe_padding[0], self.img_size[1] + self.iwe_padding[1]))

        img = torch.zeros(iwe_size, dtype=torch.float64).to(self.device)
        idx_image, gaussian_norm = torch.vmap(self.gaussian_event_kernel)(warped_events)

        if self.use_polarity:
            tiled_array = torch.tile(polarities.reshape(-1, 1), (1, gaussian_norm.shape[1]))
            gaussian_norm = gaussian_norm * tiled_array

        idx_image_resized_shifted = idx_image.reshape((-1, 2)) + torch.flip(torch.Tensor(self.iwe_padding), [0]).to(
            self.device)
        gaussian_norm_resized = gaussian_norm.reshape((-1,))

        zero = torch.tensor([0.]).to(self.device)
        one = torch.tensor([1.]).to(self.device)

        valid_indexes = torch.where(idx_image_resized_shifted[:, 0] > 0, one, zero) * torch.where(
            idx_image_resized_shifted[:, 1] > 0, one, zero) * torch.where(
            idx_image_resized_shifted[:, 0] < iwe_size[1] - 1, one, zero) * torch.where(
            iwe_size[0] - 1 > idx_image_resized_shifted[:, 1], one, zero)
        valid_indexes = valid_indexes.bool()

        img.index_put_(
            (idx_image_resized_shifted[valid_indexes, 1].long(), idx_image_resized_shifted[valid_indexes, 0].long()),
            gaussian_norm_resized[valid_indexes],
            accumulate=True)
        return img

    def translate_event_with_divergence(self, event, divergence):

        """event_0_centered_x = event[0] - self.camera_intrinsics[0, 2]
        event_0_centered_y = event[1] - self.camera_intrinsics[1, 2]

        lenght_i = torch.sqrt(event_0_centered_x ** 2 + event_0_centered_y ** 2)
        theta_i = torch.atan2(event_0_centered_y, event_0_centered_x)

        lenght_f = lenght_i - lenght_i * divergence * event[2]

        return torch.stack([lenght_f * torch.cos(theta_i) + self.camera_intrinsics[0, 2],
                            lenght_f * torch.sin(theta_i) + self.camera_intrinsics[1, 2]])"""

        event_0_centered_x = event[0] - self.camera_intrinsics[0, 2]
        event_0_centered_y = event[1] - self.camera_intrinsics[1, 2]

        event_warped_x = (event_0_centered_x + (event_0_centered_x * divergence * event[2])) + self.camera_intrinsics[
            0, 2]
        event_warped_y = (event_0_centered_y + (event_0_centered_y * divergence * event[2])) + self.camera_intrinsics[
            1, 2]

        return torch.stack([event_warped_x, event_warped_y])

    def translate_event(self, event, tran_vel, depth):

        tran_vel_mod = torch.stack(
            [
                tran_vel[0] if self.param_to_eval[0] else torch.tensor(0.),
                tran_vel[1] if self.param_to_eval[1] else torch.tensor(0.),
                tran_vel[2] if self.param_to_eval[2] else torch.tensor(0.),

            ]
        )
        time = event[2]
        point_3d = torch.stack(
            [torch.Tensor(((event[0] - self.camera_intrinsics[0, 2]) / self.camera_intrinsics[0, 0]) * depth),
             torch.Tensor(((event[1] - self.camera_intrinsics[1, 2]) / self.camera_intrinsics[1, 1]) * depth),
             depth])

        point_3d_trans = point_3d + tran_vel_mod * time
        x_warped = (point_3d_trans[0] * self.camera_intrinsics[0, 0]) / point_3d_trans[2] + self.camera_intrinsics[0, 2]
        y_warped = (point_3d_trans[1] * self.camera_intrinsics[1, 1]) / point_3d_trans[2] + self.camera_intrinsics[1, 2]
        return torch.stack([x_warped, y_warped])

    def events_warping(self, events, motion_params, depth=torch.tensor(1.)):

        # The events are assuming to be referenced agains the first event time. so the first event timestamp is always 0
        if self.motion_model == "rotation":
            warped_events = torch.vmap(self.rotate_event, in_dims=(0, None))(events, motion_params)

        elif self.motion_model == "translation":
            warped_events = torch.vmap(self.translate_event, in_dims=(0, None, None))(events, motion_params, depth)

        elif self.motion_model == "translation_divergence":
            warped_events = torch.vmap(self.translate_event_with_divergence, in_dims=(0, None))(events,
                                                                                                motion_params).squeeze()

        else:
            rospy.logerr(f"{self.motion_model} motion model not implemented!")
        return warped_events

    def loss_fn(self, motion_params, batch, depth=torch.tensor(1.)):

        iwe = self.compute_iwe(motion_params, batch, depth)
        return self.sharpness_function.fn_loss(iwe)

    def compute_iwe(self, motion_params, batch, depth=torch.tensor(1.)):

        warped_events = self.events_warping(batch, motion_params, depth)
        polarities = torch.where(batch[:, 3] > 0, 1., -1.).to(self.device)
        if self.use_bilinear_voting:
            iwe = self.accumulate_events_bilinear_voting(warped_events, polarities)
            iwe = self.gaussian_smoother(iwe.view(1, iwe.shape[0], iwe.shape[1]))
            iwe = iwe.squeeze()
        else:
            iwe = self.accumulate_events_gaussian_voting(warped_events, polarities)

        return iwe

    def jacobian_loss_fn(self, motion_params, batch, depth=torch.tensor(1.)):

        torch.cuda.empty_cache()
        motion_params.to(self.device)
        batch.to(self.device)
        depth.to(self.device)
        return self.derivative_loss_fn(motion_params, batch, depth)