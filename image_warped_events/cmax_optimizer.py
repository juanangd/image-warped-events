import torch
from torch import optim
from .image_warped_events_evaluator import ImageWarpedEventsEvaluator
import numpy as np
import time
torch.optim.Adam([torch.zeros(1)])

class CmaxOptimizer:

	def __init__(self,
				 camera_intrinsics_matrix,
				 camera_intrinsics_matrix_inverse,
				 sensor_size,
				 gauss_kernel_size,
				 sigma_kernel_size,
				 motion_model,
				 sharpness_function_type = "poisson",
				 image_area_kernel = "exponential",
				 lr = 0.1,
				 lr_scheduler_step = 5, #10
				 lr_scheduler_decay = 0.1, # 0.1
				 n_iters = 100,
				 stopping_criteria_tol = 1e-30,
				 optimizer = "Adam",
				 approximate_rmatrix = False,
				 use_bilinear_voting = False,
				 iwe_padding = (10, 10),
				 param_to_eval = [True, True, True]):

		self.iwe_evaluator = ImageWarpedEventsEvaluator(camera_intrinsics_matrix,
														camera_intrinsics_matrix_inverse,
														sensor_size,
														gauss_kernel_size,
														sigma_kernel_size,
														sharpness_function_type,
														motion_model=motion_model,
														img_area_kernel=image_area_kernel,
														approximate_rmatrix=approximate_rmatrix,
														use_bilinear_voting=use_bilinear_voting,
														iwe_padding=iwe_padding,
														param_to_eval=param_to_eval)



		self.optimizer_name = optimizer
		self.lr = lr
		self.n_iters = n_iters
		self.loss_func = self.iwe_evaluator.loss_fn

		self.lr_scheduler_step = lr_scheduler_step
		self.lr_scheduler_decay = lr_scheduler_decay
		self.stopping_criteria = stopping_criteria_tol


	def optimize(self, tensor_events, init_value, max_n_iters=100):

		self.current_vel = init_value
		self.current_vel.requires_grad = True

		optimizer= optim.Adam([self.current_vel], self.lr)
		lr_scheduler = optim.lr_scheduler.StepLR(optimizer, self.lr_scheduler_step, self.lr_scheduler_decay)

		min_loss= torch.inf
		loss_list = []
		best_value = self.current_vel

		for it in range(max_n_iters):
			optimizer.zero_grad()
			loss = self.loss_func(self.current_vel, tensor_events)
			loss_value = loss.detach().numpy()
			if loss_value < min_loss:
				best_value = self.current_vel
				min_loss = loss_value

			loss_list.append(loss_value)
			loss.backward()
			optimizer.step()
			lr_scheduler.step()

			if it > 0 and abs(loss_list[it] - loss_list[it - 1]) < self.stopping_criteria: # stop criteria
				break

		return best_value.detach().numpy() , np.array(loss_list)

	def compute_image(self, velocity, tensor_events):

		return self.iwe_evaluator.compute_iwe(velocity, tensor_events)