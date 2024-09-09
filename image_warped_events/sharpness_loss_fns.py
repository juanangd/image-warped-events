import torch
import torch.nn as nn
import torch.nn.functional as F

class SharpnessLossFunctionSuite():

    def __init__(self, sharpness_fun_type="variance", img_area_kernel="exponential"):
        """
        This class implements different types of loss function used for CMAX.
        Please refer to https://arxiv.org/abs/1904.07235 for more info.

        :param sharpness_fun_type:
        :param img_area_kernel:
        """
        self.fn_type = sharpness_fun_type
        self.img_area_kernel = img_area_kernel
        self.sobel_kernel_x = torch.Tensor([[1, 0, -1],
                                            [2, 0, -2],
                                            [1, 0, -1]])
        self.sobel_kernel_x = torch.Tensor([[1, 0, -1],
                                            [2, 0, -2],
                                            [1, 0, -1]])

        self.sobel_kernel_y = torch.reshape(torch.transpose(self.sobel_kernel_x, 0, 1),
                                            (1, 1, self.sobel_kernel_x.shape[0], self.sobel_kernel_x.shape[1])).to(torch.float64)
        self.sobel_kernel_x = torch.reshape(self.sobel_kernel_x,
                                            (1, 1, self.sobel_kernel_x.shape[0], self.sobel_kernel_x.shape[1])).to(torch.float64)


        if img_area_kernel == "exponential":
            self.kernel_img_area = lambda x: torch.Tensor([1.]) - torch.exp(-x)
        elif img_area_kernel == "gaussian":
            self.kernel_img_area = lambda x: torch.erf(x)
        elif img_area_kernel == "lorentzian":
            self.kernel_img_area = lambda x: torch.Tensor([2 / torch.pi]) * torch.arctan(x)
        elif img_area_kernel == "hyperbolic":
            self.kernel_img_area = lambda x: torch.tanh(x)
        else: # default
            self.kernel_img_area = lambda x: torch.Tensor([1.]) - torch.exp(-x)
        self.gaussian_kernel = self.create_centered_gaussian_kernel((5., 5), (1, 1))

    def fn_loss(self, image):
        """
        It computes the loss to the image input.

        :param image:
        :return:
        """
        fn_type_splitted = self.fn_type.split("-")
        if len(fn_type_splitted)==1:
            return getattr(self, f'fn_{self.fn_type}')(image)
        elif fn_type_splitted[1] == "weighted":
            return self.fn_weighted(image, fun=getattr(self, f'fn_{fn_type_splitted[0]}'))
        else:
            print("fn_type is not valid")

    def fn_magnitude_gradient(self, image):

        img_reshaped = torch.reshape(image, (1, 1, image.shape[-2], image.shape[-1]))

        grad_x = nn.functional.conv2d(img_reshaped, self.sobel_kernel_x)
        grad_y = nn.functional.conv2d(img_reshaped, self.sobel_kernel_y)

        return torch.sum(grad_x**2 + grad_y**2)


    def fn_magnitude_hessian(self, image):

        img_reshaped = torch.reshape(image, (1, 1, image.shape[-2], image.shape[-1]))

        grad_x = nn.functional.conv2d(img_reshaped, self.sobel_kernel_x)
        grad_y = nn.functional.conv2d(img_reshaped, self.sobel_kernel_y)
        grad_xx = nn.functional.conv2d(grad_x, self.sobel_kernel_x)
        grad_yy = nn.functional.conv2d(grad_y, self.sobel_kernel_x)

        return torch.sum(grad_xx ** 2 + grad_yy ** 2)

    @staticmethod
    def fn_poisson(img):

        r = torch.tensor(0.1)
        beta = torch.tensor(1.59)

        map = ((img + r).lgamma() + r * (beta).log() - (img + 1.).lgamma() - (r).lgamma() - (img + r) * (beta + 1).log())
        loss = map.sum() / img.sum()
        return -loss

    @staticmethod
    def fn_variance(img):
        return - torch.var(img)

    def fn_image_area(self, img):

        return torch.sum(self.kernel_img_area(img))

    @staticmethod
    def fn_mean_square(image):

        nels = image.shape[0] * image.shape[1]
        return torch.norm(image, p=2) **2 / nels

    @staticmethod
    def fn_mean_absolute_deviation(image):

        return torch.mean(torch.abs(image-torch.mean(image)))

    @staticmethod
    def fn_mean_absolute_value(image):

        return torch.mean(torch.abs(image))

    @staticmethod
    def fn_number_pixels_activated(image):
        activated_pixels = torch.where(image>0, torch.Tensor([1.]), torch.Tensor([0.]))
        return torch.sum(activated_pixels)

    @staticmethod
    def fn_entropy(image):

        histogram, _ = torch.histogram(image.view(-1), bins=50)
        prob = histogram.float() / histogram.sum().float()

        entropy_values = torch.where(prob>0, prob * torch.log(prob), torch.Tensor([0.]))
        return - torch.sum(entropy_values)

    def fn_image_range(self, image):

        histogram, _ = torch.histogram(image.view(-1), bins=50)
        prob = histogram.float() / histogram.sum().float()

        return self.fn_image_area(prob)

    def create_centered_gaussian_kernel(self, size, sigma=(10, 10)):

        x_coord = torch.arange(0, size[1], dtype=torch.float32) - torch.Tensor([size[1]//2])
        y_coord = torch.arange(0, size[0], dtype=torch.float32) - torch.Tensor([size[0]//2])
        rows_grid, column_grid = torch.meshgrid(y_coord, x_coord)

        # Calculate the Gaussian kernel using the formula
        gaussian_kernel = torch.exp(-(column_grid**2 / (2.0 * sigma[0]**2) + rows_grid**2 / (2.0 * sigma[1]**2)))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        return gaussian_kernel

    def fn_weighted(self, image, patch_size=(10, 10), sigma=(10, 10), fun=torch.var):

        pad_size_y = 0 if image.shape[0] % patch_size[0] ==0 else (patch_size[0] - image.shape[0] % patch_size[0]) // 2
        pad_size_x = 0 if image.shape[1] % patch_size[1] ==0 else (patch_size[1] - image.shape[1] % patch_size[1]) // 2

        image_padded = F.pad(image, (pad_size_x, pad_size_x, pad_size_y, pad_size_y))
        patches = image_padded.unfold(0, patch_size[0], patch_size[0]).unfold(1, patch_size[1], patch_size[1])
        patches_flattened = patches.reshape((-1, patches.shape[-2], patches.shape[-1]))

        contrast_patches = torch.vmap(fun, 0)(patches_flattened)
        gaussian_kernel = self.create_centered_gaussian_kernel((patches.shape[0], patches.shape[1]), sigma)
        gaussian_kernel_flattened = gaussian_kernel.view(-1)

        return torch.sum(torch.mul(gaussian_kernel_flattened, contrast_patches))

    def fn_moran_index(self, image, sigma=(1., 1.), kernel_size = (5, 5)):

        # gaussian_kernel = self.create_centered_gaussian_kernel(kernel_size, sigma)
        gaussian_kernel = self.gaussian_kernel
        weights = gaussian_kernel / 1 - gaussian_kernel[int(kernel_size[0] / 2), int(kernel_size[1] / 2)]
        weights[int(kernel_size[0] / 2), int(kernel_size[1] / 2)] = 0

        standarized_image = image - image.mean() / image.std()

        spatially_convoluted_img = F.conv2d(
            standarized_image.view(1, 1, image.shape[0], image.shape[1]),
            weights.view(1, 1, weights.shape[0], weights.shape[1]), padding='same'
        ).squeeze()

        moran_index = torch.sum(torch.mul(standarized_image, spatially_convoluted_img)) / (image.shape[0] * image.shape[1])

        return moran_index


    def fn_geary_contiguity_ratio(self, image, sigma=(1., 1.), kernel_size = (5, 5)):

        # gaussian_kernel = self.create_centered_gaussian_kernel(kernel_size, sigma)
        gaussian_kernel = self.gaussian_kernel
        weights = gaussian_kernel / 1 - gaussian_kernel[int(kernel_size[0] / 2), int(kernel_size[1] / 2)]
        weights[int(kernel_size[0] / 2), int(kernel_size[1] / 2)] = 0

        standarized_image = image - image.mean() / image.std()
        squared_standarized_image =  standarized_image ** 2


        second_term = F.conv2d(
            squared_standarized_image.view(1, 1, image.shape[0], image.shape[1]),
            weights.view(1, 1, weights.shape[0], weights.shape[1]), padding='same'
        ).squeeze()

        third_term = torch.mul(2 * standarized_image, F.conv2d(
            standarized_image.view(1, 1, image.shape[0], image.shape[1]),
            weights.view(1, 1, weights.shape[0], weights.shape[1]), padding='same'
        ).squeeze())

        c = squared_standarized_image + second_term + third_term

        return 0.5 * (1/(image.shape[0]*image.shape[1])) * torch.sum(c)