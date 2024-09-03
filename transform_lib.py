import numpy as np
import paddle
import paddle.vision.transforms as transforms
import paddle
import paddle.nn as nn

# use from https://github.com/Spijkervet/BYOL/blob/977621ae16de1f969e048a68a3f6e4dd9c3d226d/modules/transformations/simclr.py#L4
class SimCLRTransform:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    data_format is array or image
    """

    def __init__(self, size=32, gaussian=False, data_format="array"):
        s = 1
        color_jitter =  transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        if gaussian:
            self.train_transform =  transforms.Compose(
                [
                    #  transforms.ToPILImage(mode='RGB'),
                     transforms.Resize(size=size),
                     transforms.RandomResizedCrop(size=size),
                     transforms.RandomHorizontalFlip(),  # with 0.5 probability
                     transforms.RandomApply([color_jitter], p=0.8),
                     transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(kernel_size=int(0.1 * size)),
                    # RandomApply( transforms.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
                     transforms.ToTensor(),
                ]
            )
        else:
            if data_format == "array":
                self.train_transform =  transforms.Compose(
                    [
                        #  transforms.ToPILImage(mode='RGB'),
                         transforms.Resize(size=size),
                         transforms.RandomResizedCrop(size=size),
                         transforms.RandomHorizontalFlip(),  # with 0.5 probability
                         transforms.RandomApply([color_jitter], p=0.8),
                         transforms.RandomGrayscale(p=0.2),
                         transforms.ToTensor(),
                    ]
                )
            else:
                self.train_transform =  transforms.Compose(
                    [
                         transforms.RandomResizedCrop(size=size),
                         transforms.RandomHorizontalFlip(),  # with 0.5 probability
                         transforms.RandomApply([color_jitter], p=0.8),
                         transforms.RandomGrayscale(p=0.2),
                         transforms.ToTensor(),
                    ]
                )

        self.test_transform =  transforms.Compose(
            [
                 transforms.Resize(size=size),
                 transforms.ToTensor(),
            ]
        )

        self.fine_tune_transform =  transforms.Compose(
            [
                #  transforms.ToPILImage(mode='RGB'),
                 transforms.Resize(size=size),
                 transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x)


class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2D(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias_attr=False, groups=3)
        self.blur_v = nn.Conv2D(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias_attr=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor =transforms.ToTensor()
        # self.tensor_to_pil =transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = paddle.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with paddle.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        # img = self.tensor_to_pil(img)

        return img