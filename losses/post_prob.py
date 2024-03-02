import torch
from torch.nn import Module

class Post_Prob(Module):
    def __init__(self, sigma, c_size, stride, background_ratio, use_background, device):
        """
        Initializes the Post_Prob module.

        Args:
            sigma (float): The sigma value for the distance calculation.
            c_size (int): The size of the crop.
            stride (int): The stride value for the distance calculation.
            background_ratio (float): The ratio of the background distance to the crop size.
            use_background (bool): Whether to include the background distance in the calculation.
            device (torch.device): The device to be used for computation.
        """
        super(Post_Prob, self).__init__()
        assert c_size % stride == 0

        self.sigma = sigma
        self.bg_ratio = background_ratio
        self.device = device
        # coordinate is same to image space, set to constant since crop size is same
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2
        self.cood.unsqueeze_(0)
        self.softmax = torch.nn.Softmax(dim=0)
        self.use_bg = use_background

    def forward(self, points, st_sizes):
        """
        Forward pass of the Post_Prob module.

        Args:
            points (list): A list of tensors containing the coordinates of the points.
            st_sizes (list): A list of tensors containing the sizes of the crops.

        Returns:
            prob_list (list): A list of tensors containing the probability distributions for each point.
        """
        num_points_per_image = [len(points_per_image) for points_per_image in points]
        all_points = torch.cat(points, dim=0)

        if len(all_points) > 0:
            x = all_points[:, 0].unsqueeze_(1)
            y = all_points[:, 1].unsqueeze_(1)
            x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood
            y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
            y_dis.unsqueeze_(2)
            x_dis.unsqueeze_(1)
            dis = y_dis + x_dis
            dis = dis.view((dis.size(0), -1))

            dis_list = torch.split(dis, num_points_per_image)
            prob_list = []
            for dis, st_size in zip(dis_list, st_sizes):
                if len(dis) > 0:
                    if self.use_bg:
                        min_dis = torch.clamp(torch.min(dis, dim=0, keepdim=True)[0], min=0.0)
                        d = st_size * self.bg_ratio
                        bg_dis = (d - torch.sqrt(min_dis))**2
                        dis = torch.cat([dis, bg_dis], 0)  # concatenate background distance to the last
                    dis = -dis / (2.0 * self.sigma ** 2)
                    prob = self.softmax(dis)
                else:
                    prob = None
                prob_list.append(prob)
        else:
            prob_list = []
            for _ in range(len(points)):
                prob_list.append(None)
        return prob_list


