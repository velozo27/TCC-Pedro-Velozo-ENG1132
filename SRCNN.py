import torch.nn as nn
import torch


class SRCNN(nn.Module):
    def __init__(self, f2=1):
        super(SRCNN, self).__init__()
        padding = [2,2,2]
        if f2 == 5:
            padding = [9 // 2, 5 // 2, 5 // 2]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=padding[0])
        self.conv2 = nn.Conv2d(64, 32, kernel_size=f2, padding=padding[1])
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=padding[2])

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# SRCNN Original que estavamos usando inicialmente no projeto
# class SRCNN(nn.Module):
#     def __init__(self, num_channels=3, use_padding=True, initialization='normal'):
#         super(SRCNN, self).__init__()
#         self.initialization = initialization
#         self.using_padding = use_padding
#         padding = [0, 0, 0]
#         if use_padding:
#             padding = [9 // 2, 5 // 2, 5 // 2]
#         # self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=padding[0])
#         # self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=padding[1])
#         # self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=padding[2])

#         self.conv1 = nn.Conv2d(
#             3, 64, kernel_size=9, padding=(2, 2)
#         )
#         self.conv2 = nn.Conv2d(
#             64, 32, kernel_size=1, padding=(2, 2)
#         )
#         self.conv3 = nn.Conv2d(
#             32, 3, kernel_size=5, padding=(2, 2)
#         )

#         self.relu = nn.ReLU(inplace=True)
        
#         # self.apply(self._init_weights)
#         # self._init_weights()

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.conv3(x)
#         return x
    
#     def _init_weights(self):
#       # The filter weights of each layer are initialized by drawing randomly 
#       # from a Gaussian distribution with zero mean and standard deviation 0.001 (and 0 for biases)
#       for module in self.modules():
#         if isinstance(module, nn.Conv2d):
#           if self.initialization == "normal":
#             module.weight.data.normal_(mean=0.0, std=0.001)
#           elif self.initialization == "xavier":
#             nn.init.xavier_normal_(module.weight.data)
#           else:
#             nn.init.eye_(module.weight.data)

#           if module.bias is not None:
#               module.bias.data.zero_()


# # class SRCNN(nn.Module):
# #     def __init__(self, num_channels=3, use_padding=True, initialization='normal'):
# #         super(SRCNN, self).__init__()
# #         self.initialization = initialization
# #         self.using_padding = use_padding
# #         padding = [0, 0, 0]
# #         if use_padding:
# #             padding = [9 // 2, 5 // 2, 5 // 2]
# #         self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=padding[0])
# #         self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=padding[1])
# #         self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=padding[2])
# #         self.relu = nn.ReLU(inplace=True)
        
# #         # self.apply(self._init_weights)
# #         # self._init_weights()

# #     def forward(self, x):
# #         x = self.relu(self.conv1(x))
# #         x = self.relu(self.conv2(x))
# #         x = self.conv3(x)
# #         return x
    
# #     def _init_weights(self):
# #       # The filter weights of each layer are initialized by drawing randomly 
# #       # from a Gaussian distribution with zero mean and standard deviation 0.001 (and 0 for biases)
# #       for module in self.modules():
# #         if isinstance(module, nn.Conv2d):
# #           if self.initialization == "normal":
# #             module.weight.data.normal_(mean=0.0, std=0.001)
# #           elif self.initialization == "xavier":
# #             nn.init.xavier_normal_(module.weight.data)
# #           else:
# #             nn.init.eye_(module.weight.data)

# #           if module.bias is not None:
# #               module.bias.data.zero_()
