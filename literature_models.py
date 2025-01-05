import torch
import torch.nn as nn
import torch.nn.functional as F

class J_CNN3DModel(nn.Module):
    def __init__(self,input_channels,num_classes):
        super(J_CNN3DModel, self).__init__()
        
        # 1st Conv Layer
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=16, kernel_size=(3, 3, 3), stride=1, padding=0)
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=3)
        
        # 2nd Conv Layer
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(5, 5, 5), stride=1, padding=0)
        self.pool2 = nn.MaxPool3d(kernel_size=(4, 4, 4), stride=4)
        
        # 3rd Conv Layer
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(7, 7, 7), stride=1, padding=0)
        self.pool3 = nn.MaxPool3d(kernel_size=(5, 5, 5), stride=5)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=128, out_features=16)  # Update input size based on input dimensions
        self.fc2 = nn.Linear(in_features=16, out_features=num_classes)
        
    def forward(self, x):
        # 1st Conv Layer
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # 2nd Conv Layer
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # 3rd Conv Layer
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Flatten
        x = torch.flatten(x,start_dim=1)

   
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)



class JAL_model(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(JAL_model, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=4, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(4, momentum=0.9)
        self.dropout1 = nn.Dropout3d(0.5)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv3d(in_channels=4, out_channels=8, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(8, momentum=0.9)
        self.dropout2 = nn.Dropout3d(0.2)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        
        # Fully Connected Layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 45 * 64 * 128, num_classes)  # Adjust input size based on flattened dimensions

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)
        x = self.pool1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.pool2(x)

        # Fully Connected Layer
        x = self.flatten(x)
        x = self.fc(x)

        return x


class J_CNN3DEncoder(nn.Module):
    def __init__(self, input_channels=1,base_channels = 64):
        """
        A 3D CNN that outputs a feature map [B, C_out, D_out, H_out, W_out].
        
        Args:
            input_channels (int): Number of channels in the input (e.g., 1 for grayscale).
        """
        super(J_CNN3DEncoder, self).__init__()
        
        # 1st Conv + Pool
        self.conv1 = nn.Conv3d(
            in_channels=input_channels, 
            out_channels=16, 
            kernel_size=(3, 3, 3),
            stride=1,
            padding=0
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=3)
        
        # 2nd Conv + Pool
        self.conv2 = nn.Conv3d(
            in_channels=16, 
            out_channels=32,
            kernel_size=(5, 5, 5), 
            stride=1, 
            padding=0
        )
        self.pool2 = nn.MaxPool3d(kernel_size=(4, 4, 4), stride=4)
        
        # 3rd Conv + Pool
        self.conv3 = nn.Conv3d(
            in_channels=32, 
            out_channels=base_channels, 
            kernel_size=(7, 7, 7), 
            stride=1, 
            padding=0
        )
        self.pool3 = nn.MaxPool3d(kernel_size=(5, 5, 5), stride=5)

    def forward(self, x):
        """
        Forward pass returning the 3D feature map.
        
        Shape:
            Input:  [B, input_channels, D, H, W]
            Output: [B, 64, D_out, H_out, W_out]  (depending on input size)
        """
        
        # 1st block
        x = F.relu(self.conv1(x))
        x = self.pool1(x)  # -> [B, 16, ...]

        # 2nd block
        x = F.relu(self.conv2(x))
        x = self.pool2(x)  # -> [B, 32, ...]

        # 3rd block
        x = F.relu(self.conv3(x))
        x = self.pool3(x)  # -> [B, 64, ...]

        # Return the feature map without flattening
        return x
        
# Example usage
# model = JAL_model()
# print(model)
