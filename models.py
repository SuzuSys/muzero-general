import math
from abc import ABC, abstractmethod

import torch


class MuZeroNetwork:
    def __new__(cls, config):
        if config.network == "fullyconnected":
            return MuZeroFullyConnectedNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,
            )
        elif config.network == "resnet":
            return MuZeroResidualNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.blocks,
                config.channels,
                config.reduced_channels_reward,
                config.reduced_channels_value,
                config.reduced_channels_policy,
                config.resnet_fc_reward_layers,
                config.resnet_fc_value_layers,
                config.resnet_fc_policy_layers,
                config.support_size,
                config.downsample,
                # ADDED ------------------------------------------------------------------------------
                config.reduced_channels_choice,
                config.resnet_fc_choice_layers,
                config.num_choice
                # ------------------------------------------------------------------------------------
            )
        else:
            raise NotImplementedError(
                'The network parameter should be "fullyconnected" or "resnet".'
            )


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


##################################
######## Fully Connected #########


class MuZeroFullyConnectedNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1

        self.representation_network = torch.nn.DataParallel(
            mlp(
                observation_shape[0]
                * observation_shape[1]
                * observation_shape[2]
                * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],
                fc_representation_layers,
                encoding_size,
            )
        )

        self.dynamics_encoded_state_network = torch.nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
            )
        )
        self.dynamics_reward_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_reward_layers, self.full_support_size)
        )

        self.prediction_policy_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_policy_layers, self.action_space_size)
        )
        self.prediction_value_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_value_layers, self.full_support_size)
        )

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)

        reward = self.dynamics_reward_network(next_encoded_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


###### End Fully Connected #######
##################################


##################################
############# ResNet #############


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )
# Added ------------------------------------------------------------------------------------
def conv3(in_channels, out_channels):
    return torch.nn.Conv1d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    )
# ------------------------------------------------------------------------------------------


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        # FIXED ----------------------------------------------------------------------------
        self.conv1 = conv3(num_channels, num_channels)
        self.bn1 = torch.nn.BatchNorm1d(num_channels)
        self.conv2 = conv3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm1d(num_channels)
        # ----------------------------------------------------------------------------------

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = torch.nn.functional.relu(out)
        return out


# Downsample observations before representation network (See paper appendix Network Architecture)
class DownSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.resblocks1 = torch.nn.ModuleList(
            [ResidualBlock(out_channels // 2) for _ in range(2)]
        )
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.resblocks2 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.conv2(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


class DownsampleCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, h_w):
        super().__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, mid_channels, kernel_size=h_w[0] * 2, stride=4, padding=2
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(h_w)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x


class RepresentationNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        num_blocks,
        num_channels,
        downsample, # FALSE
    ):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            if self.downsample == "resnet":
                self.downsample_net = DownSample(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                )
            elif self.downsample == "CNN":
                self.downsample_net = DownsampleCNN(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                    (
                        math.ceil(observation_shape[1] / 16),
                        math.ceil(observation_shape[2] / 16),
                    ),
                )
            else:
                raise NotImplementedError('downsample should be "resnet" or "CNN".')
        # FIXED -----------------------------------------------------------------------------------
        # backgammon's obs shape: (batch, 9, 14, 8)
        self.conv = torch.nn.Conv2d(
            observation_shape[0] * (stacked_observations + 1) + stacked_observations,
            num_channels,
            kernel_size=(3, observation_shape[2]),
            stride=1,
            padding=(1,0),
            bias=False
        )
        # self.conv's output: (batch, num_channels, 14, 1)
        self.bn = torch.nn.BatchNorm1d(num_channels)
        # -----------------------------------------------------------------------------------------
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            # backgammon's obs shape: (batch, 9, 14, 8)
            x = self.conv(x)[:,:,:,0]
            # x's shape: (batch, 256, 14)
            x = self.bn(x)
            x = torch.nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x


class DynamicsNetwork(torch.nn.Module):
    def __init__(
        self,
        num_blocks,
        num_channels,
        increase_channels,
        reduced_channels_reward,
        fc_reward_layers,
        block_output_size_reward,
    ):
        super().__init__()
        # FIXED ------------------------------------------------------------------------------------
        self.conv = conv3(num_channels + increase_channels, num_channels)
        self.bn = torch.nn.BatchNorm1d(num_channels)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )
        # reward process was deleted!!!
        # ------------------------------------------------------------------------------------------

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.relu(x)
        for block in self.resblocks:
            x = block(x)
        state = x
        # FIXED ------------------------------------------------------------------------------------
        reward = torch.zeros(len(x), 1).to(x.device)
        # ------------------------------------------------------------------------------------------
        return state, reward


class PredictionNetwork(torch.nn.Module):
    def __init__(
        self,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
        # ADDED ---------------------------------------------------------------
        reduced_channels_choice,
        block_output_size_choice,
        fc_choice_layers,
        num_choice
        # ----------------------------------------------------------------------
    ):
        super().__init__()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.block_output_size_choice = block_output_size_choice
        # FIXED --------------------------------------------------------------------------------------
        self.conv1_value = torch.nn.Conv1d(num_channels, reduced_channels_value, kernel_size=1)
        self.conv1_policy = torch.nn.Conv1d(num_channels, reduced_channels_policy, kernel_size=1)
        self.conv1_choice = torch.nn.Conv1d(num_channels, reduced_channels_choice, kernel_size=1)
        self.fc_value = mlp(
            self.block_output_size_value, 
            fc_value_layers, 
            1,
            torch.nn.Sigmoid
        )
        # --------------------------------------------------------------------------------------------
        self.fc_policy = mlp(
            self.block_output_size_policy,
            fc_policy_layers,
            action_space_size,
        )
        # ADDED --------------------------------------------------------------------------------------
        self.fc_choice = mlp(
            self.block_output_size_choice,
            fc_choice_layers,
            num_choice
        )
        # --------------------------------------------------------------------------------------------

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.conv1_value(x)
        policy = self.conv1_policy(x)
        choice = self.conv1_choice(x) # ADDED --------------------------------------------------------
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        choice = choice.view(-1, self.block_output_size_choice) # ADDED ------------------------------
        # FIXED -----------------------------------------------------------------------------------------
        value = self.fc_value(value) # tensor to scalar*batch [0, 1] (tensor type)
        # -----------------------------------------------------------------------------------------------
        policy = self.fc_policy(policy)
        choice = self.fc_choice(choice)
        return policy, value, choice


class MuZeroResidualNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        reduced_channels_value,
        reduced_channels_policy,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        support_size,
        downsample,
        # ADDED --------------------------------------------------------------------------
        reduced_channels_choice,
        fc_choice_layers,
        num_choice,
        # --------------------------------------------------------------------------------
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        # block_output_size_reward = (
        #    (
        #        reduced_channels_reward
        #        * math.ceil(observation_shape[1] / 16)
        #        * math.ceil(observation_shape[2] / 16)
        #    )
        #    if downsample
        #    else (reduced_channels_reward * observation_shape[1] * observation_shape[2])
        #)

        self.num_choice = num_choice # ADDED ---------------------------------------------

        block_output_size_value = (
            (
                reduced_channels_value
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            # FIXED ---------------------------------------------------------------------------------------------
            else (reduced_channels_value * observation_shape[1])
            # ---------------------------------------------------------------------------------------------------
        )

        block_output_size_policy = (
            (
                reduced_channels_policy
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            # FIXED ---------------------------------------------------------------------------------------------
            else (reduced_channels_policy * observation_shape[1])
            # ---------------------------------------------------------------------------------------------------
        )

        # ADDED -------------------------------------------------------------------------------------------------
        block_output_size_choice = reduced_channels_choice * observation_shape[1]
        # -------------------------------------------------------------------------------------------------------

        self.representation_network = torch.nn.DataParallel(
            RepresentationNetwork(
                observation_shape,
                stacked_observations,
                num_blocks,
                num_channels,
                downsample,
            )
        )

        self.dynamics_network = torch.nn.DataParallel(
            DynamicsNetwork(
                num_blocks,
                num_channels, # FIXED ------------------------------------------------------
                2 + self.num_choice, # FIXED -----------------------------------------------
                reduced_channels_reward,
                fc_reward_layers,
                None,
            )
        )

        self.prediction_network = torch.nn.DataParallel(
            PredictionNetwork(
                action_space_size,
                num_blocks,
                num_channels,
                reduced_channels_value,
                reduced_channels_policy,
                fc_value_layers,
                fc_policy_layers,
                self.full_support_size,
                block_output_size_value,
                block_output_size_policy,
                # ADDED --------------------------------------------------------------------------
                reduced_channels_choice,
                block_output_size_choice,
                fc_choice_layers,
                num_choice
                # --------------------------------------------------------------------------------
            )
        )

    def prediction(self, encoded_state):
        policy, value, choice = self.prediction_network(encoded_state)
        return policy, value, choice

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        # encoded_state's shape: (batch, channels, 14)
        # Scale encoded state between [0, 1] (See appendix paper Training)
        # FIXED ----------------------------------------------------------------------------------------
        min_encoded_state = encoded_state.min(2, keepdim=True)[0] # shape: (batch, channels, 1)
        max_encoded_state = encoded_state.max(2, keepdim=True)[0] # shape: (batch, channels, 1)
        # ----------------------------------------------------------------------------------------------
        scale_encoded_state = max_encoded_state - min_encoded_state # shape: (batch, channels, 1)
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized # (batch, channels, 14)

    def dynamics(self, encoded_state, action, choice):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        # encoded_state's shape: (batch, channels, 14)
        # action's shape: (batch, 1)
        # choice's shape: (batch, 1)
        # FIXED ---------------------------------------------------------------------------
        points = encoded_state.shape[2]
        action_one_hot = (
            torch.zeros((action.shape[0], 2*points+1)).to(action.device) # (batch, 29)
            .scatter(dim=1, index=action, value=1)[:,1:] # (batch, 28)
            .view((action.shape[0], 2, points)) # (batch, 2, 14)
        )
        choice_one_hot = (
            torch.zeros((choice.shape[0], self.num_choice)).to(action.device) # (batch, num_choice)
            .scatter(dim=1, index=choice, value=1) # (batch, num_choice)
            .unsqueeze(2) # (batch, num_choice, 1)
            .expand(-1, -1, points) # (batch, num_choice, 14)
        )
        x = torch.cat((encoded_state, action_one_hot, choice_one_hot), dim=1) # shape: (batch, channels+2+num_choice, 14)
        # -----------------------------------------------------------------------------------
        next_encoded_state, reward = self.dynamics_network(x)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        # FIXED ----------------------------------------------------------------------------
        min_next_encoded_state = next_encoded_state.min(2, keepdim=True)[0] # shape: (batch, channels, 1)
        max_next_encoded_state = next_encoded_state.max(2, keepdim=True)[0] # shape: (batch, channels, 1)
        # ----------------------------------------------------------------------------------
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state # shape: (batch, channels, 1)
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state
        return next_encoded_state_normalized, reward
    # representation and prediction
    def initial_inference(self, observation):
        # observation shape: (batch, channels, 14)
        encoded_state = self.representation(observation)
        policy_logits, value, choice_logits = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        # FIXED --------------------------------------------------------------------------------------
        reward = torch.zeros(len(observation),1).to(observation.device)
        # --------------------------------------------------------------------------------------------
        return (
            value, # scalar
            reward, # scalar
            policy_logits, # (1, 29)
            encoded_state,
            choice_logits # (1,10) # ADDED --------------------------------------------------------------------
        )
    # dynamics and prediction
    def recurrent_inference(self, encoded_state, action, choice):
        next_encoded_state, reward = self.dynamics(encoded_state, action, choice)
        policy_logits, value, choice_logits = self.prediction(next_encoded_state)
        return (
            value, 
            reward, 
            policy_logits, 
            next_encoded_state, 
            choice_logits
        )

########### End ResNet ###########
##################################


def mlp(
    input_size, # scalar
    layer_sizes, # [scalar, scalar, scalar]
    output_size, # scalar
    output_activation=torch.nn.Identity, # 恒等関数
    activation=torch.nn.ELU, # reluをなめらかにしたもの
):
    sizes = [input_size] + layer_sizes + [output_size] # [scalar, scalar, scalar, scalar, scalar]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)

