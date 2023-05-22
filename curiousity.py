import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class ICMModel(nn.Module):
    def __init__(self, state_size, action_size, feature_size, model_path, num):
        super(ICMModel, self).__init__()

        self.device = torch.device('cpu')
        self.model_path = model_path + '/icm'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.feature = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, feature_size)
        ).double()

        self.inverse_net = nn.Sequential(
            nn.Linear(feature_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        ).double()

        self.forward_net = nn.Sequential(
            nn.Linear(feature_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_size)
        ).double()
        self.n = int(num)
        if os.path.exists(self.model_path + '/' + num + '_actor_params.pkl'):
            self._load_model()

    def forward(self, inputs):
        state, next_state, action = inputs

        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action

        pred_action = torch.cat((encode_state, encode_next_state))

        pred_action = self.inverse_net(pred_action)

        pred_next_state_feature_orig = torch.cat((encode_state, 0.01*action))
        pred_next_state_feature = self.forward_net(pred_next_state_feature_orig)
        return encode_next_state, pred_next_state_feature, pred_action

    def intrinsic(self, s, s_, a):
        encode_next_state, pred_next_state_feature, pred_action = self.forward([s, s_, a])
        intrinsic_reward = 10*F.mse_loss(encode_next_state, pred_next_state_feature, reduction='none').mean(-1)
        inverse_loss = 0.001 * F.mse_loss(pred_action, a.double(), reduction='none').mean(-1)
        return intrinsic_reward, inverse_loss

    def save_model(self):
        self.n += 1
        num = str(self.n)
        torch.save(self.feature.state_dict(), self.model_path + '/' + num + '_feature.pkl')
        torch.save(self.inverse_net.state_dict(), self.model_path + '/' + num + '_inverse.pkl')
        torch.save(self.forward_net.state_dict(), self.model_path + '/' + num + '_forward.pkl')

    def _load_model(self):
        num = str(self.n)
        self.feature.load_state_dict(torch.load(self.model_path + '/' + num + '_feature.pkl'))
        self.inverse_net.load_state_dict(torch.load(self.model_path + '/' + num + '_inverse.pkl'))
        self.forward_net.load_state_dict(torch.load(self.model_path + '/' + num + '_forward.pkl'))
