import torch
from torch import linalg as LA
import torch.nn as nn
import torch.nn.functional as F
from models.avsg_moudules import MLP
#########################################################################33
#########################################################################33

def get_agents_decoder(opt, device):

    assert opt.agent_feat_vec_coord_labels == ['centroid_x', 'centroid_y', 'yaw_cos', 'yaw_sin',
                                                'extent_length', 'extent_width', 'speed',
                                                'is_CAR', 'is_CYCLIST', 'is_PEDESTRIAN']
    if opt.agents_decoder_model == 'LSTM':
        return AgentsDecoderLstm(opt, device)
    # if opt.agents_decoder_model == 'GRU':
    #     return AgentsDecoderGRU(opt, device)
    # elif opt.agents_decoder_model == 'GRU_attn':
    #     return AgentsDecoderGRUAttn(opt, device)
    elif opt.agents_decoder_model == 'MLP':
        return AgentsDecoderMLP(opt, device)
    else:
        raise NotImplementedError
#########################################################################################


def project_to_agent_feat(raw_vec):
    assert raw_vec.ndim == 1
    # Project the generator output to the feature vectors domain:
    agent_feat = torch.cat([
        # Coordinates 0,1 are centroid x,y - no need to project
        raw_vec[0:2],
        # Coordinates 2,3 are yaw_cos, yaw_sin - project to unit circle
        raw_vec[2:4] / LA.vector_norm(raw_vec[2:4], ord=2),
        # Coordinates 4,5,6 are extent_length, extent_width, speed project to positive numbers
        F.softplus(raw_vec[4:7]),
        # Coordinates 7,8,9 are one-hot vector - project to 3-simplex
        F.softmax(raw_vec[7:10], dim=0)
    ])
    return agent_feat
#########################################################################################

class AgentsDecoderLstm(nn.Module):

    def __init__(self, opt, device):
        super(AgentsDecoderLstm, self).__init__()
        self.device = device
        self.dim_hid = opt.agents_dec_dim_hid
        self.dim_latent_map = opt.dim_latent_map
        self.dim_in = self.dim_agent_noise + self.dim_latent_map
        self.dim_agent_feat_vec = len(opt.agent_feat_vec_coord_labels)
        self.num_agents = opt.num_agents
        self.n_stacked = opt.agents_dec_n_stacked_rnns
        self.lstm = nn.LSTM(input_size=self.dim_in,
                            proj_size=self.dim_agent_feat_vec,  # output per step dim
                            hidden_size=self.dim_hid,
                            batch_first=True,
                            num_layers=opt.agents_dec_n_stacked_rnns,
                            bias=False)  # disabling the bias gives more variation per step
          # input to self.lstm should be of size [batch_size x num_agents x n_feat]


    def forward(self, map_latent, latent_noise):

        # input to self.lstm should be of size [batch_size x num_agents x n_feat]
        in_seq = torch.cat(map_latent.repeat([1, self.self.num_agents, 1]),
                           latent_noise)
        out = self.lstm(in_seq)
        outs, (hn, cn) = out
        agents_feat_vec_list = []
        for i_agent in range(self.num_agents):
            out_agent = outs[0, i_agent, :]
            # Project the generator output to the feature vectors domain:
            agent_feat = project_to_agent_feat(out_agent)
            agents_feat_vec_list.append(agent_feat)
        agents_feat_vecs = torch.stack(agents_feat_vec_list)
        return agents_feat_vecs
#########################################################################################
# ##############################################################################################

class AgentsDecoderMLP(nn.Module):
    def __init__(self, opt, device):
        super(AgentsDecoderMLP, self).__init__()
        self.device = device
        self.agents_dec_dim_hid = opt.agents_dec_dim_hid

        self.agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels
        self.dim_agent_feat_vec = len(opt.agent_feat_vec_coord_labels)
        self.num_agents = opt.num_agents
        self.dim_latent_map = opt.dim_latent_map
        self.d_in = self.dim_agent_feat_vec * self.num_agents + opt.dim_latent_map
        self.d_out = self.dim_agent_feat_vec * self.num_agents


        self.decoder = MLP(d_in=self.d_in,
                           d_out=self.d_out,
                           d_hid=self.agents_dec_dim_hid,
                           n_layers=4,
                           device=self.device,
                           bias=False)

    def forward(self, map_latent, latent_noise):
        latent_noise_f = torch.flatten(latent_noise)
        in_vec = torch.cat([map_latent, latent_noise_f])
        out_vec = self.decoder(in_vec)

        agents_feat_vec_list = []
        for i_agent in range(self.num_agents):
            output_feat = out_vec[i_agent * self.dim_agent_feat_vec:(i_agent + 1) * self.dim_agent_feat_vec]
            # Project the generator output to the feature vectors domain:
            agent_feat = project_to_agent_feat(output_feat)
            agents_feat_vec_list.append(agent_feat)
        agents_feat_vecs = torch.stack(agents_feat_vec_list)
        return agents_feat_vecs

#########

#########################################################################################

#
# class AgentsDecoderLstm(nn.Module):
#
#     def __init__(self, opt, device):
#         super(AgentsDecoderLstm, self).__init__()
#         self.use_bias_flag = False
#         self.device = device
#         self.agents_dec_dim_hid = opt.agents_dec_dim_hid
#         self.dim_agent_feat_vec = len(opt.agent_feat_vec_coord_labels)
#         self.num_agents = opt.num_agents
#         self.dim_latent_scene = opt.dim_latent_scene
#         self.lstm = nn.LSTM(input_size=self.dim_latent_scene,
#                             batch_first=True,
#                             hidden_size=self.agents_dec_dim_hid,
#                             num_layers=opt.lst_num_layers,
#                             bias=self.use_bias_flag)
#           # input to self.lstm should be of size [batch_size x num_agents x n_feat]
#         self.out_mlp = MLP(d_in=self.dim_latent_scene,
#                            d_out=self.dim_agent_feat_vec,
#                            d_hid=self.dim_latent_scene,
#                            n_layers=opt.agents_dec_out_layers,
#                            device=self.device,
#                            bias=self.use_bias_flag)
#
#     def forward(self, scene_latent, n_agents):
#
#         # input to self.lstm should be of size [batch_size x num_agents x n_feat]
#         in_seq = scene_latent.repeat([1, n_agents, 1])
#         out = self.lstm(in_seq)
#         outs, (hn, cn) = out
#         agents_feat_vec_list = []
#         for i_agent in range(self.num_agents):
#             out_agent = self.out_mlp(outs[0, i_agent, :])
#             # Project the generator output to the feature vectors domain:
#             agent_feat = project_to_agent_feat(out_agent)
#             agents_feat_vec_list.append(agent_feat)
#         agents_feat_vecs = torch.stack(agents_feat_vec_list)
#         return agents_feat_vecs
# #########################################################################################

# class DecoderUnitGru(nn.Module):
#
#     def __init__(self, opt, dim_context, dim_out):
#         super(DecoderUnitGru, self).__init__()
#         dim_hid = dim_context
#         self.device = opt.device
#         self.dim_hid = dim_hid
#         self.dim_out = dim_out
#         self.gru = nn.GRUCell(dim_hid, dim_hid)
#         self.input_mlp = MLP(d_in=dim_hid,
#                              d_out=dim_hid,
#                              d_hid=dim_hid,
#                              n_layers=opt.gru_in_layers,
#                              device=self.device)
#         self.out_mlp = MLP(d_in=dim_hid,
#                            d_out=dim_out,
#                            d_hid=dim_hid,
#                            n_layers=opt.agents_dec_out_layers,
#                            device=self.device)
#
#     def forward(self, context_vec, prev_hidden):
#
#         gru_input = self.input_mlp(context_vec)
#         gru_input = F.relu(gru_input)
#         hidden = self.gru(gru_input.unsqueeze(0), prev_hidden.unsqueeze(0))
#         hidden = hidden[0]
#         output_feat = self.out_mlp(hidden)
#
#         # Project the generator output to the feature vectors domain:
#         agent_feat = project_to_agent_feat(output_feat)
#         return agent_feat, hidden
# #########################################################################################
#
#
# class AgentsDecoderGRU(nn.Module):
#
#     def __init__(self, opt, device):
#         super(AgentsDecoderGRU, self).__init__()
#         self.device = device
#         self.dim_latent_scene = opt.dim_latent_scene
#         self.agents_dec_dim_hid = opt.agents_dec_dim_hid
#         self.dim_agent_feat_vec = len(opt.agent_feat_vec_coord_labels)
#         self.num_agents = opt.num_agents
#         self.decoder_unit = DecoderUnitGru(opt,
#                                            dim_context=self.dim_latent_scene,
#                                            dim_out=self.dim_agent_feat_vec)
#
#     def forward(self, scene_latent, n_agents):
#         prev_hidden = scene_latent
#         agents_feat_vec_list = []
#         for i_agent in range(n_agents):
#             agent_feat, next_hidden = self.decoder_unit(
#                 context_vec=scene_latent,
#                 prev_hidden=prev_hidden)
#             prev_hidden = next_hidden
#             agents_feat_vec_list.append(agent_feat)
#         agents_feat_vecs = torch.stack(agents_feat_vec_list)
#         return agents_feat_vecs
# #########################################################################################
#
#
# class DecoderUnitGRUAttn(nn.Module):
#
#     def __init__(self, opt, dim_context, dim_out):
#         super(DecoderUnitGRUAttn, self).__init__()
#         dim_hid = dim_context
#         self.device = opt.device
#         self.dim_hid = dim_hid
#         self.dim_out = dim_out
#         self.gru = nn.GRUCell(dim_hid, dim_hid)
#         self.input_mlp = MLP(d_in=dim_hid,
#                              d_out=dim_hid,
#                              d_hid=dim_hid,
#                              n_layers=opt.gru_in_layers,
#                              device=self.device)
#         self.out_mlp = MLP(d_in=dim_hid,
#                            d_out=dim_out,
#                            d_hid=dim_hid,
#                            n_layers=opt.agents_dec_out_layers,
#                            device=self.device)
#         self.attn_mlp = MLP(d_in=dim_hid,
#                             d_out=dim_hid,
#                             d_hid=dim_hid,
#                             n_layers=opt.gru_attn_layers,
#                             device=self.device)
#
#     def forward(self, context_vec, prev_hidden):
#         attn_scores = self.attn_mlp(prev_hidden)
#         # the input layer takes in the attention-applied context concatenated with the previous out features
#         attn_weights = F.softmax(attn_scores, dim=0)
#         gru_input = attn_weights * context_vec
#         gru_input = self.input_mlp(gru_input)
#         gru_input = F.relu(gru_input)
#         hidden = self.gru(gru_input.unsqueeze(0), prev_hidden.unsqueeze(0))
#         hidden = hidden[0]
#         output_feat = self.out_mlp(hidden)
#
#         # Project the generator output to the feature vectors domain:
#         agent_feat = project_to_agent_feat(output_feat)
#         return agent_feat, hidden
#
#
# #########################################################################################
#
#
# class AgentsDecoderGRUAttn(nn.Module):
#     # based on:
#     # * Show, Attend and Tell: Neural Image Caption Generation with Visual Attention  https://arxiv.org/abs/1502.03044\
#     # * https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
#     # * https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
#     # * https://towardsdatascience.com/image-captions-with-attention-in-tensorflow-step-by-step-927dad3569fa
#
#     def __init__(self, opt, device):
#         super(AgentsDecoderGRUAttn, self).__init__()
#         self.device = device
#         self.dim_latent_scene = opt.dim_latent_scene
#         self.agents_dec_dim_hid = opt.agents_dec_dim_hid
#         self.agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels
#         self.dim_agent_feat_vec = len(opt.agent_feat_vec_coord_labels)
#         self.num_agents = opt.num_agents
#         self.decoder_unit = DecoderUnitGRUAttn(opt,
#                                                dim_context=self.dim_latent_scene,
#                                                dim_out=self.dim_agent_feat_vec)
#
#     def forward(self, scene_latent, n_agents):
#         prev_hidden = scene_latent
#         agents_feat_vec_list = []
#         for i_agent in range(n_agents):
#             agent_feat, next_hidden = self.decoder_unit(
#                 context_vec=scene_latent,
#                 prev_hidden=prev_hidden)
#             prev_hidden = next_hidden
#             agents_feat_vec_list.append(agent_feat)
#         agents_feat_vecs = torch.stack(agents_feat_vec_list)
#         return agents_feat_vecs
#
#
# # # Sample hard categorical using "Straight-through" , returns one-hot vector
# # stop_flag = F.gumbel_softmax(logits=stop_score, tau=1, hard=True)
# # if i_agent > 0 and stop_flag > 0.5:
# #     # Stop flag is ignored at i=0, since we want at least one agent (including the AV)  in the scene
# #     break
# # else:
# #########################################################################################
