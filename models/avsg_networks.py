
import torch
import torch.nn as nn
from models.avsg_net_moudules import MLP, PointNet
from models.avsg_map_encoder import MapEncoder
from models.avsg_agents_decoder import AgentsDecoderGRU, AgentsDecoderMLP

##############################################################################################

class SceneGenerator(nn.Module):

    def __init__(self, opt):
        super(SceneGenerator, self).__init__()
        self.device = opt.device
        self.dim_latent_scene_noise = opt.dim_latent_scene_noise
        self.dim_latent_scene = opt.dim_latent_scene
        self.dim_latent_map = opt.dim_latent_map
        self.map_enc = MapEncoder(opt)
        self.scene_embedder_out = MLP(d_in=self.dim_latent_scene_noise + self.dim_latent_map,
                                      d_out=self.dim_latent_scene,
                                      d_hid=self.dim_latent_scene,
                                      n_layers=opt.n_layers_scene_embedder_out,
                                      device=self.device)
        if opt.agents_decoder_model == 'GRU':
            self.agents_dec = AgentsDecoderGRU(opt, self.device)
        elif opt.agents_decoder_model == 'MLP':
            self.agents_dec = AgentsDecoderMLP(opt, self.device)
        else:
            raise ValueError
        # Debug - print parameter names:  [x[0] for x in self.named_parameters()]
        self.batch_size = opt.batch_size
        if self.batch_size != 1:
            raise NotImplementedError

    def forward(self, conditioning):
        """Standard forward"""
        map_feat = conditioning['map_feat']
        n_agents = conditioning['n_agents']
        map_latent = self.map_enc(map_feat)
        latent_noise_std = torch.std(map_latent).detach()
        latent_noise = torch.randn(self.dim_latent_scene_noise, device=self.device) * latent_noise_std
        scene_latent = torch.concat([map_latent, latent_noise], dim=0)
        scene_latent = self.scene_embedder_out(scene_latent)
        agents_feat_vec_list = self.agents_dec(scene_latent, n_agents)
        agents_feat_vecs = torch.stack(agents_feat_vec_list)
        return agents_feat_vecs


#########################################################################################


class SceneDiscriminator(nn.Module):

    def __init__(self, opt):
        super(SceneDiscriminator, self).__init__()
        self.device = opt.device
        self.agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels
        self.dim_agent_feat_vec = len(opt.agent_feat_vec_coord_labels)
        self.dim_latent_all_agents = opt.dim_latent_all_agents
        self.dim_latent_map = opt.dim_latent_map
        self.dim_latent_scene = opt.dim_latent_scene
        self.map_enc = MapEncoder(opt)
        self.agents_enc = PointNet(d_in=self.dim_agent_feat_vec,
                                   d_out=self.dim_latent_all_agents,
                                   d_hid=self.dim_latent_all_agents,
                                   n_layers=opt.n_discr_pointnet_layers,
                                   device=self.device)
        self.out_mlp = MLP(d_in=self.dim_latent_map + self.dim_latent_all_agents,
                           d_out=1,
                           d_hid=self.dim_latent_scene,
                           n_layers=opt.n_discr_out_mlp_layers,
                           device=self.device)

    def forward(self, conditioning, agents_feat_vecs):
        """Standard forward."""
        map_feat = conditioning['map_feat']
        map_latent = self.map_enc(map_feat)
        agents_latent = self.agents_enc(agents_feat_vecs)
        scene_latent = torch.cat([map_latent, agents_latent])
        pred_fake = self.out_mlp(scene_latent)
        ''' 
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        '''
        return pred_fake
#########################################################################################
