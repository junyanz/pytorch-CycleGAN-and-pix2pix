import numpy as np
import torch


#########################################################################################


def agents_feat_vecs_to_dicts(agents_feat_vecs):
    agents_feat_dicts = []
    n_agents = agents_feat_vecs.shape[0]
    for i_agent in range(n_agents):
        agent_feat_vec = agents_feat_vecs[i_agent]
        agent_feat_dict = ({'centroid': agent_feat_vec[:2],
                            'yaw': torch.atan2(agent_feat_vec[3], agent_feat_vec[2]),
                            'extent': agent_feat_vec[4:6],
                            'speed': agent_feat_vec[6],
                            'agent_label_id': torch.argmax(agent_feat_vec[7:10])})
        for key in agent_feat_dict.keys():
            agent_feat_dict[key] = agent_feat_dict[key].detach().cpu().numpy()
        agents_feat_dicts.append(agent_feat_dict)
    return agents_feat_dicts


#########################################################################################


def agents_feat_dicts_to_vecs(agent_feat_vec_coord_labels, agents_feat_dicts, device):
    dim_agent_feat_vec = len(agent_feat_vec_coord_labels)
    assert agent_feat_vec_coord_labels == ['centroid_x', 'centroid_y', 'yaw_cos', 'yaw_sin',
                                           'extent_length', 'extent_width', 'speed',
                                           'is_CAR', 'is_CYCLIST', 'is_PEDESTRIAN']
    agents_feat_vecs = []
    for agent_dict in agents_feat_dicts:
        agent_feat_vec = torch.zeros(dim_agent_feat_vec, device=device)
        assert agent_dict['centroid'].shape == torch.Size([1, 2])
        agent_feat_vec[0] = agent_dict['centroid'][0, 0]
        agent_feat_vec[1] = agent_dict['centroid'][0, 1]
        agent_feat_vec[2] = torch.cos(agent_dict['yaw'])
        agent_feat_vec[3] = torch.sin(agent_dict['yaw'])
        assert agent_dict['extent'].shape == torch.Size([1, 2])
        agent_feat_vec[4] = agent_dict['extent'][0, 0]
        agent_feat_vec[5] = agent_dict['extent'][0, 1]
        agent_feat_vec[6] = agent_dict['speed']
        # agent type ['CAR', 'CYCLIST', 'PEDESTRIAN'] is represented in one-hot encoding
        agent_feat_vec[7] = agent_dict['agent_label_id'] == 0
        agent_feat_vec[8] = agent_dict['agent_label_id'] == 1
        agent_feat_vec[9] = agent_dict['agent_label_id'] == 2
        assert agent_feat_vec[7:].sum() == 1
        agents_feat_vecs.append(agent_feat_vec)
    agents_feat_vecs = torch.stack(agents_feat_vecs)
    return agents_feat_vecs


#########################################################################################


def pre_process_scene_data(scene_data, opt):
    num_agents = opt.num_agents
    agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels
    polygon_name_order = opt.polygon_name_order
    device = opt.device
    # We assume this order of coordinates:
    assert agent_feat_vec_coord_labels == ['centroid_x', 'centroid_y',
                                           'yaw_cos', 'yaw_sin',
                                           'extent_length', 'extent_width', 'speed',
                                           'is_CAR', 'is_CYCLIST', 'is_PEDESTRIAN']

    agents_feat_vecs = filter_and_preprocess_agent_feat(
        scene_data['agents_feat'],
        num_agents, agent_feat_vec_coord_labels,
        device)

    # Map features - Move to device
    map_feat = dict()
    for poly_type in polygon_name_order:
        map_feat[poly_type] = []
        poly_elems = scene_data['map_feat'][poly_type]
        map_feat[poly_type] = [poly_elem.to(device) for poly_elem in poly_elems]

    if opt.augmentation_type == 'none':
        pass
    elif opt.augmentation_type == 'rotate_and_translate':
        # --------------------------------------
        # Random augmentation: rotation & translation
        # --------------------------------------
        aug_rot = np.random.rand(1).squeeze() * 2 * np.pi
        rot_mat = np.array([[np.cos(aug_rot), -np.sin(aug_rot)],
                            [np.sin(aug_rot), np.cos(aug_rot)]])
        rot_mat = torch.from_numpy(rot_mat).to(device=device, dtype=torch.float32)

        pos_shift_std = 50  # [m]
        pos_shift = torch.rand(2, device=device) * pos_shift_std

        for ag in agents_feat_vecs:
            # Rotate the centroid (x,y)
            ag[:2] = rot_mat @ ag[:2]
            # Rotate the yaw angle (in unit vec form)
            ag[2:4] = rot_mat @ ag[2:4]
            # Translate centroid
            ag[:2] += pos_shift

        for poly_type in map_feat.keys():
            for i_elem, poly_elem in enumerate(map_feat[poly_type]):
                for i_point in range(poly_elem.shape[1]):
                    poly_elem[0, i_point, :] = rot_mat @ poly_elem[0, i_point, :]
                map_feat[poly_type][i_elem] = poly_elem
                map_feat[poly_type][i_elem] += pos_shift
    elif opt.augmentation_type == 'Gaussian_data':
        # Replace all the agent features data to gaussian samples... for debug
        agents_feat_vecs = agents_feat_vecs * 0 + torch.randn_like(agents_feat_vecs)
        # Set zero to all map features
        for poly_type in map_feat.keys():
            for i_elem, poly_elem in enumerate(map_feat[poly_type]):
                map_feat[poly_type][i_elem] *= 0.
        ####### Debug ###
        # import matplotlib.pyplot as plt
        # n_bins = 10
        # hist, bin_edges = torch.histogram(agents_feat_vecs, bins=n_bins)
        # plt.bar(0.5 * (bin_edges[1:] + bin_edges[:-1]), hist, align='center')
        # plt.xlabel('Bins')
        # plt.ylabel('Frequency')
        # plt.show()
        ####### Debug ###

    else:
        raise NotImplementedError(f'Unrecognized opt.augmentation_type  {opt.augmentation_type}')
    real_agents = agents_feat_vecs
    conditioning = {'map_feat': map_feat, 'n_agents': opt.num_agents}
    return real_agents, map_feat, conditioning
#########################################################################################


#########################################################################################


def filter_and_preprocess_agent_feat(agent_feat, num_agents, agent_feat_vec_coord_labels, device):
    # We assume this order of coordinates:
    assert agent_feat_vec_coord_labels == ['centroid_x', 'centroid_y',
                                           'yaw_cos', 'yaw_sin',
                                           'extent_length', 'extent_width', 'speed',
                                           'is_CAR', 'is_CYCLIST', 'is_PEDESTRIAN']

    # --------------------------------------
    # Filter out the selected agents
    # --------------------------------------

    # Agent features -
    agent_dists_to_ego = [np.linalg.norm(agent_dict['centroid'][0, :]) for agent_dict in agent_feat]

    # Change to vector form, Move to device
    agents_feat_vecs = agents_feat_dicts_to_vecs(agent_feat_vec_coord_labels,
                                                 agent_feat,
                                                 device)
    agents_dists_order = np.argsort(agent_dists_to_ego)

    agents_inds = agents_dists_order[:num_agents]  # take the closest agent to the ego
    np.random.shuffle(agents_inds)  # shuffle so that the ego won't always be first
    agents_feat_vecs = agents_feat_vecs[agents_inds]

    return agents_feat_vecs


#########################################################################################


def get_agents_descriptions(agents_feat_dicts):
    txt_descript = []
    for i, ag in enumerate(agents_feat_dicts):
        x, y = ag['centroid']
        yaw_deg = np.degrees(ag['yaw'])
        length, width = ag['extent']
        if ag['agent_label_id'] == 0:
            type_label = 'Car'
        elif ag['agent_label_id'] == 1:
            type_label = 'Cyclist'
        elif ag['agent_label_id'] == 2:
            type_label = 'Pedestrian'
        else:
            raise ValueError
        txt_descript.append(
            f"#{i}, {type_label}, ({x:.1f},{y:.1f}), {yaw_deg:.1f}\u00B0, {length:.1f}\u00D7{width:.1f}")
    return txt_descript


#########################################################################################

def calc_agents_feats_stats(dataset, agent_feat_vec_coord_labels, device, num_agents, polygon_name_order):
    ##### Find data normalization parameters

    dim_agent_feat_vec = len(agent_feat_vec_coord_labels)
    sum_agent_feat = torch.zeros(dim_agent_feat_vec, device=device)
    count = 0
    for scene in dataset:
        agents_feat_dict = scene['agents_feat']
        is_valid, agents_feat_vec = filter_and_preprocess_agent_feat(agents_feat_dict, num_agents, agent_feat_vec_coord_labels, device)
        if is_valid:
            sum_agent_feat += agents_feat_vec.sum(dim=0)  # sum all agents in the scene
            count += agents_feat_vec.shape[0]  # count num agents summed
    agent_feat_mean = sum_agent_feat / count  # avg across all agents in all scenes

    count = 0
    sum_sqr_div_agent_feat = torch.zeros(dim_agent_feat_vec, device=device)
    for scene in dataset:
        agents_feat_dict = scene['agents_feat']
        is_valid, agents_feat_vec = filter_and_preprocess_agent_feat(agents_feat_dict, num_agents, agent_feat_vec_coord_labels, device)
        if is_valid:
            count += agents_feat_vec.shape[0]  # count num agents summed
            sum_sqr_div_agent_feat += torch.sum(
                torch.pow(agents_feat_vec - agent_feat_mean, 2), dim=0)  # sum all agents in the scene
    agent_feat_std = torch.sqrt(sum_sqr_div_agent_feat / count)

    return agent_feat_mean, agent_feat_std
    #########################################################################################
    #
    #
    # def get_normalized_agent_feat(self, feat):
    #     nrm_feat = torch.clone(feat)
    #     nrm_feat[:, self.agent_feat_to_nrm] -= self.agent_feat_mean[self.agent_feat_to_nrm]
    #     nrm_feat[:, self.agent_feat_to_nrm] /= self.agent_feat_std[self.agent_feat_to_nrm]
    #     return nrm_feat
    #########################################################################################


