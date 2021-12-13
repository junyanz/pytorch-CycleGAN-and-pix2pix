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


def pre_process_scene_data(scene_data, num_agents, agent_feat_vec_coord_labels, polygon_name_order, device):
    # if there are too few agents in the scene - skip it
    if len(scene_data['agents_feat']) < num_agents:
        return False, None, None

    # Map features - Move to device
    map_feat = dict()
    for poly_type in polygon_name_order:
        map_feat[poly_type] = []
        poly_elems = scene_data['map_feat'][poly_type]
        map_feat[poly_type] = [poly_elem.to(device) for poly_elem in poly_elems]

    # Agent features -
    agent_dists_to_ego = [np.linalg.norm(agent_dict['centroid'][0, :]) for agent_dict in scene_data['agents_feat']]

    # Change to vector form, Move to device
    agents_feat_vecs = agents_feat_dicts_to_vecs(agent_feat_vec_coord_labels,
                                                 scene_data['agents_feat'],
                                                 device)
    agents_dists_order = np.argsort(agent_dists_to_ego)

    # n_agents = np.random.randint(low=self.min_num_agents, high=self.max_num_agents+1)
    n_agents = num_agents
    agents_feat_vecs = agents_feat_vecs[agents_dists_order[:n_agents]]
    conditioning = {'map_feat': map_feat, 'n_agents': n_agents}
    real_agents = agents_feat_vecs
    return True, real_agents, conditioning
