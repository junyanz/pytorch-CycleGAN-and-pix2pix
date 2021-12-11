
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
