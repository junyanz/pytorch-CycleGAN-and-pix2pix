import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
import wandb
from avsg_utils import agents_feat_vecs_to_dicts, pre_process_scene_data, get_agents_descriptions
from models.networks import cal_gradient_penalty

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


######################################################################


def plot_poly_elems(ax, poly, facecolor='0.4', alpha=0.3, edgecolor='black', label='', is_closed=False, linewidth=1):
    first_plt = True
    for elem in poly:
        x = elem[0, :, 0].detach().cpu()
        y = elem[0, :, 1].detach().cpu()
        if first_plt:
            first_plt = False
        else:
            label = None
        if is_closed:
            ax.fill(x, y, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth, label=label)
        else:
            ax.plot(x, y, alpha=alpha, color=edgecolor, linewidth=linewidth, label=label)


##############################################################################################


def plot_lanes(ax, left_lanes, right_lanes, facecolor='0.4', alpha=0.3, edgecolor='black', label='', linewidth=1):
    # assert len(left_lanes) == len(right_lanes)
    n_elems = min(len(left_lanes), len(right_lanes))
    first_plt = True
    for i in range(n_elems):
        x_left = left_lanes[i][0, :, 0]
        y_left = left_lanes[i][0, :, 1]
        x_right = right_lanes[i][0, :, 0]
        y_right = right_lanes[i][0, :, 1]
        x = torch.cat((x_left, torch.flip(x_right, [0]))).detach().cpu()
        y = torch.cat((y_left, torch.flip(y_right, [0]))).detach().cpu()
        if first_plt:
            first_plt = False
        else:
            label = None
        ax.fill(x, y, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth, label=label)


##############################################################################################


def plot_rectangles(ax, centroids, extents, yaws, label, facecolor, alpha=0.7, edgecolor='black'):
    n_elems = len(centroids)
    first_plt = True
    for i in range(n_elems):
        if first_plt:
            first_plt = False
        else:
            label = None
        height = extents[i][0]
        width = extents[i][1]
        angle = yaws[i]
        angle_deg = float(np.degrees(angle))
        xy = centroids[i] \
             - 0.5 * height * np.array([np.cos(angle), np.sin(angle)]) \
             - 0.5 * width * np.array([-np.sin(angle), np.cos(angle)])
        rect = Rectangle(xy, height, width, angle_deg, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor, linewidth=1, label=label)

        ax.add_patch(rect)


##############################################################################################


def visualize_scene_feat(agents_feat, map_feat):
    centroids = np.stack([af['centroid'] for af in agents_feat])
    yaws = np.stack([af['yaw'] for af in agents_feat])
    speed = np.stack([af['speed'] for af in agents_feat])
    # print('agents types: ', [af['agent_label_id'] for af in agents_feat])
    X = centroids[:, 0]
    Y = centroids[:, 1]
    U = speed * np.cos(yaws)
    V = speed * np.sin(yaws)

    fig, ax = plt.subplots()
    plot_lanes(ax, map_feat['lanes_left'], map_feat['lanes_right'], facecolor='grey', alpha=0.3, edgecolor='black',
               label='Lanes')
    plot_poly_elems(ax, map_feat['lanes_mid'], facecolor='lime', alpha=0.4, edgecolor='lime', label='Lanes mid',
                    is_closed=False, linewidth=1)
    plot_poly_elems(ax, map_feat['crosswalks'], facecolor='orange', alpha=0.3, edgecolor='orange', label='Crosswalks',
                    is_closed=True)

    n_agents = len(agents_feat)
    if n_agents > 0:
        extents = [af['extent'] for af in agents_feat]
        plot_rectangles(ax, centroids[1:], extents[1:], yaws[1:], label='non-ego', facecolor='saddlebrown')
        plot_rectangles(ax, [centroids[0]], [extents[0]], [yaws[0]], label='ego', facecolor='red')
        valid = speed > 1e-4
        if valid.any():
            ax.quiver(X[valid], Y[valid], U[valid], V[valid], units='xy', color='black', width=0.5)
    ax.grid()
    plt.legend()
    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image


##############################################################################################


def get_metrics_stats_and_images(model, train_dataset, eval_dataset, opt, i_epoch, epoch_iter, total_iters):
    """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""

    datasets = {'train': train_dataset, 'val': eval_dataset}
    wandb_logs = {}
    visuals_dict = {}
    model.eval()
    stats_n_maps = opt.stats_n_maps  # how many maps to average over the metrics
    vis_n_maps = opt.vis_n_maps  # how many maps to visualize
    vis_n_generator_runs = opt.vis_n_generator_runs  # how many sampled fake agents per map to visualize
    metrics = dict()
    metrics_type_names = ['G/loss_GAN', 'G/loss_reconstruct', 'G/loss_total',
                          'D/loss_classify_real', 'D/loss_classify_fake', 'D/loss_grad_penalty',
                          'D/loss_total', 'D/logit(fake)', 'D/logit(real)']

    for dataset_name, dataset in datasets.items():
        metrics_names = [f'{dataset_name}/{type_name}' for type_name in metrics_type_names]

        for metric_name in metrics_names:
            metrics[metric_name] = np.zeros(stats_n_maps)

        assert vis_n_generator_runs >= 1
        map_id = 0
        log_label = 'null'
        for scene_data in dataset:
            if map_id >= stats_n_maps:
                break
            real_agents_vecs, _, conditioning = pre_process_scene_data(scene_data, opt)

            if map_id < vis_n_maps:
                # Add an image of the map & real agents to wandb logs
                log_label = f"{dataset_name}/Epoch#{1 + i_epoch} iter#{1 + epoch_iter} Map#{1 + map_id}"
                img, wandb_img = get_wandb_image(model, conditioning, real_agents_vecs, label='real_agents')
                visuals_dict[f'{dataset_name}_map_{map_id}_real_agents'] = img
                if opt.use_wandb:
                    wandb_logs[log_label] = [wandb_img]

            for i_generator_run in range(vis_n_generator_runs):
                fake_agents_vecs = model.netG(conditioning).detach()  # detach since we don't backpropp

                # calculate the metrics for only for the first generated agents set per map:
                if i_generator_run == 0:
                    # Feed real agents set to discriminator
                    d_out_for_real = model.netD(conditioning, real_agents_vecs).detach()  # detach since we don't backpropp
                    # pred_is_real_for_real_binary = (pred_is_real_for_real > 0).to(torch.float32)
                    d_out_for_fake = model.netD(conditioning, fake_agents_vecs).detach()  # detach since we don't backpropp
                    # pred_is_real_for_fake_binary = (pred_is_real_for_fake > 0).to(torch.float32)
                    loss_D_fake = model.criterionGAN(prediction=d_out_for_fake,
                                                     target_is_real=False)  # D wants to correctly classsify
                    loss_D_real = model.criterionGAN(prediction=d_out_for_real,
                                                     target_is_real=True)  # D wants to correctly classsify
                    loss_G_GAN = model.criterionGAN(prediction=d_out_for_fake,
                                                    target_is_real=True)  # G tries to make D wrongly classify the fake sample (make D output "True"
                    loss_G_reconstruct = model.criterion_reconstruct(fake_agents_vecs, real_agents_vecs)

                    loss_D_grad_penalty = cal_gradient_penalty(model.netD, conditioning, real_agents_vecs,
                                                               fake_agents_vecs, model)

                    metrics[f'{dataset_name}/G/loss_GAN'][map_id] = loss_G_GAN
                    metrics[f'{dataset_name}/G/loss_reconstruct'][map_id] = loss_G_reconstruct
                    metrics[f'{dataset_name}/G/loss_total'][map_id] = loss_G_GAN + loss_G_reconstruct * opt.lambda_reconstruct
                    metrics[f'{dataset_name}/D/loss_classify_real'][map_id] = loss_D_real
                    metrics[f'{dataset_name}/D/loss_classify_fake'][map_id] = loss_D_fake
                    metrics[f'{dataset_name}/D/loss_grad_penalty'][map_id] = loss_D_grad_penalty
                    metrics[f'{dataset_name}/D/loss_total'][map_id] = loss_D_fake + loss_D_real + model.lambda_gp * loss_D_grad_penalty
                    metrics[f'{dataset_name}/D/logit(fake)'][map_id] = d_out_for_fake
                    metrics[f'{dataset_name}/D/logit(real)'][map_id] = d_out_for_real

                # Add an image of the map & fake agents to wandb logs
                if map_id < vis_n_maps and i_generator_run < vis_n_generator_runs:
                    img, wandb_img = get_wandb_image(model, conditioning, fake_agents_vecs, label='real_agents')
                    visuals_dict[f'{dataset_name}_map_#{map_id + 1}_fake_#{i_generator_run + 1}'] = img
                    if opt.use_wandb:
                        wandb_logs[log_label].append(wandb_img)
            map_id += 1

    # Average over the maps:
    for key, val in metrics.items():
        metrics[key] = val.mean()

    # additional metrics:
    metrics['run/LR'] = model.lr
    metrics['run/epoch'] = 1 + i_epoch
    metrics['run/total_iters'] = total_iters

    if opt.use_wandb:
        wandb.log(metrics)

    print('Eval metrics: ' + ', '.join([f'{key}: {val:.2f}' for key, val in metrics.items()]))

    # wandb.log(info_dict)
    # # Show also in table of current vales:
    # run_time_str = strfdelta(datetime.timedelta(seconds=time.time() - run_start_time), '%H:%M:%S')
    # table_columns = ['Runtime'] + list(info_dict.keys())
    # table_data_row = [run_time_str] + list(info_dict.values())
    # table_data_rows = [table_data_row]
    # wandb_logs[f"Epoch #{1+i_epoch} iter #{1+epoch_iter}"] = \
    #     wandb.Table(columns=table_columns, data=table_data_rows)

    if opt.isTrain:
        model.train()
    return visuals_dict, wandb_logs


#########################################################################################

def get_wandb_image(model, conditioning, agents_vecs, label='real_agents'):
    agents_feat_dicts = agents_feat_vecs_to_dicts(agents_vecs)
    real_map = conditioning['map_feat']
    img = visualize_scene_feat(agents_feat_dicts, real_map)
    pred_is_real = torch.sigmoid(model.netD(conditioning, agents_vecs)).item()
    caption = f'{label}\npred_is_real={pred_is_real:.2}\n'
    caption += '\n'.join(get_agents_descriptions(agents_feat_dicts))
    wandb_img = wandb.Image(img, caption=caption)
    return img, wandb_img

#########################################################################################
