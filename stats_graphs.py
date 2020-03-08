import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--stats_path', type=str,
                    help='Path to the stats files')
args = parser.parse_args()
stats_path = args.stats_path
path_loss_log = os.path.join(stats_path, 'loss_log.txt')

with open(path_loss_log, 'r') as f:
    loss_log = f.read()

loss_log = re.split('\n',
                    re.sub('\n\n', '\n',
                    re.sub('================ Training Loss .+ ================\n', '',
                    re.sub('[()]', '', loss_log))))

loss_dict = {}


eval_list = ['epoch', 'iters', 'time', 'G_GAN', 'G_L1', 'D_real', 'D_fake']

# Generator loss:
#        self.loss_G_GAN = self.criterionGAN(pred_fake_gen, True)
#        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
#        self.loss_G = self.loss_G_GAN + self.loss_G_L1
#Loss G Gan and G L1 are combined for the generator loss

# Discriminator loss:
#        self.loss_D_real = self.criterionGAN(pred_real, True)
#        self.loss_D_fake = self.criterionGAN(pred_fake_discr, False)
#        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

for element in eval_list:
    loss_dict[element] = []



for l, line in enumerate(loss_log):
    if len(line) > 0:
        number = '[+-]?([0-9]*[.])?[0-9]+'
        for element in eval_list:
            epoch = re.sub('epoch' + ': ', '', re.search('epoch' + ': ' + number, line)[0])
            stat = re.sub(str(element) + ': ', '', re.search(str(element) + ': ' + number, line)[0])
            loss_dict[str(element)].append(float(stat))


loss_df = pd.DataFrame(data=loss_dict)

average_loss_df = loss_df.groupby('epoch').mean()


plot_df = average_loss_df

fig, ax = plt.subplots(nrows=2, ncols=2)
fig.tight_layout()

ax[0,0].plot(range(0,len(plot_df['G_GAN'])), plot_df['G_GAN'])
ax[0,0].title.set_text('G_GAN')
ax[0,1].plot(range(0,len(plot_df['G_L1'])), plot_df['G_L1'])
ax[0,1].title.set_text('G_L1')
ax[1,0].plot(range(0,len(plot_df['D_real'])), plot_df['D_real'])
ax[1,0].title.set_text('D_real')
ax[1,1].plot(range(0,len(plot_df['D_fake'])), plot_df['D_fake'])
ax[1,1].title.set_text('D_fake')

fig.savefig(os.path.join(stats_path, 'stats_graph.png'), dpi=300, bbox_inches='tight')

