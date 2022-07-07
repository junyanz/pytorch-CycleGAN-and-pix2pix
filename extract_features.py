import os
from data import create_dataset
from models import create_model
from data import create_dataset
from options.test_options import TestOptions
import torch as torch
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle as p
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

PERPLEXITY=32
N_COMP=3

def train_svm(embeddings, filename, reduce_dim=False, dim_red_method="tsne"):
    label = np.zeros(len(list(embeddings.keys())))
    for i, key in enumerate(embeddings.keys()):
        if key.split('_')[0] == 'morph':
            label[i] = 1
        else:
            label[i] = 0
    
    data = list(embeddings.values())
    if reduce_dim==True:
        if dim_red_method == "tsne":
            tsne = TSNE(n_components=N_COMP, verbose=1, random_state=123, perplexity=PERPLEXITY, method="exact")
            data = tsne.fit_transform(list(embeddings.values()))
            

    clf = svm.SVC(probability=True)
    clf.fit(np.array(data), label)

    #save model
    #filename = 'trained_svm.sav'
    p.dump(clf, open(filename, 'wb'))

def plot_tsne(embeddings):
    tsne = TSNE(n_components=2, verbose=1, random_state=123, perplexity=20, n_iter=5000)
    tsne_results = tsne.fit_transform(list(embeddings.values()))

    label = np.zeros(len(list(embeddings.keys())))
    for i, key in enumerate(embeddings.keys()):
        if key.split('_')[0] == 'morph':
            label[i] = 0
        else:
            label[i] = 1

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=label)
    plt.legend(handles=scatter.legend_elements()[0], labels=['morph','non-morph'])
    f.savefig('tsne.png', dpi=300, bbox_inches='tight')
    
    plt.close()


def train_knn(n, embeddings):
    label = np.zeros(len(list(embeddings.keys())))
    for i, key in enumerate(embeddings.keys()):
        if key.split('_')[0] == 'morph':
            label[i] = 1
        else:
            label[i] = 0
    data = list(embeddings.values())
    neigh = KNeighborsClassifier(n_neighbors=n)
    neigh.fit(np.array(data), label)

    filename = 'trained_knn_' + str(n) + '_.sav'
    p.dump(neigh, open(filename, 'wb'))
    

def load_svm(embeddings, filename, reduce_dim=False, dim_red_method="tsne"):
    label = np.zeros(len(list(embeddings.keys())))
    #filename = 'trained_svm.sav'
    loaded_svm = p.load(open(filename, 'rb'))
    data = list(embeddings.values())
    if reduce_dim==True:
        tsne = TSNE(n_components=N_COMP, verbose=1, random_state=123, perplexity=PERPLEXITY)
        data = tsne.fit_transform(list(embeddings.values()))
    for i, key in enumerate(embeddings.keys()):
        if key.split('_')[0] == 'morph':
            label[i] = 1
        else:
            label[i] = 0
    #result = loaded_svm.score(np.array(data), label)
    pred_label = loaded_svm.predict(data)
    result = confusion_matrix(label, pred_label)
    
    print((result[0][0] + result[1][1])/ len(list(embeddings.keys())))
    print(result)
    return loaded_svm.predict_proba(data), label

def extract_features():
    opt = TestOptions().parse()  # get test options
    num_threads = 0   # test code only supports num_threads = 0
    batch_size = 1    # test code only supports batch_size = 1
    serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    dataset = create_dataset(opt)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt) 
    model.eval()
    #layer = model._modules.get('DataParallel.module.model.model[1].model[3].model[3].model[3].model[3].model[3].model[3].model[2]')
    net = model.netG.module.model.model[1].model[3].model[3].model[3].model[3].model[3].model[3].model[2]#getattr(model, 'net' + layer1)
    layer1 = net

    embeddings = {}
    for i, data in enumerate(dataset):
        embedding_tran1 = torch.zeros([1, 512])
        filename = data['A_paths'][0].split('/')[-1]
        model.set_input(data)  # unpack data from data loader
        def copy_data(model, input, o):
            embedding_tran1.copy_(o.data.reshape(o.data.size(1)))
        h = layer1.register_forward_hook(copy_data)        
        model.test()           # run inference
        print(i)
        
        embeddings[filename] = embedding_tran1.detach().numpy().squeeze(0)
        h.remove()

    return embeddings

def plot_roc(probabilities, labels):
    class_1 = np.zeros(len(probabilities))
    class_0 = np.zeros(len(probabilities))
    label_1 = np.ones(len(probabilities))
    label_0 = np.zeros(len(probabilities))
    for i, prob in enumerate(probabilities):
        class_0[i] = prob[0]
        class_1[i] = prob[1]
    
    prob = np.append(class_0, class_1)
    #labels = np.append(label_0, label_1)
    fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
    get_bpcer_apcer(5, fpr, tpr)
    get_bpcer_apcer(10, fpr, tpr)
    fig, axs = plt.subplots(1, 1)
    axs.plot(100*(1-tpr), 100*fpr, 'b-o', linewidth=2, label='ROC')
    plt.grid(True)
    axs.legend(loc='best')
    axs.set_xlabel('APCER (%)', fontsize=14, fontweight='bold')
    axs.set_ylabel('BPCER (%)', fontsize=14, fontweight='bold')
    axs.set_title('ROC Curve ({} gen. scores, {} imp. scores)'.format(np.count_nonzero(labels==1), np.count_nonzero(labels==0)), fontsize=18, fontweight='bold')
    fig.savefig('roc.png', dpi=300, bbox_inches='tight')
    plt.close()


def get_bpcer_apcer(apcer, fpr, tpr):
    fpr = 100*fpr
    tpr = 100*(1-tpr)
    min_diff = 100
    index = -1
    for i, f in enumerate(fpr):
        diff = abs(f-apcer)
        if diff < min_diff:
            min_diff = diff
            index = i

    print(f"Minimum:{min_diff} index:{index} fpr:{fpr[index]} bpcer:{tpr[index]}")



if __name__ == '__main__':
    embeddings = extract_features()
    N_COMP=2
    #train_svm(embeddings, '/research/iprobe-protichi/svm_experiment/MorGAN/trained_svm_tsne_2.sav', True)
    train_svm(embeddings, '/research/iprobe-protichi/svm_experiment/MorGAN/trained_svm.sav')
    N_COMP=3
    #train_svm(embeddings, '/research/iprobe-protichi/svm_experiment/MorGAN/trained_svm_tsne_2.sav', True)
    N_COMP=4
    #train_svm(embeddings, '/research/iprobe-protichi/svm_experiment/MorGAN/trained_svm_tsne_2.sav', True)
    #probs, labels = load_svm(embeddings, '/research/iprobe-protichi/svm_experiment/MorGAN/trained_svm.sav', True)
    probs, labels = load_svm(embeddings, '/research/iprobe-protichi/svm_experiment/MorGAN/trained_svm.sav')
    plot_roc(probs, labels)
    
    #knn
    #train_knn(5, embeddings)
    #load_svm(embeddings, 'trained_knn_5_.sav')
    #print(probs)
    #plot_roc(probs, labels)

    #plot_tsne(embeddings)



