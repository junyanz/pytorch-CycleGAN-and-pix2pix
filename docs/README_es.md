<img src='https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/raw/master/imgs/horse2zebra.gif' align="right" width=384>

<br><br><br>

# CycleGAN y pix2pix en PyTorch

Implementacion en PyTorch de Unpaired Image-to-Image Translation.

Este codigo fue escrito por [Jun-Yan Zhu](https://github.com/junyanz) y [Taesung Park](https://github.com/taesung), y con ayuda de [Tongzhou Wang](https://ssnl.github.io/).

Esta implementacion de PyTorch produce resultados comparables o mejores que nuestros original software de Torch. Si te gustaria producir los mismos resultados que en documento oficial, echa un vistazo al codigo original [CycleGAN Torch](https://github.com/junyanz/CycleGAN) y [pix2pix Torch](https://github.com/phillipi/pix2pix) 

**Aviso**: El software actual funciona correctamente en PyTorch 0.41+. Para soporte en PyTorch 0.1-0.3: [branch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/pytorch0.3.1).

Puede encontrar información útil en [training/test tips](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md) y [preguntas frecuentes](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md). Para implementar modelos y conjuntos de datos personalizados, consulte nuestro [templates](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/README_es.md#modelo-y-dataset-personalizado). Para ayudar a los usuarios a comprender y adaptar mejor nuestra base de código, proporcionamos un [overview](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/overview.md) de la estructura de código de este repositorio.

**CycleGAN: [Proyecto](https://junyanz.github.io/CycleGAN/) |  [PDF](https://arxiv.org/pdf/1703.10593.pdf) |  [Torch](https://github.com/junyanz/CycleGAN) |
[Guia de Tensorflow Core](https://www.tensorflow.org/tutorials/generative/cyclegan) | [PyTorch Colab](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb)**

<img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg" width="800"/>

**Pix2pix:  [Proyeto](https://phillipi.github.io/pix2pix/) |  [PDF](https://arxiv.org/pdf/1611.07004.pdf) |  [Torch](https://github.com/phillipi/pix2pix) |
[Guia de Tensorflow Core](https://www.tensorflow.org/tutorials/generative/cyclegan) | [PyTorch Colab](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb)**

<img src="https://phillipi.github.io/pix2pix/images/teaser_v3.png" width="800px"/>


**[EdgesCats Demo](https://affinelayer.com/pixsrv/) | [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow) | por [Christopher Hesse](https://twitter.com/christophrhesse)**

<img src='https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/imgs/edges2cats.jpg' width="400px"/>

Si usa este código para su investigación, cite:

Unpaired Image-to-Image Translation usando Cycle-Consistent Adversarial Networks.<br>
[Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/)\*,  [Taesung Park](https://taesung.me/)\*, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In ICCV 2017. (* contribucion igualitaria) [[Bibtex]](https://junyanz.github.io/CycleGAN/CycleGAN.txt)


Image-to-Image Translation usando Conditional Adversarial Networks.<br>
[Phillip Isola](https://people.eecs.berkeley.edu/~isola), [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In CVPR 2017. [[Bibtex]](http://people.csail.mit.edu/junyanz/projects/pix2pix/pix2pix.bib)

## Charlas y curso
Presentacion en PowerPoint de Pix2pix: [keynote](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/pix2pix.key) | [pdf](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/pix2pix.pdf),
Presentacion en PowerPoint de CycleGAN: [pptx](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/CycleGAN.pptx) | [pdf](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/CycleGAN.pdf)

Asignación del curso CycleGAN [codigo](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip) y [handout](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf) diseñado por el Prof. [Roger Grosse](http://www.cs.toronto.edu/~rgrosse/) for [CSC321](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/) "Intro to Neural Networks and Machine Learning" en la universidad de Toronto. Póngase en contacto con el instructor si desea adoptarlo en su curso.

## Colab Notebook
TensorFlow Core CycleGAN Tutorial: [Google Colab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb) | [Codigo](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb)

Guia de TensorFlow Core pix2pix : [Google Colab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb) | [Codigo](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb)

PyTorch Colab notebook: [CycleGAN](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb) y [pix2pix](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb)

## Otras implementaciones
### CycleGAN
<p><a href="https://github.com/leehomyc/cyclegan-1"> [Tensorflow]</a> (por Harry Yang),
<a href="https://github.com/architrathore/CycleGAN/">[Tensorflow]</a> (por Archit Rathore),
<a href="https://github.com/vanhuyz/CycleGAN-TensorFlow">[Tensorflow]</a> (por Van Huy),
<a href="https://github.com/XHUJOY/CycleGAN-tensorflow">[Tensorflow]</a> (por Xiaowei Hu),
<a href="https://github.com/LynnHo/CycleGAN-Tensorflow-Simple"> [Tensorflow-simple]</a> (por Zhenliang He),
<a href="https://github.com/luoxier/CycleGAN_Tensorlayer"> [TensorLayer]</a> (por luoxier),
<a href="https://github.com/Aixile/chainer-cyclegan">[Chainer]</a> (por Yanghua Jin),
<a href="https://github.com/yunjey/mnist-svhn-transfer">[Minimal PyTorch]</a> (por yunjey),
<a href="https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/CycleGAN">[Mxnet]</a> (por Ldpe2G),
<a href="https://github.com/tjwei/GANotebooks">[lasagne/Keras]</a> (por tjwei),
<a href="https://github.com/simontomaskarlsson/CycleGAN-Keras">[Keras]</a> (por Simon Karlsson)
</p>
</ul>

### pix2pix
<p><a href="https://github.com/affinelayer/pix2pix-tensorflow"> [Tensorflow]</a> (por Christopher Hesse),
<a href="https://github.com/Eyyub/tensorflow-pix2pix">[Tensorflow]</a> (por Eyyüb Sariu),
<a href="https://github.com/datitran/face2face-demo"> [Tensorflow (face2face)]</a> (por Dat Tran),
<a href="https://github.com/awjuliani/Pix2Pix-Film"> [Tensorflow (film)]</a> (por Arthur Juliani),
<a href="https://github.com/kaonashi-tyc/zi2zi">[Tensorflow (zi2zi)]</a> (por Yuchen Tian),
<a href="https://github.com/pfnet-research/chainer-pix2pix">[Chainer]</a> (por mattya),
<a href="https://github.com/tjwei/GANotebooks">[tf/torch/keras/lasagne]</a> (por tjwei),
<a href="https://github.com/taey16/pix2pixBEGAN.pytorch">[Pytorch]</a> (por taey16)
</p>
</ul>

## Requerimientos
- Linux o macOS
- Python 3
- CPU o NVIDIA GPU usando CUDA CuDNN

## Inicio
### Instalación

- Clone este repositorio:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

- Instale [PyTorch](http://pytorch.org) 0.4+ y sus otras dependencias (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) y [dominate](https://github.com/Knio/dominate)).
  - Para uso de pip, por favor escriba el comando `pip install -r requirements.txt`.
  - Para uso de Conda, proporcionamos un script de instalación `./scripts/conda_deps.sh`. De forma alterna, puede crear un nuevo entorno Conda usando `conda env create -f environment.yml`.
  - Para uso de Docker, Proporcionamos la imagen Docker y el archivo Docker preconstruidos. Por favor, consulte nuestra página
 [Docker](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/docker.md).

### CycleGAN entreanimiento/test
- Descargar el dataset de CycleGAN (e.g. maps):
```bash
bash ./datasets/download_cyclegan_dataset.sh maps
```
- Para ver los resultados del entrenamiento y las gráficas de pérdidas, `python -m visdom.server` y haga clic en la URL
 http://localhost:8097.
- Entrenar el modelo:
```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
Para ver más resultados intermedios, consulte `./checkpoints/maps_cyclegan/web/index.html`.
- Pruebe el modelo:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
-Los resultados de la prueba se guardarán en un archivo html aquí: `./results/maps_cyclegan/latest_test/index.html`.

### pix2pix entrenamiento/test
- Descargue el dataset de pix2pix (e.g.[facades](http://cmp.felk.cvut.cz/~tylecr1/facade/)):
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
- Para ver los resultados del entrenamiento y las gráficas de pérdidas `python -m visdom.server`, haga clic en la URL http://localhost:8097.
- Para entrenar el modelo:
```bash
#!./scripts/train_pix2pix.sh
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
Para ver más resultados intermedios, consulte `./checkpoints/facades_pix2pix/web/index.html`.

- Pruebe el modelo (`bash ./scripts/test_pix2pix.sh`):
```bash
#!./scripts/test_pix2pix.sh
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
- Los resultados de la prueba se guardarán en un archivo html aquí: `./results/facades_pix2pix/test_latest/index.html`. Puede encontrar más scripts en `scripts` directory.
- Para entrenar y probar modelos de colorización basados en pix2pix, agregue la linea `--model colorization` y `--dataset_mode colorization`. Para más detalles de nuestro entrenamiento [tips](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md#notes-on-colorization).

### Aplicar un modelo pre-entrenado (CycleGAN)
- Puedes descargar un modelo previamente entrenado (e.g. horse2zebra) con el siguiente script:
```bash
bash ./scripts/download_cyclegan_model.sh horse2zebra
```
- El modelo pre-entrenado se guarda en `./checkpoints/{name}_pretrained/latest_net_G.pth`. Revise [aqui](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_cyclegan_model.sh#L3) para todos los modelos CycleGAN disponibles.

- Para probar el modelo, también debe descargar el dataset horse2zebra:
```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

- Luego genere los resultados usando:
```bash
python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
```
- La opcion `--model test` ise usa para generar resultados de CycleGAN de un solo lado. Esta opción configurará automáticamente
 `--dataset_mode single`, carga solo las imágenes de un conjunto. Por el contrario, el uso de `--model cycle_gan` requiere cargar y generar resultados en ambas direcciones, lo que a veces es innecesario. Los resultados se guardarán en `./results/`. Use `--results_dir {directory_path_to_save_result}` para especificar el directorio de resultados.

- Para sus propios experimentos, es posible que desee especificar `--netG`, `--norm`, `--no_dropout` para que coincida con la arquitectura del generador del modelo entrenado.

### Aplicar un modelo pre-entrenado (pix2pix)
Descargue un modelo pre-entrenado con `./scripts/download_pix2pix_model.sh`.

- Revise [aqui](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_pix2pix_model.sh#L3) para todos los modelos pix2pix disponibles. Por ejemplo, si desea descargar el modelo label2photo en el dataset:
```bash
bash ./scripts/download_pix2pix_model.sh facades_label2photo
```
- Descarga el dataset facades de pix2pix:
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
- Luego genere los resultados usando:
```bash
python test.py --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained
```
- Tenga en cuenta que `--direction BtoA` como Facades dataset's, son direcciones A o B para etiquetado de fotos.

- Si desea aplicar un modelo previamente entrenado a una colección de imágenes de entrada (en lugar de pares de imágenes), use la opcion `--model test`. Vea `./scripts/test_single.sh` obre cómo aplicar un modelo a Facade label maps (almacenados en el directorio `facades/testB`).

- Vea una lista de los modelos disponibles actualmente en `./scripts/download_pix2pix_model.sh`

## [Docker](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/docker.md)
Proporcionamos la imagen Docker y el archivo Docker preconstruidos que pueden ejecutar este repositorio de código. Ver [docker](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/docker.md).

## [Datasets](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md)
Descargue los conjuntos de datos pix2pix / CycleGAN y cree sus propios conjuntos de datos.

## [Entretanimiento/Test Tips](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md)
Las mejores prácticas para entrenar y probar sus modelos.

## [Preguntas frecuentes](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md)
Antes de publicar una nueva pregunta, primero mire las preguntas y respuestas anteriores y los problemas existentes de GitHub.

## Modelo y Dataset personalizado
Si planea implementar modelos y conjuntos de datos personalizados para sus nuevas aplicaciones, proporcionamos un conjunto de datos [template](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/template_dataset.py) y un modelo [template](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/template_model.py) como punto de partida.


## [Estructura de codigo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/overview.md)
Para ayudar a los usuarios a comprender mejor y usar nuestro código, presentamos brevemente la funcionalidad e implementación de cada paquete y cada módulo.

## Solicitud de Pull
Siempre puede contribuir a este repositorio enviando un [pull request](https://help.github.com/articles/about-pull-requests/).
Por favor ejecute `flake8 --ignore E501 .` y `python ./scripts/test_before_push.py` antes de realizar un Pull en el código, asegure de también actualizar la estructura del código [overview](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/overview.md) en consecuencia si agrega o elimina archivos.


## Citación
Si utiliza este código para su investigación, cite nuestros documentos.
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

## Proyectos relacionados
**[CycleGAN-Torch](https://github.com/junyanz/CycleGAN) |
[pix2pix-Torch](https://github.com/phillipi/pix2pix) | [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)|
[BicycleGAN](https://github.com/junyanz/BicycleGAN) | [vid2vid](https://tcwang0509.github.io/vid2vid/) | [SPADE/GauGAN](https://github.com/NVlabs/SPADE)**<br>
**[iGAN](https://github.com/junyanz/iGAN) | [GAN Dissection](https://github.com/CSAILVision/GANDissect) | [GAN Paint](http://ganpaint.io/)**

## Cat Paper Collection
Si amas a los gatos y te encanta leer gráficos geniales, computer vision y documentos de aprendizaje, echa un vistazo a Cat Paper [Collection](https://github.com/junyanz/CatPapers).

## Agradecimientos
Nuestro código fue inspirado en [pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan).
