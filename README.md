# Conditional Pokémon Generation with Pix2Pix

This repository is a personal project exploring conditional image generation of Pokémon using a fork of the pix2pix framework. The input domain consists of silhouette masks colored by the Pokémon's primary type, and the output domain is the generated Pokémon image.

Credit for the pix2pix implementation goes to: [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesungp), and supported by [Tongzhou Wang](https://github.com/SsnL).

Image-to-Image Translation with Conditional Adversarial Networks.<br>
[Phillip Isola](https://people.eecs.berkeley.edu/~isola), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In CVPR 2017. [[Bibtex]](https://www.cs.cmu.edu/~junyanz/projects/pix2pix/pix2pix.bib)


## TODO

### 🔧 Data Preparation
- [ ] Collect or create silhouette masks of Pokémon.

- [ ] Assign and apply colors to masks based on Pokémon primary types.

- [ ] Pair masks with corresponding official Pokémon images for training.


🏋️ Training
- [ ] Define training/validation split.

- [ ] Train on a subset of Pokémon (e.g., starters or first 150).

- [ ] Train on full dataset

- [ ] Experiment with hyperparameters (batch size, learning rate, epochs, etc.).


🎨 Output & Evaluation
- [ ] Generate Pokémon from unseen mask/type combinations.

- [ ] Compare generated results with ground truth Pokémon designs.

- [ ] Explore diversity and consistency of generations by varying the mask slightly.

📈 Enhancements
- [ ]Add a GUI or notebook interface to draw/select a silhouette and type.

- [ ] Add secondary type handling (e.g., gradient or dual-color silhouettes).

📚 Documentation

- [ ] Document how to run the project and reproduce results.

- [ ] Add sample inputs/outputs in the repo.