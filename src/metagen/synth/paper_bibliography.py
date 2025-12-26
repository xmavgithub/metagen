"""
MetaGen Paper Bibliography System

Dynamic bibliography generation based on spec modalities, architectures, and objectives.
Generates BibTeX entries relevant to the specific model being synthesized.

Note: BibTeX author fields intentionally exceed line length limits.
"""
# ruff: noqa: E501

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metagen.specs.schema import ModelSpec

# =============================================================================
# Core References (always included)
# =============================================================================

CORE_REFERENCES = {
    "vaswani2017attention": """@inproceedings{vaswani2017attention,
  title={Attention is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and
          Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and
          Kaiser, {\\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}""",
    "ho2020denoising": """@inproceedings{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  pages={6840--6851},
  year={2020}
}""",
    "elsken2019neural": """@article{elsken2019neural,
  title={Neural Architecture Search: A Survey},
  author={Elsken, Thomas and Metzen, Jan Hendrik and Hutter, Frank},
  journal={Journal of Machine Learning Research},
  volume={20},
  number={55},
  pages={1--21},
  year={2019}
}""",
}

# =============================================================================
# Modality-Specific References
# =============================================================================

TEXT_REFERENCES = {
    "devlin2019bert": """@inproceedings{devlin2019bert,
  title={{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={Proceedings of NAACL-HLT},
  pages={4171--4186},
  year={2019}
}""",
    "brown2020language": """@inproceedings{brown2020language,
  title={Language Models are Few-Shot Learners},
  author={Brown, Tom and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and
          Kaplan, Jared D and Dhariwal, Prafulla and Neelakantan, Arvind and
          Shyam, Pranav and Sastry, Girish and Askell, Amanda and others},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  pages={1877--1901},
  year={2020}
}""",
    "touvron2023llama": """@article{touvron2023llama,
  title={{LLaMA}: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and
          Martinet, Xavier and Lachaux, Marie-Anne and
          Lacroix, Timoth{\\'e}e and Rozi{\\`e}re, Baptiste and
          Goyal, Naman and Hambro, Eric and Azhar, Faisal and others},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}""",
    "radford2019language": """@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and
          Luan, David and Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI Blog},
  volume={1},
  number={8},
  pages={9},
  year={2019}
}""",
}

IMAGE_REFERENCES = {
    "dosovitskiy2021image": """@inproceedings{dosovitskiy2021image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and
          Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and
          Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and
          Gelly, Sylvain and others},
  booktitle={International Conference on Learning Representations},
  year={2021}
}""",
    "rombach2022high": """@inproceedings{rombach2022high,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and
          Esser, Patrick and Ommer, Bj{\\\"o}rn},
  booktitle={Proceedings of the IEEE/CVF CVPR},
  pages={10684--10695},
  year={2022}
}""",
    "he2016deep": """@inproceedings{he2016deep,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={770--778},
  year={2016}
}""",
    "krizhevsky2012imagenet": """@inproceedings{krizhevsky2012imagenet,
  title={{ImageNet} Classification with Deep Convolutional Neural Networks},
  author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  booktitle={Advances in Neural Information Processing Systems},
  volume={25},
  year={2012}
}""",
}

AUDIO_REFERENCES = {
    "oord2016wavenet": """@article{oord2016wavenet,
  title={{WaveNet}: A Generative Model for Raw Audio},
  author={Oord, Aaron van den and Dieleman, Sander and Zen, Heiga and
          Simonyan, Karen and Vinyals, Oriol and Graves, Alex and
          Kalchbrenner, Nal and Senior, Andrew and Kavukcuoglu, Koray},
  journal={arXiv preprint arXiv:1609.03499},
  year={2016}
}""",
    "dhariwal2020jukebox": """@inproceedings{dhariwal2020jukebox,
  title={Jukebox: A Generative Model for Music},
  author={Dhariwal, Prafulla and Jun, Heewoo and Payne, Christine and
          Kim, Jong Wook and Radford, Alec and Sutskever, Ilya},
  booktitle={International Conference on Machine Learning},
  pages={2488--2497},
  year={2020}
}""",
    "kong2020hifi": """@inproceedings{kong2020hifi,
  title={{HiFi-GAN}: Generative Adversarial Networks for High Fidelity Speech},
  author={Kong, Jungil and Kim, Jaehyeon and Bae, Jaekyoung},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  pages={17022--17033},
  year={2020}
}""",
}

VIDEO_REFERENCES = {
    "arnab2021vivit": """@inproceedings{arnab2021vivit,
  title={{ViViT}: A Video Vision Transformer},
  author={Arnab, Anurag and Dehghani, Mostafa and Heigold, Georg and Sun, Chen and Lu{\\v{c}}i{\\'c}, Mario and Schmid, Cordelia},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6836--6846},
  year={2021}
}""",
    "ho2022video": """@article{ho2022video,
  title={Video Diffusion Models},
  author={Ho, Jonathan and Salimans, Tim and Gritsenko, Alexey and Chan, William and Norouzi, Mohammad and Fleet, David J},
  journal={arXiv preprint arXiv:2204.03458},
  year={2022}
}""",
    "tran2015learning": """@inproceedings{tran2015learning,
  title={Learning Spatiotemporal Features with {3D} Convolutional Networks},
  author={Tran, Du and Bourdev, Lubomir and Fergus, Rob and Torresani, Lorenzo and Paluri, Manohar},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={4489--4497},
  year={2015}
}""",
}

MULTIMODAL_REFERENCES = {
    "radford2021learning": """@inproceedings{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International Conference on Machine Learning},
  pages={8748--8763},
  year={2021}
}""",
    "alayrac2022flamingo": """@inproceedings{alayrac2022flamingo,
  title={Flamingo: a Visual Language Model for Few-Shot Learning},
  author={Alayrac, Jean-Baptiste and Donahue, Jeff and Luc, Pauline and Miech, Antoine and Barr, Iain and Hasson, Yana and Lenc, Karel and Mensch, Arthur and Millican, Katherine and Reynolds, Malcolm and others},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  pages={23716--23736},
  year={2022}
}""",
    "liu2024visual": """@inproceedings{liu2024visual,
  title={Visual Instruction Tuning},
  author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}""",
}

# =============================================================================
# Architecture-Specific References
# =============================================================================

NAS_REFERENCES = {
    "liu2019darts": """@inproceedings{liu2019darts,
  title={{DARTS}: Differentiable Architecture Search},
  author={Liu, Hanxiao and Simonyan, Karen and Yang, Yiming},
  booktitle={International Conference on Learning Representations},
  year={2019}
}""",
    "zoph2018learning": """@inproceedings{zoph2018learning,
  title={Learning Transferable Architectures for Scalable Image Recognition},
  author={Zoph, Barret and Vasudevan, Vijay and Shlens, Jonathon and Le, Quoc V},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={8697--8710},
  year={2018}
}""",
    "tan2019efficientnet": """@inproceedings{tan2019efficientnet,
  title={{EfficientNet}: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International Conference on Machine Learning},
  pages={6105--6114},
  year={2019}
}""",
    "real2020automl": """@inproceedings{real2020automl,
  title={{AutoML-Zero}: Evolving Machine Learning Algorithms From Scratch},
  author={Real, Esteban and Liang, Chen and So, David and Le, Quoc},
  booktitle={International Conference on Machine Learning},
  pages={8007--8019},
  year={2020}
}""",
    "cai2020once": """@inproceedings{cai2020once,
  title={Once-for-All: Train One Network and Specialize it for Efficient Deployment},
  author={Cai, Han and Gan, Chuang and Wang, Tianzhe and Zhang, Zhekai and Han, Song},
  booktitle={International Conference on Learning Representations},
  year={2020}
}""",
}

META_LEARNING_REFERENCES = {
    "ha2016hypernetworks": """@article{ha2016hypernetworks,
  title={HyperNetworks},
  author={Ha, David and Dai, Andrew and Le, Quoc V},
  journal={arXiv preprint arXiv:1609.09106},
  year={2016}
}""",
    "finn2017model": """@inproceedings{finn2017model,
  title={Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks},
  author={Finn, Chelsea and Abbeel, Pieter and Levine, Sergey},
  booktitle={International Conference on Machine Learning},
  pages={1126--1135},
  year={2017}
}""",
}

# =============================================================================
# Training Objective References
# =============================================================================

OBJECTIVE_REFERENCES = {
    "contrastive": {
        "chen2020simple": """@inproceedings{chen2020simple,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle={International Conference on Machine Learning},
  pages={1597--1607},
  year={2020}
}""",
    },
    "diffusion": {
        "song2021score": """@inproceedings{song2021score,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Song, Yang and Sohl-Dickstein, Jascha and Kingma, Diederik P and Kumar, Abhishek and Ermon, Stefano and Poole, Ben},
  booktitle={International Conference on Learning Representations},
  year={2021}
}""",
    },
    "autoregressive": {
        "radford2018improving": """@article{radford2018improving,
  title={Improving Language Understanding by Generative Pre-Training},
  author={Radford, Alec and Narasimhan, Karthik and Salimans, Tim and Sutskever, Ilya},
  journal={OpenAI},
  year={2018}
}""",
    },
}


def generate_bibliography(spec: ModelSpec) -> str:
    """Generate a BibTeX bibliography based on the model specification.

    Dynamically selects relevant references based on:
    - Input/output modalities (text, image, audio, video, multimodal)
    - Architecture family (transformer, cnn, diffusion)
    - Training objectives (contrastive, diffusion, autoregressive)

    Args:
        spec: Model specification to generate bibliography for.

    Returns:
        BibTeX string with all relevant references.

    Example:
        >>> from metagen.specs.schema import ModelSpec
        >>> spec = ModelSpec(name="test", modality={"inputs": ["text"], "outputs": ["text"]})
        >>> bib = generate_bibliography(spec)
        >>> "vaswani2017attention" in bib
        True
    """
    references: dict[str, str] = {}

    # Always include core references
    references.update(CORE_REFERENCES)

    # Add modality-specific references
    modalities = _get_modalities(spec)

    if "text" in modalities:
        references.update(TEXT_REFERENCES)
    if "image" in modalities:
        references.update(IMAGE_REFERENCES)
    if "audio" in modalities:
        references.update(AUDIO_REFERENCES)
    if "video" in modalities:
        references.update(VIDEO_REFERENCES)
    if _is_multimodal(modalities):
        references.update(MULTIMODAL_REFERENCES)

    # Add architecture-specific references
    references.update(NAS_REFERENCES)
    references.update(META_LEARNING_REFERENCES)

    # Add objective-specific references
    objectives = _get_objectives(spec)
    for objective in objectives:
        if objective in OBJECTIVE_REFERENCES:
            references.update(OBJECTIVE_REFERENCES[objective])

    # Sort by citation key for determinism
    sorted_refs = sorted(references.items(), key=lambda x: x[0])

    return "\n\n".join(ref for _, ref in sorted_refs) + "\n"


def get_citation_keys(spec: ModelSpec) -> list[str]:
    """Get list of citation keys that will be included in the bibliography.

    Useful for generating \\cite{} commands in the paper.

    Args:
        spec: Model specification.

    Returns:
        Sorted list of BibTeX citation keys.
    """
    keys = list(CORE_REFERENCES.keys())

    modalities = _get_modalities(spec)

    if "text" in modalities:
        keys.extend(TEXT_REFERENCES.keys())
    if "image" in modalities:
        keys.extend(IMAGE_REFERENCES.keys())
    if "audio" in modalities:
        keys.extend(AUDIO_REFERENCES.keys())
    if "video" in modalities:
        keys.extend(VIDEO_REFERENCES.keys())
    if _is_multimodal(modalities):
        keys.extend(MULTIMODAL_REFERENCES.keys())

    keys.extend(NAS_REFERENCES.keys())
    keys.extend(META_LEARNING_REFERENCES.keys())

    objectives = _get_objectives(spec)
    for objective in objectives:
        if objective in OBJECTIVE_REFERENCES:
            keys.extend(OBJECTIVE_REFERENCES[objective].keys())

    return sorted(set(keys))


def get_modality_citations(spec: ModelSpec) -> dict[str, list[str]]:
    """Get citation keys organized by category.

    Args:
        spec: Model specification.

    Returns:
        Dictionary mapping category names to lists of citation keys.
    """
    citations: dict[str, list[str]] = {
        "core": list(CORE_REFERENCES.keys()),
        "nas": list(NAS_REFERENCES.keys()),
        "meta_learning": list(META_LEARNING_REFERENCES.keys()),
    }

    modalities = _get_modalities(spec)

    if "text" in modalities:
        citations["text"] = list(TEXT_REFERENCES.keys())
    if "image" in modalities:
        citations["image"] = list(IMAGE_REFERENCES.keys())
    if "audio" in modalities:
        citations["audio"] = list(AUDIO_REFERENCES.keys())
    if "video" in modalities:
        citations["video"] = list(VIDEO_REFERENCES.keys())
    if _is_multimodal(modalities):
        citations["multimodal"] = list(MULTIMODAL_REFERENCES.keys())

    return citations


def _get_modalities(spec: ModelSpec) -> set[str]:
    """Extract all modalities from spec."""
    modalities = set()
    for m in spec.modality.inputs + spec.modality.outputs:
        modalities.add(m.lower())
    return modalities


def _get_objectives(spec: ModelSpec) -> set[str]:
    """Extract training objectives from spec."""
    return {obj.lower() for obj in spec.training.objective}


def _is_multimodal(modalities: set[str]) -> bool:
    """Check if spec is multimodal (has multiple different modalities)."""
    standard_modalities = {"text", "image", "audio", "video", "3d"}
    found = modalities & standard_modalities
    return len(found) >= 2
