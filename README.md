# Object Noun Dictionary

**Author:** Philipp Allgeuer

This repository provides the code for generating the Object Noun Dictionary, an exhaustive collection of simple and compound nouns in the English language (plus alternate spellings, plural forms, and relative word frequencies) as detailed in, and required for, the WACV 2025 paper:

Philipp Allgeuer, Kyra Ahrens, and Stefan Wermter: *Unconstrained Open Vocabulary Image Classification: Zero-Shot Transfer from Text to Image via CLIP Inversion*

The aim of the Object Noun Dictionary is to provide a reference dictionary for image classification and/or object detection class labels for computer vision tasks, in particular open vocabulary ones. The final generated dictionary is available at `datasets/object_noun_curated.json`. For instructions how to regenerate the dataset, refer in particular to the `Noun Curation` section of the `commands.txt` file.

## Citation

The associated WACV 2025 paper is available on [arXiv](https://www.arxiv.org/abs/2407.11211) (see Section 3.2 and Appendix B). If you use this project in your research, please cite this GitHub repository as well as the paper:

```bibtex
@Misc{github_object_noun_dict,
    title = {{O}bject {N}oun {D}ictionary},
    author = {Philipp Allgeuer},
    url = {https://github.com/pallgeuer/object_noun_dictionary},
}

@InProceedings{allgeuer_novic_2025,
    author    = {Philipp Allgeuer and Kyra Ahrens and Stefan Wermter},
    title     = {Unconstrained Open Vocabulary Image Classification: {Z}ero-Shot Transfer from Text to Image via {CLIP} Inversion},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year      = {2025},
```
