"""SAIP: Summarization-Aware Iterative Pseudo-Labeling for Dense Video Captioning.

Reference implementation of Sections 3.1-3.4 of

    "Summarization-Aware Iterative Pseudo-Labeling for Dense Video Captioning"

The package covers the annotation-free part of the pipeline only:

    saip.candidates   Section 3.1  candidate-event over-generation
    saip.sfs          Section 3.2  SFS scoring (Eqs. 4-10) and greedy selection
    saip.bcnet        Section 3.3  boundary calibration network and iterative refinement
    saip.calibration  Section 3.4  cross-video statistical calibration
    saip.pipeline     orchestration of the full loop and pseudo-label dataset output

Training of the downstream DVC models (PDVC / Vid2Seq) is outside the scope of
this repository.
"""

__version__ = "1.0.0"

#: Dimensionality of the frame-level BLIP visual feature h_t (Eq. 1).
FEATURE_DIM = 768

__all__ = ["FEATURE_DIM", "__version__"]
