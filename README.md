# Evaluating Generative Text-to-Motion for Occupational Wellbeing Recommender Systems

This repository contains the official implementation, technical details, scripts, and additional visualizations for the paper:
**Evaluating Generative Text-to-Motion for Occupational Wellbeing Recommender Systems**
by _Gaetano Dibenedetto, Stefano Labianca, Andrea Romano, and Pasquale Lops_.

---

## 📊 Phase 1: Zero-Shot Evaluation

To reproduce the zero-shot quantitative evaluation:

- The progressive prompting strategies (Qualitative, Direct Quantitative, and Relative Anatomical) are located in `data/prompts/`.
- Run the evaluation script to test the models on the overhead reaching task and calculate the Mean Absolute Error (MAE) and Mean Relative Error (MRE):
  ```bash
  python ...
  ```

## 🏋️ Phase 2: Domain Adaptation & Custom Dataset

Because standard datasets like HumanML3D are omnidirectional MoCap, we constructed a custom dataset of occupational lifting scenarios starting from [SAFELIFT](https://github.com/GaetanoDibenedetto/IUI26) videos. We augmented this data using horizontal flipping, resulting in 578 occupational lifting poses and 2,312 textual descriptions.

- 3D poses were extracted from monocular video using SMPLer-X.
- To run the fine-tuning pipeline for MoMask (either Task-Specific or Mixed-Domain):
  ```bash
  python ...
  ```

### 📝 Dataset Annotation Process

As noted in the paper, standard HumanML3D annotations are highly descriptive (averaging 12 words) and capture micro-actions. In contrast, our occupational annotations were generated via an automated process, resulting in shorter (averaging 9.25 words) and more repetitive descriptions tailored to safety constraints.

Here is how the 2,312 text-motion pairs were generated (4 descriptions per pose):

1. **Automated Extraction:** [Briefly explain here how you extracted the text—e.g., "We used a deterministic Natural Language Generation template based on the calculated RNLE variables like Horizontal distance and Vertical height."]
2. **Template Variations:** For each extracted 3D posture, we generated 4 semantic variations to create the text-motion pairs.
3. **Example Annotations:**
   - _Description 1 (Direct):_ "A worker lifting a box from the floor."
   - _Description 2 (Safety-focused):_ "A person safely lifting a load with a straight spine."
   - _Description 3 (Metric-based):_ "Worker lifting a 15kg load close to the body core."
   - _Description 4 (Action-oriented):_ "Bending the knees to lift an object from the ground."
