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

- 3D poses were extracted from monocular video using [SMPLer-X](https://github.com/MotrixLab/SMPLer-X);
- To run the fine-tuning pipeline for MoMask (either Task-Specific or Mixed-Domain):
  - Generate the annotations for the **SAFELIFT** dataset using the code inside the `section-3-2/code/annotation-script` folder;
  - Use SMPLer-X for the extraction of 3D poses;
  - After configuring the [HumanML3D project](https://github.com/EricGuo5513/HumanML3D), process the previously extracted 3D poses in order to generate the motion representations required for training;
  - After configuring the [MoMask project](https://github.com/EricGuo5513/momask-codes), it is possible to proceed with the fine-tuning of the model by executing the following Bash scripts:

  ```bash
  # run-train-t2m.sh: Run the finetuning process for the M-Trasformer
  python train_t2m_transformer.py --is_continue --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns --gpu_id 0 --dataset_name t2m --batch_size 10 --max_epoch 514 --vq_name rvq_nq6_dc512_nc512_noshare_qdp0.2
  ```

  ```bash
  # run-train-res.sh: Run the finetuning process for the R-Trasformer
  python train_res_transformer.py --is_continue --name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw --gpu_id 0 --dataset_name t2m --batch_size 10 --max_epoch 490 --vq_name rvq_nq6_dc512_nc512_noshare_qdp0.2 --cond_drop_prob 0.2 --share_weight
  ```

- Techinal Aspects:
  - The retraining process was carried out over 150 epochs using an NVIDIA RTX 3090 GPU with 24 GB of VRAM, requiring approximately 15 days of computation;
  - The fine-tuning process was performed directly on the original model checkpoints, extending training by an additional 50 epochs. This stage was
    carried out on an NVIDIA GTX Titan X GPU equipped with 12 GB of VRAM, with a total computation time of approximately 6 hours.

### 📝 Dataset Annotation Process

As noted in the paper, standard HumanML3D annotations are highly descriptive (averaging 12 words) and capture micro-actions. In contrast, our occupational annotations were generated via an automated process, resulting in shorter (averaging 9.25 words) and more repetitive descriptions tailored to safety constraints.

Here is how the 2,312 text-motion pairs were generated (4 descriptions per pose):

1. **Automated Extraction:** Broadly speaking, we prepared a set of templates for our annotations. The annotations are sourced from a JSON file, where each entry contains metadata associated with a lifting action recorded in a video, including the subject's demographic information, the sizes of a box being moved and the initial and final positions of the handled object;
2. **Template Variations:** For each extracted 3D posture, four semantic variations were generated to construct the text-motion pairs. These variations capture the direction of the movement, the gender of the subject, a combination of both, and an ergonomic assessment derived from the vertical displacement of the handled object;
3. **Example Annotations:**
   - _Description 1 (Direction):_ "a person is lifting a box from the ground."
   - _Description 2 (Gender):_ "a man is moving something, with both hands."
   - _Description 3 (Direction and Gender):_ "a man is picking up something, with both hands."
   - _Description 4 (Ergonomic Assessment):_ "a person is moving a box to a slightly higher position."

You can find the full implementation inside the `section-3-2/code/annotation-script` folder.
