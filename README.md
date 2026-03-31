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

### 📝 Dataset Annotation Process

As noted in the paper, standard HumanML3D annotations are highly descriptive (averaging 12 words) and capture micro-actions. In contrast, our occupational annotations were generated via an automated process, resulting in shorter (averaging 9.25 words) and more repetitive descriptions tailored to safety constraints.

You can find the implementation for generating these annotations in the `section-3-2/code/annotation-script` folder. To run the automated script (`build_prompts.py`):
1. Ensure Python 3.10 is installed.
2. Create and enable a virtual environment (e.g., `.venv`) and install dependencies via `pip install -r requirements.txt`.
3. The script extracts metadata from a JSON file and outputs the generated text prompts and a CSV file mapping each video to its texts.

**How Text-Motion Pairs were Created:**
Each extracted 3D pose in our [annotations](section-3-2/code/annotation-script/annotations.json) relies on metadata capturing demographic information (gender, age), box dimensions, and initial/final handled object positions. For each pose, the script generates four semantic variations (yielding 2,312 text-motion pairs) along with mirrored equivalents (`M*.txt`):

1. **Direction Description (`format_with_direction`):** Determines the primary action by comparing the initial (`height_start`) and final (`height_end`) vertical positions of the box. If the final height is equal to or greater than the starting height, the action is classified as a "lifting" motion "from the ground". Otherwise, it is classified as a "laying" motion "on the ground".
   - _Example:_ "a person is lifting a box from the ground."
   
2. **Gender Description (`format_with_gender`):** Focuses exclusively on the demographic attribute of the subject based on the `subject_gender` metadata. To add linguistic variance, it abstracts the handled object to a generic "something" and explicitly denotes the bimanual nature of the task ("with both hands").
   - _Example:_ "a man is moving something, with both hands."
   
3. **Gender + Action Description (`format_with_gender_and_action`):** Combines the components of the previous two variations. It incorporates the subject's gender and adapts the verb depending on the vertical displacement vector ("picking up" for upwards motion, "putting down" for downwards motion).
   - _Example:_ "a man is picking up something, with both hands."
   
4. **Ergonomic Assessment using RNLE variables (`format_with_niosh`):** Translates complex biomechanical variables into natural language by leveraging the **Distance Multiplier (DM)** from the **Revised NIOSH Lifting Equation (RNLE)**. 
   - First, the absolute vertical displacement (`D`) is calculated and clamped within standard NIOSH boundaries ($25 \text{ cm} \le D \le 175 \text{ cm}$).
   - The Distance Multiplier is computed using the formula $DM = 0.82 + (4.5 / D)$.
   - The action's specific $DM$ is then compared against the median $DM$ computed across the entire dataset. This relative comparison dictates a quantitative modifier ("slightly" or "much"), which is paired with the direction of the movement ("higher" or "lower") to form the final phrase.
   - _Example:_ "a person is moving a box to a slightly higher position."

### 🛠️ Fine-tuning Pipeline

To run the fine-tuning pipeline for MoMask (either Task-Specific or Mixed-Domain):

1. Generate the annotations for the **SAFELIFT** dataset using the steps above.
2. Extract 3D poses from the monocular videos using [SMPLer-X](https://github.com/MotrixLab/SMPLer-X).
3. Configure the [HumanML3D project](https://github.com/EricGuo5513/HumanML3D) within `section-3-2/code/HumanML3D`, and process the extracted 3D poses to generate the motion representations required for training.
4. Configure the [MoMask project](https://github.com/EricGuo5513/momask-codes) within `section-3-2/code/MoMask`.
5. Proceed with fine-tuning the model by executing the following Bash scripts:

   ```bash
   # run-train-t2m.sh: Run the finetuning process for the M-Trasformer
   python train_t2m_transformer.py --is_continue --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns --gpu_id 0 --dataset_name t2m --batch_size 10 --max_epoch 514 --vq_name rvq_nq6_dc512_nc512_noshare_qdp0.2
   ```

   ```bash
   # run-train-res.sh: Run the finetuning process for the R-Trasformer
   python train_res_transformer.py --is_continue --name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw --gpu_id 0 --dataset_name t2m --batch_size 10 --max_epoch 490 --vq_name rvq_nq6_dc512_nc512_noshare_qdp0.2 --cond_drop_prob 0.2 --share_weight
   ```

**Technical Aspects:**
- The retraining process was carried out over 150 epochs using an NVIDIA RTX 3090 GPU with 24 GB of VRAM, requiring approximately 15 days of computation.
- The fine-tuning process was performed directly on the original model checkpoints, extending training by an additional 50 epochs. This stage was carried out on an NVIDIA GTX Titan X GPU equipped with 12 GB of VRAM, with a total computation time of approximately 6 hours.
