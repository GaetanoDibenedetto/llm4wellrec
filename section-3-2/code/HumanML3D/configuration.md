## Configuration

- Python's version: **3.8.20**;
- Create and enable a virtual enviroment, for instance `.venv`;
- Install all the depencencies from the `requirements.txt` file;
- Download the `en_core_web_sm` model using the command: `python -m spacy download en_core_web_sm`;
- Download the datasets:
  - Follow the AMASS dataset installation instructions provided in the `raw_pose_processing.ipynb` file;
  - To obtain the AMASS dataset annotations, download the zipped folder `texts.zip` from the [original repository](https://github.com/EricGuo5513/HumanML3D/blob/main/HumanML3D/texts.zip). Once downloaded, move it into the HumanML3D folder and extract its contents;
  - To obtain the HumanAct12 dataset poses, download the zipped folder `humanact12.zip` from the [original repository](https://github.com/EricGuo5513/HumanML3D/blob/main/pose_data/humanact12.zip). Once downloaded, move it into the `pose_data` folder and extract its contents;
