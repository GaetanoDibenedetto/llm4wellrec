## Configurazione

- Impostare la versione 3.8.20 di Python
- Creare e abilitare un ambiente virtuale
- Installare le dipendenze tramite il `requirements.txt`
- Installare il modello `en_core_web_sm` con il comando: `python -m spacy download en_core_web_sm`
- Scaricare i dataset
  - Seguire l'installazione del dataset AMASS presente nel file `raw_pose_processing.ipynb`
  - Per ottenere le annotazioni del dataset AMASS, scaricare la cartella zippata `texts.zip` dal [repository originale](https://github.com/EricGuo5513/HumanML3D/blob/main/HumanML3D/texts.zip). Una volta scaricata, spostarla all'interno della cartella `HumanML3D` e poi unzippa
  - Per ottenere le pose del dataset HumanAct12, scaricare la cartella zippata `humanact12.zip` dal [repository originale](https://github.com/EricGuo5513/HumanML3D/blob/main/pose_data/humanact12.zip), spostarla all'interno della cartella `pose_data` e poi unzippa
  - Estrarre la cartella zippata `pose_data/humanact12.zip` lasciando il suo contenuto nella cartella `pose_data`
