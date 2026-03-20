import spacy

from tqdm import tqdm
import codecs as cs
from os.path import join as pjoin
import os
import csv
import pandas as pd
import shutil

BASE_PATH = "../video/annotation-script"
PROCESSED_FILE_PATH = "./processed_text"
CORPUS_PATH = "./corpus.csv"

nlp = spacy.load('en_core_web_sm')

def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list


def process_humanml3d(corpus):
    # text_save_path = './dataset/pose_data_raw/texts'
    desc_all = corpus
    for i in tqdm(range(len(desc_all))):
        caption = desc_all.iloc[i]['caption']
        start = desc_all.iloc[i]['from']
        end = desc_all.iloc[i]['to']
        name = desc_all.iloc[i]['new_joint_name']
        word_list, pose_list = process_text(caption)
        tokens = ' '.join(['%s/%s'%(word_list[i], pose_list[i]) for i in range(len(word_list))])
        processed_file_name = name.replace('npy', 'txt')
        
        with cs.open(pjoin(PROCESSED_FILE_PATH, processed_file_name), 'a+') as f:
            f.write('%s#%s#%s#%s\n'%(caption, tokens, start, end))


def process_kitml(corpus):
    text_save_path = './dataset/kit_mocap_dataset/texts'
    desc_all = corpus
    for i in tqdm(range(len(desc_all))):
        caption = desc_all.iloc[i]['desc']
        start = 0.0
        end = 0.0
        name = desc_all.iloc[i]['data_id']
        word_list, pose_list = process_text(caption)
        tokens = ' '.join(['%s/%s' % (word_list[i], pose_list[i]) for i in range(len(word_list))])
        with cs.open(pjoin(text_save_path, name + '.txt'), 'a+') as f:
            f.write('%s#%s#%s#%s\n' % (caption, tokens, start, end))



class CorpusHandler:    
    def __get_npy_names(self, file_path):
        file_name = os.path.basename(file_path).replace(".txt", ".npy")

        return file_name

    def __read_from_csv(self, csv_path):
        csv_rows = []

        with open(csv_path) as csv_file:
            csv_iter =  csv.reader(csv_file, delimiter=",")
            
            for row in csv_iter:
                _, _, text_file_path, mirrored_text_file_path = row

                is_txt_proccessed = self.__pose_arealdy_proccesed(text_file_path)
                is_mirrored_txt_proccessed = self.__pose_arealdy_proccesed(
                    mirrored_text_file_path
                )

                avaiable_files = []
                if not is_txt_proccessed:
                    print(f"[WARNING] Missing files: {text_file_path}")
                else:
                    avaiable_files.append(text_file_path)

                if not is_mirrored_txt_proccessed:
                    print(f"[WARNING] Missing files: {mirrored_text_file_path}")
                else:
                    avaiable_files.append(mirrored_text_file_path)
                
                total_read_lines = 0
                for avaiable_file in avaiable_files:
                    txt_name = os.path.basename(avaiable_file)
                    
                    print(f"Working on: {txt_name}")
                    lines = self.__read(avaiable_file)

                    for line in lines:
                        csv_rows.append([
                            line, 0.0, 0.0, txt_name.replace("txt", "npy")
                        ])
                    total_read_lines += len(lines)
                    
                print(f"Read {total_read_lines} line(s)")

        return csv_rows

    def __read(self, txt_path):
        full_path = os.path.join(BASE_PATH, txt_path)
        with open(full_path) as f:
            lines = f.readlines()
            lines = [ line.replace("\n", "") for line in lines]

            return lines

    def __pose_arealdy_proccesed(self, file_path):
        processed_poses = os.listdir("./HumanML3D/new_joints")
        npy_name = self.__get_npy_names(file_path)

        return npy_name in processed_poses

    def create_corpus(self):
        csv_paths = os.path.join(BASE_PATH, "video-csv")
        csv_files = sorted(
            os.listdir(csv_paths)
        )

        content_per_csv = {}
        for csv_file in csv_files: 
            print(f"Running for '{csv_file}' file")
            csv_path = os.path.join(csv_paths, csv_file)
            csv_rows = self.__read_from_csv(csv_path)
            content_per_csv[csv_file] = csv_rows

        if os.path.exists(CORPUS_PATH):
            os.remove(CORPUS_PATH)

        with open(CORPUS_PATH, mode="w") as csv_handler:
            writer = csv.writer(csv_handler, delimiter=",")
            writer.writerow(["caption", "from", "to", "new_joint_name"])
            row_saved = 0

            for csv_name, rows in content_per_csv.items():
                print(f"Saving content for '{csv_name}' file in {CORPUS_PATH} file")
                writer.writerows(rows)
                row_saved += len(rows)

            print(f"Saved {row_saved} row(s)")


if __name__ == "__main__":
    if os.path.exists(PROCESSED_FILE_PATH):
        shutil.rmtree(PROCESSED_FILE_PATH)
        os.mkdir(PROCESSED_FILE_PATH)

    handler = CorpusHandler()
    handler.create_corpus()
    corpus = pd.read_csv(CORPUS_PATH)
    
    print("Processing corpus using HumanML3D method")
    process_humanml3d(corpus)