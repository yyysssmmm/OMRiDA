import os
import sys
import zipfile
from datamodule import CROHMEDatamodule
from lit_posformer import LitPosFormer
from pytorch_lightning import Trainer, seed_everything
import os

base_dir = os.path.dirname(__file__)

seed_everything(7)

def cal_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    if m*n == 0:
        return m+n
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range (m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            a = dp[i-1][j] + 1
            b = dp[i][j-1] + 1
            c = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                c += 1
            dp[i][j] = min(a, b, c)
    return dp[m][n]

def main(category, ckp_path):

    trainer = Trainer(logger=False, accelerator="gpu", devices=1)

    dm = CROHMEDatamodule(test_year=category, eval_batch_size = 1)

    model = LitPosFormer.load_from_checkpoint(ckp_path)
    trainer.test(model, datamodule=dm)
    caption = {}
    with zipfile.ZipFile(os.path.join(base_dir, "test_data_for_PosFormer.zip")) as archive:
        with archive.open(f"data/{category}/caption.txt", "r") as f:
            caption_lines = [line.decode('utf-8').strip() for line in f.readlines()]
            for caption_line in caption_lines:
                caption_parts = caption_line.split()
                caption_file_name = caption_parts[0]
                caption_string = ' '.join(caption_parts[1:])
                caption[caption_file_name] = caption_string

    
    with zipfile.ZipFile(os.path.join(base_dir, "result.zip")) as archive:
        exprate=[0,0,0,0]
        file_list = archive.namelist()
        txt_files = [file for file in file_list if file.endswith('.txt')]
        for txt_file in txt_files:
            file_name = txt_file.rstrip('.txt')
            with archive.open(txt_file) as f:
                lines = f.readlines()
                pred_string = lines[1].decode('utf-8').strip()[1:-1]
                if file_name in caption:
                    caption_string = caption[file_name]
                else:
                    print(file_name,"not found in caption file")
                    continue
                caption_parts = caption_string.strip().split()
                pred_parts = pred_string.strip().split()
                if caption_string == pred_string:
                    exprate[0]+=1
                else:
                    error_num=cal_distance(pred_parts,caption_parts)
                    if error_num<=3:
                        exprate[error_num]+=1
        tot = len(txt_files)
        exprate_final=[]
        for i in range(1,5):
            exprate_final.append(100*sum(exprate[:i])/tot)
        print(category,"exprate",exprate_final)



TEST_CATEGORY = ["pme_2014", "pme_2016", "pme_2019", "pme_im2latex", "hme_2014", "hme_2016", "hme_2019"]

if __name__ == "__main__":

    import argparse
    import yaml
    import sys

    ckp_path = os.path.join(base_dir, "best.ckpt")

    def load_config(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.abspath(os.path.join(base_dir, "../../config.yaml")))
    args = parser.parse_args()
    config = load_config(args.config)

    for category in TEST_CATEGORY:
        main(category, ckp_path)


    set_seed(config["misc"]["seed"])
