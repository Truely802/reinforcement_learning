import glob
import pandas as pd
import os
import sys

module_dir = os.getcwd()
sys.path.append(module_dir)

from src.config import data_dir

def fix_feat():
    pass

if __name__ == '__main__':

    main_dir = ''
    out_path = os.path.join(data_dir, 'categories')
    os.makedirs(out_path, exist_ok=True)

    for filepath in glob.glob(main_dir):
       categ_name = os.path.basename(filepath)
       row_df = pd.read_csv(filepath)
       df_prep = fix_feat(row_df)
       df_prep.to_csv(os.path.join(out_path, categ_name + '.csv'))