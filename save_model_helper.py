import os
import pandas as pd
from datetime import datetime
from google.colab import files
import time
import torch

# import os
# if 'COLAB_GPU' in os.environ:
#    print("I'm running on Colab")

## Salvando o Dataframe no drive e localmente para cada *epoch*
def download_df(df, train_loss, validation_loss, epoch):
  date = datetime.today().strftime('%Y-%m-%d %H:%M:%S').replace(' ', '_')
  file_name = f'SRCNN_epoch={epoch}_{date}_srcnn_avg_train_loss_{train_loss}_avg_validation_loss_{validation_loss}.csv'
  df.to_csv(file_name)
  files.download(file_name)
    

SLEEP_TIME = 30

def save_df(df_path, train_loss, validation_loss, epoch, epoch_duration):
    # Check if the file exists
    if not os.path.isfile(df_path):
        df = pd.DataFrame({'Train Loss': [train_loss], 'Validation Loss': [validation_loss], 'epoch': [epoch], 'epoch_duration [s]': [epoch_duration]})
        download_df(df, train_loss, validation_loss, epoch) # for backup
        df.to_csv(df_path, index=False)
        time.sleep(SLEEP_TIME) # stop concurrency attempt
        return

    df = pd.read_csv(df_path)
    new_df_rows = {'Train Loss': [train_loss], 'Validation Loss': [validation_loss], 'epoch': [epoch], 'epoch_duration [s]': [epoch_duration]}
    df = df.append(pd.DataFrame(new_df_rows))
    download_df(df, train_loss, validation_loss, epoch) # for backup
    df.to_csv(df_path, index=False)
    time.sleep(SLEEP_TIME) # stop concurrency attempt
    
def save_epoch(directory, model, current_epoch, train_loss, validation_loss):
    # saving each epoch, since it's taking forever to train
    date = datetime.today().strftime('%Y-%m-%d %H:%M:%S').replace(' ', '_')
    model_save_path_name = f"{directory}/{date}_epoch={current_epoch}_srcnn_avg_train_loss_{train_loss}_avg_validation_loss_{validation_loss}"
    torch.save(model.state_dict(), model_save_path_name)