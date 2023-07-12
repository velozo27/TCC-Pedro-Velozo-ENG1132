import os
import torch
import re

# - Função para pegar a epoch que paramos de executar na última sessão
def get_epoch_number_from_path(s: str) -> int or None:
    epoch_pattern = r"epoch=(\d+)"
    match = re.search(epoch_pattern, s)

    if match:
        epoch_number = int(match.group(1))
        return epoch_number
    else:
        return None

# - Os arquivos estão sendo salvos no drive como no comando abaixo, a função `get_epoch_number_from_path` acima extrai o número da epoch através do nome do arquivo
def get_most_recent_epoch(directory: str):
    """
    Returns the most recent epoch number from the given directory.

    Args:
        directory (str): The directory path containing the epoch models.

    Returns:
        int: The most recent epoch number.
        str: The model load path name for the most recent epoch.
    """
    most_recent_epoch = -1
    model_load_path_name = ''

    for filename in os.listdir(directory):
        epoch = get_epoch_number_from_path(filename)
        if epoch is not None and epoch > most_recent_epoch:
            most_recent_epoch = epoch
            model_load_path_name = os.path.join(directory, filename)

    return most_recent_epoch, model_load_path_name

def load_state_dict(model, model_load_path_name):
    """
    Loads the state dictionary of a PyTorch model from the specified file.

    Args:
        model (torch.nn.Module): The PyTorch model to load the state dictionary into.
        model_load_path_name (str): The path to the state dictionary file.

    Returns:
        None
    """
    if not model_load_path_name:
        print('model_load_path_name does not exist. This is a new training instance')
    elif torch.cuda.is_available():
        model.load_state_dict(torch.load(model_load_path_name))
    else:
        model.load_state_dict(torch.load(model_load_path_name, map_location=torch.device('cpu')))
    
    return model

def get_current_epoch(index, t, starting_epoch):
    return t
    
#     if EXPERIMENTAL_MODE:
#       return t

#     # finding the current epoch number logic
#     if index == 0 and starting_epoch is None:
#       current_epoch = 0
#       starting_epoch = 0
#     elif starting_epoch >= 0:  
#       current_epoch = starting_epoch + t
#     else:
#       current_epoch = 0
#     return current_epoch