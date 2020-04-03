import importlib
from IPython.display import FileLink


__all__ = ['mount_gdrive', 'output_file_link']


def mount_gdrive() -> str:
    """Mount Google Drive storage of the current Google account and return the root path.

    Functionality only available in Google Colab Enviroment; otherwise, it raises a RuntimeError.
    """
    if (importlib.util.find_spec("google.colab") is None):
        raise RuntimeError("Cannot mount Google Drive outside of Google Colab.")

    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=True)
    root_dir = "/content/gdrive/My Drive/"
    
    return root_dir


def output_file_link(path:str):  
    FileLink(path)
