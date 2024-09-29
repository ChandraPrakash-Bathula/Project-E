import pymupdf
import docx
import requests
from bs4 import BeautifulSoup
import torch
from transformers import is_torch_npu_available

def extract_content_from_file(file_paths: list[str] | str):
        text = ""
        
        file_paths = [file_paths] if isinstance(file_paths, str) else file_paths
        try:
            for file_path in file_paths:
                # Handle PDF file
                if file_path.endswith(".pdf"):
                    pdf = pymupdf.open(file_path)
                    for page in pdf:
                        text += page.get_text()

                # Handle word file
                elif file_path.endswith(".docx"):
                    doc = docx.Document(file_path)
                    for para in doc.paragraphs:
                        text += para.text # + "\n"

                # Handle webpage
                elif file_path.startswith("http://") or file_path.startswith("https://"):
                    response = requests.get(file_path)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text += soup.get_text(separator='\n')

                # Handle plain text files
                elif file_path.endswith(".txt"):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text += file.read()
                
                # Handle any other file like code file
                try:
                    with open(file_path, 'r') as file:
                        text += file.read()
                except:
                    raise ValueError("Couldn't read the given file: {}".format(file_path))
        
        except Exception as e:
            raise ValueError("Error whil reding the file: {}".format(file_path))

        return text.strip()

def get_current_device():
    """
    Returns the name of the available device
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    elif is_torch_npu_available():
        return "npu"
    else:
        return "cpu"