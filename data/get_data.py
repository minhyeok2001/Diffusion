import os
import kaggle

"""
FLOW
1. 데이터를 찾아보고, dir 없으면 생성, 있으면 그냥 사용하는 식으로
2. 데이터 다운시에는 kaggle 통해서 자동으로 받을 수 있도록 -> kaggle.json 활용
https://www.kaggle.com/datasets/andrewmvd/animal-faces
"""

DATA_PATH = "../data/dataset"

## 실행되는 워킹디렉토리는 아마 vae 같은 개별 폴더일 것이므로, data 폴더 내부를 검색
if not os.path.exists(DATA_PATH) :
    os.mkdir(DATA_PATH)
    
    ## kaggle.json 찾고 다운로드 -> mac 환경
    if os.path.exists(os.path.expanduser("~/.kaggle")) :  ## 여기서도 expanduser 꼭 써야지 !! 안쓰면 이거 인식 못함
        os.system("chmod 600 ~/.kaggle/kaggle.json")

    ## kaggle.json이 없다면, colab인 것으로 인지하고 file 업로드 요청
    else :
        from google.colab import files
        uploaded = files.upload()
        os.makedirs("/root/.config/kaggle", exist_ok=True)
        os.rename(list(uploaded)[0], "/root/.config/kaggle/kaggle.json") 
        
    
    print("Downloading AFHQ dataset from Kaggle...")
    kaggle.api.dataset_download_files(
        "andrewmvd/animal-faces",
        path=DATA_PATH,
        unzip=True,
        quiet=False 
    )
    
    print("Download Complete !!")
    
    
    