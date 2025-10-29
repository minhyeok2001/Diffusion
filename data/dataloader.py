import torchvision.transforms as T
import torch
import os
from PIL import Image

"""
상황 : data/dataset/afhq/train , val / cat dog wild 내부에 사진이 있음

폴더에서 가져올때, class 명을 그냥 정수 매핑하여 가져 올 것인가?  -> CS492에서는 정수 매핑으로 받아오는 것을 확인. 이를 채용

dataloader를 만들건데, 사진하고 클래스를 매핑해주는 데이터로더를 만들거임 

한 배치 안에서 cat dog wild가 골고루 나왔으면 좋겠으므로, 하나씩 뽑는 방식을 채용 
"""

"""
FLOW

0. 배치 내부에 균형있게 데이터를 가져오기를 기대하므로, cat : 5153, dog : 4739, wild : 4738 를 최소 단위로 잘라서 리스트를 가져오자

1. 데이터들을 리스트로 받아온다, 단 받아올때 1,2,3 정수 매핑 해두기. 리턴은 [이미지,정수값]

2. 배치는 무조건 3의 배수로 진행. 
"""

# torch.Size([3, 3, 3, 512, 512]) 로 들어오는 문제를 방지하기 위한 함수
def collate_ft(batch):
    imgs_list, labels_list = zip(*batch)
    
    imgs = torch.stack(imgs_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    
    B,K,C,H,W = imgs.shape
    imgs = imgs.view(B*K,C,H,W) 
    labels = labels.view(-1)         
    return imgs, labels

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,test=False):
        super().__init__()
        
        # 1. 리스트 리턴하기 -> 경로는 최상위 폴더 실행 기준
        BASE_PATH = "data/dataset/afhq/train"

        self.test = test
        self.basepath = BASE_PATH
        self.transform = T.Compose([
                T.Resize((128, 128)),
                T.ToTensor()
            ])  
        if self.test :
            ## 어차피 reconstruction이니까 클래스 불균형보다는 우선 cat 남은거 많이 활용하도록 설계
            self.len = 300 ## 300개만 써보자 test로
            test_cat_path = os.path.join(BASE_PATH,"cat")
            test_cat_list = os.listdir(test_cat_path)[-300:]
            self.lists = test_cat_list
            
        else :
            category = ["cat","dog","wild"]
            paths = {}
            lists = {}
            length = {}
            
            for item in category:
                paths[item] = os.path.join(BASE_PATH, item)
                lists[item] = os.listdir(paths[item])
                length[item] = len(lists[item])
                
            min_len = min(length["cat"],length["dog"],length['wild']) ## 4738개
            
            self.lists = lists
            self.category = category
            self.len = min_len

        
    def __len__(self):
        ## 원래는 여기서 전체 데이터셋 개수를 반환해야하지만, 3종류를 하나씩 묶어서 리턴할거임. 즉 idx 상으로는 1/3이 되어야하므로 그냥 min_len 사용
        return self.len
    
    def __getitem__(self,idx):
        ## 여기서는 path만 받아서, idx에 맞는걸 로드해서 리턴할 수 있도록 함 
        
        if self.test :
            path = os.path.join(self.basepath,'cat',self.lists[idx])
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            #print("img shape : ", img.shape) # torch.Size([3, 512, 512])
            return img, torch.tensor(1)
            
        else :
            imgs = []
            labels = []
            for category_idx,(item,category) in enumerate(zip(self.lists.values(),self.category)) :
                path = os.path.join(self.basepath,category,item[idx])
                img = Image.open(path).convert("RGB")
                img = self.transform(img)
                #print("img shape : ", img.shape) # torch.Size([3, 512, 512])
                imgs.append(img)     ## cat : 1 , dog : 2 , wild : 3
                labels.append(category_idx+1) 

        return torch.stack(imgs,dim=0), torch.tensor(labels)
        

def test():
    data = CustomDataset(test=True)
    dataloader = torch.utils.data.DataLoader(data)
    
    a,b = next(iter(dataloader))
    print(a,b)
    
#test()


print("dataloader.py executed...")