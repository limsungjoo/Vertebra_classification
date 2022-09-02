import torch
import torch.nn as nn
import torch.optim as optim

def load_resnet(resnet_type, pretrained=True):
    if resnet_type == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif resnet_type == 'resnet34':
        return resnet34(pretrained=pretrained)
    elif resnet_type == 'resnet50':
        return resnet50(pretrained=pretrained)
    else:
        raise ValueError("resnet18, resnet34, and resnet50 are supported now.")

class image_combineNet(nn.Module):
    def __init__(self,n_classes,resnet_type,base_n_filter = 16,savefeature=False):
        super(image_combineNet,self).__init__()

        self.n_classes = n_classes
        self.base_n_filter = base_n_filter
        self.savefeature = savefeature
        
        self.left_resnet = load_resnet(resnet_type,pretrained=True)
        self.right_resnet = load_resnet(resnet_type,pretrained=True)
        
        self.left_resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.left_resnet.fc.in_features, 100)
        )
         
        self.right_resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.right_resnet.fc.in_features, 100)
        )
        
        self.concat_layer = nn.Sequential(

        )
        
    def forward(self,img_left,img_right):
        left_fc = self.left_resnet(img_left)
        right_fc = self.right_resnet(img_right)
        concat_fc = torch.cat((left_fc, right_fc), 1)
        output =  self.concat_layer
        
        if self.savefeature:
            output = nn.Linear(200, 100)(concat_fc)
            
            return output
        
        else:    
            output = nn.Linear(200, 100)(concat_fc)
            output = nn.Linear(100, self.n_classes)(output)
            
            return output

def main():
    net = image_combineNet(2,"resnet34")
    

if __name__ == '__main__':
    main()