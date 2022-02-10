import torch
from torch import nn
import torchvision # Pre-trained CNN

class ResnetTagger(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        resnet = torchvision.models.resnet101(pretrained=True) # Pretrained ResNet101 model
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.33),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )

        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        out = self.base_model(x)
        return self.sigm(out)
    
    

class CrossTagger(nn.Module):

    def __init__(self, n_adjs, n_nouns, hiddensize=1024, consider_anp_targets=False, fractorise_from="reps"):
        super().__init__()
        
        resnet = torchvision.models.resnet101(pretrained=True) # Pretrained ResNet101 model
        
        # This layer is shared for both adj&noun
        resnet.fc = nn.Sequential(
            nn.Linear(in_features=resnet.fc.in_features, out_features=hiddensize), 
        )
        self.base_model = resnet
        
        # Separate FCs for adj/noun
        self.adj_fc = nn.Sequential(
            nn.Dropout(p=0.33),
            nn.Linear(in_features=hiddensize, out_features=n_adjs),
        )
        
        self.noun_fc = nn.Sequential(
            nn.Dropout(p=0.33),
            nn.Linear(in_features=hiddensize, out_features=n_nouns),
        )
        
        self.cross_fc = nn.Sequential(
            nn.Linear(in_features=n_adjs*n_nouns, out_features=hiddensize),
        )

        self.out_fc = nn.Sequential(
#             nn.Dropout(p=0.2),
            nn.Linear(in_features=hiddensize, out_features=n_adjs*n_nouns),
        )
        
        self.sigm = nn.Sigmoid()
        self.consider_anp_targets = consider_anp_targets
        self.fractorise_from = fractorise_from


    def forward(self, x):
        resnet_out = self.base_model(x) # ---->to be added to the cross-mul 
        # Train to predict adj/noun separately
        adj_reps, noun_reps = self.adj_fc(resnet_out), self.noun_fc(resnet_out)
        adj_out, noun_out = self.sigm(adj_reps), self.sigm(noun_reps)
        
        if self.consider_anp_targets:
            # Multiply the adj/noun reps|sigmoids and go thru another fc layer

            if self.fractorise_from=="sigmoids": # cf. "semantic-ANP"
                cross_reps = self.cross_multiply(adj_out, noun_out) # AxN Sigmoids matrix of size= B, num_adjs*num_nouns 
            if self.fractorise_from=="reps": # cf. "visual-ANP"
                cross_reps = self.cross_multiply(adj_reps, noun_reps) # AxN reps matrix of size= B, num_adjs*num_nouns 

            cross_reps = self.cross_fc(cross_reps) # BxA*N -> BxHidden
            cross_out = self.out_fc( resnet_out+cross_reps ) # BxHidden + BxHidden: ResNet rep + Cross reps
            cross_out = self.sigm(cross_out) 
        else:
            # Simply multiply the adj/noun sigmoids
            cross_out = self.cross_multiply(adj_out, noun_out)
        
        return cross_out, (adj_out, noun_out)
    
    def cross_multiply(self, t1,t2):
        '''
        Given two tensors of size BxN and BxM, return a tensor of size Bx(N*M),
        where each element in dim1 is i*j for i in N and for j in M
        '''
        # Each element in out-tensor is cross-multiplication of N and M
        out = torch.einsum("di,dj->dij", (t1,t2))
        # Flatten all but the batch dim
        return torch.flatten(out, start_dim=1)