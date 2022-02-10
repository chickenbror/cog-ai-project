# import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from datetime import datetime
from models import * # import functions
from dataloader import * #import functions

#-----------------------------------------------------------------------------------
# PARAMETERS FOR DATA
flickr8k_path = '/scratch/lt2318-h21-resources/03-image-captioning/data/flickr8k/'
flickr8K_caps = flickr8k_path + 'captions.txt'
flickr8K_imgs = flickr8k_path + 'Images/'
min_freq=150

# PARAMETERS FOR TRAINING
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
augment_imgs=False, 
batch_size=40
epochs = 10
learning_rate = 1e-4
model_name = "model_name"
#-----------------------------------------------------------------------------------

def train_val_split(dataset, val_portion=0.2):
    random.shuffle(dataset)
    val_split = int( len(dataset) * val_portion)
    train, val = dataset[val_split:], dataset[:val_split]
    print(f"Total no of data: {len(dataset)}. Split to train:val = {len(train)}:{len(val)}.")
    return train, val


# def chart_losses(epochs, epoch_train_val_losses):
#     # ===Draw the line chart===================================
#     the_epochs = list(range(1,epochs+1)) # X-axis
#     train_losses, val_losses = zip(*epoch_train_val_losses)

#     # Lines
#     plt.plot(the_epochs, train_losses, label='train loss')
#     plt.plot(the_epochs, val_losses, label='val loss')


#     plt.title('Losses over iterations')
#     plt.xlabel('epoch')
#     plt.ylabel('train/val loss')
#     plt.legend()
#     plt.show()
    
#=============================================================
def train(model, model_name, trainXY, valXY, label_idx,
          device=device, augment_imgs=False,
          batch_size=48, epochs = 10, learning_rate = 1e-4):
    
    model=model.to(device)

    # Grad descend / Loss fn / Dataset iterators
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#     optimizer = adabound.AdaBound(model.parameters(), lr=learning_rate, final_lr=0.1)
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) # LR adjustment
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.BCELoss()

    train_loader = DataLoader(trainXY, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valXY, batch_size=batch_size, shuffle=False)

    # Epoch (Train+Val) loop

    # Record time lapses and stats in each epoch 
    train_start_time = datetime.now()
    epoch_train_val_losses = []

    for e in range(1,epochs+1):
        start_time = datetime.now()

        #===TRAIN==============================================
        model.train()
        train_loss = 0
        for i, xy in enumerate(train_loader):
            imgs = xy[0]
            if augment_imgs:
                imgs = augment_img_tensors(imgs) #Random augment train imgs
                
            targets = xy[label_idx] # 1,2,3 for anp, adj, noun
            imgs, targets = imgs.to(device), targets.to(device)

            output = model(imgs)

            loss = criterion(output, targets)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {e} avg train loss {train_loss/(i+1)}", end='\r')
        print()
#         scheduler.step() # Adject LR after an epoch
        
        #===Save model============================================
        torch.save(model, f"trained-models/{model_name}.pt")

        #===VAL/TEST==============================================
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, xy in enumerate(val_loader):
                imgs = xy[0] # No augment
                targets = xy[label_idx]
                imgs, targets = imgs.to(device), targets.to(device)
                output = model(imgs)

                loss = criterion(output, targets)
                val_loss += loss.item()
                print(f"Epoch {e} avg val loss   {val_loss/(i+1)}", end='\r')
        print()

        #===EPOCH INFO and STATS=======================================
        lapsed_time = datetime.now()-start_time
        total_lapsed_time = datetime.now()-train_start_time
        print(f"Epoch {e} train+val time:",str(lapsed_time).split('.')[0], 
              "Total lapsed time:", str(total_lapsed_time).split('.')[0],
             )
        epoch_train_val_losses.append((train_loss, val_loss))

    #Finish all epochs    
    print('DONE!')
#     chart_losses(epochs, epoch_train_val_losses)
    
    return model

def train_crossmodel(model, model_name, trainXY, valXY,
          device=device, consider_anp_targets=False, augment_imgs=False, 
          batch_size=40, epochs = 10, learning_rate = 1e-4):
    
    model.consider_anp_targets = consider_anp_targets
    if consider_anp_targets and model.consider_anp_targets:
        print('Consider ANP gold labels when training.')
    
    model=model.to(device)

    # Grad descend / Loss fn / Dataset iterators
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#     optimizer = adabound.AdaBound(model.parameters(), lr=learning_rate, final_lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) # LR adjustment
    criterion = nn.BCELoss()

    train_loader = DataLoader(trainXY, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valXY, batch_size=batch_size, shuffle=False)

    # Epoch (Train+Val) loop

    # Record time lapses and stats in each epoch 
    train_start_time = datetime.now()
    epoch_train_val_losses = []

    for e in range(1,epochs+1):
        start_time = datetime.now()

        #===TRAIN==============================================
        model.train()
        train_loss = 0
        for i, xy in enumerate(train_loader):
            imgs = xy[0]
            if augment_imgs:
                imgs = augment_img_tensors(imgs) #Random augment train imgs
            adj_targets, noun_targets = xy[2], xy[3] # 1,2,3 for anp, adj, noun
            
            imgs, adj_targets, noun_targets = imgs.to(device), adj_targets.to(device), noun_targets.to(device)

            cross_out, (adj_out, noun_out) = model(imgs)
            
            # The loss is based on learning adjs/nouns separately
            if consider_anp_targets:
                anp_targets = xy[1].to(device)
                loss = criterion(cross_out, anp_targets) + criterion(adj_out, adj_targets) + criterion(noun_out, noun_targets)
            else:
                loss = criterion(adj_out, adj_targets) + criterion(noun_out, noun_targets)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {e} avg train loss {train_loss/(i+1)}", end='\r')
        print()
#         scheduler.step() # Adject LR after an epoch
        
        #===Save model============================================
        torch.save(model, f"trained-models/{model_name}.pt")

        #===VAL/TEST==============================================
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, xy in enumerate(val_loader):
                imgs = xy[0] # No augment
                adj_targets, noun_targets = xy[2], xy[3] # 1,2,3 for anp, adj, noun
                imgs, adj_targets, noun_targets = imgs.to(device), adj_targets.to(device), noun_targets.to(device)
                cross_out, (adj_out, noun_out) = model(imgs)

                if consider_anp_targets:
                    anp_targets = xy[1].to(device)
                    loss = criterion(cross_out, anp_targets) + criterion(adj_out, adj_targets) + criterion(noun_out, noun_targets)
                else:
                    loss = criterion(adj_out, adj_targets) + criterion(noun_out, noun_targets)
                val_loss += loss.item()
                print(f"Epoch {e} avg val loss   {val_loss/(i+1)}", end='\r')
        print()

        #===EPOCH INFO and STATS=======================================
        lapsed_time = datetime.now()-start_time
        total_lapsed_time = datetime.now()-train_start_time
        print(f"Epoch {e} train+val time:",str(lapsed_time).split('.')[0], 
              "Total lapsed time:", str(total_lapsed_time).split('.')[0],
             )
        epoch_train_val_losses.append((train_loss, val_loss))

    #Finish all epochs    
    print('DONE!')
#     chart_losses(epochs, epoch_train_val_losses)
    
    return model




def main():
    
    # Prepare and split datasets
    print("Parsing...")
    adj_classes, noun_classes, ANP_classes, fn_ANPs = get_imgs_with_ANPs(flickr8K_caps, min_freq=min_freq)
    data = list(fn_ANPs.items())
    
    print("Splitting data...")
    train_fn_anp, val_fn_anp = train_val_split(data)
    
    # Load pickled datasets or make new ones
    try:
        with open("trainXY.pkl", "rb") as trainf, open("valXY.pkl", "rb") as valf:
            print("Load previously saved train and val data...")
            trainXY = pickle.load(trainf)
            valXY = pickle.load(valf)
    except FileNotFoundError:
        print("Reading image files and creating multi-hot encodings...")
        trainXY = convert_fn_anp_to_xy(train_fn_anp, flickr8K_imgs, ANP_classes, adj_classes, noun_classes)
        valXY = convert_fn_anp_to_xy(val_fn_anp, flickr8K_imgs, ANP_classes, adj_classes, noun_classes)
        with open("trainXY.pkl", "wb") as trainf, open("valXY.pkl", "wb") as valf:
            pickle.dump(trainXY, trainf)
            pickle.dump(valXY, valf)
    
    
    # Train
    
    # TRAINING CHOICE
    choice = ""  # adj-tagger, noun-tagger, anp-tagger, fact-an, fact-reps-plus, fact-sigmoids-plus
    
    while choice not in ("anp-tagger", "adj-tagger", "noun-tagger", 
                         "fact-an", "fact-reps-plus", "fact-sigmoids-plus"):
        choice = input("\nPlease choose one of the models to train: \nadj-tagger\nnoun-tagger\nanp-tagger\nfact-an\nfact-reps-plus\nfact-sigmoids-plus\n")
        
    print(f"\nTraining {choice}...")
    
    # Simple models
    if choice in ("anp-tagger", "adj-tagger", "noun-tagger"):
        if choice=="anp-tagger":
            num_classes = len(ANP_classes)
            label_idx = 1
        if choice=="adj-tagger":
            num_classes = len(adj_classes)
            label_idx = 2
        if choice=="noun-tagger":
            num_classes = len(noun_classes)
            label_idx = 3
        
    
        m = ResnetTagger(num_classes)
        m = train(model=m, model_name=model_name, trainXY=trainXY, valXY=valXY, label_idx=label_idx, 
                    device=device, augment_imgs=augment_imgs,
                    batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)
        
    # Factorisation models
    elif choice in ("fact-an", "fact-reps-plus", "fact-sigmoids-plus"):
        
        if choice=="fact-an":
            m = CrossTagger( len(adj_classes), len(noun_classes) )
            consider_anp_targets=False
            
        if choice=="fact-reps-plus":
            m = CrossTagger( len(adj_classes), len(noun_classes), fractorise_from="reps")
            consider_anp_targets=True
            
        if choice=="fact-sigmoids-plus":
            m = CrossTagger( len(adj_classes), len(noun_classes), fractorise_from="sigmoids")
            consider_anp_targets=True
            
        m = train_crossmodel( model=m, model_name=model_name, trainXY=trainXY, valXY=valXY,
                  device=device, consider_anp_targets=consider_anp_targets, augment_imgs=augment_imgs, 
                  batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)
    

        
        
if __name__ == '__main__':
    main()