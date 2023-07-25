import torch
import torch.nn as nn
import torch.nn.functional as F


batch_size = 256

# logger = open('./error.txt', 'w')

class Models:
    @staticmethod
    def getModel(modelType, numClass, numCell, maxLen):
        if modelType == 'cnn':
            return CNN(maxLen, numClass, numCell)
        elif modelType == 'multimodal_TI':
            return MultiModal_CNN_textImg(maxLen, numClass, numCell)
        elif modelType == 'multimodal_TC':
            return MultiModal_CNN_textCode(maxLen, numClass, numCell)
        elif modelType == 'multimodal_TIC':
            return MultiModal_CNN_TIC(maxLen, numClass, numCell)

class MultiModal_CNN_TIC(nn.Module):
    def __init__(self, maxLen, numClass, numCell):
        super(MultiModal_CNN_TIC, self).__init__()
        
        # text #
        self.maxLen = maxLen
        self.conv2d_filter2 = nn.Conv2d(1, 64, (2, numCell))
        self.conv2d_filter3 = nn.Conv2d(1, 64, (3, numCell))
        self.conv2d_filter4 = nn.Conv2d(1, 64, (4, numCell))
        self.conv2d_filter5 = nn.Conv2d(1, 64, (5, numCell))

        self.maxpool_filter2 = nn.MaxPool1d(maxLen-1)
        self.maxpool_filter3 = nn.MaxPool1d(maxLen-2)
        self.maxpool_filter4 = nn.MaxPool1d(maxLen-3)
        self.maxpool_filter5 = nn.MaxPool1d(maxLen-4)
        self.out = nn.Linear(64*4, 64*4)
        self.out_end = nn.Linear(64*4, numClass)

        # image #
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5),
            nn.ReLU(),

            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5),
            nn.ReLU(),            

            nn.MaxPool2d(kernel_size=5,stride=5),            

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),            

            nn.MaxPool2d(kernel_size=5,stride=5)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*9*9,256),                                      
            nn.ReLU(),                                              
        )

    def forward(self, x_t, x_c, x_i):
        ################ text-layer ################
        x_t = x_t.view(x_t.size(0), 1, x_t.size(1), x_t.size(2))
        
        out2_t = self.conv2d_filter2(x_t)
        out3_t = self.conv2d_filter3(x_t)
        out4_t = self.conv2d_filter4(x_t)
        out5_t = self.conv2d_filter5(x_t)

        out2_t = out2_t.view(out2_t.size(0), out2_t.size(1), -1)
        out3_t = out3_t.view(out3_t.size(0), out3_t.size(1), -1)
        out4_t = out4_t.view(out4_t.size(0), out4_t.size(1), -1)
        out5_t = out5_t.view(out5_t.size(0), out5_t.size(1), -1)

        out2_t = self.maxpool_filter2(out2_t)
        out3_t = self.maxpool_filter3(out3_t)
        out4_t = self.maxpool_filter4(out4_t)
        out5_t = self.maxpool_filter5(out5_t)

        out2_t = out2_t.view(x_t.size()[0], -1)
        out3_t = out3_t.view(x_t.size()[0], -1)
        out4_t = out4_t.view(x_t.size()[0], -1)
        out5_t = out5_t.view(x_t.size()[0], -1)

        out_t = torch.cat((out2_t, out3_t, out4_t, out5_t), dim=1)
        out_t = F.relu(out_t)

        out_t = self.out(out_t)        
        
        ################ code-layer ################
        x_c = x_c.view(x_c.size(0), 1, x_c.size(1), x_c.size(2))
        
        out2_c = self.conv2d_filter2(x_c)
        out3_c = self.conv2d_filter3(x_c)
        out4_c = self.conv2d_filter4(x_c)
        out5_c = self.conv2d_filter5(x_c)

        out2_c = out2_c.view(out2_c.size(0), out2_c.size(1), -1)
        out3_c = out3_c.view(out3_c.size(0), out3_c.size(1), -1)
        out4_c = out4_c.view(out4_c.size(0), out4_c.size(1), -1)
        out5_c = out5_c.view(out5_c.size(0), out5_c.size(1), -1)

        out2_c = self.maxpool_filter2(out2_c)
        out3_c = self.maxpool_filter3(out3_c)
        out4_c = self.maxpool_filter4(out4_c)
        out5_c = self.maxpool_filter5(out5_c)

        out2_c = out2_c.view(x_c.size()[0], -1)
        out3_c = out3_c.view(x_c.size()[0], -1)
        out4_c = out4_c.view(x_c.size()[0], -1)
        out5_c = out5_c.view(x_c.size()[0], -1)

        out_c = torch.cat((out2_c, out3_c, out4_c, out5_c), dim=1)
        out_c = F.relu(out_c)

        out_c = self.out(out_c)

        ################ image-layer ################
        out_i = self.layer(x_i)
                
        try:
            out_i = out_i.view(batch_size,-1)
        except:
        ################ pointWise-layer ################
            out = out_t.mul(out_c)
            out = self.out_end(out)
            return F.log_softmax(out, dim=1)

        if out_i.size(0) != out_c.size(0):
        ################ pointWise-layer ################
            out = out_t.mul(out_c)
            out = self.out_end(out)
            return F.log_softmax(out, dim=1)
        else:
            out_i = self.fc_layer(out_i)

        ################ pointWise-layer ################
        out = out_t.mul(out_c)
        out = out.mul(out_i)
        out = self.out_end(out)

        return F.log_softmax(out, dim=1)

class MultiModal_CNN_textCode(nn.Module):
    def __init__(self, maxLen, numClass, numCell):
        super(MultiModal_CNN_textCode, self).__init__()

        self.maxLen = maxLen
        self.conv2d_filter2 = nn.Conv2d(1, 64, (2, numCell))
        self.conv2d_filter3 = nn.Conv2d(1, 64, (3, numCell))
        self.conv2d_filter4 = nn.Conv2d(1, 64, (4, numCell))
        self.conv2d_filter5 = nn.Conv2d(1, 64, (5, numCell))

        self.maxpool_filter2 = nn.MaxPool1d(maxLen-1)
        self.maxpool_filter3 = nn.MaxPool1d(maxLen-2)
        self.maxpool_filter4 = nn.MaxPool1d(maxLen-3)
        self.maxpool_filter5 = nn.MaxPool1d(maxLen-4)
        self.out = nn.Linear(64*4, 64*4)
        self.out_end = nn.Linear(64*4, numClass)

    def forward(self, x_t, x_c):
        ################ text-layer ################
        x_t = x_t.view(x_t.size(0), 1, x_t.size(1), x_t.size(2))
        
        out2_t = self.conv2d_filter2(x_t)
        out3_t = self.conv2d_filter3(x_t)
        out4_t = self.conv2d_filter4(x_t)
        out5_t = self.conv2d_filter5(x_t)

        out2_t = out2_t.view(out2_t.size(0), out2_t.size(1), -1)
        out3_t = out3_t.view(out3_t.size(0), out3_t.size(1), -1)
        out4_t = out4_t.view(out4_t.size(0), out4_t.size(1), -1)
        out5_t = out5_t.view(out5_t.size(0), out5_t.size(1), -1)

        out2_t = self.maxpool_filter2(out2_t)
        out3_t = self.maxpool_filter3(out3_t)
        out4_t = self.maxpool_filter4(out4_t)
        out5_t = self.maxpool_filter5(out5_t)

        out2_t = out2_t.view(x_t.size()[0], -1)
        out3_t = out3_t.view(x_t.size()[0], -1)
        out4_t = out4_t.view(x_t.size()[0], -1)
        out5_t = out5_t.view(x_t.size()[0], -1)

        out_t = torch.cat((out2_t, out3_t, out4_t, out5_t), dim=1)
        out_t = F.relu(out_t)

        out_t = self.out(out_t)
        
        
        ################ code-layer ################
        x_c = x_c.view(x_c.size(0), 1, x_c.size(1), x_c.size(2))
        
        out2_c = self.conv2d_filter2(x_c)
        out3_c = self.conv2d_filter3(x_c)
        out4_c = self.conv2d_filter4(x_c)
        out5_c = self.conv2d_filter5(x_c)

        out2_c = out2_c.view(out2_c.size(0), out2_c.size(1), -1)
        out3_c = out3_c.view(out3_c.size(0), out3_c.size(1), -1)
        out4_c = out4_c.view(out4_c.size(0), out4_c.size(1), -1)
        out5_c = out5_c.view(out5_c.size(0), out5_c.size(1), -1)

        out2_c = self.maxpool_filter2(out2_c)
        out3_c = self.maxpool_filter3(out3_c)
        out4_c = self.maxpool_filter4(out4_c)
        out5_c = self.maxpool_filter5(out5_c)

        out2_c = out2_c.view(x_c.size()[0], -1)
        out3_c = out3_c.view(x_c.size()[0], -1)
        out4_c = out4_c.view(x_c.size()[0], -1)
        out5_c = out5_c.view(x_c.size()[0], -1)

        out_c = torch.cat((out2_c, out3_c, out4_c, out5_c), dim=1)
        out_c = F.relu(out_c)

        out_c = self.out(out_c)

        ################ pointWise-layer ################
        out = out_t.mul(out_c)
        out = self.out_end(out)

        return F.log_softmax(out, dim=1)

class MultiModal_CNN_textImg(nn.Module):
    def __init__(self, maxLen, numClass, numCell):
        super(MultiModal_CNN_textImg, self).__init__()
        
        # text #
        self.maxLen = maxLen
        self.conv2d_filter2 = nn.Conv2d(1, 64, (2, numCell))
        self.conv2d_filter3 = nn.Conv2d(1, 64, (3, numCell))
        self.conv2d_filter4 = nn.Conv2d(1, 64, (4, numCell))
        self.conv2d_filter5 = nn.Conv2d(1, 64, (5, numCell))

        self.maxpool_filter2 = nn.MaxPool1d(maxLen-1)
        self.maxpool_filter3 = nn.MaxPool1d(maxLen-2)
        self.maxpool_filter4 = nn.MaxPool1d(maxLen-3)
        self.maxpool_filter5 = nn.MaxPool1d(maxLen-4)
        self.out = nn.Linear(64*4, 64*4)
        self.out_end = nn.Linear(64*4, numClass)

        # image #
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5),
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=5,stride=5),            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=5,stride=5)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*9*9,256),                                      
            nn.ReLU(),                                
        )

    def forward(self, x_t, x_i):
        ################ text-layer ################
        x_t = x_t.view(x_t.size(0), 1, x_t.size(1), x_t.size(2))
        
        out2_t = self.conv2d_filter2(x_t)
        out3_t = self.conv2d_filter3(x_t)
        out4_t = self.conv2d_filter4(x_t)
        out5_t = self.conv2d_filter5(x_t)

        out2_t = out2_t.view(out2_t.size(0), out2_t.size(1), -1)
        out3_t = out3_t.view(out3_t.size(0), out3_t.size(1), -1)
        out4_t = out4_t.view(out4_t.size(0), out4_t.size(1), -1)
        out5_t = out5_t.view(out5_t.size(0), out5_t.size(1), -1)

        out2_t = self.maxpool_filter2(out2_t)
        out3_t = self.maxpool_filter3(out3_t)
        out4_t = self.maxpool_filter4(out4_t)
        out5_t = self.maxpool_filter5(out5_t)

        out2_t = out2_t.view(x_t.size()[0], -1)
        out3_t = out3_t.view(x_t.size()[0], -1)
        out4_t = out4_t.view(x_t.size()[0], -1)
        out5_t = out5_t.view(x_t.size()[0], -1)

        out_t = torch.cat((out2_t, out3_t, out4_t, out5_t), dim=1)
        out_t = F.relu(out_t)

        out_t = self.out(out_t)        
        
        ################ image-layer ################
        out_i = self.layer(x_i)
                
        try:
            out_i = out_i.view(batch_size,-1)
        except:
        ################ pointWise-layer ################
            out = self.out_end(out_t)
            return F.log_softmax(out, dim=1)

        if out_i.size(0) != out_t.size(0):
        ################ pointWise-layer ################
            out = self.out_end(out_t)
            return F.log_softmax(out, dim=1)
        else:
            out_i = self.fc_layer(out_i)

        ################ pointWise-layer ################
        out = out_t.mul(out_i)
        out = self.out_end(out)

        return F.log_softmax(out, dim=1)

class CNN(nn.Module):
    def __init__(self, maxLen, numClass, numCell):
        super(CNN, self).__init__()

        self.maxLen = maxLen
        self.conv2d_filter2 = nn.Conv2d(1, 64, (2, numCell))
        self.conv2d_filter3 = nn.Conv2d(1, 64, (3, numCell))
        self.conv2d_filter4 = nn.Conv2d(1, 64, (4, numCell))
        self.conv2d_filter5 = nn.Conv2d(1, 64, (5, numCell))

        self.maxpool_filter2 = nn.MaxPool1d(maxLen-1)
        self.maxpool_filter3 = nn.MaxPool1d(maxLen-2)
        self.maxpool_filter4 = nn.MaxPool1d(maxLen-3)
        self.maxpool_filter5 = nn.MaxPool1d(maxLen-4)
        # self.out2 = nn.Linear(64*4, 64*4)
        self.out = nn.Linear(64*4, numClass)
        

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))

        out2 = self.conv2d_filter2(x)
        out3 = self.conv2d_filter3(x)
        out4 = self.conv2d_filter4(x)
        out5 = self.conv2d_filter5(x)

        out2 = out2.view(out2.size(0), out2.size(1), -1)
        out3 = out3.view(out3.size(0), out3.size(1), -1)
        out4 = out4.view(out4.size(0), out4.size(1), -1)
        out5 = out5.view(out5.size(0), out5.size(1), -1)

        out2 = self.maxpool_filter2(out2)
        out3 = self.maxpool_filter3(out3)
        out4 = self.maxpool_filter4(out4) 
        out5 = self.maxpool_filter5(out5)

        out2 = out2.view(x.size()[0], -1)
        out3 = out3.view(x.size()[0], -1)
        out4 = out4.view(x.size()[0], -1)
        out5 = out5.view(x.size()[0], -1)

        out = torch.cat((out2, out3, out4, out5), dim=1)
        out = F.relu(out)
        out = self.out(out)       
        return F.log_softmax(out, dim=1)