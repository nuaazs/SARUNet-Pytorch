import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from torch.autograd import Function
import torchvision.models as models
class CtLoss(nn.Module):
    def __init__(self):
        super(CtLoss,self).__init__()
        self.num_classes = 25
        return

    @staticmethod
    def forward(real,fake):
        ct_array_real = real.cpu().detach().numpy()[0,0,:,:]
        ct_array_fake = fake.cpu().detach().numpy()[0,0,:,:]
        hu_list = [-999999,-950,-120,-88,-53,-23,7,18,80,120,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,99999999]
        # for i in range(ct_array_real.shape[0]):
        #     for j in range(ct_array_real.shape[1]):
        #         if ct_array_real[i,j]<-950:
        #             ct_array_real[i,j]=1
        #         elif ct_array_real[i,j]<-120:
        #             ct_array_real[i,j]=2
        #         elif ct_array_real[i,j]<-88:
        #             ct_array_real[i,j]=3
        #         elif ct_array_real[i,j]<-53:
        #             ct_array_real[i,j]=4
        #         elif ct_array_real[i,j]<-23:
        #             ct_array_real[i,j]=5
        #         elif ct_array_real[i,j]<7:
        #             ct_array_real[i,j]=6
        #         elif ct_array_real[i,j]<18:
        #             ct_array_real[i,j]=7
        #         elif ct_array_real[i,j]<80:
        #             ct_array_real[i,j]=8
        #         elif ct_array_real[i,j]<120:
        #             ct_array_real[i,j]=9
        #         elif ct_array_real[i,j]<200:
        #             ct_array_real[i,j]=10
        #         elif ct_array_real[i,j]<300:
        #             ct_array_real[i,j]=11
        #         elif ct_array_real[i,j]<400:
        #             ct_array_real[i,j]=12
        #         elif ct_array_real[i,j]<500:
        #             ct_array_real[i,j]=13
        #         elif ct_array_real[i,j]<600:
        #             ct_array_real[i,j]=14
        #         elif ct_array_real[i,j]<700:
        #             ct_array_real[i,j]=15
        #         elif ct_array_real[i,j]<800:
        #             ct_array_real[i,j]=16
        #         elif ct_array_real[i,j]<900:
        #             ct_array_real[i,j]=17
        #         elif ct_array_real[i,j]<1000:
        #             ct_array_real[i,j]=18
        #         elif ct_array_real[i,j]<1100:
        #             ct_array_real[i,j]=19
        #         elif ct_array_real[i,j]<1200:
        #             ct_array_real[i,j]=20
        #         elif ct_array_real[i,j]<1300:
        #             ct_array_real[i,j]=21
        #         elif ct_array_real[i,j]<1400:
        #             ct_array_real[i,j]=22
        #         elif ct_array_real[i,j]<1500:
        #             ct_array_real[i,j]=23
        #         elif ct_array_real[i,j]<1600:
        #             ct_array_real[i,j]=24
        #         else:
        #             ct_array_real[i,j]=25
                
        #         if ct_array_fake[i,j]<-950:
        #             ct_array_fake[i,j]=1
        #         elif ct_array_fake[i,j]<-120:
        #             ct_array_fake[i,j]=2
        #         elif ct_array_fake[i,j]<-88:
        #             ct_array_fake[i,j]=3
        #         elif ct_array_fake[i,j]<-53:
        #             ct_array_fake[i,j]=4
        #         elif ct_array_fake[i,j]<-23:
        #             ct_array_fake[i,j]=5
        #         elif ct_array_fake[i,j]<7:
        #             ct_array_fake[i,j]=6
        #         elif ct_array_fake[i,j]<18:
        #             ct_array_fake[i,j]=7
        #         elif ct_array_fake[i,j]<80:
        #             ct_array_fake[i,j]=8
        #         elif ct_array_fake[i,j]<120:
        #             ct_array_fake[i,j]=9
        #         elif ct_array_fake[i,j]<200:
        #             ct_array_fake[i,j]=10
        #         elif ct_array_fake[i,j]<300:
        #             ct_array_fake[i,j]=11
        #         elif ct_array_fake[i,j]<400:
        #             ct_array_fake[i,j]=12
        #         elif ct_array_fake[i,j]<500:
        #             ct_array_fake[i,j]=13
        #         elif ct_array_fake[i,j]<600:
        #             ct_array_fake[i,j]=14
        #         elif ct_array_fake[i,j]<700:
        #             ct_array_fake[i,j]=15
        #         elif ct_array_fake[i,j]<800:
        #             ct_array_fake[i,j]=16
        #         elif ct_array_fake[i,j]<900:
        #             ct_array_fake[i,j]=17
        #         elif ct_array_fake[i,j]<1000:
        #             ct_array_fake[i,j]=18
        #         elif ct_array_fake[i,j]<1100:
        #             ct_array_fake[i,j]=19
        #         elif ct_array_fake[i,j]<1200:
        #             ct_array_fake[i,j]=20
        #         elif ct_array_fake[i,j]<1300:
        #             ct_array_fake[i,j]=21
        #         elif ct_array_fake[i,j]<1400:
        #             ct_array_fake[i,j]=22
        #         elif ct_array_fake[i,j]<1500:
        #             ct_array_fake[i,j]=23
        #         elif ct_array_fake[i,j]<1600:
        #             ct_array_fake[i,j]=24
        #         else:
        #             ct_array_fake[i,j]=25

        
        result = nn.L1Loss()(torch.Tensor(ct_array_real),torch.Tensor(ct_array_fake))
        # result = 0
        # for i in range(self.num_classes):
        #     real_slice = real[i]
        #     fake_slice = fake[i]
        #     result += nn.BCELoss()(torch.Tensor(real_slice),torch.Tensor(fake_slice))
        # result = result/self.num_classes
        return result
    @staticmethod
    def backward(grad_output):
        return grad_output
    
# Loss functions
class PerceptualLoss():
	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = models.vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
            
			if i == conv_3_3_layer:
				break
		return model
		
	def __init__(self, loss):
		self.criterion = loss
		self.contentFunc = self.contentFunc()
			
	def get_loss(self, fakeIm, realIm):
		f_fake = self.contentFunc.forward(fakeIm)
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss
