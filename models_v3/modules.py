import torch
import torch.nn as nn
import torch.nn.functional as NF

class double_conv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(double_conv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1)
			nn.BatchNorm2d(out_ch)
			nn.ReLU(inplace=True)
			nn.Conv2d(in_ch, out_ch, 3, padding=1)
			nn.BatchNorm2d(out_ch)
			nn.ReLU(inplace=True)
		)
	def forward(self, x):
		return self.conv(x)

class down(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(down, self).__init__()
		self.down = nn.Sequential(
			nn.MaxPool2d(2)
			double_conv(in_ch, out_ch)
		)
	def forward(self, x):
		return self.down(x)

class up(nn.Module):
	def __init__(self, in_ch, out_ch, bilinear=True):
		super(up, self).__init__()
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		else:
			self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
		self.conv = double_conv(in_ch, out_ch)
	def forward(self, x1, x2):
		x1 = self.up(x1)
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

class inconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(inconv, self).__init__()
		self.conv = double_conv(in_ch, out_ch)
	def forward(self, x):
		return self.conv(x)

class outconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(inconv, self).__init__()
		self.conv = nn.Conv(in_ch, out_ch, 1)
	def forward(self, x):
		return self.conv(x)

