import argparse
from dataprocess.dataset import ImageForgery
from models import ShuffleNet2, ShuffleNet24, ShuffleNet25
from models import MobileNet2, ShuffleNet, SqueezeNet, ResNet
from models import MobileNetV3_Large, MobileNetV3_Small, BetterShuffleNet, SENet18
import torch as t
from torch.utils import data
import torch.nn as nn
import copy
from PIL import Image
from torchvision import transforms
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:51200"

t.cuda.empty_cache() 

ap = argparse.ArgumentParser()
ap.add_argument("-gpu", "--use_gpu", type=int, default=1,
                help="use gpu or not")
ap.add_argument("-bs", "--batchsize", type=int, default=128,
                help="the batch size of input")
ap.add_argument("-t", "--train", type=int, default=1,
                help="choose training or valdating")
ap.add_argument("-pre", "--pretrained", type=str, default="None",
                help="select a pretrained model")
ap.add_argument("-e", "--epochs", type=int, default=50,
                help="epochs of training")
ap.add_argument("-path", "--datapath", type=str, default="./Dataset",
                help="path of training dataset")
ap.add_argument("-m", "--model", type=str, default="ShuffleNet2",
                help="the type of model")
ap.add_argument("-c", "--classes", type=int, default=2,
                help="the number of classes of dataset")
ap.add_argument("-s", "--inputsize", type=int, default=224,
                help="the size of image")
ap.add_argument("-nt", "--nettype", type=int, default=1,
                help="type of network")

def train_model(model, dataloaders, loss_fn, optimizer, num_epochs=5):
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.
	val_acc_history = []
	
	start_time = t.cuda.Event(enable_timing=True)
	end_time = t.cuda.Event(enable_timing=True)
	
	start_memory = t.cuda.memory_allocated(device)
	
	start_time.record()

	print('\n ', args["model"])
	for epoch in range(num_epochs):
		print('\n Epoch', epoch+1)
		for phase in ["train", "val"]:
			running_loss = 0.
			running_corrects = 0.
			if phase == "train":
				model.train()
			else:
				model.eval()
				
			for inputs, labels in dataloaders[phase]:
				inputs, labels = inputs.to(device), labels.to(device)
                
				with t.autograd.set_grad_enabled(phase=="train"):
					outputs = model(inputs) 
					loss = loss_fn(outputs, labels) 
				
				preds = outputs.argmax(dim=1)
                
				if phase == "train":
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
                
				running_loss += loss.item() * inputs.size(0)
				running_corrects += t.sum(preds.view(-1) == labels.view(-1)).item()
                
			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc = running_corrects / len(dataloaders[phase].dataset)
			
			print("Phase {} loss: {}, acc: {}".format(phase, epoch_loss, epoch_acc))
            
			if phase == "val" and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())
            
			if phase == "val":
				val_acc_history.append(epoch_acc)
    
	model.load_state_dict(best_model_wts)  
	# Stop timer
	end_time.record()
	
	t.cuda.synchronize()
	
	elapsed_time = start_time.elapsed_time(end_time)
	
	end_memory = t.cuda.memory_allocated(device)
	memory_usage = end_memory - start_memory

	return model, val_acc_history, elapsed_time / 1000.0, memory_usage / 1024.0 / 1024.0


def test_model(model, dataloader, loss_fn):
	import time
	model.eval()
	running_loss = 0.
	running_corrects = 0.
	records = []
	total_len = 0
	
	for inputs, labels in dataloader:
		img_len = len(inputs)
		inputs, labels = inputs.to(device), labels.to(device)
		# 
		start = time.time()
		outputs = model(inputs)
		# 
		end = time.time()
		fps = img_len/1
		records.append(fps)
		loss = loss_fn(outputs, labels) 
		preds = outputs.argmax(dim=1)

		running_loss += loss.item() * inputs.size(0)
		running_corrects += t.sum(preds.view(-1) == labels.view(-1)).item()
	
	epoch_loss = running_loss / len(dataloader.dataset)
	epoch_acc = running_corrects / len(dataloader.dataset)

	print("On val dataset loss: {}, acc: {}".format(epoch_loss, epoch_acc))
	import numpy as np
	print("{} FPS".format(np.mean(records)))


if __name__ == '__main__':
	args = vars(ap.parse_args())
	path = args["datapath"]
	train_sign = args["train"]
	epochs = args["epochs"]

	batchsize = args["batchsize"]
	dataloader = {}
	if train_sign:
		train_dataset = ImageForgery(path, train=True)
		
		train_loader = data.DataLoader(train_dataset,
	                               batch_size = batchsize,
	                               shuffle=True)
		dataloader["train"] = train_loader
	
	val_dataset = ImageForgery(path, train=False, test=False)                               
	val_loader = data.DataLoader(val_dataset,
                             batch_size = batchsize,
                             shuffle=True)
	dataloader["val"] = val_loader

	model_path = args["pretrained"]
	num_classes = args["classes"]
	input_size = args["inputsize"]
	net_type = args["nettype"]
	model_type = args["model"]
	
	if model_type == "ShuffleNet2":
		model = ShuffleNet2(num_classes, input_size, net_type)
	elif model_type == "MobileNet2":
		model = MobileNet2(num_classes, input_size, net_type)
	elif model_type == "MobileNetV3_Large":
		model = MobileNetV3_Large(num_classes)
	elif model_type == "MobileNetV3_Small":
		model = MobileNetV3_Small(num_classes)
	elif "efficientnet" in model_type.lower():
		model = EfficientNet.from_name(model_type)
	elif model_type == "BetterShuffleNet":
		model = BetterShuffleNet(num_classes)
	elif model_type == "SENet18":
		model = ShuffleNet(num_classes)
	elif model_type == "ShuffleNet":
		model = SENet18(num_classes)
	elif model_type == "SqueezeNet":
		model = SqueezeNet.squeezenet(num_classes)
	elif model_type == "ResNet":
		model = ResNet.ResNet50(num_classes)
	elif model_type == "ShuffleNet24":
		model = ShuffleNet24([24, 116, 232, 464, 1024], [4, 8, 4], num_classes)
	elif model_type == "ShuffleNet25":
		model = ShuffleNet25([24, 116, 232, 464, 1024], [4, 8, 4], num_classes)
	else:
		print("We don't implement the model, please choose ShuffleNet2 or MobileNet2")
	if model_path != "None":
		model.load_state_dict(t.load(model_path))

	device = t.device("cuda" if t.cuda.is_available() else "cpu")
	# device = t.device("cpu")
	model.to(device)

	loss_fn = nn.CrossEntropyLoss()
	optimizer = t.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
	
	total_params = sum(p.numel() for p in model.parameters())
	print(f"Total number of parameters: {total_params}")

	if train_sign:
		model, val_logs, execution_time, memory_usage = train_model(model, dataloader, loss_fn, optimizer, epochs)
		
		# store the model
		import time
		pkl_path = "./save/" + model_type + "-" + "batchsize" + str(batchsize) + "-epoch" + str(epochs) + "-" + path.split("/")[2] + "-" + str(int(time.time())) + '.pkl'
		t.save(model.state_dict(), pkl_path)
		print(f"Total waktu eksekusi: {execution_time} detik")
		print(f"Penggunaan memori: {memory_usage} MB")
		print("Model saved to", pkl_path)  
	else:
		test_model(model, dataloader['val'], loss_fn)
		
		model.eval()
		
		for inputs, labels in dataloader['val']:
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			probabilities = t.nn.functional.softmax(outputs[0], dim=0)
			top5_prob, top5_catid = t.topk(probabilities, 1)
		# model.eval()
		# input_image = Image.open('bare.1012.png')
		# preprocess = transforms.Compose([
		# 	transforms.Resize(256),
		# 	transforms.CenterCrop(224),
		# 	# transforms.RandomHorizontalFlip(),
		# 	transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		# ])
		# input_tensor = preprocess(input_image)
		# input_batch = input_tensor.unsqueeze(0)

		# with t.no_grad():
		# 	output = model(input_batch)
		
		# probabilities = t.nn.functional.softmax(output[0], dim=0)

		# with open("classes.txt", "r") as f:
		# 	categories = [s.strip() for s in f.readlines()]
		# # Show top categories per image
		# top5_prob, top5_catid = t.topk(probabilities, 3)
		# for i in range(top5_prob.size(0)):
		# 	print(categories[top5_catid[i]], top5_prob[i].item())
	
