#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U-Net
"""

from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import EpiDatasets
import U_Net

BATCH_SIZE = 30
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 0.001
EPOCH = 10

# 前処理
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
target_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224), interpolation=0), torchvision.transforms.ToTensor()])

# 訓練データとテストデータの用意
trainset = EpiDatasets.EpiDatasets(train=True, transform=transform, target_transform=target_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = EpiDatasets.EpiDatasets(train=False, transform=transform, target_transform=target_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

# Networkの準備と損失関数と最適化手法
device = torch.device("cuda:0")
u_net = U_Net.U_Net(3,2)
u_net = u_net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(u_net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

# 訓練とテスト
train_loss_value=[]      #trainingのlossを保持するlist
train_acc_value=[]       #trainingのaccuracyを保持するlist
train_class_acc_value=[]
test_loss_value=[]       #testのlossを保持するlist
test_acc_value=[]        #testのaccuracyを保持するlist
test_class_acc_value=[] 

for epoch in range(EPOCH):
    print('epoch', epoch+1)    #epoch数の出力

    sum_loss = 0.0          #lossの合計
    sum_correct = 0         #正解率の合計
    sum_total = 0           #dataの数の合計
    sum_class_pixels = np.zeros(2)
    sum_class_correct = np.zeros(2)

    # 訓練
    train_num = 1
    u_net.train()
    for (inputs, labels) in trainloader:
        print('train', train_num)
        train_num = train_num + 1

        print('inputs', inputs)
        print('labels', labels)

        inputs = inputs.to(device)
        
        optimizer.zero_grad()
        outputs = u_net(inputs)
        row_diff = labels.size()[2] - outputs.size()[2]
        col_diff = labels.size()[3] - outputs.size()[3]
        row_slice = row_diff // 2
        col_slice = col_diff // 2
        labels = labels[:, :, row_slice:(labels.size()[2]-row_slice), col_slice:(labels.size()[3]-col_slice)]

        labels = torch.reshape(labels, (labels.shape[0], labels.shape[2], labels.shape[3]))
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        loss = criterion(outputs, labels)

        sum_loss += loss.item()                                      #lossを足していく
        _, predicted = torch.max(outputs, 1)                         #出力の最大値の添字(予想位置)を取得
        sum_total += labels.shape[0]*labels.shape[1]*labels.shape[2] #labelの数を足していくことでデータの総和を取る
        sum_correct += (predicted == labels).sum().item()            #予想位置と実際の正解を比べ,正解している数だけ足す

        for num in range(2):
            sum_class_pixels[num] += (labels == num).sum().item()
            sum_class_correct[num] += ((predicted == labels) & (labels == num)).sum().item()

        loss.backward()
        optimizer.step()       

    #print("train mean loss={}, accuracy={}".format(sum_loss*BATCH_SIZE/len(trainloader.dataset), float(sum_correct/sum_total))) 　#lossとaccuracy出力
    train_loss_value.append(sum_loss*BATCH_SIZE/len(trainloader.dataset))  #traindataのlossをグラフ描画のためにlistに保持
    train_acc_value.append(float(sum_correct/sum_total))   #traindataのaccuracyをグラフ描画のためにlistに保持
    train_class_acc_value.append((sum_class_correct/sum_class_pixels).sum()/2)
    print('train mean loss=', train_loss_value[epoch])
    print('train mean accuracy=', train_acc_value[epoch])
    print('train class mean accuracy=', train_class_acc_value[epoch])

    sum_loss = 0.0
    sum_correct = 0
    sum_total = 0
    sum_class_pixels = np.zeros(2)
    sum_class_correct = np.zeros(2)

    # テスト
    u_net.eval()
    with torch.no_grad():
        test_num = 1
        for (inputs, labels) in testloader:
            print('test_num', test_num)
            test_num = test_num + 1

            inputs = inputs.to(device)

            outputs = u_net(inputs)

            row_diff = labels.size()[2] - outputs.size()[2]
            col_diff = labels.size()[3] - outputs.size()[3]
            row_slice = row_diff // 2
            col_slice = col_diff // 2
            labels = labels[:, :, row_slice:(labels.size()[2]-row_slice), col_slice:(labels.size()[3]-col_slice)]

            labels = torch.reshape(labels, (labels.shape[0], labels.shape[2], labels.shape[3]))
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            loss = criterion(outputs, labels)

            sum_loss += loss.item()                            #lossを足していく
            _, predicted = outputs.max(1)
            sum_total += labels.shape[0]*labels.shape[1]*labels.shape[2]
            sum_correct += (predicted == labels).sum().item()

            for num in range(2):
                sum_class_pixels[num] += (labels == num).sum().item()
                sum_class_correct[num] += ((predicted == labels) & (labels == num)).sum().item()

    #print("test  mean loss={}, accuracy={}".format(sum_loss*BATCH_SIZE/len(testloader.dataset), float(sum_correct/sum_total)))
    test_loss_value.append(sum_loss*BATCH_SIZE/len(testloader.dataset))
    test_acc_value.append(float(sum_correct/sum_total))
    test_class_acc_value.append((sum_class_correct/sum_class_pixels).sum()/2)
    print('test mean loss=', test_loss_value[epoch])
    print('test mean accuracy=', test_acc_value[epoch])
    print('test class mean accuracy=', test_class_acc_value[epoch])


image_num = 0

u_net.eval()
with torch.no_grad():
    for (inputs, labels) in testloader:
        inputs = inputs.to(device)

        outputs = u_net(inputs)
        row_diff = labels.size()[2] - outputs.size()[2]
        col_diff = labels.size()[3] - outputs.size()[3]
        row_slice = row_diff // 2
        col_slice = col_diff // 2
        labels = labels[:, :, row_slice:(labels.size()[2]-row_slice), col_slice:(labels.size()[3]-col_slice)]

        labels = torch.reshape(labels, (labels.shape[0], labels.shape[2], labels.shape[3]))
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        labels = labels.to(device)

        _, predicted = torch.max(outputs, 1)                         #出力の最大値の添字(予想位置)を取得

        if image_num < 100:
            for num in range(BATCH_SIZE):
                input_image = inputs[num,:,:,:]
                input_image = input_image.to(torch.device('cpu'))
                input_image = (input_image + 1) * 0.5
                input_r = input_image[0,:,:]
                input_g = input_image[1,:,:]
                input_b = input_image[2,:,:]
                input_image = np.zeros([224,224,3])
                input_image[:,:,0] = input_r
                input_image[:,:,1] = input_g
                input_image[:,:,2] = input_b
                plt.imshow(input_image)
                plt.savefig("input/"+str(image_num)+".png")

                label_image = labels[num,:,:]
                label_image = label_image.to(torch.device('cpu'))
                label_image = label_image.detach().numpy()
                plt.imshow(label_image, cmap= 'gray')
                plt.savefig("label/"+str(image_num)+".png")

                predicted_image = predicted[num,:,:]
                predicted_image = predicted_image.to(torch.device('cpu'))
                predicted_image = predicted_image.detach().numpy()
                plt.imshow(predicted_image, cmap = 'gray')
                plt.savefig("predicted/"+str(image_num)+".png")

                image_num = image_num + 1


    
plt.figure(figsize=(6,6))      #グラフ描画用
#以下グラフ描画
plt.plot(range(EPOCH), train_loss_value, color='red', linewidth=1.5)
plt.plot(range(EPOCH), test_loss_value, color='blue', linewidth=1.5)
plt.xlim(0, EPOCH)
plt.ylim(0, 0.5)
plt.xlabel('EPOCH', fontsize='20')
plt.ylabel('LOSS', fontsize='20')
plt.legend(['train loss', 'test loss'])
#plt.title('loss', fontsize='20')
plt.savefig("loss_image.png")
plt.clf()

plt.plot(range(EPOCH), train_acc_value, color='red', linewidth=1.5)
plt.plot(range(EPOCH), test_acc_value, color='blue', linewidth=1.5)
plt.xlim(0, EPOCH)
plt.ylim(0, 1)
plt.xlabel('EPOCH', fontsize='20')
plt.ylabel('ACCURACY', fontsize='20')
plt.legend(['train acc', 'test acc'])
#plt.title('accuracy', fontsize='20')
plt.savefig("accuracy_image.png")
plt.clf()

plt.plot(range(EPOCH), train_class_acc_value)
plt.plot(range(EPOCH), test_class_acc_value, c='#00ff00')
plt.xlim(0, EPOCH)
plt.ylim(0, 1)
plt.xlabel('EPOCH')
plt.ylabel('CLASS ACCURACY')
plt.legend(['train acc', 'test acc'])
plt.title('class accuracy')
plt.savefig("class_accuracy_image.png")
