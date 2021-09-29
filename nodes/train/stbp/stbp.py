"""

[1] 데모에서는 cuda를 사용하는지? 아니면, cpu-only로 하는지?
[2] 전달되는 데이터에 대한 분석이 필요하다.
[3] 입력된 데이터는 학습을 위해 입력 데이터와 목표 데이터로 구성이 되어야 한다.
[4] 

"""

import argparse

import torch.nn as nn
import torch.optim as optim

from n3ml.model import Wu2018


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--latency', default=20, type=int)
parser.add_argument('--lr', default=0.001, type=float)

opt = parser.parse_args()

# TODO: [1] 데모에서는 cuda를 사용하는지? 아니면, cpu-only로 하는지?
model = Wu2018(batch_size=opt.batch_size, time_interval=opt.latency)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)

while True:
    data = input()

    # TODO: [2] Analyse the data and its format to feed it into a model.

    # TODO: [3] 입력된 데이터는 학습을 위해 입력 데이터와 목표 데이터로 구성이 되어야 한다.
    images, labels = data

    outputs = model(images)

