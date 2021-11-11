import torch
import torch.nn as nn

from n3ml.model import Wu2018


images = None
labels = None

pretrained = None
num_classes = 10
batch_size = None
num_steps = None

model = Wu2018(batch_size=batch_size)
model.load_state_dict(torch.load(pretrained)['model'])

if torch.cuda.is_available():
    images = images.cuda()
    labels = labels.cuda()
    model.cuda()

criterion = nn.MSELoss()

with torch.no_grad():
    outs = model(images, num_steps)

    labels_ = torch.zeros(torch.numel(labels), num_classes, device=labels.cuda())
    labels_ = labels_.scatter_(1, labels.view(-1, 1), 1)

    loss = criterion(outs, labels_)

    num_correct = torch.argmax(outs, dim=1).eq(labels).sum(dim=0)
