import torch

from n3ml.model import VGG16, SVGG16


images = torch.rand(size=(1000, 3, 32, 32))
labels = torch.rand(size=(1000,))

batch_size = images.size(0)
num_steps = 2500
pretrained = 'pretrained/vgg16_acc_9289.pt'
saved = 'pretrained/svgg16_ths_vgg16_acc_9289.pt'

ann = VGG16()
ann.load_state_dict(torch.load(pretrained)['model'])

snn = SVGG16(ann, batch_size=batch_size)
snn.update_threshold(torch.load(saved))

if torch.cuda.is_available():
    images = images.cuda()
    labels = labels.cuda()
    snn.cuda()

snn.eval()

print(snn)

with torch.no_grad():
    outs = snn(images, num_steps=num_steps)

    num_corrects = torch.argmax(outs, dim=1).eq(labels).sum(dim=0).item()
    total_images = images.size(0)

    print("accuracy: {}".format(1.0*num_corrects/total_images))
