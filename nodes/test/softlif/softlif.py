"""
soft LIf 방법을 이용한 추론 코드

Network structure를 알고 있어야 한다는 말은 fixed network structure를 사용한다는 말 아닌가?
왜냐하면, .pt에저장되는 정보는 학습된 가중치에 대한 정보가 된다. 가중치가 없는 레이어는
.pt에 저장되지 않기 때문에 학습에 어떤 network를 사용했는지 정확하게 알 수는 없다.
"""

""" """
from n3ml.model import Hunsberger2015_ANN
from n3ml.model import Hunsberger2015_SNN

images = None
labels = None

pretrained = 'pretrained/huns2015_ann.pth'
num_steps = 20

"""
Dynamic network는 처음에 레이어를 가지고 있지 않는 상태이다.
학습된 모델 정보를 불러오는 과정에서 레이어도 추가를 하고,
학습된 정보도 저장을 하는 방식으로 지원이 되어야 한다.
"""
ann = torch.load(pretrained)['model']
snn = Hunsberger2015_SNN(ann)

if torch.cuda.is_available():
	images = images.cuda()
	labels = labels.cuda()
	snn.cuda()

criterion = None

with torch.no_grad():
	outs = snn(images, num_steps)

	""" outs 형태에 따른 outs 처리"""
	
	loss = criterion(outs, labels)
	num_corrects = torch.argmax(outs, dim=1).eq(labels).sum(dim=0)

