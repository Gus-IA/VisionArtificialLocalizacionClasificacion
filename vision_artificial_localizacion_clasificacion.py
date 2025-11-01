import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import random
import albumentations as A

device = "cpu"

train = torchvision.datasets.VOCDetection('./data', download=True)
print(len(train))


classes = ["background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor"]


def get_sample(ix):
  img, label = train[ix]
  img_np = np.array(img)
  anns = label['annotation']['object']
  if type(anns) is not list:
    anns = [anns]
  labels = np.array([classes.index(ann['name']) for ann in anns])
  bbs = [ann['bndbox'] for ann in anns]
  bbs = np.array([[int(bb['xmin']), int(bb['ymin']),int(bb['xmax'])-int(bb['xmin']),int(bb['ymax'])-int(bb['ymin'])] for bb in bbs])
  anns = (labels, bbs)
  return img_np, anns

def plot_anns(img, anns, ax=None, bg=-1):
  # anns is a tuple with (labels, bbs)
  # bbs is an array of bounding boxes in format [x_min, y_min, width, height] 
  # labels is an array containing the label 
  if not ax:
    fig, ax = plt.subplots(figsize=(10, 6))
  ax.imshow(img)
  labels, bbs = anns
  for lab, bb in zip(labels, bbs):
    if bg == -1 or lab != bg:
      x, y, w, h = bb
      rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
      text = ax.text(x, y - 10, classes[lab], {'color': 'red'})
      text.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
      ax.add_patch(rect)


def norm(bb, shape):
  # normalize bb
  # shape = (heigh, width)
  # bb = [x_min, y_min, width, height]
  h, w = shape
  return np.array([bb[0]/w, bb[1]/h, bb[2]/w, bb[3]/h])

def unnorm(bb, shape):
  # normalize bb
  # shape = (heigh, width)
  # bb = [x_min, y_min, width, height]
  h, w = shape
  return np.array([bb[0]*w, bb[1]*h, bb[2]*w, bb[3]*h])


img_np, anns = get_sample(4445)
plot_anns(img_np, anns)
plt.show()


def block(c_in, c_out, k=3, p=1, s=1, pk=2, ps=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(c_in, c_out, k, padding=p, stride=s),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(pk, stride=ps)
    )

def block2(c_in, c_out):
    return torch.nn.Sequential(
        torch.nn.Linear(c_in, c_out),
        torch.nn.ReLU()
    )

class Model(torch.nn.Module):
  def __init__(self, n_classes, n_channels=3):
    super().__init__()
    self.conv1 = block(n_channels, 8)
    self.conv2 = block(8, 16)
    self.conv3 = block(16, 32)
    self.conv4 = block(32, 64)
    self.fc1 = block2(64*6*6, 100)
    self.fc2_loc = torch.nn.Linear(100, 4)
    self.fc2_cls = torch.nn.Linear(100, n_classes)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = x.view(x.shape[0], -1)
    x = self.fc1(x)
    x_loc = self.fc2_loc(x)
    x_cls = self.fc2_cls(x)
    return x_loc, x_cls
  
model = Model(n_classes = len(classes))
output_loc, output_cls = model(torch.randn(64, 3, 100, 100))
print(output_loc.shape, output_cls.shape)



# with coco format the bb is expected in 
# [x_min, y_min, width, height] 
def get_aug(aug, min_area=0., min_visibility=0.):
    return A.Compose(aug, bbox_params={'format': 'coco', 'min_area': min_area, 'min_visibility': min_visibility, 'label_fields': ['labels']})

trans = get_aug([A.Resize(100, 100)])

labels, bbs = anns
augmented = trans(**{'image': img_np, 'bboxes': bbs, 'labels': labels})
img, bbs, labels = augmented['image'], augmented['bboxes'], augmented['labels']

print(img.shape, bbs, labels)


bb, label = bbs[0], labels[0]
print(bb, label)

bb_norm = norm(bb, img.shape[:2])
print(bb_norm)


plot_anns(img, (labels, bbs))
plt.show()


def fit(model, X, y1, y2, epochs=1, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_loc = torch.nn.L1Loss()
    criterion_cls = torch.nn.CrossEntropyLoss()
    for epoch in range(1, epochs+1):
        model.train()
        train_loss_loc, train_loss_cls = [], []
        X, y1, y2 = X.to(device), y1.to(device), y2.to(device)
        optimizer.zero_grad()
        y_hat1, y_hat2 = model(X)
        loss_loc = criterion_loc(y_hat1, y1)
        loss_cls = criterion_cls(y_hat2, y2)
        loss = loss_loc + loss_cls
        loss.backward()
        optimizer.step()
        train_loss_loc.append(loss_loc.item())
        train_loss_cls.append(loss_cls.item())
        print(f"Epoch {epoch}/{epochs} loss_loc {np.mean(train_loss_loc):.5f} loss_cls {np.mean(train_loss_cls):.5f}")




model = Model(n_classes = len(classes))
img_tensor = torch.FloatTensor(img / 255.).permute(2,0,1).unsqueeze(0)
bb_tensor = torch.FloatTensor(bb_norm).unsqueeze(0)
label_tensor = torch.tensor(label).long().unsqueeze(0)
fit(model, img_tensor, bb_tensor, label_tensor, epochs=30)



model.eval()
pred_bb_norm, pred_cls = model(img_tensor.to(device))
pred_bb = unnorm(pred_bb_norm[0].detach().numpy(), img.shape[:2])
pred_cls = torch.argmax(pred_cls, axis=1)[0]
plot_anns(img, ([pred_cls], [pred_bb]))
plt.show()


