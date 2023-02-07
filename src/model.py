from torchvision.models.detection import fasterrcnn_resnet50_fpn

def get_faster_rcnn():
    return fasterrcnn_resnet50_fpn(num_classes=2)