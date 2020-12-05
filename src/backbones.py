import torchvision.models as models

resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()
squeezenet = models.squeezenet1_0()
densenet = models.densenet161()
inception = models.inception_v3(init_weights=True)
googlenet = models.googlenet(init_weights=True)
shufflenet = models.shufflenet_v2_x1_0()
mobilenet = models.mobilenet_v2()
resnext50_32x4d = models.resnext50_32x4d()
wide_resnet50_2 = models.wide_resnet50_2()
mnasnet = models.mnasnet1_0()

''' Download pretrained weights files
resnet18_p = models.resnet18(pretrained=True)
alexnet_p = models.alexnet(pretrained=True)
vgg16_p = models.vgg16(pretrained=True)
squeezenet_p = models.squeezenet1_0(pretrained=True)
densenet_p = models.densenet161(pretrained=True)
inception_p = models.inception_v3(pretrained=True)
googlenet_p = models.googlenet(pretrained=True)
shufflenet_p = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet_p = models.mobilenet_v2(pretrained=True)
resnext50_32x4d_p = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2_p = models.wide_resnet50_2(pretrained=True)
mnasnet_p = models.mnasnet1_0(pretrained=True)
'''
