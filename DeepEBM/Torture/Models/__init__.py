from . import GAN
from . import Classifier

model_dict = {
    "GAN_resnet_cifar_G": GAN.resnet_cifar.Generator,
    "GAN_resnet_cifar_D": GAN.resnet_cifar.Discriminator,
    "Classifier_resnet_cifar_18": Classifier.resnet_cifar.resnet18,
    "Classifier_resnet_cifar_34": Classifier.resnet_cifar.resnet34,
    "Classifier_resnet_cifar_50": Classifier.resnet_cifar.resnet50,
    "Classifier_resnet_cifar_101": Classifier.resnet_cifar.resnet101,
    "Classifier_resnet_cifar_152": Classifier.resnet_cifar.resnet152,
    "Classifier_resnet_imagenet_18": Classifier.resnet_imagenet.resnet18,
    "Classifier_resnet_imagenet_34": Classifier.resnet_imagenet.resnet34,
    "Classifier_resnet_imagenet_50": Classifier.resnet_imagenet.resnet50,
    "Classifier_resnet_imagenet_101": Classifier.resnet_imagenet.resnet101,
    "Classifier_resnet_imagenet_152": Classifier.resnet_imagenet.resnet152
}


def get_model(name):
    return model_dict[name]
