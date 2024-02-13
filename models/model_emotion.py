#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time: 2022/9/1 8:32 下午
# @Author: hanluyt

import torch
import torch.nn as nn
from .resnet_backbone import resnet50, load_state_dict
import torchvision
import torch.nn.functional as F

class Encoder(nn.Module):
    """Emotion encoder and non-emotion encoder"""
    def __init__(self):
        super(Encoder, self).__init__()
        model = resnet50()
        load_state_dict(model, "backbone_weight/resnet50_ft_weight.pkl")  # pre-trained model for face identification
        self.base = nn.Sequential(*list(model.children())[:-1])
        self.linear1 = nn.Linear(2048, 256)
        self.linear2 = nn.Linear(256, 256)
        # self.norm1 = nn.BatchNorm1d(256)
        # self.norm2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.base(x)  # (bs, 2048, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.linear1(x))) # (bs, 256)
        out = self.linear2(x) # (bs, 256)
        return out


class Encoder_18(nn.Module):
    """Perceptual encoder"""
    def __init__(self):
        super(Encoder_18, self).__init__()
        # ResNet18 = torchvision.models.resnet18(pretrained=False)
        ResNet18 = torchvision.models.resnet18(weights=None)

        checkpoint = torch.load('backbone_weight/resnet18_msceleb.pth')
        ResNet18.load_state_dict(checkpoint['state_dict'], strict=True)

        self.base = nn.Sequential(*list(ResNet18.children())[:-1])  # (bs, 512, 1, 1)

        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 256)
        # self.norm1 = nn.BatchNorm1d(256)
        # self.norm2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.base(x)  # (bs, 512, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.linear1(x)))  # (bs, 256)
        out = self.linear2(x)  # (bs, 256)
        return out

class Pred_Class(nn.Module):
    def __init__(self, n_class=7, atten=False):
        # receive emotion embedding: [batch, in_channel]
        super(Pred_Class, self).__init__()
        self.n_class = n_class
        self.atten = atten
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, n_class)
        self.relu = nn.ReLU(True)
        if self.atten:
            self.alpha = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())


    def forward(self, x):
        if self.atten:
            attention_weight = self.alpha(x)
            out = self.linear2((self.relu(self.linear1(x)))) * attention_weight
            return attention_weight, out
        else:
            out = self.linear2((self.relu(self.linear1(x))))
        return out


class ConCept(nn.Module):
    def __init__(self,  n_class=7, baseline=False, atten=False):
        super(ConCept, self).__init__()
        self.emotion_encoder = Encoder()
        self.baseline=baseline
        self.atten = atten

        if self.baseline:
            self.pred_class = Pred_Class(n_class, atten=False)
        else:
            self.identity_encoder = Encoder()
            for k, v in self.identity_encoder.named_parameters():
                if 'base' in k:
                    v.requires_grad = False

            self.pred_class = Pred_Class(n_class, atten=self.atten)


    def forward(self, x):
        emotion_feature = self.emotion_encoder(x)
        if self.baseline:
            logit = self.pred_class(emotion_feature)
            return emotion_feature, logit
        else:
            identity_feature = self.identity_encoder(x)
            emotion_feature = F.normalize(emotion_feature,dim=1)
            identity_feature = F.normalize(identity_feature, dim=1)

            if self.atten:
                attention_weight, logit = self.pred_class(emotion_feature)  # [bs, 1]
                return emotion_feature, identity_feature, attention_weight, logit
            else:
                logit = self.pred_class(emotion_feature)
                return emotion_feature, identity_feature, logit


class PerCept(nn.Module):
    def __init__(self,  n_class=7, baseline=False):
        super(PerCept, self).__init__()
        self.percept_encoder = Encoder_18()
        self.pred_class = Pred_Class(n_class)
        self.baseline = baseline

    def forward(self, x):
        percept_feature = self.percept_encoder(x)
        percept_feature = F.normalize(percept_feature, dim=1)
        logit = self.pred_class(percept_feature)

        if self.baseline:
            return logit
        else:
            return percept_feature, logit


"""Append an identity recognition classifier after the nonemotion 
   features and fine-tune the conceptual pathway using the IMAGEN training data."""
class IdentityRecog(nn.Module):
    def __init__(self, pth_file, n_identity=6, fix_emotion=True):
        super(IdentityRecog, self).__init__()
        self.concept = ConCept(n_class=7, atten=True)
        checkpoint = torch.load(pth_file, map_location='cpu')
        self.concept.load_state_dict(checkpoint['model'])
        self.pred_class = Pred_Class(n_class=n_identity, atten=False)
        self.fix_emotion = fix_emotion

        if self.fix_emotion:
            for k, v in self.concept.named_parameters():
                if 'emotion_encoder' or 'pred_class' in k:
                    v.requires_grad = False
                if 'identity_encoder' in k:
                    v.requires_grad = True
        else:
            for k, v in self.concept.named_parameters():
                if 'pred_class' in k:
                    v.requires_grad = False
                else:
                    v.requires_grad=True
            self.emotion_pred = Pred_Class(n_class=3, atten=False)

    def forward(self, x):
        emotion_feature, identity_feature, attention_weight, emo_logit = self.concept(x)
        if self.fix_emotion:
            ide_logit = self.pred_class(identity_feature)
            return emotion_feature, identity_feature, ide_logit
        else:
            emo_logit = self.emotion_pred(emotion_feature)
            ide_logit = self.pred_class(identity_feature)
            return emotion_feature, identity_feature, emo_logit, ide_logit









