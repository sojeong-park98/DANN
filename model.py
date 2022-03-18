import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # 모델에서 출력할 때에는 기울기를 그대로
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        # 오차 역전파 할 때에는 기울기에 -를 붙여서
        output = grad_output.neg() * ctx.alpha
        return output, None

class Feature_extractor(nn.Module):
    def __init__(self):
        super(Feature_extractor, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(48)
        self.MaxPool = nn.MaxPool2d(2)
        self.DropOut = nn.Dropout2d()
        self.relu = nn.ReLU(True)

    def forward(self, input):
        # ? X 3 X 28 X 28
        x = self.conv1(input)
        # ? X 32 X 24 X 24
        x = self.bn1(x)
        x = self.relu(x)
        x = self.MaxPool(x)
        # ? X 32 X 12 X 12
        x = self.conv2(x)
        # ? X 48 X 8 X 8
        x = self.bn2(x)
        x = self.relu(x)
        x = self.DropOut(x)
        x = self.MaxPool(x)
        # ? X 48 X 4 X 4

        feat = x.view(-1, 48 * 4 * 4)
        # ? X 768

        return feat

class Class_classifier(nn.Module):
    def __init__(self, num_classes = 10):
        super(Class_classifier, self).__init__()

        self.linear1 = nn.Linear(48 * 4 * 4, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.linear2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.linear3 = nn.Linear(100, num_classes)
        self.DropOut = nn.Dropout2d()
        self.relu = nn.ReLU(True)

    def forward(self, x):
        # ? X 768
        x = self.linear1(x)
        # ? X 100
        x = self.bn1(x)
        x = self.relu(x)
        x = self.DropOut(x)
        x = self.linear2(x)
        # ? X 100
        x = self.bn2(x)
        x = self.relu(x)
        x = self.linear3(x)
        # ? X 10 => 10는 클래스의 개수

        x = F.log_softmax(x, dim=1)

        return x

class Domain_classifier(nn.Module):
    def __init__(self):
        super(Domain_classifier, self).__init__()

        self.linear1 = nn.Linear(48 * 4 * 4, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU(True)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)
        # ? X 768
        x = self.linear1(x)
        # ? X 100
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # ? X 1 => 1는 소스or타겟

        x = F.sigmoid(x)

        return x

class DANN():
    def __init__(self, num_classes, cuda, lr, n_epoch):

        # 전체 에폭 수와, 쿠다 사용 여부 저장 (클래스 내의 함수에서 self.~로 접근하기 위함)
        self.n_epoch = n_epoch
        self.cuda = cuda

        # 모델 불러오기
        self.feature_extractor = Feature_extractor()
        self.class_classifier = Class_classifier(num_classes=num_classes)
        self.domain_classifier = Domain_classifier()

        # 쿠다 사용 시
        if self.cuda:
            self.feature_extractor = self.feature_extractor.cuda()
            self.class_classifier = self.class_classifier.cuda()
            self.domain_classifier = self.domain_classifier.cuda()

        # 옵티마이저 정의
        self.optimizer = optim.SGD(list(self.feature_extractor.parameters())+list(self.class_classifier.parameters())+list(self.domain_classifier.parameters()), lr=lr, momentum=0.9)

    def train(self, epoch, dataloader_source, dataloader_target):

        len_dataloader = len(dataloader_target)
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        iter_idx = 0
        while iter_idx < len_dataloader:
            # alpha는 훈련의 진행에 따라 0=>1로 점점 증가
            p = float(iter_idx + epoch * len_dataloader) / self.n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # 모델의 기울기 초기화
            self.feature_extractor.zero_grad()
            self.class_classifier.zero_grad()
            self.domain_classifier.zero_grad()

            ############################
            #       소스 훈련 파트       #
            ############################
            # 훈련을 위해 소스 데이터로더로부터 현재 배치에 활용할 데이터를 불러오기
            # iteration의 수가 타겟 로더 기준으로 맞춰져있기 때문에, 소스 로더가 끝날 경우 새로 불러와야 함
            try:
                data_source = data_source_iter.next()
            except:
                data_source_iter = iter(dataloader_source)
                data_source = data_source_iter.next()

            s_img, s_label = data_source

            # 만약 이미지 데이터가 1차원이라면 모델 입력을 위해 3차원으로 변경
            if s_img.shape[1] == 1:
                img_3_channel = torch.zeros((s_img.shape[0], 3, s_img.shape[2], s_img.shape[3]))
                img_3_channel[:, 0, :, :] = s_img.squeeze(1)
                img_3_channel[:, 1, :, :] = s_img.squeeze(1)
                img_3_channel[:, 2, :, :] = s_img.squeeze(1)
                s_img = img_3_channel

            if self.cuda:
                s_img, s_label = s_img.cuda(), s_label.cuda()

            # 소스 도메인은 0으로 레이블 지정
            s_domain_label = torch.zeros(s_img.shape[0]).unsqueeze(1)
            if self.cuda:
                s_domain_label = s_domain_label.cuda()

            # 모델로부터 소스도메인에 대한 계산
            source_feat = self.feature_extractor(s_img)
            s_class_pred = self.class_classifier(source_feat)
            s_domain_pred = self.domain_classifier(source_feat, alpha=alpha)

            # 손실 함수
            s_class_loss = F.nll_loss(s_class_pred, s_label)  # 클래스 예측에 대한 크로스 엔트로피 로스
            s_domain_loss = F.binary_cross_entropy(s_domain_pred, s_domain_label)  # 도메인에 대한 크로스 엔트로피 로스

            ############################
            #       타겟 훈련 파트       #
            ############################
            # training model using target data
            data_target = data_target_iter.next()
            t_img, _ = data_target
            if t_img.shape[1] == 1:
                img_3_channel = torch.zeros((t_img.shape[0], 3, t_img.shape[2], t_img.shape[3]))
                img_3_channel[:, 0, :, :] = t_img.squeeze(1)
                img_3_channel[:, 1, :, :] = t_img.squeeze(1)
                img_3_channel[:, 2, :, :] = t_img.squeeze(1)
                t_img = img_3_channel
            if self.cuda:
                t_img = t_img.cuda()

            # 타겟 도메인은 1으로 레이블 지정
            t_domain_label = torch.ones(t_img.shape[0]).unsqueeze(1)
            if self.cuda:
                t_domain_label = t_domain_label.cuda()

            # 모델로부터 타겟도메인에 대한 계산
            target_feat = self.feature_extractor(t_img)
            t_domain_pred = self.domain_classifier(target_feat, alpha=alpha)

            # 손실 함수
            t_domain_loss = F.binary_cross_entropy(t_domain_pred, t_domain_label)  # 도메인에 대한 크로스 엔트로피 로스

            # 소스 도메인의 손실함수와 타겟 도메인의 손실함수를 모두 더해주기
            loss = s_class_loss + s_domain_loss + t_domain_loss
            # 기울기 계산
            loss.backward()
            # 최적화를 위한 모델 업데이트
            self.optimizer.step()

            iter_idx += 1

            if iter_idx%20 == 0:
              print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                  % (epoch, iter_idx, len_dataloader, s_class_loss.cpu().data.numpy(),
                     s_domain_loss.cpu().data.numpy(), t_domain_loss.cpu().data.numpy()))

    def test(self, dataloader_test):

        n_total = 0
        n_correct = 0

        iter_idx = 0
        len_dataloader = len(dataloader_test)
        data_iter = iter(dataloader_test)

        while iter_idx < len_dataloader:
            data = data_iter.next()
            img, label = data
            if self.cuda:
                img, label = img.cuda(), label.cuda()
            feat = self.feature_extractor(img)
            class_pred = self.class_classifier(feat)

            # class_pred는 현재 원핫 구조 [? X class_num]
            class_out = class_pred.argmax(1)
            # class_out은 가장 큰 확률 값을 갖는 인덱스

            n_correct += class_out.eq(label).sum()
            n_total += len(img)
            iter_idx += 1

        return n_correct/n_total