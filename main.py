# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch.backends.cudnn as cudnn
import torch.utils.data
from dataset.dataloader import create_dataset
from model import DANN

# 쿠다 사용이 가능한지 확인
if torch.cuda.is_available():
    #print("Use CUDA")
    cuda = True
    cudnn.benchmark = True
else:
    print("Use unavailable")
    cuda = False

# 랜덤시드 셋팅
torch.manual_seed(1234)

# 하이퍼 파라미터 셋팅
lr = 0.001
batch_size = 128
n_epoch = 100
image_size = 28  # 입력 이미지를 28 X 28로 리사이즈

# 데이터셋 정의
num_classes = 10
source_dataset = 'MNIST'
target_dataset = 'mnist_m'

# Pres the green button in the gutter to run the script.
if __name__ == '__main__':

    model = DANN(num_classes, cuda, lr, n_epoch)

    dataloader_source_train, dataloader_source_test = create_dataset(source_dataset, batch_size, image_size)
    dataloader_target_train, dataloader_target_test = create_dataset(target_dataset, batch_size, image_size)

    # epoch만큼 훈련
    best_acc = 0
    for epoch in range(n_epoch):
        # 훈련
        print("Epoch %dth training start!" % (epoch))
        model.feature_extractor.train()
        model.class_classifier.train()
        model.domain_classifier.train()
        model.train(epoch, dataloader_source_train, dataloader_target_train)
        # 테스트
        model.feature_extractor.eval()
        model.class_classifier.eval()
        model.domain_classifier.eval()

        print("Epoch %dth testing..." % (epoch))
        acc = model.test(dataloader_target_test)
        print("Current acc: %f", acc)

        # 최고 어큐러시 모델 저장
        if acc >= best_acc:
            best_acc = acc
            torch.save(model, './best_model.pt')

        print("Best acc: %f", best_acc)

    print('done')
# See PyCharmhelp at https://www.jetbrains.com/help/pycharm/
