import torch
import torchtext
import spacy
from torch.utils.tensorboard import SummaryWriter

import dataclasses
# print(torch.cuda.is_available())

train, test = torchtext.datasets.AG_NEWS(root='.data', split=('train', 'test'))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.targets = []
        self.samples = []
        for target, sample in data:
            self.targets.append(target)
            self.samples.append(sample)
        self.n_targets = len(torch.unique(torch.tensor(self.targets)))

    def __getitem__(self, index):
        return self.samples[index], torch.tensor(self.targets[index]) # self.sample[index] is x, torch.tensor(self.target[index] is y


class CollateFn:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def __call__(self, batch):

        return torch.nn.utils.rnn.pack_sequence(
            [
                torch.stack([torch.from_numpy(token.vector) for token in doc])
                for doc in self.nlp.pipe(batch[0])
            ],
            enforce_sorted=False,
        ), batch[1]


class MedLstm(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = torch.nn.LSTM(input_size=96, num_layers=3, hidden_size=200, batch_first=True, bidirectional=True)
        self.classifier = torch.nn.Sequential(torch.nn.Hardswish(), torch.nn.Linear(in_features=400, out_features=num_classes))

    def setup(self, device, loss_func, optimizer: torch.optim.Optimizer, tensorboard, scheduler=None):
        self.device = device
        self.summary_writer = tensorboard
        self.to(device)
        self.loss_func = loss_func
        self.optimizer = optimizer

    def forward(self, samples):
        _, (h_n, _) = self.features(samples)
        reshape_last = h_n.reshape(self.features.num_layers, 2 if self.features.bidirectional else 1,
        h_n.shape[1], self.features.hidden_size)[-1]
        concatenated = torch.cat((reshape_last[0], reshape_last[1]), dim=-1).squeeze(dim=0)
        final = self.classifier(concatenated)
        return final

    def train_step(self, samples, targets):
        self.train()
        outputs = self(samples)
        loss = self.loss_func(outputs, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def predict(self, samples):
        return torch.argmax(self(samples))

    def accuracy(self, y_preds, targets):
        return sum(y_preds==targets)/len(y_preds)

    def dataloader(self, dataset, *args, **kwargs):
        return torch.utils.data.DataLoader(dataset, *args, **kwargs)

    def evaluate_step(self, samples, targets):
        with torch.no_grad():
            outputs = self(samples)
            y_preds = self.predict(outputs)
            loss = self.loss_func(outputs, targets)
            accuracy_step = self.accuracy(y_preds, targets)
        return loss, accuracy_step

    def training(self, epochs, train_dataset, val_dataset):
        dataloader_dict = {'batch_size': 64, 'shuffle': True}
        train_loader = self.dataloader(train_dataset, **dataloader_dict)
        val_loader = self.dataloader(val_dataset)
        for epoch in range(epochs):
            total_accuracy = 0
            total_train_loss = 0
            total_validation_loss = 0
            total_validation_accuracy = 0
            train_step = 0
            val_step = 0
            for samples, targets in train_loader:
                samples, targets = samples.to(self.device), targets.to(self.device)
                train_step += 1
                train_loss = self.train_step(samples, targets)
                total_train_loss += train_loss
            for samples, targets in val_loader:
                samples, targets = samples.to(self.device), targets.to(self.device)
                val_step += 1
                val_loss, val_acc = self.val_step(samples, targets)
                total_validation_loss += val_loss
                total_validation_accuracy += val_acc
            total_train_loss = total_train_loss/train_step
            total_validation_loss = total_validation_loss/val_step
            total_validation_accuracy = total_validation_accuracy/val_step
            self.summary_writer.add_scaler('val_loss', total_validation_loss, epoch)
            self.summary_writer.add_scaler('val_acc', total_validation_accuracy, epoch)
            self.summary_writer.add_scaler('train_loss', total_train_loss, epoch)
            print('Total_train_loss:', total_train_loss, 'Total_validation_loss:', total_validation_loss,
                    'Total_validation_accuracy:', total_validation_accuracy)

test_dataset = Dataset(test)
model = MedLstm(num_classes=10)
collate = CollateFn()
print(model(torch.randn(50, 13, 96)))


# for target, sample in train:
#     print(target)
#     print(sample)
#     break
