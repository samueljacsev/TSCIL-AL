import torch
from agents.base import BaseLearner
from utils.buffer.buffer import Buffer


class ExperienceReplay(BaseLearner):
    """
    Follow the Paper: "On Tiny Episodic Memories in Continual Learning"
    Use the merged mini-batch (mini-batch from dataset and mini-batch from the memory buffer) to update one step.
    """
    def __init__(self, model, args):
        super(ExperienceReplay, self).__init__(model, args)
        args.eps_mem_batch = args.batch_size
        args.retrieve = 'random'
        args.update = 'random'
        self.buffer = Buffer(model, args)
        self.ncm_classifier = args.ncm_classifier
        print('ER mode: {}, NCM classifier: {}'.format(self.er_mode, self.ncm_classifier))

    def train_epoch(self, dataloader, epoch):
        total = 0
        correct = 0
        epoch_loss = 0
             
        self.model.train()
        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)

            if y.size == 1:
                y.unsqueeze()

            self.optimizer.zero_grad()

            if self.task_now > 0:  # Replay after 1st task
                x_buf, y_buf = self.buffer.retrieve(x=x, y=y)
                combined_batch = torch.cat((x_buf, x))
                combined_labels = torch.cat((y_buf, y))
                total += combined_labels.size(0)
                outputs = self.model(combined_batch)
                loss_ce = self.criterion(outputs, combined_labels)
            else:
                total += y.size(0)
                outputs = self.model(x)
                loss_ce = self.criterion(outputs, y)
                
            loss_ce.backward()
            self.optimizer_step(epoch=epoch)
        
            if self.er_mode == 'online':
                self.buffer.update(x, y)

            epoch_loss += loss_ce
            prediction = torch.argmax(outputs, dim=1)
            if self.task_now > 0:
                correct += prediction.eq(combined_labels).sum().item()
            else:
                correct += prediction.eq(y).sum().item()

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)

        return epoch_loss, epoch_acc
