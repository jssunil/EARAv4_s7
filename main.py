# main.py
from dataset import generate_train_test_loader
from cifar10_architecture import CIFAR10_C1C2C3C40
from trainer import Trainer
from visualization import plot_training_history
from config import Config
import torch.optim as optim

def main():
    # Load data
    train_loader, test_loader = generate_train_test_loader(
        Config.DATA_PATH, Config.CIFAR10_MEAN, Config.CIFAR10_STD, Config.BATCH_SIZE
    )
    
    # Initialize model
    model = CIFAR10_C1C2C3C40(dropout_value=Config.DROPOUT_VALUE).to(Config.DEVICE)
    
    # Initialize optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.STEP_SIZE, gamma=Config.GAMMA)
    
    # Initialize trainer
    trainer = Trainer(model, Config.DEVICE, optimizer, scheduler)
    
    # Training loop
    train_losses, test_losses = [], []
    train_acc, test_acc = [], []
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f'Epoch {epoch+1}')
        acc, loss = trainer.train_epoch(train_loader)
        train_acc.append(acc)
        train_losses.append(loss)
        
        acc, loss = trainer.test_epoch(test_loader)
        test_acc.append(acc)
        test_losses.append(loss)
        
        if epoch < Config.STEP_SIZE:
            scheduler.step()

if __name__ == "__main__":
    main()
