import torch
from modules.helper.calculate_metrics import calculate_metrics
from pathlib import Path


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criteration,
        device=None,
        print_every=1,
        save_dir=None,
        checkpoint_every=None
    ):
        # store training block
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criteration = criteration

        # Use GPU if available
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.print_every = print_every
        self.save_dir = save_dir
        self.checkpoint_every = checkpoint_every

        # Move model to device
        self.model.to(self.device)

    def _run_epoch(self, dataloader, training=True):

        total_loss = 0.0

        all_preds = []
        all_targets = []

        # Set model mode
        if training:
            self.model.train()
        else:
            self.model.eval()

        for x, y in dataloader:

            # Move batch to device
            x = x.to(self.device)
            y = y.to(self.device)

            # Clear old gradients
            if training:
                self.optimizer.zero_grad()

            # allow gradients only while training
            with torch.set_grad_enabled(training):

                # Forward pass
                outputs = self.model(x)

                # Calculate loss
                loss = self.criteration(outputs, y)

                # Backward pass and update weights
                if training:
                    loss.backward()
                    self.optimizer.step()

            # sum loss
            total_loss += loss.item()

            # predicted value
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

        # Average loss for all batches
        avg_loss = total_loss / len(dataloader)

        # Combine all predictions and targets
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        return avg_loss, all_preds, all_targets

    def _save_model(self, filename):

        if self.save_dir is None:
            return

        save_path = Path(self.save_dir)

        # create the directory if not present
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.model.state_dict(),
            save_path / filename
        )

    def train(self, epochs):

        history = []

        for epoch in range(1, epochs + 1):

            # run train epoch
            train_loss, train_preds, train_targets = self._run_epoch(
                self.train_loader,
                training=True
            )

            # run val epoch
            val_loss, val_preds, val_targets = self._run_epoch(
                self.val_loader,
                training=False
            )
            
            #get metrics
            train_metrics = calculate_metrics(actual_value=train_targets, predicted_value=train_preds)
            val_metrics = calculate_metrics(actual_value=val_targets, predicted_value=val_preds)
            
            new_metrics_data = {
                "epoch":epoch,
                
                "train_loss": train_loss,
                "val_loss":val_loss
            }
            
            for item in train_metrics:
                key = f"train_{items.lower()}"
                new_metrics_data[key] = train_metrics[item]
                
            for item in val_metrics:
                key = f"val_{items.lower()}"
                new_metrics_data[key] = val_metrics[item]
                
            history.append(new_metrics_data)

            # print every progress
            if epoch % self.print_every == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )

            # save the checkpoint
            if (
                self.save_dir is not None
                and self.checkpoint_every is not None
                and epoch % self.checkpoint_every == 0
            ):
                self._save_model(f"checkpoint_{epoch}.pt")

        # save the final model
        self._save_model("final.pt")

        return history