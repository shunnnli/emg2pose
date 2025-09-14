def train_model(model, train_loader, val_loader=None, optimizer=None, epochs=10, device='cpu'):
    """
    Generic training loop for a model using its own compute_loss.
    Works for both unsupervised models (HNet, VQWav2Vec, Contrastive) and supervised (if any).
    """
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for (x, y) in train_loader:
            x = x.to(device)
            # Compute loss (for unsupervised, y is ignored inside compute_loss)
            loss = model.compute_loss(x, target=y.to(device) if isinstance(y, torch.Tensor) else None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        if val_loader:
            # If validation loader is provided, compute validation loss (unsupervised loss or supervised as appropriate)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (x, y) in val_loader:
                    x = x.to(device)
                    val_loss += model.compute_loss(x, target=y.to(device) if isinstance(y, torch.Tensor) else None).item()
            val_loss /= len(val_loader)
            model.train()
            print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")
        else:
            print(f"Epoch {epoch}: train_loss={avg_loss:.4f}")
    return model

def evaluate_classifier(rep_model, classifier, data_loader, device='cpu'):
    """
    Evaluate classification accuracy of the classifier (with rep_model providing embeddings).
    """
    rep_model.to(device).eval()
    classifier.to(device).eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (x, y) in data_loader:
            x = x.to(device)
            y = y.to(device)
            # Get embedding from representation model
            emb = rep_model.get_embedding(x)
            # Get predictions from classifier
            logits = classifier(emb)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total if total > 0 else 0
    return acc

def train_classifier(rep_model, classifier, train_loader, val_loader=None, epochs=5, device='cpu'):
    """
    Train the classifier on top of a frozen representation model's embeddings.
    rep_model: a pretrained representation model (frozen during classifier training)
    classifier: an nn.Module that takes embedding -> predicts class
    """
    # Freeze representation model parameters
    rep_model.eval()
    for param in rep_model.parameters():
        param.requires_grad = False
    classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    for epoch in range(1, epochs+1):
        classifier.train()
        total_loss = 0.0
        for (x, y) in train_loader:
            x = x.to(device)
            y = y.to(device)
            # Compute embedding (no grad for rep_model)
            with torch.no_grad():
                emb = rep_model.get_embedding(x)
            # Forward through classifier and compute loss
            logits = classifier(emb)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        # Validation accuracy
        val_acc = 0.0
        if val_loader:
            val_acc = evaluate_classifier(rep_model, classifier, val_loader, device)
            print(f"Epoch {epoch}: classifier_train_loss={avg_loss:.4f}, val_acc={val_acc*100:.2f}%")
        else:
            print(f"Epoch {epoch}: classifier_train_loss={avg_loss:.4f}")
    return classifier



class ActionClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(ActionClassifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
    def forward(self, emb):
        # emb: [B, embedding_dim]
        return self.fc(emb)  # logits for each class
    
# Example usage of classifier:
# For HNet embedding_dim = latent_channels (after pooling)
# For VQWav2Vec embedding_dim = encoder_dim (GRU hidden size)
# For ContrastiveModel embedding_dim = as defined
