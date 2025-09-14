from models import HNetModel, VQWav2VecModel, ContrastiveModel
from trainfunctions import train_model, train_classifier, evaluate_classifier
from trainfunctions import ActionClassifier

num_channels = 8
num_classes = 10
seq_length = 1000
train_loader = None
val_loader = None

# Assume train_loader and val_loader are defined as above, and device = 'cuda' or 'cpu'
device = 'cpu'

# 1. Train H-Net Autoencoder
hnet = HNetModel(input_channels=num_channels, latent_channels=64)
print("Training H-Net autoencoder...")
train_model(hnet, train_loader, val_loader, epochs=5, device=device)
# Train classifier on H-Net embeddings
print("Training classifier on H-Net embeddings...")
hnet_classifier = ActionClassifier(embedding_dim=64, num_classes=num_classes)
train_classifier(hnet, hnet_classifier, train_loader, val_loader, epochs=5, device=device)
hnet_acc = evaluate_classifier(hnet, hnet_classifier, val_loader, device=device)
print(f"H-Net representation classification accuracy: {hnet_acc*100:.2f}%")

# 2. Train VQ-wav2vec style model
vq_model = VQWav2VecModel(input_channels=num_channels, encoder_dim=64, codebook_size=128, code_dim=64)
print("\nTraining VQ-wav2vec style model...")
train_model(vq_model, train_loader, val_loader, epochs=5, device=device)
print("Training classifier on VQ-wav2vec embeddings...")
vq_classifier = ActionClassifier(embedding_dim=64, num_classes=num_classes)
train_classifier(vq_model, vq_classifier, train_loader, val_loader, epochs=5, device=device)
vq_acc = evaluate_classifier(vq_model, vq_classifier, val_loader, device=device)
print(f"VQ-wav2vec representation classification accuracy: {vq_acc*100:.2f}%")

# 3. Train contrastive SimCLR-style model
# contrast_model = ContrastiveModel(input_channels=num_channels, embedding_dim=64)
# print("\nTraining contrastive model...")
# train_model(contrast_model, train_loader, val_loader, epochs=5, device=device)
# print("Training classifier on contrastive embeddings...")
# contrast_classifier = ActionClassifier(embedding_dim=64, num_classes=num_classes)
# train_classifier(contrast_model, contrast_classifier, train_loader, val_loader, epochs=5, device=device)
# contrast_acc = evaluate_classifier(contrast_model, contrast_classifier, val_loader, device=device)
# print(f"Contrastive representation classification accuracy: {contrast_acc*100:.2f}%")
