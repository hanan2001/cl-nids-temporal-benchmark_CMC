"""
Boucle d'entraînement avec projection de gradient RGGP+
VERSION PAPER-READY - Scheduler doux + compute_val_loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.focal_loss import FocalLoss
import torch.nn.functional as F
def label_smoothing_loss(outputs, targets, smoothing=0.1):
    """Label Smoothing pour améliorer la généralisation"""
    n_classes = outputs.size(1)
    with torch.no_grad():
        true_dist = torch.zeros_like(outputs)
        true_dist.fill_(smoothing / (n_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    return torch.mean(torch.sum(-true_dist * F.log_softmax(outputs, dim=1), dim=1))
class Trainer:
    def __init__(self, model, gpm, config, use_focal_loss=True, focal_gamma=2.0, risk_mapping=None):
        self.model = model
        self.gpm = gpm
        self.config = config
        self.device = config.DEVICE
        self.risk_mapping = risk_mapping
        
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.criterion = FocalLoss(gamma=focal_gamma, risk_mapping=risk_mapping)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        # 🔥 SCHEDULER PLUS INTELLIGENT

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.loss_history = []
        self.acc_history = []
        self.val_loss_history = []
    
    def train_epoch(self, train_loader, use_projection=True):
        """Entraîne une époque avec ou sans projection de gradient"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        projection_used = 0
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            # 🔥 LABEL SMOOTHING POUR MEILLEURE GÉNÉRALISATION
            if self.use_focal_loss:
                loss = self.criterion(outputs, labels)
            else:
                loss = label_smoothing_loss(outputs, labels, smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if use_projection and len(self.gpm.memory) > 0:
                try:
                    projected_grads = self.gpm.project_gradient(self.model)
                    self.gpm.apply_projected_gradients(self.model, projected_grads)
                    projection_used += 1
                except Exception as e:
                    pass
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        if use_projection and projection_used > 0:
            stats = self.gpm.get_stats()
            print(f"   🔄 Projection RGGP+: {projection_used} batches")
        
        avg_loss = total_loss / max(1, len(train_loader))
        avg_acc = correct / max(1, total)
        
        self.loss_history.append(avg_loss)
        self.acc_history.append(avg_acc)
        
        return avg_loss, avg_acc
    
    def evaluate(self, test_loader):
        """Évalue le modèle sur un dataset de test"""
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy, all_predictions, all_labels
    
    def compute_val_loss(self, val_loader, step_scheduler=True):
        """
        Calcule la loss de validation pour early stopping
        
        Args:
            val_loader: DataLoader de validation
            step_scheduler: Si True, avance le scheduler
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        
        avg_loss = total_loss / max(1, len(val_loader))
        self.val_loss_history.append(avg_loss)
        
        # 🔥 Scheduler step UNIQUEMENT si demandé
        if step_scheduler:
            self.scheduler.step(avg_loss)
    
        return avg_loss
    
    def evaluate_by_risk_group(self, test_loader, risk_mapping):
        """Évalue la performance par groupe de risque"""
        risk_groups = {
            'critical': {'threshold': 0.7, 'correct': 0, 'total': 0},
            'high': {'threshold': 0.6, 'correct': 0, 'total': 0},
            'medium': {'threshold': 0.4, 'correct': 0, 'total': 0},
            'low': {'threshold': 0.0, 'correct': 0, 'total': 0}
        }
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                for i, label in enumerate(labels.cpu().numpy()):
                    risk = risk_mapping.get(label, 0.5)
                    group = None
                    for g in risk_groups:
                        if risk >= risk_groups[g]['threshold']:
                            group = g
                            break
                    if group is None:
                        group = 'low'
                    
                    risk_groups[group]['total'] += 1
                    if predicted[i].cpu().numpy() == label:
                        risk_groups[group]['correct'] += 1
        
        for group, data in risk_groups.items():
            total = data['total']
            if total > 0:
                data['accuracy'] = data['correct'] / total
            else:
                data['accuracy'] = 0.0
        
        return risk_groups