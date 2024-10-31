import torch
import torch.nn.functional as F

def binary_segmentation_loss(predictions, ground_truths, aux1=None, aux2=None, aux3=None, 
                             smooth=1e-8, alpha=0.5, beta=0.5, pos_weight=1.0, aux_weight=0.4):
    """
    Combined Tversky Loss and Weighted Binary Cross-Entropy Loss for binary segmentation with auxiliary outputs.
    
    Arguments:
    - predictions: The predicted logits from the main output of the model (before sigmoid activation).
    - ground_truths: The ground truth binary labels (0 or 1).
    - aux1, aux2, aux3: Auxiliary predicted logits from the model (before sigmoid activation).
    - smooth: A smoothing factor to avoid division by zero (default 1e-8).
    - alpha: Weight for false positives in Tversky loss (default 0.5).
    - beta: Weight for false negatives in Tversky loss (default 0.5).
    - pos_weight: Weight for the positive class in weighted BCE (default 1.0).
    - aux_weight: Weighting factor for auxiliary losses (default 0.4).
    
    Returns:
    - Combined loss value as a single scalar tensor.
    """
    
    def compute_main_loss(pred, gt):
        """
        Compute the combined Tversky and Weighted BCE loss for the main prediction.
        """
        # Flatten the predictions and ground truths
        pred = pred.view(-1)
        gt = gt.view(-1).float()
        
        # Apply sigmoid to get probabilities
        pred_sigmoid = torch.sigmoid(pred)
        
        # Tversky Loss
        TP = (pred_sigmoid * gt).sum()
        FP = ((1 - gt) * pred_sigmoid).sum()
        FN = (gt * (1 - pred_sigmoid)).sum()
        tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        tversky_loss = 1 - tversky
        
        # Weighted Binary Cross-Entropy Loss
        weights = gt * pos_weight + (1 - gt)
        bce_loss = F.binary_cross_entropy_with_logits(pred, gt, weight=weights)
        
        # Combined Loss
        return tversky_loss + bce_loss
    
    def compute_aux_loss(aux_pred, gt):
        """
        Compute the combined Tversky and Weighted BCE loss for auxiliary predictions.
        """
        return compute_main_loss(aux_pred, gt)
    
    # Compute main loss
    main_loss = compute_main_loss(predictions, ground_truths)
    
    # Initialize total loss with main loss
    total_loss = main_loss
    
    # Compute and add auxiliary losses 
    if aux1 is not None:
        aux1_loss = compute_aux_loss(aux1, ground_truths)
        total_loss += aux_weight * aux1_loss
    
    if aux2 is not None:
        aux2_loss = compute_aux_loss(aux2, ground_truths)
        total_loss += aux_weight * aux2_loss
    
    if aux3 is not None:
        aux3_loss = compute_aux_loss(aux3, ground_truths)
        total_loss += aux_weight * aux3_loss
    
    return total_loss