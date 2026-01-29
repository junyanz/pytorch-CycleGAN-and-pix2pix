import torch



def content_adversarial_loss(G, D, x, y):

    # Possibly add something to remove log(0) errors?

    # Possibly should return a negative number?

    return torch.log(1 - D(G(x))).mean() + torch.log(D(y)).mean()

# def cycle_consistency_loss_function(G, F, x, f):
#     """
#     Calculates the cycle consistency loss
    
#     G: LF -> HF Generator
#     F: HF -> LF Generator
#     x: 
#     f: 
#     """

#     l1_loss = nn.L1Loss()

#     # Perform LF -> HF
#     pred_x = G(F(x))
#     pred_x_loss = l1_loss(pred_x)

#     # Perform HF -> LF
#     pred_y = G(F(y))
#     pred_y_loss = l1_loss(pred_y)

#     # Combine to calculate loss
#     return pred_x_loss + pred_y_loss


# def identity_mapping_loss(G, F, x, y):

#     l1_loss = nn.L1Loss()

#     return l1_loss(G(y), y) + l1_loss(F(x), x)




