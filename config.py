class Config:
    # Data parameters
    DATASET_NAME = 'birds'  # 'birds' or 'flowers'
    EMBEDDING_DIM = 1024  # dimension of the text embedding
    Z_DIM = 100  # dimension of the noise vector

    # Stage I parameters
    STAGE1_G_LR = 0.0002  # learning rate for stage 1 generator
    STAGE1_D_LR = 0.0002  # learning rate for stage 1 discriminator
    STAGE1_G_HDIM = 128  # hidden dimension for stage 1 generator
    STAGE1_D_HDIM = 64   # hidden dimension for stage 1 discriminator
    STAGE1_IMAGE_SIZE = 64  # size of image in stage 1
    
    # Stage II parameters
    STAGE2_G_LR = 0.0002  # learning rate for stage 2 generator
    STAGE2_D_LR = 0.0002  # learning rate for stage 2 discriminator
    STAGE2_G_HDIM = 128  # hidden dimension for stage 2 generator
    STAGE2_D_HDIM = 64   # hidden dimension for stage 2 discriminator
    STAGE2_IMAGE_SIZE = 256  # size of image in stage 2
    
    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 600
    SNAPSHOT_INTERVAL = 50
    NUM_EXAMPLES = 6  # number of examples to visualize during training
    
    # Conditioning Augmentation
    CA_DIM = 128  # dimension of the augmented conditioning vector
    
    # Adam optimizer parameters
    BETA1 = 0.5
    BETA2 = 0.999
    
    # KL divergence regularization weight
    LAMBDA = 1.0
