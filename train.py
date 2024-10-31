# Instantiate the data module
data_module = MedicalSegmentationDataModule()


data_module.setup()


train_dataloader = data_module.train_dataloader()
val_dataloader = data_module.val_dataloader()

# Instantiate the Trainer
trainer = pl.Trainer(
    gradient_clip_val=1.0,
    accumulate_grad_batches=4,
    max_epochs=TrainingConfig.NUM_EPOCHS,  
    enable_model_summary=False,  
    callbacks=[model_checkpoint, lr_rate_monitor],  
    precision=16,  
    logger=tensorboard_logger,
    accelerator="auto",  
)


# Start training
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
