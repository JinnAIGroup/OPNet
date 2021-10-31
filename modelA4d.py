'''   JLL, 2021.10.21, 10.31
modelA4d = UNet + efficientnet + imagenet
UNet https://github.com/YassineYousfi
efficientnet https://github.com/qubvel/segmentation_models.pytorch
comma10k data:
   imgs: *.png (874, 1164, 3) = (H, W, C)
  masks: *.png (874, 1164, 1)
modelA4a I/O

1. Task: Multiclass semantic segmentation (NUM_CLASSES = 5)
2. Input: comma10k
   Imgs = /home/jinn/dataAll/comma10k/imgs/*.png
   YUV  = /home/jinn/dataAll/comma10k/imgs_yuv/*.h5
   Msks = /home/jinn/dataAll/comma10k/masks/*.png
   png imgs = (H, W, C) = (874, 1164, 3); masks = (874, 1164, 1);  imgs2 = (1208, 1928, 3);
3. Output:
   plt.title("Training Loss")
   plt.title("Training Accuracy")
   plt.title("Validation Loss")
   plt.title("Validation Accuracy")
   plot_predictions(train_images[:4], colormap, model=model)
     binary mask: one-hot encoded tensor = (?, ?, ?)
     visualize: RGB segmentation masks (each pixel by a unique color corresponding
       to each predicted label from the human_colormap.mat file)
4. No Run:
'''
