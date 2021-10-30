'''   JLL, 2021.10.21
modelA4d = UNet + efficientnet + imagenet
UNet https://github.com/YassineYousfi
efficientnet https://github.com/qubvel/segmentation_models.pytorch
Input data: comma10k (png)
png imgs = (H, W, C) = (874, 1164, 3); masks = (874, 1164, 1);  imgs2 = (1208, 1928, 3);

1. Task: Multiclass semantic segmentation
   png imgs = (H, W, C) = (874, 1164, 3); masks = (874, 1164, 1);  imgs2 = (1208, 1928, 3);
   NUM_CLASSES = 5
   Imgs = /home/jinn/dataAll/comma10k/imgs/*.png
   Msks = /home/jinn/dataAll/comma10k/masks/*.png
2. Input: comma10k
   Imgs = /home/jinn/dataAll/comma10k/imgs/*.png
   YUV  = /home/jinn/dataAll/comma10k/imgs_yuv/*.h5
   Msks = /home/jinn/dataAll/comma10k/masks/*.png
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
