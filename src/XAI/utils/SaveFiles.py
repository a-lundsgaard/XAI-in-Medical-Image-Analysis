import os
import shutil
import matplotlib.pyplot as plt

class PLTSaver:

    def __init__(self, xai_method, custom_save_dir="default"):
        self.custom_save_dir = custom_save_dir
        self.data_dir = f"outputs/{xai_method}/"
        self.xai_method = xai_method
        self.save_output = False

    def getDataDir(self):
        return self.data_dir + self.custom_save_dir

    def set_custom_save_dir(self, custom_save_dir, save_output=False):
        self.save_output = save_output
        self.custom_save_dir = custom_save_dir
        if save_output:
            self.createFolder()

    def createFolder(self):
        """
        Save the saliency maps as images in the specified directory.
        """
        path = self.getDataDir()
        if self.save_output:
            if not os.path.exists(path):
                os.makedirs(path)
                #shutil.rmtree(path)
            #os.makedirs(path)
    
    def handleSaveImage(self, id, plt: plt, name):
        """
        Saves the generated image to the specified directory.
        Args:
        - output_dir: Directory where the image will be saved.
        - image: Image to be saved.
        - label: Label of the image.
        """
        if self.save_output:
            print("save name: ",f"{self.getDataDir()}/{id}_{name}.png")
            plt.savefig(f"{self.getDataDir()}/{id}_{name}.png")
