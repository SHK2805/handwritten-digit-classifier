from torchvision import transforms

class Transformer:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) # Normalize the image data to [-1 , 1]
        ])

    def __call__(self, img):
        """
        The __call__ method in Python is a special method that allows an instance of a class to be called as if it were a function
        :param img: PIL image
        :return: transformed image
        Usage:
                transformer = Transformer()
                img = Image.open('path/to/image.jpg')
                transformed_img = transformer(img)
        """
        return self.transform(img)

    def get_transform(self):
        return self.transform