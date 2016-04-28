import matplotlib.pyplot as plt
import matplotlib.cm as cm

def display_image(expected_int, image):
    print('displaying #: {}'.format(int(expected_int)))
    plt.imshow(image, cmap=cm.binary)
    plt.show(block=False)
    input('Press Enter')
