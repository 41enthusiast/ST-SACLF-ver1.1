import cv2
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm

def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size )

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size )

    # knowing the image center,
    # compute the coordinates of the top left and bottom right corner
    tl_x = (center_x - int(image_width))*x
    tl_y = (center_y - int(image_height))*y

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return int(tl_x), int(tl_y), int(br_x), int(br_y)

def tsne_features_vis(features, images, name, lbls, plot_size=512):
    tsne = TSNE(n_components=2).fit_transform(features)

    tx = tsne[:, 0]
    ty = tsne[:, 1]
    minmaxnorm = lambda x: (x -x.min())/(x.max()-x.min())
    tx = minmaxnorm(tx)
    ty = minmaxnorm(ty)

    tsne_plot = 255*np.ones((plot_size, plot_size, 3))
    for img, lbl,x,y in tqdm(
            zip(images, lbls, tx, ty),
            desc='Building the T-SNE plot',
            total=len(images)
    ):
        img = cv2.cvtColor(np.float32(img.permute(1,2,0).numpy()), cv2.COLOR_RGB2BGR)
        #img = cv2.rectangle(img, (0,0), tuple(img.shape[:2]), colors[lbl], 2)
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(img, x, y, plot_size, 0)
        print(tl_x, tl_y, br_x, br_y )
        tsne_plot[tl_y:br_y, tl_x:br_x,:] = img
    cv2.imshow('t-SNE', tsne_plot)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
    cv2.imwrite(f'{name}.png', tsne_plot*255)






