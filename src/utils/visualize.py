from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image, ImageDraw


def draw_graph(G, radius=1, img=None) -> np.array:
    '''
    :param G: networkx graph
    :param radius: int - radius how big must be the plotted node
    :param img: PIL Image - on which we will plot graph
    :return: numpy array with image
    '''
    if img is None:
        img = Image.new('RGB', (100, 100))
    draw = ImageDraw.Draw(img)
    points = []
    edges = []
    # nodes
    for idx, node in enumerate(G.nodes):
        # center
        x = G.nodes[idx]['x']
        y = G.nodes[idx]['y']
        points_left = (x - radius, y - radius)
        points_right = (x + radius, y + radius)
        points.append([points_left, points_right])
    # edges
    for e in G.edges:
        node_A = G.nodes[e[0]]

        node_B = G.nodes[e[1]]
        edges.append([node_A['x'], node_A['y'], node_B['x'], node_B['y']])

    for points_pair in points:
        draw.ellipse(points_pair, fill=(255, 0, 0, 255))

    for edge in edges:
        draw.line(edge, fill=(255, 0, 0, 255))

    arr = np.asarray(img)
    img.close()
    imshow(arr)
    return arr
    pass
