from view import *
from match import *
from sfm import *
import numpy as np
import logging
import argparse
import Video_to_frames


def run(Object):
    Video_to_frames.convert(Object)
    logging.basicConfig(level=logging.INFO)
    views = create_views(Object, "jpg")
    matches = create_matches(views)
    K = np.load("./camera_params/mtx.npy")
    dist = np.load("./camera_params/dist.npy")
    sfm = SFM(views, matches, K)
    sfm.reconstruct()
    sfm.visualize_points()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Structure from Motion on a given object.')
    parser.add_argument('Object', type=str, help='The name of the object to process')
    args = parser.parse_args()
    run(args.Object)
