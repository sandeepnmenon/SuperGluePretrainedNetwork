import torch
from models.matching import Matching
import cv2
import numpy as np
import argparse


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

def read_image(path, device):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    image = image.astype('float32')

    inp = frame2tensor(image, device)
    return image, inp

def get_superglue_matches(left_image_file_path, right_image_file_path, scene='indoor', force_cpu=True):
    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1 # -1 means no limit
        },
        'superglue': {
            'weights': scene,
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
        }
    }
    matching = Matching(config).eval().to(device)

    # Load the image pair.
    image0, inp0 = read_image(
        left_image_file_path, device)
    image1, inp1 = read_image(
        right_image_file_path, device)

    with torch.no_grad():
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        left_keypoints, right_keypoints = pred['keypoints0'], pred['keypoints1']
        matches, confidence = pred['matches0'], pred['matching_scores0']
    
    # Keep the matching keypoints.
    valid = matches > -1
    left_features_matched = np.float32(left_keypoints[valid])
    right_features_matched = np.float32(right_keypoints[matches[valid]])
    match_confidence = confidence[valid]

    return left_features_matched, right_features_matched, match_confidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_image_file_path', type=str, required=True)
    parser.add_argument('--right_image_file_path', type=str, required=True)
    parser.add_argument('--scene', type=str, default='indoor')
    parser.add_argument('--force_cpu', default=False)
    args = parser.parse_args()
    left_image_file_path = args.left_image_file_path
    right_image_file_path = args.right_image_file_path
    scene = args.scene
    force_cpu = args.force_cpu
    
    left_features_matched, right_features_matched, match_confidence = get_superglue_matches(left_image_file_path, right_image_file_path, scene, force_cpu)
    print(left_features_matched)
    print(right_features_matched)
    print(match_confidence)