# Ensemble de fonctions utiles 
import cv2
from typing import Tuple, List
import os.path as osp
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
#from nuscenes.nuscenes import NuScenes

def get_color(category_name: str) -> Tuple[int, int, int]:
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """
    if 'bicycle' in category_name or 'motorcycle' in category_name:
        return 255, 61, 99  # Red
    elif 'vehicle' in category_name or category_name in ['bus', 'car', 'construction_vehicle', 'trailer', 'truck']:
        return 255, 158, 0  # Orange
    elif 'pedestrian' in category_name:
        return 0, 0, 230  # Blue
    elif 'cone' in category_name or 'barrier' in category_name:
        return 0, 0, 0  # Black
    else:
        return 255, 0, 255  # Magenta

def render_scene_channel_with_predict(nusc,
                        scene_token: str,
                        channel: str = 'CAM_FRONT',
                        freq: float = 10,
                        imsize: Tuple[float, float] = (640, 360),
                        out_path: str = None) -> None:
    """
    Renders a full scene for a particular camera channel.
    :param scene_token: Unique identifier of scene to render.
    :param channel: Channel to render.
    :param freq: Display frequency (Hz).
    :param imsize: Size of image to render. The larger the slower this will run.
    :param out_path: Optional path to write a video file of the rendered frames.
    """
    valid_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    assert imsize[0] / imsize[1] == 16 / 9, "Aspect ratio should be 16/9."
    assert channel in valid_channels, 'Input channel {} not valid.'.format(channel)

    if out_path is not None:
        assert osp.splitext(out_path)[-1] == '.avi'

    # Get records from DB
    scene_rec = nusc.get('scene', scene_token)
    sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
    sd_rec = nusc.get('sample_data', sample_rec['data'][channel])

    # Open CV init
    name = '{}: {} (Space to pause, ESC to exit)'.format(scene_rec['name'], channel)
    cv2.namedWindow(name)
    cv2.moveWindow(name, 0, 0)

    if out_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(out_path, fourcc, freq, imsize)
    else:
        out = None

    has_more_frames = True
    while has_more_frames:

        # Get data from DB
        impath, boxes, camera_intrinsic = nusc.get_sample_data(sd_rec['token'],
                                                                    box_vis_level=BoxVisibility.ANY)

        # Load and render
        if not osp.exists(impath):
            raise Exception('Error: Missing image %s' % impath)
        im = cv2.imread(impath)
        for box in boxes:
            c = get_color(box.name)
            box.render_cv2(im, view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Render
        im = cv2.resize(im, imsize)
        cv2.imshow(name, im)
        if out_path is not None:
            out.write(im)

        key = cv2.waitKey(10)  # Images stored at approx 10 Hz, so wait 10 ms.
        if key == 32:  # If space is pressed, pause.
            key = cv2.waitKey()

        if key == 27:  # if ESC is pressed, exit
            cv2.destroyAllWindows()
            break

        if not sd_rec['next'] == "":
            sd_rec = nusc.get('sample_data', sd_rec['next'])
        else:
            has_more_frames = False

    cv2.destroyAllWindows()
    if out_path is not None:
        out.release()
