# Brouillon pour stocker des bouts de codes qui peuvent reservir

# 15/06


# 1 
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
anntoken = "bc3180b07f8e4a728f504ded654df56f"
ann_record = nusc.get('sample_annotation',anntoken)
sample_record = nusc.get('sample', ann_record['sample_token'])
boxes, cam = [], []
cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
print(cams)

inst_token = nusc.get('instance',ann_record['instance_token'])
print(inst_token)

cams_check = []
for cam in cams:
    _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=BoxVisibility.ANY,
                                            selected_anntokens=[anntoken])
    if len(boxes) > 0:
        cams_check += [cam]

print(cams_check)

#['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
#{'token': 'c1958768d48640948f6053d04cffd35b', 'category_token': 'fd69059b62a3469fbaef25340c0eab7f', 'nbr_annotations': 39, 'first_annotation_token': '49f76277d07541c5a584aa14c9d28754', 'last_annotation_token': 'bc3180b07f8e4a728f504ded654df56f'}
#['CAM_FRONT', 'CAM_FRONT_LEFT']


# dans cette extrait, j'ai oublié de changer sample_record à chaque tour de boucle
# normalement boxes est censé être vide sauf quand on sera sur le bon sample de la bonne annotation
# mais non, a chaque fois quasiment j'avais des boxes retourné, je n'ai pas trouvé pourquoi. 
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

def find_vehicle_follow(instance_token):
    instance = nusc.get('instance',instance_token)
    last_token = instance["last_annotation_token"]
    curr_token = instance["first_annotation_token"]

    while curr_token != last_token:
        curr_ann = nusc.get('sample_annotation',curr_token)
        curr_sample = nusc.get('sample',curr_ann['sample_annotation'])
        cams_check = []
        for cam in cams:
            _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=BoxVisibility.ANY,
                                                    selected_anntokens=[curr_token])
            if len(boxes) > 0:
                cams_check += [cam]
        print(cams_check)
        curr_token = curr_ann['next']
        
        #nusc.render_annotation(curr_token)

        
find_vehicle_follow("c1958768d48640948f6053d04cffd35b")


# v2 14h, je decale sur une autre méthode (pas ce code):  

from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

# renvoie 

# renvoie vrai et un dataframe rempli s'il y a un véhicule en face, faux et dataframe vide sinon
def find_vehicle_in_front(instance_token,utime):
    instance = nusc.get('instance',instance_token)
    last_token = instance["last_annotation_token"]
    curr_token = instance["first_annotation_token"]
    # sans vitesse du véhicule en face actuellement
    columns = ["distance,throttle,ego_speed"]
    dataframe = pd.DataFrame(columns=columns)
    rows_list = []
    
    while curr_token != last_token:
        curr_ann = nusc.get('sample_annotation',curr_token)
        curr_sample = nusc.get('sample',curr_ann['sample_token'])
        cams_check = []
        
        # récupérer les caméras qui ont vu l'annotation
        for cam in cams:
            _, boxes, _ = nusc.get_sample_data(curr_sample['data'][cam], box_vis_level=BoxVisibility.ANY,
                                                    selected_anntokens=[curr_token])
            if len(boxes) > 0:
                cams_check += [cam]
        #print(cams_check)
        curr_token = curr_ann['next']    
        
        #calcul distance entre ego et le vehicule
        lidar = nusc.get('sample_data',curr_sample['data']['LIDAR_TOP'])
        ego_pos = nusc.get('ego_pose',lidar['ego_pose_token'])
        dist = np.linalg.norm(np.array(ego_pos['translation']) - np.array(curr_ann['translation']))
        #print(dist)
        dic = {'distance':dist}
        rows_list += [dic]
        print(curr_sample["timestamp"] in utime)
        print(curr_sample["timestamp"])
    print(len(rows_list))
        
    
dic_scene = nusc_can.get_messages(scene_test['name'],'vehicle_monitor')
utime = [ d["utime"] for d in dic_scene ]
print(len(utime))
print(utime)
find_vehicle_in_front("c1958768d48640948f6053d04cffd35b",utime)

scene_test = nusc.scene[58]
dic_scene = nusc_can.get_messages(scene_test['name'],'vehicle_monitor')
features = ["vehicle_speed","steering","throttle","left_signal","right_signal"]
df_scene = pd.DataFrame.from_dict(dic_scene)[features]
#dic_scene



# 


last = nusc.get('sample',scene['last_sample_token'])
while(curr_sample['timestamp'] < last['timestamp']):
    #print(curr_sample['timestamp'] )
    list_sample += [curr_sample['timestamp']]
    curr_sample = nusc.get('sample',curr_sample['next'])
    i += 1
print(i)
print(curr_sample['timestamp'])
print(len(utime[i:]))
print([list_sample[i] - list_sample[i+1] for i in range(len(list_sample)-1)])

###
