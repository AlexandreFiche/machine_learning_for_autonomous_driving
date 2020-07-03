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

# 16 juin

from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

# renvoie une liste des informations du vehicule (meme nombre que le nombre d'annotation)
# par defaut ce nombre peut être differents car les timestamps ne sont pas les meme
def get_list_info(instance_token):
    instance = nusc.get('instance',instance_token)
    ann = nusc.get('sample_annotation',instance["first_annotation_token"])
    sample = nusc.get('sample',ann['sample_token'])
    scene = nusc.get('scene',sample['scene_token'])
    dict_scene = nusc_can.get_messages(scene['name'],'vehicle_monitor')
    curr_sample = sample
    i = 0
    list_info = []
    last = nusc.get('sample',scene['last_sample_token'])
    
    while(curr_sample['timestamp'] < last['timestamp']):
        if(curr_sample['timestamp'] > dict_scene[i]['utime'] and i < len(dict_scene)-1):
            i += 1
        list_info += [dict_scene[i]]
        curr_sample = nusc.get('sample',curr_sample['next'])
    if(curr_sample['timestamp'] < dict_scene[i]['utime'] and i < len(dict_scene)-1):
        i += 1
    list_info += [dict_scene[i]]
    
    return list_info

# renvoie vrai et un un tableau rempli si l'instance est en face d'ego
def find_vehicle_in_front(instance_token):
    instance = nusc.get('instance',instance_token)
    last_token = instance["last_annotation_token"]
    curr_token = instance["first_annotation_token"]
    info_list = get_list_info(instance_token)
    rows_list = []
    i = 0
    # Pour chaque enregistrement de l'annoation on ajoute une ligne avec les elements
    while curr_token != last_token:
        curr_ann = nusc.get('sample_annotation',curr_token)
        curr_sample = nusc.get('sample',curr_ann['sample_token'])
        cams_check = []
        
        # récupérer les caméras qui ont vu l'annotation
        _, boxes, _ = nusc.get_sample_data(curr_sample['data']['CAM_FRONT'], box_vis_level=BoxVisibility.ANY,
                                                selected_anntokens=[curr_token])
        if len(boxes) > 0 and abs(info_list[i]['steering']) < 100:
            #calcul distance entre ego et le vehicule
            lidar = nusc.get('sample_data',curr_sample['data']['LIDAR_TOP'])
            ego_pos = nusc.get('ego_pose',lidar['ego_pose_token'])
            dist = np.linalg.norm(np.array(ego_pos['translation']) - np.array(curr_ann['translation']))
            dic = {'distance':dist,'throttle':info_list[i]['throttle'],'ego_speed':info_list[i]['vehicle_speed']
                  ,'brake':info_list[i]['brake']}
            rows_list += [dic]
        curr_token = curr_ann['next']   
        i +=1
    #print(len(rows_list)," lignes ajoutées")
    return len(rows_list)!=0,rows_list


out.release()


blackint = nusc_can.can_blacklist
blacklist = [ "scene-0"+ str(i) for i in blackint]

# Liste toutes les instances d'une scene
def get_instances_scene(scene):
    sample = nusc.get('sample',scene['first_sample_token'])
    list_instances = []
    
    while sample['token'] != scene['last_sample_token']:
        anns = sample['anns']
        for ann_token in anns:
            ann = nusc.get('sample_annotation',ann_token)
            instance = nusc.get('instance',ann['instance_token'])
            category = nusc.get('category',instance['category_token'])
            if not instance in list_instances and "vehicle" in category['name']:
                #print(category['name'])
                list_instances += [instance]
        sample = nusc.get('sample',sample['next'])
    return list_instances


# Explore chaque scene, puis chaque instance de cette scene qui est un vehicle en mouvement (devant)
# Cree un dataframe avec pour entree distance au vehicle, ego_vitesse, ego_accel, ego_brake 
# et vehicle_vitesse (pas mtn)
def build_dataframe_for_vehicle_in_front():
    scenes = nusc.scene[:100]
    list_rows = []
    for s in scenes:
        if s not in blacklist and s not in ["scene-003"]:
            list_instances = get_instances_scene(s)
            for inst in list_instances:
                ok, res = find_vehicle_in_front(inst['token'])
                if ok:
                    list_rows += res
    dataframe = pd.DataFrame.from_dict(list_rows)
    print(dataframe)
    print(dataframe.describe())
    return dataframe
    
#find_vehicle_in_front("c1958768d48640948f6053d04cffd35b")
# 15k ligne sans contrainte sur steering (100 scenes)
df_vehicle = build_dataframe_for_vehicle_in_front()


# 17 juin modification pretraitement: sauvegarde des fonctions

from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

# renvoie une liste des informations du vehicule (meme nombre que le nombre d'annotation)
# par defaut ce nombre peut être differents car les timestamps ne sont pas les meme
def get_list_info(instance_token):
    instance = nusc.get('instance',instance_token)
    ann = nusc.get('sample_annotation',instance["first_annotation_token"])
    sample = nusc.get('sample',ann['sample_token'])
    scene = nusc.get('scene',sample['scene_token'])
    dict_scene = nusc_can.get_messages(scene['name'],'vehicle_monitor')
    curr_sample = sample
    i = 0
    list_info = []
    last = nusc.get('sample',scene['last_sample_token'])
    
    while(curr_sample['timestamp'] < last['timestamp']):
        if(curr_sample['timestamp'] > dict_scene[i]['utime'] and i < len(dict_scene)-1):
            i += 1
        list_info += [dict_scene[i]]
        curr_sample = nusc.get('sample',curr_sample['next'])
    if(curr_sample['timestamp'] < dict_scene[i]['utime'] and i < len(dict_scene)-1):
        i += 1
    list_info += [dict_scene[i]]
    
    return list_info

# renvoie vrai et un un tableau rempli si l'instance est en face d'ego
def find_vehicle_in_front(instance_token):
    instance = nusc.get('instance',instance_token)
    last_token = instance["last_annotation_token"]
    curr_token = instance["first_annotation_token"]
    info_list = get_list_info(instance_token)
    rows_list = []
    i = 0
    # Pour chaque enregistrement de l'annoation on ajoute une ligne avec les elements
    while curr_token != last_token:
        curr_ann = nusc.get('sample_annotation',curr_token)
        curr_sample = nusc.get('sample',curr_ann['sample_token'])
        scene = scene = nusc.get('scene',curr_sample['scene_token'])
        cams_check = []
        
        # récupérer les caméras qui ont vu l'annotation
        _, boxes, _ = nusc.get_sample_data(curr_sample['data']['CAM_FRONT'], box_vis_level=BoxVisibility.ANY,
                                                selected_anntokens=[curr_token])
        if len(boxes) > 0 and abs(info_list[i]['steering']) < 100:
            #calcul distance entre ego et le vehicule
            lidar = nusc.get('sample_data',curr_sample['data']['LIDAR_TOP'])
            ego_pos = nusc.get('ego_pose',lidar['ego_pose_token'])
            dist = np.linalg.norm(np.array(ego_pos['translation']) - np.array(curr_ann['translation']))
            dic = {'distance':dist,'throttle':info_list[i]['throttle'],'ego_speed':info_list[i]['vehicle_speed']
                  ,'brake':info_list[i]['brake'],'future_throttle':info_list[i+1]['throttle'],'future_brake':info_list[i+1]['brake']}
            rows_list += [dic]
            if info_list[i]['brake'] > 10:
                #print(scene['name'])
                pass
                
        curr_token = curr_ann['next']   
        i +=1
    #print(len(rows_list)," lignes ajoutées")
    return len(rows_list)!=0,rows_list


def show_infos(dataframe,num_frame):
    if num_frame < taille:
        cv2.putText(im, 'vitesse:'+ str(dataframe.at[int(num_frame/25),'ego_speed']), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
def gestion(dataframe):
    i = 0
    nb_ligne = dataframe.shape[0]
    sample = nusc.get('sample',scene['first_sample_token'])
    list_instances = []
    
    while sample['token'] != scene['last_sample_token']:
        #print(sample['timestamp'],' a ')
        df = dataframe[dataframe['timestamp'] == sample['timestamp']]
        i += 1
        if i == 6:
            i = 0
            sample = nusc.get('sample',sample['next'])


# renvoie vrai et un un tableau rempli si l'instance est en face d'ego
def find_vehicle_in_front(instance_token):
    instance = nusc.get('instance',instance_token)
    last_token = instance["last_annotation_token"]
    curr_token = instance["first_annotation_token"]
    info_list = get_list_info(instance_token)
    rows_list = []
    i = 0
    # Pour chaque enregistrement de l'annoation on ajoute une ligne avec les elements
    while curr_token != last_token:
        curr_ann = nusc.get('sample_annotation',curr_token)
        curr_sample = nusc.get('sample',curr_ann['sample_token'])
        scene = scene = nusc.get('scene',curr_sample['scene_token'])
        cams_check = []
        
        # récupérer les caméras qui ont vu l'annotation
        _, boxes, _ = nusc.get_sample_data(curr_sample['data']['CAM_FRONT'], box_vis_level=BoxVisibility.ANY,
                                                selected_anntokens=[curr_token])
        if len(boxes) > 0 and abs(info_list[i]['steering']) < 100:
            #calcul distance entre ego et le vehicule
            lidar = nusc.get('sample_data',curr_sample['data']['LIDAR_TOP'])
            ego_pos = nusc.get('ego_pose',lidar['ego_pose_token'])
            dist = np.linalg.norm(np.array(ego_pos['translation']) - np.array(curr_ann['translation']))
            dic = {'scene':scene['name'],'timestamp':curr_sample['timestamp'],'inst_token':instance_token,'ann_token':curr_token,'distance':round(dist,3),'throttle':info_list[i]['throttle'],'ego_speed':round(info_list[i]['vehicle_speed'],3)
                  ,'brake':info_list[i]['brake'],'future_throttle':info_list[i+1]['throttle'],'future_brake':info_list[i+1]['brake']}
            rows_list += [dic]
            if info_list[i]['brake'] > 10:
                #print(scene['name'])
                pass
                
        curr_token = curr_ann['next']   
        i +=1


# 18 juin

# renvoie vrai et un un tableau rempli si l'instance est en face d'ego
def find_vehicle_in_front_b(instance_token):
    instance = nusc.get('instance',instance_token)
    last_token = instance["last_annotation_token"]
    curr_token = instance["first_annotation_token"]
    info_list = get_list_info(instance_token)
    rows_list = []
    i = 0
    # Pour chaque enregistrement de l'annoation on ajoute une ligne avec les elements
    while curr_token != last_token:
        curr_ann = nusc.get('sample_annotation',curr_token)
        curr_sample = nusc.get('sample',curr_ann['sample_token'])
        scene = scene = nusc.get('scene',curr_sample['scene_token'])
        cams_check = []
        
        # récupérer les caméras qui ont vu l'annotation
        _, boxes, _ = nusc.get_sample_data(curr_sample['data']['CAM_FRONT'], box_vis_level=BoxVisibility.ANY,
                                                selected_anntokens=[curr_token])
        if len(boxes) > 0 and abs(info_list[i]['steering']) < 100:
            #calcul distance entre ego et le vehicule
            lidar = nusc.get('sample_data',curr_sample['data']['LIDAR_TOP'])
            ego_pos = nusc.get('ego_pose',lidar['ego_pose_token'])
            dist = np.linalg.norm(np.array(ego_pos['translation']) - np.array(curr_ann['translation']))
            dic = {'scene':scene['name'],'timestamp':curr_sample['timestamp'],'inst_token':instance_token,'ann_token':curr_token,'distance':round(dist,3),'throttle':info_list[i]['throttle'],'ego_speed':round(info_list[i]['vehicle_speed'],3)
                  ,'brake':info_list[i]['brake'],'future_throttle':info_list[i+1]['throttle'],'future_brake':info_list[i+1]['brake']}
            rows_list += [dic]
            if info_list[i]['brake'] > 10:
                #print(scene['name'])
                pass
                
        curr_token = curr_ann['next']   
        i +=1
 
    print(len(rows_list),len(info_list))
    return len(rows_list)!=0,rows_list


from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

def get_list_info_v2(instance_token):
    instance = nusc.get('instance',instance_token)
    ann = nusc.get('sample_annotation',instance["first_annotation_token"])
    sample = nusc.get('sample',ann['sample_token'])
    scene = nusc.get('scene',sample['scene_token'])
    dict_scene = nusc_can.get_messages(scene['name'],'vehicle_monitor')
    curr_sample = sample
    i = 0
    list_info = []
    last = nusc.get('sample',scene['last_sample_token'])
    while curr_sample['timestamp'] <= dict_scene[i]['utime']:
        i += 1
        curr_sample = nusc.get('sample',curr_sample['next'])
        
    while(curr_sample['timestamp'] < last['timestamp']):
        if(curr_sample['timestamp'] > dict_scene[i]['utime'] and i < len(dict_scene)-1):
            i += 1
        list_info += [dict_scene[i]]
        curr_sample = nusc.get('sample',curr_sample['next'])
    if(curr_sample['timestamp'] < dict_scene[i]['utime'] and i < len(dict_scene)-1):
        i += 1
    list_info += [dict_scene[i]]
    
    return list_info
# 1532402936198962 1532402936699359 1532402937198682

# renvoie une liste des informations du vehicule (meme nombre que le nombre d'annotation)
# par defaut ce nombre peut être differents car les timestamps ne sont pas les meme
def get_list_info(instance_token):
    instance = nusc.get('instance',instance_token)
    ann = nusc.get('sample_annotation',instance["first_annotation_token"])
    sample = nusc.get('sample',ann['sample_token'])
    scene = nusc.get('scene',sample['scene_token'])
    dict_scene = nusc_can.get_messages(scene['name'],'vehicle_monitor')
    curr_sample = sample
    i = 0
    list_info = []
    last = nusc.get('sample',scene['last_sample_token'])
    
    while(curr_sample['timestamp'] < last['timestamp']):
        if(curr_sample['timestamp'] > dict_scene[i]['utime'] and i < len(dict_scene)-1):
            i += 1
        list_info += [dict_scene[i]]
        curr_sample = nusc.get('sample',curr_sample['next'])
    if(curr_sample['timestamp'] < dict_scene[i]['utime'] and i < len(dict_scene)-1):
        i += 1
    list_info += [dict_scene[i]]
    #print([ e['utime'] - 1532402900000000  for e in dict_scene])
    return list_info

# 19 juin
df_ego = df[df['inst_token'] == "vehicle_info"]
#df_ego

list_vec = [(df_ego.iloc[i+1]['ego_pos'][0] - df_ego.iloc[i]['ego_pos'][0],
            df_ego.iloc[i+1]['ego_pos'][1] - df_ego.iloc[i]['ego_pos'][1]) 
            for i in range(df_ego.shape[0]-1) ]
list_vitesse = [df_ego.iloc[i]['ego_speed'] for i in range(df_ego.shape[0]-1)]
list_vec_norm = [ (v[0]/np.sqrt((v[0]*v[0] + v[1]*v[1])),v[1]/np.sqrt((v[0]*v[0] + v[1]*v[1])))
            for v in list_vec ]
#print(list_vec)
list_vec_norm

for i in range(df_ego.shape[0]-1):
    # tuple(map(operator.add, df_ego.iloc[i]['ego_pos'],
    r = [e * list_vitesse[i]/3.6*0.5 for e in list_vec_norm[i]]
    #print(list_vec_norm[i])
    new_pos =   list(map(operator.add, df_ego.iloc[i]['ego_pos'],r))
    new_pos = [round(e,3) for e in new_pos]
    
    print(new_pos,df_ego.iloc[i+1]['ego_pos'])

# 23 juin

for box in boxes:
    #angle = 0
    if  (((box.center[0] > -2 - angle and box.center[0] < 2 - angle and box.center[2] < 20) or
        (box.center[0] > -3 - angle and box.center[0] < 3 - angle and box.center[2] < 40 and box.center[2] > 20) or 
        (box.center[0] > -6 - angle and box.center[0] < 6 - angle and box.center[2] > 40)) and "vehicle" in box.name
        and box.center[2] < dmin):
        dmin = box.center[2]
        minbox = box 

# Affichage informations
if sample['token'] != scene['last_sample_token']:
    if not df_curr.empty:
        #print("passe")
        if dmin != 999:
            cv2.line(im, (int(800+minbox.center[0]*20), 100), (int(800+minbox.center[0]*20), 800), (255, 255, 0), thickness=2)
            cv2.putText(im, 'Center:'+ str(round(minbox.center[0],3))+"\n      "+str(round(minbox.center[2],2)), 
                        (int(800+minbox.center[0]*10),250), 
                        font, 
                        fontScale, 
                        (255, 0, 255),
                        lineType)
        cv2.line(im, (int(1200-angle*20), 100), (int(1200-angle*20), 800), (255, 0, 0), thickness=2)
        cv2.line(im, (int(400-angle*20), 100), (int(400-angle*20), 800), (255, 0, 0), thickness=2)



import cv2
from typing import Tuple, List
import os.path as osp
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
import operator

# parametres pour cv2
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (0,500)
fontScale              = 1
fontColor              = (255,255,255)
color              = (255,0,0)
lineType               = 2
pas = (0,50)

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

def affichage(im,df_curr):
           
    cv2.putText(im, 'Vitesse:'+ str(df_curr.iat[0,9]), 
        bottomLeftCornerOfText, 
        font, 
        fontScale, 
        fontColor,
        lineType)
    cv2.putText(im, 'Angle volant:'+ str(df_curr.iat[0,8]/20), 
        tuple(map(operator.add, bottomLeftCornerOfText,(0,50))), 
        font, 
        fontScale, 
        fontColor,
        lineType)
    cv2.putText(im, 'Acceleration:'+ str(df_curr.iat[0,10]), 
        tuple(map(operator.add, bottomLeftCornerOfText,(0,100))), 
        font, 
        fontScale, 
        fontColor,
        lineType)

    cv2.putText(im, 'Frein:'+ str(df_curr.iat[0,11]), 
        tuple(map(operator.add, bottomLeftCornerOfText,(0,150))), 
        font, 
        fontScale, 
        fontColor,
        lineType)
    cv2.putText(im, 'Acceleration (Pred):'+ str(df_curr.iat[0,12]), 
        tuple(map(operator.add, bottomLeftCornerOfText,(0,200))), 
        font, 
        fontScale, 
        fontColor,
        lineType)

    cv2.putText(im, 'Frein (Pred):'+ str(df_curr.iat[0,11]), 
        tuple(map(operator.add, bottomLeftCornerOfText,(0,250))), 
        font, 
        fontScale, 
        fontColor,
        lineType)

    if df_curr.shape[0] > 1:
        cv2.putText(im, 'Distance:'+ str(df_curr.iloc[1]['distance']), 
            tuple(map(operator.add, bottomLeftCornerOfText,(0,300))), 
            font, 
            fontScale, 
            color,
            lineType)    
        
def draw_rect(im,selected_corners, color):
    prev = selected_corners[-1]
    for corner in selected_corners:
        cv2.line(im,
                 (int(prev[0]), int(prev[1])),
                 (int(corner[0]), int(corner[1])),
                 color, 2)
        prev = corner
        
def render_scene_channel_with_predict(nusc,
                        scene_token: str, dataframe,
                        channel: str = 'CAM_FRONT',
                        freq: float = 10,
                        imsize: Tuple[float, float] = (960, 540),
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

    # parametres pour cv2
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 1
    fontColor              = (255,255,255)
    color              = (255,0,0)
    lineType               = 2
    pas = (0,50)
    # 900* 1600
    # parametres pour afficher infos
    i = 0
    taille = dataframe.shape[0]
    scene_token = nusc.field2token('scene', 'name', dataframe.at[0,'scene'])[0]
    scene = nusc.get('scene',scene_token)
    sample = nusc.get('sample',scene['first_sample_token'])
    df_curr = dataframe[dataframe['timestamp'] == sample['timestamp']]
    df_curr = df_curr.sort_values(by='distance').reset_index(drop=True)
    print(df_curr)
    has_more_frames = True
    angle = df_curr.iat[0,8]
    xmin = 10
    xmax = - 10
    colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155))
    # -30.671 39.22340
    borne_a = 600
    borne_b = 1000
    while has_more_frames:
        
        ann = df_curr[df_curr["inst_token"]=="98300b9c4acb4da9a7aecd0084650265"]
        ann_tok = ann['ann_token']
        # selected_anntokens=[ann_tok.iat[0]]
        # Get data from DB
        impath, boxes, camera_intrinsic = nusc.get_sample_data(sd_rec['token'],
                                                                    box_vis_level=BoxVisibility.ANY)
                                        

        # Load and render
        if not osp.exists(impath):
            raise Exception('Error: Missing image %s' % impath)
        im = cv2.imread(impath)

        dmin = 999
        minbox = None

        for box in boxes:
            corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
            if (box.center[2] < dmin and corners.T[4][0] < borne_b-angle and corners.T[6][0] > borne_a-angle 
                and "vehicle" in box.name):
                dmin = box.center[2]
                minbox = box
                
            if box.center[0] < xmin:
                xmin = box.center[0]
            if box.center[0] > xmax:
                xmax = box.center[0]
                
            #print(box.center,ann["distance"].iat[0])
        if dmin != 999:
            c = get_color(minbox.name)
            #minbox.render_cv2(im, view=camera_intrinsic, normalize=True, colors=(c, c, c))
            corners = view_points(minbox.corners(), camera_intrinsic, normalize=True)[:2, :]
            #draw_rect(im,corners.T[:4], colors[0][::-1])
            draw_rect(im,corners.T[4:], colors[1][::-1])
            #print(corners.T[4:])



        # Affichage informations
        if sample['token'] != scene['last_sample_token']:
            if not df_curr.empty:
                #print("passe")
                if dmin != 999:
                    cv2.line(im, (int((corners.T[4][0]+corners.T[6][0])/2), 400), (int((corners.T[4][0]+corners.T[6][0])/2), 600), (255, 255, 0), thickness=2)
                    cv2.putText(im, 'Center:'+ str(round(minbox.center[0],3))+"\n      "+str(round(minbox.center[2],2)), 
                                (int(800+minbox.center[0]*10),250), 
                                font, 
                                fontScale, 
                                (255, 0, 255),
                                lineType)
                cv2.line(im, (int(borne_b-angle), 400), (int(borne_b-angle), 600), (255, 0, 0), thickness=2)
                cv2.line(im, (int(borne_a-angle), 400), (int(borne_a-angle), 600), (255, 0, 0), thickness=2)

                affichage(im,df_curr)


            else:
                print(sample['timestamp'])
            if i%6 == 0 and i != 0:
                sample = nusc.get('sample',sample['next'])
                df_curr = dataframe[dataframe['timestamp'] == sample['timestamp']]
                df_curr = df_curr.sort_values(by='distance').reset_index(drop=True)
                #print("changement")
                if not df_curr.empty:
                    angle = df_curr.iat[0,8]
                #angle = 0
        else:
            print("fin des données ",i)

            
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
        i += 1
    print("nombre de frame: ",i)
    print(xmin,xmax)
    cv2.destroyAllWindows()
    if out_path is not None:
        out.release()

# 01 Juillet

# Fonction qui déroule une scene en se basant sur les predictions faites, 
# Point de départ = pos initial puis après calcul à chaque tour de boucle par rapport aux retours des modèles
def predict_scene_v1(scene_name):
    my_scene_token = nusc.field2token('scene', 'name', scene_name)[0]
    scene = nusc.get('scene',my_scene_token)
    #nusc.render_scene_channel(my_scene_token, 'CAM_FRONT')
    df = build_dataframe_for_one_scene(scene,False)
    df_ego = df[df['inst_token'] == "vehicle_info"]
    
    # Initialisation des paramètres
    speed = df_ego.iloc[0]['ego_speed']
    A = df_ego.iloc[0]['ego_pos'][:2]
    B = df_ego.iloc[1]['ego_pos'][:2]
    AB =   [round(B[0] - A[0],3),round(B[1] - A[1],3)]
    ABn = round(AB[0]/np.sqrt((AB[0]*AB[0] + AB[1]*AB[1])),3),round(AB[1]/np.sqrt((AB[0]*AB[0] + AB[1]*AB[1])),3)
    #print(A,B,AB,ABn)
    log = []
    features = ["distance","ego_speed","throttle","brake"]
    sample = nusc.get('sample',scene['first_sample_token'])
    last = scene['last_sample_token']
    i = 0
    throttle = 0
    brake = 0
    print("Position Predite,    Position Reel,    Distance, vitesse,  accélération,  freinage")
    # Boucle
    while i != 30 and sample['token'] != last:
        speed = round(speed,3)
        distance = compute_distance_cheat(A,ABn,df[df['timestamp']==sample['timestamp']])
        data = [[distance,speed,throttle,brake]]
        data = [[distance,speed]]
        throttle = model_t.predict(data)
        brake = model_b.predict(data)
        if throttle[0] < 0:
            throttle[0] = 0.0
        if brake[0] < 0:
            brake[0] = 0.0
        print(A,df_ego.iloc[i]['ego_pos'][:2],distance,speed,throttle,brake)


        #throttle = 0
        #brake = 0
        speed = speed   + throttle[0]/10 - brake[0] - 0.5
        if speed < 0:
            speed = 0 
        # Calcul nouveau point
        A = B
        deplacement = [e * speed/3.6*0.5 for e in ABn]
        #B = list(map(operator.add, B,deplacement))
        i += 1
        B = df_ego.iloc[i]['ego_pos'][:2]
        B = [round(b,3) for b in B]
        sample = nusc.get('sample',sample['next'])
        AB =   [round(B[0] - A[0],3),round(B[1] - A[1],3)]
        ABn = round(AB[0]/np.sqrt((AB[0]*AB[0] + AB[1]*AB[1])),3),round(AB[1]/np.sqrt((AB[0]*AB[0] + AB[1]*AB[1])),3)
        log += [ABn]
    return log

# Premiere version , ne marche pas
def compute_distance(pos,ABn,dataframe):
    #dist = np.linalg.norm(np.array(ego['translation']) - np.array(curr_ann['translation']))
    dataframe = dataframe.drop(columns=['distance'])
    taille = dataframe.shape[0]
    dmin = 99
    mini = 0
    for i in range(taille):
        row = dataframe.iloc[i]
        if row["inst_token"] != "vehicle_info":
            distance_ego = np.linalg.norm(np.array(pos) - np.array(row['object_pos'][:2]))
            distance_vecteur_vitesse = np.absolute(p[1] - a * p[0] - c)/ np.sqrt(a*a + 1)
            if distance_ego < dmin:
                mind = distance
                mini = i
    print("Distance:",mind," ",dataframe.iloc[mini]['inst_token']," ",dataframe.iloc[mini]['object_pos'])
    return mind
    
    #ego_pos = [round(e,3) for e in ego['translation']]
    #object_pos = [round(e,3) for e in curr_ann['translation']]


    #
scene_name = 'scene-0006'
my_scene_token = nusc.field2token('scene', 'name', scene_name)[0]
scene = nusc.get('scene',my_scene_token)
df_scene = build_dataframe_for_one_scene(scene,False)
df = df_scene
#display(df_scene)
liste_temps = sorted(set(df_scene['timestamp'].to_list()))
#liste_temps = np.sort(np.unique(df_scene['timestamp'].to_numpy()))
print(liste_temps)
list_pos = df[df['timestamp']==1531884156948944]['object_pos'].to_list()
print(list_pos)
a = np.transpose(np.asarray(list_pos))
df[(df['timestamp']==1531884156948944) & (df['inst_token']!='vehicle_info')]

