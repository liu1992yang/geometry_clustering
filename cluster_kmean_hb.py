### First version June 2018
### This version fixed some bugs and allows more flexibility in multiprocessing 
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import functools
import math
import os,re,sys

from itertools import combinations, compress
from scipy.spatial import distance
from scipy import signal
from sklearn.cluster import KMeans
from sklearn import metrics



X_H_COVALENT = 1.1 #X = C, O, N
X_H_HB = 2.5 #X = O,N
wdir = os.getcwd()


def read_reference(key_atom_fname):
  """
  Read from a ',' split key-atom file and return a list of int
  remove duplicates
  """
  reference_atom_num = set()
  try:
    with open(key_atom_fname, "r") as fin :
      for line in fin:
        content = line.strip()
        if content != '':
          reference_atom_num.update([int(i) for i in content.split(',')])
      return list(reference_atom_num)
  except OSError:
    print('OS error')
    sys.exit()
  except ValueError:
    print('Could not convert data to integer')
    sys.exit()


def get_indexes(reference_atom_num):
  reference_atom_num.sort()
  return [x-1 for x in reference_atom_num]

def get_index_pair(indexes_of_interest):
  #make pairs of all index combinations
  return list(combinations(indexes_of_interest,2))
    

def gen_distance_colnames(pairs):
  #generate column names of pairs
  headers = ['file']
  for pair in pairs:
    headers.append(str(pair[0]+1) + "_" + str(pair[1]+1))
  return headers

    
def read_energy_file(energy_fname):
  """
  read energy file and generate list of pairs[(comfile, energy),(),...]
  """
  file_energy = []
  try:
    with open(os.path.join(wdir,energy_fname),'r') as fin_energy:
      for line in fin_energy:
        arr = line.strip().split()
        if len(arr) < 2 or '' in arr:
          continue
        comfile, curr_energy = get_comfile_name(arr[0]), float(arr[1])
        file_energy.append((comfile, curr_energy))
      return file_energy
  except IOError:
    print("Could not read " + energy_fname)
    sys.exit()


def get_comfile_name(energy_name):
  #keep extracted file format
  return energy_name.strip()
  
     
    
def read_geomfile(filename):
  """
  read one geometry file and return nested list
  """
  data=[]
  with open(os.path.join(wdir, filename),'r') as fin:
    for line in fin:
      if line.strip() == '':
        continue
      if line.startswith("  "):
        arr = re.split(" +", line.strip())
        if len(arr) != 4:
            continue
        data.append((arr[0], float(arr[1]), float(arr[2]),float(arr[3])))
    return {idx: atom for idx, atom in enumerate(data)}
  raise ValueError("Error parsing geom file")



def get_coords(coords_dict,index):
  """
  type coords_dict: List[str, float, float, float]
  rtype: tup(float, float, float)
  """  
  return coords_dict[index][1:]

  
def get_type(coords_dict,index):
  """
  Need to get atom type to be considered for hydrogen bonding
  rtype: str 
  """
  return coords_dict[index][0]


def subset_by_type(coords_dict):  
  """
  subset the coords_dict indexes by atom type
  only need H, N, O, F
  """
  indexes_h = []
  indexes_nof = []
  for idx in coords_dict.keys():
    type_idx = get_type(coords_dict, idx)
    if type_idx == 'H':
      indexes_h.append(idx)
    if type_idx in ('N', 'O', 'F'): 
      indexes_nof.append(idx)
  return indexes_h, indexes_nof      

 
def dist(coords_dict, index1, index2):
  """
  Given index1, index2, calculate distance: float
  """
  return distance.euclidean(get_coords(coords_dict,index1), get_coords(coords_dict,index2))


def x_h_link(distance):
  """
  Determine whether X(C,O,N) and H atom is covalently linked
  type distance: float
  rtype: bool
  """
  return distance <= X_H_COVALENT

def cos_ahb_dists(d_ah, d_hb, d_ab):
  """
  given three distances, calculate cosine angle a-h-b
  d_ah: distance of atom a to hydrogen
  """
  return (d_ah**2 + d_hb**2 - d_ab**2)/(2*d_ah*d_hb)  

    
def cos_index(coords_dict, index_a, index_b, d_ah, d_hb):
  """
  calculates the cosine of bond angle a-h-b based on 
  bond lengh a-h and h-b
  """
  d_ab = dist(coords_dict,index_a, index_b) 
  return cos_ahb_dists(d_ah, d_hb, d_ab)

  
def filter_hx_dists(h_index, x_index_list, coords_dict):
  """
  given ONE hydrogen index, loop through the x_indexes (N,O,F)
  calculate distance and 
  (1)filter by X_H_HB to rule out impossible H-bond
  (2)filter by X_H_COVALENT to rule out impossible Hydrogen case
  rtype: [(x1, dis_x1-H),(x2, dist_x2-H)...] filtered
  """
  hx_dists = []
  for x_idx in x_index_list:
    dist_xh = dist(coords_dict, h_index, x_idx)
    if dist_xh >= X_H_HB:
      continue
    hx_dists.append((x_idx, dist_xh))
  #if qualified number of distance < 2
  # return empty list
  if len(hx_dists) < 2:
    return []
  #put the lowest distance first, which is covalent case  
  hx_dists.sort(key = lambda elem: elem[1]) 
  if x_h_link(hx_dists[0][1]):
    return hx_dists
  return []
  
def gen_legit_list(h_index_list, x_index_list, coords_dict):
  """
  based on whole h_list, get legit combinations
  {(h1,coval_x, coval_dist):[(x2, dist2), ..], (h4 ,coval_x1, coval_dist1):[(x3, dist3),..]..}
  """
  hx_legit = {}
  for h_idx in h_index_list:
    hx_dists = filter_hx_dists(h_idx, x_index_list, coords_dict)
    if hx_dists == []:
      continue
    coval_idx, coval_dist = hx_dists[0]
    hx_legit[(h_idx, coval_idx, coval_dist)] = hx_dists[1:]  
  return hx_legit
 
 
def get_hbond(hx_legit, coords_dict):
  """
  {(h1,coval_x1): [y1, y2], (h2, coval_x2): [y4]...}
  """
  hbonds = {}
  for h_a_pair, h_b_dists in hx_legit.items():
    idx_h, idx_a, d_ah = h_a_pair
    #creat filter that cosine a-h-b <= -0.5 (>=120degree) that qualifies h-bond angle
    filt = list(map(lambda x: cos_index(coords_dict, idx_a, x[0], d_ah, x[1])<= -0.5, h_b_dists))
    if sum(filt) > 0: #not empty, at least one hbond
      hbonds[(idx_h, idx_a)] = list(map(lambda x: x[0], compress(h_b_dists, filt)))
  return hbonds


  
def hb_pattern(coords_dict):
  """
  get total h_bonding_expression for one geometry file
  """
  h_index_list, x_index_list = subset_by_type(coords_dict)
  hx_legit = gen_legit_list(h_index_list, x_index_list, coords_dict)
  hbonds = get_hbond(hx_legit, coords_dict)
  group_pattern = []
  for key, value in hbonds.items():
    group_pattern.extend((key[0],key[1], x_idx) for x_idx in value)
  return ','.join(sorted(list(map(lambda x: format_hb(x[0],x[1],x[2], coords_dict), group_pattern))))

  
  
def format_hb(h_idx, index_a, index_b, coords_dict):
  a_type = get_type(coords_dict, index_a)
  b_type = get_type(coords_dict, index_b)
  return '{0}{1}-H{2}_{3}{4}'.format(a_type, str(index_a+1), str(h_idx+1), b_type, str(index_b+1))


def get_pair_distances(coords_dict, idx_pairs):
  """
  get index pair distances
  return List[float]
  """
  pair_dist = []
  for pair in idx_pairs:
    pair_dist.append(dist(coords_dict,pair[0],pair[1]))
  return pair_dist



### MACHINE LEARNING FUNCTION  

def get_rmsd_score(dists_data,list_k):
  """
  type dists_data: DataFrame
  type K: List[int]
  """  
  rmsd = []
  score =[]
  assert 0 <len(list_k) <= dists_data.shape[0]
  for k in list_k:
    kmeanModel = KMeans(n_clusters=k,max_iter=500).fit(dists_data)
    kmeanModel.fit(dists_data)
    rmsd.append(math.sqrt(kmeanModel.inertia_/len(dists_data)))
    score.append(kmeanModel.score(dists_data))
  return rmsd, score

def max_dist_to_line_k(score, list_k):#list
  n_points = len(list_k)
  all_coords = np.vstack((range(n_points),score)).T
  np.array([range(n_points), score])
  first_point = all_coords[0]
  line_vec = all_coords[-1] - first_point
  lineVecNorm = line_vec / np.sqrt(np.sum(line_vec**2))
  vecFromFirst = all_coords - first_point
  scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, n_points, 1), axis=1)
  vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
  vecToLine = vecFromFirst - vecFromFirstParallel
  distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
  return list_k[np.argmax(distToLine)]  


def get_best_k(rmsd, score, list_k, max_k):
  assert len(rmsd) == len(list_k)
  rmsd_smoothed = signal.savgol_filter(rmsd,window_length=7,polyorder=5)
  score_smoothed = signal.savgol_filter(score,window_length=7,polyorder=5)
  k_names = ['rmsd','score', 'rmsd_smoothed','score_smoothed']
  k_values = list(map(lambda x: max_dist_to_line_k(x, list_k),[rmsd,score,rmsd_smoothed,score_smoothed])) 
  best_k = math.ceil(np.array(k_values).mean())
  #PLOT ELBOW
  plt.figure()
  plt.plot(list_k,rmsd,"ro-")
  plt.plot(list_k,rmsd_smoothed, "bx-")
  plt.xlabel('k')
  plt.ylabel('RMSD to centroid')
  plt.title('The Elbow Method showing the optimal k')

  plt.savefig(os.path.join(wdir,"elbow_maxk_"+str(max_k)+".png"))
  plt.close()
  
  for name,value in zip(k_names, k_values):
     print('best k from {0}:{1};'.format(name, str(value)))
  print('Use {} for k mean clustering'.format(str(best_k)))
  return best_k


def best_k_cluster(best_k,max_iteration, data):
  km = KMeans(n_clusters = best_k, max_iter = max_iteration)
  result = km.fit_predict(data)
  result_df = pd.DataFrame({"cluster":result})
  hbond_kmean = pd.concat([hb_df.iloc[:,[0,1]], result_df,hb_df.iloc[:,2]], axis=1)
  hbond_kmean['hb_id'] = pd.Categorical(hbond_kmean['H-bond'].astype(str)).codes
  hbond_kmean = hbond_kmean.iloc[:,[0,1,2,4,3]]
  hbond_kmean.sort_values(by = ['cluster','hb_id','energy'], inplace = True) #sort by cluster, then hb_id, then energy
  return hbond_kmean
 
 
 
if __name__ == '__main__':
  if len(sys.argv) < 4 :
    print("Usage: python hb_cluster_sys_demo.py energyFile keyAtomFile maxK")
    sys.exit()
  
  energy_file = sys.argv[1]
  key_atom_fname = sys.argv[2]
  MAX_K = int(sys.argv[3])
  
  hb_energy_file = "hb_energy_" + energy_file 
  distance_file_out = "distance_read_from_energy" 
  ##get reference atoms
  reference_atom_num = read_reference(key_atom_fname)
  index_pairs = get_index_pair(get_indexes(reference_atom_num))
  ##Read the energy file and store (filename, energy) pairs
  #process filename_energy and generate list of coordinates
  energy_table = read_energy_file(energy_file)
  fns, energies = zip(*energy_table) #unzip to get two tuples
  file_coords = list(map(read_geomfile, fns))
  
  ##apply h-bond function on each filename (map), convert to df1, then write out[filename, energy, h-bond]
  #combine fname, energy, hbond pattern
  hb_df = pd.DataFrame(list(zip(fns, energies, list(map(hb_pattern,file_coords)))))
  hb_df.columns = ['file','energy','H-bond']  
  hb_df.to_csv(os.path.join(wdir,hb_energy_file), sep="\t",index = False)
  
  ##apply get-distances pairs on each filename(map), convert to df2, then write[filename, pair1, pair2,..]
  #filename from fns, list[dists] from file_coords
  pair_dists = list(map(functools.partial(get_pair_distances,idx_pairs = index_pairs), file_coords))
  fn_dists = [[fns[i]] + pair_dists[i] for i in range(len(fns))] #extend to flattened rows
  dists_df = pd.DataFrame(fn_dists)
  dists_df.columns = gen_distance_colnames(index_pairs)
  dists_df.to_csv(os.path.join(wdir,distance_file_out), sep = '\t', index = False)
  
  ##Machine learning based on existing df
  print("Now starting clustering")
  list_k = list(range(1,MAX_K))
  dists_data = dists_df.iloc[:,1:]
  #print(dists_data.shape)
  labels = dists_df.iloc[:,0]
  rmsd, score = get_rmsd_score(dists_data, list_k)
  best_k = get_best_k(rmsd, score, list_k, MAX_K)
  hbond_kmean = best_k_cluster(best_k , 500, dists_data)
  hbond_kmean.to_csv(os.path.join(wdir,"kmean"+str(best_k)+"_hb_energy_sorted_w_hbid.tsv"),sep="\t",index=False )

