import os
import numpy as np
import time
from helper import R_to_angle
from params import par


# transform poseGT [R|t] to [theta_x, theta_y, theta_z, x, y, z]
# save as .npy file
def create_pose_data():
	info = {'00': [0, 4540], '01': [0, 1100], '02': [0, 4660], '03': [0, 800], '04': [0, 270], '05': [0, 2760], '06': [0, 1100], '07': [0, 1100],'08': [1100, 5170], '09': [0, 1590], '10': [0, 1200]}
	start_t = time.time()
	for video in info.keys():
		fn = '{}{}.txt'.format(par.pose_dir, video)
		print('Transforming {}...'.format(fn))
		with open(fn) as f:
			lines = [line.split('\n')[0] for line in f.readlines()] 
			poses = [ R_to_angle([float(value) for value in l.split(' ')]) for l in lines]  # list of pose (pose=list of 12 floats)
			poses = np.array(poses)
			#base_fn = os.path.splitext(fn)[0]
			base_fn = './' +os.path.splitext(os.path.split(fn)[1])[0]
			np.save(base_fn+'.npy', poses)
			print('Video {}: shape={}'.format(video, poses.shape))
	print('elapsed time = {}'.format(time.time()-start_t))

def create_imu_data():
	info = {'04': [0, 270]}
	start_t = time.time()
	for video in info.keys():
		fn = '{}{}.txt'.format(par.imu_dir, video)
		print('Transforming {}...'.format(fn))
		with open(fn) as f:
			lines = [line.split('\n')[0] for line in f.readlines()]
			imus = [float(value) for value in (l.split(' ') for l in lines)]  # list of imu data
			imus = np.array(imus)
			base_fn = './' +os.path.splitext(os.path.split(fn)[1])[0]
			np.save(base_fn+'.npy', imus)
			print('Video {}: shape={}'.format(video, imus.shape))
	print('elapsed time = {}'.format(time.time()-start_t))



if __name__ == '__main__':
	create_pose_data()
	create_imu_data()


