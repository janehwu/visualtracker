import numpy as np
import math
import random as rand

import simulatePeople as sim

from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

Np = 600 #Number of particles
B = 200
Tv = 1/25.0
BETA = 10.0
v_x = 1 #m/s
a_x = math.exp(-BETA*Tv)
b_x = v_x * math.sqrt(1 - math.pow(a_x,2))
MAX_TIME = 24 #seconds
SIGMA = 1

class Particle(object):
	def __init__(self, pos, velocity):
		self.pos = pos
		self.velocity = velocity

'''
obs_i = Z_i,k or X_i,k (position)
pred_i = X_i,k or X_i,k-1
'''
def dynamicProb(obs_i, pred_i):
	return math.exp(-math.pow(np.linalg.norm(obs_i - pred_i), 2)/(2.0*math.pow(SIGMA,2)))

def runPF():
	#Initialize people
	p1TrajX = []
	p1TrajY = []
	p1TrajZ = []
	p2TrajX = []
	p2TrajY = []
	p2TrajZ = []
	people = [sim.Person([0,0,1.7], 0), sim.Person([5,5,1.7], math.pi)]
	p1TrajX.append(people[0].curPos[0])
	p1TrajY.append(people[0].curPos[1])
	p1TrajZ.append(people[0].curPos[2])
	p2TrajX.append(people[1].curPos[0])
	p2TrajY.append(people[1].curPos[1])
	p2TrajZ.append(people[1].curPos[2])
	m = 2 #Number of people

	#Initialize particle filter
	Z_0 = [p.curPos for p in people]
	for p in people:
		p.randomMove()
	Z_1 = [p.curPos for p in people]
	p1TrajX.append(people[0].curPos[0])
	p1TrajY.append(people[0].curPos[1])
	p1TrajZ.append(people[0].curPos[2])
	p2TrajX.append(people[1].curPos[0])
	p2TrajY.append(people[1].curPos[1])
	p2TrajZ.append(people[1].curPos[2])

	#For each particle
	particles = []
	weights = []
	for i in range(Np):
		r = rand.randint(0,m-1)
		#First pair
		vel0 = (Z_1[r] - Z_0[0])/Tv
		#vel0 = a_x*prev_vel0 + b_x*rand.random()
		pos0 = Z_1[r] + Tv*vel0
		s = 0
		if r == 0:
			s = 1
		#Second pair
		vel1 = (Z_1[s] - Z_0[1])/Tv
		#vel1 = a_x*prev_vel0 + b_x*rand.random()
		pos1 = Z_1[s] + Tv*vel1

		particles.append(Particle(np.array([pos0, pos1]), np.array([vel0, vel1])))
		weights.append(1/float(Np)) #Equal weights

	t = Tv*2
	predictedPos = [[],[]]
	while(t < MAX_TIME):
		#New observations
		for p in people:
			p.randomMove()
		Z_t = [p.curPos for p in people]
		p1TrajX.append(people[0].curPos[0])
		p1TrajY.append(people[0].curPos[1])
		p1TrajZ.append(people[0].curPos[2])
		p2TrajX.append(people[1].curPos[0])
		p2TrajY.append(people[1].curPos[1])
		p2TrajZ.append(people[1].curPos[2])

		#Re-sample
		new_particles = np.random.choice(particles, B + Np,True, weights) #TODO: Fix sampling distribution later
		new_particles = new_particles[B:] #Remove first B samples

		positions = [n.pos for n in new_particles]

		#Advance samples
		for n in range(Np):
			prev_vel = new_particles[n].velocity
			vel = prev_vel
			for i in range(m):
				# vel[i][0] = a_x*prev_vel[i][0] + b_x*rand.random()
				# vel[i][1] = a_x*prev_vel[i][1] + b_x*rand.random()
				vel[i] = (Z_t[i] - new_particles[n].pos[i])/Tv
			pos = new_particles[n].pos + Tv*vel

			new_particles[n] = Particle(pos, vel)

		#Update weights
		new_weights = []
		sumWeights = 0
		for n in range(Np):
			w = dynamicProb(Z_t, new_particles[n].pos)
			new_weights.append(w)
			sumWeights += w
		if sumWeights == 0:
			print "Weird"
			new_weights = [1/float(Np) for w in new_weights]
		else:
			new_weights = [w/sumWeights for w in new_weights]

		#Ramdomly select speaker i
		i = rand.randint(0,m-1)

		#TODO: Sample new state X_star
		q_weights = []
		for n in range(Np):
			q_weights.append(0)
		#Compute acceptance ratio
		#Draw randomly to accept/reject

		#Output position/velocity for this time
		for i in range(m):
			pos = np.array([0.0,0.0,0.0])
			bestPos = new_particles[0].pos[i]
			bestWeight = new_weights[0]
			for n in range(Np):
				if new_weights[n] > bestWeight:
					bestPos = new_particles[n].pos[i]
				#pos += new_weights[n]*new_particles[n].pos[i] / sumWeights
			predictedPos[i].append(bestPos)

		particles = new_particles
		weights = new_weights
		t += Tv

	#Plot results
	fig = plt.figure()
	ax = p3.Axes3D(fig)
	x = [p[0] for p in predictedPos[0]]
	y = [p[1] for p in predictedPos[0]]
	z = [p[2] for p in predictedPos[0]]
	ax.scatter(x,y,z,depthshade=False)
	x = [p[0] for p in predictedPos[1]]
	y = [p[1] for p in predictedPos[1]]
	z = [p[2] for p in predictedPos[1]]
	ax.scatter(x,y,z,c="red",depthshade=False)

	fig = plt.figure()
	ax = p3.Axes3D(fig)
	ax.scatter(p1TrajX,p1TrajY,p1TrajZ,depthshade=False)
	ax.scatter(p2TrajX,p2TrajY,p2TrajZ,c="red",depthshade=False)

	plt.show()

runPF()
