import numpy as np
from matplotlib  import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.animation import FuncAnimation

#We define our class of functions related to the evolution of our nbody simulation.
#It is based on the framework given by Professor Sievers in nbody_slow.py
class particles:
	def __init__(self,m=1.0,npart=1000,soft=0.01,G=1.0,dt=0.1,vrange=0,gridsize=1000,size=5,body2=0,periodic=0,mass_scale=0):
		#Parameters of the simulation
		self.opts={}
		self.opts['soft']=soft
		self.opts['n']=npart
		self.opts['G']=G
		self.opts['dt']=dt
		self.opts['vrange']=vrange
		self.opts['grid']=gridsize
		self.opts['m']=m
		self.opts['size']=size
		self.opts['2body']=body2 #We put body2=1 for Part 2)
		self.opts['periodic']=periodic #We put periodic=1 for non periodic BC, 0 otherwise.
		self.opts['mass_scale']=mass_scale

		#Definition of the necessary functions
		self.initial_position()
		self.initial_velocity()
		self.masses()
		self.partloc()
		self.greenfcn()
		self.potential()
		self.get_forces()
		self.evolve()

	def initial_position(self):
		#2 body case (Part 2). We require orbiting initial conditions for the particles.
		#We do so by starting both particle at opposite ends of a circle and by giving 
		#them opposite and equal velocities.
		if self.opts['2body']==1:
			self.x=np.zeros([2,1])
			self.y=np.zeros([2,1])
			self.y[0]=6*self.opts['grid']/16
			self.y[1]=10*self.opts['grid']/16
			self.x[0]=self.opts['grid']/2
			self.x[1]=self.opts['grid']/2

		#Part 3 section b). When we start with non periodic boundary conditions, we want
		#to ensure no particles are on the boundary.
		elif self.opts['periodic']==1:
			self.x=np.random.rand(self.opts['n'])*(self.opts['grid']-2.5)+1.501
			self.y=np.random.rand(self.opts['n'])*(self.opts['grid']-2.5)+1.501
		# General case for n particles	
		else:
			self.x=np.random.rand(self.opts['n'])*(self.opts['grid']-1)
			self.y=np.random.rand(self.opts['n'])*(self.opts['grid']-1)

		


	def initial_velocity(self):
		#2 body case (Part 2). Here we implement the opposite and equal velocities.
		if self.opts['2body']==1:
			self.vx=np.zeros([2,1])
			self.vy=np.zeros([2,1])
			self.vx[0]=-self.opts['vrange']
			self.vx[1]=self.opts['vrange']

		#General case for n particles. This includes both periodic and non periodic
		#boundary conditions.
		else:

			self.vx=self.opts['vrange']*np.random.randn(self.opts['n'])
			self.vy=self.opts['vrange']*np.random.randn(self.opts['n'])

	def masses(self):
		if self.opts['mass_scale']==1:
			kx=np.fft.fftshift(self.x)
			ky=np.fft.fftshift(self.y)
			k=np.real(np.sqrt(kx**2+ky**2))
			self.m=np.ones(self.opts['n'])*k**(-3)*self.opts['m']
		else:
			self.m=np.ones(self.opts['n'])*self.opts['m']


	def partloc(self):
		#This function is used to allocate particles to the nearest grid point
		self.grid=np.zeros([self.opts['grid'],self.opts['grid']])

		self.ixy=np.asarray([np.round(self.x),np.round(self.y)],dtype='int')
	
		#the value at each grid point is the total mass allocated there.
		for i in range(self.opts['n']):
			self.grid[self.ixy[0,i],self.ixy[1,i]]= self.grid[self.ixy[0,i],self.ixy[1,i]]+self.m[i]


		return self.ixy, self.grid

	def greenfcn(self):
		#initial parameters necessary to set up our Green's function
		self.greenfcngrid=np.zeros([self.opts['grid'],self.opts['grid']])#memory allocation
		size=np.arange(self.opts['grid']) #gridsize to define the meshgrid.
		n=self.opts['grid'] #gridsize
		soft=self.opts['soft'] #softening parameter

		#Definition of the softened radius
		xx,yy=np.meshgrid(size,size)
		xflat=np.ravel(xx)
		yflat=np.ravel(yy)
		rsqr=(xflat)**2+(yflat)**2+soft**2
		rsqr[rsqr<soft]=soft
		r=np.sqrt(rsqr)

		#Definition of the Green's function
		greenfcn=np.ones([self.opts['grid'],self.opts['grid']])
		flatgrid=np.ravel(greenfcn)
		flatgrid=1/(4*np.pi*r)
		
		#We implement the Green's function at each corner of the grid, to avoid central bias
		self.greenfcngrid=flatgrid.reshape([self.opts['grid'],self.opts['grid']])
		self.greenfcngrid[n//2:,:]=np.flip(self.greenfcngrid[:n//2,:],0)
		self.greenfcngrid[:,n//2:]=np.flip(self.greenfcngrid[:,:n//2],1)


	
	def potential(self):
		#We define our potential as the convolution of the density (which is the grid variable)
		#and the green's function. This way we avoid considering particle to particle interactions
		#individually. Furthermore, the use of fft allows us to perform the convolution efficiently.
		ixy, grid=self.partloc()

		V=np.real(np.fft.ifft2(np.fft.fft2(self.greenfcngrid)*np.fft.fft2(grid)))
		#Non Periodic Case. We ensure that the potential is 0 on boundary at all times so that particles
		#will remain inside the domain.
		if self.opts['periodic']==1:

			V=0.5*(np.roll(V,1,axis=1)+V)
			V=0.5*(np.roll(V,1,axis=0)+V)
			V[:,0]=0
			V[:,-1]=0
			V[0,:]=0
			V[-1,:]=0
		#General Case. We average over direct neighboring grid points at each grid point to ensure
		#that a particle will not move if it's all by itself.
		else:
			V=0.5*(np.roll(V,1,axis=1)+V)
			V=0.5*(np.roll(V,1,axis=0)+V)
		return V

	def get_forces(self):
		#Here we use the fact that the force is the gradient of the potential. We therefore use a 
		#first order approximation to the gradient.
		V=self.potential()
		self.Fx=-0.5*(np.roll(V,-1,axis=0)-np.roll(V,1,axis=0))*self.grid
		self.Fy=-0.5*(np.roll(V,-1,axis=1)-np.roll(V,1,axis=1))*self.grid
		#self.forcex,self.forcey=np.gradient(V)
		return self.Fx, self.Fy

	def evolve(self):
		#This is the function we iterate to make the particles move. Their positions are updated 
		#according to the velocity they had in the previous step, and we update their velocity
		#according to the force they feel at their new position.
		
		self.x+=self.vx*self.opts['dt']
		self.y+=self.vy*self.opts['dt']
		self.x=(self.x)%(self.opts['grid']-1)
		self.y=(self.y)%(self.opts['grid']-1)
		ixy, grid=self.partloc()
		Fx, Fy = self.get_forces()
		for i in range(self.opts['n']):
			self.vx[i]+=-Fx[ixy[0,i],ixy[1,i]]*self.opts['dt']/self.m[i]
			self.vy[i]+=-Fy[ixy[0,i],ixy[1,i]]*self.opts['dt']/self.m[i]
		#Potential Energy
		self.pot=-0.5*np.sum(self.potential())
		#Kinetic Energy
		self.kinene=0.5*self.m*(np.sum(self.vx**2)+np.sum(self.vy**2))
		#Total Energy
		self.energy=self.pot+self.kinene
		

		
		


















if __name__=='__main__':
	plt.ion()

	#Here I will code the different options we need to implement for the various parts of the assignment.
	#I will comment out the Part 1, Part 2, and Part 3.

	######## PART 1 ###############
	#Comment out the next three lines to run the code
	
	#n=1
	#gridsize=50
	#part=particles(m=10,npart=n,soft=0.1,dt=3,vrange=0,gridsize=gridsize)

	######## PART 2 ################
	#Comment out the next three lines to run the code

	#n=2
	#gridsize=50
	#part=particles(m=10,npart=n,soft=0.1,dt=3,vrange=0.2,gridsize=gridsize,body2=1,periodic=0)

	######### PART 3 a) #############
	#Comment out the next three lines to run the code

	#n=100000
	#gridsize=500
	#part=particles(m=1/n,npart=n,soft=10,dt=100,vrange=0,gridsize=gridsize,periodic=0)
	
	######### PART 3 b) #############
	#Same set up as part 3 a), but we set the non periodic boundary conditions.
	#Comment out the next three lines to run the code


	#n=100000
	#gridsize=500
	#part=particles(m=1/n,npart=n,soft=10,dt=100,vrange=0,gridsize=gridsize,periodic=1)

	######### PART 4 ############
	#Same set up as part 3 a), but we set initial mass fluctuations to k**-3.

	n=100000
	gridsize=500
	part=particles(m=1/n,npart=n,soft=10,dt=100,vrange=0,gridsize=gridsize,periodic=0, mass_scale=1)
	
	######## ANIMATION ##############
	#The next lines are used to generate gifs that show the evolution of the nbody simulation.
	fig=plt.figure()
	graph=fig.add_subplot(111, autoscale_on=False,xlim=(0,gridsize),ylim=(0,gridsize))
	graph.set_title('Periodic Boundary Conditions')

	###### PART 1-2-3
	frame=graph.imshow(part.grid)
	###### PART 4 (Comment out)
	#frame=graph.imshow(part.grid,norm=LogNorm())

	def animate(i):
		global part, graph, fig
		part.evolve()
		frame.set_data(part.grid)
		return frame

	anim =FuncAnimation(fig, animate, frames=450, interval=50)
	anim.save('prob4.gif', writer='imagemagick')
	

	############# PLOT STEP BY STEP ##################
	#The next lines are used if we want to see the nbody simulation in real time. Furthermore,
	#this will print the energy at every step. 
	
	for i in range(0,500):
		part.evolve()
		plt.clf()
		print(part.energy)
		###### PART 1-2-3
		plt.pcolormesh(part.grid)
		###### PART 4 (Comment out)
		#plt.pcolormesh(part.grid, norm=LogNorm())
		plt.pause(0.01)




	


		
