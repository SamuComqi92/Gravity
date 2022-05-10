import numpy as np
import matplotlib.pyplot as plt

class Orbits :	
	"""
	Class to simulation the orbit of a planet around the Sun. Equations for updating 
	the velocity and the position of the planet with respect to the Sun are simply 
	based on the Gravitational Law where the force is proportional to the products 
	of masses divided by their squared distance (i.e., F = GMm/r2). 
	"""
	def __init__(self, Pos_planet, Epochs, v_planet = None) :       
		"""
		- Delta_t 			Time interval between one epoch and the next (in seconds)
		- AU				Astronomycal Unit corresponding to the mean distance of Earth from the Sun
		- Mass_sun			Mass of the Sun
		- G					Gravitational constant
		- Pos_planet		Position of the planet. It can be an intenger (distance in AU from the Sun) or an array (coordinates x-y)
		- Epochs            Number of epochs of the simulation (the final time will be Epochs*Delta_t)
		- v_planet			Velocity of the planet. It can be an intenger (velocity in km/s) or an array (velocity components along axes x-y)
		"""
		self.Delta_t = 24*3600              		#Time interval
		self.AU = 1.49597870707e11          		#In mters
		self.Mass_sun = 1.9891e30           		#In kg 
		self.G = 6.6743e-11                 		#In the SI
		self.Pos_sun = np.array([0, 0])     		#Sun position 
		self.Pos_planet = self.AU*Pos_planet 		#In meters
		self.Epochs = Epochs				
		self.v_planet = v_planet
		
		#Transform Pos_planet into an array if float
		self.flag = 0 							#Transformation flag
		if type(self.Pos_planet) is float :
			self.flag = 1
			self.Pos_planet = np.array([-self.Pos_planet,0])
		
		#Initial distance and acceleration
		self.d0 = np.sqrt((self.Pos_sun[0]-self.Pos_planet[0])**2 + (self.Pos_sun[1]-self.Pos_planet[1])**2)
		self.a_sun0 = self.G*self.Mass_sun/self.d0**2
		
		#Transform v_planet into an array
		#If only the modulus is given, the velocity is transformed into an array perpendicular to the radius of the orbit
		if (self.v_planet is None) and (self.flag == 1) :
			self.v_planet = np.array([0,np.sqrt(self.a_sun0*self.d0)])
		elif (type(self.v_planet) is int) and (self.flag == 1) :
			self.v_planet = 1000*np.array([0,self.v_planet]) 
		elif (self.flag == 0) :
			alpha = np.arctan(self.Pos_planet[1]/self.Pos_planet[0])
			sign1 = -np.sign(self.Pos_planet[1])
			if self.Pos_planet[1] == 0 :
				sign1 = -1
			sign0 = -sign1
			if (self.v_planet is None) :
				self.v_planet = (np.sqrt(self.a_sun0*self.d0))*np.array([sign0*np.cos(alpha),sign1*np.sin(alpha)]) 
			elif (type(self.v_planet) is int) :
				self.v_planet = 1000*self.v_planet*np.array([sign0*np.cos(alpha),sign1*np.sin(alpha)]) 
		
		#History will contain the posizione of the planet at different epochs
		self.History = self.Pos_planet
		self.NewPos = self.Pos_planet
		self.NewVel = self.v_planet
		
	
	#Function for the acceleration sign according to the planet position with respect to the Sun	
	def acceleration(A, B, alpha, a_sun) :
		sign0 = np.sign(B[0]-A[0])
		sign1 = np.sign(B[0]-A[0])
		if B[0] == A[0] :
			sign0 = 1
		elif B[1] == A[1] :
			sign1 = -1
		return a_sun*np.array([sign0*np.cos(alpha), sign1*np.sin(alpha)])	
	
	#Method to draw the orbit and show the animation of the planet at different epochs
	def Draw(self, Pause = 0.01) :
		#Compute the total history of the planet position
		for t in range(self.Epochs) :
			print(f'\rHistory: epoch number {t+1}/{self.Epochs}', end='') 
			d = np.sqrt((self.Pos_sun[0]-self.NewPos[0])**2 + (self.Pos_sun[1]-self.NewPos[1])**2)
			d_x = abs(self.Pos_sun[0]-self.NewPos[0]) 
			d_y = abs(self.Pos_sun[1]-self.NewPos[1])
			alpha = np.arcsin(d_y/d)
			a_sun = self.G*self.Mass_sun/d**2
			self.NewVel = self.NewVel + Orbits.acceleration(self.NewPos, self.Pos_sun, alpha, a_sun)*self.Delta_t
			self.NewPos = self.NewPos + self.NewVel*self.Delta_t
			self.History = np.vstack([self.History,self.NewPos])
		
		#Animated plot
		print("\nAnimation ready!")
		for i in range(self.Epochs) :
			fig = plt.figure(1)
			ax = fig.add_subplot(111, facecolor='black')
			plt.plot([self.Pos_sun[0]],[self.Pos_sun[1]],"ow", markersize=10)
			plt.plot(self.History[:,0],self.History[:,1],'--g',linewidth=0.5)
			plotto = plt.plot(self.History[i,0],self.History[i,1],'oy',markersize=3)
			plt.title("Day {}".format(i))
			plt.pause(Pause)
			plt.clf()


