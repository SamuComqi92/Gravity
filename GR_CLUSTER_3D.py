import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class Cluster3D : 
    """
    Class to simulate the gravitational interactions between objects in a 3D cluster.
    Equations for updating the velocities and the positions of each object are 
    simply based on the Gravitational Law where the force is proportional to the 
    products of masses divided by their squared distance (i.e., F = GMm/r2). 
    Inside the cluster, each object feels the force of all the others: these latter
    will be described as a single object located in the center of mass with a total
    mass given by the sum of the single masses. When two or more objects are close 
    to each other (below acertain distance threshold), they will form one single 
    object assuming a purely anelastic collision. 
    """
    def __init__(self, Number_obj, Epochs, Radius, Mass_obj = 1e33, Pause = 0.01, flag_mass = True, flag_rot = False) :
        """
        - Delta_t         Time interval between one epoch and the next
        - AU              Astronomycal Unit corresponding to the mean distance of Earth from the Sun
        - G               Gravitational constant
        - Number_obj      Number of objects in the simulation
        - Epochs          Number of epochs of the simulation (the final time will be Epochs*Delta_t)
        - Radius          Radius of the circle in which the positions of the objects are initialized
        - Mass_obj        Reference mass of the objects
        - Pause           Time between one plot (corresponding to one epoch) and the next
        - flag_mass       If True, masses are initialized randomly in a range [1-3]*Mass_obj
        - flag_rot        If True, the final animation will rotate around the z-axis
        """
        self.Delta_t = 0.1*24*3600             # Time interval for simulation updates (0.1 days in seconds)
        self.AU = 1.49597870707*1e11           # Astronomical Unit in meters
        self.G = 6.6743*1e-11                  # Gravitational constant in SI units
        self.Number_obj = Number_obj           # Initial number of objects
        self.Mass_obj = Mass_obj               # Reference mass for objects
        self.Epochs = Epochs                   # Total number of simulation epochs
        self.Radius = Radius                   # Radius for initial object placement
        self.Pause = Pause                     # Pause duration between animation frames
        self.flag_mass = flag_mass             # Flag to determine if masses are random
        self.flag_rot = flag_rot               # Flag to enable rotation of the 3D plot

        # Random position (in meters) of the objects in a spherical region
        # These lines generate random spherical coordinates (t, u, v) and convert them to Cartesian (x, y, z)
        # to initialize the positions of the objects within a sphere of given Radius.
        t = np.random.uniform(0,1, size = self.Number_obj)
        u = np.random.uniform(0,1, size = self.Number_obj)
        v = np.random.uniform(0,1, size = self.Number_obj)
        x = self.Radius*np.sqrt(t)*np.cos(2*np.pi*u)*np.sin(np.pi*v)
        y = self.Radius*np.sqrt(t)*np.sin(2*np.pi*u)*np.sin(np.pi*v)
        z = self.Radius*np.sqrt(t)*np.cos(np.pi*v)
        
		
        self.Pos_planets = self.AU*np.vstack([x,y,z]).T 

        # Mass (in kg) of the objects (random or similar mass)
        if self.flag_mass == True :
            # If flag_mass is True, assign random masses (1, 2, or 3 times Mass_obj)
            self.Mass_tot = self.Mass_obj*np.random.randint(1,4,size=(self.Number_obj,1))
        else :
            # Otherwise, all objects have the reference Mass_obj
            self.Mass_tot = self.Mass_obj*np.ones((self.Number_obj,1))
        
        # Random velocities (in km/s) of the objects in the cluster (components along the x,y, and z axis)
        Velocities = []
        for i in range(self.Number_obj) :
            # Initialize velocities with random values between -10 km/s and 10 km/s
            Velocities.append(10000*(np.random.rand(1,3)*2-1)[0]) 
        
		# Convert list of velocities to a NumPy array
		self.Velocities = np.array(Velocities) 
    
	######################################################################################################################################################
    # Function to compute the sign of the acceleration components along the x,y, and x axis
    def Sign(A) :
        # Returns -1 if A is non-negative, and 1 if A is negative. Used for direction of acceleration.
        return np.where(A>=0,-1,1)
            
    def acceleration(A, B, alpha1, alpha2, aa) :
        Final_a = []
        for i in range(len(A)) :
            # Calculates the acceleration vector for each object based on its position (A[i]),
            # the center of mass position (B[i]), angular components (alpha1, alpha2),
            # and the magnitude of acceleration (aa[i]).
            # Returns the array of acceleration vectors
            a = Cluster3D.Sign(A[i]-B[i]) * aa[i]*np.array([np.cos(alpha1[i])*np.cos(alpha2[i]),np.cos(alpha1[i])*np.sin(alpha2[i]), np.sin(alpha1[i])])
            Final_a.append(a)
        return np.array(Final_a) 
    
    # Function to Start and Draw the simulation
    def Draw(self) :
        for t in range(self.Epochs) :
            # Show the current epoch
            print(f'\rEpoch number {t+1}/{self.Epochs}', end='') 
            
            # Compute the center of mass with respect to each object in the cluster
            Pos_CM = []
            Mass_CM = []
            for i in range(len(self.Pos_planets)) :
                H = np.delete(self.Pos_planets,i,axis=0) 		# Get all positions except the current object's
                MM = np.delete(self.Mass_tot,i,axis=0)   		# Get all masses except the current object's
                
		# Calculate the center of mass for the remaining objects
                Pos_CM.append((H*MM).sum(axis=0)/(np.sum(MM)))
                Mass_CM.append(np.sum(MM)) 				# Sum of masses of the remaining objects
            Pos_CM = np.array(Pos_CM)
            Mass_CM = np.array(Mass_CM)

	    # Compute the distances of each object from the corresponding center of mass
            Distances = []
            Accelerations = []
            for i in range(len(self.Pos_planets)) :
                # Calculate the Euclidean distance between the object and its corresponding CM
                Distances.append( np.sqrt(np.sum((Pos_CM[i] - self.Pos_planets[i])**2)) )
                if Distances[i] == 0 :
                    Accelerations.append(0) # Avoid division by zero if distance is 0
                else :
                    # Calculate the magnitude of gravitational acceleration (GM/r^2)
                    Accelerations.append(self.G*Mass_CM[i]/Distances[i]**2)
            Distances = np.array(Distances)
            Accelerations = np.array(Accelerations)
            
            # Compute the angle between the object position and the corresponding center of mass
            # These calculations determine the directional components for the acceleration vector.
            d_x = abs(self.Pos_planets[:,0]-Pos_CM[:,0])
            d_y = abs(self.Pos_planets[:,1]-Pos_CM[:,1])
            d_z = abs(self.Pos_planets[:,2]-Pos_CM[:,2])
            ddxy = np.sqrt(d_x**2+d_y**2)
            
	    # Calculate spherical angles alpha1 (polar) and alpha2 (azimuthal)
            alpha1 = np.nan_to_num(np.arctan(d_z/ddxy)) 
            alpha2 = np.nan_to_num(np.arctan(d_y/d_x))
        
            # Updating the velocity and the position of each object
            # Update velocities using acceleration and time interval (v_new = v_old + a*dt)
            self.Velocities = self.Velocities + Cluster3D.acceleration(self.Pos_planets, Pos_CM, alpha1, alpha2, Accelerations)*self.Delta_t
            
	    # Update positions using new velocities and time interval (p_new = p_old + v_new*dt)
            self.Pos_planets = self.Pos_planets + self.Velocities*self.Delta_t
            
            # Check for collisions that will be treated as purely anelastic
            # If two or more objects are close to each other at a distance 
            # less than 1e11 meters, they will form one single object. 
            # Calculate upper triangular matrix of distances between all pairs of objects
            D_section = np.triu((np.sqrt((np.array([x-y for x in self.Pos_planets for y in self.Pos_planets])**2).sum(axis=1))).reshape(len(self.Pos_planets),len(self.Pos_planets)))
            
	    # Find indices of objects that are close (distance > 0 and < 1e11 meters)
            indices = np.where((D_section>0) & (D_section<1e11))
            if len(indices[0])>0 : # If there are any close objects
                # Binary collisions (check if each index appears only once in both rows of 'indices')
                if len(np.unique(indices[0]))+len(np.unique(indices[1])) == 2*len(indices[0]) :
                    # Update all the arrays with the new object
                    for i in range(len(indices[0])) :
                        # Get properties of the two colliding objects
                        a1 = self.Pos_planets[indices[0][i]]
                        a2 = self.Pos_planets[indices[1][i]]
                        vel1 = self.Velocities[indices[0][i]]
                        vel2 = self.Velocities[indices[1][i]]
                        mass1 = self.Mass_tot[indices[0][i]]
                        mass2 = self.Mass_tot[indices[1][i]]
                        
			# Calculate new position, velocity, and mass after an inelastic collision
                        New_pos = (a1+a2)*0.5
                        New_vel = (vel1+vel2)*0.5
                        New_mass = mass1+mass2
                        
			# Add the new combined object to the arrays
                        self.Pos_planets = np.vstack([self.Pos_planets,New_pos])
                        self.Velocities = np.vstack([self.Velocities,New_vel])
                        self.Mass_tot = np.vstack([self.Mass_tot,New_mass])
                    
		    # Delete the old information (the two original colliding objects)
                    self.Pos_planets = np.delete(self.Pos_planets,indices,axis=0)
                    self.Velocities = np.delete(self.Velocities,indices,axis=0)
                    self.Mass_tot = np.delete(self.Mass_tot,indices,axis=0)
                
		# Collisions with more than two objects (iteratively handle multiple collisions)
                else :
                    # Keep processing as long as there are collisions
		    while len(indices[0])>0 : 
                        # Take the first pair of colliding objects
                        a1 = self.Pos_planets[indices[0][0]]
                        a2 = self.Pos_planets[indices[1][0]]
                        vel1 = self.Velocities[indices[0][0]]
                        vel2 = self.Velocities[indices[1][0]]
                        mass1 = self.Mass_tot[indices[0][0]]
                        mass2 = self.Mass_tot[indices[1][0]]
                        
			# Calculate new properties of the combined object
                        New_pos = (a1+a2)*0.5
                        New_vel = (vel1+vel2)*0.5
                        New_mass = mass1+mass2
                        
			# Add the new combined object
                        self.Pos_planets = np.vstack([self.Pos_planets,New_pos])
                        self.Velocities = np.vstack([self.Velocities,New_vel])
                        self.Mass_tot = np.vstack([self.Mass_tot,New_mass])
                        
			# Delete the old information (the two objects that collided)
                        self.Pos_planets = np.delete(self.Pos_planets,indices,axis=0)
                        self.Velocities = np.delete(self.Velocities,indices,axis=0)
                        self.Mass_tot = np.delete(self.Mass_tot,indices,axis=0)
                        
			# Find new indices for potential further collisions after the current one
                        D_section = np.triu((np.sqrt((np.array([x-y for x in self.Pos_planets for y in self.Pos_planets])**2).sum(axis=1))).reshape(len(self.Pos_planets),len(self.Pos_planets)))
                        indices = np.where((D_section>0) & (D_section<1e11))
                
            # Creating the plot for visualization
            # Define plot limits based on initial radius
            Xlim_l = -1.5*self.Radius*self.AU
            Xlim_u = 1.5*self.Radius*self.AU
            Ylim_l = -1.5*self.Radius*self.AU
            Ylim_u = 1.5*self.Radius*self.AU
            Zlim_l = -1.5*self.Radius*self.AU
            Zlim_u = 1.5*self.Radius*self.AU
            
            # Plot setup
            fig = plt.figure(1)
            fig.set_facecolor('black') # Set figure background to black
            ax = fig.add_subplot(111, projection='3d') 	# Create a 3D subplot
            
            # Set view angle for rotation if flag_rot is True (and rotate around Z-axis over time). Otherwise, static view angle
            if self.flag_rot == True :
                ax.view_init(25,30+0.5*t)
            else :
                ax.view_init(25,45)

	    # Set subplot background to black and Hide grid lines
            ax.set_facecolor('black')
            ax.grid(False)
            
            # Set axis labels and their colors
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.zaxis.label.set_color('white')
            
            # Set axis limits
            ax.set_xlim([Xlim_l,Xlim_u])
            ax.set_ylim([Ylim_l,Ylim_u])
            ax.set_zlim([Zlim_l,Zlim_u])
            
            # Make the pane backgrounds transparent
            ax.w_xaxis.pane.fill = ax.w_yaxis.pane.fill = ax.w_zaxis.pane.fill = False
            
            # Scatter plot of objects
            # Marker size is scaled by object mass relative to reference mass
            ax.scatter(self.Pos_planets[:,0],self.Pos_planets[:,1],self.Pos_planets[:,2], 
                       s = np.array([int(x) for x in (self.Mass_tot/self.Mass_obj)]), 
                       c="white", label="Objects", depthshade=0)
            
            # Set plot title showing current day, epoch, and number of objects
            ax.set_title("Day {:.4f} -- Epoch: {}/{}\nNumber of objects: {}/{}".format((t+1)*0.1,t+1,self.Epochs,len(self.Mass_tot),self.Number_obj), color = "white")
            
            plt.pause(self.Pause) # Pause for animation effect
            plt.clf() # Clear the current figure for the next epoch's plot

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
# SIMULATION
if __name__ == "__main__":
    # Create an instance of Cluster3D with specified parameters
    simulation1 = Cluster3D(Number_obj = 50, Epochs = 200, Radius = 20, Pause = 0.05, flag_rot = True)
    simulation1.Draw() # Run the simulation
    print("\nSimulation(s) finished.")
