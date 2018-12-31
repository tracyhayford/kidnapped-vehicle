/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang (template)
 *              Completion by Tracy Hayford
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// DONE: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	// number of particles chosen based 
	num_particles = 100;
	// These lines creates normal (Gaussian) distributions for x, y and theta.
   	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

  	for (int i = 0; i < num_particles; i++) {
	  	Particle p1;
      	p1.id = i;
		// initialize particle to input x, y & theta with Gaussian noise added
	  	p1.x = dist_x(gen);
		p1.y = dist_y(gen);
		p1.theta = dist_theta(gen);
		// initialize all weights to 1
		p1.weight = 1;
		
      	particles.push_back(p1);
		
		// initialize all weights to 1
		weights.push_back(1);
    }
	
	// particle filter is now initialized
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// DONE: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	// This line creates a normal (Gaussian) distribution for x, y and theta.
   	default_random_engine gen;
  
  	for (int i = 0; i < num_particles; i++) {
      
      	double next_x;
      	double next_y;
      	double next_theta;
      	if (yaw_rate == 0) {
          	// simple calculations using trig since the yaw rate is 0
          	next_x = particles[i].x + (velocity*delta_t*cos(particles[i].theta));
          	next_y = particles[i].y + (velocity*delta_t*sin(particles[i].theta));
			next_theta = particles[i].theta;
        }
        else {
          	// prediction calculation from lesson 6, section 8 (when yaw rate is non-zero)
      		next_x = particles[i].x + ((velocity/yaw_rate)*(sin(particles[i].theta + (yaw_rate*delta_t)) - sin(particles[i].theta)));
			next_y = particles[i].y + ((velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate*delta_t))));
			next_theta = particles[i].theta + (yaw_rate*delta_t);
        }

		// generate normal distributions using the next position as the mean and the std dev from the input
		normal_distribution<double> dist_x(next_x, std_pos[0]);
		normal_distribution<double> dist_y(next_y, std_pos[1]);
		normal_distribution<double> dist_theta(next_theta, std_pos[2]);
      
		// new positions from a normal distribution around the calculated next position/theta
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// DONE: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	// The predicted vector comes from landmarks and the observations are the transformed observations
	
	// loop through all observations
	for (unsigned int i = 0; i < observations.size(); i++) {
		// select the observation
		LandmarkObs obs = observations[i];
		
		// initialize the minimum to the maximum double
		double min_dist = numeric_limits<double>::max();
		
		// initialize id to nothing selected (invalid index)
		int sel_pred_id = -1;
		
		// loop through all predictions and find the nearest neighbor
		for (unsigned int j = 0; j < predicted.size(); j++) {
			// select the prediction
			LandmarkObs pred = predicted[j];
			// calculate the distance to this prediction
			double this_dist = dist(obs.x, obs.y, pred.x, pred.y);
			if (this_dist < min_dist) {
				min_dist = this_dist;
				// set id so we can connect it to a landmark later
				sel_pred_id = pred.id;
				// cout << "Found one - Observation: " << i << "; " << sel_pred_id << endl;
			}
		}
		// set the observation id to the id of the nearest neighbor
		observations[i].id = sel_pred_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// DONE: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  	for (unsigned int p = 0; p < particles.size(); p++) {
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;
		
		vector<LandmarkObs> trans_observations;
		LandmarkObs obs;
		
		for (unsigned int i = 0; i < observations.size(); i++) {
			LandmarkObs trans_obs;
			obs = observations[i];
			
			// transform from vehicle coordinates to map coordinates
			trans_obs.x = particles[p].x+(obs.x*cos(particles[p].theta)-obs.y*sin(particles[p].theta));
			trans_obs.y = particles[p].y+(obs.x*sin(particles[p].theta)+obs.y*cos(particles[p].theta));
			trans_observations.push_back(trans_obs);
		}
		
		// initalize weight product
		particles[p].weight = 1.0;
		
		// Create a vector to hold valid map landmarks
		vector<LandmarkObs> lminrange;
		
		// Loop through each landmark
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++){
			// copy this landmark
			LandmarkObs lm;
			lm.id = map_landmarks.landmark_list[j].id_i;
			lm.x = map_landmarks.landmark_list[j].x_f;
			lm.y = map_landmarks.landmark_list[j].y_f;
			// Find distance between landmark and particle
			double range = dist(lm.x, lm.y, particles[p].x, particles[p].y);
			// 
			if (range < sensor_range){
				// landmark is in sensor range, save it to the vector
				lminrange.push_back(lm);
				// cout << "In range:" << lm.id << endl;
			}
		}
		
		// associate observations with landmarks that are in range
		dataAssociation(lminrange, trans_observations);
		
		for (unsigned int i = 0; i < trans_observations.size(); i++) {
			int association = 0;
			
			for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
				if (map_landmarks.landmark_list[j].id_i == trans_observations[i].id) {
					association = j;
					break;
                }
			}
			
			// If there is an association, calculate the new weight (Lesson 6, section 20)
			if (association!=0) {
              	// cout << "Observation: " << i << "; Association: " << association << endl;
				double meas_x = trans_observations[i].x;
				double meas_y = trans_observations[i].y;
				double mu_x = map_landmarks.landmark_list[association].x_f;
				double mu_y = map_landmarks.landmark_list[association].y_f;
				// cout << "Meas x: " << meas_x << "; meas y: " << meas_y << endl;
				// cout << "Mu x: " << mu_x << "; mu y: " << mu_y << endl;
				// cout << M_PI << " " << std_landmark[0] << " " << std_landmark[1] << endl;
				long double expon = ((pow(meas_x-mu_x, 2.0)/(2 * pow(std_landmark[0], 2.0))) + (pow(meas_y-mu_y, 2.0)/(2 * pow(std_landmark[1], 2.0)))); 
				long double scale = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
				// cout << scale << " " << expon << endl;
				long double multiplier = scale*exp(-expon);
				if (multiplier > 0) {
					particles[p].weight *= multiplier;
					// cout << "Updated weight: " << multiplier << endl;
				}
				// cout << "Particle: " << p << "; Weight: " << particles[p].weight << endl;
			}
			associations.push_back(association+1);
			sense_x.push_back(trans_observations[i].x);
			sense_y.push_back(trans_observations[i].y);
		}
		
		particles[p] = SetAssociations(particles[p], associations, sense_x, sense_y);
		weights[p] = particles[p].weight;
	}
}

void ParticleFilter::resample() {
	// DONE: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  	default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());
  
	vector<Particle> resample_particles;
  
	for (int i = 0; i < num_particles; i++) {
		resample_particles.push_back(particles[distribution(gen)]); 
	}
	
	particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
  
  	// TLH - added to return the updated particle
  	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
