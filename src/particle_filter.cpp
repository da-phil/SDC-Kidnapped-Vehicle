/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
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
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine gen;

	// Ceate a normal (Gaussian) distribution for x, y, and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	weights.resize(num_particles);

	// Create a particle and draw x, y and theta from distributions
	struct Particle particle;
	particle.weight = 1.0;

	for (int i = 0; i < num_particles; ++i) {
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = fmod(dist_theta(gen), 2*M_PI);
		particles.push_back(particle);
	}

	is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	double yaw_rate_dt = yaw_rate*delta_t;
	// Ceate a zero-mean normal (Gaussian) distribution for x, y, and theta
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);	
	for (Particle& p: particles) {
		// handle division by zero case:
		if (fabs(yaw_rate) < std::numeric_limits<double>::epsilon()) {
			p.x += cos(p.theta) * velocity * delta_t;
			p.y += sin(p.theta) * velocity * delta_t;
			// p.theta doesn't change
		} else {
			p.x += dist_x(gen) + (velocity / yaw_rate) * (sin(p.theta + yaw_rate_dt) - sin(p.theta));
			p.y += dist_y(gen) + (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate_dt));
			p.theta = fmod(p.theta + dist_theta(gen) + yaw_rate_dt, 2*M_PI);
		}
	}
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (LandmarkObs& observation: observations) {
		double last_dist = 100.0;
		double new_dist = 0.0;
		int closest_id = 0;
		for (LandmarkObs predict: predicted) {
			new_dist = dist(observation.x, observation.y, predict.x, predict.y);
			if (new_dist < last_dist) {
				last_dist = new_dist;
				closest_id = predict.id;
			}
		}
		observation.id = closest_id;
		cout << "found association with id " << closest_id << endl;
	}
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	for (Particle& p: particles)
	{
		// predict landmarks next to the car, within range
		std::vector<LandmarkObs> predicted;
		LandmarkObs obs;
		for (auto lm: map_landmarks.landmark_list) {
			if (dist(p.x, p.y, lm.x_f, lm.y_f) < sensor_range) {
				obs.id = lm.id_i;
				obs.x  = lm.x_f;
				obs.y  = lm.y_f;
				predicted.push_back(obs);
			}
		}

		// in order to compare map landmarks with observed landmarks we have to transform
		// the observed landmarks into the map coordinate frame
		std::vector<LandmarkObs> obs_transformed;
		for (LandmarkObs obs: observations) {
			LandmarkObs transformed;
			transformed.x  = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
			transformed.y  = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
			obs_transformed.push_back(transformed);
		}

		// find association between map landmarks and observed landmarks
		dataAssociation(predicted, obs_transformed);
		
		// start with a particle weight of 1.0
		p.weight = 1.0;
		
		for (LandmarkObs obs: obs_transformed) {
			for (LandmarkObs pred: predicted) {
				if (pred.id == obs.id) {
					cout << "Particle " << p.id << " - found landmark correspondence (" << obs.id << "), calc weights..." << endl;
					double pos[] = {pred.x, pred.y};
					double mu[]  = {obs.x, obs.y};
					p.weight *= multivariate_gaussian(pos, mu, std_landmark);
					weights[p.id] = p.weight;
					break;					
				}
			}
		}
	}


}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

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
