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

#define NUM_PARTICLES 100
#define EPSILON 0.0001

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = NUM_PARTICLES;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

  particles.resize(num_particles);
	for (Particle& particle : particles) {
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1;
	}
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[1]);

  for (Particle& particle : particles) {
		// motion model
		if (fabs(yaw_rate) < EPSILON) {
			particle.x += velocity * delta_t * cos(particle.theta);
			particle.y += velocity * delta_t * sin(particle.theta);
		} else {
			particle.x += velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
			particle.y += velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
			particle.theta += yaw_rate * delta_t;
		}
		// motion noise
		particle.x += noise_x(gen);
		particle.y += noise_y(gen);
		particle.theta += noise_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  for (LandmarkObs& observation : observations) {
		double min_dist = std::numeric_limits<double>::max();
		int best_id = -1;
		for (LandmarkObs& prediction : predicted) {
			double new_dist = dist(prediction.x, prediction.y, observation.x, observation.y);
			if (new_dist < min_dist) {
				min_dist = new_dist;
				best_id = prediction.id;
			}
		}
		observation.id = best_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks) {

	weights.clear();
	for (Particle& particle : particles) {

    // filter observations in range
		vector<LandmarkObs> predictions;
    for (Map::single_landmark_s& landmark : map_landmarks.landmark_list) {
			if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) <= sensor_range) {
				predictions.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
			}
		}

    // particle coordinates -> map coordinates
		vector<LandmarkObs> observations_map;
		for (LandmarkObs& obs : observations) {
			double x_map = particle.x + obs.x * cos(particle.theta) - obs.y * sin(particle.theta);
			double y_map = particle.y + obs.x * sin(particle.theta) + obs.y * cos(particle.theta);
			observations_map.push_back(LandmarkObs{-1, x_map, y_map});
		}

		dataAssociation(predictions, observations_map);

		particle.weight = 1;
		for (LandmarkObs& obs : observations_map) {
			Map::single_landmark_s m;
			for (Map::single_landmark_s& tmp : map_landmarks.landmark_list) {
				if (tmp.id_i == obs.id) {
					m = tmp;
				}
			}
			double dx = std_landmark[0];
			double dy = std_landmark[1];
			double pxy = exp(-((pow(obs.x - m.x_f, 2) / (2 * pow(dx, 2))) + (pow(obs.y - m.y_f, 2) / (2 * pow(dy, 2))))) / (2 * M_PI * dx * dy);
			particle.weight *=  pxy;
		}
    weights.push_back(particle.weight);
	}
}

void ParticleFilter::resample() {
  discrete_distribution<> dist(weights.begin(), weights.end());
	vector<Particle> new_particles;
	new_particles.resize(num_particles);
	for (int i = 0; i < num_particles; i++) {
    new_particles[i] = particles[dist(gen)];
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

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
