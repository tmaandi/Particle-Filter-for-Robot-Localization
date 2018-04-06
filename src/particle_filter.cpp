/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *  Modified on: April 5, 2018
 *      Author: Tarun Maandi
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

static vector<LandmarkObs> Vehicle2MapConv(const Particle& particle, const vector<LandmarkObs>& observations);

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  /* This function initializes all particles to first position (based on estimates of
	   x, y, theta and their uncertainties from GPS) and all weights to 1.
	   It also adds random Gaussian noise to each particle.
   */

  /*
   * Set the number of particles
   */
  num_particles = 20;

  /*
   * Initialize weights
   */
  weights.assign (num_particles, 1);


  std::random_device rd{};
  std::mt19937 gen{rd()};

  std::normal_distribution<> noise_xd{0,std[0]};
  std::normal_distribution<> noise_yd{0,std[1]};
  std::normal_distribution<> noise_thetad{0,std[2]};
  /*
   * Initialize particles
   */
  particles.reserve(num_particles);

  for(int i = 0; i < num_particles; ++i)
  {
    particles[i].x = x + noise_xd(gen);
    particles[i].y = y + noise_yd(gen);
    particles[i].theta = theta + noise_thetad(gen);
    particles[i].id = i;
    particles[i].weight = weights[i];
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  /*
   * This fucntion updates the particle's postion and orientation after each time
   * step, based on particle's velocity, yaw rate and translation process noise
   */
  std::random_device rd{};
  std::mt19937 gen{rd()};

  std::normal_distribution<> noise_xd{0,std_pos[0]};
  std::normal_distribution<> noise_yd{0,std_pos[1]};
  std::normal_distribution<> noise_thetad{0,std_pos[2]};

  double noise_x, noise_y, noise_theta;

  for (int i = 0; i < num_particles; ++i)
  {
    noise_x = noise_xd(gen);
    noise_y = noise_yd(gen);
    noise_theta = noise_thetad(gen);

    double theta = particles[i].theta;

    if (abs(yaw_rate) > 0.0001)
    {
      particles[i].x = particles[i].x + \
          (velocity * (sin(theta + yaw_rate * delta_t) - sin(theta)))/yaw_rate + noise_x;
      particles[i].y = particles[i].y + \
          (velocity * (cos(theta) - cos(theta + yaw_rate * delta_t)))/yaw_rate + noise_y;
      particles[i].theta = theta + yaw_rate * delta_t + noise_theta;
    }
    else
    {
      particles[i].x = particles[i].x + velocity * cos(theta) * delta_t + noise_x;
      particles[i].y = particles[i].y + velocity * sin(theta) * delta_t + noise_y;
      particles[i].theta = theta;
    }
  }

}

void ParticleFilter::dataAssociation(const vector<Map::single_landmark_s> landmark_list, std::vector<LandmarkObs>& observations) {

  /*
   * This function uses the nearest neightbour method to assign the closest
   * landmark to each observation, expressed in the map coordinates at each
   * time step
   */
  for (int i = 0; i < observations.size(); ++i)
  {
    double d_temp = dist(observations[i].x, observations[i].y,\
                         (double)landmark_list[0].x_f, (double)landmark_list[0].y_f);

    observations[i].id = landmark_list[0].id_i;

    for (int j = 1; j < landmark_list.size(); ++j)
    {
      double d = dist(observations[i].x, observations[i].y,\
                      (double)landmark_list[j].x_f, (double)landmark_list[j].y_f);

      if (d < d_temp)
      {
        observations[i].id = landmark_list[j].id_i;
        d_temp = d;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  /*
   * In this function, a weight is assigned to each particle, based on the
   * likelihood of observations of the ladmark positions in the map corrdinates,
   * correlating with the actual landmark positions in the map coordinates.
   */
  vector<LandmarkObs> observations_mapC;

  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];

  /*
   * Declaring particle association vectors
   */
  vector<int> associations;
  vector<double> sense_x;
  vector<double> sense_y;

  for (int i = 0; i < num_particles; ++i)
  {
    weights[i] = 1;

     /*
     * Conversion of observations to map coordinates
     */
    observations_mapC = Vehicle2MapConv(particles[i], observations);
    /*
     * Calling dataAssociation() to update observations_mapC id's matching the closest landmarks
     */
    dataAssociation(map_landmarks.landmark_list, observations_mapC);

    /*
     * Updating weights by calculating multi-variate Gaussian probability
     */

    /*
     * gauss_norm = (1/(2 * pi * sig_x * sig_y))
     *
     * exponent= ((x_obs - mu_x)^2)/(2 * sig_x^2) + ((y_obs - mu_y)^2)/(2 * sig_y^2)
     *
     */

    /*
     * Counter to check if all matching landmarks for a given particle
     * are out of sensor range
     */
    int count = 0;

    double gauss_norm, exponent;

    for (int j = 0; j < observations_mapC.size(); ++j)
    {
      for (int k = 0; k < map_landmarks.landmark_list.size(); ++k)
      {
        if(observations_mapC[j].id == map_landmarks.landmark_list[k].id_i)
        {
          double x_obs = observations_mapC[j].x;
          double y_obs = observations_mapC[j].y;
          double mu_x = (double)map_landmarks.landmark_list[k].x_f;
          double mu_y = (double)map_landmarks.landmark_list[k].y_f;

          /*
           * Set associations for each particle,
           * one landmark at a time
           */
          associations.push_back(map_landmarks.landmark_list[k].id_i);
          sense_x.push_back(x_obs);
          sense_y.push_back(y_obs);

          /*
           * Distance between particle and landmark
           */
          double x_dd = pow((particles[i].x - mu_x), 2);
          double y_dd = pow((particles[i].y - mu_y), 2);

          double d = sqrt(x_dd + y_dd);

          if (d > sensor_range)
          {
            count += 1;
          }

          double x_error = pow((x_obs - mu_x), 2);
          double y_error = pow((y_obs - mu_y), 2);

          /*
           * calculate normalization term
           */
          gauss_norm = 1/(2 * M_PI * sig_x * sig_y);

          /*
           * calculate exponent
           */
          exponent = x_error / (2 * pow(sig_x, 2)) + \
              y_error / (2 * pow(sig_y, 2));

          /*
           * update weight using normalization terms and exponent
           */
          weights[i] *= (gauss_norm * exp(-exponent));

          break;
        }
      }
    }

    /*
     * If all observations are beyond the sensor range,
     * set the particle weight to 0
     */
    if (count == observations_mapC.size())
    {
      weights[i] = 0;
    }
    particles[i].weight = weights[i];

    /*
     * Set all associations for the current particle
     */
    SetAssociations(particles[i], associations, sense_x, sense_y);

    /*
     * Clear temporary association data
     */
    associations.clear();
    sense_x.clear();
    sense_y.clear();
  }
}

void ParticleFilter::resample() {

  /*
   * This functions resamples the particles with replacement,
   * based on the normalized weights assigned to each particle,
   * which are calculated in the updateWeights() function
   */
  std::random_device rd{};
  std::mt19937 gen{rd()};

  std::discrete_distribution<> d(weights.begin(), weights.end());

  vector<Particle> updated_particles(num_particles);

  for (int i = 0; i < weights.size(); ++i)
  {
    updated_particles[i] = particles[d(gen)];
  }

  particles = updated_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    /*
       particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
       associations: The landmark id that goes along with each listed association
       sense_x: the associations x mapping already converted to world coordinates
       sense_y: the associations y mapping already converted to world coordinates
    */

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

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

static vector<LandmarkObs> Vehicle2MapConv(const Particle& particle, const vector<LandmarkObs>& observations)
{
  /*
   * This function translates the observations from vehicle's frame of reference
   * to map's frame of reference using homogenous transformation
   */
  vector<LandmarkObs> observations_mapC(observations.size());

  double x_p, y_p, x_c, y_c, theta;
  x_p = particle.x;
  y_p = particle.y;
  theta = particle.theta;

  for (int i = 0; i < observations.size(); ++i)
  {
    x_c = observations[i].x;
    y_c = observations[i].y;
    observations_mapC[i].id = observations[i].id;
    observations_mapC[i].x = x_p + (cos(theta) * x_c) - (sin(theta) * y_c);
    observations_mapC[i].y = y_p + (sin(theta) * x_c) + (cos(theta) * y_c);
  }

  return observations_mapC;
}

