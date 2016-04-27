#include <random>

void SampleEta(int d, vec& eta_m, vec& u_m, vec& a_m, vector<mat>& z, int N, int K){
	// initialize the value of eta
	double eta_sample, new_eta, old_eta, sum_eta, C, mu, sigma, accept;
	sum_eta = sum(eta_m);
	std::default_random_engine generator;

	// Gibbs sampling
	for(int k = 0; k < K; k++){
		int iter = 0;
		old_eta = eta_sample(k);
		C = sum_eta - eta_m(k);
		eta_sample = 0;
		for(iter < 1000){
			iter++;
			// propose from the normal distribution
			sigma = 1 / gamma;
			mu = dot(u_m.row(i), a_m.row(d)) + z[d](n, i) / gamma;
  			normal_distribution<double> distribution(mu, sigma);
  			new_eta = distribution(generator);

			// compute the rejection criteria
  			accept = min((C + exp(new_eta)) / (C + exp(old_eta)), 1);

			// get the new sample
			uniform_real_distribution<double> distribution(0.0, 1.0);
			double prob = distribution(generator);
			if(prob > accept) new_eta = old_eta;

			eta_sample += new_eta;
			old_eta = new_eta;
		}
		eta_m(k) = eta_sample / 1000;
	}
	return;
}